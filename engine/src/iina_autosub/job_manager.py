from __future__ import annotations

import asyncio
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from iina_autosub.models import GenerateRequest, JobStage, JobState
from iina_autosub.pipeline.audio import CancelledError
from iina_autosub.pipeline.orchestrator import run_generation_pipeline

ACTIVE_STAGES = frozenset(
    {
        "queued",
        "probing",
        "downloading_model",
        "extracting_audio",
        "vad",
        "asr",
        "segmenting",
        "translating",
        "rendering",
    }
)


class JobManager:
    """Thread-safe single-GPU job scheduler with SSE broadcast and cancellation."""

    def __init__(self, max_history: int = 50):
        self._jobs: dict[str, JobState] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._subscribers: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[JobState]]]] = {}
        self._lock = threading.Lock()
        # Single worker serializes Metal GPU inference so Apple Silicon unified memory never OOMs
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="autosub-worker")
        self._max_history = max_history

    def submit(self, req: GenerateRequest) -> JobState:
        resolved_path = str(Path(req.video_path).expanduser().resolve())
        req = req.model_copy(update={"video_path": resolved_path})

        with self._lock:
            # Deduplicate active job for the exact same movie
            for existing in self._jobs.values():
                if existing.video_path == resolved_path and existing.stage in ACTIVE_STAGES:
                    return existing.model_copy()

            job_id = uuid.uuid4().hex[:12]
            state = JobState(
                job_id=job_id,
                video_path=resolved_path,
                stage="queued",
                progress=0.0,
                message="已加入生成队列",
            )
            cancel_event = threading.Event()
            self._jobs[job_id] = state
            self._cancel_events[job_id] = cancel_event
            self._prune_history_locked()

        self._executor.submit(self._run_job, job_id, req, cancel_event)
        return state.model_copy()

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            state = self._jobs.get(job_id)
            return state.model_copy() if state else None

    def list_jobs(self) -> list[JobState]:
        with self._lock:
            return sorted(
                (j.model_copy() for j in self._jobs.values()),
                key=lambda x: x.created_at,
                reverse=True,
            )

    def cancel(self, job_id: str) -> JobState | None:
        with self._lock:
            state = self._jobs.get(job_id)
            ev = self._cancel_events.get(job_id)
            if state is None:
                return None
            if ev is not None:
                ev.set()
            if state.stage == "queued":
                state.stage = "cancelled"
                state.message = "任务已取消"
                state.updated_at = time.time()
                self._notify_locked(job_id, state)
            return state.model_copy()

    def subscribe(self, job_id: str, loop: asyncio.AbstractEventLoop) -> asyncio.Queue[JobState]:
        queue: asyncio.Queue[JobState] = asyncio.Queue(maxsize=64)
        with self._lock:
            self._subscribers.setdefault(job_id, []).append((loop, queue))
            current = self._jobs.get(job_id)
            if current is not None:
                queue.put_nowait(current.model_copy())
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue[JobState]) -> None:
        with self._lock:
            subs = self._subscribers.get(job_id)
            if not subs:
                return
            self._subscribers[job_id] = [(lp, q) for lp, q in subs if q is not queue]
            if not self._subscribers[job_id]:
                self._subscribers.pop(job_id, None)

    def _update(
        self,
        job_id: str,
        *,
        stage: JobStage | None = None,
        progress: float | None = None,
        message: str | None = None,
        detail: str | None = None,
        subtitle_path: str | None = None,
        error: str | None = None,
        reused_existing: bool | None = None,
        used_embedded_sub: bool | None = None,
        segments_count: int | None = None,
    ) -> None:
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return
            if stage is not None:
                state.stage = stage
            if progress is not None:
                state.progress = round(progress, 4)
            if message is not None:
                state.message = message
            if detail is not None:
                state.detail = detail
            if subtitle_path is not None:
                state.subtitle_path = subtitle_path
            if error is not None:
                state.error = error
            if reused_existing is not None:
                state.reused_existing = reused_existing
            if used_embedded_sub is not None:
                state.used_embedded_sub = used_embedded_sub
            if segments_count is not None:
                state.segments_count = segments_count
            state.updated_at = time.time()
            self._notify_locked(job_id, state)

    def _notify_locked(self, job_id: str, state: JobState) -> None:
        snapshot = state.model_copy()
        for loop, queue in self._subscribers.get(job_id, []):
            def _push(q: asyncio.Queue[JobState] = queue, item: JobState = snapshot) -> None:
                if q.full():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                try:
                    q.put_nowait(item)
                except asyncio.QueueFull:
                    pass

            try:
                loop.call_soon_threadsafe(_push)
            except RuntimeError:
                pass

    def _run_job(self, job_id: str, req: GenerateRequest, cancel_event: threading.Event) -> None:
        if cancel_event.is_set():
            self._update(job_id, stage="cancelled", message="任务已取消")
            return

        def on_progress(stage: JobStage, progress: float, message: str, detail: str) -> None:
            self._update(job_id, stage=stage, progress=progress, message=message, detail=detail)

        try:
            result = run_generation_pipeline(req, cancel_event, on_progress)
            self._update(
                job_id,
                stage="done",
                progress=1.0,
                message=str(result.get("message") or "字幕已生成"),
                detail=str(result.get("detail") or ""),
                subtitle_path=str(result["subtitle_path"]),
                reused_existing=bool(result["reused_existing"]),
                used_embedded_sub=bool(result["used_embedded_sub"]),
                segments_count=int(result["segments_count"]),
            )
        except CancelledError:
            self._update(job_id, stage="cancelled", message="任务已取消")
        except Exception as exc:
            self._update(
                job_id,
                stage="error",
                message=f"生成失败: {exc}",
                error=str(exc),
            )

    def _prune_history_locked(self) -> None:
        if len(self._jobs) <= self._max_history:
            return
        finished = sorted(
            (j for j in self._jobs.values() if j.stage not in ACTIVE_STAGES),
            key=lambda x: x.updated_at,
        )
        for old in finished[: max(0, len(self._jobs) - self._max_history)]:
            self._jobs.pop(old.job_id, None)
            self._cancel_events.pop(old.job_id, None)

    def shutdown(self) -> None:
        with self._lock:
            for ev in self._cancel_events.values():
                ev.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
