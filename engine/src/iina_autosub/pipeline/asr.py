from __future__ import annotations

import gc
import sys
import threading
from collections.abc import Callable

import numpy as np

from iina_autosub.config import resolve_asr_repo
from iina_autosub.hf_download import ensure_hf_repo_downloaded
from iina_autosub.models import TranscriptSegment, WordTiming
from iina_autosub.pipeline.audio import CancelledError
from iina_autosub.pipeline.langs import LanguageProfile
from iina_autosub.pipeline.vad import SAMPLE_RATE, SpeechSpan


def is_hallucination_text(text: str, profile: LanguageProfile) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    for pattern in profile.hallucination_regexes:
        if pattern.search(cleaned):
            return True
    return False


def clear_mlx_whisper_cache() -> None:
    """Explicitly drop cached Whisper weights and reclaim Apple Silicon unified memory."""
    try:
        mt_mod = sys.modules.get("mlx_whisper.transcribe")
        holder = getattr(mt_mod, "ModelHolder", None) if mt_mod is not None else None
        if holder is not None:
            holder.model = None
            holder.model_path = None
    except Exception:
        pass

    gc.collect()
    try:
        import mlx.core as mx

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass


def transcribe_windows(
    audio: np.ndarray,
    windows: list[SpeechSpan],
    profile: LanguageProfile,
    asr_model: str,
    hf_mirror: bool = False,
    cancel_event: threading.Event | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
    download_cb: Callable[[float, str], None] | None = None,
) -> list[TranscriptSegment]:
    """Run MLX Whisper over VAD-grouped speech windows with progress, hallucination filter, and cleanup."""
    if not windows:
        return []

    repo_id = resolve_asr_repo(asr_model)
    local_model_path = ensure_hf_repo_downloaded(
        repo_id,
        hf_mirror=hf_mirror,
        cancel_event=cancel_event,
        progress_cb=download_cb,
    )

    import mlx_whisper

    segments: list[TranscriptSegment] = []
    next_id = 1
    total = len(windows)
    prev_text = ""
    consecutive_dups = 0

    try:
        for idx, win in enumerate(windows):
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError("任务已取消")

            if progress_cb:
                label = (
                    f"加载语音识别模型 {repo_id.split('/')[-1]}..."
                    if idx == 0
                    else f"语音识别中 ({idx}/{total} 段)"
                )
                progress_cb(idx / total, label)

            start_sample = max(0, int(round(win.start_sec * SAMPLE_RATE)))
            end_sample = min(len(audio), int(round(win.end_sec * SAMPLE_RATE)))
            if end_sample - start_sample < int(0.15 * SAMPLE_RATE):
                continue

            chunk_audio = np.ascontiguousarray(audio[start_sample:end_sample], dtype=np.float32)

            # Request word timestamps when supported; gracefully fall back if a distilled model errors
            transcribe_kwargs = {
                "path_or_hf_repo": local_model_path,
                "language": profile.whisper_lang,
                "task": "transcribe",
                "word_timestamps": True,
                "hallucination_silence_threshold": 1.5,
                "condition_on_previous_text": False,
                "no_speech_threshold": 0.65,
                "compression_ratio_threshold": 2.4,
                "initial_prompt": profile.whisper_initial_prompt,
                "verbose": False,
            }
            try:
                result = mlx_whisper.transcribe(chunk_audio, **transcribe_kwargs)
            except Exception:
                transcribe_kwargs["word_timestamps"] = False
                transcribe_kwargs.pop("hallucination_silence_threshold", None)
                result = mlx_whisper.transcribe(chunk_audio, **transcribe_kwargs)

            for raw_seg in result.get("segments", []):
                no_speech_prob = float(raw_seg.get("no_speech_prob") or 0.0)
                compression_ratio = float(raw_seg.get("compression_ratio") or 1.0)
                if no_speech_prob > 0.72 or compression_ratio > 2.4:
                    continue

                text = str(raw_seg.get("text") or "").strip()
                if is_hallucination_text(text, profile):
                    continue

                if text == prev_text:
                    consecutive_dups += 1
                    if consecutive_dups >= 2:
                        continue
                else:
                    prev_text = text
                    consecutive_dups = 0

                seg_start = win.start_sec + float(raw_seg.get("start") or 0.0)
                seg_end = min(
                    win.end_sec,
                    win.start_sec + float(raw_seg.get("end") or (win.end_sec - win.start_sec)),
                )
                if seg_end <= seg_start + 0.05:
                    seg_end = seg_start + 0.2

                words: list[WordTiming] = []
                for w in raw_seg.get("words") or []:
                    w_text = str(w.get("word") or "")
                    if not w_text:
                        continue
                    w_start = win.start_sec + float(w.get("start") or 0.0)
                    w_end = win.start_sec + float(w.get("end") or w_start)
                    w_prob = float(w.get("probability") or 1.0)
                    words.append(
                        WordTiming(
                            word=w_text,
                            start=max(seg_start, w_start),
                            end=max(w_start, min(win.end_sec, w_end)),
                            probability=w_prob,
                        )
                    )

                segments.append(
                    TranscriptSegment(
                        id=next_id,
                        start=round(seg_start, 3),
                        end=round(seg_end, 3),
                        text=text,
                        words=words,
                    )
                )
                next_id += 1

            if progress_cb:
                progress_cb((idx + 1) / total, f"语音识别中 ({idx + 1}/{total} 段)")
    finally:
        clear_mlx_whisper_cache()

    return segments
