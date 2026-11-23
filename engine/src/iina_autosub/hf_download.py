from __future__ import annotations

import fnmatch
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

from iina_autosub.config import configure_hf_endpoint
from iina_autosub.pipeline.audio import CancelledError

MLX_MODEL_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.npz",
    "*.model",
    "*.tiktoken",
    "*.txt",
    "*.py",
]


def _hf_hub_cache_dir() -> Path:
    custom = os.environ.get("HF_HUB_CACHE")
    if custom:
        return Path(custom).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _repo_cache_dir(repo_id: str) -> Path:
    folder_name = f"models--{repo_id.replace('/', '--')}"
    return _hf_hub_cache_dir() / folder_name


def _dir_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file() and not entry.is_symlink():
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _matches_patterns(filename: str) -> bool:
    return any(fnmatch.fnmatch(filename, pat) for pat in MLX_MODEL_PATTERNS)


def is_hf_repo_cached(repo_id: str) -> bool:
    """Fast local-only check whether an MLX model repo already has weights cached on disk."""
    local_candidate = Path(repo_id).expanduser()
    if local_candidate.is_dir():
        return True
    snapshots = _repo_cache_dir(repo_id) / "snapshots"
    if not snapshots.is_dir():
        return False
    for snap in snapshots.iterdir():
        if not snap.is_dir():
            continue
        has_config = (snap / "config.json").exists()
        has_weights = any(snap.glob("*.safetensors")) or any(snap.glob("*.npz"))
        if has_config and has_weights:
            return True
    return False


def _estimate_repo_bytes(repo_id: str) -> int:
    try:
        from huggingface_hub import HfApi

        info = HfApi().repo_info(repo_id, files_metadata=True)
        total = 0
        for sib in info.siblings or []:
            if sib.size and _matches_patterns(sib.rfilename):
                total += int(sib.size)
        return total
    except Exception:
        return 0


def ensure_hf_repo_downloaded(
    repo_id: str,
    hf_mirror: bool = False,
    cancel_event: threading.Event | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
) -> str:
    """Return local cached path for a Hugging Face repo, reporting live progress (incl. hf-xet)."""
    local_candidate = Path(repo_id).expanduser()
    if local_candidate.is_dir():
        return str(local_candidate)

    configure_hf_endpoint(hf_mirror)
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm as base_tqdm

    # Fast local cache hit -> zero network overhead
    try:
        cached_path = snapshot_download(
            repo_id,
            local_files_only=True,
            allow_patterns=MLX_MODEL_PATTERNS,
        )
        if cached_path and Path(cached_path).is_dir():
            return cached_path
    except Exception:
        pass

    short_name = repo_id.split("/")[-1]
    expected_bytes = _estimate_repo_bytes(repo_id)
    size_str = f" (约 {expected_bytes / (1024**3):.1f} GB)" if expected_bytes > 50_000_000 else ""
    if progress_cb:
        progress_cb(0.0, f"首次下载模型 {short_name}{size_str}（下载完成后后续秒开）...")

    repo_blobs = _repo_cache_dir(repo_id) / "blobs"
    xet_cache = _hf_hub_cache_dir().parent / "xet"
    base_blobs = _dir_bytes(repo_blobs)
    base_xet = _dir_bytes(xet_cache)
    t_start = time.monotonic()

    lock = threading.Lock()
    state = {"tqdm_done": 0, "tqdm_total": expected_bytes, "last_report": 0.0}
    stop_monitor = threading.Event()

    def _emit_progress(done_bytes: int, total_bytes: int) -> None:
        if not progress_cb:
            return
        elapsed = max(0.5, time.monotonic() - t_start)
        speed_mb = (done_bytes / (1024**2)) / elapsed
        done_gb = done_bytes / (1024**3)
        if total_bytes > 0:
            frac = min(0.98, done_bytes / total_bytes)
            tot_gb = total_bytes / (1024**3)
            progress_cb(
                frac,
                f"首次下载模型 {short_name} ({int(frac * 100)}% · {done_gb:.2f}/{tot_gb:.2f} GB · {speed_mb:.1f} MB/s)",
            )
        else:
            progress_cb(
                0.5,
                f"首次下载模型 {short_name} (已下载 {done_gb:.2f} GB · {speed_mb:.1f} MB/s)",
            )

    def _disk_monitor() -> None:
        # Tracks actual bytes written on disk because hf-xet bypasses Python tqdm_class
        while not stop_monitor.wait(1.0):
            if cancel_event is not None and cancel_event.is_set():
                return
            delta_blobs = max(0, _dir_bytes(repo_blobs) - base_blobs)
            delta_xet = max(0, _dir_bytes(xet_cache) - base_xet)
            disk_done = max(delta_blobs, delta_xet)
            with lock:
                done = max(disk_done, state["tqdm_done"])
                total = max(expected_bytes, state["tqdm_total"])
            if done >= 1_000_000:
                _emit_progress(done, total)

    class _ProgressTqdm(base_tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.total and self.unit in {"B", "iB"}:
                with lock:
                    state["tqdm_total"] = max(state["tqdm_total"], int(self.total))

        def update(self, n=1):
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError("模型下载已取消")
            super().update(n)
            if self.unit in {"B", "iB"} and n:
                now = time.monotonic()
                with lock:
                    state["tqdm_done"] += int(n)
                    if now - state["last_report"] >= 0.8 and state["tqdm_done"] >= 1_000_000:
                        state["last_report"] = now
                        _emit_progress(state["tqdm_done"], max(expected_bytes, state["tqdm_total"]))

    monitor_thread = threading.Thread(target=_disk_monitor, daemon=True)
    monitor_thread.start()
    try:
        downloaded = snapshot_download(
            repo_id,
            allow_patterns=MLX_MODEL_PATTERNS,
            tqdm_class=_ProgressTqdm,
        )
    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=1.5)

    if progress_cb:
        progress_cb(1.0, f"模型 {short_name} 准备就绪")
    return downloaded
