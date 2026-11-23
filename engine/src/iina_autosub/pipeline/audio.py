from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

from iina_autosub.ffmpeg_bin import get_ffmpeg

SURROUND_LAYOUT_HINTS = ("5.1", "7.1", "5.0", "6.1")


class CancelledError(RuntimeError):
    """Raised when an active subtitle generation job is cancelled."""


def _run_cancellable(cmd: list[str], cancel_event: threading.Event | None = None) -> tuple[int, str]:
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        while proc.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise CancelledError("任务已取消")
            time.sleep(0.15)
        stderr = proc.stderr.read() if proc.stderr else ""
        return proc.returncode or 0, stderr
    finally:
        if proc.stderr:
            proc.stderr.close()


def extract_audio_wav(
    video_path: str | Path,
    output_wav: str | Path,
    audio_stream_index: int | None = None,
    channels: int = 2,
    channel_layout: str = "",
    center_channel_boost: bool = True,
    cancel_event: threading.Event | None = None,
) -> Path:
    """Extract 16kHz mono PCM WAV from video, boosting Front Center channel on 5.1/7.1 surround."""
    ffmpeg = get_ffmpeg()
    out_path = Path(output_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    map_spec = f"0:{audio_stream_index}" if audio_stream_index is not None else "0:a:0"
    is_surround = channels >= 6 or any(h in (channel_layout or "") for h in SURROUND_LAYOUT_HINTS)

    base_cmd = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-map",
        map_spec,
        "-vn",
        "-sn",
        "-dn",
    ]

    if center_channel_boost and is_surround:
        # Weight Front Center (dialogue) heavily while keeping light stereo ambience.
        # FL/FR/FC exist across 5.1, 5.1(side), 7.1, and 7.1(wide).
        af_cmd = [
            *base_cmd,
            "-af",
            "pan=mono|c0=0.70*FC+0.15*FL+0.15*FR,highpass=f=80",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(out_path),
        ]
        rc, _stderr = _run_cancellable(af_cmd, cancel_event)
        if rc == 0 and out_path.is_file() and out_path.stat().st_size > 44:
            return out_path

    fallback_cmd = [
        *base_cmd,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(out_path),
    ]
    rc, stderr = _run_cancellable(fallback_cmd, cancel_event)
    if rc != 0 or not out_path.is_file():
        raise RuntimeError(f"FFmpeg 提取音频失败: {stderr.strip()}")
    return out_path
