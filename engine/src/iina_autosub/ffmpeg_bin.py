from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path

EXTRA_SEARCH_DIRS = (
    Path("/opt/homebrew/bin"),
    Path("/usr/local/bin"),
    Path.home() / ".local" / "bin",
)


def _find_on_system(name: str) -> str | None:
    env_override = os.environ.get(f"IINA_AUTOSUB_{name.upper()}")
    if env_override and Path(env_override).is_file():
        return env_override

    found = shutil.which(name)
    if found:
        return found

    for directory in EXTRA_SEARCH_DIRS:
        candidate = directory / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


@lru_cache(maxsize=1)
def get_ffmpeg_binaries() -> tuple[str, str]:
    """Return absolute paths to (ffmpeg, ffprobe), auto-fetching static binaries if needed."""
    ffmpeg = _find_on_system("ffmpeg")
    ffprobe = _find_on_system("ffprobe")
    if ffmpeg and ffprobe:
        return ffmpeg, ffprobe

    try:
        from static_ffmpeg import run as static_ffmpeg_run

        static_ffmpeg_bin, static_ffprobe_bin = (
            static_ffmpeg_run.get_or_fetch_platform_executables_else_raise()
        )
        return ffmpeg or static_ffmpeg_bin, ffprobe or static_ffprobe_bin
    except Exception as exc:
        raise RuntimeError(
            "未找到 ffmpeg / ffprobe，且自动获取 static-ffmpeg 失败。"
            "请先安装 ffmpeg（例如 `brew install ffmpeg`）或检查网络连接。"
        ) from exc


def get_ffmpeg() -> str:
    return get_ffmpeg_binaries()[0]


def get_ffprobe() -> str:
    return get_ffmpeg_binaries()[1]
