from __future__ import annotations

import hashlib
import json
from pathlib import Path

from iina_autosub.config import ASR_CACHE_DIR, ensure_dirs
from iina_autosub.models import TranscriptSegment

SAMPLE_BYTES = 65536


def compute_video_fingerprint(video_path: str | Path) -> str:
    """Fast O(1) fingerprint from file size + head/middle/tail samples."""
    path = Path(video_path)
    stat = path.stat()
    size = stat.st_size
    hasher = hashlib.sha256()
    hasher.update(str(size).encode("utf-8"))

    with path.open("rb") as fh:
        hasher.update(fh.read(SAMPLE_BYTES))
        if size > SAMPLE_BYTES * 2:
            fh.seek(size // 2)
            hasher.update(fh.read(SAMPLE_BYTES))
        if size > SAMPLE_BYTES:
            fh.seek(max(0, size - SAMPLE_BYTES))
            hasher.update(fh.read(SAMPLE_BYTES))

    return hasher.hexdigest()[:24]


def _asr_cache_file(
    fingerprint: str,
    audio_stream_index: int | None,
    asr_model: str,
    source_lang: str,
) -> Path:
    ensure_dirs()
    key = hashlib.sha256(
        f"{fingerprint}|{audio_stream_index}|{asr_model}|{source_lang}".encode()
    ).hexdigest()[:24]
    return ASR_CACHE_DIR / f"{key}.json"


def load_cached_transcript(
    video_path: str | Path,
    audio_stream_index: int | None,
    asr_model: str,
    source_lang: str,
) -> list[TranscriptSegment] | None:
    try:
        fp = compute_video_fingerprint(video_path)
        cache_path = _asr_cache_file(fp, audio_stream_index, asr_model, source_lang)
        if not cache_path.is_file():
            return None
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return [TranscriptSegment.model_validate(item) for item in payload.get("segments", [])]
    except Exception:
        return None


def save_cached_transcript(
    video_path: str | Path,
    audio_stream_index: int | None,
    asr_model: str,
    source_lang: str,
    segments: list[TranscriptSegment],
) -> Path:
    fp = compute_video_fingerprint(video_path)
    cache_path = _asr_cache_file(fp, audio_stream_index, asr_model, source_lang)
    payload = {
        "fingerprint": fp,
        "video_name": Path(video_path).name,
        "audio_stream_index": audio_stream_index,
        "asr_model": asr_model,
        "source_lang": source_lang,
        "segments": [seg.model_dump() for seg in segments],
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache_path
