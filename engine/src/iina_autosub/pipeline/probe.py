from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pysubs2

from iina_autosub.ffmpeg_bin import get_ffmpeg, get_ffprobe
from iina_autosub.models import AudioTrack, ProbeResult, SubtitleTrack, TranscriptSegment
from iina_autosub.pipeline.langs import get_language_profile

TEXT_SUBTITLE_CODECS = frozenset(
    {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text", "ttml"}
)
SUBTITLE_EXTENSIONS = frozenset({".ass", ".srt", ".ssa", ".vtt"})
ZH_LANG_CODES = frozenset(
    {"zh", "zho", "chi", "chs", "cht", "zh-cn", "zh-tw", "zh-hk", "zh-hans", "zh-hant", "cmn"}
)
ZH_TOKEN_RE = re.compile(
    r"(^|[\.\-_\s\[\(])(zh|zho|chi|chs|cht|cn|zh-hans|zh-hant|zh-cn|zh-tw|zh-ja|zh-en|sc|tc|gb|big5|chinese|中文|简体|繁體|简中|繁中|双语|雙語)($|[\.\-_\s\]\)])",
    re.IGNORECASE,
)
COMMENTARY_RE = re.compile(r"commentary|director|コメンタリー|副音声|解説", re.IGNORECASE)
ASS_TAG_RE = re.compile(r"\{[^}]*\}")


def _is_chinese_label(language: str, title: str) -> bool:
    lang = (language or "").strip().lower()
    if lang in ZH_LANG_CODES or lang.startswith("zh"):
        return True
    return bool(ZH_TOKEN_RE.search(title or ""))


def _is_source_label(language: str, title: str, iso_tags: frozenset[str]) -> bool:
    lang = (language or "").strip().lower()
    if lang in iso_tags:
        return True
    low_title = (title or "").lower()
    if "ja" in iso_tags and any(tok in low_title for tok in ("日本語", "japanese", "jpn")):
        return True
    return False


def find_existing_zh_subtitles(video_path: str | Path) -> list[str]:
    vpath = Path(video_path)
    parent = vpath.parent
    if not parent.is_dir():
        return []
    stem = vpath.stem
    stem_lower = stem.lower()
    matches: list[str] = []

    for child in sorted(parent.iterdir()):
        if not child.is_file():
            continue
        if child.suffix.lower() not in SUBTITLE_EXTENSIONS:
            continue
        child_stem = child.stem
        if not child_stem.lower().startswith(stem_lower):
            continue
        remainder = child_stem[len(stem) :]
        if remainder == "":
            # Exact-match .srt/.ass with no language tag: only treat as Chinese if file contains CJK
            if _subtitle_file_has_chinese(child):
                matches.append(str(child))
            continue
        if remainder[0] not in ".-_ [(":
            continue
        if ZH_TOKEN_RE.search(remainder):
            matches.append(str(child))
    return matches


def _subtitle_file_has_chinese(sub_path: Path) -> bool:
    try:
        sample = sub_path.read_bytes()[:32768].decode("utf-8", errors="ignore")
        # Distinguish Chinese from pure Japanese by checking lack of hiragana/katakana dominance
        cjk_count = sum(1 for ch in sample if "\u4e00" <= ch <= "\u9fff")
        kana_count = sum(1 for ch in sample if "\u3040" <= ch <= "\u30ff")
        return cjk_count >= 15 and cjk_count > kana_count * 2
    except Exception:
        return False


def _select_recommended_audio(tracks: list[AudioTrack], iso_tags: frozenset[str]) -> int | None:
    if not tracks:
        return None
    non_commentary = [t for t in tracks if not t.is_commentary]
    pool = non_commentary or tracks

    lang_matches = [t for t in pool if _is_source_label(t.language, t.title, iso_tags)]
    if lang_matches:
        defaults = [t for t in lang_matches if t.is_default]
        return (defaults[0] if defaults else lang_matches[0]).stream_index

    defaults = [t for t in pool if t.is_default]
    if defaults:
        return defaults[0].stream_index
    return pool[0].stream_index


def probe_video(video_path: str | Path, source_lang: str = "ja") -> ProbeResult:
    vpath = Path(video_path).expanduser().resolve()
    if not vpath.is_file():
        raise FileNotFoundError(f"找不到视频文件: {vpath}")

    profile = get_language_profile(source_lang)
    ffprobe = get_ffprobe()
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(vpath),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe 解析失败: {proc.stderr.strip()}")

    data = json.loads(proc.stdout or "{}")
    fmt = data.get("format", {})
    duration_sec = float(fmt.get("duration") or 0.0)

    audio_tracks: list[AudioTrack] = []
    subtitle_tracks: list[SubtitleTrack] = []
    embedded_zh_streams: list[int] = []
    embedded_source_text_streams: list[int] = []

    for stream in data.get("streams", []):
        codec_type = stream.get("codec_type")
        index = int(stream.get("index", 0))
        tags = stream.get("tags") or {}
        disposition = stream.get("disposition") or {}
        language = str(tags.get("language") or tags.get("LANGUAGE") or "").strip()
        title = str(tags.get("title") or tags.get("TITLE") or "").strip()
        is_default = bool(disposition.get("default", 0))

        if codec_type == "audio":
            audio_tracks.append(
                AudioTrack(
                    stream_index=index,
                    codec=str(stream.get("codec_name") or ""),
                    language=language,
                    title=title,
                    channels=int(stream.get("channels") or 2),
                    channel_layout=str(stream.get("channel_layout") or ""),
                    sample_rate=int(stream.get("sample_rate") or 0),
                    is_default=is_default,
                    is_commentary=bool(
                        disposition.get("comment", 0) or COMMENTARY_RE.search(title)
                    ),
                )
            )
        elif codec_type == "subtitle":
            codec = str(stream.get("codec_name") or "").lower()
            is_text = codec in TEXT_SUBTITLE_CODECS
            subtitle_tracks.append(
                SubtitleTrack(
                    stream_index=index,
                    codec=codec,
                    language=language,
                    title=title,
                    is_default=is_default,
                    is_text_based=is_text,
                )
            )
            if _is_chinese_label(language, title):
                embedded_zh_streams.append(index)
            elif is_text and _is_source_label(language, title, profile.iso_tags):
                embedded_source_text_streams.append(index)

    recommended_audio = _select_recommended_audio(audio_tracks, profile.iso_tags)
    existing_zh = find_existing_zh_subtitles(vpath)

    return ProbeResult(
        video_path=str(vpath),
        duration_sec=duration_sec,
        audio_tracks=audio_tracks,
        subtitle_tracks=subtitle_tracks,
        recommended_audio_stream=recommended_audio,
        existing_zh_subtitles=existing_zh,
        embedded_zh_streams=embedded_zh_streams,
        embedded_source_text_streams=embedded_source_text_streams,
    )


def extract_embedded_subtitle(
    video_path: str | Path,
    stream_index: int,
    work_dir: Path,
) -> list[TranscriptSegment]:
    ffmpeg = get_ffmpeg()
    out_srt = work_dir / f"embedded_{stream_index}.srt"
    cmd = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-map",
        f"0:{stream_index}",
        "-c:s",
        "srt",
        str(out_srt),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not out_srt.is_file():
        raise RuntimeError(f"提取内嵌字幕流 #{stream_index} 失败: {proc.stderr.strip()}")

    subs = pysubs2.load(str(out_srt), encoding="utf-8")
    segments: list[TranscriptSegment] = []
    idx = 1
    for event in subs:
        if event.is_comment:
            continue
        text = ASS_TAG_RE.sub("", event.plaintext).replace("\n", " ").strip()
        if not text:
            continue
        start = max(0.0, event.start / 1000.0)
        end = max(start + 0.1, event.end / 1000.0)
        segments.append(TranscriptSegment(id=idx, start=start, end=end, text=text))
        idx += 1
    return segments
