from __future__ import annotations

from pathlib import Path

import pysubs2

from iina_autosub.models import OutputFormat, SubtitleStyle, TranslatedSegment
from iina_autosub.pipeline.langs import LanguageProfile


def _escape_ass_text(text: str) -> str:
    """Prevent literal braces or newlines in dialogue from breaking ASS override tags."""
    return (
        (text or "")
        .replace("\r\n", " ")
        .replace("\n", " ")
        .replace("{", "｛")
        .replace("}", "｝")
        .strip()
    )


def build_output_path(
    video_path: str | Path,
    profile: LanguageProfile,
    subtitle_style: SubtitleStyle,
    output_format: OutputFormat,
) -> Path:
    vpath = Path(video_path)
    suffix = (
        profile.bilingual_suffix if subtitle_style == "bilingual" else profile.zh_only_suffix
    )
    return vpath.with_name(f"{vpath.stem}.{suffix}.{output_format}")


def render_subtitles(
    segments: list[TranslatedSegment],
    output_path: str | Path,
    profile: LanguageProfile,
    subtitle_style: SubtitleStyle = "bilingual",
    output_format: OutputFormat = "ass",
) -> Path:
    """Render IINA-optimized .ass (with native macOS PingFang SC + Hiragana Sans) or .srt."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    subs = pysubs2.SSAFile()
    subs.info["Title"] = out_path.stem
    subs.info["ScriptType"] = "v4.00+"
    subs.info["PlayResX"] = "1920"
    subs.info["PlayResY"] = "1080"
    subs.info["WrapStyle"] = "0"
    subs.info["ScaledBorderAndShadow"] = "yes"

    # Primary Simplified Chinese style tuned for IINA + macOS Retina
    zh_style = pysubs2.SSAStyle(
        fontname=profile.primary_font,
        fontsize=58,
        primarycolor=pysubs2.Color(255, 255, 255, 0),
        secondarycolor=pysubs2.Color(255, 255, 255, 0),
        outlinecolor=pysubs2.Color(16, 16, 16, 25),
        backcolor=pysubs2.Color(0, 0, 0, 135),
        bold=False,
        borderstyle=1,
        outline=2.4,
        shadow=1.2,
        alignment=pysubs2.Alignment.BOTTOM_CENTER,
        marginl=80,
        marginr=80,
        marginv=48,
    )

    # Secondary Japanese style using macOS native Hiragino Sans
    src_style = pysubs2.SSAStyle(
        fontname=profile.secondary_font,
        fontsize=34,
        primarycolor=pysubs2.Color(228, 224, 210, 15),
        secondarycolor=pysubs2.Color(228, 224, 210, 15),
        outlinecolor=pysubs2.Color(16, 16, 16, 35),
        backcolor=pysubs2.Color(0, 0, 0, 145),
        bold=False,
        borderstyle=1,
        outline=1.8,
        shadow=1.0,
        alignment=pysubs2.Alignment.BOTTOM_CENTER,
        marginl=80,
        marginr=80,
        marginv=44,
    )

    subs.styles["ZH"] = zh_style
    subs.styles["SRC"] = src_style
    subs.styles["Default"] = zh_style

    for seg in segments:
        start_ms = max(0, int(round(seg.start * 1000)))
        end_ms = max(start_ms + 150, int(round(seg.end * 1000)))
        zh = _escape_ass_text(seg.target_text)
        src = _escape_ass_text(seg.source_text)
        if not zh and not src:
            continue

        if output_format == "ass":
            if subtitle_style == "bilingual" and src:
                # Single stacked event: Chinese prominently on top, Japanese neatly below
                text = f"{zh}\\N{{\\rSRC}}{src}"
            else:
                text = zh or src
            subs.events.append(
                pysubs2.SSAEvent(start=start_ms, end=end_ms, style="ZH", text=text)
            )
        else:
            if subtitle_style == "bilingual" and src:
                text = f"{zh}\\N{src}"
            else:
                text = zh or src
            subs.events.append(pysubs2.SSAEvent(start=start_ms, end=end_ms, text=text))

    tmp_path = out_path.with_suffix(f".{output_format}.tmp")
    subs.save(str(tmp_path), encoding="utf-8", format_=output_format)
    tmp_path.replace(out_path)
    return out_path
