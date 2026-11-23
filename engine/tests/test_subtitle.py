from __future__ import annotations

from pathlib import Path

import pysubs2

from iina_autosub.models import TranslatedSegment
from iina_autosub.pipeline.langs import get_language_profile
from iina_autosub.pipeline.subtitle import build_output_path, render_subtitles


def test_render_bilingual_ass_for_iina(tmp_path: Path) -> None:
    ja = get_language_profile("ja")
    video = tmp_path / "Spirited.Away.2001.mkv"
    out_path = build_output_path(video, ja, "bilingual", "ass")
    assert out_path.name == "Spirited.Away.2001.zh-ja.ass"

    segs = [
        TranslatedSegment(
            id=1,
            start=1.25,
            end=3.5,
            source_text="千尋、{早く}来て！",
            target_text="千寻，快过来！",
        ),
    ]
    saved = render_subtitles(segs, out_path, ja, subtitle_style="bilingual", output_format="ass")
    assert saved.is_file()

    loaded = pysubs2.load(str(saved), encoding="utf-8")
    assert loaded.info["PlayResX"] == "1920"
    assert loaded.info["PlayResY"] == "1080"
    assert loaded.styles["ZH"].fontname == "PingFang SC"
    assert loaded.styles["SRC"].fontname == "Hiragino Sans"
    assert len(loaded.events) == 1
    assert loaded.events[0].text == "千寻，快过来！\\N{\\rSRC}千尋、｛早く｝来て！"


def test_render_zh_only_srt(tmp_path: Path) -> None:
    ja = get_language_profile("ja")
    video = tmp_path / "movie.mp4"
    out_path = build_output_path(video, ja, "zh_only", "srt")
    assert out_path.name == "movie.zh-Hans.srt"

    segs = [
        TranslatedSegment(id=1, start=0.5, end=2.0, source_text="ありがとう", target_text="谢谢你")
    ]
    saved = render_subtitles(segs, out_path, ja, subtitle_style="zh_only", output_format="srt")
    content = saved.read_text(encoding="utf-8")
    assert "谢谢你" in content
    assert "ありがとう" not in content
