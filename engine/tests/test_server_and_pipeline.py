from __future__ import annotations

import subprocess
import time
from pathlib import Path

from fastapi.testclient import TestClient

from iina_autosub.ffmpeg_bin import get_ffmpeg
from iina_autosub.models import TranscriptSegment, TranslatedSegment
from iina_autosub.server import create_app


def _create_movie_with_embedded_ja(tmp_path: Path) -> Path:
    ffmpeg = get_ffmpeg()
    srt = tmp_path / "embedded.srt"
    srt.write_text(
        "\n".join(
            [
                "1\n00:00:00,200 --> 00:00:01,000\nおはようございます。",
                "2\n00:00:01,100 --> 00:00:02,000\n今日はいい天気ですね。",
                "3\n00:00:02,100 --> 00:00:03,000\n学校に行きましょう。",
                "4\n00:00:03,100 --> 00:00:04,000\nちょっと待ってください。",
                "5\n00:00:04,100 --> 00:00:05,000\nありがとうございます。",
            ]
        ),
        encoding="utf-8",
    )
    movie = tmp_path / "demo_film.mkv"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x180:d=5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=5",
            "-i",
            str(srt),
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-map",
            "2:s",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            "-c:s",
            "srt",
            "-metadata:s:a:0",
            "language=jpn",
            "-metadata:s:s:0",
            "language=jpn",
            str(movie),
        ],
        check=True,
    )
    return movie


def test_server_health_probe_and_end_to_end_job(tmp_path: Path, monkeypatch) -> None:
    movie = _create_movie_with_embedded_ja(tmp_path)

    class FakeTranslator:
        def prepare(self):
            pass

        def translate_segments(self, segments: list[TranscriptSegment], **_kwargs):
            return [
                TranslatedSegment(
                    id=s.id,
                    start=s.start,
                    end=s.end,
                    source_text=s.text,
                    target_text=f"中文字幕 {s.id}",
                )
                for s in segments
            ]

    monkeypatch.setattr(
        "iina_autosub.pipeline.orchestrator.create_translator",
        lambda _cfg, **_kwargs: FakeTranslator(),
    )

    app = create_app()
    with TestClient(app) as client:
        dash = client.get("/")
        assert dash.status_code == 200
        assert "iina-autosub" in dash.text

        h = client.get("/health")
        assert h.status_code == 200
        hdata = h.json()
        assert hdata["status"] == "ok"
        assert hdata["memory_gb"] >= 1
        assert len(hdata["local_llm_presets"]) >= 3
        assert "cached" in hdata["local_llm_presets"][0]
        assert "size_hint" in hdata["local_llm_presets"][0]

        p = client.get("/probe", params={"path": str(movie), "source_lang": "ja"})
        assert p.status_code == 200
        assert len(p.json()["embedded_source_text_streams"]) == 1

        resp = client.post(
            "/jobs",
            json={
                "video_path": str(movie),
                "source_lang": "ja",
                "subtitle_style": "bilingual",
                "output_format": "ass",
            },
        )
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        # Poll until done and verify the VERY FIRST done state already carries subtitle_path atomically
        state = resp.json()
        for _ in range(50):
            r = client.get(f"/jobs/{job_id}")
            assert r.status_code == 200
            state = r.json()
            if state["stage"] in {"done", "error", "cancelled"}:
                assert state["subtitle_path"], "done stage emitted before subtitle_path was populated!"
                break
            time.sleep(0.05)

        assert state["stage"] == "done", state
        assert state["used_embedded_sub"] is True
        assert state["segments_count"] == 5
        sub_path = Path(state["subtitle_path"])
        assert sub_path.is_file()
        assert sub_path.name == "demo_film.zh-ja.ass"
        assert "中文字幕 1" in sub_path.read_text(encoding="utf-8")

        # Submitting again without force_regenerate should immediately reuse existing subtitle via SSE
        resp2 = client.post("/jobs", json={"video_path": str(movie)})
        job_id2 = resp2.json()["job_id"]
        done_event = None
        with client.stream("GET", f"/jobs/{job_id2}/events") as sse:
            for line in sse.iter_lines():
                if line.startswith("data:"):
                    import json as _json

                    ev = _json.loads(line[5:].strip())
                    if ev["stage"] == "done":
                        done_event = ev
                        break
        assert done_event is not None
        assert done_event["reused_existing"] is True
        assert done_event["subtitle_path"] == str(sub_path)
