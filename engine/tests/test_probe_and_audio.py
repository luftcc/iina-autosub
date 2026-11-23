from __future__ import annotations

import subprocess
from pathlib import Path

from iina_autosub.ffmpeg_bin import get_ffmpeg
from iina_autosub.pipeline.audio import extract_audio_wav
from iina_autosub.pipeline.probe import (
    extract_embedded_subtitle,
    find_existing_zh_subtitles,
    probe_video,
)
from iina_autosub.pipeline.vad import read_wav_float32


def _make_sample_video(tmp_path: Path, channels: int = 2, with_ja_sub: bool = False) -> Path:
    ffmpeg = get_ffmpeg()
    wav = tmp_path / "tone.wav"
    layout = "5.1" if channels == 6 else "stereo"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration=2,aformat=channel_layouts={layout}",
            str(wav),
        ],
        check=True,
    )

    if with_ja_sub:
        srt = tmp_path / "ja.srt"
        srt.write_text(
            "1\n00:00:00,200 --> 00:00:01,500\nおはようございます。\n\n"
            "2\n00:00:01,600 --> 00:00:02,000\n今日はいい天気ですね。\n",
            encoding="utf-8",
        )
        out = tmp_path / "sample_ja.mkv"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=320x180:d=2",
                "-i",
                str(wav),
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
                str(out),
            ],
            check=True,
        )
        return out

    out = tmp_path / f"sample_{channels}ch.mkv"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x180:d=2",
            "-i",
            str(wav),
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            "-metadata:s:a:0",
            "language=jpn",
            str(out),
        ],
        check=True,
    )
    return out


def test_probe_and_embedded_ja_extract(tmp_path: Path) -> None:
    video = _make_sample_video(tmp_path, channels=2, with_ja_sub=True)
    result = probe_video(video, source_lang="ja")
    assert result.duration_sec >= 1.5
    assert len(result.audio_tracks) == 1
    assert result.recommended_audio_stream == result.audio_tracks[0].stream_index
    assert len(result.embedded_source_text_streams) == 1
    assert result.existing_zh_subtitles == []

    segs = extract_embedded_subtitle(video, result.embedded_source_text_streams[0], tmp_path)
    assert len(segs) == 2
    assert segs[0].text == "おはようございます。"
    assert segs[1].text == "今日はいい天気ですね。"


def test_extract_audio_51_center_boost(tmp_path: Path) -> None:
    video = _make_sample_video(tmp_path, channels=6, with_ja_sub=False)
    probe = probe_video(video, source_lang="ja")
    track = probe.audio_tracks[0]
    assert track.channels == 6

    wav = extract_audio_wav(
        video,
        tmp_path / "out_16k.wav",
        audio_stream_index=track.stream_index,
        channels=track.channels,
        channel_layout=track.channel_layout,
        center_channel_boost=True,
    )
    audio, sr = read_wav_float32(wav)
    assert sr == 16000
    assert len(audio) >= 16000


def test_find_existing_zh_subtitles(tmp_path: Path) -> None:
    movie = tmp_path / "Your.Name.2016.mkv"
    movie.write_bytes(b"0000")
    # Non-Chinese unrelated files should not match
    (tmp_path / "Your.Name.2016.en.srt").write_text("hello", encoding="utf-8")
    (tmp_path / "Your.Name.2016.school.srt").write_text("school", encoding="utf-8")
    assert find_existing_zh_subtitles(movie) == []

    zh = tmp_path / "Your.Name.2016.zh-ja.ass"
    zh.write_text("subtitle", encoding="utf-8")
    found = find_existing_zh_subtitles(movie)
    assert found == [str(zh)]
