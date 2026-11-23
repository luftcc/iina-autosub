from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from iina_autosub.ffmpeg_bin import get_ffmpeg
from iina_autosub.pipeline.vad import (
    SAMPLE_RATE,
    SileroOnnxVAD,
    SpeechSpan,
    ensure_vad_model,
    fallback_fixed_windows,
    group_speech_windows,
    read_wav_float32,
)


def test_group_speech_windows_merges_and_splits() -> None:
    spans = [
        SpeechSpan(0.5, 4.0),
        SpeechSpan(4.8, 10.0),  # gap 0.8s -> merges with previous
        SpeechSpan(20.0, 25.0),  # gap 10s -> separate window
        SpeechSpan(30.0, 95.0),  # 65s continuous -> must split <= 26s
    ]
    # Construct fake probabilities with clear dips around 50s and 72s
    n_chunks = int(100 * SAMPLE_RATE / 512)
    probs = np.full(n_chunks, 0.9, dtype=np.float32)
    probs[int(50.0 * SAMPLE_RATE / 512)] = 0.05
    probs[int(72.0 * SAMPLE_RATE / 512)] = 0.05

    windows = group_speech_windows(spans, probs=probs, max_window_sec=26.0, max_gap_sec=1.8)
    assert windows[0] == SpeechSpan(0.5, 10.0)
    assert windows[1] == SpeechSpan(20.0, 25.0)
    # Every split window respects max_window_sec
    for win in windows:
        assert win.end_sec - win.start_sec <= 26.1
    # Ensure coverage of the long span is contiguous
    long_windows = [w for w in windows if w.start_sec >= 29.9]
    assert len(long_windows) >= 3
    assert long_windows[0].start_sec == 30.0
    assert long_windows[-1].end_sec == 95.0


def test_fallback_fixed_windows() -> None:
    windows = fallback_fixed_windows(65.0, window_sec=28.0)
    assert len(windows) == 3
    assert windows[0] == SpeechSpan(0.0, 28.0)
    assert windows[-1] == SpeechSpan(56.0, 65.0)


def test_real_silero_onnx_vad_on_japanese_speech(tmp_path: Path) -> None:
    if shutil.which("say") is None:
        return
    aiff = tmp_path / "ja.aiff"
    wav = tmp_path / "ja_16k.wav"
    subprocess.run(
        [
            "say",
            "-v",
            "Kyoko",
            "おはようございます。今日はとてもいい天気ですね。",
            "-o",
            str(aiff),
        ],
        check=True,
    )
    ffmpeg = get_ffmpeg()
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(aiff),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(wav),
        ],
        check=True,
    )
    speech, sr = read_wav_float32(wav)
    # Surround speech with 2 seconds of silence on both ends
    silence = np.zeros(2 * sr, dtype=np.float32)
    combined = np.concatenate([silence, speech, silence])

    model_path = ensure_vad_model(hf_mirror=False)
    vad = SileroOnnxVAD(model_path)
    spans, probs = vad.detect_speech_spans(combined, sample_rate=sr)

    assert len(spans) >= 1
    assert spans[0].start_sec >= 1.4
    assert spans[-1].end_sec <= (len(combined) / sr) - 1.2
    assert float(probs.max()) > 0.75

