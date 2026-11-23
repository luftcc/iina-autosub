from __future__ import annotations

import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

import httpx
import numpy as np
import onnxruntime as ort

from iina_autosub.config import (
    SILERO_VAD_FILENAME,
    SILERO_VAD_MIRROR_URLS,
    SILERO_VAD_URLS,
    VAD_MODEL_DIR,
    ensure_dirs,
    sync_macos_system_proxy,
)
from iina_autosub.pipeline.audio import CancelledError

SAMPLE_RATE = 16000
WINDOW_SIZE = 512  # 32ms @ 16kHz
CONTEXT_SIZE = 64  # Silero VAD v5 required prefix context


@dataclass(frozen=True)
class SpeechSpan:
    start_sec: float
    end_sec: float


def ensure_vad_model(
    hf_mirror: bool = False,
    progress_cb: Callable[[str], None] | None = None,
) -> Path:
    """Download Silero VAD v5 ONNX (~2MB) once if not already cached."""
    ensure_dirs()
    sync_macos_system_proxy()
    model_path = VAD_MODEL_DIR / SILERO_VAD_FILENAME
    if model_path.is_file() and model_path.stat().st_size > 100_000:
        return model_path

    urls = SILERO_VAD_MIRROR_URLS if hf_mirror else SILERO_VAD_URLS
    last_err: Exception | None = None
    tmp_path = model_path.with_suffix(".onnx.tmp")

    for url in urls:
        try:
            if progress_cb:
                progress_cb(f"下载语音检测模型 silero_vad.onnx ({url.split('/')[2]})...")
            with httpx.Client(follow_redirects=True, timeout=60.0) as client:
                with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    with tmp_path.open("wb") as fh:
                        for chunk in resp.iter_bytes(chunk_size=65536):
                            fh.write(chunk)
            if tmp_path.stat().st_size > 100_000:
                tmp_path.replace(model_path)
                return model_path
        except Exception as exc:
            last_err = exc
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    raise RuntimeError(f"无法下载 Silero VAD 模型: {last_err}") from last_err


def read_wav_float32(wav_path: str | Path) -> tuple[np.ndarray, int]:
    """Read 16-bit mono PCM WAV directly into float32 numpy array normalized to [-1, 1]."""
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError(f"仅支持 16-bit PCM WAV，当前采样宽度为 {sampwidth * 8}-bit")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    return audio, sr


class SileroOnnxVAD:
    """Pure ONNXRuntime Silero VAD v5 detector (zero PyTorch dependency)."""

    def __init__(self, model_path: str | Path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        input_metas = {inp.name: inp for inp in self.session.get_inputs()}
        self._input_names = set(input_metas.keys())
        self._is_v5 = "state" in self._input_names
        sr_meta = input_metas.get("sr")
        self._sr_is_scalar = bool(sr_meta is not None and len(sr_meta.shape) == 0)

    def _compute_probs(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        cancel_event: threading.Event | None = None,
        progress_cb: Callable[[float], None] | None = None,
    ) -> np.ndarray:
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Silero VAD 期望 16kHz 音频，当前为 {sample_rate}Hz")

        n_samples = len(audio)
        if n_samples == 0:
            return np.zeros(0, dtype=np.float32)

        pad_len = (-n_samples) % WINDOW_SIZE
        if pad_len:
            audio = np.pad(audio, (0, pad_len), mode="constant")

        n_chunks = len(audio) // WINDOW_SIZE
        probs = np.empty(n_chunks, dtype=np.float32)

        sr_arr = (
            np.array(sample_rate, dtype=np.int64)
            if self._sr_is_scalar
            else np.array([sample_rate], dtype=np.int64)
        )

        if self._is_v5:
            state = np.zeros((2, 1, 128), dtype=np.float32)
            context = np.zeros(CONTEXT_SIZE, dtype=np.float32)
            for idx in range(n_chunks):
                if cancel_event is not None and (idx & 255) == 0 and cancel_event.is_set():
                    raise CancelledError("任务已取消")
                start = idx * WINDOW_SIZE
                chunk = audio[start : start + WINDOW_SIZE]
                x = np.concatenate([context, chunk])[np.newaxis, :].astype(np.float32)
                out, state = self.session.run(None, {"input": x, "state": state, "sr": sr_arr})
                probs[idx] = float(out[0][0])
                context = chunk[-CONTEXT_SIZE:]
                if progress_cb and (idx & 511) == 0:
                    progress_cb(idx / n_chunks)
        else:
            # Compatibility path if a v4 ONNX (h/c states) is ever supplied
            h = np.zeros((2, 1, 64), dtype=np.float32)
            c = np.zeros((2, 1, 64), dtype=np.float32)
            for idx in range(n_chunks):
                if cancel_event is not None and (idx & 255) == 0 and cancel_event.is_set():
                    raise CancelledError("任务已取消")
                start = idx * WINDOW_SIZE
                chunk = audio[start : start + WINDOW_SIZE][np.newaxis, :].astype(np.float32)
                out, h, c = self.session.run(
                    None, {"input": chunk, "h": h, "c": c, "sr": sr_arr}
                )
                probs[idx] = float(out[0][0])
                if progress_cb and (idx & 511) == 0:
                    progress_cb(idx / n_chunks)

        if progress_cb:
            progress_cb(1.0)
        return probs

    def detect_speech_spans(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        threshold: float = 0.45,
        neg_threshold: float = 0.30,
        min_speech_sec: float = 0.25,
        min_silence_sec: float = 0.35,
        speech_pad_sec: float = 0.20,
        cancel_event: threading.Event | None = None,
        progress_cb: Callable[[float], None] | None = None,
    ) -> tuple[list[SpeechSpan], np.ndarray]:
        probs = self._compute_probs(audio, sample_rate, cancel_event, progress_cb)
        if len(probs) == 0:
            return [], probs

        chunk_sec = WINDOW_SIZE / float(sample_rate)
        total_sec = len(audio) / float(sample_rate)
        min_silence_chunks = max(1, int(round(min_silence_sec / chunk_sec)))

        triggered = False
        speech_start_idx = 0
        silence_start_idx: int | None = None
        raw_spans: list[SpeechSpan] = []

        for idx, prob in enumerate(probs):
            if prob >= threshold:
                if not triggered:
                    triggered = True
                    speech_start_idx = idx
                silence_start_idx = None
            elif triggered and prob < neg_threshold:
                if silence_start_idx is None:
                    silence_start_idx = idx
                if idx - silence_start_idx >= min_silence_chunks:
                    start_s = speech_start_idx * chunk_sec
                    end_s = silence_start_idx * chunk_sec
                    if end_s - start_s >= min_speech_sec:
                        raw_spans.append(SpeechSpan(start_s, end_s))
                    triggered = False
                    silence_start_idx = None

        if triggered:
            start_s = speech_start_idx * chunk_sec
            end_s = (
                silence_start_idx * chunk_sec if silence_start_idx is not None else total_sec
            )
            if end_s - start_s >= min_speech_sec:
                raw_spans.append(SpeechSpan(start_s, end_s))

        # Apply padding and merge overlapping spans
        padded: list[SpeechSpan] = []
        for span in raw_spans:
            s = max(0.0, span.start_sec - speech_pad_sec)
            e = min(total_sec, span.end_sec + speech_pad_sec)
            if padded and s <= padded[-1].end_sec:
                padded[-1] = SpeechSpan(padded[-1].start_sec, max(padded[-1].end_sec, e))
            else:
                padded.append(SpeechSpan(s, e))

        return padded, probs


def group_speech_windows(
    spans: list[SpeechSpan],
    probs: np.ndarray | None = None,
    max_window_sec: float = 26.0,
    max_gap_sec: float = 1.8,
) -> list[SpeechSpan]:
    """Merge adjacent speech spans into <=26s Whisper-optimal chunks cut strictly on silence/dips."""
    if not spans:
        return []

    chunk_sec = WINDOW_SIZE / float(SAMPLE_RATE)

    def _split_long_span(span: SpeechSpan) -> list[SpeechSpan]:
        if span.end_sec - span.start_sec <= max_window_sec:
            return [span]
        pieces: list[SpeechSpan] = []
        cur = span.start_sec
        while span.end_sec - cur > max_window_sec:
            # Search for lowest speech probability in [cur + 14s, cur + max_window_sec]
            search_lo = cur + max_window_sec * 0.55
            search_hi = min(span.end_sec - 1.0, cur + max_window_sec)
            cut = search_hi
            if probs is not None and len(probs) > 0 and search_hi > search_lo:
                lo_idx = max(0, int(search_lo / chunk_sec))
                hi_idx = min(len(probs) - 1, int(search_hi / chunk_sec))
                if hi_idx > lo_idx:
                    rel_min = int(np.argmin(probs[lo_idx : hi_idx + 1]))
                    cut = (lo_idx + rel_min) * chunk_sec
            pieces.append(SpeechSpan(cur, cut))
            cur = cut
        if span.end_sec - cur > 0.2:
            pieces.append(SpeechSpan(cur, span.end_sec))
        return pieces

    normalized: list[SpeechSpan] = []
    for sp in spans:
        normalized.extend(_split_long_span(sp))

    grouped: list[SpeechSpan] = [normalized[0]]
    for sp in normalized[1:]:
        prev = grouped[-1]
        gap = sp.start_sec - prev.end_sec
        merged_len = sp.end_sec - prev.start_sec
        if gap <= max_gap_sec and merged_len <= max_window_sec:
            grouped[-1] = SpeechSpan(prev.start_sec, sp.end_sec)
        else:
            grouped.append(sp)
    return grouped


def fallback_fixed_windows(duration_sec: float, window_sec: float = 28.0) -> list[SpeechSpan]:
    windows: list[SpeechSpan] = []
    cur = 0.0
    while cur < duration_sec:
        end = min(duration_sec, cur + window_sec)
        if end - cur > 0.2:
            windows.append(SpeechSpan(cur, end))
        cur = end
    return windows
