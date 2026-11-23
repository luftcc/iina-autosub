from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path
from collections.abc import Callable

from iina_autosub.cache import load_cached_transcript, save_cached_transcript
from iina_autosub.config import TMP_WORK_DIR, ensure_dirs, resolve_asr_repo
from iina_autosub.models import GenerateRequest, JobStage, TranscriptSegment
from iina_autosub.pipeline.asr import transcribe_windows
from iina_autosub.pipeline.audio import CancelledError, extract_audio_wav
from iina_autosub.pipeline.langs import get_language_profile
from iina_autosub.pipeline.probe import extract_embedded_subtitle, probe_video
from iina_autosub.pipeline.segmenter import resegment_transcript
from iina_autosub.pipeline.subtitle import build_output_path, render_subtitles
from iina_autosub.pipeline.vad import (
    SileroOnnxVAD,
    ensure_vad_model,
    fallback_fixed_windows,
    group_speech_windows,
    read_wav_float32,
)
from iina_autosub.translators import create_translator

ProgressCallback = Callable[[JobStage, float, str, str], None]


def run_generation_pipeline(
    req: GenerateRequest,
    cancel_event: threading.Event,
    on_progress: ProgressCallback,
) -> dict[str, object]:
    """Execute end-to-end one-shot subtitle generation and return result metadata."""
    ensure_dirs()
    profile = get_language_profile(req.source_lang)

    def check_cancel() -> None:
        if cancel_event.is_set():
            raise CancelledError("任务已取消")

    def report(stage: JobStage, progress: float, message: str, detail: str = "") -> None:
        on_progress(stage, max(0.0, min(1.0, progress)), message, detail)

    check_cancel()
    report("probing", 0.02, "正在分析影片音轨与字幕...", Path(req.video_path).name)
    probe = probe_video(req.video_path, source_lang=req.source_lang)

    # 1. Reuse existing external Chinese subtitle unless user forced regeneration
    if not req.force_regenerate and probe.existing_zh_subtitles:
        existing = probe.existing_zh_subtitles[0]
        return {
            "subtitle_path": existing,
            "message": "检测到已有中文字幕，直接加载",
            "detail": Path(existing).name,
            "reused_existing": True,
            "used_embedded_sub": False,
            "segments_count": 0,
        }

    work_dir = Path(tempfile.mkdtemp(prefix="job_", dir=str(TMP_WORK_DIR)))
    try:
        segments: list[TranscriptSegment] | None = None
        used_embedded_sub = False

        # 2. Fast-path: if movie has embedded Japanese text subtitles, skip ASR and translate directly
        if req.use_embedded_source_sub and probe.embedded_source_text_streams:
            stream_idx = probe.embedded_source_text_streams[0]
            try:
                report(
                    "extracting_audio",
                    0.12,
                    "检测到内嵌日语字幕，直接提取跳过语音识别...",
                    f"字幕流 #{stream_idx}",
                )
                extracted = extract_embedded_subtitle(probe.video_path, stream_idx, work_dir)
                if len(extracted) >= 5:
                    segments = extracted
                    used_embedded_sub = True
            except Exception:
                segments = None

        # 3. Audio -> VAD -> ASR path (with ASR disk cache)
        if segments is None:
            chosen_stream = (
                req.audio_track_index
                if req.audio_track_index is not None
                else probe.recommended_audio_stream
            )
            chosen_track = next(
                (t for t in probe.audio_tracks if t.stream_index == chosen_stream),
                None,
            )
            asr_repo = resolve_asr_repo(req.asr_model)

            if not req.ignore_cache:
                cached = load_cached_transcript(
                    probe.video_path, chosen_stream, asr_repo, profile.code
                )
                if cached:
                    report(
                        "asr",
                        0.58,
                        f"命中语音识别缓存（共 {len(cached)} 句），跳过识别",
                        asr_repo.split("/")[-1],
                    )
                    segments = cached

            if segments is None:
                check_cancel()
                track_desc = (
                    f"音轨 #{chosen_stream} ({chosen_track.language or 'unknown'}, {chosen_track.channels}ch)"
                    if chosen_track
                    else "默认音轨"
                )
                report("extracting_audio", 0.05, "正在提取电影音轨...", track_desc)
                wav_path = extract_audio_wav(
                    probe.video_path,
                    work_dir / "audio_16k.wav",
                    audio_stream_index=chosen_stream,
                    channels=chosen_track.channels if chosen_track else 2,
                    channel_layout=chosen_track.channel_layout if chosen_track else "",
                    center_channel_boost=req.center_channel_boost,
                    cancel_event=cancel_event,
                )

                check_cancel()
                audio, sr = read_wav_float32(wav_path)
                audio_dur = len(audio) / float(sr) if sr else 0.0

                if req.enable_vad and audio_dur > 0:
                    vad_model = ensure_vad_model(
                        hf_mirror=req.hf_mirror,
                        progress_cb=lambda msg: report("downloading_model", 0.09, msg),
                    )
                    report("vad", 0.10, "正在检测电影有效人声区间（Silero VAD）...")
                    detector = SileroOnnxVAD(vad_model)
                    spans, probs = detector.detect_speech_spans(
                        audio,
                        sample_rate=sr,
                        cancel_event=cancel_event,
                        progress_cb=lambda p: report(
                            "vad",
                            0.10 + 0.06 * p,
                            f"语音区间检测中 {int(p * 100)}%",
                        ),
                    )
                    windows = group_speech_windows(spans, probs=probs)
                    if not windows and audio_dur > 0.5:
                        windows = fallback_fixed_windows(audio_dur)
                else:
                    windows = fallback_fixed_windows(audio_dur)

                check_cancel()
                report(
                    "asr",
                    0.16,
                    f"开始日语语音识别（共 {len(windows)} 个语音段）...",
                    asr_repo.split("/")[-1],
                )
                raw_segments = transcribe_windows(
                    audio=audio,
                    windows=windows,
                    profile=profile,
                    asr_model=asr_repo,
                    hf_mirror=req.hf_mirror,
                    cancel_event=cancel_event,
                    progress_cb=lambda p, msg: report("asr", 0.16 + 0.42 * p, msg, asr_repo),
                    download_cb=lambda p, msg: report(
                        "downloading_model", 0.16 + 0.02 * p, msg, asr_repo
                    ),
                )

                check_cancel()
                report("segmenting", 0.59, "正在优化日语断句与时间轴...")
                segments = resegment_transcript(raw_segments, profile)
                if segments:
                    save_cached_transcript(
                        probe.video_path, chosen_stream, asr_repo, profile.code, segments
                    )

        if not segments:
            raise RuntimeError("未在影片中检测到可识别的日语对白，请检查所选音轨。")

        # 4. Translate Japanese -> Simplified Chinese (Local MLX or Cloud API)
        check_cancel()
        mode_label = (
            "本地离线大模型" if req.translator.mode == "local" else f"云端 API ({req.translator.cloud_model})"
        )
        report("translating", 0.61, f"准备加载{mode_label}（共 {len(segments)} 句台词）...", mode_label)
        translator = create_translator(
            req.translator,
            hf_mirror=req.hf_mirror,
            download_cb=lambda p, msg: report(
                "downloading_model", 0.61 + 0.03 * p, msg, mode_label
            ),
            cancel_event=cancel_event,
        )
        translator.prepare()
        translated = translator.translate_segments(
            segments=segments,
            video_path=probe.video_path,
            profile=profile,
            cancel_event=cancel_event,
            progress_cb=lambda p, msg: report("translating", 0.65 + 0.31 * p, msg, mode_label),
        )

        # 5. Render IINA-adapted subtitle file alongside the movie
        check_cancel()
        out_path = build_output_path(
            probe.video_path,
            profile=profile,
            subtitle_style=req.subtitle_style,
            output_format=req.output_format,
        )
        report("rendering", 0.97, "正在生成 IINA 适配字幕文件...", out_path.name)
        saved_path = render_subtitles(
            segments=translated,
            output_path=out_path,
            profile=profile,
            subtitle_style=req.subtitle_style,
            output_format=req.output_format,
        )

        return {
            "subtitle_path": str(saved_path),
            "message": f"字幕已生成（共 {len(translated)} 句）",
            "detail": saved_path.name,
            "reused_existing": False,
            "used_embedded_sub": used_embedded_sub,
            "segments_count": len(translated),
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
