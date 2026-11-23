from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

TranslatorMode = Literal["local", "cloud"]
OutputFormat = Literal["ass", "srt"]
SubtitleStyle = Literal["bilingual", "zh_only"]
JobStage = Literal[
    "queued",
    "probing",
    "downloading_model",
    "extracting_audio",
    "vad",
    "asr",
    "segmenting",
    "translating",
    "rendering",
    "done",
    "cancelled",
    "error",
]


class AudioTrack(BaseModel):
    stream_index: int
    codec: str = ""
    language: str = ""
    title: str = ""
    channels: int = 2
    channel_layout: str = ""
    sample_rate: int = 0
    is_default: bool = False
    is_commentary: bool = False


class SubtitleTrack(BaseModel):
    stream_index: int
    codec: str = ""
    language: str = ""
    title: str = ""
    is_default: bool = False
    is_text_based: bool = False


class ProbeResult(BaseModel):
    video_path: str
    duration_sec: float = 0.0
    audio_tracks: list[AudioTrack] = Field(default_factory=list)
    subtitle_tracks: list[SubtitleTrack] = Field(default_factory=list)
    recommended_audio_stream: int | None = None
    existing_zh_subtitles: list[str] = Field(default_factory=list)
    embedded_zh_streams: list[int] = Field(default_factory=list)
    embedded_source_text_streams: list[int] = Field(default_factory=list)


class WordTiming(BaseModel):
    word: str
    start: float
    end: float
    probability: float = 1.0


class TranscriptSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: list[WordTiming] = Field(default_factory=list)


class TranslatedSegment(BaseModel):
    id: int
    start: float
    end: float
    source_text: str
    target_text: str


class TranslatorConfig(BaseModel):
    mode: TranslatorMode = "local"
    local_model: str = "auto"
    cloud_base_url: str = "https://api.deepseek.com/v1"
    cloud_api_key: str = ""
    cloud_model: str = "deepseek-chat"
    temperature: float = 0.2
    batch_size: int = 25
    context_before: int = 8
    context_after: int = 2
    max_concurrency: int = 3


class GenerateRequest(BaseModel):
    video_path: str
    source_lang: str = "ja"
    target_lang: str = "zh-Hans"
    audio_track_index: int | None = None
    subtitle_style: SubtitleStyle = "bilingual"
    output_format: OutputFormat = "ass"
    asr_model: str = "mlx-community/whisper-large-v3-turbo"
    translator: TranslatorConfig = Field(default_factory=TranslatorConfig)
    use_embedded_source_sub: bool = True
    force_regenerate: bool = False
    ignore_cache: bool = False
    hf_mirror: bool = False
    center_channel_boost: bool = True
    enable_vad: bool = True


class JobState(BaseModel):
    job_id: str
    video_path: str
    stage: JobStage = "queued"
    progress: float = 0.0
    message: str = "等待开始"
    detail: str = ""
    subtitle_path: str | None = None
    error: str | None = None
    reused_existing: bool = False
    used_embedded_sub: bool = False
    segments_count: int = 0
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
