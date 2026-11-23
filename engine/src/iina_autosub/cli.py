from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from pathlib import Path

from iina_autosub import __version__
from iina_autosub.config import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    detect_memory_gb,
    is_apple_silicon,
    recommend_local_llm_repo,
)
from iina_autosub.ffmpeg_bin import get_ffmpeg_binaries
from iina_autosub.models import GenerateRequest, TranslatorConfig
from iina_autosub.pipeline.orchestrator import run_generation_pipeline
from iina_autosub.pipeline.probe import probe_video
from iina_autosub.pipeline.vad import ensure_vad_model
from iina_autosub.server import run_server


def _cmd_doctor(_args: argparse.Namespace) -> int:
    ram = detect_memory_gb()
    print(f"iina-autosub v{__version__}")
    print(f"  Apple Silicon : {'Yes' if is_apple_silicon() else 'No'}")
    print(f"  Unified RAM   : {ram} GB")
    print(f"  推荐本地模型   : {recommend_local_llm_repo(ram)}")

    try:
        ffmpeg, ffprobe = get_ffmpeg_binaries()
        print(f"  ffmpeg        : {ffmpeg}")
        print(f"  ffprobe       : {ffprobe}")
    except Exception as exc:
        print(f"  ffmpeg        : ERROR ({exc})")
        return 1

    iina_app = Path("/Applications/IINA.app")
    print(f"  IINA.app      : {'Installed' if iina_app.is_dir() else 'Not found in /Applications'}")

    for mod in ("mlx.core", "mlx_whisper", "mlx_lm", "onnxruntime", "pysubs2", "fastapi", "openai"):
        try:
            __import__(mod)
            print(f"  module {mod:12s}: OK")
        except Exception as exc:
            print(f"  module {mod:12s}: MISSING ({exc})")
            return 1
    return 0


def _cmd_fetch_assets(args: argparse.Namespace) -> int:
    ffmpeg, ffprobe = get_ffmpeg_binaries()
    print(f"ffmpeg ready : {ffmpeg}")
    print(f"ffprobe ready: {ffprobe}")
    vad_path = ensure_vad_model(hf_mirror=args.hf_mirror, progress_cb=print)
    print(f"VAD model    : {vad_path}")
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    result = probe_video(args.video, source_lang=args.source_lang)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    req = GenerateRequest(
        video_path=args.video,
        source_lang=args.source_lang,
        audio_track_index=args.audio_track,
        subtitle_style=args.style,
        output_format=args.format,
        asr_model=args.asr_model,
        use_embedded_source_sub=not args.no_embedded_sub,
        force_regenerate=args.force,
        ignore_cache=args.no_cache,
        hf_mirror=args.hf_mirror,
        enable_vad=not args.no_vad,
        translator=TranslatorConfig(
            mode=args.mode,
            local_model=args.local_model,
            cloud_base_url=args.cloud_base_url,
            cloud_api_key=args.cloud_api_key,
            cloud_model=args.cloud_model,
        ),
    )

    cancel_event = threading.Event()

    def on_progress(stage: str, progress: float, message: str, detail: str) -> None:
        pct = int(round(progress * 100))
        suffix = f" ({detail})" if detail else ""
        print(f"[{pct:3d}%] [{stage}] {message}{suffix}", flush=True)

    result = run_generation_pipeline(req, cancel_event, on_progress)
    sub_path = str(result["subtitle_path"])
    print(f"\n完成！字幕已保存至: {sub_path}")

    if args.open_iina:
        subprocess.Popen(["open", "-a", "IINA", str(Path(args.video).expanduser().resolve())])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="iina-autosub", description="IINA 日→中一次性离线字幕生成引擎")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doc = sub.add_parser("doctor", help="检查系统环境、Apple Silicon 与依赖")
    p_doc.set_defaults(func=_cmd_doctor)

    p_fetch = sub.add_parser("fetch-assets", help="预下载静态 ffmpeg 与轻量 Silero VAD 模型")
    p_fetch.add_argument("--hf-mirror", action="store_true", help="使用国内 HF 镜像源")
    p_fetch.set_defaults(func=_cmd_fetch_assets)

    p_serve = sub.add_parser("serve", help="启动本地 HTTP 引擎服务")
    p_serve.add_argument("--host", default=DEFAULT_HOST)
    p_serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    p_serve.set_defaults(func=lambda a: (run_server(a.host, a.port) or 0))

    p_probe = sub.add_parser("probe", help="解析电影音轨与字幕流")
    p_probe.add_argument("video", help="电影文件路径")
    p_probe.add_argument("--source-lang", default="ja")
    p_probe.set_defaults(func=_cmd_probe)

    p_gen = sub.add_parser("generate", help="为单部电影一次性生成中文字幕")
    p_gen.add_argument("video", help="电影文件路径")
    p_gen.add_argument("--source-lang", default="ja")
    p_gen.add_argument("--audio-track", type=int, default=None, help="指定 ffprobe 音轨流编号")
    p_gen.add_argument("--style", choices=["bilingual", "zh_only"], default="bilingual")
    p_gen.add_argument("--format", choices=["ass", "srt"], default="ass")
    p_gen.add_argument("--asr-model", default="mlx-community/whisper-large-v3-turbo")
    p_gen.add_argument("--mode", choices=["local", "cloud"], default="local")
    p_gen.add_argument("--local-model", default="auto", help="auto / light-7b / balanced-8b / pro-14b / ultra-32b 或任意 HF MLX 仓库名")
    p_gen.add_argument("--cloud-base-url", default="https://api.deepseek.com/v1")
    p_gen.add_argument("--cloud-api-key", default="")
    p_gen.add_argument("--cloud-model", default="deepseek-chat")
    p_gen.add_argument("--hf-mirror", action="store_true")
    p_gen.add_argument("--no-embedded-sub", action="store_true")
    p_gen.add_argument("--no-vad", action="store_true")
    p_gen.add_argument("--no-cache", action="store_true")
    p_gen.add_argument("--force", action="store_true", help="即使同目录已有中字也强制重新生成")
    p_gen.add_argument("--open-iina", action="store_true", help="生成后用 IINA 打开影片")
    p_gen.set_defaults(func=_cmd_generate)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
