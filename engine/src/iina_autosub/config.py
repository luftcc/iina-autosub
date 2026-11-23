from __future__ import annotations

import os
import platform
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_HOST = os.environ.get("IINA_AUTOSUB_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("IINA_AUTOSUB_PORT", "18421"))

CACHE_ROOT = Path(
    os.environ.get("IINA_AUTOSUB_CACHE_DIR", str(Path.home() / ".cache" / "iina-autosub"))
).expanduser()
VAD_MODEL_DIR = CACHE_ROOT / "vad"
ASR_CACHE_DIR = CACHE_ROOT / "asr_cache"
TMP_WORK_DIR = CACHE_ROOT / "tmp"
LOG_DIR = Path.home() / "Library" / "Logs" / "iina-autosub"

HF_MIRROR_ENDPOINT = "https://hf-mirror.com"

SILERO_VAD_FILENAME = "silero_vad.onnx"
SILERO_VAD_URLS = [
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
    "https://huggingface.co/runanywhere/silero-vad-v5/resolve/main/silero_vad.onnx",
]
SILERO_VAD_MIRROR_URLS = [
    "https://hf-mirror.com/runanywhere/silero-vad-v5/resolve/main/silero_vad.onnx",
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
]


@dataclass(frozen=True)
class ModelPreset:
    id: str
    label: str
    repo_id: str
    min_ram_gb: int
    size_hint: str
    description: str


ASR_MODEL_PRESETS: list[ModelPreset] = [
    ModelPreset(
        id="whisper-large-v3-turbo",
        label="Whisper Large v3 Turbo (推荐均衡)",
        repo_id="mlx-community/whisper-large-v3-turbo",
        min_ram_gb=8,
        size_hint="~1.6 GB",
        description="速度极快，词级时间轴稳定，兼顾日语与片中偶发英文对白",
    ),
    ModelPreset(
        id="whisper-large-v3-turbo-4bit",
        label="Whisper Large v3 Turbo 4-bit (极速省内存)",
        repo_id="mlx-community/whisper-large-v3-turbo-4bit",
        min_ram_gb=8,
        size_hint="~0.5 GB",
        description="更低内存占用与更快下载体积，适合 8GB/16GB 轻薄本",
    ),
    ModelPreset(
        id="kotoba-whisper-v2.1",
        label="Kotoba Whisper v2.1 (纯日语专精)",
        repo_id="enet45/kotoba-whisper-v2.1-mlx-fp16",
        min_ram_gb=8,
        size_hint="~1.5 GB",
        description="针对日语语料深度微调，纯日剧/日本电影识别字错率更低",
    ),
]

LOCAL_LLM_PRESETS: list[ModelPreset] = [
    ModelPreset(
        id="light-7b",
        label="轻量 7B (Qwen2.5-7B-4bit)",
        repo_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
        min_ram_gb=16,
        size_hint="~4.4 GB",
        description="约 4.5GB 显存占用，速度快、格式稳定，16GB Mac 流畅运行",
    ),
    ModelPreset(
        id="balanced-8b",
        label="均衡 8B (Qwen3-8B-4bit)",
        repo_id="mlx-community/Qwen3-8B-4bit",
        min_ram_gb=16,
        size_hint="~4.7 GB",
        description="新一代 Qwen3 8B 量化版（关闭思考模式），速度与口语译文均衡",
    ),
    ModelPreset(
        id="pro-14b",
        label="高质 14B (Qwen3-14B-4bit)",
        repo_id="mlx-community/Qwen3-14B-4bit",
        min_ram_gb=24,
        size_hint="~8.3 GB",
        description="上下文理解与长句表现优秀，推荐 24GB–64GB Mac",
    ),
    ModelPreset(
        id="ultra-32b",
        label="旗舰 32B (Qwen3-32B-4bit)",
        repo_id="mlx-community/Qwen3-32B-4bit",
        min_ram_gb=64,
        size_hint="~18.4 GB",
        description="接近云端顶级模型观感，适合 64GB/128GB 高配 Apple Silicon Mac",
    ),
]

LOCAL_LLM_PRESET_MAP: dict[str, str] = {p.id: p.repo_id for p in LOCAL_LLM_PRESETS}
ASR_PRESET_MAP: dict[str, str] = {p.id: p.repo_id for p in ASR_MODEL_PRESETS}


def ensure_dirs() -> None:
    for d in (CACHE_ROOT, VAD_MODEL_DIR, ASR_CACHE_DIR, TMP_WORK_DIR):
        d.mkdir(parents=True, exist_ok=True)


def detect_memory_gb() -> int:
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=2).strip()
        return max(1, round(int(out) / (1024**3)))
    except Exception:
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return max(1, round((pages * page_size) / (1024**3)))
    except Exception:
        return 16


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def recommend_local_llm_repo(mem_gb: int | None = None) -> str:
    ram = mem_gb if mem_gb is not None else detect_memory_gb()
    if ram >= 48:
        return LOCAL_LLM_PRESET_MAP["pro-14b"]
    if ram >= 22:
        return LOCAL_LLM_PRESET_MAP["balanced-8b"]
    return LOCAL_LLM_PRESET_MAP["light-7b"]


def resolve_local_llm_repo(model_spec: str, mem_gb: int | None = None) -> str:
    spec = (model_spec or "").strip()
    if not spec or spec == "auto":
        return recommend_local_llm_repo(mem_gb)
    if spec in LOCAL_LLM_PRESET_MAP:
        return LOCAL_LLM_PRESET_MAP[spec]
    return spec


def resolve_asr_repo(model_spec: str) -> str:
    spec = (model_spec or "").strip()
    if not spec or spec == "auto":
        return ASR_PRESET_MAP["whisper-large-v3-turbo"]
    if spec in ASR_PRESET_MAP:
        return ASR_PRESET_MAP[spec]
    return spec


def sync_macos_system_proxy() -> dict[str, str]:
    """Propagate live macOS System Settings proxy (e.g. Clash/Surge) into env for httpx/hf_hub/openai.

    LaunchAgents do not inherit shell HTTP_PROXY variables, and httpx/hf-xet do not read
    macOS SystemConfiguration directly. Syncing urllib.request.getproxies() solves this cleanly.
    """
    applied: dict[str, str] = {}
    # Respect explicit user environment overrides if already set externally
    if os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"):
        return applied

    try:
        proxies = urllib.request.getproxies()
    except Exception:
        return applied

    http_proxy = proxies.get("http")
    https_proxy = proxies.get("https") or http_proxy
    if https_proxy and https_proxy.startswith(("http://", "https://")):
        os.environ["HTTPS_PROXY"] = https_proxy
        os.environ["https_proxy"] = https_proxy
        applied["HTTPS_PROXY"] = https_proxy
    if http_proxy and http_proxy.startswith(("http://", "https://")):
        os.environ["HTTP_PROXY"] = http_proxy
        os.environ["http_proxy"] = http_proxy
        applied["HTTP_PROXY"] = http_proxy
    if applied:
        os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,::1")
        os.environ.setdefault("no_proxy", "127.0.0.1,localhost,::1")
    return applied


def configure_hf_endpoint(use_mirror: bool) -> str:
    """Apply Hugging Face mirror endpoint (including live constants patching for huggingface-hub 2.x)."""
    sync_macos_system_proxy()
    endpoint = HF_MIRROR_ENDPOINT if use_mirror else "https://huggingface.co"

    if use_mirror:
        os.environ["HF_ENDPOINT"] = HF_MIRROR_ENDPOINT
    elif os.environ.get("HF_ENDPOINT") == HF_MIRROR_ENDPOINT:
        os.environ.pop("HF_ENDPOINT", None)
    else:
        endpoint = os.environ.get("HF_ENDPOINT") or "https://huggingface.co"

    # Patch already-imported huggingface_hub constants so runtime toggles take effect immediately
    hf_constants = sys.modules.get("huggingface_hub.constants")
    if hf_constants is not None:
        hf_constants.ENDPOINT = endpoint
        hf_constants.HUGGINGFACE_CO_URL_HOME = f"{endpoint}/"
        hf_constants.HUGGINGFACE_CO_URL_TEMPLATE = (
            f"{endpoint}/{{repo_id}}/resolve/{{revision}}/{{filename}}"
        )
    for mod_name, mod in list(sys.modules.items()):
        if mod_name.startswith("huggingface_hub") and mod is not None:
            if hasattr(mod, "ENDPOINT"):
                mod.ENDPOINT = endpoint
            if hasattr(mod, "HUGGINGFACE_CO_URL_TEMPLATE"):
                mod.HUGGINGFACE_CO_URL_TEMPLATE = (
                    f"{endpoint}/{{repo_id}}/resolve/{{revision}}/{{filename}}"
                )
    return endpoint
