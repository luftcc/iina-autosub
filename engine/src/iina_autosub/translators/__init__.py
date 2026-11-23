from __future__ import annotations

import threading
from collections.abc import Callable

from iina_autosub.models import TranslatorConfig
from iina_autosub.translators.base import BaseTranslator
from iina_autosub.translators.mlx_local import MLXLocalTranslator
from iina_autosub.translators.openai_api import OpenAICompatTranslator


def create_translator(
    config: TranslatorConfig,
    hf_mirror: bool = False,
    download_cb: Callable[[float, str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> BaseTranslator:
    if config.mode == "cloud":
        return OpenAICompatTranslator(config)
    return MLXLocalTranslator(
        config,
        hf_mirror=hf_mirror,
        download_cb=download_cb,
        cancel_event=cancel_event,
    )
