from __future__ import annotations

import gc
import threading
from collections.abc import Callable
from typing import Any

from iina_autosub.config import resolve_local_llm_repo
from iina_autosub.hf_download import ensure_hf_repo_downloaded
from iina_autosub.models import TranslatorConfig
from iina_autosub.translators.base import BaseTranslator


class MLXLocalTranslator(BaseTranslator):
    """Offline Apple Silicon MLX-LM translator with thinking-mode suppression & memory cleanup."""

    supports_parallel: bool = False

    def __init__(
        self,
        config: TranslatorConfig,
        hf_mirror: bool = False,
        download_cb: Callable[[float, str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ):
        super().__init__(config)
        self.hf_mirror = hf_mirror
        self.repo_id = resolve_local_llm_repo(config.local_model)
        self._download_cb = download_cb
        self._cancel_event = cancel_event
        self._model: Any = None
        self._tokenizer: Any = None

    def prepare(self) -> None:
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        local_model_path = ensure_hf_repo_downloaded(
            self.repo_id,
            hf_mirror=self.hf_mirror,
            cancel_event=self._cancel_event,
            progress_cb=self._download_cb,
        )
        from mlx_lm import load

        self._model, self._tokenizer = load(local_model_path)

    def _build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self._tokenizer, "apply_chat_template"):
            # Disable Qwen3 thinking mode when supported for 3-5x faster clean JSON output
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        return f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"

    def chat_complete(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        self._ensure_loaded()
        from mlx_lm import generate

        prompt = self._build_prompt(system_prompt, user_prompt)
        gen_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "verbose": False,
        }

        try:
            from mlx_lm.sample_utils import make_sampler

            gen_kwargs["sampler"] = make_sampler(temp=self.config.temperature, top_p=0.9)
        except Exception:
            gen_kwargs["temp"] = self.config.temperature

        return str(generate(self._model, self._tokenizer, **gen_kwargs))

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import mlx.core as mx

            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass
