from __future__ import annotations

from openai import OpenAI

from iina_autosub.config import sync_macos_system_proxy
from iina_autosub.models import TranslatorConfig
from iina_autosub.translators.base import BaseTranslator


class OpenAICompatTranslator(BaseTranslator):
    """Cloud API translator supporting any OpenAI-compatible endpoint (DeepSeek/OpenAI/Gemini/Claude)."""

    supports_parallel: bool = True

    def __init__(self, config: TranslatorConfig):
        super().__init__(config)
        sync_macos_system_proxy()
        api_key = (config.cloud_api_key or "").strip()
        base_url = (config.cloud_base_url or "").strip()
        if not api_key:
            raise ValueError("已选择云端 API 翻译模式，但未配置 API Key。")
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=90.0,
            max_retries=2,
        )
        self._model = (config.cloud_model or "deepseek-chat").strip()

    def chat_complete(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            # Some OpenAI-compatible gateways reject response_format; retry without it
            if "response_format" in str(exc).lower() or "json_object" in str(exc).lower():
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=max_tokens,
                )
            else:
                raise
        return resp.choices[0].message.content or ""

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
