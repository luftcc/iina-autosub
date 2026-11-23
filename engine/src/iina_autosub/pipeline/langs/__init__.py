from __future__ import annotations

from dataclasses import dataclass
import re

from iina_autosub.pipeline.langs.ja import JAPANESE_PROFILE


@dataclass(frozen=True)
class LanguageProfile:
    code: str
    whisper_lang: str | None
    iso_tags: frozenset[str]
    display_name: str
    secondary_font: str
    primary_font: str
    bilingual_suffix: str
    zh_only_suffix: str
    sentence_end_chars: str
    soft_break_chars: str
    max_source_chars: int
    min_duration_sec: float
    max_duration_sec: float
    pause_split_sec: float
    hallucination_regexes: tuple[re.Pattern[str], ...]
    whisper_initial_prompt: str
    glossary_system_prompt: str
    translation_system_prompt: str


LANGUAGE_PROFILES: dict[str, LanguageProfile] = {
    "ja": JAPANESE_PROFILE,
}


def get_language_profile(lang_code: str) -> LanguageProfile:
    normalized = (lang_code or "ja").strip().lower()
    if normalized in {"ja", "jp", "jpn", "japanese"}:
        return LANGUAGE_PROFILES["ja"]
    raise ValueError(f"当前暂不支持源语言 '{lang_code}'，首批支持日语 (ja)。")
