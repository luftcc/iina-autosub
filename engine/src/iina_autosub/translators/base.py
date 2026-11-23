from __future__ import annotations

import json
import re
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from collections.abc import Callable

from iina_autosub.models import TranscriptSegment, TranslatedSegment, TranslatorConfig
from iina_autosub.pipeline.audio import CancelledError
from iina_autosub.pipeline.langs import LanguageProfile

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
UNCLOSED_THINK_RE = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def strip_thinking(text: str) -> str:
    cleaned = THINK_BLOCK_RE.sub("", text or "")
    cleaned = UNCLOSED_THINK_RE.sub("", cleaned)
    return cleaned.strip()


def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = strip_thinking(raw_text)
    fence_match = CODE_FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try direct parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Extract outermost {...}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidate = text[start : end + 1]
        candidate = TRAILING_COMMA_RE.sub(r"\1", candidate)
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"无法从模型响应中解析 JSON 对象: {text[:240]}")


def _kana_ratio(text: str) -> float:
    non_space = [ch for ch in text if not ch.isspace()]
    if not non_space:
        return 0.0
    kana = sum(1 for ch in non_space if "\u3040" <= ch <= "\u30ff")
    return kana / len(non_space)


def _needs_retranslation(source_text: str, target_text: str, source_lang: str) -> bool:
    t = (target_text or "").strip()
    if not t:
        return True
    if source_lang == "ja" and len(t) >= 4 and _kana_ratio(t) > 0.35:
        # Still mostly untranslated Japanese kana
        if _kana_ratio(source_text) > 0.15:
            return True
    return False


def _clean_zh_line(text: str) -> str:
    cleaned = " ".join((text or "").replace("\r", " ").replace("\n", " ").split()).strip()
    # Subtitle convention: drop trailing single Chinese period unless ellipsis
    if cleaned.endswith("。") and not cleaned.endswith("。。"):
        cleaned = cleaned[:-1].rstrip()
    return cleaned


def _clean_movie_title(video_path: str) -> str:
    stem = Path(video_path).stem
    # Strip common release tags like [1080p], (2023), BluRay, x265
    cleaned = re.sub(r"\[[^\]]*\]|\([^\)]*\)", " ", stem)
    cleaned = re.sub(
        r"\b(1080p|2160p|720p|4k|bluray|bdrip|webrip|web-dl|x264|x265|hevc|aac|flac|dts|remux)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[\._]+", " ", cleaned)
    return " ".join(cleaned.split()).strip() or stem


class BaseTranslator(ABC):
    supports_parallel: bool = False

    def __init__(self, config: TranslatorConfig):
        self.config = config

    @abstractmethod
    def chat_complete(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Execute a single chat completion and return assistant text."""

    def prepare(self) -> None:  # noqa: B027
        """Optional hook to download/load model weights before translation starts."""

    def close(self) -> None:  # noqa: B027
        """Release any held model/GPU resources."""

    def extract_glossary(
        self,
        movie_title: str,
        segments: list[TranscriptSegment],
        profile: LanguageProfile,
        cancel_event: threading.Event | None = None,
    ) -> tuple[dict[str, str], str]:
        if len(segments) < 10:
            return {}, ""
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("任务已取消")

        # Sample up to 40 representative lines across the whole film
        sample_limit = 40
        step = max(1, len(segments) // sample_limit)
        samples = [seg.text for seg in segments[::step][:sample_limit]]
        payload = {
            "movie_title": movie_title,
            "sample_lines": samples,
        }
        try:
            raw = self.chat_complete(
                system_prompt=profile.glossary_system_prompt,
                user_prompt=json.dumps(payload, ensure_ascii=False),
                max_tokens=1024,
            )
            data = extract_json_object(raw)
            glossary_raw = data.get("glossary") or {}
            glossary: dict[str, str] = {}
            if isinstance(glossary_raw, dict):
                for k, v in list(glossary_raw.items())[:25]:
                    ks, vs = str(k).strip(), str(v).strip()
                    if ks and vs:
                        glossary[ks] = vs
            tone = str(data.get("tone") or "").strip()[:120]
            return glossary, tone
        except CancelledError:
            raise
        except Exception:
            return {}, ""

    def _translate_single_batch(
        self,
        batch: list[TranscriptSegment],
        context_before: list[dict[str, str]],
        context_after: list[str],
        movie_title: str,
        glossary: dict[str, str],
        tone: str,
        profile: LanguageProfile,
        extra_instruction: str = "",
    ) -> dict[int, str]:
        payload: dict[str, Any] = {
            "movie_title": movie_title,
            "tone": tone,
            "glossary": glossary,
            "context_before": context_before,
            "items": [{"id": seg.id, profile.code: seg.text} for seg in batch],
            "context_after": context_after,
        }
        system_prompt = profile.translation_system_prompt
        if extra_instruction:
            system_prompt = f"{system_prompt}\n\n补充强调：{extra_instruction}"

        max_tokens = min(4096, max(768, len(batch) * 90))
        raw = self.chat_complete(
            system_prompt=system_prompt,
            user_prompt=json.dumps(payload, ensure_ascii=False),
            max_tokens=max_tokens,
        )
        parsed = extract_json_object(raw)
        items = parsed.get("items")
        if not isinstance(items, list):
            raise ValueError("JSON 响应缺少 items 数组")

        result: dict[int, str] = {}
        expected_ids = {seg.id for seg in batch}
        for entry in items:
            if not isinstance(entry, dict):
                continue
            raw_id = entry.get("id")
            if raw_id is None:
                continue
            try:
                seg_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if seg_id not in expected_ids:
                continue
            zh = _clean_zh_line(str(entry.get("zh") or entry.get("translation") or ""))
            if zh:
                result[seg_id] = zh
        return result

    def _translate_batch_with_retry(
        self,
        batch: list[TranscriptSegment],
        context_before: list[dict[str, str]],
        context_after: list[str],
        movie_title: str,
        glossary: dict[str, str],
        tone: str,
        profile: LanguageProfile,
        cancel_event: threading.Event | None = None,
    ) -> dict[int, str]:
        resolved: dict[int, str] = {}

        # Attempt 1: full batch
        try:
            first = self._translate_single_batch(
                batch, context_before, context_after, movie_title, glossary, tone, profile
            )
            for seg in batch:
                cand = first.get(seg.id, "")
                if cand and not _needs_retranslation(seg.text, cand, profile.code):
                    resolved[seg.id] = cand
        except CancelledError:
            raise
        except Exception:
            pass

        # Attempt 2: retry only missing or suspiciously untranslated items (split if large)
        missing = [seg for seg in batch if seg.id not in resolved]
        if missing:
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError("任务已取消")
            sub_size = max(1, min(10, (len(missing) + 1) // 2)) if len(missing) > 6 else len(missing)
            for i in range(0, len(missing), sub_size):
                sub = missing[i : i + sub_size]
                try:
                    retry_map = self._translate_single_batch(
                        sub,
                        context_before[-4:],
                        context_after[:2],
                        movie_title,
                        glossary,
                        tone,
                        profile,
                        extra_instruction="上次有遗漏或未译成简体中文的条目，请务必全部翻译为简体中文并严格保留每个 id。",
                    )
                    for seg in sub:
                        cand = retry_map.get(seg.id, "")
                        if cand and not _needs_retranslation(seg.text, cand, profile.code):
                            resolved[seg.id] = cand
                except CancelledError:
                    raise
                except Exception:
                    pass

        # Attempt 3: single-line fallback for any stubborn stragglers
        still_missing = [seg for seg in batch if seg.id not in resolved]
        for seg in still_missing:
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError("任务已取消")
            try:
                one_map = self._translate_single_batch(
                    [seg],
                    context_before[-2:],
                    [],
                    movie_title,
                    glossary,
                    tone,
                    profile,
                    extra_instruction="仅翻译这 1 句为简体中文，严格输出 JSON。",
                )
                if seg.id in one_map:
                    resolved[seg.id] = one_map[seg.id]
                    continue
            except CancelledError:
                raise
            except Exception:
                pass
            # Last resort: keep source text so timestamps never drop out
            resolved[seg.id] = seg.text

        return resolved

    def translate_segments(
        self,
        segments: list[TranscriptSegment],
        video_path: str,
        profile: LanguageProfile,
        cancel_event: threading.Event | None = None,
        progress_cb: Callable[[float, str], None] | None = None,
    ) -> list[TranslatedSegment]:
        if not segments:
            return []

        movie_title = _clean_movie_title(video_path)
        batch_size = max(5, min(60, self.config.batch_size))
        ctx_before_n = max(0, self.config.context_before)
        ctx_after_n = max(0, self.config.context_after)

        try:
            if progress_cb:
                progress_cb(0.02, "正在分析影片人名与专有名词表...")
            glossary, tone = self.extract_glossary(movie_title, segments, profile, cancel_event)

            batches: list[tuple[int, list[TranscriptSegment]]] = []
            for start_idx in range(0, len(segments), batch_size):
                batches.append((start_idx, segments[start_idx : start_idx + batch_size]))

            total_batches = len(batches)
            translations: dict[int, str] = {}

            if self.supports_parallel and self.config.max_concurrency > 1 and total_batches > 1:
                completed = 0
                with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as pool:
                    future_map = {}
                    for start_idx, batch in batches:
                        if cancel_event is not None and cancel_event.is_set():
                            raise CancelledError("任务已取消")
                        ctx_before = [
                            {profile.code: s.text}
                            for s in segments[max(0, start_idx - ctx_before_n) : start_idx]
                        ]
                        after_start = start_idx + len(batch)
                        ctx_after = [
                            s.text for s in segments[after_start : after_start + ctx_after_n]
                        ]
                        fut = pool.submit(
                            self._translate_batch_with_retry,
                            batch,
                            ctx_before,
                            ctx_after,
                            movie_title,
                            glossary,
                            tone,
                            profile,
                            cancel_event,
                        )
                        future_map[fut] = batch

                    for fut in as_completed(future_map):
                        if cancel_event is not None and cancel_event.is_set():
                            raise CancelledError("任务已取消")
                        batch_map = fut.result()
                        translations.update(batch_map)
                        completed += 1
                        if progress_cb:
                            progress_cb(
                                0.06 + 0.94 * (completed / total_batches),
                                f"云端翻译中 ({completed}/{total_batches} 批)",
                            )
            else:
                for b_idx, (start_idx, batch) in enumerate(batches):
                    if cancel_event is not None and cancel_event.is_set():
                        raise CancelledError("任务已取消")
                    if progress_cb:
                        progress_cb(
                            0.06 + 0.94 * (b_idx / total_batches),
                            f"本地翻译中 ({b_idx + 1}/{total_batches} 批)",
                        )
                    prev_slice = segments[max(0, start_idx - ctx_before_n) : start_idx]
                    ctx_before = [
                        {profile.code: s.text, "zh": translations.get(s.id, "")}
                        for s in prev_slice
                    ]
                    after_start = start_idx + len(batch)
                    ctx_after = [
                        s.text for s in segments[after_start : after_start + ctx_after_n]
                    ]
                    batch_map = self._translate_batch_with_retry(
                        batch,
                        ctx_before,
                        ctx_after,
                        movie_title,
                        glossary,
                        tone,
                        profile,
                        cancel_event,
                    )
                    translations.update(batch_map)
                    if progress_cb:
                        progress_cb(
                            0.06 + 0.94 * ((b_idx + 1) / total_batches),
                            f"本地翻译中 ({b_idx + 1}/{total_batches} 批)",
                        )

            return [
                TranslatedSegment(
                    id=seg.id,
                    start=seg.start,
                    end=seg.end,
                    source_text=seg.text,
                    target_text=translations.get(seg.id, seg.text),
                )
                for seg in segments
            ]
        finally:
            self.close()
