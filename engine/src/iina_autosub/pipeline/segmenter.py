from __future__ import annotations

from iina_autosub.models import TranscriptSegment, WordTiming
from iina_autosub.pipeline.langs import LanguageProfile

MIN_GAP_SEC = 0.04


def _clean_text(text: str) -> str:
    return " ".join(text.strip().split())


def _split_segment_without_words(
    seg: TranscriptSegment,
    profile: LanguageProfile,
) -> list[TranscriptSegment]:
    """Proportionally split a long non-word-timed Japanese segment along punctuation."""
    text = _clean_text(seg.text)
    if len(text) <= profile.max_source_chars and (seg.end - seg.start) <= profile.max_duration_sec:
        return [TranscriptSegment(id=0, start=seg.start, end=seg.end, text=text)]

    pieces: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        cur = "".join(buf).strip()
        if ch in profile.sentence_end_chars and len(cur) >= 6:
            pieces.append(cur)
            buf = []
        elif ch in profile.soft_break_chars and len(cur) >= int(profile.max_source_chars * 0.65):
            pieces.append(cur)
            buf = []
        elif len(cur) >= profile.max_source_chars:
            pieces.append(cur)
            buf = []
    if buf:
        rem = "".join(buf).strip()
        if rem:
            if pieces and len(rem) <= 4 and len(pieces[-1]) + len(rem) <= profile.max_source_chars + 4:
                pieces[-1] += rem
            else:
                pieces.append(rem)

    if len(pieces) <= 1:
        return [TranscriptSegment(id=0, start=seg.start, end=seg.end, text=text)]

    total_chars = max(1, sum(len(p) for p in pieces))
    total_dur = max(0.2, seg.end - seg.start)
    out: list[TranscriptSegment] = []
    cursor = seg.start
    for idx, piece in enumerate(pieces):
        share = (len(piece) / total_chars) * total_dur
        end = seg.end if idx == len(pieces) - 1 else min(seg.end, cursor + max(0.35, share))
        out.append(TranscriptSegment(id=0, start=cursor, end=max(cursor + 0.15, end), text=piece))
        cursor = end
    return out


def _segment_from_words(
    words: list[WordTiming],
    profile: LanguageProfile,
) -> list[TranscriptSegment]:
    if not words:
        return []

    out: list[TranscriptSegment] = []
    buf: list[WordTiming] = []

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        text = "".join(w.word for w in buf).strip()
        if text:
            start = buf[0].start
            end = max(start + 0.15, buf[-1].end)
            out.append(TranscriptSegment(id=0, start=start, end=end, text=text))
        buf = []

    for idx, word in enumerate(words):
        if buf:
            gap = word.start - buf[-1].end
            cur_text = "".join(w.word for w in buf).strip()
            cur_dur = buf[-1].end - buf[0].start
            if gap >= profile.pause_split_sec:
                flush()
            elif len(cur_text) + len(word.word) > profile.max_source_chars and len(cur_text) >= 6:
                flush()
            elif cur_dur >= profile.max_duration_sec and len(cur_text) >= 6:
                flush()

        buf.append(word)
        joined = "".join(w.word for w in buf).strip()
        dur = buf[-1].end - buf[0].start
        last_char = joined[-1] if joined else ""

        if last_char in profile.sentence_end_chars and len(joined) >= 5:
            flush()
        elif (
            last_char in profile.soft_break_chars
            and (len(joined) >= int(profile.max_source_chars * 0.65) or dur >= 3.6)
        ):
            flush()
        elif idx == len(words) - 1:
            flush()

    return out


def resegment_transcript(
    segments: list[TranscriptSegment],
    profile: LanguageProfile,
) -> list[TranscriptSegment]:
    """Re-segment Japanese ASR output into clean, natural subtitle lines with non-overlapping timings."""
    if not segments:
        return []

    raw_out: list[TranscriptSegment] = []
    word_buffer: list[WordTiming] = []

    def flush_words() -> None:
        nonlocal word_buffer
        if word_buffer:
            raw_out.extend(_segment_from_words(word_buffer, profile))
            word_buffer = []

    for seg in segments:
        if seg.words:
            if word_buffer and (seg.words[0].start - word_buffer[-1].end) >= profile.pause_split_sec * 1.5:
                flush_words()
            word_buffer.extend(seg.words)
        else:
            flush_words()
            raw_out.extend(_split_segment_without_words(seg, profile))
    flush_words()

    # Merge tiny orphan fragments with adjacent segment when close in time
    merged: list[TranscriptSegment] = []
    for seg in raw_out:
        text = _clean_text(seg.text)
        if not text:
            continue
        seg = TranscriptSegment(id=0, start=seg.start, end=seg.end, text=text)
        if (
            merged
            and len(seg.text) <= 4
            and (seg.end - seg.start) < 0.55
            and (seg.start - merged[-1].end) <= 0.35
            and len(merged[-1].text) + len(seg.text) <= profile.max_source_chars + 4
            and (seg.end - merged[-1].start) <= profile.max_duration_sec
        ):
            merged[-1] = TranscriptSegment(
                id=0,
                start=merged[-1].start,
                end=max(merged[-1].end, seg.end),
                text=merged[-1].text + seg.text,
            )
        else:
            merged.append(seg)

    # Normalize duration bounds and enforce non-overlapping monotonic timestamps
    final: list[TranscriptSegment] = []
    for idx, seg in enumerate(merged):
        start = max(0.0, seg.start)
        end = max(start + 0.25, seg.end)
        if end - start < profile.min_duration_sec:
            next_start = merged[idx + 1].start if idx + 1 < len(merged) else end + 1.0
            end = min(start + profile.min_duration_sec, max(end, next_start - MIN_GAP_SEC))
        if final and start < final[-1].end + MIN_GAP_SEC:
            midpoint = final[-1].end
            if start < midpoint:
                final[-1] = TranscriptSegment(
                    id=final[-1].id,
                    start=final[-1].start,
                    end=max(final[-1].start + 0.15, start - MIN_GAP_SEC),
                    text=final[-1].text,
                )
            start = max(start, final[-1].end + MIN_GAP_SEC)
            end = max(start + 0.2, end)
        final.append(
            TranscriptSegment(
                id=len(final) + 1,
                start=round(start, 3),
                end=round(end, 3),
                text=seg.text,
            )
        )
    return final
