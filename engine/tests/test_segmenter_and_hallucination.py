from __future__ import annotations

from iina_autosub.models import TranscriptSegment, WordTiming
from iina_autosub.pipeline.asr import is_hallucination_text
from iina_autosub.pipeline.langs import get_language_profile
from iina_autosub.pipeline.segmenter import resegment_transcript


def test_japanese_hallucination_filter() -> None:
    ja = get_language_profile("ja")
    assert is_hallucination_text("ご視聴ありがとうございました。", ja)
    assert is_hallucination_text("チャンネル登録と高評価よろしくお願いします", ja)
    assert is_hallucination_text("字幕作成: 太郎", ja)
    assert is_hallucination_text("ああああああああ", ja)
    assert is_hallucination_text("……", ja)
    assert not is_hallucination_text("おはようございます、先輩。", ja)
    assert not is_hallucination_text("ちょっと待って！", ja)


def test_resegment_word_timings_on_punctuation_and_pauses() -> None:
    ja = get_language_profile("ja")
    words = [
        WordTiming(word="おはよう", start=0.0, end=0.5),
        WordTiming(word="ございます。", start=0.5, end=1.1),
        WordTiming(word="今日", start=1.2, end=1.5),
        WordTiming(word="は", start=1.5, end=1.6),
        WordTiming(word="いい", start=1.6, end=1.9),
        WordTiming(word="天気ですね。", start=1.9, end=2.6),
        # Large pause before next utterance
        WordTiming(word="行きましょう。", start=4.0, end=4.8),
    ]
    raw = [TranscriptSegment(id=1, start=0.0, end=4.8, text="".join(w.word for w in words), words=words)]
    out = resegment_transcript(raw, ja)
    assert len(out) == 3
    assert [s.text for s in out] == ["おはようございます。", "今日はいい天気ですね。", "行きましょう。"]
    assert [s.id for s in out] == [1, 2, 3]
    # Monotonic non-overlapping
    for a, b in zip(out, out[1:], strict=False):
        assert a.end < b.start


def test_resegment_long_segment_without_words() -> None:
    ja = get_language_profile("ja")
    long_text = "昨日駅の前で偶然昔の友達に会ったんだよ。本当に驚いたよ！十年ぶりだったからね。"
    raw = [TranscriptSegment(id=1, start=10.0, end=19.0, text=long_text)]
    out = resegment_transcript(raw, ja)
    assert len(out) == 3
    assert "".join(s.text for s in out) == long_text
    assert out[0].start == 10.0
    assert out[-1].end <= 19.0
