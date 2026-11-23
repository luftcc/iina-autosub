from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class _JapaneseProfileData:
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


_HALLUCINATION_PATTERNS = (
    # Classic Japanese Whisper silence/BGM hallucinations
    re.compile(r"^\s*(ご視聴|視聴)(いただき)?ありがとう(ございました|ございます)[。！!]*\s*$"),
    re.compile(r"チャンネル登録|高評価|グッドボタン|通知ボタン"),
    re.compile(r"字幕(作成|制作|提供|協力|翻訳|by)|翻訳[:：]|Amara\.org", re.IGNORECASE),
    re.compile(r"^\s*最後までご覧(いただき|頂き)ありがとう.*$"),
    re.compile(r"^\s*(ご覧|お聞き)いただきありがとう.*$"),
    re.compile(r"^\s*提供でお送り(しました|します|いたしました)[。！!]*\s*$"),
    re.compile(r"^\s*次回の動画でお会いしましょう.*$"),
    re.compile(r"^\s*(MBC|NHK|TBS|フジテレビ)?ニュース.*お伝えしました.*$"),
    re.compile(r"^\s*[\[\(（【]?(音楽|BGM|拍手|笑い|歓声|無音|沈黙)[\]\)）】]?\s*$"),
    # Pure punctuation / filler noise
    re.compile(r"^[\s。、，,\.．・…\-ー～〜!！?？♪♫★☆\(\)（）「」『』\[\]【】]+$"),
    # Extreme single-character repetition (e.g. ああああああ)
    re.compile(r"^(.)\1{5,}$"),
)

_GLOSSARY_SYSTEM_PROMPT = """你是一位资深的日剧与日本电影中文字幕总监。
下面给你影片文件名和部分日语对白抽样，请你提取全片最关键的专有名词与称谓词表，用于统一后续整部电影的译名。

要求：
1. 只提取确信出现的人名、昵称、地名、组织/学校/作品名、特殊称谓（最多 25 条）。
2. 译名应符合中国大陆影视字幕习惯，简洁自然。
3. 严格输出 JSON 对象，格式如下，不要输出任何额外解释：
{"glossary": {"原文": "中文译名"}, "tone": "一句简短描述影片语气风格（如：现代都市日常/悬疑冷峻/青春校园轻快）"}"""

_TRANSLATION_SYSTEM_PROMPT = """你是一位专业日本电影与日剧中文字幕译者。请将给定的日语台词批量翻译为自然、地道、口语化的简体中文字幕。

核心准则：
1. **口语自然，拒绝翻译腔**：像真人影视字幕组一样表达，根据语气和人物关系自然还原日语省略的主语（我/你/他/她/大家）；日常应答词如「そっか / そうか / なるほど / うん」请按语境译为“这样啊 / 是吗 / 嗯 / 对”，避免机械译成“原来如此”。
2. **忠实原意、不漏细节**：保留关键时间、地点与具体动作（如「今日」译出“今天”、「風邪ひかないように」译出“别感冒了”），不要过度意译或删减信息。
3. **称谓与敬语得体**：结合上下文自然处理「〜さん / 〜先輩 / 〜ちゃん / 〜くん / 先生 / 部長」等称呼，不要生硬机械全翻成“xx桑”。
4. **精炼易读**：单条中文字幕尽量精炼（通常不超过 22 个汉字，必要时保留意群不硬砍信息），不使用书信式长定语，句末通常不加中文句号（问号、感叹号、省略号按需保留）。
5. **严格遵循术语表与上下文**：保持人名、地名、设定与前后已译台词一致；参考前后文避免语义断裂，但**只翻译 `items` 数组里的句子**。
6. **绝对 1:1 对齐**：严格返回合法 JSON，格式为：
   {"items": [{"id": 1, "zh": "中文译文"}, ...]}
   必须与输入的 `items` 长度完全一致、每个 `id` 严格对应、绝不合并、绝不漏行、绝不输出 JSON 以外的任何文字。"""


JAPANESE_PROFILE = _JapaneseProfileData(
    code="ja",
    whisper_lang="ja",
    iso_tags=frozenset({"ja", "jp", "jpn", "japanese"}),
    display_name="日语 → 简体中文",
    secondary_font="Hiragino Sans",
    primary_font="PingFang SC",
    bilingual_suffix="zh-ja",
    zh_only_suffix="zh-Hans",
    sentence_end_chars="。！？!?…」』",
    soft_break_chars="、,，…・ ",
    max_source_chars=30,
    min_duration_sec=0.8,
    max_duration_sec=5.8,
    pause_split_sec=0.45,
    hallucination_regexes=_HALLUCINATION_PATTERNS,
    whisper_initial_prompt="以下は日本語の映画・ドラマの会話です。自然な句読点を付けて正確に書き起こしてください。",
    glossary_system_prompt=_GLOSSARY_SYSTEM_PROMPT,
    translation_system_prompt=_TRANSLATION_SYSTEM_PROMPT,
)
