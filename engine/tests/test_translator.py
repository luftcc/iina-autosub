from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from iina_autosub.models import TranscriptSegment, TranslatorConfig
from iina_autosub.pipeline.langs import get_language_profile
from iina_autosub.translators.base import BaseTranslator, extract_json_object, strip_thinking
from iina_autosub.translators.openai_api import OpenAICompatTranslator


class StubTranslator(BaseTranslator):
    def __init__(self, config: TranslatorConfig):
        super().__init__(config)
        self.calls: list[dict] = []
        self._failed_once = False

    def chat_complete(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        payload = json.loads(user_prompt)
        self.calls.append(payload)
        if "sample_lines" in payload:
            return '<think>reasoning...</think>{"glossary": {"健太": "健太"}, "tone": "青春校园"}'

        items = payload["items"]
        out = []
        for idx, item in enumerate(items):
            if not self._failed_once and idx == 1:
                # Simulate model accidentally leaving Japanese kana untranslated on first try
                self._failed_once = True
                out.append({"id": item["id"], "zh": "おはようございます"})
            else:
                out.append({"id": item["id"], "zh": f"译文{item['id']}。"})
        return f"```json\n{json.dumps({'items': out}, ensure_ascii=False)}\n```"


def test_strip_thinking_and_json_extract() -> None:
    raw = '<think>let me translate\nstep 1</think>\n```json\n{"items": [{"id": 1, "zh": "你好"}],}\n```'
    assert strip_thinking(raw).startswith("```json")
    parsed = extract_json_object(raw)
    assert parsed["items"][0]["zh"] == "你好"


def test_sliding_window_and_untranslated_retry() -> None:
    ja = get_language_profile("ja")
    segs = [
        TranscriptSegment(id=i, start=float(i), end=float(i) + 0.8, text=f"おはようございます{i}")
        for i in range(1, 13)
    ]
    cfg = TranslatorConfig(mode="local", batch_size=5, context_before=3, context_after=2)
    translator = StubTranslator(cfg)
    result = translator.translate_segments(segs, "/movies/Your.Name.[1080p].mkv", ja)

    assert len(result) == 12
    # Item 2 should have been retried and fixed from kana to Chinese, and trailing "。" stripped
    assert all(r.target_text == f"译文{r.id}" for r in result)
    # Glossary pass should have run once + batch calls + 1 retry call
    batch_calls = [c for c in translator.calls if "items" in c]
    assert batch_calls[0]["glossary"] == {"健太": "健太"}
    # Second full batch should carry translated context_before from batch 1
    second_full = next(c for c in batch_calls if c["items"][0]["id"] == 6)
    assert len(second_full["context_before"]) == 3
    assert second_full["context_before"][-1]["zh"] == "译文5"
    assert len(second_full["context_after"]) == 2


def test_openai_compat_translator_against_mock_server() -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            user_payload = json.loads(body["messages"][1]["content"])
            items = [{"id": it["id"], "zh": f"云端译文{it['id']}"} for it in user_payload["items"]]
            resp = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"items": items}, ensure_ascii=False),
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
            raw = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, _format, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        ja = get_language_profile("ja")
        cfg = TranslatorConfig(
            mode="cloud",
            cloud_base_url=f"http://127.0.0.1:{port}/v1",
            cloud_api_key="sk-test",
            cloud_model="deepseek-chat",
            batch_size=5,
        )
        segs = [
            TranscriptSegment(id=1, start=0.0, end=1.0, text="ありがとう"),
            TranscriptSegment(id=2, start=1.5, end=2.5, text="さようなら"),
        ]
        tr = OpenAICompatTranslator(cfg)
        out = tr.translate_segments(segs, "/tmp/demo.mkv", ja)
        assert [o.target_text for o in out] == ["云端译文1", "云端译文2"]
    finally:
        server.shutdown()
        server.server_close()
