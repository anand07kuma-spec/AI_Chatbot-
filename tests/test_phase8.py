"""Phase 8: model-mode selector + response-language preference (backend)."""

from unittest.mock import patch

import main


# ---------- GET /api/models ----------

class TestModelModes:

    def test_returns_configured_modes(self, client):
        r = client.get("/api/models")
        body = r.json()
        assert r.status_code == 200
        assert body["ok"] is True
        ids = [m["id"] for m in body["modes"]]
        assert ids == ["auto", "fast", "reasoning"]
        # non-auto modes must expose the real configured model id
        for m in body["modes"][1:]:
            assert m["model"]

    def test_auto_has_empty_model(self, client):
        body = client.get("/api/models").json()
        auto = body["modes"][0]
        assert auto["id"] == "auto"
        assert auto["model"] == ""


# ---------- reply_lang ----------

def _post_and_capture(client, payload):
    """POST /api/chat with patched get_response; return the system prompt used."""
    captured = {}

    def fake_get_response(msgs, model=None, system_prompt=None):
        captured["prompt"] = system_prompt
        return {"ok": True, "reply": "ok", "model": "m", "task": "fast"}

    with patch.object(main, "get_response", side_effect=fake_get_response):
        r = client.post("/api/chat", json=payload)
    assert r.status_code == 200
    return captured.get("prompt")


_MSG = [{"role": "user", "content": "hello"}]


class TestReplyLang:

    def test_reply_lang_appended(self, client):
        prompt = _post_and_capture(client, {"messages": _MSG, "reply_lang": "Hindi"})
        assert prompt is not None
        assert "response language" in prompt
        assert "Hindi" in prompt

    def test_reply_lang_auto_no_override(self, client):
        prompt = _post_and_capture(client, {"messages": _MSG, "reply_lang": "auto"})
        # None → default SYSTEM_PROMPT used inside chatbot.py
        assert prompt is None

    def test_reply_lang_empty_no_override(self, client):
        prompt = _post_and_capture(client, {"messages": _MSG, "reply_lang": "  "})
        assert prompt is None

    def test_reply_lang_combined_with_coding_prompt(self, client):
        prompt = _post_and_capture(
            client,
            {"messages": _MSG, "mode": "code", "reply_lang": "Hinglish"},
        )
        assert "coding assistant" in prompt.lower()
        assert "Hinglish" in prompt

    def test_reply_lang_capped(self, client):
        prompt = _post_and_capture(
            client, {"messages": _MSG, "reply_lang": "x" * 500}
        )
        # language text itself is capped at 40 chars (no prompt flooding)
        assert "x" * 41 not in prompt
        assert "x" * 40 in prompt

    def test_reply_lang_stream_endpoint(self, client):
        captured = {}

        async def fake_stream_response(msgs, model=None, images=None,
                                       request_id=None, system_prompt=None):
            captured["prompt"] = system_prompt
            yield {"type": "done", "model": "m", "task": "fast"}

        with patch.object(main, "stream_response", side_effect=fake_stream_response):
            r = client.post(
                "/api/chat/stream", json={"messages": _MSG, "reply_lang": "Urdu"}
            )
        assert r.status_code == 200
        assert captured.get("prompt") and "Urdu" in captured["prompt"]
