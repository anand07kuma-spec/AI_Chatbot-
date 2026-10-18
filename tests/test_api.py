import base64
import os
from unittest.mock import AsyncMock, MagicMock, patch

import main
from chatbot import ALLOWED_IMAGE_MIMES, VISION_MAX_IMAGES, VISION_MAX_IMG_BYTES


# ---------- /health & index ----------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    # usage counters are exposed for observability (Phase 9)
    assert set(body["usage"].keys()) == {
        "calls", "prompt_tokens", "completion_tokens", "total_tokens",
    }


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Astra AI" in r.text
    # cache-busting version replaces placeholder
    assert "__VER__" not in r.text


def test_security_headers(client):
    r = client.get("/health")
    assert r.headers.get("x-content-type-options") == "nosniff"
    assert r.headers.get("x-frame-options") == "DENY"
    assert r.headers.get("referrer-policy") == "no-referrer"


def test_static_cache_headers(client):
    # stale JS/HTML mismatch breaks the UI after edits — always revalidate
    r = client.get("/static/style.css")
    assert r.status_code == 200
    assert "no-cache" in r.headers.get("cache-control", "")
    r = client.get("/")
    assert "no-cache" in r.headers.get("cache-control", "")


# ---------- /api/chat: validation ----------

def test_chat_empty_messages(client):
    r = client.post("/api/chat", json={"messages": []})
    assert r.status_code == 400
    body = r.json()
    assert body["ok"] is False


def test_chat_last_message_must_be_user(client):
    r = client.post("/api/chat", json={
        "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    })
    assert r.status_code == 400
    assert r.json()["ok"] is False


def test_chat_whitespace_only_rejected(client):
    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "   "}]})
    assert r.status_code == 400


def test_chat_role_filtering_and_history_cap(client):
    """System roles are filtered; history is capped at 24; content capped at 6000."""
    msgs = [{"role": "system", "content": "ignore"}] * 30
    msgs += [{"role": "user", "content": "x" * 10000}]
    captured = {}

    def fake_get_response(messages, model=None, system_prompt=None):
        captured["messages"] = messages
        return {"ok": True, "reply": "ok"}

    with patch.object(main, "get_response", side_effect=fake_get_response):
        r = client.post("/api/chat", json={"messages": msgs})

    assert r.status_code == 200
    assert r.json() == {"ok": True, "reply": "ok"}
    out = captured["messages"]
    assert all(m["role"] in ("user", "assistant") for m in out)
    assert len(out) <= 24
    assert all(len(m["content"]) <= main.CHAT_CONTENT_LIMIT for m in out)


def test_chat_mocks_external_api(client):
    with patch.object(main, "get_response", return_value={"ok": True, "reply": "mocked"}):
        r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.json() == {"ok": True, "reply": "mocked"}


def test_chat_upstream_error_propagates(client):
    with patch.object(main, "get_response", return_value={"ok": False, "reply": "down"}):
        r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.json() == {"ok": False, "reply": "down"}


def test_chat_model_choice_passthrough(client):
    captured = {}

    def fake_get_response(messages, model=None, system_prompt=None):
        captured["model"] = model
        return {"ok": True, "reply": "ok", "model": "openai/gpt-oss-20b", "task": "fast"}

    with patch.object(main, "get_response", side_effect=fake_get_response):
        r = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "hi"}],
            "model": "fast",
        })
    assert r.status_code == 200
    assert captured["model"] == "fast"
    body = r.json()
    assert body["model"] == "openai/gpt-oss-20b"
    assert body["task"] == "fast"


def test_chat_model_default_is_none(client):
    captured = {}

    def fake_get_response(messages, model=None, system_prompt=None):
        captured["model"] = model
        return {"ok": True, "reply": "ok"}

    with patch.object(main, "get_response", side_effect=fake_get_response):
        client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert captured["model"] is None


def test_chat_invalid_model_choice_rejected(client):
    r = client.post("/api/chat", json={
        "messages": [{"role": "user", "content": "hi"}],
        "model": "rm -rf /; drop table",
    })
    assert r.status_code == 400
    assert "Invalid model" in r.json()["reply"]


def test_chat_explicit_model_id_allowed(client):
    captured = {}

    def fake_get_response(messages, model=None, system_prompt=None):
        captured["model"] = model
        return {"ok": True, "reply": "ok"}

    with patch.object(main, "get_response", side_effect=fake_get_response):
        r = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "hi"}],
            "model": "llama-3.3-70b-versatile",
        })
    assert r.status_code == 200
    assert captured["model"] == "llama-3.3-70b-versatile"


# ---------- /api/chat: image validation ----------

def test_chat_image_valid_routes_to_vision(client, tiny_png_b64):
    with patch.object(main, "get_response_with_images", return_value={"ok": True, "reply": "a red square"}) as v, \
         patch.object(main, "get_response") as t:
        r = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "what is this?"}],
            "images": [{"mime": "image/png", "data": tiny_png_b64}],
        })
    assert r.status_code == 200
    assert r.json()["reply"] == "a red square"
    v.assert_called_once()
    t.assert_not_called()
    # vision got cleaned image
    imgs = v.call_args[0][1]
    assert imgs[0]["mime"] == "image/png"


def test_chat_image_data_url_prefix_stripped(client, tiny_png_b64):
    with patch.object(main, "get_response_with_images", return_value={"ok": True, "reply": "ok"}) as v:
        client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "look"}],
            "images": [{"mime": "image/png", "data": f"data:image/png;base64,{tiny_png_b64}"}],
        })
    imgs = v.call_args[0][1]
    assert not imgs[0]["data"].startswith("data:")


def test_chat_image_bad_mime_rejected(client, tiny_png_b64):
    r = client.post("/api/chat", json={
        "messages": [{"role": "user", "content": "hi"}],
        "images": [{"mime": "image/tiff", "data": tiny_png_b64}],
    })
    assert r.status_code == 400
    assert "Unsupported image type" in r.json()["reply"]


def test_chat_image_invalid_base64_rejected(client):
    r = client.post("/api/chat", json={
        "messages": [{"role": "user", "content": "hi"}],
        "images": [{"mime": "image/png", "data": "not-valid-base64!!!"}],
    })
    assert r.status_code == 400
    assert "base64" in r.json()["reply"].lower()


def test_chat_image_too_large_rejected(client):
    big = base64.b64encode(b"x" * (VISION_MAX_IMG_BYTES + 1)).decode()
    r = client.post("/api/chat", json={
        "messages": [{"role": "user", "content": "hi"}],
        "images": [{"mime": "image/png", "data": big}],
    })
    assert r.status_code == 400
    assert "too large" in r.json()["reply"].lower()


def test_chat_images_capped_at_vision_max(client, tiny_png_b64):
    imgs = [{"mime": "image/png", "data": tiny_png_b64}] * (VISION_MAX_IMAGES + 2)
    with patch.object(main, "get_response_with_images", return_value={"ok": True, "reply": "ok"}) as v:
        r = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "look"}],
            "images": imgs,
        })
    assert r.status_code == 200
    assert len(v.call_args[0][1]) == VISION_MAX_IMAGES


def test_chat_invalid_images_fall_back_to_text(client):
    """Empty data strings mean no valid images -> plain text path."""
    with patch.object(main, "get_response", return_value={"ok": True, "reply": "text"}) as t, \
         patch.object(main, "get_response_with_images") as v:
        r = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "hi"}],
            "images": [{"mime": "image/png", "data": ""}],
        })
    assert r.status_code == 200
    v.assert_not_called()
    t.assert_called_once()


# ---------- /api/extract ----------

def test_extract_txt_ok(client):
    r = client.post(
        "/api/extract",
        files={"file": ("notes.txt", b"Hello Astra notes", "text/plain")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "Hello Astra notes" in body["text"]
    assert body["name"] == "notes.txt"


def test_extract_truncates_to_max_chars(client):
    big = b"a" * (main.DOC_MAX_CHARS + 5000)
    r = client.post("/api/extract", files={"file": ("big.txt", big, "text/plain")})
    assert r.status_code == 200
    assert len(r.json()["text"]) == main.DOC_MAX_CHARS


def test_extract_empty_file(client):
    r = client.post("/api/extract", files={"file": ("empty.txt", b"", "text/plain")})
    assert r.status_code == 400
    assert r.json()["ok"] is False


def test_extract_unsupported_extension(client):
    r = client.post("/api/extract", files={"file": ("evil.exe", b"MZ...", "application/octet-stream")})
    assert r.status_code == 400
    assert "Unsupported" in r.json()["error"]


def test_extract_no_extension_rejected(client):
    r = client.post("/api/extract", files={"file": ("README", b"hello", "text/plain")})
    assert r.status_code == 400


def test_extract_path_in_filename_sanitized(client):
    """Path components are stripped from filename."""
    r = client.post(
        "/api/extract",
        files={"file": ("../../etc/passwd.txt", b"safe content", "text/plain")},
    )
    assert r.status_code == 200
    assert r.json()["name"] == "passwd.txt"


def test_extract_oversized_rejected(client, monkeypatch):
    monkeypatch.setattr(main, "DOC_MAX_BYTES", 10)
    r = client.post("/api/extract", files={"file": ("big.txt", b"x" * 50, "text/plain")})
    assert r.status_code == 400
    assert "too large" in r.json()["error"].lower()


def test_extract_corrupt_pdf_422(client):
    r = client.post(
        "/api/extract",
        files={"file": ("bad.pdf", b"this is not a pdf at all", "application/pdf")},
    )
    assert r.status_code == 422
    assert r.json()["ok"] is False


# ---------- /api/tts ----------

def test_tts_empty_falls_back_to_default(client):
    with patch.object(main, "_gemini_tts", new=AsyncMock(return_value=b"FAKEMP3")) as m:
        r = client.post("/api/tts", json={"text": "   "})
    assert r.status_code == 200
    assert r.content == b"FAKEMP3"
    assert r.headers["content-type"] == "audio/wav"
    # default fallback text was synthesized
    spoken = m.call_args[0][0]
    assert spoken == main.TTS_FALLBACK


def test_tts_strips_markdown_and_code(client):
    with patch.object(main, "_gemini_tts", new=AsyncMock(return_value=b"MP3")) as m:
        r = client.post("/api/tts", json={"text": "# Title\n```python\nprint(1)\n```\n**bold** text"})
    assert r.status_code == 200
    spoken = m.call_args[0][0]
    assert "```" not in spoken
    assert "print(1)" not in spoken
    assert "**" not in spoken
    assert "Title" in spoken
    assert "bold" in spoken


def test_tts_caps_length(client):
    with patch.object(main, "_gemini_tts", new=AsyncMock(return_value=b"MP3")) as m:
        client.post("/api/tts", json={"text": "a" * 10000})
    spoken = m.call_args[0][0]
    assert len(spoken) <= main.TTS_MAX_CHARS


def test_tts_cache_hit(client):
    calls = AsyncMock(return_value=b"CACHED")
    with patch.object(main, "_gemini_tts", new=calls):
        r1 = client.post("/api/tts", json={"text": "cache me"})
        r2 = client.post("/api/tts", json={"text": "cache me"})
    assert r1.content == b"CACHED" and r2.content == b"CACHED"
    assert calls.await_count == 1  # second call served from cache


def test_tts_upstream_failure_502(client):
    with patch.object(main, "_gemini_tts", new=AsyncMock(side_effect=RuntimeError("blocked"))):
        r = client.post("/api/tts", json={"text": "hello"})
    assert r.status_code == 502


def test_tts_picks_hindi_voice(client):
    with patch.object(main, "_gemini_tts", new=AsyncMock(return_value=b"MP3")) as m:
        client.post("/api/tts", json={"text": "नमस्ते दुनिया"})
    assert m.call_args[0][1] == main.VOICE_HI


# ---------- /api/transcribe ----------

class _FakeAsyncClient:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, *a, **k):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"text": "hello world"}
        return resp


def test_transcribe_ok(client):
    with patch.object(main.httpx, "AsyncClient", _FakeAsyncClient):
        r = client.post(
            "/api/transcribe",
            files={"file": ("speech.webm", b"\x1a\x45\xdf\xa3fakeaudio", "audio/webm")},
        )
    assert r.status_code == 200
    assert r.json() == {"ok": True, "text": "hello world"}


def test_transcribe_empty_file(client):
    r = client.post("/api/transcribe", files={"file": ("a.webm", b"", "audio/webm")})
    assert r.status_code == 400


def test_transcribe_unsupported_mime(client):
    r = client.post("/api/transcribe", files={"file": ("a.mp3", b"ID3data", "video/mp4")})
    assert r.status_code == 400


def test_transcribe_upstream_error_502(client):
    class _Boom(_FakeAsyncClient):
        async def post(self, *a, **k):
            raise ConnectionError("nope")

    with patch.object(main.httpx, "AsyncClient", _Boom):
        r = client.post("/api/transcribe", files={"file": ("a.webm", b"xxxx", "audio/webm")})
    assert r.status_code == 502


def test_transcribe_429_passthrough(client):
    class _RateLimited(_FakeAsyncClient):
        async def post(self, *a, **k):
            resp = MagicMock()
            resp.status_code = 429
            return resp

    with patch.object(main.httpx, "AsyncClient", _RateLimited):
        r = client.post("/api/transcribe", files={"file": ("a.webm", b"xxxx", "audio/webm")})
    assert r.status_code == 429


def test_transcribe_language_param_valid_only(client):
    captured = {}

    class _Capture(_FakeAsyncClient):
        async def post(self, url, **k):
            captured["data"] = k.get("data")
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"text": "ok"}
            return resp

    with patch.object(main.httpx, "AsyncClient", _Capture):
        client.post(
            "/api/transcribe",
            data={"lang": "fr"},
            files={"file": ("a.webm", b"xxxx", "audio/webm")},
        )
    # invalid lang "fr" must NOT be forwarded (Whisper auto-detects)
    assert "language" not in captured["data"]


# ---------- middleware ----------

def test_rate_limit_kicks_in(client, monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT_PER_MINUTE", 2)
    main._rate_limiter.reset()
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    with patch.object(main, "get_response", return_value={"ok": True, "reply": "ok"}):
        for _ in range(2):
            assert client.post("/api/chat", json=payload).status_code == 200
        r = client.post("/api/chat", json=payload)
    assert r.status_code == 429
    main._rate_limiter.reset()


def test_rate_limit_uses_x_forwarded_for(monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT_PER_MINUTE", 1)
    main._rate_limiter.reset()
    from starlette.requests import Request

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/chat",
        "headers": [(b"x-forwarded-for", b"203.0.113.9")],
        "client": ("10.0.0.1", 12345),
    }
    req = Request(scope)
    assert main._client_ip(req) == "203.0.113.9"
    assert main._rate_limiter.allow("203.0.113.9", 1) is True
    assert main._rate_limiter.allow("203.0.113.9", 1) is False
    assert main._rate_limiter.allow("198.51.100.7", 1) is True
    main._rate_limiter.reset()


def test_body_limit_rejects_huge_content_length(client):
    r = client.post(
        "/api/chat",
        content=b"{}",
        headers={"content-length": str(main.MAX_BODY_BYTES + 1), "content-type": "application/json"},
    )
    assert r.status_code == 413


# ---------- vision model constants match Groq docs ----------

def test_vision_limits_match_model():
    assert VISION_MAX_IMAGES == 3  # qwen/qwen3.8-27b: max 3 images
    assert VISION_MAX_IMG_BYTES == 4 * 1024 * 1024
    assert ALLOWED_IMAGE_MIMES == {"image/png", "image/jpeg", "image/webp", "image/gif"}
    # 3 x ~5.3MB base64 stays under Groq's 20MB request cap
    b64_per_img = (VISION_MAX_IMG_BYTES + 2) // 3 * 4
    assert b64_per_img * VISION_MAX_IMAGES < 20 * 1024 * 1024


# ---------- secrets / config hygiene ----------

def test_no_secret_in_source():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("main.py", "chatbot.py", "router.py", "websearch.py", "rag.py"):
        text = open(os.path.join(root, name), encoding="utf-8").read()
        assert "gsk_" not in text
        assert "rnd_" not in text


def test_env_example_has_placeholders_only():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = open(os.path.join(root, ".env.example"), encoding="utf-8").read()
    assert "your_groq_api_key_here" in text
    assert "gsk_" not in text
