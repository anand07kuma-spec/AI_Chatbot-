"""Phase 6 voice tests: voice selection module + TTS/STT endpoints (mocked)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import main
import voice


# ---------- voice.pick_voice / detect_script / get_voice ----------

class TestVoiceSelection:

    def test_pref_beats_detection(self):
        # English text, Hindi preference → Hindi voice (user choice wins)
        assert voice.pick_voice("Hello world", "hi") == voice.get_voice("hi")

    def test_romanized_hinglish_not_detected(self):
        # Romanized Hinglish must NOT be guessed as Hindi
        assert voice.detect_script("namaste kya haal hai") is None
        assert voice.pick_voice("namaste kya haal hai", "auto") == voice.get_voice("en")

    def test_devanagari_detected_hindi(self):
        assert voice.detect_script("नमस्ते दुनिया") == "hi"
        assert voice.pick_voice("नमस्ते दुनिया", "auto") == voice.get_voice("hi")

    def test_arabic_detected_urdu(self):
        assert voice.detect_script("السلام علیکم") == "ur"

    def test_env_voice_override(self, monkeypatch):
        monkeypatch.setenv("ASTRA_VOICE_EN", "en-XX-CustomNeural")
        assert voice.get_voice("en") == "en-XX-CustomNeural"

    def test_invalid_preference_falls_to_auto(self):
        assert voice.pick_voice("hello", "fr") == voice.get_voice("en")


# ---------- voice.strip_for_speech ----------

class TestStripForSpeech:

    def test_removes_html_tags(self):
        out = voice.strip_for_speech("<b>bold</b> and <script>alert(1)</script> text")
        assert "<" not in out
        assert "bold" in out
        assert "alert" not in out

    def test_removes_code_and_urls(self):
        out = voice.strip_for_speech("See https://example.com/x for ```code block``` and `inline`")
        assert "https://" not in out
        assert "code block" not in out
        assert "inline" in out

    def test_keeps_plain_hindi(self):
        out = voice.strip_for_speech("नमस्ते, आपका स्वागत है।")
        assert "नमस्ते" in out


# ---------- /api/tts with lang ----------

class TestTTSLang:

    def test_lang_pref_selects_voice(self, client):
        with patch.object(main, "_gemini_tts", new=AsyncMock(return_value=b"MP3")) as m:
            r = client.post("/api/tts", json={"text": "Hello", "lang": "hi"})
        assert r.status_code == 200
        assert m.call_args[0][1] == main.VOICE_HI

    def test_auto_lang_devanagari_selects_hindi(self, client):
        with patch.object(main, "_gemini_tts", new=AsyncMock(return_value=b"MP3")) as m:
            client.post("/api/tts", json={"text": "नमस्ते दोस्त", "lang": "auto"})
        assert m.call_args[0][1] == main.VOICE_HI

    def test_lang_ignored_for_invalid_value(self, client):
        with patch.object(main, "_gemini_tts", new=AsyncMock(return_value=b"MP3")) as m:
            client.post("/api/tts", json={"text": "Hello", "lang": "fr"})
        assert m.call_args[0][1] == main.VOICE_EN

    def test_cache_separate_per_lang(self, client, monkeypatch):
        # distinct voices per language → distinct cache keys → two calls
        monkeypatch.setenv("ASTRA_VOICE_EN", "VoiceEn")
        monkeypatch.setenv("ASTRA_VOICE_HI", "VoiceHi")
        calls = AsyncMock(return_value=b"WAV")
        with patch.object(main, "_gemini_tts", new=calls):
            client.post("/api/tts", json={"text": "Hello", "lang": "en"})
            client.post("/api/tts", json={"text": "Hello", "lang": "hi"})
        assert calls.await_count == 2

    def test_html_not_spoken(self, client):
        with patch.object(main, "_gemini_tts", new=AsyncMock(return_value=b"MP3")) as m:
            client.post("/api/tts", json={"text": "<img src=x onerror=alert(1)> Hi"})
        spoken = m.call_args[0][0]
        assert "<" not in spoken
        assert "onerror" not in spoken


# ---------- /api/transcribe error paths ----------

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
        resp.json.return_value = {"text": "hello"}
        return resp


class TestTranscribeErrors:

    def test_silent_audio_422_with_error(self, client):
        class _Silent(_FakeAsyncClient):
            async def post(self, *a, **k):
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = {"text": "   "}
                return resp

        with patch.object(main.httpx, "AsyncClient", _Silent):
            r = client.post(
                "/api/transcribe",
                files={"file": ("a.webm", b"xxxx", "audio/webm")},
            )
        assert r.status_code == 422
        body = r.json()
        assert body["ok"] is False
        assert "speech" in body["error"].lower()

    def test_bad_mime_400_with_message(self, client):
        r = client.post(
            "/api/transcribe",
            files={"file": ("a.mp3", b"data", "video/mp4")},
        )
        assert r.status_code == 400
        assert "unsupported" in r.json()["error"].lower()

    def test_lang_hi_forwarded(self, client):
        captured = {}

        class _Capture(_FakeAsyncClient):
            async def post(self, url, **k):
                captured["data"] = k.get("data")
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = {"text": "नमस्ते"}
                return resp

        with patch.object(main.httpx, "AsyncClient", _Capture):
            r = client.post(
                "/api/transcribe",
                data={"lang": "hi"},
                files={"file": ("a.webm", b"xxxx", "audio/webm")},
            )
        assert captured["data"].get("language") == "hi"
        assert r.status_code == 200

    def test_lang_ur_forwarded(self, client):
        captured = {}

        class _Capture(_FakeAsyncClient):
            async def post(self, url, **k):
                captured["data"] = k.get("data")
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = {"text": "hello"}
                return resp

        with patch.object(main.httpx, "AsyncClient", _Capture):
            client.post(
                "/api/transcribe",
                data={"lang": "ur"},
                files={"file": ("a.webm", b"xxxx", "audio/webm")},
            )
        assert captured["data"].get("language") == "ur"

    def test_oversize_413(self, client, monkeypatch):
        monkeypatch.setattr(main, "AUDIO_MAX_BYTES", 10)
        r = client.post(
            "/api/transcribe",
            files={"file": ("a.webm", b"x" * 50, "audio/webm")},
        )
        assert r.status_code == 413
        assert "large" in r.json()["error"].lower()
