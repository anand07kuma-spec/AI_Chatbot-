"""Phase 7: coding assistance + sandbox execution tests (mocked)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import coder
import execution
import main


# ---------- coder.is_coding_request ----------

class TestCodingDetection:

    def test_code_fence_detected(self):
        assert coder.is_coding_request("```python\nprint(1)\n```")

    def test_error_phrase_detected(self):
        assert coder.is_coding_request("I get a TypeError when running this, can you debug?")

    def test_refactor_phrase_detected(self):
        assert coder.is_coding_request("Please refactor this function for readability")

    def test_test_generation_detected(self):
        assert coder.is_coding_request("write unit tests for my Calculator class")

    def test_file_extension_detected(self):
        assert coder.is_coding_request("Explain what utils.py does")

    def test_plain_chat_not_detected(self):
        assert not coder.is_coding_request("What's the capital of France?")
        assert not coder.is_coding_request("Tell me a joke about cats")
        assert not coder.is_coding_request("")


# ---------- coder.extract_first_code_block / detect_request_language ----------

class TestCodeBlockHelpers:

    def test_extract_with_lang(self):
        block = coder.extract_first_code_block("```python\nx = 1\n```")
        assert block == ("python", "x = 1")

    def test_extract_without_lang(self):
        block = coder.extract_first_code_block("```\nhello\n```")
        assert block == ("text", "hello")

    def test_extract_none(self):
        assert coder.extract_first_code_block("no code here") is None

    def test_detect_language_from_fence(self):
        assert coder.detect_request_language("```cpp\nint main(){}\n```") == "cpp"

    def test_detect_language_from_extension(self):
        assert coder.detect_request_language("check main.rs for bugs") == "rust"

    def test_detect_language_none(self):
        assert coder.detect_request_language("hello") is None


# ---------- /api/sandbox status ----------

class TestSandboxStatus:

    def test_disabled_by_default(self, client):
        r = client.get("/api/sandbox")
        body = r.json()
        assert r.status_code == 200
        assert body["enabled"] is False
        assert body["languages"] == []

    def test_enabled_reports_languages(self, client, monkeypatch):
        monkeypatch.setenv("ASTRA_SANDBOX_URL", "http://sandbox.test")
        r = client.get("/api/sandbox")
        body = r.json()
        assert body["enabled"] is True
        assert "python" in body["languages"]
        assert "cpp" in body["languages"]


# ---------- /api/execute: disabled, validation ----------

class TestExecuteValidation:

    def test_disabled_returns_503(self, client):
        r = client.post("/api/execute", json={"language": "python", "code": "print(1)"})
        assert r.status_code == 503
        body = r.json()
        assert body["ok"] is False
        assert "not configured" in body["error"].lower()

    def test_disallowed_language_400(self, client, monkeypatch):
        monkeypatch.setenv("ASTRA_SANDBOX_URL", "http://sandbox.test")
        r = client.post("/api/execute", json={"language": "brainfuck", "code": "+++"})
        assert r.status_code == 400
        assert "not allowed" in r.json()["error"]

    def test_empty_code_400(self, client, monkeypatch):
        monkeypatch.setenv("ASTRA_SANDBOX_URL", "http://sandbox.test")
        r = client.post("/api/execute", json={"language": "python", "code": "   "})
        assert r.status_code == 400

    def test_oversize_code_413(self, client, monkeypatch):
        monkeypatch.setenv("ASTRA_SANDBOX_URL", "http://sandbox.test")
        r = client.post("/api/execute", json={
            "language": "python",
            "code": "x" * (execution.CODE_MAX_BYTES + 1),
        })
        assert r.status_code == 413

    def test_oversize_stdin_413(self, client, monkeypatch):
        monkeypatch.setenv("ASTRA_SANDBOX_URL", "http://sandbox.test")
        r = client.post("/api/execute", json={
            "language": "python",
            "code": "print(input())",
            "stdin": "y" * (execution.STDIN_MAX_BYTES + 1),
        })
        assert r.status_code == 413


# ---------- /api/execute: mocked sandbox behavior ----------

class _FakeSandboxResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None):
        return _FakeSandboxResp(200, {
            "version": "3.10",
            "run": {"stdout": "hello\n", "stderr": "", "code": 0, "signal": None},
        })


class TestExecuteMocked:

    def test_success(self, client, monkeypatch):
        monkeypatch.setenv("ASTRA_SANDBOX_URL", "http://sandbox.test")
        with patch.object(main.httpx, "AsyncClient", _FakeAsyncClient):
            r = client.post("/api/execute", json={
                "language": "python",
                "code": "print('hello')",
            })
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["stdout"] == "hello\n"
        assert body["exit_code"] == 0
        assert body["language"] == "python"

    def test_timeout_504(self, client, monkeypatch):
        monkeypatch.setenv("ASTRA_SANDBOX_URL", "http://sandbox.test")

        class _Timeout(_FakeAsyncClient):
            async def post(self, *a, **k):
                raise main.httpx.TimeoutException("slow")

        with patch.object(main.httpx, "AsyncClient", _Timeout):
            r = client.post("/api/execute", json={
                "language": "python",
                "code": "while True: pass",
            })
        assert r.status_code == 504
        assert "timed out" in r.json()["error"].lower()

    def test_unreachable_502(self, client, monkeypatch):
        monkeypatch.setenv("ASTRA_SANDBOX_URL", "http://sandbox.test")

        class _Down(_FakeAsyncClient):
            async def post(self, *a, **k):
                raise main.httpx.ConnectError("refused")

        with patch.object(main.httpx, "AsyncClient", _Down):
            r = client.post("/api/execute", json={
                "language": "python",
                "code": "print(1)",
            })
        assert r.status_code == 502

    def test_output_truncated(self, client, monkeypatch):
        monkeypatch.setenv("ASTRA_SANDBOX_URL", "http://sandbox.test")
        big = "A" * (execution.OUTPUT_MAX_CHARS + 500)

        class _Big(_FakeAsyncClient):
            async def post(self, *a, **k):
                return _FakeSandboxResp(200, {
                    "version": "3.10",
                    "run": {"stdout": big, "stderr": "", "code": 0},
                })

        with patch.object(main.httpx, "AsyncClient", _Big):
            r = client.post("/api/execute", json={
                "language": "python",
                "code": "print('A'*99999)",
            })
        stdout = r.json()["stdout"]
        assert len(stdout) < len(big)
        assert "truncated" in stdout

    def test_compile_error_surfaces_stderr(self, client, monkeypatch):
        monkeypatch.setenv("ASTRA_SANDBOX_URL", "http://sandbox.test")

        class _CompileFail(_FakeAsyncClient):
            async def post(self, *a, **k):
                return _FakeSandboxResp(200, {
                    "version": "11",
                    "compile": {"code": 1, "stderr": "error: expected ';'", "stdout": ""},
                    "run": {"stdout": "", "stderr": "", "code": None},
                })

        with patch.object(main.httpx, "AsyncClient", _CompileFail):
            r = client.post("/api/execute", json={
                "language": "c",
                "code": "int main(){ return 0 }",
            })
        body = r.json()
        assert body["ok"] is True
        assert body["exit_code"] == 1
        assert "expected" in body["stderr"]

    def test_no_secrets_forwarded(self, client, monkeypatch):
        """Host env secrets must never reach the sandbox request."""
        monkeypatch.setenv("ASTRA_SANDBOX_URL", "http://sandbox.test")
        monkeypatch.setenv("ASTRA_SANDBOX_KEY", "provider-key-abc")
        monkeypatch.setenv("GROQ_API_KEY", "gsk_super_secret")
        captured = {}

        class _Capture(_FakeAsyncClient):
            async def post(self, url, json=None, headers=None):
                captured["json"] = json
                captured["headers"] = headers
                return _FakeSandboxResp(200, {
                    "version": "3.10",
                    "run": {"stdout": "", "stderr": "", "code": 0},
                })

        with patch.object(main.httpx, "AsyncClient", _Capture):
            client.post("/api/execute", json={
                "language": "python",
                "code": "import os; print(os.environ)",
            })

        body_str = str(captured["json"])
        assert "gsk_super_secret" not in body_str
        assert "GROQ_API_KEY" not in body_str
        # provider key IS sent as auth, but only that
        assert captured["headers"].get("Authorization") == "Bearer provider-key-abc"
        # payload only carries language/version/files/stdin
        assert set(captured["json"].keys()) <= {
            "language", "version", "files", "stdin",
            "compile_timeout", "run_timeout",
            "compile_memory_limit", "run_memory_limit",
        }


# ---------- coding-mode chat integration ----------

class TestCodingModeChat:

    def test_mode_code_forces_coding_prompt(self, client):
        captured = {}

        def fake_route_chat(messages, api_key, system_prompt, requested_model=None):
            captured["prompt"] = system_prompt
            return {"ok": True, "reply": "Here is the code.", "model": "m", "task": "fast"}

        with patch.object(main, "get_response", side_effect=lambda msgs, model=None, system_prompt=None: fake_route_chat(msgs, "k", system_prompt or "default")):
            r = client.post("/api/chat", json={
                "messages": [{"role": "user", "content": "hello"}],
                "mode": "code",
            })
        assert r.status_code == 200
        assert "coding assistant" in captured["prompt"].lower()

    def test_auto_detect_code_fence_uses_coding_prompt(self, client):
        captured = {}

        def fake_get_response(msgs, model=None, system_prompt=None):
            captured["prompt"] = system_prompt or ""
            return {"ok": True, "reply": "ok", "model": "m", "task": "fast"}

        with patch.object(main, "get_response", side_effect=fake_get_response):
            r = client.post("/api/chat", json={
                "messages": [{"role": "user", "content": "```python\nprint(1)\n```\nExplain this"}],
            })
        assert r.status_code == 200
        assert "coding assistant" in captured["prompt"].lower()

    def test_normal_chat_uses_default_prompt(self, client):
        captured = {}

        def fake_get_response(msgs, model=None, system_prompt=None):
            captured["prompt"] = system_prompt  # None → default SYSTEM_PROMPT
            return {"ok": True, "reply": "ok", "model": "m", "task": "fast"}

        with patch.object(main, "get_response", side_effect=fake_get_response):
            r = client.post("/api/chat", json={
                "messages": [{"role": "user", "content": "What's the weather?"}],
            })
        assert r.status_code == 200
        assert captured["prompt"] is None  # falls back to SYSTEM_PROMPT in chatbot.py
