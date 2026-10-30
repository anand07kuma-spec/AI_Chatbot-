"""Unit tests for the model router: classification, resolution, provider
errors, retries, fallbacks, and metrics hygiene."""

import logging
from types import SimpleNamespace

import httpx
import pytest

import router
from router import (
    ALLOWED_IMAGE_MIMES,
    DEFAULT_FAST_MODEL,
    DEFAULT_REASONING_MODEL,
    DEFAULT_STT_MODEL,
    DEFAULT_VISION_MODEL,
    VISION_MAX_IMAGES,
    build_text_payload,
    build_vision_payload,
    classify_task,
    fast_model,
    is_valid_model_choice,
    reasoning_model,
    resolve_text_model,
    route_chat,
    route_vision,
    router_mode,
    stt_model,
    vision_model,
)

API_KEY = "gsk_test_key_not_real"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "ASTRA_FAST_MODEL",
        "ASTRA_REASONING_MODEL",
        "ASTRA_VISION_MODEL",
        "ASTRA_STT_MODEL",
        "ASTRA_ROUTER_MODE",
        "ASTRA_MODEL_TIMEOUT",
        "GROQ_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(router, "RETRY_BACKOFF_SEC", 0)


class FakeResp:
    def __init__(self, status_code=200, data=None, text=""):
        self.status_code = status_code
        self._data = data if data is not None else {}
        self.text = text or str(self._data)

    def json(self):
        return self._data


def _ok_data(content="hello from model"):
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


# ---------- classification ----------

def test_classify_short_greeting_is_fast():
    assert classify_task([{"role": "user", "content": "hi"}]) == "fast"
    assert classify_task([{"role": "user", "content": "what is 2+2"}]) == "fast"


def test_classify_complex_keywords_are_reasoning():
    assert classify_task([{"role": "user", "content": "Prove that the square root of 2 is irrational"}]) == "reasoning"
    assert classify_task([{"role": "user", "content": "Explain the time complexity of quicksort"}]) == "reasoning"
    assert classify_task([{"role": "user", "content": "Help me debug a race condition in my thread pool"}]) == "reasoning"


def test_classify_code_block_is_reasoning():
    msg = "Review this:\n```python\ndef f(x):\n    return f(x-1)\n```"
    assert classify_task([{"role": "user", "content": msg}]) == "reasoning"


def test_classify_long_message_is_reasoning():
    assert classify_task([{"role": "user", "content": "a" * 600}]) == "reasoning"


def test_classify_long_history_is_reasoning():
    msgs = [{"role": "user", "content": "x" * 2000}, {"role": "assistant", "content": "y" * 2000},
            {"role": "user", "content": "tell me more"}]
    assert classify_task(msgs) == "reasoning"


def test_classify_uncertain_defaults_to_reasoning():
    # 200 chars, no signal: quality-safe default
    msgs = [{"role": "user", "content": "b" * 200}]
    assert classify_task(msgs) == "reasoning"


def test_classify_empty_defaults_to_reasoning():
    assert classify_task([]) == "reasoning"


# ---------- configuration & resolution ----------

def test_default_models():
    assert fast_model() == "openai/gpt-oss-20b"
    assert reasoning_model() == "openai/gpt-oss-120b"
    assert vision_model() == "qwen/qwen3.8-27b"
    assert stt_model() == "whisper-large-v3-turbo"
    assert router_mode() == "auto"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("ASTRA_FAST_MODEL", "my/fast-model")
    monkeypatch.setenv("ASTRA_REASONING_MODEL", "my/reason-model")
    monkeypatch.setenv("ASTRA_VISION_MODEL", "my/vision-model")
    monkeypatch.setenv("ASTRA_STT_MODEL", "my/stt-model")
    assert fast_model() == "my/fast-model"
    assert reasoning_model() == "my/reason-model"
    assert vision_model() == "my/vision-model"
    assert stt_model() == "my/stt-model"


def test_legacy_groq_model_alias(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "legacy/model")
    assert reasoning_model() == "legacy/model"
    monkeypatch.setenv("ASTRA_REASONING_MODEL", "new/model")
    assert reasoning_model() == "new/model"  # explicit beats legacy


def test_invalid_router_mode_falls_back_to_auto(monkeypatch):
    monkeypatch.setenv("ASTRA_ROUTER_MODE", "banana")
    assert router_mode() == "auto"


def test_resolve_requested_modes():
    assert resolve_text_model("fast", "fast") == fast_model()
    assert resolve_text_model("fast", "reasoning") == reasoning_model()
    assert resolve_text_model("fast", "some/custom-model") == "some/custom-model"


def test_resolve_auto_follows_classification():
    assert resolve_text_model("fast", None) == fast_model()
    assert resolve_text_model("reasoning", None) == reasoning_model()
    assert resolve_text_model("fast", "auto") == fast_model()


def test_resolve_router_mode_forces_tier(monkeypatch):
    monkeypatch.setenv("ASTRA_ROUTER_MODE", "fast")
    assert resolve_text_model("reasoning", "auto") == fast_model()
    monkeypatch.setenv("ASTRA_ROUTER_MODE", "reasoning")
    assert resolve_text_model("fast", "auto") == reasoning_model()


def test_resolve_client_choice_beats_router_mode(monkeypatch):
    monkeypatch.setenv("ASTRA_ROUTER_MODE", "fast")
    assert resolve_text_model("fast", "reasoning") == reasoning_model()


@pytest.mark.parametrize("value,valid", [
    ("auto", True),
    ("fast", True),
    ("REASONING", True),
    ("openai/gpt-oss-120b", True),
    ("llama-3.3-70b-versatile", True),
    ("rm -rf /", False),
    ("a" * 120, False),
    ("", False),
    ("has space", False),
])
def test_is_valid_model_choice(value, valid):
    assert is_valid_model_choice(value) is valid


# ---------- payload adapters ----------

def test_build_text_payload_shape():
    payload = build_text_payload("m1", [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}])
    assert payload["model"] == "m1"
    assert payload["stream"] is False
    assert payload["messages"][0]["role"] == "system"
    assert payload["max_tokens"] == 2048


def test_build_vision_payload_caps_images():
    images = [{"mime": "image/png", "data": "AAA"}] * 10
    payload = build_vision_payload(
        "vis", [{"role": "user", "content": "what?"}], images, "sys"
    )
    last = payload["messages"][-1]
    assert last["role"] == "user"
    assert isinstance(last["content"], list)
    img_parts = [p for p in last["content"] if p["type"] == "image_url"]
    assert len(img_parts) == VISION_MAX_IMAGES
    assert img_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")


# ---------- route_chat: provider errors ----------

def _patch_post(monkeypatch, responses):
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(json)
        resp = responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp

    # router now posts via the pooled sync_client (http_pool.sync_client)
    monkeypatch.setattr(router, "sync_client", SimpleNamespace(post=fake_post))
    return calls


def test_route_chat_success(monkeypatch):
    _patch_post(monkeypatch, [FakeResp(200, _ok_data("answer"))])
    result = route_chat(
        [{"role": "user", "content": "hi"}], API_KEY, "sys"
    )
    assert result == {
        "ok": True,
        "reply": "answer",
        "model": fast_model(),
        "task": "fast",
    }


def test_route_chat_complex_uses_reasoning_model(monkeypatch):
    calls = _patch_post(monkeypatch, [FakeResp(200, _ok_data("deep"))])
    result = route_chat(
        [{"role": "user", "content": "Prove Euler's identity step by step"}], API_KEY, "sys"
    )
    assert result["ok"] is True
    assert calls[0]["model"] == reasoning_model()


def test_route_chat_falls_back_on_404(monkeypatch):
    calls = _patch_post(monkeypatch, [FakeResp(404, {"error": "model_decommissioned"}), FakeResp(200, _ok_data("fb"))])
    result = route_chat([{"role": "user", "content": "hi"}], API_KEY, "sys")
    assert result["ok"] is True
    assert result["reply"] == "fb"
    assert calls[0]["model"] == fast_model()
    assert calls[1]["model"] == reasoning_model()


def test_route_chat_no_fallback_for_explicit_model(monkeypatch):
    calls = _patch_post(monkeypatch, [FakeResp(404, {"error": "model_not_found"})])
    result = route_chat(
        [{"role": "user", "content": "hi"}], API_KEY, "sys", requested_model="custom/gone-model"
    )
    assert result["ok"] is False
    assert "not available" in result["reply"]
    assert len(calls) == 1  # explicit id: clear error, no silent switch


def test_route_chat_404_on_both_models_clear_error(monkeypatch):
    _patch_post(monkeypatch, [FakeResp(404), FakeResp(404)])
    result = route_chat([{"role": "user", "content": "hi"}], API_KEY, "sys")
    assert result["ok"] is False
    assert "not available" in result["reply"]


def test_route_chat_auth_error_no_retry(monkeypatch):
    calls = _patch_post(monkeypatch, [FakeResp(401, {"error": {"message": "bad key"}})])
    result = route_chat([{"role": "user", "content": "hi"}], API_KEY, "sys")
    assert result["ok"] is False
    assert "API key" in result["reply"]
    assert len(calls) == 1


def test_route_chat_rate_limit_retries_then_fails(monkeypatch):
    calls = _patch_post(monkeypatch, [FakeResp(429), FakeResp(429)])
    result = route_chat([{"role": "user", "content": "hi"}], API_KEY, "sys")
    assert result["ok"] is False
    assert "rate limit" in result["reply"].lower()
    assert len(calls) == router.MAX_ATTEMPTS


def test_route_chat_5xx_retried_then_success(monkeypatch):
    calls = _patch_post(monkeypatch, [FakeResp(503), FakeResp(200, _ok_data("recovered"))])
    result = route_chat([{"role": "user", "content": "hi"}], API_KEY, "sys")
    assert result["ok"] is True
    assert len(calls) == 2


def test_route_chat_timeout_returns_clear_error(monkeypatch):
    _patch_post(monkeypatch, [httpx.TimeoutException("t"), httpx.TimeoutException("t")])
    result = route_chat([{"role": "user", "content": "hi"}], API_KEY, "sys")
    assert result["ok"] is False
    assert "too long" in result["reply"]


def test_route_chat_network_error_returns_clear_error(monkeypatch):
    _patch_post(monkeypatch, [ConnectionError("boom"), ConnectionError("boom")])
    result = route_chat([{"role": "user", "content": "hi"}], API_KEY, "sys")
    assert result["ok"] is False
    assert "try again" in result["reply"].lower()


def test_route_chat_empty_answer_is_error(monkeypatch):
    _patch_post(monkeypatch, [FakeResp(200, _ok_data(""))])
    result = route_chat([{"role": "user", "content": "hi"}], API_KEY, "sys")
    assert result["ok"] is False


# ---------- route_vision ----------

def test_route_vision_uses_vision_model(monkeypatch):
    calls = _patch_post(monkeypatch, [FakeResp(200, _ok_data("a red square"))])
    images = [{"mime": "image/png", "data": "AAA"}]
    result = route_vision([{"role": "user", "content": "what is this?"}], images, API_KEY, "sys")
    assert result["ok"] is True
    assert result["model"] == vision_model()
    assert result["task"] == "vision"
    assert calls[0]["model"] == vision_model()


def test_route_vision_no_text_fallback_on_404(monkeypatch):
    """Vision must never silently fall back to a text model."""
    calls = _patch_post(monkeypatch, [FakeResp(404), FakeResp(200, _ok_data("should not happen"))])
    result = route_vision(
        [{"role": "user", "content": "look"}], [{"mime": "image/png", "data": "AAA"}], API_KEY, "sys"
    )
    assert result["ok"] is False
    assert "not available" in result["reply"]
    assert len(calls) == 1
    assert calls[0]["model"] == vision_model()


# ---------- metrics hygiene ----------

def test_logs_contain_metadata_but_not_secrets(monkeypatch, caplog):
    _patch_post(monkeypatch, [FakeResp(200, _ok_data("secret reply content"))])
    with caplog.at_level(logging.INFO, logger="astra.router"):
        route_chat([{"role": "user", "content": "the prompt itself is secret"}], API_KEY, "sys")
    text = caplog.text
    assert "openai/gpt-oss-20b" in text
    assert "latency_ms=" in text
    assert "tokens=15" in text
    assert API_KEY not in text
    assert "gsk_" not in text
    assert "secret reply content" not in text
    assert "the prompt itself is secret" not in text
