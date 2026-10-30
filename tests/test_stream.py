"""Tests for the SSE streaming chat endpoint and router stream_chat."""

import asyncio
import json
import logging

import httpx
import pytest

import router
from router import fast_model, reasoning_model, vision_model, stream_chat


API_KEY = "gsk_test_key_not_real"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "ASTRA_FAST_MODEL",
        "ASTRA_REASONING_MODEL",
        "ASTRA_VISION_MODEL",
        "ASTRA_ROUTER_MODE",
        "ASTRA_MODEL_TIMEOUT",
        "ASTRA_STREAM_IDLE_TIMEOUT",
        "GROQ_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(router, "RETRY_BACKOFF_SEC", 0)


# ---------- fake streaming httpx.AsyncClient ----------

def _sse_data(obj) -> str:
    return f"data: {json.dumps(obj)}"


def _delta_lines(texts, finish=True):
    lines = [_sse_data({"choices": [{"delta": {"content": t}}]}) for t in texts]
    if finish:
        lines.append(_sse_data({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
        lines.append("data: [DONE]")
    return lines


class FakeStreamResp:
    def __init__(self, status_code=200, lines=None, exc_at=None):
        self.status_code = status_code
        self._lines = lines or []
        self._exc_at = exc_at

    async def aiter_lines(self):
        for i, line in enumerate(self._lines):
            if self._exc_at is not None and i == self._exc_at:
                raise httpx.ReadError("connection lost")
            yield line

    async def aread(self):
        return b"upstream error"


class FakeStreamCM:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *args):
        return False


class FakeAsyncClient:
    """Stands in for httpx.AsyncClient with scripted .stream() responses."""

    def __init__(self, responses, calls):
        self._responses = responses  # shared list: each .stream() pops the next
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def stream(self, method, url, json=None, headers=None, timeout=None):
        self._calls.append({"method": method, "url": url, "json": json, "headers": headers})
        return FakeStreamCM(self._responses.pop(0))


def _patch_stream(monkeypatch, responses):
    calls = []

    # stream_chat now uses the pooled per-loop client from http_pool
    monkeypatch.setattr(router, "get_async_client", lambda: FakeAsyncClient(responses, calls))
    return calls


def _parse_sse(body: str):
    """Parse an SSE body into a list of (event, data) tuples."""
    events = []
    for frame in body.strip().split("\n\n"):
        ev, data = None, None
        for line in frame.split("\n"):
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        if ev is not None:
            events.append((ev, data))
    return events


# ---------- router.stream_chat unit tests ----------

def test_stream_chat_emits_meta_deltas_done(monkeypatch):
    _patch_stream(monkeypatch, [FakeStreamResp(200, _delta_lines(["Hel", "lo"]))])

    async def run():
        return [ev async for ev in stream_chat(
            [{"role": "user", "content": "hi"}], API_KEY, "sys",
        )]

    events = asyncio.run(run())
    assert [e["type"] for e in events] == ["meta", "delta", "delta", "done"]
    assert events[0]["model"] == fast_model()
    assert events[0]["task"] == "fast"
    assert events[0]["request_id"]
    assert "".join(e["text"] for e in events[1:3]) == "Hello"
    assert events[-1]["request_id"] == events[0]["request_id"]


def test_stream_chat_uses_reasoning_model(monkeypatch):
    calls = _patch_stream(monkeypatch, [FakeStreamResp(200, _delta_lines(["deep"]))])

    async def run():
        return [ev async for ev in stream_chat(
            [{"role": "user", "content": "Prove Euler's identity step by step"}],
            API_KEY, "sys",
        )]

    events = asyncio.run(run())
    assert events[0]["model"] == reasoning_model()
    assert calls[0]["json"]["model"] == reasoning_model()
    assert calls[0]["json"]["stream"] is True


def test_stream_chat_falls_back_on_404(monkeypatch):
    calls = _patch_stream(monkeypatch, [
        FakeStreamResp(404),
        FakeStreamResp(200, _delta_lines(["fb"])),
    ])

    async def run():
        return [ev async for ev in stream_chat(
            [{"role": "user", "content": "hi"}], API_KEY, "sys",
        )]

    events = asyncio.run(run())
    assert events[0]["model"] == reasoning_model()  # fallback tier
    assert calls[0]["json"]["model"] == fast_model()
    assert calls[1]["json"]["model"] == reasoning_model()
    assert [e["type"] for e in events] == ["meta", "delta", "done"]


def test_stream_chat_rate_limit_safe_error(monkeypatch):
    _patch_stream(monkeypatch, [FakeStreamResp(429), FakeStreamResp(429)])

    async def run():
        return [ev async for ev in stream_chat(
            [{"role": "user", "content": "hi"}], API_KEY, "sys",
        )]

    events = asyncio.run(run())
    assert [e["type"] for e in events] == ["error"]
    assert "rate limit" in events[0]["reply"].lower()
    assert events[0]["request_id"]


def test_stream_chat_no_retry_after_partial_output(monkeypatch):
    """A mid-stream provider failure must NOT restart (would duplicate text)."""
    lines = _delta_lines(["partial "], finish=False) + ["data: {}"]
    calls = _patch_stream(monkeypatch, [
        FakeStreamResp(200, lines, exc_at=1),
    ])

    async def run():
        return [ev async for ev in stream_chat(
            [{"role": "user", "content": "hi"}], API_KEY, "sys",
        )]

    events = asyncio.run(run())
    types = [e["type"] for e in events]
    assert types == ["meta", "delta", "error"]  # one delta kept, then safe error
    assert len(calls) == 1  # no second provider call
    assert "try again" in events[-1]["reply"].lower()


def test_stream_chat_vision_route(monkeypatch):
    calls = _patch_stream(monkeypatch, [FakeStreamResp(200, _delta_lines(["a cat"]))])
    images = [{"mime": "image/png", "data": "AAA"}]

    async def run():
        return [ev async for ev in stream_chat(
            [{"role": "user", "content": "what is this?"}], API_KEY, "sys",
            images=images,
        )]

    events = asyncio.run(run())
    assert events[0]["task"] == "vision"
    assert events[0]["model"] == vision_model()
    assert calls[0]["json"]["model"] == vision_model()
    assert calls[0]["json"]["stream"] is True


def test_stream_chat_vision_no_text_fallback(monkeypatch):
    calls = _patch_stream(monkeypatch, [FakeStreamResp(404)])

    async def run():
        return [ev async for ev in stream_chat(
            [{"role": "user", "content": "look"}], API_KEY, "sys",
            images=[{"mime": "image/png", "data": "AAA"}],
        )]

    events = asyncio.run(run())
    assert [e["type"] for e in events] == ["error"]
    assert len(calls) == 1


def test_stream_chat_cancel_closes_cleanly(monkeypatch):
    """Closing the generator mid-stream (client disconnect) must not hang or raise."""
    _patch_stream(monkeypatch, [FakeStreamResp(200, _delta_lines(["a", "b", "c"]))])

    async def run():
        gen = stream_chat([{"role": "user", "content": "hi"}], API_KEY, "sys")
        first = await gen.__anext__()
        assert first["type"] == "meta"
        second = await gen.__anext__()
        assert second["type"] == "delta"
        await gen.aclose()  # simulates client disconnect

    asyncio.run(run())  # must complete without error


def test_stream_chat_logs_metadata_not_secrets(monkeypatch, caplog):
    _patch_stream(monkeypatch, [FakeStreamResp(200, _delta_lines(["secret reply"]))])
    with caplog.at_level(logging.INFO, logger="astra.router"):
        asyncio.run(_drain(
            stream_chat([{"role": "user", "content": "the prompt"}], API_KEY, "sys")
        ))
    text = caplog.text
    assert "request_id=" in text
    assert "chunks=1" in text
    assert API_KEY not in text
    assert "gsk_" not in text
    assert "secret reply" not in text
    assert "the prompt" not in text


async def _drain(gen):
    async for _ in gen:
        pass


# ---------- /api/chat/stream endpoint tests ----------

def test_stream_endpoint_sse_contract(client, monkeypatch):
    _patch_stream(monkeypatch, [FakeStreamResp(200, _delta_lines(["Hel", "lo"]))])
    r = client.post("/api/chat/stream", json={
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers.get("x-request-id")
    events = _parse_sse(r.text)
    kinds = [e[0] for e in events]
    assert kinds[0] == "meta"
    assert kinds[-1] == "done"
    assert "delta" in kinds
    assert events[0][1]["model"] == fast_model()
    assert events[0][1]["request_id"] == r.headers["x-request-id"]


def test_stream_endpoint_text_joins(client, monkeypatch):
    _patch_stream(monkeypatch, [FakeStreamResp(200, _delta_lines(["Yes,", " it ", "works."]))])
    r = client.post("/api/chat/stream", json={
        "messages": [{"role": "user", "content": "hi"}],
    })
    events = _parse_sse(r.text)
    text = "".join(d["text"] for e, d in events if e == "delta")
    assert text == "Yes, it works."


def test_stream_endpoint_model_choice(client, monkeypatch):
    calls = _patch_stream(monkeypatch, [FakeStreamResp(200, _delta_lines(["ok"]))])
    r = client.post("/api/chat/stream", json={
        "messages": [{"role": "user", "content": "hi"}],
        "model": "reasoning",
    })
    assert r.status_code == 200
    assert calls[0]["json"]["model"] == reasoning_model()


def test_stream_endpoint_validation_errors_are_json(client):
    r = client.post("/api/chat/stream", json={"messages": []})
    assert r.status_code == 400
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["ok"] is False

    r = client.post("/api/chat/stream", json={
        "messages": [{"role": "user", "content": "hi"}],
        "model": "bad model; rm",
    })
    assert r.status_code == 400
    assert "Invalid model" in r.json()["reply"]


def test_stream_endpoint_upstream_error_event(client, monkeypatch):
    _patch_stream(monkeypatch, [FakeStreamResp(401)])
    r = client.post("/api/chat/stream", json={
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200  # stream started; error rides the protocol
    events = _parse_sse(r.text)
    assert [e[0] for e in events] == ["error"]
    assert "API key" in events[0][1]["reply"]


def test_stream_endpoint_images_use_vision(client, monkeypatch, tiny_png_b64):
    calls = _patch_stream(monkeypatch, [FakeStreamResp(200, _delta_lines(["a square"]))])
    r = client.post("/api/chat/stream", json={
        "messages": [{"role": "user", "content": "what is this?"}],
        "images": [{"mime": "image/png", "data": tiny_png_b64}],
    })
    assert r.status_code == 200
    events = _parse_sse(r.text)
    assert events[0][1]["task"] == "vision"
    assert calls[0]["json"]["model"] == vision_model()


def test_stream_endpoint_history_capped(client, monkeypatch):
    captured = {}

    class CaptureClient(FakeAsyncClient):
        def stream(self, method, url, json=None, headers=None, timeout=None):
            captured["json"] = json
            return super().stream(method, url, json=json, headers=headers, timeout=timeout)

    responses = [FakeStreamResp(200, _delta_lines(["ok"]))]

    monkeypatch.setattr(router, "get_async_client", lambda: CaptureClient(responses, []))
    msgs = [{"role": "system", "content": "x"}] * 30
    msgs += [{"role": "user", "content": "hi"}]
    r = client.post("/api/chat/stream", json={"messages": msgs})
    assert r.status_code == 200
    sent = captured["json"]["messages"]
    # system prompt + capped history (24)
    assert len(sent) <= 25
    assert sent[0]["role"] == "system"


def test_stream_endpoint_no_secret_in_body(client, monkeypatch):
    _patch_stream(monkeypatch, [FakeStreamResp(200, _delta_lines(["hello"]))])
    r = client.post("/api/chat/stream", json={
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert "gsk_" not in r.text
    assert "test-key-not-real" not in r.text


def test_non_streaming_chat_still_works(client, monkeypatch):
    """The legacy /api/chat endpoint is unchanged."""
    from unittest.mock import patch
    import main
    with patch.object(main, "get_response", return_value={"ok": True, "reply": "ok"}):
        r = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert r.headers["content-type"].startswith("application/json")
    assert r.json() == {"ok": True, "reply": "ok"}
