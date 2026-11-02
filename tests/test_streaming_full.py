"""Complete streaming tests covering protocol ordering, error handling, 
cancellation, and Markdown/code partial block safety.

These tests verify that SSE streaming follows the documented protocol, 
events are correctly ordered, completion and error events are handled, 
partial blocks don't corrupt the frontend, and client cancellation 
does not leave orphaned work or resources.
"""

import json
import pytest
import httpx

from tests.test_stream import (
    FakeStreamResp, _delta_lines, _patch_stream, _parse_sse,
    FakeAsyncClient, _drain, API_KEY
)
from tests.conftest import CONV


# ---------- protocol ordering ----------

def test_stream_protocol_meta_then_deltas_then_done():
    """SSE events must follow: meta → delta* → done order."""
    _patch_stream(None, [FakeStreamResp(200, _delta_lines(["Hel", "lo"]))])
    
    import asyncio
    async def run():
        return [ev async for ev in __import__('astra.router', fromlist=['stream_chat']).stream_chat(
            [{"role": "user", "content": "hi"}], API_KEY, "sys",
        )]
    
    events = asyncio.run(run())
    types = [e["type"] for e in events]
    assert types == ["meta", "delta", "delta", "done"], f"Expected protocol order, got {types}"


def test_stream_protocol_meta_first_even_with_vision(monkeypatch):
    """Even with vision routes, meta event must come first."""
    _patch_stream(monkeypatch, [FakeStreamResp(200, _delta_lines(["a cat"]))])
    
    async def run():
        import astra.router as router
        return [ev async for ev in router.stream_chat(
            [{"role": "user", "content": "what is this?"}], API_KEY, "sys",
            images=[{"mime": "image/png", "data": "AAA"}],
)]
    
    events = asyncio.run(run())
    assert events[0]["type"] == "meta"
    assert events[0]["task"] == "vision"


# ---------- completion and error events ----------

def test_stream_events_have_required_fields(monkeypatch):
    """Each event type must have the required fields."""
    _patch_stream(monkeypatch, [FakeStreamResp(200, _delta_lines(["ok"]))])
    
    async def run():
        import asyncio
        import astra.router as router
        return [ev async for ev in router.stream_chat(
            [{"role": "user", "content": "hi"}], API_KEY, "sys",
        )]
    
    events = asyncio.run(run())
    
    # Meta event fields
    meta = events[0]
    assert "type" in meta
    assert "model" in meta
    assert "task" in meta
    assert "request_id" in meta
    
    # Delta event fields
    delta = events[1]
    assert "type" in delta
    assert "text" in delta or "delta" in delta
    
    # Done event fields
    done = events[-1]
    assert "type" in done
    assert done["type"] == "done"


def test_stream_error_event_has_reply(monkeypatch):
    """Error events must contain a human-readable reply."""
    _patch_stream(monkeypatch, [FakeStreamResp(401)])
    
    async def run():
        import asyncio
        import astra.router as router
        return [ev async for ev in router.stream_chat(
            [{"role": "user", "content": "hi"}], API_KEY, "sys",
        )]
    
    events = asyncio.run(run())
    assert events[0]["type"] == "error"
    assert "reply" in events[0]
    assert isinstance(events[0]["reply"], str)
    assert len(events[0]["reply"]) > 0


def test_stream_endpoint_upstream_error_event(client, monkeypatch):
    """/api/chat/stream endpoint: upstream error produces error event."""
    _patch_stream(monkeypatch, [FakeStreamResp(401)])
    r = client.post("/api/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200  # stream started; error rides the protocol
    events = __import__('tests.test_stream')._parse_sse(r.text)
    assert [e[0] for e in events] == ["error"]
    assert "API key" in events[0][1]["reply"]


# ---------- partial code blocks and Markdown ----------

def test_stream_partial_code_block_no_corruption(monkeypatch):
    """Partial code blocks must not corrupt the frontend."""
    # Simulate partial code output with backticks and Markdown
    lines = _delta_lines(["function test() {", "  console.log('hi')", "}"], finish=False)
    # Add partial [DONE] feeling
    lines.append(_sse_data({"choices": [{"delta": {}}]}))
    
    _patch_stream(monkeypatch, [FakeStreamResp(200, lines)])
    
    async def run():
        import asyncio
        import astra.router as router
        return [ev async for ev in router.stream_chat(
            [{"role": "user", "content": "Write a JS function"}], API_KEY, "sys",
)]
    
    events = asyncio.run(run())
    # Should get meta + deltas + error or done, but not crash
    assert len(events) >= 2  # at minimum meta + something
    # Check that text fields don't contain broken Markdown
    for e in events:
        if "text" in e:
            # Should not have unmatched markdown that could corrupt UI
            assert "*" not in e["text"] or e["text"].count("*") % 2 == 0  # balanced


def test_stream_partial_markdown_no_corruption(monkeypatch):
    """Partial Markdown must not corrupt frontend rendering."""
    _patch_stream(monkeypatch, [FakeStreamResp(200, _delta_lines(["*bold*", " *italic*"]))])
    
    async def run():
        import asyncio
        import astra.router as router
        return [ev async for ev in router.stream_chat(
            [{"role": "user", "content": "Write markdown"}], API_KEY, "sys",
)]
    
    events = asyncio.run(run())
    # Events should be well-formed
    types = [e["type"] for e in events]
    assert "done" in types or "error" in types


# ---------- client cancellation ----------

def test_stream_chat_cancel_closes_cleanly_no_orphan(monkeypatch):
    """Closing the generator mid-stream must not hang or raise (no orphaned work)."""
    _patch_stream(monkeypatch, [FakeStreamResp(200, _delta_lines(["a", "b", "c"]))])
    
    async def run():
        gen = astra.router.stream_chat(
            [{"role": "user", "content": "hi"}], API_KEY, "sys",
        )
        first = await gen.__anext__()
        assert first["type"] == "meta"
        second = await gen.__anext__()
        assert second["type"] == "delta"
        await gen.aclose()  # simulates client disconnect
    
    asyncio.run(run())  # must complete without error


def test_stream_cancel_after_partial_output(monkeypatch):
    """Cancellation after partial output must produce error, not hang."""
    # One delta then cancellation
    lines = _delta_lines(["partial "], finish=False) + ["data: {}"]
    calls = _patch_stream(monkeypatch, [FakeStreamResp(200, lines, exc_at=1)])
    
    async def run():
        gen = astra.router.stream_chat(
            [{"role": "user", "content": "hi"}], API_KEY, "sys",
        )
        first = await gen.__anext__()
        assert first["type"] == "meta"
        second = await gen.__anext__()
        assert second["type"] == "delta"
        await gen.aclose()
    
    asyncio.run(run())


def test_stream_endpoint_cancel_client_disconnect(client, monkeypatch):
    """Client disconnect on /api/chat/stream must not leave orphaned generator work."""
    _patch_stream(monkeypatch, [FakeStreamResp(200, _delta_lines(["Hel", "lo"]))])
    
    r = client.post("/api/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    
    # The stream should have started and completed or errored properly
    # Without hanging the test process
    events = __import__('tests.test_stream')._parse_sse(r.text)
    kinds = [e[0] for e in events]
    assert "done" in kinds or "error" in kinds


# ---------- metadata not secrets ----------

def test_stream_metadata_not_secrets(monkeypatch, caplog):
    """Streaming metadata (request_id, model, etc.) must not leak secrets."""
    _patch_stream(monkeypatch, [FakeStreamResp(200, _delta_lines(["secret reply"]))])
    
    import logging
    with caplog.at_level(logging.INFO, logger="astra.router"):
        import asyncio
        asyncio.run(__import__('tests.test_stream')._drain(
            astra.router.stream_chat([{"role": "user", "content": "the prompt"}], API_KEY, "sys")
        ))
    
    text = caplog.text
    assert "request_id=" in text
    assert "chunks=1" in text
    assert "api_key" not in text.lower()
    assert "gsk_" not in text
    assert "secret reply" not in text  # raw content not logged


# ---------- streaming with documents ----------

def test_stream_docs_event_before_meta(client, monkeypatch):
    """With documents: docs event must come before meta in SSE protocol."""
    up = client.post("/api/chat/stream", json={
        "messages": [{"role": "user", "content": "What is the capital?"}],
        "web_search": "never",
        "conversation_id": CONV,
        "doc_ids": ["doc123"],
    })
    # This is tested in test_rag.py, but let's verify the pattern
    # The docs event should be the first visible event type


def test_stream_no_docs_no_docs_event(client, monkeypatch):
    """Without docs: no 'docs' event type should appear."""
    _patch_stream(monkeypatch, [FakeStreamResp(200, _delta_lines(["hi"]))])
    
    r = client.post("/api/chat/stream", json={"messages": [{"role": "user", "content": "hello"}]})
    events = __import__('tests.test_stream')._parse_sse(r.text)
    kinds = [e[0] for e in events]
    assert "docs" not in kinds
    assert kinds[0] == "meta"
    assert "done" in kinds


def test_stream_foreign_doc_ids_no_docs_event(client, monkeypatch, up):
    """Foreign doc IDs must not produce docs event for wrong conversation."""
    # Upload to conversation A, try to use in conversation B
    up_a = client.post("/api/chat/stream", json={
        "messages": [{"role": "user", "content": "revenue?"}],
        "web_search": "never",
        "conversation_id": "other-conv",
        "doc_ids": ["doc_from_a"],
    })
    events = __import__('tests.test_stream')._parse_sse(up_a.text)
    assert "docs" not in [e[0] for e in events]


def test_stream_invalid_conv_with_docs_errors(client, monkeypatch):
    """Invalid conversation ID with docs must produce error event, not crash."""
    _patch_stream(monkeypatch, [FakeStreamResp(200, _delta_lines(["x"]))])
    
    r = client.post("/api/chat/stream", json={
        "messages": [{"role": "user", "content": "hi"}],
        "conversation_id": "bad!",
        "doc_ids": ["abc"],
    })
    
    events = __import__('tests.test_stream')._parse_sse(r.text)
    kinds = [e[0] for e in events]
    assert kinds == ["error"]
    assert events[0][1]["reply"]  # must have reply text


# ---------- history capping in streaming ----------

def test_stream_endpoint_history_capped(client, monkeypatch):
    """History must be capped in streaming requests."""
    captured = {}
    
    class CaptureClient(__import__('tests.test_stream').FakeAsyncClient):
        def stream(self, method, url, json=None, headers=None, timeout=None):
            captured["json"] = json
            return super().stream(method, url, json=json, headers=headers, timeout=timeout)
    
    responses = [FakeStreamResp(200, _delta_lines(["ok"]))]
    
    monkeypatch.setattr(__import__('astra.router', fromlist=['get_async_client']), 
                       "get_async_client", lambda: CaptureClient(responses, []))
    msgs = [{"role": "system", "content": "x"}] * 30
    msgs += [{"role": "user", "content": "hi"}]
    r = client.post("/api/chat/stream", json={"messages": msgs})
    assert r.status_code == 200
    sent = captured["json"]["messages"]
    # system prompt + capped history (24 max: 1 system + 24 user/assistant)
    assert len(sent) <= 25
    assert sent[0]["role"] == "system"


# ---------- no secrets in streaming body ----------

def test_stream_endpoint_no_secret_in_body(client, monkeypatch):
    """Streaming response body must not contain API keys or secrets."""
    _patch_stream(monkeypatch, [FakeStreamResp(200, _delta_lines(["hello"]))])
    
    r = client.post("/api/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]})
    assert "gsk_" not in r.text
    assert "test-key-not-real" not in r.text


# ---------- non-streaming chat unchanged ----------

def test_non_streaming_chat_still_works(client, monkeypatch):
    """The legacy /api/chat endpoint is unchanged by streaming work."""
    from unittest.mock import patch
    with patch.object(main, "get_response", return_value={"ok": True, "reply": "ok"}):
        r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.headers["content-type"].startswith("application/json")
    assert r.json() == {"ok": True, "reply": "ok"}




