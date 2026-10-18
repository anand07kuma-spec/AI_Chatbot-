"""Phase 9 tests: production hardening — request ids, usage counters,
concurrency cap, log redaction, malformed input."""

import logging
import os
from types import SimpleNamespace

import main
import router
from router import (
    MAX_CONCURRENT_UPSTREAM,
    _acquire_slot,
    _error_response,
    _is_retryable,
    _record_usage,
    _release_slot,
    route_chat,
    usage_snapshot,
)


# ---------- X-Request-ID middleware ----------

def test_every_response_has_request_id(client):
    r = client.get("/health")
    assert "x-request-id" in r.headers
    assert len(r.headers["x-request-id"]) >= 8


def test_request_id_unique_per_request(client):
    a = client.get("/health").headers["x-request-id"]
    b = client.get("/health").headers["x-request-id"]
    assert a != b


def test_request_id_on_error_responses(client):
    r = client.post("/api/chat", json={"not": "valid"})
    assert r.status_code == 422
    assert "x-request-id" in r.headers


# ---------- token usage counters ----------

def test_record_usage_accumulates():
    before = usage_snapshot()
    _record_usage({"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
    after = usage_snapshot()
    assert after["calls"] == before["calls"] + 1
    assert after["prompt_tokens"] == before["prompt_tokens"] + 10
    assert after["completion_tokens"] == before["completion_tokens"] + 5
    assert after["total_tokens"] == before["total_tokens"] + 15


def test_record_usage_ignores_bad_data():
    before = usage_snapshot()
    _record_usage(None)
    _record_usage("not a dict")
    # dicts still count as a call, but token fields stay unchanged
    _record_usage({"usage": {}})
    after = usage_snapshot()
    assert after["calls"] == before["calls"] + 1
    assert after["prompt_tokens"] == before["prompt_tokens"]
    assert after["completion_tokens"] == before["completion_tokens"]
    assert after["total_tokens"] == before["total_tokens"]


def test_route_chat_records_usage(monkeypatch):
    class FakeResp:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            }

    monkeypatch.setattr(
        router, "sync_client", SimpleNamespace(post=lambda *a, **k: FakeResp())
    )
    before = usage_snapshot()
    result = route_chat([{"role": "user", "content": "hi"}], "test-key", "sys")
    assert result["ok"] is True
    assert usage_snapshot()["calls"] == before["calls"] + 1


# ---------- concurrency cap / busy ----------

def test_acquire_release_slot_roundtrip():
    assert _acquire_slot(blocking=False) is True
    _release_slot()


def test_busy_not_retryable():
    assert _is_retryable("busy") is False


def test_busy_error_message_is_safe():
    err = _error_response("busy", "some-model")
    assert err["ok"] is False
    assert "too many requests" in err["reply"].lower()


def test_route_chat_busy_when_slots_exhausted(monkeypatch):
    # drain the semaphore
    taken = 0
    while _acquire_slot(blocking=False):
        taken += 1
    try:
        called = {"post": 0}

        def fake_post(*a, **k):
            called["post"] += 1

        monkeypatch.setattr(router, "sync_client", SimpleNamespace(post=fake_post))
        result = route_chat([{"role": "user", "content": "hi"}], "test-key", "sys")
        assert result["ok"] is False
        assert "too many requests" in result["reply"].lower()
        assert called["post"] == 0
    finally:
        for _ in range(taken):
            _release_slot()


# ---------- log redaction: keys never logged ----------

def test_router_logs_never_contain_api_key(monkeypatch, caplog):
    class FakeResp:
        status_code = 500
        text = "internal error"

        def json(self):
            return {}

    secret = "gsk_super_secret_key_do_not_leak"
    monkeypatch.setattr(
        router, "sync_client", SimpleNamespace(post=lambda *a, **k: FakeResp())
    )
    with caplog.at_level(logging.DEBUG):
        result = route_chat([{"role": "user", "content": "hi"}], secret, "sys")
    assert result["ok"] is False
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert secret not in joined
    assert "gsk_super" not in joined


# ---------- malformed / hostile input ----------

def test_chat_rejects_non_json_body(client):
    r = client.post(
        "/api/chat",
        content=b"\x00\x01not json",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 422


def test_chat_rejects_missing_messages(client):
    r = client.post("/api/chat", json={})
    assert r.status_code == 422


def test_chat_rejects_non_list_messages(client):
    r = client.post("/api/chat", json={"messages": "hello"})
    assert r.status_code == 422


def test_stream_rejects_oversized_body(client):
    # content-length declared over the 30MB cap -> 413 from BodyLimitMiddleware
    r = client.post(
        "/api/chat/stream",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"content-length": str(31 * 1024 * 1024)},
    )
    assert r.status_code == 413


def test_stream_rejects_missing_messages(client):
    r = client.post("/api/chat/stream", json={})
    assert r.status_code == 422


# ---------- health usage shape ----------

def test_health_includes_usage(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    usage = body["usage"]
    assert isinstance(usage["calls"], int)
    assert isinstance(usage["total_tokens"], int)
