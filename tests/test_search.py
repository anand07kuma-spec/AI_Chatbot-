"""Tests for the live web search service: policy, providers, caching, SSE."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

import main
import websearch
from websearch import (
    SEARCH_UNAVAILABLE_MSG,
    SearchResult,
    TavilyProvider,
    augment_with_search,
    decide_search,
    format_search_context,
    maybe_search,
    sanitize_url,
    sources_payload,
)
from test_stream import FakeStreamResp, _delta_lines, _patch_stream, _parse_sse


@pytest.fixture(autouse=True)
def _clean_search_state(monkeypatch):
    websearch._cache.clear()
    monkeypatch.setattr(websearch, "_last_fetch", 0.0)
    monkeypatch.setattr(websearch, "SEARCH_MIN_INTERVAL", 0.0)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("ASTRA_SEARCH_PROVIDER", raising=False)


def _fake_provider(results=None, exc=None, name="fake"):
    provider = TavilyProvider.__new__(TavilyProvider)  # duck-typed stand-in
    calls = []
    provider.name = name

    def search(query, max_results=5):
        calls.append(query)
        if exc is not None:
            raise exc
        return list(results or [])

    provider.search = search
    provider.calls = calls
    return provider


# ---------- decision policy ----------

def test_decide_never_mode_skips():
    assert decide_search("latest news today", "never") == (False, "mode_never")


def test_decide_always_mode_forces():
    assert decide_search("hello", "always")[0] is True


def test_decide_auto_time_sensitive():
    for q in (
        "what is the latest news about Mars missions?",
        "current price of gold per gram",
        "aaj ka weather kya hai",
        "iPhone release date 2026",
        "stock price of TCS today",
    ):
        should, reason = decide_search(q, "auto")
        assert should is True, q
        assert reason in ("time_sensitive", "year_mentioned")


def test_decide_auto_conceptual_skips():
    for q in (
        "explain recursion with an example",
        "write a python function to sort a list",
        "what is the difference between a list and a tuple",
    ):
        should, _ = decide_search(q, "auto")
        assert should is False, q


def test_decide_auto_sources_requested():
    for q in (
        "explain quantum computing with sources",
        "cite your sources for that claim",
        "can you give me links for further reading",
    ):
        should, reason = decide_search(q, "auto")
        assert should is True, q
        assert reason == "sources_requested"


def test_sanitize_url_allows_http_only():
    assert sanitize_url("https://example.com/a?b=1") == "https://example.com/a?b=1"
    assert sanitize_url("http://example.com") == "http://example.com"
    assert sanitize_url("javascript:alert(1)") == ""
    assert sanitize_url("data:text/html,x") == ""
    assert sanitize_url("ftp://example.com") == ""
    assert sanitize_url("") == ""
    assert sanitize_url("https://" + "a" * 600) == ""


# ---------- Tavily provider parsing ----------

def _tavily_response(results):
    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": results}

    return FakeResp()


def test_tavily_parses_and_filters_results(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _tavily_response([
            {
                "title": "Mars mission update",
                "url": "https://example.com/mars",
                "content": "Snippet about the Mars mission " * 20,
                "published_date": "2026-10-10",
            },
            {"title": "bad", "url": "javascript:alert(1)", "content": "x"},
            {"title": "no url", "url": "", "content": "x"},
        ])

    monkeypatch.setattr(websearch, "sync_client", SimpleNamespace(post=fake_post))
    provider = TavilyProvider("tvly-test-key")
    out = provider.search("mars news", 5)

    assert captured["url"] == "https://api.tavily.com/search"
    assert captured["headers"]["Authorization"] == "Bearer tvly-test-key"
    assert captured["json"]["query"] == "mars news"
    assert len(out) == 1
    assert out[0].url == "https://example.com/mars"
    assert out[0].published == "2026-10-10"
    assert len(out[0].snippet) <= 300


def test_tavily_http_error_propagates(monkeypatch):
    def fake_post(url, **kwargs):
        class FakeResp:
            def raise_for_status(self):
                raise httpx.HTTPStatusError(
                    "401", request=httpx.Request("POST", url), response=httpx.Response(401)
                )

        return FakeResp()

    monkeypatch.setattr(websearch, "sync_client", SimpleNamespace(post=fake_post))
    with pytest.raises(httpx.HTTPStatusError):
        TavilyProvider("tvly-test-key").search("q", 5)


# ---------- maybe_search orchestration ----------

def test_maybe_search_skips_conceptual():
    outcome = maybe_search([{"role": "user", "content": "explain recursion"}], "auto")
    assert outcome.status == "skipped"


def test_maybe_search_fails_without_provider(monkeypatch):
    monkeypatch.setattr(websearch, "get_search_provider", lambda: None)
    outcome = maybe_search([{"role": "user", "content": "latest news"}], "always")
    assert outcome.status == "failed"
    assert outcome.error


def test_maybe_search_ok_with_fake_provider(monkeypatch):
    provider = _fake_provider(results=[
        SearchResult("T1", "https://example.com/1", "s1", "2026-10-01"),
        SearchResult("T2", "https://example.com/2", "s2", None),
    ])
    monkeypatch.setattr(websearch, "get_search_provider", lambda: provider)
    outcome = maybe_search(
        [{"role": "user", "content": "latest AI news today?"}], "auto"
    )
    assert outcome.status == "ok"
    assert len(outcome.results) == 2
    assert outcome.provider == "fake"
    assert outcome.retrieved


def test_maybe_search_caches_repeated_queries(monkeypatch):
    provider = _fake_provider(results=[SearchResult("T", "https://e.com", "s")])
    monkeypatch.setattr(websearch, "get_search_provider", lambda: provider)
    msgs = [{"role": "user", "content": "current gold price?"}]
    first = maybe_search(msgs, "auto")
    second = maybe_search(msgs, "auto")
    assert first.status == second.status == "ok"
    assert len(provider.calls) == 1  # second call hit the cache


def test_maybe_search_upstream_failure_is_safe(monkeypatch):
    provider = _fake_provider(exc=RuntimeError("boom secret"))
    monkeypatch.setattr(websearch, "get_search_provider", lambda: provider)
    outcome = maybe_search([{"role": "user", "content": "latest news"}], "always")
    assert outcome.status == "failed"
    assert outcome.error == SEARCH_UNAVAILABLE_MSG
    assert "secret" not in outcome.error


def test_maybe_search_never_mode_ignores_time_keywords(monkeypatch):
    provider = _fake_provider(results=[SearchResult("T", "https://e.com")])
    monkeypatch.setattr(websearch, "get_search_provider", lambda: provider)
    outcome = maybe_search([{"role": "user", "content": "latest news"}], "never")
    assert outcome.status == "skipped"
    assert provider.calls == []


def test_maybe_search_skips_when_no_user_message():
    outcome = maybe_search([{"role": "assistant", "content": "latest news"}], "auto")
    assert outcome.status == "skipped"


# ---------- prompt augmentation & sources ----------

def test_format_search_context_marks_untrusted_and_cites():
    outcome = websearch.SearchOutcome(
        status="ok",
        results=[SearchResult("Mars update", "https://example.com/m", "snippet text", "2026-10-10")],
        provider="fake",
        retrieved="2026-10-13T00:00:00Z",
    )
    block = format_search_context(outcome)
    assert "<web_search_results" in block
    assert "untrusted" in block
    assert "[1] Mars update" in block
    assert "https://example.com/m" in block
    assert "Never invent" in block


def test_augment_appends_only_to_last_user_message():
    outcome = websearch.SearchOutcome(
        status="ok",
        results=[SearchResult("T", "https://e.com/x", "s")],
        provider="fake",
        retrieved="2026-10-13T00:00:00Z",
    )
    msgs = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "latest news?"},
    ]
    out = augment_with_search(msgs, outcome)
    assert "web_search_results" not in out[0]["content"]
    assert "web_search_results" not in out[1]["content"]
    assert "web_search_results" in out[2]["content"]
    assert out[2]["content"].startswith("latest news?")
    # originals untouched (copy-on-write)
    assert "web_search_results" not in msgs[2]["content"]


def test_augment_noop_on_failed_outcome():
    msgs = [{"role": "user", "content": "hi"}]
    outcome = websearch.SearchOutcome(status="failed", error="x")
    assert augment_with_search(msgs, outcome) == msgs


def test_sources_payload_numbered_and_public():
    outcome = websearch.SearchOutcome(
        status="ok",
        results=[
            SearchResult("A", "https://e.com/a", "secret snippet", "2026-01-01"),
            SearchResult("B", "https://e.com/b", "another", None),
        ],
    )
    payload = sources_payload(outcome)
    assert [p["n"] for p in payload] == [1, 2]
    assert payload[0]["published"] == "2026-01-01"
    # snippets must NOT reach the frontend payload
    assert all("snippet" not in p for p in payload)


# ---------- /api/chat integration ----------

def test_chat_json_includes_sources(client, monkeypatch):
    provider = _fake_provider(results=[SearchResult("T", "https://e.com/1", "s", "2026-10-01")])
    monkeypatch.setattr(websearch, "get_search_provider", lambda: provider)

    with patch.object(main, "get_response", return_value={"ok": True, "reply": "answer [1]"}):
        r = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "current gold price?"}],
            "web_search": "auto",
        })

    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["sources"][0]["url"] == "https://e.com/1"
    assert body["sources"][0]["n"] == 1


def test_chat_json_search_failure_does_not_crash(client, monkeypatch):
    provider = _fake_provider(exc=RuntimeError("upstream down"))
    monkeypatch.setattr(websearch, "get_search_provider", lambda: provider)

    with patch.object(main, "get_response", return_value={"ok": True, "reply": "plain"}):
        r = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "latest news?"}],
            "web_search": "always",
        })

    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["reply"] == "plain"
    assert body["search"]["status"] == "failed"
    assert "sources" not in body


def test_chat_never_mode_no_search_call(client, monkeypatch):
    provider = _fake_provider(results=[SearchResult("T", "https://e.com")])
    monkeypatch.setattr(websearch, "get_search_provider", lambda: provider)

    with patch.object(main, "get_response", return_value={"ok": True, "reply": "ok"}):
        r = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "latest news today"}],
            "web_search": "never",
        })

    assert r.status_code == 200
    assert provider.calls == []
    assert "sources" not in r.json()


def test_chat_augmented_prompt_reaches_model(client, monkeypatch):
    provider = _fake_provider(results=[SearchResult("T", "https://e.com/1", "snippet")])
    monkeypatch.setattr(websearch, "get_search_provider", lambda: provider)
    captured = {}

    def fake_get_response(messages, model=None, system_prompt=None):
        captured["messages"] = messages
        return {"ok": True, "reply": "ok"}

    with patch.object(main, "get_response", side_effect=fake_get_response):
        client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "latest news?"}],
            "web_search": "always",
        })

    last = captured["messages"][-1]["content"]
    assert "<web_search_results" in last
    assert "untrusted" in last


# ---------- /api/chat/stream integration ----------

def test_stream_search_started_ok_done(client, monkeypatch):
    provider = _fake_provider(results=[
        SearchResult("Mars update", "https://e.com/mars", "s", "2026-10-10")
    ])
    monkeypatch.setattr(websearch, "get_search_provider", lambda: provider)
    _patch_stream(monkeypatch, [FakeStreamResp(200, _delta_lines(["Hel", "lo"]))])

    r = client.post("/api/chat/stream", json={
        "messages": [{"role": "user", "content": "latest Mars news today?"}],
        "web_search": "auto",
    })

    events = _parse_sse(r.text)
    kinds = [e[0] for e in events]
    assert kinds[0] == "search"
    assert events[0][1]["status"] == "started"
    assert kinds[1] == "search"
    assert events[1][1]["status"] == "ok"
    assert events[1][1]["sources"][0]["url"] == "https://e.com/mars"
    assert "meta" in kinds
    assert "delta" in kinds
    assert "done" in kinds
    # request_id consistency across search and meta events
    search_rid = events[0][1]["request_id"]
    meta = next(d for k, d in events if k == "meta")
    assert meta["request_id"] == search_rid


def test_stream_search_failed_still_streams(client, monkeypatch):
    provider = _fake_provider(exc=RuntimeError("boom"))
    monkeypatch.setattr(websearch, "get_search_provider", lambda: provider)
    _patch_stream(monkeypatch, [FakeStreamResp(200, _delta_lines(["ok"]))])

    r = client.post("/api/chat/stream", json={
        "messages": [{"role": "user", "content": "latest news?"}],
        "web_search": "always",
    })

    events = _parse_sse(r.text)
    kinds = [e[0] for e in events]
    search_events = [d for k, d in events if k == "search"]
    assert search_events[0]["status"] == "started"
    assert search_events[-1]["status"] == "failed"
    assert search_events[-1]["error"]
    assert "boom" not in json.dumps(events)  # internals never leak
    assert "meta" in kinds and "done" in kinds


def test_stream_conceptual_question_has_no_search_events(client, monkeypatch):
    provider = _fake_provider(results=[SearchResult("T", "https://e.com")])
    monkeypatch.setattr(websearch, "get_search_provider", lambda: provider)
    _patch_stream(monkeypatch, [FakeStreamResp(200, _delta_lines(["hi"]))])

    r = client.post("/api/chat/stream", json={
        "messages": [{"role": "user", "content": "hi"}],
    })

    events = _parse_sse(r.text)
    kinds = [e[0] for e in events]
    assert "search" not in kinds
    assert kinds[0] == "meta"
    assert provider.calls == []


def test_stream_never_mode_no_search(client, monkeypatch):
    provider = _fake_provider(results=[SearchResult("T", "https://e.com")])
    monkeypatch.setattr(websearch, "get_search_provider", lambda: provider)
    _patch_stream(monkeypatch, [FakeStreamResp(200, _delta_lines(["hi"]))])

    r = client.post("/api/chat/stream", json={
        "messages": [{"role": "user", "content": "latest news today"}],
        "web_search": "never",
    })

    events = _parse_sse(r.text)
    assert "search" not in [e[0] for e in events]
    assert provider.calls == []


# ---------- hygiene ----------

def test_no_secret_in_websearch_source():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = open(os.path.join(root, "websearch.py"), encoding="utf-8").read()
    assert "tvly-" not in text
    assert "gsk_" not in text
