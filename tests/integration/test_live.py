"""Integration tests — REAL API calls to Groq / Tavily.

Skipped by default so the unit suite stays offline and fast.
Enable with:   ASTRA_INTEGRATION=1 python -m pytest tests/integration -m integration -v

Requires a real GROQ_API_KEY (TAVILY_API_KEY optional — search tests skip
without it). These tests exercise the live provider contract that mocks
cannot verify (auth, model ids, SSE shape, rate-limit behaviour).
"""

import os

import pytest

pytestmark = pytest.mark.integration

API_KEY = os.getenv("GROQ_API_KEY", "")


def _has_real_key() -> bool:
    return bool(API_KEY) and API_KEY != "test-key-not-real"


needs_groq = pytest.mark.skipif(
    not _has_real_key(), reason="GROQ_API_KEY not set to a real key"
)
needs_tavily = pytest.mark.skipif(
    not os.getenv("TAVILY_API_KEY"), reason="TAVILY_API_KEY not set"
)


@needs_groq
def test_live_route_chat_roundtrip():
    import router

    result = router.route_chat(
        [{"role": "user", "content": "Reply with the single word: pong"}],
        API_KEY,
        "You are a test harness. Reply exactly as instructed.",
    )
    assert result["ok"] is True
    assert result["reply"].strip()
    assert result["model"]


@needs_groq
def test_live_stream_first_chunk_arrives():
    import asyncio

    import router

    async def run():
        events = []
        async for ev in router.stream_chat(
            [{"role": "user", "content": "Say hi"}],
            API_KEY,
            "You are a test harness.",
        ):
            events.append(ev)
        return events

    events = asyncio.run(run())
    types = [e["type"] for e in events]
    assert types[0] == "meta"
    assert "delta" in types
    assert types[-1] == "done"


@needs_groq
def test_live_models_resolve():
    import router

    for task in ("fast", "reasoning"):
        model = router.resolve_text_model(task)
        assert router.is_valid_model_choice(model)


@needs_tavily
def test_live_tavily_search_returns_results():
    from websearch import TavilyProvider

    provider = TavilyProvider(os.environ["TAVILY_API_KEY"])
    results = provider.search("latest news about python", 3)
    assert results
    for r in results:
        assert r.url.startswith(("http://", "https://"))
