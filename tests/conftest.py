import base64
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")
os.environ["RATE_LIMIT_PER_MINUTE"] = "0"  # disable rate limiting for most tests
# Guest mode: never touch a real Supabase from the test suite — the in-memory
# store backend has the same contract and is reset per test.
os.environ["ASTRA_STORE"] = "memory"
# RAG: force cheap local embeddings so tests never hit OpenAI even if a real
# key is present in the developer's environment.
os.environ["ASTRA_EMBEDDINGS_PROVIDER"] = "hash"

import main  # noqa: E402
import rag  # noqa: E402
import store  # noqa: E402


@pytest.fixture()
def client():
    store.set_store(store.MemoryStore())  # fresh data for every test
    main._index_html_cache = None
    main._rate_limiter.reset()
    main._tts_cache.clear()
    rag.get_doc_store().clear()
    with TestClient(main.app) as c:
        yield c
    main._rate_limiter.reset()
    rag.get_doc_store().clear()


@pytest.fixture()
def tiny_png_b64():
    """1x1 transparent PNG as base64."""
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
        "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    return base64.b64encode(png).decode()
