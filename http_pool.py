"""Shared httpx clients with connection pooling.

Every upstream call used to create a fresh httpx.Client (new TCP + TLS
handshake per request). These module-level clients reuse keep-alive
connections to api.groq.com / api.tavily.com / api.openai.com.

- sync_client:   thread-safe httpx.Client for blocking endpoints (chat,
                 search, embeddings). Per-call `timeout=` still overrides.
- get_async_client(): one AsyncClient per event loop (streams), so each
                 event loop gets its own connection pool; weakly keyed so
                 closed loops are collected.
"""

import asyncio
import threading
import weakref

import httpx

_LIMITS = httpx.Limits(
    max_connections=20,
    max_keepalive_connections=10,
    keepalive_expiry=30.0,
)
_DEFAULT_TIMEOUT = httpx.Timeout(60.0)

sync_client = httpx.Client(limits=_LIMITS, timeout=_DEFAULT_TIMEOUT)

_async_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient]" = (
    weakref.WeakKeyDictionary()
)
_async_lock = threading.Lock()


def get_async_client() -> httpx.AsyncClient:
    """Return the pooled AsyncClient for the *current* event loop."""
    loop = asyncio.get_running_loop()
    with _async_lock:
        client = _async_clients.get(loop)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(limits=_LIMITS, timeout=_DEFAULT_TIMEOUT)
            _async_clients[loop] = client
        return client
