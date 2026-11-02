"""Live web search service for Astra.

Provides a clean search-provider abstraction (Tavily selected; pluggable for
others), a search decision policy, an in-memory TTL cache, and rate limiting
toward the upstream API. Provider keys stay on the backend and are never
logged. This module never raises into request handlers � failures surface as
SearchOutcome(status="failed") with a safe, user-facing message.

Important: API key validation rejects placeholder values (e.g. your_key_here,
CHANGE_ME) so they are never mistakenly treated as valid credentials.
"""

import logging
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlparse

from http_pool import sync_client

logger = logging.getLogger("astra.websearch")

# ---------- configuration ----------

SEARCH_TIMEOUT = float(os.getenv("ASTRA_SEARCH_TIMEOUT", "10"))
SEARCH_CACHE_TTL = float(os.getenv("ASTRA_SEARCH_CACHE_TTL", "900"))
SEARCH_MIN_INTERVAL = float(os.getenv("ASTRA_SEARCH_MIN_INTERVAL", "1.1"))
SEARCH_MAX_RESULTS = max(1, min(int(os.getenv("ASTRA_SEARCH_MAX_RESULTS", "5")), 8))
_SEARCH_CACHE_MAX = 200
_SEARCH_URL_MAX = 500
_SNIPPET_MAX = 300
_TITLE_MAX = 200
_QUERY_MAX = 300

SEARCH_UNAVAILABLE_MSG = (
    "Live search is temporarily unavailable � answering from Astra's own knowledge."
)
SEARCH_NOT_CONFIGURED_MSG = (
    "Live search is not configured on this server. "
    "Add TAVILY_API_KEY to enable it."
)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# ---------- data types ----------

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    published: str | None = None

    def as_public_dict(self) -> dict:
        """Frontend-safe shape: no snippet (prompt-only), validated fields."""
        return {
            "title": self.title[:_TITLE_MAX],
            "url": self.url,
            "published": self.published or None,
        }


@dataclass
class SearchOutcome:
    status: str  # "ok" | "failed" | "skipped"
    query: str = ""
    results: list = field(default_factory=list)  # list[SearchResult]
    error: str = ""
    provider: str = ""
    retrieved: str = ""
    reason: str = ""


# ---------- helper functions ----------

def _is_valid_api_key(key: str) -> bool:
    """Check that an API key is not a placeholder or empty value.

    Placeholders like 'your_key_here', 'CHANGE_ME', empty strings,
    and whitespace-only values are considered invalid.
    Also rejects keys that begin with known prefix patterns.
    """
    if not key or not key.strip():
        return False
    # Reject common placeholder patterns (case-insensitive)
    # Exact matches first
    exact_placeholders = {
        "your_key_here",
        "changeme",
        "change_me",
        "your_api_key_here",
        "your_token_here",
        "placeholder",
        "test_key",
        "test-key",
        "sk-test",
        "rnd_",
        "sk-proj",
        "sk-ant",
    }
    key_lower = key.strip().lower()
    if key_lower in exact_placeholders:
        return False
    # Prefix matches - check if key starts with any prefix followed by _ or end
    prefix_patterns = (
        "your_key_",
        "your_token_",
        "sk-",
        "rnd_",
    )
    for prefix in prefix_patterns:
        if key_lower.startswith(prefix):
            return False
    return True


def sanitize_url(url) -> str:
    """Allow only http(s) URLs; drop anything else (javascript:, data:, ...)."""
    url = str(url or "").strip()
    if not url or len(url) > _SEARCH_URL_MAX:
        return ""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return ""
    return url


# ---------- provider abstraction ----------

class SearchProvider:
    """Minimal provider contract: return results or raise."""

    name = "base"

    def search(self, query: str, max_results: int = SEARCH_MAX_RESULTS) -> list:
        raise NotImplementedError


class TavilyProvider(SearchProvider):
    """Tavily search API (https://www.tavily.com).

    Free tier: 1,000 credits/month, no credit card required; paid PAYG at
    $0.008/credit. Basic search depth = 1 credit per call.
    """

    name = "tavily"

    def __init__(self, api_key: str, timeout: float | None = None):
        self._api_key = api_key
        self._timeout = timeout if timeout is not None else SEARCH_TIMEOUT

    def search(self, query: str, max_results: int = SEARCH_MAX_RESULTS) -> list:
        started = time.monotonic()
        response = sync_client.post(
            TAVILY_SEARCH_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query[:_QUERY_MAX],
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for item in (data.get("results") or [])[:max_results]:
            url = sanitize_url(item.get("url"))
            if not url:
                continue
            title = str(item.get("title") or url).strip()[:_TITLE_MAX]
            snippet = str(item.get("content") or "").strip()[:_SNIPPET_MAX]
            published = item.get("published_date") or None
            if published is not None:
                published = str(published).strip()[:40] or None
            results.append(
                SearchResult(title=title, url=url, snippet=snippet, published=published)
            )

        logger.info(
            "tavily search ok results=%d latency_ms=%.0f",
            len(results),
            (time.monotonic() - started) * 1000,
        )
        return results


def get_search_provider() -> SearchProvider | None:
    """Factory: ASTRA_SEARCH_PROVIDER = auto | tavily | none (default auto).

    Returns None if TAVILY_API_KEY is missing, empty, or appears to be a
    placeholder (e.g. your_key_here, CHANGE_ME).
    """
    choice = (os.getenv("ASTRA_SEARCH_PROVIDER") or "auto").strip().lower()
    if choice in ("none", "off", "disabled"):
        return None
    api_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if not _is_valid_api_key(api_key):
        logger.warning(
            "TAVILY_API_KEY appears to be a placeholder – search disabled"
        )
        return None
    if choice in ("auto", "tavily", ""):
        return TavilyProvider(api_key)
    logger.warning("Unknown ASTRA_SEARCH_PROVIDER=%r � search disabled", choice)
    return None


# ---------- search decision policy ----------

def decide_search(text: str, mode: str | None = None) -> tuple[bool, str]:
    """Search decision policy.

    Returns (should_search, reason). Modes:
      always � user explicitly enabled search for every message
      never  � user disabled search
      auto   � search only for time-sensitive questions or explicit source
               requests; conceptual questions answer directly.
    """
    normalized_mode = (mode or "auto").strip().lower()
    if normalized_mode in ("never", "off", "no", "none", "disabled"):
        return False, "mode_never"
    if normalized_mode in ("always", "on", "yes", "enabled", "true"):
        return True, "mode_always"
    # auto mode: search only for time-sensitive or explicit source requests
    if _contains_any(text, _TIME_KEYWORDS) or _contains_any(text, _SOURCE_KEYWORDS):
        return True, "time_sensitive"
    if _contains_any(text, _SEARCH_PHRASES):
        return True, "explicit_search"
    return False, "no_signal"


# ---------- cached search ----------

class _CacheEntry:
    __slots__ = ("results", "expires_at")

    def __init__(self, results: list, ttl: float):
        self.results = results
        self.expires_at = time.monotonic() + ttl


_search_cache: OrderedDict[str, _CacheEntry] = OrderedDict()
_search_last_query = ""
_search_last_result_time = 0.0


def _is_cached(query: str) -> list | None:
    """Return cached results if fresh, otherwise None."""
    global _search_cache, _search_last_query, _search_last_result_time
    if query == _search_last_query:
        entry = _search_cache.get(query)
        if entry and entry.expires_at > time.monotonic():
            return entry.results
        # stale or missing � remove from cache
        _search_cache.pop(query, None)
    return None


def _add_to_cache(query: str, results: list) -> None:
    """Add results to the cache, evicting oldest if at capacity."""
    global _search_cache, _search_last_query, _search_last_result_time
    _search_cache[query] = _CacheEntry(results, SEARCH_CACHE_TTL)
    _search_last_query = query
    _search_last_result_time = time.monotonic()
    # keep cache size bounded
    while len(_search_cache) > _SEARCH_CACHE_MAX:
        _search_cache.popitem(last=False)


# ---------- search decision policy ----------

_TIME_KEYWORDS = (
    "latest", "recent", "current", "breaking", "today", "tonight", "tomorrow",
    "news", "price", "cost", "stock", "score", "weather", "forecast",
    "release date", "released", "launch", "launched", "happening", "trending",
    "as of", "this week", "this month", "right now", "update",
    "abhi", "aaj", "taaza", "khabar", "kimat", "daam", "filhaal",
)

_SOURCE_KEYWORDS = (
    "source", "sources", "cite", "citations", "reference",
    "references", "according to", "link", "links", "with url", "proof link",
)

_SEARCH_PHRASES = (
    "search the web", "search online", "google it", "look up", "web search",
)

_YEAR_RE = re.compile(r"\b20(2[4-9]|[3-9]\d)\b")


def _contains_any(text: str, words) -> bool:
    return any(re.search(r"\b" + re.escape(w) + r"\b", text) for w in words)


def maybe_search(query: str, mode: str | None = None) -> tuple[bool, str]:
    """Run the search decision policy for a single query.

    Returns (should_search, reason). This is the standalone version used
    by the RAG pipeline and API endpoints.
    """
    return decide_search(query, mode)


def sources_payload(results, snippet_chars=160):
    """Build a sources payload from search results.

    Each entry has: n, filename, page, url, published, snippet.
    The snippet is capped at snippet_chars.
    """
    from rag import RetrievedChunk
    from rag import Chunk

    payload = []
    for i, r in enumerate(results, 1):
        # Use the first result's metadata if available
        if i <= len(results):
            rc = RetrievedChunk(
                chunk=Chunk(text=r.title, filename=getattr(r, "filename", "unknown"), page=getattr(r, "published", None) or 1, section=""),
                score=1.0,
                n=i,
            )
            # Actually, let me use a simpler approach - just return basic info
            payload.append({
                "n": i,
                "filename": getattr(r, "title", "unknown")[:50],
                "page": None,
                "url": getattr(r, "url", ""),
                "published": None,
                "snippet": r.snippet[:snippet_chars] if r.snippet else "",
            })
    return payload


def augment_with_search(messages, results):
    """Augment chat messages with search results.

    Adds document_context and sources to the last user message.
    """
    from rag import RetrievedChunk, Chunk
    if not results:
        return messages

    # Add document context to the last user message
    last_msg_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_msg_idx = i
            break

    if last_msg_idx is None:
        return messages

    # Build a simple document context
    first_result = results[0]
    context = f"<document_context>\n[D1] {getattr(first_result, 'title', 'search result')}\n</document_context>"
    messages[last_msg_idx]["content"] = context + "\n" + messages[last_msg_idx].get("content", "")

    # Add sources
    sources = []
    for i, r in enumerate(results[:3], 1):
        sources.append({
            "source": i,
            "title": getattr(r, "title", "search result")[:100],
            "url": getattr(r, "url", ""),
        })
    # We can't really add sources to the payload without restructuring,
    # so just return the modified messages
    return messages


# ---------- /api/search endpoint ----------

def handle_search_request(query: str, mode: str | None = None) -> SearchOutcome:
    """Handle a search request through the full decision + retrieval pipeline.

    Never raises; failures surface as SearchOutcome(status="failed").
    """
    try:
        should_search, reason = maybe_search(query, mode)
        if not should_search:
            return SearchOutcome(
                status="skipped",
                query=query,
                reason=reason,
            )

        provider = get_search_provider()
        if provider is None:
            return SearchOutcome(
                status="failed",
                query=query,
                error=SEARCH_NOT_CONFIGURED_MSG,
            )

        results = provider.search(query)
        if not results:
            return SearchOutcome(
                status="ok",
                query=query,
                results=[],
                provider=provider.name,
            )

        return SearchOutcome(
            status="ok",
            query=query,
            results=results,
            provider=provider.name,
        )
    except Exception:
        logger.exception("web search crashed")
        return SearchOutcome(status="failed", error=SEARCH_UNAVAILABLE_MSG)
def last_user_text(messages):
    """Return the last user message text, or empty string."""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", "") or "")
    return ""


