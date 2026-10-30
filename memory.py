"""Long-term device-controlled memory.

Separation of concerns:
- Short-term context = the rolling message window already sent to the model.
- Long-term memory   = Memory rows in the store, scoped to the device.

Consent model:
- Nothing is saved silently.
- After each assistant reply, a lightweight heuristic suggests personal
  facts from the LAST USER message as status="pending" (never injected).
- The UI shows pending suggestions; the user approves (-> active) or
  rejects (-> deleted). The user can also add/edit/delete memories
  manually at any time.
- Injection only happens when the device has memory_enabled=True AND the
  memory status is "active" AND it is relevant to the current query.

We deliberately do NOT use embeddings for memory relevance (keeps the
dependency surface small); a simple token-overlap score with a threshold
is enough for a personal memory list (typically < 100 entries).
"""

import re

from store import StoreError, get_store

# conservative patterns — high precision over recall; false positives are
# annoying, misses are fine because users can add memories manually.
_SUGGESTION_PATTERNS: list[tuple[str, str, re.Pattern]] = [
    ("fact", "name", re.compile(
        r"\b(?:my name is|i am called|call me|i'm)\s+([A-Z][a-zA-Z]{1,30}(?:\s[A-Z][a-zA-Z]{1,30})?)\b")),
    ("preference", "likes", re.compile(
        r"\bi (?:really )?(?:like|love|enjoy)\s+([a-z0-9][a-z0-9\s\-]{2,50})", re.I)),
    ("preference", "dislikes", re.compile(
        r"\bi (?:really )?(?:dislike|hate|don't like)\s+([a-z0-9][a-z0-9\s\-]{2,50})", re.I)),
    ("fact", "location", re.compile(
        r"\bi (?:live|am based|am from)\s+in\s+([A-Z][a-zA-Z\s]{1,40})", re.I)),
    ("fact", "works_in", re.compile(
        r"\bi work(?:ing)?\s+(?:in|at|as)\s+([a-zA-Z][a-zA-Z\s\-]{2,50})", re.I)),
]

_MAX_SUGGESTIONS_PER_MESSAGE = 2
_MIN_MEMORY_SCORE = 0.25
_TOKEN_RE = re.compile(r"[a-z0-9\u0900-\u097f]+")


def suggest_memories(user_text: str) -> list[dict]:
    """Heuristic extraction of candidate personal facts (needs consent)."""
    text = (user_text or "").strip()
    if not text or len(text) < 8:
        return []
    seen: set[str] = set()
    out: list[dict] = []
    for kind, label, pattern in _SUGGESTION_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        value = m.group(1).strip(" .,!?;:\"'")
        if len(value) < 3 or value.lower() in seen:
            continue
        seen.add(value.lower())
        content = {
            "name": f"The user's name is {value}.",
            "likes": f"The user likes {value}.",
            "dislikes": f"The user does not like {value}.",
            "location": f"The user lives in {value}.",
            "works_in": f"The user works in {value}.",
        }.get(label, f"The user mentioned: {value}.")
        out.append({"content": content, "kind": kind})
        if len(out) >= _MAX_SUGGESTIONS_PER_MESSAGE:
            break
    return out


def save_pending_suggestions(device_id: str, user_text: str) -> list[dict]:
    """Insert pending suggestions for this device (deduped against existing).

    Returns the pending rows that were created (empty list when there is
    nothing to do) so callers can either count them or ship them to the UI —
    the SSE `memory` event needs the actual rows, not a count. Never raises
    StoreError for 'nothing to do'; real store failures propagate as StoreError.
    """
    store = get_store()
    dev = store.ensure_device(device_id)
    if not dev.get("memory_enabled", True):
        return []
    suggestions = suggest_memories(user_text)
    if not suggestions:
        return []
    existing_norm = {
        c.strip().lower() for c in
        (m.get("content") or "" for m in store.list_memories(device_id))
    }
    created: list[dict] = []
    for s in suggestions:
        if s["content"].strip().lower() in existing_norm:
            continue
        created.append(
            store.add_memory(device_id, s["content"], s["kind"], status="pending")
        )
        existing_norm.add(s["content"].strip().lower())
    return created


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def score_memory(query: str, memory_content: str) -> float:
    q = _tokenize(query)
    m = _tokenize(memory_content)
    if not q or not m:
        return 0.0
    return len(q & m) / len(q)


def get_relevant_memories(device_id: str, query: str, limit: int = 5) -> list[dict]:
    """Top-k ACTIVE memories relevant to the query. Empty if memory disabled."""
    store = get_store()
    dev = store.get_device(device_id)
    if dev is None or not dev.get("memory_enabled", True):
        return []
    rows = store.list_memories(device_id, status="active")
    if not rows:
        return []
    scored = [(score_memory(query, m.get("content") or ""), m) for m in rows]
    scored = [x for x in scored if x[0] >= _MIN_MEMORY_SCORE]
    scored.sort(key=lambda t: t[0], reverse=True)
    return [m for _, m in scored[:limit]]


def format_memory_context(memories: list) -> str:
    """Delimited block for the model prompt (untrusted-data style)."""
    if not memories:
        return ""
    lines = ["<user_memory>"]
    lines.append(
        "The user previously chose to share these personal facts. Use them "
        "only when relevant to the current question; never invent additional "
        "personal details and never mention this block explicitly."
    )
    for m in memories:
        lines.append(f"- {m.get('content') if isinstance(m, dict) else m.content}")
    lines.append("</user_memory>")
    return "\n".join(lines)


def augment_with_memories(messages: list, memories: list,
                          content_limit: int = 9000) -> list:
    """Copy-on-write: prepend memory block to the last user message."""
    block = format_memory_context(memories)
    if not block or not messages:
        return messages
    messages = [dict(m) for m in messages]
    last = messages[-1]
    last["content"] = (block + "\n\n" + str(last["content"]))[:content_limit]
    return messages
