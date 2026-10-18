"""Pluggable persistence for guest-device data (no login, no accounts).

Backends:
- SupabaseStore — PostgREST over HTTPS using the service-role key. Used when
  SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY are configured (production/dev).
- MemoryStore — thread-safe in-process dicts. Used when Supabase is not
  configured or ASTRA_STORE=memory (tests, offline development).

Data model (Supabase tables, text PKs — see README / SQL in repo docs):
    devices(device_id PK, memory_enabled bool, created_at)
    conversations(id PK, device_id, title, created_at, updated_at)
    messages(id PK, conversation_id FK cascade, role, content, created_at)
    memories(id PK, device_id, content, kind, status, created_at, updated_at)

Identity: the browser keeps a generated UUID in localStorage and sends it as
an X-Device-Id header. Every query is scoped by that device id — there is no
user table, no cookie and no login. Rows of other devices are never readable.

NOTE: values in this module are secrets-free by design — API keys are only
read from env, never logged or returned.
"""

import logging
import os
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import quote

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("astra.store")

_SERVICE_TIMEOUT = 10.0
_LIST_LIMIT = 200
_MSG_FETCH_LIMIT = 5000


class StoreError(Exception):
    """Persistence failure with a user-safe message (no internals leaked)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _new_id() -> str:
    return uuid.uuid4().hex


def _qs(pairs) -> str:
    """URL-encoded query string (PostgREST-safe: %, spaces, commas encoded)."""
    return "&".join(
        f"{quote(str(k), safe='')}={quote(str(v), safe='')}" for k, v in pairs
    )


# --------------------------------------------------------------------------
# In-memory backend (tests / offline fallback)
# --------------------------------------------------------------------------

class MemoryStore:
    """Thread-safe process-local store with the same contract as SupabaseStore."""

    def __init__(self):
        self._lock = threading.RLock()
        self._devices: dict[str, dict] = {}
        self._conversations: dict[str, dict] = {}
        self._messages: dict[str, list[dict]] = {}
        self._memories: dict[str, list[dict]] = {}

    # -- devices ----------------------------------------------------------
    def get_device(self, device_id: str) -> dict | None:
        with self._lock:
            dev = self._devices.get(device_id)
            return dict(dev) if dev else None

    def ensure_device(self, device_id: str) -> dict:
        with self._lock:
            dev = self._devices.get(device_id)
            if dev is None:
                dev = {"device_id": device_id, "memory_enabled": True, "created_at": _now()}
                self._devices[device_id] = dev
            return dict(dev)

    def set_memory_enabled(self, device_id: str, enabled: bool) -> dict | None:
        with self._lock:
            dev = self._devices.get(device_id)
            if dev is None:
                return None
            dev["memory_enabled"] = bool(enabled)
            return dict(dev)

    # -- conversations ----------------------------------------------------
    def create_conversation(self, device_id: str, conv_id: str, title: str) -> dict:
        with self._lock:
            row = {
                "id": conv_id,
                "device_id": device_id,
                "title": title[:200],
                "created_at": _now(),
                "updated_at": _now(),
            }
            self._conversations[conv_id] = row
            self._messages.setdefault(conv_id, [])
            return dict(row)

    def list_conversations(self, device_id: str, q: str = "") -> list[dict]:
        with self._lock:
            rows = [dict(c) for c in self._conversations.values()
                    if c["device_id"] == device_id]
        rows.sort(key=lambda c: c.get("updated_at") or "", reverse=True)
        rows = rows[:_LIST_LIMIT]
        q_norm = (q or "").strip().lower()
        out = []
        for row in rows:
            count = len(self._messages.get(row["id"], []))
            if q_norm:
                hit = q_norm in (row.get("title") or "").lower()
                if not hit:
                    hit = any(q_norm in (m.get("content") or "").lower()
                              for m in self._messages.get(row["id"], []))
                if not hit:
                    continue
            row["message_count"] = count
            out.append(row)
        return out

    def get_conversation(self, device_id: str, conv_id: str) -> dict | None:
        with self._lock:
            row = self._conversations.get(conv_id)
            if row is None or row["device_id"] != device_id:
                return None
            return dict(row)

    def rename_conversation(self, device_id: str, conv_id: str, title: str) -> dict | None:
        with self._lock:
            row = self._conversations.get(conv_id)
            if row is None or row["device_id"] != device_id:
                return None
            row["title"] = title[:200]
            row["updated_at"] = _now()
            return dict(row)

    def touch_conversation(self, conv_id: str) -> None:
        with self._lock:
            row = self._conversations.get(conv_id)
            if row is not None:
                row["updated_at"] = _now()

    def delete_conversation(self, device_id: str, conv_id: str) -> bool:
        with self._lock:
            row = self._conversations.get(conv_id)
            if row is None or row["device_id"] != device_id:
                return False
            del self._conversations[conv_id]
            self._messages.pop(conv_id, None)
            return True

    def clear_conversations(self, device_id: str) -> int:
        with self._lock:
            ids = [cid for cid, c in self._conversations.items()
                   if c["device_id"] == device_id]
            for cid in ids:
                del self._conversations[cid]
                self._messages.pop(cid, None)
            return len(ids)

    # -- messages ---------------------------------------------------------
    def add_messages(self, conv_id: str, items: list[dict]) -> None:
        with self._lock:
            bucket = self._messages.setdefault(conv_id, [])
            for item in items:
                bucket.append({
                    "id": _new_id(),
                    "conversation_id": conv_id,
                    "role": item["role"],
                    "content": item["content"],
                    "created_at": _now(),
                })

    def list_messages(self, conv_id: str) -> list[dict]:
        with self._lock:
            return [dict(m) for m in self._messages.get(conv_id, [])]

    def count_messages(self, conv_ids: list[str]) -> dict[str, int]:
        with self._lock:
            return {cid: len(self._messages.get(cid, [])) for cid in conv_ids}

    # -- memories ---------------------------------------------------------
    def add_memory(self, device_id: str, content: str, kind: str,
                   status: str = "active") -> dict:
        with self._lock:
            row = {
                "id": _new_id(),
                "device_id": device_id,
                "content": content,
                "kind": kind or "fact",
                "status": status,
                "created_at": _now(),
                "updated_at": _now(),
            }
            self._memories.setdefault(device_id, []).append(row)
            return dict(row)

    def list_memories(self, device_id: str, status: str = "") -> list[dict]:
        with self._lock:
            rows = [dict(m) for m in self._memories.get(device_id, [])]
        if status:
            rows = [m for m in rows if m["status"] == status]
        rows.sort(key=lambda m: m.get("updated_at") or "", reverse=True)
        return rows

    def get_memory(self, device_id: str, memory_id: str) -> dict | None:
        with self._lock:
            for m in self._memories.get(device_id, []):
                if m["id"] == memory_id:
                    return dict(m)
        return None

    def update_memory(self, device_id: str, memory_id: str, content: str) -> dict | None:
        with self._lock:
            for m in self._memories.get(device_id, []):
                if m["id"] == memory_id:
                    m["content"] = content
                    m["updated_at"] = _now()
                    return dict(m)
        return None

    def set_memory_status(self, device_id: str, memory_id: str, status: str) -> dict | None:
        with self._lock:
            for m in self._memories.get(device_id, []):
                if m["id"] == memory_id:
                    m["status"] = status
                    m["updated_at"] = _now()
                    return dict(m)
        return None

    def delete_memory(self, device_id: str, memory_id: str) -> bool:
        with self._lock:
            rows = self._memories.get(device_id, [])
            for i, m in enumerate(rows):
                if m["id"] == memory_id:
                    del rows[i]
                    return True
        return False

    def delete_all_memories(self, device_id: str) -> int:
        with self._lock:
            rows = self._memories.pop(device_id, [])
            return len(rows)


# --------------------------------------------------------------------------
# Supabase backend (PostgREST)
# --------------------------------------------------------------------------

class SupabaseStore:
    """PostgREST client scoped by device_id on every query (defence in depth:
    RLS is enabled on the tables with no policies, and our queries never rely
    on cross-device access)."""

    def __init__(self, url: str, service_key: str):
        self._base = url.rstrip("/") + "/rest/v1"
        self._headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(timeout=_SERVICE_TIMEOUT, headers=self._headers)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    # -- transport --------------------------------------------------------
    def _request(self, method: str, path: str, *, query: str = "",
                 body=None, prefer: str | None = None):
        url = f"{self._base}/{path}"
        if query:
            url = f"{url}?{query}"
        headers = {"Prefer": prefer} if prefer else None
        try:
            resp = self._client.request(method, url, json=body, headers=headers)
        except httpx.HTTPError as error:
            logger.warning("store network error method=%s path=%s: %s",
                           method, path, type(error).__name__)
            raise StoreError("Database unavailable. Please try again.") from error
        if resp.status_code >= 400:
            logger.warning("store upstream status=%s method=%s path=%s body=%s",
                           resp.status_code, method, path, resp.text[:200])
            if resp.status_code in (401, 403):
                raise StoreError("Database rejected the request. Check SUPABASE keys.")
            raise StoreError("Database unavailable. Please try again.")
        if not resp.content:
            return []
        try:
            data = resp.json()
        except ValueError as error:
            raise StoreError("Database returned an invalid response.") from error
        return data

    # -- devices ----------------------------------------------------------
    def get_device(self, device_id: str) -> dict | None:
        rows = self._request("GET", "devices",
                             query=_qs([("device_id", f"eq.{device_id}"), ("limit", "1")]))
        return rows[0] if rows else None

    def ensure_device(self, device_id: str) -> dict:
        existing = self.get_device(device_id)
        if existing is not None:
            return existing
        self._request(
            "POST", "devices",
            body={"device_id": device_id, "memory_enabled": True, "created_at": _now()},
            prefer="resolution=ignore-duplicates",
        )
        created = self.get_device(device_id)
        if created is None:
            raise StoreError("Could not initialise this device. Please try again.")
        return created

    def set_memory_enabled(self, device_id: str, enabled: bool) -> dict | None:
        rows = self._request(
            "PATCH", "devices",
            query=_qs([("device_id", f"eq.{device_id}")]),
            body={"memory_enabled": bool(enabled)},
            prefer="return=representation",
        )
        return rows[0] if rows else None

    # -- conversations ----------------------------------------------------
    def create_conversation(self, device_id: str, conv_id: str, title: str) -> dict:
        now = _now()
        rows = self._request(
            "POST", "conversations",
            body={"id": conv_id, "device_id": device_id, "title": title[:200],
                  "created_at": now, "updated_at": now},
            prefer="return=representation",
        )
        if not rows:
            raise StoreError("Could not create the conversation.")
        return rows[0]

    def list_conversations(self, device_id: str, q: str = "") -> list[dict]:
        rows = self._request(
            "GET", "conversations",
            query=_qs([
                ("device_id", f"eq.{device_id}"),
                ("order", "updated_at.desc"),
                ("limit", str(_LIST_LIMIT)),
            ]),
        )
        q_norm = (q or "").strip().lower()
        if q_norm and rows:
            ids = ",".join(r["id"] for r in rows)
            matches = self._request(
                "GET", "messages",
                query=_qs([
                    ("conversation_id", f"in.({ids})"),
                    ("content", f"ilike.%{q_norm}%"),
                    ("select", "conversation_id"),
                    ("limit", str(_MSG_FETCH_LIMIT)),
                ]),
            )
            hit_ids = {m["conversation_id"] for m in matches}
            rows = [r for r in rows
                    if q_norm in (r.get("title") or "").lower() or r["id"] in hit_ids]
        if rows:
            ids = ",".join(r["id"] for r in rows)
            counts = Counter(
                m["conversation_id"] for m in self._request(
                    "GET", "messages",
                    query=_qs([
                        ("conversation_id", f"in.({ids})"),
                        ("select", "conversation_id"),
                        ("limit", str(_MSG_FETCH_LIMIT)),
                    ]),
                )
            )
            for r in rows:
                r["message_count"] = counts.get(r["id"], 0)
        return rows

    def get_conversation(self, device_id: str, conv_id: str) -> dict | None:
        rows = self._request(
            "GET", "conversations",
            query=_qs([("device_id", f"eq.{device_id}"), ("id", f"eq.{conv_id}"),
                       ("limit", "1")]),
        )
        return rows[0] if rows else None

    def rename_conversation(self, device_id: str, conv_id: str, title: str) -> dict | None:
        rows = self._request(
            "PATCH", "conversations",
            query=_qs([("device_id", f"eq.{device_id}"), ("id", f"eq.{conv_id}")]),
            body={"title": title[:200], "updated_at": _now()},
            prefer="return=representation",
        )
        return rows[0] if rows else None

    def touch_conversation(self, conv_id: str) -> None:
        self._request(
            "PATCH", "conversations",
            query=_qs([("id", f"eq.{conv_id}")]),
            body={"updated_at": _now()},
        )

    def delete_conversation(self, device_id: str, conv_id: str) -> bool:
        # messages first (works with or without ON DELETE CASCADE)
        self._request("DELETE", "messages",
                      query=_qs([("conversation_id", f"eq.{conv_id}")]))
        rows = self._request(
            "DELETE", "conversations",
            query=_qs([("device_id", f"eq.{device_id}"), ("id", f"eq.{conv_id}")]),
            prefer="return=representation",
        )
        return bool(rows)

    def clear_conversations(self, device_id: str) -> int:
        convs = self._request(
            "GET", "conversations",
            query=_qs([("device_id", f"eq.{device_id}"), ("select", "id"),
                       ("limit", str(_MSG_FETCH_LIMIT))]),
        )
        ids = [c["id"] for c in convs]
        if ids:
            self._request("DELETE", "messages",
                          query=_qs([("conversation_id", f"in.({','.join(ids)})")]))
        rows = self._request(
            "DELETE", "conversations",
            query=_qs([("device_id", f"eq.{device_id}")]),
            prefer="return=representation",
        )
        return len(rows)

    # -- messages ---------------------------------------------------------
    def add_messages(self, conv_id: str, items: list[dict]) -> None:
        if not items:
            return
        now = _now()
        self._request(
            "POST", "messages",
            body=[
                {"id": _new_id(), "conversation_id": conv_id, "role": item["role"],
                 "content": item["content"], "created_at": now}
                for item in items
            ],
            prefer="return=representation",
        )

    def list_messages(self, conv_id: str) -> list[dict]:
        return self._request(
            "GET", "messages",
            query=_qs([("conversation_id", f"eq.{conv_id}"),
                       ("order", "created_at.asc,id.asc"),
                       ("limit", str(_MSG_FETCH_LIMIT))]),
        )

    def count_messages(self, conv_ids: list[str]) -> dict[str, int]:
        if not conv_ids:
            return {}
        rows = self._request(
            "GET", "messages",
            query=_qs([("conversation_id", f"in.({','.join(conv_ids)})"),
                       ("select", "conversation_id"),
                       ("limit", str(_MSG_FETCH_LIMIT))]),
        )
        counts = Counter(r["conversation_id"] for r in rows)
        return {cid: counts.get(cid, 0) for cid in conv_ids}

    # -- memories ---------------------------------------------------------
    def add_memory(self, device_id: str, content: str, kind: str,
                   status: str = "active") -> dict:
        now = _now()
        rows = self._request(
            "POST", "memories",
            body={"id": _new_id(), "device_id": device_id, "content": content,
                  "kind": kind or "fact", "status": status,
                  "created_at": now, "updated_at": now},
            prefer="return=representation",
        )
        if not rows:
            raise StoreError("Could not save the memory.")
        return rows[0]

    def list_memories(self, device_id: str, status: str = "") -> list[dict]:
        pairs = [("device_id", f"eq.{device_id}"), ("order", "updated_at.desc")]
        if status:
            pairs.append(("status", f"eq.{status}"))
        return self._request("GET", "memories", query=_qs(pairs))

    def get_memory(self, device_id: str, memory_id: str) -> dict | None:
        rows = self._request(
            "GET", "memories",
            query=_qs([("device_id", f"eq.{device_id}"), ("id", f"eq.{memory_id}"),
                       ("limit", "1")]),
        )
        return rows[0] if rows else None

    def update_memory(self, device_id: str, memory_id: str, content: str) -> dict | None:
        rows = self._request(
            "PATCH", "memories",
            query=_qs([("device_id", f"eq.{device_id}"), ("id", f"eq.{memory_id}")]),
            body={"content": content, "updated_at": _now()},
            prefer="return=representation",
        )
        return rows[0] if rows else None

    def set_memory_status(self, device_id: str, memory_id: str, status: str) -> dict | None:
        rows = self._request(
            "PATCH", "memories",
            query=_qs([("device_id", f"eq.{device_id}"), ("id", f"eq.{memory_id}")]),
            body={"status": status, "updated_at": _now()},
            prefer="return=representation",
        )
        return rows[0] if rows else None

    def delete_memory(self, device_id: str, memory_id: str) -> bool:
        rows = self._request(
            "DELETE", "memories",
            query=_qs([("device_id", f"eq.{device_id}"), ("id", f"eq.{memory_id}")]),
            prefer="return=representation",
        )
        return bool(rows)

    def delete_all_memories(self, device_id: str) -> int:
        rows = self._request(
            "DELETE", "memories",
            query=_qs([("device_id", f"eq.{device_id}")]),
            prefer="return=representation",
        )
        return len(rows)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

_store_lock = threading.Lock()
_store = None


def get_store():
    """Return the process-wide store (Supabase when configured, else memory)."""
    global _store
    if _store is not None:
        return _store
    with _store_lock:
        if _store is not None:
            return _store
        mode = (os.getenv("ASTRA_STORE") or "").strip().lower()
        url = (os.getenv("SUPABASE_URL") or "").strip()
        key = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
        if mode == "memory":
            _store = MemoryStore()
            logger.info("store backend: memory (ASTRA_STORE=memory)")
        elif url and key:
            _store = SupabaseStore(url, key)
            logger.info("store backend: supabase host=%s",
                        url.split("//", 1)[-1].split("/", 1)[0])
        else:
            _store = MemoryStore()
            logger.info("store backend: memory (SUPABASE_URL not set)")
        return _store


def set_store(store) -> None:
    """Test hook: install a specific store instance."""
    global _store
    with _store_lock:
        _store = store
