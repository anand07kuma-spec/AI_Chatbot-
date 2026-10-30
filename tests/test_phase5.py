"""Guest mode: device-scoped persistence, conversations, isolation, memory.

Identity is the X-Device-Id header (see device.py) — there is no login, so
the auth/account/CSRF suites from the account era are gone. What matters now:
CRUD works per device, one device can never see another device's rows, chat
turns persist, and memory consent (pending -> approve/reject) is honoured.
"""

import pytest
from fastapi.testclient import TestClient

DEV_A = "device-aaaa1111"
DEV_B = "device-bbbb2222"


def _as(device_id, **kw):
    return {"headers": {"X-Device-Id": device_id}, **kw}


def _create_conv(client, title="New chat", device=DEV_A):
    r = client.post("/api/conversations", json={"title": title}, headers={"X-Device-Id": device})
    assert r.status_code == 200, r.text
    return r.json()["conversation"]["id"]


def _add_memory(client, content="likes green tea", kind="preference", device=DEV_A):
    r = client.post("/api/memories", json={"content": content, "kind": kind},
                    headers={"X-Device-Id": device})
    assert r.status_code == 200, r.text
    return r.json()["memory"]["id"]


def _patch_chat(client, monkeypatch, reply="ok"):
    import main
    monkeypatch.setattr(main, "get_response",
                        lambda *a, **kw: {"ok": True, "reply": reply})


# ------------------------------------------------------------ device id ----

class TestDeviceHeader:
    def test_missing_header_400(self, client):
        r = client.get("/api/conversations")
        assert r.status_code == 400
        r = client.get("/api/memories")
        assert r.status_code == 400

    def test_invalid_header_400(self, client):
        r = client.get("/api/conversations", headers={"X-Device-Id": "x"})
        assert r.status_code == 400

    def test_chat_works_without_header_but_is_not_persisted(self, client, monkeypatch):
        _patch_chat(client, monkeypatch, "hi!")
        r = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "hi"}],
            "conversation_id": "g" + "a" * 12,
        })
        assert r.status_code == 200
        r = client.get("/api/conversations", headers={"X-Device-Id": DEV_A})
        assert r.json()["conversations"] == []


# ------------------------------------------------------- conversations ----

class TestConversations:
    def test_create_and_list(self, client):
        conv_id = _create_conv(client, "First chat")
        assert conv_id
        r = client.get("/api/conversations", headers={"X-Device-Id": DEV_A})
        assert r.status_code == 200
        assert len(r.json()["conversations"]) == 1
        assert r.json()["conversations"][0]["title"] == "First chat"
        assert r.json()["conversations"][0]["message_count"] == 0

    def test_create_with_client_supplied_id(self, client):
        conv_id = "c" + "z" * 12
        r = client.post("/api/conversations", json={"title": "mine", "id": conv_id},
                        headers={"X-Device-Id": DEV_A})
        assert r.status_code == 200
        assert r.json()["conversation"]["id"] == conv_id
        # duplicate id -> 409
        r = client.post("/api/conversations", json={"title": "again", "id": conv_id},
                        headers={"X-Device-Id": DEV_A})
        assert r.status_code == 409

    def test_create_invalid_id_400(self, client):
        r = client.post("/api/conversations", json={"title": "x", "id": "bad id!"},
                        headers={"X-Device-Id": DEV_A})
        assert r.status_code == 400

    def test_rename(self, client):
        conv_id = _create_conv(client, "Old")
        r = client.patch(f"/api/conversations/{conv_id}", json={"title": "Renamed"},
                         headers={"X-Device-Id": DEV_A})
        assert r.status_code == 200
        r = client.get(f"/api/conversations/{conv_id}", headers={"X-Device-Id": DEV_A})
        assert r.json()["conversation"]["title"] == "Renamed"

    def test_search_by_title(self, client):
        _create_conv(client, "Python tips")
        _create_conv(client, "Cooking recipes")
        r = client.get("/api/conversations", params={"q": "python"},
                       headers={"X-Device-Id": DEV_A})
        assert len(r.json()["conversations"]) == 1
        assert "Python" in r.json()["conversations"][0]["title"]

    def test_search_by_message_content(self, client, monkeypatch):
        conv_id = _create_conv(client, "Chat about something")
        _patch_chat(client, monkeypatch, "samosa recipe coming up")
        client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "how do I make samosas?"}],
            "conversation_id": conv_id,
        }, headers={"X-Device-Id": DEV_A})
        r = client.get("/api/conversations", params={"q": "samosa"},
                       headers={"X-Device-Id": DEV_A})
        assert [c["id"] for c in r.json()["conversations"]] == [conv_id]
        # message_count badge was computed
        assert r.json()["conversations"][0]["message_count"] == 2

    def test_list_messages_empty(self, client):
        conv_id = _create_conv(client, "Empty")
        r = client.get(f"/api/conversations/{conv_id}/messages",
                       headers={"X-Device-Id": DEV_A})
        assert r.status_code == 200
        assert r.json()["messages"] == []

    def test_export_markdown(self, client):
        conv_id = _create_conv(client, "Export me")
        r = client.get(f"/api/conversations/{conv_id}/export",
                       headers={"X-Device-Id": DEV_A})
        assert r.status_code == 200
        assert "text/markdown" in r.headers.get("content-type", "")
        assert "Export me" in r.text

    def test_delete_one(self, client):
        conv_id = _create_conv(client, "Bye")
        r = client.delete(f"/api/conversations/{conv_id}", headers={"X-Device-Id": DEV_A})
        assert r.status_code == 200
        r = client.get(f"/api/conversations/{conv_id}", headers={"X-Device-Id": DEV_A})
        assert r.status_code == 404

    def test_clear_all_requires_confirm(self, client):
        _create_conv(client, "A")
        r = client.delete("/api/conversations", headers={"X-Device-Id": DEV_A})
        assert r.status_code == 400
        r = client.delete("/api/conversations", params={"confirm": "true"},
                          headers={"X-Device-Id": DEV_A})
        assert r.status_code == 200
        r = client.get("/api/conversations", headers={"X-Device-Id": DEV_A})
        assert r.json()["conversations"] == []


# ------------------------------------------------- cross-device isolation ----

class TestDeviceIsolation:
    def test_cross_device_conversation_404(self, client):
        conv_id = _create_conv(client, "A private", device=DEV_A)
        for method, url in [
            ("GET", f"/api/conversations/{conv_id}"),
            ("GET", f"/api/conversations/{conv_id}/messages"),
            ("GET", f"/api/conversations/{conv_id}/export"),
            ("DELETE", f"/api/conversations/{conv_id}"),
        ]:
            r = client.request(method, url, headers={"X-Device-Id": DEV_B})
            assert r.status_code == 404, f"{method} {url} -> {r.status_code}"

    def test_cross_device_rename_404(self, client):
        conv_id = _create_conv(client, "A", device=DEV_A)
        r = client.patch(f"/api/conversations/{conv_id}", json={"title": "stolen"},
                         headers={"X-Device-Id": DEV_B})
        assert r.status_code == 404

    def test_cross_device_list_is_empty(self, client):
        _create_conv(client, "A", device=DEV_A)
        _add_memory(client, "likes tea", device=DEV_A)
        r = client.get("/api/conversations", headers={"X-Device-Id": DEV_B})
        assert r.json()["conversations"] == []
        r = client.get("/api/memories", headers={"X-Device-Id": DEV_B})
        assert r.json()["memories"] == []

    def test_cross_device_memory_404(self, client):
        mem_id = _add_memory(client, "likes tea", device=DEV_A)
        r = client.delete(f"/api/memories/{mem_id}", headers={"X-Device-Id": DEV_B})
        assert r.status_code == 404
        r = client.patch(f"/api/memories/{mem_id}", json={"content": "hacked"},
                         headers={"X-Device-Id": DEV_B})
        assert r.status_code == 404

    def test_docs_endpoint_needs_no_device(self, client):
        # RAG doc store is conversation-scoped, not identity-scoped
        r = client.post(
            "/api/docs",
            files={"file": ("t.txt", b"hello world", "text/plain")},
            data={"conversation_id": "c" + "a" * 12},
        )
        assert r.status_code == 200


# -------------------------------------------------------- persistence ----

class TestPersistence:
    def test_chat_persists_messages(self, client, monkeypatch):
        conv_id = _create_conv(client, "P")
        _patch_chat(client, monkeypatch, "hi!")
        r = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "hello"}],
            "conversation_id": conv_id,
        }, headers={"X-Device-Id": DEV_A})
        assert r.status_code == 200
        msgs = client.get(f"/api/conversations/{conv_id}/messages",
                          headers={"X-Device-Id": DEV_A}).json()["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "hi!"

    def test_chat_autocreates_conversation(self, client, monkeypatch):
        # client may generate the id locally before the server ever saw it
        _patch_chat(client, monkeypatch, "yo")
        conv_id = "n" + "b" * 12
        r = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "fresh chat"}],
            "conversation_id": conv_id,
        }, headers={"X-Device-Id": DEV_A})
        assert r.status_code == 200
        r = client.get(f"/api/conversations/{conv_id}", headers={"X-Device-Id": DEV_A})
        assert r.status_code == 200
        assert r.json()["conversation"]["title"]  # title from first message

    def test_persist_survives_reconnect(self, client, monkeypatch):
        import main
        conv_id = _create_conv(client, "S")
        _patch_chat(client, monkeypatch, "ok")
        client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "ping"}],
            "conversation_id": conv_id,
        }, headers={"X-Device-Id": DEV_A})
        # new client instance = fresh cookie-less jar; same device header reads it
        with TestClient(main.app) as c2:
            msgs = c2.get(f"/api/conversations/{conv_id}/messages",
                          headers={"X-Device-Id": DEV_A}).json()["messages"]
            assert len(msgs) == 2
            # a different device sees nothing
            r = c2.get(f"/api/conversations/{conv_id}", headers={"X-Device-Id": DEV_B})
            assert r.status_code == 404


# ------------------------------------------------------------- memory ----

class TestMemory:
    def _make_memory(self, client):
        return _add_memory(client, "User likes green tea")

    def test_add_and_list(self, client):
        self._make_memory(client)
        r = client.get("/api/memories", headers={"X-Device-Id": DEV_A})
        assert r.status_code == 200
        mems = r.json()["memories"]
        assert len(mems) == 1
        assert mems[0]["status"] == "active"

    def test_toggle_off_stops_injection(self, client, monkeypatch):
        captured = {}
        import main
        orig = main.augment_with_memories

        def spy(messages, mems):
            captured["hits"] = mems
            return orig(messages, mems)

        monkeypatch.setattr(main, "augment_with_memories", spy)
        _patch_chat(client, monkeypatch, "ok")
        _add_memory(client, "likes green tea")
        r = client.patch("/api/memories/preferences", json={"memory_enabled": False},
                         headers={"X-Device-Id": DEV_A})
        assert r.status_code == 200
        captured.pop("hits", None)
        client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "what tea should I drink?"}],
            "conversation_id": "m" + "a" * 12,
        }, headers={"X-Device-Id": DEV_A})
        assert "hits" not in captured  # no injection when disabled

    def test_injection_when_enabled(self, client, monkeypatch):
        captured = {}
        import main
        orig = main.augment_with_memories

        def spy(messages, mems):
            captured["hits"] = mems
            return orig(messages, mems)

        monkeypatch.setattr(main, "augment_with_memories", spy)
        _patch_chat(client, monkeypatch, "ok")
        _add_memory(client, "likes green tea")
        client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "should I drink green tea?"}],
            "conversation_id": "n" + "a" * 12,
        }, headers={"X-Device-Id": DEV_A})
        assert "hits" in captured  # injection happens when enabled

    def test_suggestions_pending_and_approve(self, client):
        r = client.post("/api/memories/suggest", json={"content": "Hi, my name is Anand"},
                        headers={"X-Device-Id": DEV_A})
        assert r.status_code == 200
        assert r.json()["created"] >= 1
        r = client.get("/api/memories", params={"status": "pending"},
                       headers={"X-Device-Id": DEV_A})
        pending = r.json()["memories"]
        assert len(pending) >= 1
        mem_id = pending[0]["id"]
        r = client.post(f"/api/memories/{mem_id}/approve", headers={"X-Device-Id": DEV_A})
        assert r.status_code == 200
        r = client.get("/api/memories", params={"status": "active"},
                       headers={"X-Device-Id": DEV_A})
        assert any(m["id"] == mem_id for m in r.json()["memories"])

    def test_suggest_reject_deletes(self, client):
        r = client.post("/api/memories/suggest", json={"content": "I live in Delhi"},
                        headers={"X-Device-Id": DEV_A})
        assert r.status_code == 200
        r = client.get("/api/memories", params={"status": "pending"},
                       headers={"X-Device-Id": DEV_A})
        pending = r.json()["memories"]
        assert pending
        mem_id = pending[0]["id"]
        r = client.post(f"/api/memories/{mem_id}/reject", headers={"X-Device-Id": DEV_A})
        assert r.status_code == 200
        r = client.get("/api/memories", headers={"X-Device-Id": DEV_A})
        assert all(m["id"] != mem_id for m in r.json()["memories"])

    def test_pending_never_injected(self, client, monkeypatch):
        # pending suggestions must NOT reach the prompt until approved
        captured = {}
        import main
        orig = main.augment_with_memories

        def spy(messages, mems):
            captured["hits"] = mems
            return orig(messages, mems)

        monkeypatch.setattr(main, "augment_with_memories", spy)
        _patch_chat(client, monkeypatch, "ok")
        r = client.post("/api/memories/suggest",
                        json={"content": "my name is Anand Kumar"},
                        headers={"X-Device-Id": DEV_A})
        assert r.json()["created"] >= 1
        captured.pop("hits", None)
        client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "what is my name?"}],
            "conversation_id": "p" + "a" * 12,
        }, headers={"X-Device-Id": DEV_A})
        # injection may run but must not include the pending memory
        if "hits" in captured:
            assert all(m.get("status") == "active" for m in captured["hits"])

    def test_delete_all_memories(self, client):
        _add_memory(client, "fact one")
        _add_memory(client, "fact two")
        r = client.delete("/api/memories", params={"confirm": "true"},
                          headers={"X-Device-Id": DEV_A})
        assert r.status_code == 200
        r = client.get("/api/memories", headers={"X-Device-Id": DEV_A})
        assert r.json()["memories"] == []

    def test_preferences_toggle(self, client):
        r = client.get("/api/memories/preferences", headers={"X-Device-Id": DEV_A})
        assert r.json()["memory_enabled"] is True
        r = client.patch("/api/memories/preferences", json={"memory_enabled": False},
                         headers={"X-Device-Id": DEV_A})
        assert r.status_code == 200
        assert r.json()["memory_enabled"] is False
        r = client.get("/api/memories/preferences", headers={"X-Device-Id": DEV_A})
        assert r.json()["memory_enabled"] is False
