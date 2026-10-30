"""Conversation CRUD: create, rename, list, search, messages, export,
delete-one, clear-all. Every endpoint is scoped to the caller's device id
(guest mode — no accounts; see device.py / store.py).
"""

import io
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from device import DEVICE_ID_RE, get_device_id
from store import StoreError, get_store

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


class CreateIn(BaseModel):
    title: str = Field(default="New chat", max_length=200)
    id: str | None = Field(default=None, max_length=64)  # optional client id


class RenameIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)


def _conv_public(c: dict, message_count: int | None = None) -> dict:
    out = {
        "id": c.get("id"),
        "title": c.get("title"),
        "created_at": c.get("created_at"),
        "updated_at": c.get("updated_at"),
    }
    if message_count is not None:
        out["message_count"] = message_count
    return out


def _message_public(m: dict) -> dict:
    return {
        "id": m.get("id"),
        "role": m.get("role"),
        "content": m.get("content"),
        "created_at": m.get("created_at"),
    }


def get_owned_conversation(device_id: str, conv_id: str) -> dict:
    """Fetch a conversation ONLY if it belongs to this device (404 otherwise,
    so one guest never learns whether another guest's chat id exists)."""
    conv = get_store().get_conversation(device_id, conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conv


@router.post("")
def create_conversation(payload: CreateIn, device_id: str = Depends(get_device_id)):
    conv_id = (payload.id or "").strip() or uuid.uuid4().hex
    if not DEVICE_ID_RE.match(conv_id):
        raise HTTPException(status_code=400, detail="Invalid conversation id.")
    try:
        store = get_store()
        store.ensure_device(device_id)
        if store.get_conversation(device_id, conv_id) is not None:
            raise HTTPException(status_code=409, detail="Conversation already exists.")
        conv = store.create_conversation(device_id, conv_id, payload.title.strip()[:200])
    except StoreError as error:
        raise HTTPException(status_code=503, detail=str(error))
    return {"ok": True, "conversation": _conv_public(conv, 0)}


@router.get("")
def list_conversations(q: str = "", device_id: str = Depends(get_device_id)):
    """List this device's conversations (newest first), optional search.

    Search matches conversation title OR any message content (case-insensitive).
    """
    try:
        rows = get_store().list_conversations(device_id, q=q or "")
    except StoreError as error:
        raise HTTPException(status_code=503, detail=str(error))
    return {
        "ok": True,
        "conversations": [
            _conv_public(c, c.get("message_count", 0)) for c in rows
        ],
    }


@router.get("/{conv_id}")
def get_conversation(conv_id: str, device_id: str = Depends(get_device_id)):
    try:
        conv = get_owned_conversation(device_id, conv_id)
    except StoreError as error:
        raise HTTPException(status_code=503, detail=str(error))
    return {"ok": True, "conversation": _conv_public(conv)}


@router.patch("/{conv_id}")
def rename_conversation(conv_id: str, payload: RenameIn,
                        device_id: str = Depends(get_device_id)):
    try:
        conv = get_owned_conversation(device_id, conv_id)
        renamed = get_store().rename_conversation(
            device_id, conv["id"], payload.title.strip()[:200]
        )
    except StoreError as error:
        raise HTTPException(status_code=503, detail=str(error))
    return {"ok": True, "conversation": _conv_public(renamed or conv)}


@router.get("/{conv_id}/messages")
def list_messages(conv_id: str, device_id: str = Depends(get_device_id)):
    try:
        conv = get_owned_conversation(device_id, conv_id)
        msgs = get_store().list_messages(conv["id"])
    except StoreError as error:
        raise HTTPException(status_code=503, detail=str(error))
    return {"ok": True, "messages": [_message_public(m) for m in msgs]}


@router.get("/{conv_id}/export")
def export_conversation(conv_id: str, device_id: str = Depends(get_device_id)):
    """Export one conversation as Markdown text."""
    try:
        conv = get_owned_conversation(device_id, conv_id)
        msgs = get_store().list_messages(conv["id"])
    except StoreError as error:
        raise HTTPException(status_code=503, detail=str(error))
    lines = [f"# {conv.get('title') or 'Chat'}", ""]
    for m in msgs:
        who = "You" if m.get("role") == "user" else "Astra AI"
        lines.append(f"**{who}:**")
        lines.append(str(m.get("content") or ""))
        lines.append("")
    body = "\n".join(lines)
    title = conv.get("title") or "chat"
    safe = "".join(ch for ch in title if ch.isalnum() or ch in " -_").strip() or "chat"
    return Response(
        content=body,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{safe[:60]}.md"'},
    )


@router.delete("/{conv_id}")
def delete_conversation(conv_id: str, device_id: str = Depends(get_device_id)):
    try:
        get_owned_conversation(device_id, conv_id)
        deleted = get_store().delete_conversation(device_id, conv_id)
    except StoreError as error:
        raise HTTPException(status_code=503, detail=str(error))
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"ok": True}


@router.delete("")
def clear_all_history(confirm: bool = False, device_id: str = Depends(get_device_id)):
    """Clear ALL conversations + messages for this device (confirm=true)."""
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Send confirm=true to delete all conversations.",
        )
    try:
        get_store().clear_conversations(device_id)
    except StoreError as error:
        raise HTTPException(status_code=503, detail=str(error))
    return {"ok": True}
