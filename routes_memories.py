"""Memory management for this device: list, add, edit, approve/reject
suggestions, delete, delete-all, and the memory_enabled preference toggle."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from device import get_device_id
from memory import save_pending_suggestions
from store import StoreError, get_store

router = APIRouter(prefix="/api/memories", tags=["memories"])


class AddIn(BaseModel):
    content: str = Field(min_length=2, max_length=2000)
    kind: str = Field(default="fact", max_length=40)


class EditIn(BaseModel):
    content: str = Field(min_length=2, max_length=2000)


class PrefIn(BaseModel):
    memory_enabled: bool


def _public(m: dict) -> dict:
    return {
        "id": m.get("id"),
        "content": m.get("content"),
        "kind": m.get("kind"),
        "status": m.get("status"),
        "created_at": m.get("created_at"),
        "updated_at": m.get("updated_at"),
    }


def _get_owned(device_id: str, memory_id: str) -> dict:
    m = get_store().get_memory(device_id, memory_id)
    if m is None:
        raise HTTPException(status_code=404, detail="Memory not found.")
    return m


@router.get("")
def list_memories(status: str = "", device_id: str = Depends(get_device_id)):
    try:
        store = get_store()
        store.ensure_device(device_id)
        rows = store.list_memories(
            device_id, status=status if status in ("active", "pending") else ""
        )
    except StoreError as error:
        raise HTTPException(status_code=503, detail=str(error))
    return {"ok": True, "memories": [_public(m) for m in rows]}


# NOTE: /preferences must be registered BEFORE /{memory_id} so FastAPI
# doesn't try to parse "preferences" as a path param.
@router.get("/preferences")
def get_preferences(device_id: str = Depends(get_device_id)):
    """Current memory preference for this device."""
    try:
        dev = get_store().ensure_device(device_id)
    except StoreError as error:
        raise HTTPException(status_code=503, detail=str(error))
    return {"ok": True, "memory_enabled": bool(dev.get("memory_enabled", True))}


@router.patch("/preferences")
def set_preferences(payload: PrefIn, device_id: str = Depends(get_device_id)):
    """Enable/disable long-term memory injection."""
    try:
        store = get_store()
        store.ensure_device(device_id)
        dev = store.set_memory_enabled(device_id, payload.memory_enabled)
    except StoreError as error:
        raise HTTPException(status_code=503, detail=str(error))
    enabled = bool(dev.get("memory_enabled", True)) if dev else payload.memory_enabled
    return {"ok": True, "memory_enabled": enabled}


@router.post("")
def add_memory(payload: AddIn, device_id: str = Depends(get_device_id)):
    """Manual add — explicit consent by definition."""
    try:
        store = get_store()
        store.ensure_device(device_id)
        m = store.add_memory(device_id, payload.content.strip(),
                             payload.kind.strip() or "fact", status="active")
    except StoreError as error:
        raise HTTPException(status_code=503, detail=str(error))
    return {"ok": True, "memory": _public(m)}


@router.post("/suggest")
def trigger_suggestions(payload: AddIn, device_id: str = Depends(get_device_id)):
    """Run the heuristic extractor on provided user text (creates pending rows)."""
    try:
        created = save_pending_suggestions(device_id, payload.content)
    except StoreError as error:
        raise HTTPException(status_code=503, detail=str(error))
    return {"ok": True, "created": created}


@router.patch("/{memory_id}")
def edit_memory(memory_id: str, payload: EditIn,
                device_id: str = Depends(get_device_id)):
    try:
        _get_owned(device_id, memory_id)
        m = get_store().update_memory(device_id, memory_id, payload.content.strip())
    except StoreError as error:
        raise HTTPException(status_code=503, detail=str(error))
    if m is None:
        raise HTTPException(status_code=404, detail="Memory not found.")
    return {"ok": True, "memory": _public(m)}


@router.post("/{memory_id}/approve")
def approve_memory(memory_id: str, device_id: str = Depends(get_device_id)):
    """Consent: pending suggestion becomes active (eligible for injection)."""
    try:
        m = _get_owned(device_id, memory_id)
        if m.get("status") != "pending":
            raise HTTPException(status_code=400, detail="Memory is not pending.")
        m = get_store().set_memory_status(device_id, memory_id, "active")
    except StoreError as error:
        raise HTTPException(status_code=503, detail=str(error))
    return {"ok": True, "memory": _public(m)}


@router.post("/{memory_id}/reject")
def reject_memory(memory_id: str, device_id: str = Depends(get_device_id)):
    """Consent: pending suggestion is discarded."""
    try:
        _get_owned(device_id, memory_id)
        get_store().delete_memory(device_id, memory_id)
    except StoreError as error:
        raise HTTPException(status_code=503, detail=str(error))
    return {"ok": True}


@router.delete("/{memory_id}")
def delete_memory(memory_id: str, device_id: str = Depends(get_device_id)):
    try:
        _get_owned(device_id, memory_id)
        get_store().delete_memory(device_id, memory_id)
    except StoreError as error:
        raise HTTPException(status_code=503, detail=str(error))
    return {"ok": True}


@router.delete("")
def delete_all_memories(confirm: bool = False, device_id: str = Depends(get_device_id)):
    if not confirm:
        raise HTTPException(status_code=400, detail="Send confirm=true to delete all memories.")
    try:
        get_store().delete_all_memories(device_id)
    except StoreError as error:
        raise HTTPException(status_code=503, detail=str(error))
    return {"ok": True}
