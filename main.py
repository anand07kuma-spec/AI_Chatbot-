from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chatbot import get_response

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="NOVA AI Chatbot")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    messages: list[dict]


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/chat")
def chat(request: ChatRequest):
    messages = [
        {
            "role": m.get("role", "user"),
            "content": str(m.get("content", "")),
        }
        for m in request.messages
        if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()
    ]

    if not messages:
        return JSONResponse(
            {"ok": False, "reply": "Please enter a message."},
            status_code=400,
        )

    if messages[-1]["role"] != "user":
        return JSONResponse(
            {"ok": False, "reply": "Last message must be from the user."},
            status_code=400,
        )

    result = get_response(messages)
    return {"ok": result["ok"], "reply": result["reply"]}


@app.get("/health")
def health():
    return {"status": "ok"}
