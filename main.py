import asyncio
import io
import re
from pathlib import Path

import edge_tts
import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chatbot import get_api_key, get_response

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Astra AI Chatbot")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
WHISPER_MODEL = "whisper-large-v3-turbo"
WHISPER_PROMPT = (
    "Python, code, program, function, leap year, late, please, "
    "Hinglish, Hindi and English mix. Technical words in English."
)
VOICE_EN = "en-IN-NeerjaNeural"
VOICE_HI = "hi-IN-SwaraNeural"
VOICE_UR = "ur-IN-GulNeural"
TTS_FALLBACK = "Please check the code shared in the chat."
TTS_MAX_CHARS = 1500


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


class TTSRequest(BaseModel):
    text: str


def pick_voice(text):
    if any("ऀ" <= ch <= "ॿ" for ch in text):
        return VOICE_HI
    if any("؀" <= ch <= "ۿ" for ch in text):
        return VOICE_UR
    return VOICE_EN


def strip_markdown(text):
    """Convert markdown to plain spoken text (no symbols, no code)."""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*\*([^*]+)\*\*\*", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"\|", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.replace("*", "").replace("#", "").replace("`", "")
    return text.strip()


async def _edge_tts(text, voice):
    communicate = edge_tts.Communicate(text, voice)
    buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buffer.write(chunk["data"])
    return buffer.getvalue()


@app.post("/api/tts")
def tts(request: TTSRequest):
    spoken = strip_markdown(request.text.strip())
    if not spoken:
        spoken = TTS_FALLBACK

    spoken = spoken[:TTS_MAX_CHARS]
    voice = pick_voice(spoken)

    try:
        audio = asyncio.run(_edge_tts(spoken, voice))
    except Exception as error:
        print(f"TTS error: {error}")
        return JSONResponse({"ok": False}, status_code=502)

    if not audio:
        return JSONResponse({"ok": False}, status_code=502)

    return Response(content=audio, media_type="audio/mpeg")


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), lang: str = Form("en")):
    api_key = get_api_key()
    if not api_key:
        return JSONResponse({"ok": False, "text": ""}, status_code=400)

    content = await file.read()
    if not content:
        return JSONResponse({"ok": False, "text": ""}, status_code=400)

    mime = file.content_type or "audio/webm"
    filename = file.filename or ("audio.webm" if "webm" in mime else "audio.ogg")

    form = {
        "model": WHISPER_MODEL,
        "response_format": "json",
        "prompt": WHISPER_PROMPT,
        "temperature": "0",
    }
    if lang in ("en", "hi", "ur"):
        form["language"] = lang

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                GROQ_TRANSCRIBE_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                data=form,
                files={"file": (filename, content, mime)},
            )
    except Exception as error:
        print(f"Transcribe error: {error}")
        return JSONResponse({"ok": False, "text": ""}, status_code=502)

    if response.status_code == 429:
        return JSONResponse(
            {"ok": False, "text": ""},
            status_code=429,
        )

    if response.status_code != 200:
        print(f"Transcribe API error {response.status_code}: {response.text[:200]}")
        return JSONResponse({"ok": False, "text": ""}, status_code=502)

    try:
        data = response.json()
        text = str(data.get("text", "")).strip()
    except Exception:
        text = ""

    return {"ok": bool(text), "text": text}


@app.get("/health")
def health():
    return {"status": "ok"}
