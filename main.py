import base64
import hashlib
import io
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path

import edge_tts
import httpx
from docx import Document as DocxDocument
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from pypdf import PdfReader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from chatbot import get_api_key, get_response, get_response_with_images

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Astra AI Chatbot")

app.add_middleware(GZipMiddleware, minimum_size=500)


class StaticCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "public, max-age=3600"
        return response


app.add_middleware(StaticCacheMiddleware)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

BUILD_VER = str(int(time.time()))

_index_html_cache = None

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
TTS_MAX_CHARS = 800
TTS_RAW_LIMIT = 2500
CHAT_HISTORY_LIMIT = 24
CHAT_CONTENT_LIMIT = 6000

_tts_cache: OrderedDict = OrderedDict()
_TTS_CACHE_MAX = 60
_tts_lock = threading.Lock()


class ChatRequest(BaseModel):
    messages: list[dict]
    images: list[dict] | None = None  # [{mime, data}] base64


class ChatRequestLegacy(BaseModel):
    messages: list[dict]


@app.get("/")
def index():
    global _index_html_cache
    if _index_html_cache is None:
        _index_html_cache = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(_index_html_cache.replace("__VER__", BUILD_VER))


@app.post("/api/chat")
def chat(request: ChatRequest):
    messages = [
        {
            "role": m.get("role", "user"),
            "content": str(m.get("content", ""))[:CHAT_CONTENT_LIMIT],
        }
        for m in request.messages
        if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()
    ]

    messages = messages[-CHAT_HISTORY_LIMIT:]

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

    images = request.images or []
    if images:
        processed = []
        for img in images[:5]:
            mime = str(img.get("mime", "image/png"))
            data = str(img.get("data", ""))
            if data.startswith("data:"):
                data = data.split(",", 1)[-1]
            if data:
                processed.append({"mime": mime, "data": data})
        if processed:
            result = get_response_with_images(messages, processed)
            return {"ok": result["ok"], "reply": result["reply"]}

    result = get_response(messages)
    return {"ok": result["ok"], "reply": result["reply"]}


ALLOWED_DOC_EXTS = {".pdf", ".docx", ".txt", ".md", ".csv", ".json", ".log", ".py", ".js", ".html", ".css", ".xml", ".yml", ".yaml"}
DOC_MAX_BYTES = 10 * 1024 * 1024
DOC_MAX_CHARS = 20000


@app.post("/api/extract")
async def extract_doc(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        return JSONResponse({"ok": False, "error": "Empty file."}, status_code=400)

    if len(content) > DOC_MAX_BYTES:
        return JSONResponse({"ok": False, "error": "File too large (max 10MB)."}, status_code=400)

    filename = file.filename or "file"
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_DOC_EXTS:
        return JSONResponse(
            {"ok": False, "error": f"Unsupported file type '{ext}'. Allowed: PDF, DOCX, TXT, MD, CSV, JSON, code files."},
            status_code=400,
        )

    try:
        if ext == ".pdf":
            reader = PdfReader(io.BytesIO(content))
            parts = []
            for page in reader.pages[:50]:
                text = page.extract_text() or ""
                if text.strip():
                    parts.append(text)
            text = "\n\n".join(parts)
        elif ext == ".docx":
            doc = DocxDocument(io.BytesIO(content))
            parts = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        parts.append(" | ".join(cells))
            text = "\n".join(parts)
        else:
            text = content.decode("utf-8", errors="replace")
    except Exception as error:
        print(f"Extract error: {error}")
        return JSONResponse({"ok": False, "error": "Could not read this file."}, status_code=422)

    text = text.strip()
    if not text:
        return JSONResponse({"ok": False, "error": "No readable text found in this file."}, status_code=422)

    return {
        "ok": True,
        "name": filename,
        "chars": len(text),
        "text": text[:DOC_MAX_CHARS],
    }


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
async def tts(request: TTSRequest):
    raw = request.text.strip()[:TTS_RAW_LIMIT]
    spoken = strip_markdown(raw)
    if not spoken:
        spoken = TTS_FALLBACK

    spoken = spoken[:TTS_MAX_CHARS]
    voice = pick_voice(spoken)

    cache_key = hashlib.sha1(f"{voice}|{spoken}".encode("utf-8")).hexdigest()
    with _tts_lock:
        cached = _tts_cache.get(cache_key)
        if cached is not None:
            _tts_cache.move_to_end(cache_key)
            return Response(content=cached, media_type="audio/mpeg")

    try:
        audio = await _edge_tts(spoken, voice)
    except Exception as error:
        print(f"TTS error: {error}")
        return JSONResponse({"ok": False}, status_code=502)

    if not audio:
        return JSONResponse({"ok": False}, status_code=502)

    with _tts_lock:
        _tts_cache[cache_key] = audio
        _tts_cache.move_to_end(cache_key)
        while len(_tts_cache) > _TTS_CACHE_MAX:
            _tts_cache.popitem(last=False)

    return Response(content=audio, media_type="audio/mpeg")


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), lang: str = Form("")):
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
    if lang and lang in ("en", "hi", "ur"):
        form["language"] = lang
    # else: Whisper auto-detects the language from audio

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
