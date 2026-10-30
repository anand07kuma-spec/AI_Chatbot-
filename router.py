"""Astra model router.

Classifies each chat request (fast vs reasoning), resolves which Groq model
to use, builds the provider payload via separate adapters, calls Groq with
bounded retries, and applies a safe fallback when a text model is missing.

Env config (all optional):
    ASTRA_FAST_MODEL          default openai/gpt-oss-20b   (quick answers)
    ASTRA_REASONING_MODEL     default openai/gpt-oss-120b  (complex work;
                              legacy GROQ_MODEL is honoured as an alias)
    ASTRA_VISION_MODEL        default qwen/qwen3.8-27b     (image understanding)
    ASTRA_STT_MODEL           default whisper-large-v3-turbo
    ASTRA_ROUTER_MODE         auto | fast | reasoning      (default auto)
    ASTRA_MODEL_TIMEOUT       seconds per provider call    (default 60)
    ASTRA_STREAM_IDLE_TIMEOUT max silent seconds between
                              stream chunks                 (default 30)
"""

import asyncio
import json
import logging
import os
import re
import threading
import time
import uuid

import httpx
from dotenv import load_dotenv

from http_pool import get_async_client, sync_client

load_dotenv()

logger = logging.getLogger("astra.router")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

DEFAULT_FAST_MODEL = "openai/gpt-oss-20b"
DEFAULT_REASONING_MODEL = "openai/gpt-oss-120b"
DEFAULT_VISION_MODEL = "qwen/qwen3.8-27b"
DEFAULT_STT_MODEL = "whisper-large-v3-turbo"

# Groq qwen3.8-27b limits (console.groq.com/docs/vision):
# max 3 images per request, max 20MB per request.
VISION_MAX_IMAGES = 3
VISION_MAX_IMG_BYTES = 4 * 1024 * 1024  # base64-safe per-image cap (~5.3MB b64)
ALLOWED_IMAGE_MIMES = {"image/png", "image/jpeg", "image/webp", "image/gif"}

MAX_ATTEMPTS = 2  # 1 retry on transient failures (429 / 5xx / timeout / network)
RETRY_BACKOFF_SEC = 0.3
DEFAULT_TIMEOUT_SEC = 60.0

_MODEL_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._/-]{0,99}$")

_COMPLEXITY_RE = re.compile(
    r"(?i)\b("
    r"prove|proof|theorem|lemma|derive|derivative|integral|eigenvalue|calculus|"
    r"algorithm|big[\s-]?o|time complexity|space complexity|optimize|optimal|"
    r"dynamic programming|recursion tree|amortized|np[\s-]?hard|np[\s-]?complete|"
    r"race condition|deadlock|concurren\w*|thread[\s-]?safe|memory leak|"
    r"stack trace|segfault|segmentation fault|root cause|debug|refactor|"
    r"architecture|design pattern|step[\s-]?by[\s-]?step|in[\s-]?depth|"
    r"thoroughly|combinatorics|polynomial|matrix|determinant|"
    r"equation|inequality|probability of|proof by"
    r")\b"
)


# ---------- configuration ----------

def fast_model() -> str:
    return (os.getenv("ASTRA_FAST_MODEL") or DEFAULT_FAST_MODEL).strip() or DEFAULT_FAST_MODEL


def reasoning_model() -> str:
    val = (
        os.getenv("ASTRA_REASONING_MODEL")
        or os.getenv("GROQ_MODEL")  # legacy alias
        or DEFAULT_REASONING_MODEL
    ).strip()
    return val or DEFAULT_REASONING_MODEL


def vision_model() -> str:
    return (os.getenv("ASTRA_VISION_MODEL") or DEFAULT_VISION_MODEL).strip() or DEFAULT_VISION_MODEL


def stt_model() -> str:
    return (os.getenv("ASTRA_STT_MODEL") or DEFAULT_STT_MODEL).strip() or DEFAULT_STT_MODEL


def router_mode() -> str:
    mode = (os.getenv("ASTRA_ROUTER_MODE") or "auto").strip().lower()
    return mode if mode in ("auto", "fast", "reasoning") else "auto"


def _timeout() -> float:
    try:
        return max(1.0, float(os.getenv("ASTRA_MODEL_TIMEOUT", str(DEFAULT_TIMEOUT_SEC))))
    except ValueError:
        return DEFAULT_TIMEOUT_SEC


# ---------- task classification & model resolution ----------

def classify_task(messages) -> str:
    """Return "fast" or "reasoning" for a chat turn.

    Uncertain cases default to "reasoning" so quality never silently drops
    below the old single-model behaviour.
    """
    if not isinstance(messages, list) or not messages:
        return "reasoning"
    last = str(messages[-1].get("content", ""))
    if _COMPLEXITY_RE.search(last):
        return "reasoning"
    if "```" in last:
        return "reasoning"
    if len(last) > 500:
        return "reasoning"
    total = sum(len(str(m.get("content", ""))) for m in messages)
    if total > 4000:
        return "reasoning"
    if len(last.strip()) < 150:
        return "fast"
    return "reasoning"


def is_valid_model_choice(value: str) -> bool:
    v = value.strip()
    if v.lower() in ("auto", "fast", "reasoning"):
        return True
    return bool(_MODEL_ID_RE.match(v))


def resolve_text_model(task: str, requested: str | None = None) -> str:
    """Pick the model for a text turn.

    requested: None | "auto" | "fast" | "reasoning" | explicit Groq model id.
    Explicit client choices win; "auto" obeys ASTRA_ROUTER_MODE then classify.
    """
    req = (requested or "").strip()
    if req:
        low = req.lower()
        if low == "fast":
            return fast_model()
        if low == "reasoning":
            return reasoning_model()
        if low != "auto" and _MODEL_ID_RE.match(req):
            return req
    mode = router_mode()
    if mode == "fast":
        return fast_model()
    if mode == "reasoning":
        return reasoning_model()
    return fast_model() if task == "fast" else reasoning_model()


def _fallback_model(model: str) -> str | None:
    """Configured text fallback when a tier model 404s (never for explicit ids)."""
    if model == reasoning_model():
        return fast_model()
    if model == fast_model():
        return reasoning_model()
    return None


# ---------- payload adapters (model-specific formatting lives here) ----------

def build_text_payload(model: str, api_messages, max_tokens: int = 2048) -> dict:
    return {
        "model": model,
        "messages": api_messages,
        "max_tokens": max_tokens,
        "stream": False,
    }


def build_vision_payload(
    model: str, messages, images, system_prompt: str, max_tokens: int = 2048
) -> dict:
    api_messages = [{"role": "system", "content": system_prompt}]
    for m in messages[:-1]:
        api_messages.append({"role": m["role"], "content": str(m.get("content", ""))[:4000]})

    last = str(messages[-1].get("content", "What is in this image?"))
    parts = [{"type": "text", "text": last}]
    for img in images[:VISION_MAX_IMAGES]:
        mime = img.get("mime", "image/png")
        data = img.get("data", "")
        if data:
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"},
            })
    api_messages.append({"role": "user", "content": parts})

    return {
        "model": model,
        "messages": api_messages,
        "max_tokens": max_tokens,
        "stream": False,
    }


# ---------- provider transport ----------

# Concurrency cap: bound simultaneous upstream model calls so a burst of
# requests cannot open unbounded sockets / exhaust the Groq free-tier quota.
# Sync and async paths share the same budget (thread-safe counter).
MAX_CONCURRENT_UPSTREAM = max(1, int(os.getenv("ASTRA_MAX_CONCURRENT_UPSTREAM", "8")))
_upstream_slots = threading.BoundedSemaphore(MAX_CONCURRENT_UPSTREAM)


def _acquire_slot(blocking: bool = True, timeout: float | None = None) -> bool:
    if blocking and timeout is not None:
        return _upstream_slots.acquire(timeout=timeout)
    return _upstream_slots.acquire(blocking=blocking)


def _release_slot() -> None:
    _upstream_slots.release()


# Aggregate token usage (in-memory; per-process). Exposed via usage_snapshot()
# for observability. Counts are best-effort — reset on process restart.
_usage_lock = threading.Lock()
_token_usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _record_usage(data: dict | None) -> None:
    if not isinstance(data, dict):
        return
    usage = data.get("usage") or {}
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")
    with _usage_lock:
        _token_usage["calls"] += 1
        if isinstance(prompt, int):
            _token_usage["prompt_tokens"] += prompt
        if isinstance(completion, int):
            _token_usage["completion_tokens"] += completion
        if isinstance(total, int):
            _token_usage["total_tokens"] += total


def usage_snapshot() -> dict:
    with _usage_lock:
        return dict(_token_usage)


def _post_payload(model: str, payload: dict, api_key: str) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = sync_client.post(
            GROQ_API_URL, json=payload, headers=headers, timeout=_timeout()
        )
    except httpx.TimeoutException:
        return {"status": "timeout", "data": None}
    except Exception as error:
        # log exception type only — never request bodies or credentials
        logger.warning("router network error model=%s: %s", model, type(error).__name__)
        return {"status": "network", "data": None}

    if response.status_code == 200:
        try:
            return {"status": 200, "data": response.json()}
        except Exception:
            return {"status": "bad_json", "data": None}

    logger.warning(
        "router upstream status=%s model=%s body=%s",
        response.status_code, model, response.text[:200],
    )
    return {"status": response.status_code, "data": None}


def _post_payload_guarded(model: str, payload: dict, api_key: str) -> dict:
    """_post_payload with a concurrency slot (bounds simultaneous upstream calls)."""
    if not _acquire_slot(timeout=5.0):
        logger.warning("router busy model=%s (concurrency cap %d)", model, MAX_CONCURRENT_UPSTREAM)
        return {"status": "busy", "data": None}
    try:
        return _post_payload(model, payload, api_key)
    finally:
        _release_slot()


def _is_retryable(status) -> bool:
    # "busy" is our own concurrency cap — retrying would just re-queue.
    if status == "busy":
        return False
    return status in ("timeout", "network", 429) or (
        isinstance(status, int) and status >= 500
    )


def _call_with_retry(model: str, payload: dict, api_key: str, task: str) -> dict:
    last = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        start = time.monotonic()
        result = _post_payload_guarded(model, payload, api_key)
        latency_ms = int((time.monotonic() - start) * 1000)
        tokens = None
        data = result.get("data")
        if isinstance(data, dict):
            _record_usage(data)
            usage = data.get("usage") or {}
            tokens = usage.get("total_tokens")
        # metadata only — never API keys or prompt/reply content
        logger.info(
            "router attempt=%d task=%s model=%s status=%s latency_ms=%d tokens=%s",
            attempt, task, model, result["status"], latency_ms, tokens,
        )
        last = result
        if result["status"] == 200 or not _is_retryable(result["status"]):
            return result
        if attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_BACKOFF_SEC)
    return last


def _answer_from(data: dict) -> str:
    try:
        return str(data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


def _error_response(status, model: str | None) -> dict:
    if status == "busy":
        return {
            "ok": False,
            "reply": "Astra is handling too many requests right now. Please try again in a few seconds.",
            "model": model,
        }
    if status in (401, 403):
        return {
            "ok": False,
            "reply": "The Groq API key was rejected. Check your GROQ_API_KEY configuration.",
            "model": model,
        }
    if status == 429:
        return {
            "ok": False,
            "reply": "Groq rate limit reached. Please wait a moment and try again.",
            "model": model,
        }
    if status == 404:
        return {
            "ok": False,
            "reply": "The AI model is not available right now. Please try again later.",
            "model": model,
        }
    if status == "timeout":
        return {
            "ok": False,
            "reply": "The request took too long. Please try again.",
            "model": model,
        }
    if status == "network" or status == "bad_json":
        return {
            "ok": False,
            "reply": "Something went wrong while contacting Groq. Please try again.",
            "model": model,
        }
    if isinstance(status, int) and status >= 500:
        return {
            "ok": False,
            "reply": "Groq service is temporarily unavailable. Please try again in a moment.",
            "model": model,
        }
    return {
        "ok": False,
        "reply": f"The AI service returned an error ({status}). Please try again.",
        "model": model,
    }


# ---------- routes ----------

def route_chat(messages, api_key: str, system_prompt: str, requested_model: str | None = None) -> dict:
    """Classify, resolve, call Groq; fall back to the other text tier on 404."""
    task = classify_task(messages)
    model = resolve_text_model(task, requested_model)

    api_messages = [{"role": "system", "content": system_prompt}]
    for m in messages:
        api_messages.append({
            "role": m.get("role", "user"),
            "content": str(m.get("content", ""))[:4000],
        })

    payload = build_text_payload(model, api_messages)
    result = _call_with_retry(model, payload, api_key, task)

    if result["status"] == 404:
        fb = _fallback_model(model)
        if fb:
            logger.info("router fallback model=%s -> %s (reason=404)", model, fb)
            payload = build_text_payload(fb, api_messages)
            result = _call_with_retry(fb, payload, api_key, task)
            model = fb

    if result["status"] == 200:
        answer = _answer_from(result["data"])
        if answer:
            return {"ok": True, "reply": answer, "model": model, "task": task}
        return {
            "ok": False,
            "reply": "I couldn't generate a response. Please try again.",
            "model": model,
            "task": task,
        }

    err = _error_response(result["status"], model)
    err["task"] = task
    return err


def route_vision(messages, images, api_key: str, system_prompt: str) -> dict:
    """Vision calls always use the vision model. No text-model fallback —
    switching to a non-vision model would silently drop the images."""
    model = vision_model()
    payload = build_vision_payload(model, messages, images, system_prompt)
    result = _call_with_retry(model, payload, api_key, "vision")

    if result["status"] == 200:
        answer = _answer_from(result["data"])
        if answer:
            return {"ok": True, "reply": answer, "model": model, "task": "vision"}
        return {
            "ok": False,
            "reply": "I couldn't analyze the image. Please try again.",
            "model": model,
            "task": "vision",
        }

    err = _error_response(result["status"], model)
    err["task"] = "vision"
    return err


# ---------- streaming ----------
#
# Streaming protocol (SSE, documented in README):
#   event: meta    data: {"request_id","model","task"}
#   event: delta   data: {"request_id","text"}
#   event: done    data: {"request_id","model","task"}
#   event: error   data: {"request_id","reply"}
# Yields event dicts; main.py formats them as SSE frames.

def _idle_timeout() -> float:
    try:
        return max(1.0, float(os.getenv("ASTRA_STREAM_IDLE_TIMEOUT", "30")))
    except ValueError:
        return 30.0


async def stream_chat(
    messages,
    api_key: str,
    system_prompt: str,
    requested_model: str | None = None,
    images: list | None = None,
    request_id: str | None = None,
):
    """Async generator: stream a chat turn as event dicts (see protocol above).

    - Classification, model resolution and 404 fallback match route_chat.
    - Retries (429/5xx/timeout/network) only happen before the first delta;
      after partial output we emit a safe error event instead of restarting.
    - Client disconnects surface as GeneratorExit/CancelledError; the httpx
      stream is closed by the context managers and we just re-raise.
    """
    rid = request_id or uuid.uuid4().hex[:12]

    if images:
        task = "vision"
        model = vision_model()
        payload = build_vision_payload(model, messages, images, system_prompt)
        api_messages = None
    else:
        task = classify_task(messages)
        model = resolve_text_model(task, requested_model)
        api_messages = [{"role": "system", "content": system_prompt}]
        for m in messages:
            api_messages.append({
                "role": m.get("role", "user"),
                "content": str(m.get("content", ""))[:4000],
            })
        payload = build_text_payload(model, api_messages)
    payload["stream"] = True

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(_timeout(), read=_idle_timeout())

    attempt = 0
    tried_fallback = False
    emitted_delta = False
    while True:
        status = None
        if not _acquire_slot(blocking=False):
            logger.warning(
                "router stream busy request_id=%s model=%s (concurrency cap %d)",
                rid, model, MAX_CONCURRENT_UPSTREAM,
            )
            err = _error_response("busy", model)
            yield {"type": "error", "request_id": rid, "reply": err["reply"]}
            return
        try:
            client = get_async_client()
            async with client.stream(
                "POST", GROQ_API_URL, json=payload, headers=headers, timeout=timeout
            ) as resp:
                status = resp.status_code
                if status != 200:
                    await resp.aread()
                    logger.warning(
                        "router stream upstream status=%s model=%s request_id=%s",
                        status, model, rid,
                    )
                else:
                    yield {
                        "type": "meta",
                        "request_id": rid,
                        "model": model,
                        "task": task,
                    }
                    start = time.monotonic()
                    chunks = 0
                    finish_reason = None
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = obj.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        text = delta.get("content") or ""
                        fr = choices[0].get("finish_reason")
                        if fr:
                            finish_reason = fr
                        if text:
                            chunks += 1
                            emitted_delta = True
                            yield {"type": "delta", "request_id": rid, "text": text}
                    latency_ms = int((time.monotonic() - start) * 1000)
                    # metadata only — never keys or reply content
                    logger.info(
                        "router stream request_id=%s task=%s model=%s status=ok "
                        "latency_ms=%d chunks=%d finish=%s",
                        rid, task, model, latency_ms, chunks, finish_reason,
                    )
                    yield {
                        "type": "done",
                        "request_id": rid,
                        "model": model,
                        "task": task,
                    }
                    return
        except asyncio.CancelledError:
            logger.info("router stream cancelled request_id=%s model=%s", rid, model)
            raise
        except httpx.TimeoutException:
            status = "timeout"
            logger.warning(
                "router stream timeout request_id=%s model=%s", rid, model
            )
        except Exception as error:
            # log type only — no request bodies or credentials
            status = "network"
            logger.warning(
                "router stream error request_id=%s model=%s: %s",
                rid, model, type(error).__name__,
            )
        finally:
            _release_slot()

        # --- pre-delta failure: fallback / retry / safe error ---
        # Once any delta was emitted we must NOT retry or restart — the
        # client already has partial text; a restart would duplicate it.
        if (
            status == 404
            and task != "vision"
            and not tried_fallback
            and not emitted_delta
        ):
            fb = _fallback_model(model)
            if fb:
                logger.info(
                    "router stream fallback request_id=%s model=%s -> %s (reason=404)",
                    rid, model, fb,
                )
                tried_fallback = True
                model = fb
                if api_messages is not None:
                    payload = build_text_payload(fb, api_messages)
                    payload["stream"] = True
                continue

        if (
            not emitted_delta
            and _is_retryable(status)
            and attempt < MAX_ATTEMPTS - 1
        ):
            attempt += 1
            await asyncio.sleep(RETRY_BACKOFF_SEC)
            continue

        err = _error_response(status, model)
        yield {"type": "error", "request_id": rid, "reply": err["reply"]}
        return
