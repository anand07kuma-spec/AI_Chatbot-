"""Astra chatbot facade.

Keeps the original get_response / get_response_with_images contract and
delegates model selection, retries, fallbacks and metrics to router.py.
"""

import logging
import os

from router import (
    ALLOWED_IMAGE_MIMES,
    VISION_MAX_IMAGES,
    VISION_MAX_IMG_BYTES,
    route_chat,
    route_vision,
    stream_chat,
)
from router import DEFAULT_FAST_MODEL as DEFAULT_MODEL
from router import DEFAULT_VISION_MODEL as VISION_MODEL

logger = logging.getLogger("astra.chatbot")


def get_api_key():
    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        return api_key.strip()

    return None


SYSTEM_PROMPT = (
    "You are Astra AI, a friendly and helpful assistant. "
    "ALWAYS detect the language of the user's message and reply in EXACTLY the same language. "
    "Supported languages: English, Hindi, Hinglish (Hindi in Latin script), Urdu, "
    "Arabic, Bengali, Tamil, Telugu, Marathi, Gujarati, Punjabi, Malayalam, Kannada, "
    "Spanish, French, German, Portuguese, Russian, Chinese, Japanese, Korean, "
    "Indonesian, Turkish, Italian, Dutch, Polish, Vietnamese, Thai, and more. "
    "If the user writes in Hinglish, reply in Hinglish. If in Urdu (Roman or Nastaliq), reply in Urdu. "
    "If in Hindi (Devanagari or Latin), reply in Hindi. If in English, reply in English. "
    "NEVER mix languages unless the user does. "
    "If the user asks for code, give complete, runnable code first, "
    "followed by at most 2-3 short usage steps — do not explain concepts "
    "unless asked. Keep replies concise and conversational with minimal "
    "markdown: no headings, no bullet lists unless necessary. "
    "Put code only inside ``` code blocks. Never use LaTeX."
)

VISION_SYSTEM_PROMPT = (
    "You are Astra AI, a friendly and helpful assistant. "
    "The user has shared image(s) with you. Analyze and describe them accurately. "
    "ALWAYS detect the language of the user's message and reply in EXACTLY the same language. "
    "If the user asks for code, give complete, runnable code first. "
    "Keep replies concise and conversational with minimal markdown."
)


def get_response(messages, model: str | None = None, system_prompt: str | None = None):
    """Route a text chat turn and return {"ok": bool, "reply": str, ...}."""

    if not isinstance(messages, list) or not messages:
        return {"ok": False, "reply": "Please enter a message."}

    api_key = get_api_key()

    if not api_key:
        return {
            "ok": False,
            "reply": "API key is missing. Add GROQ_API_KEY to your .env file "
                     "(locally) or to Render Environment Variables.",
        }

    prompt = system_prompt or SYSTEM_PROMPT
    return route_chat(messages, api_key, prompt, requested_model=model)


def get_response_with_images(messages, images):
    """Route a vision chat turn and return {"ok": bool, "reply": str, ...}."""

    if not isinstance(messages, list) or not messages:
        return {"ok": False, "reply": "Please enter a message."}

    api_key = get_api_key()
    if not api_key:
        return {"ok": False, "reply": "API key is missing."}

    return route_vision(messages, images, api_key, VISION_SYSTEM_PROMPT)


async def stream_response(messages, model: str | None = None, images: list | None = None, request_id: str | None = None, system_prompt: str | None = None):
    """Async generator: stream a chat turn as event dicts (router protocol).

    Yields {"type": meta|delta|done|error, ...}. Input validation happens in
    main.py before the stream starts; here we only handle missing keys.
    """

    api_key = get_api_key()
    if not api_key:
        yield {
            "type": "error",
            "reply": "API key is missing. Add GROQ_API_KEY to your .env file "
                     "(locally) or to Render Environment Variables.",
        }
        return

    if system_prompt:
        prompt = system_prompt
    else:
        prompt = VISION_SYSTEM_PROMPT if images else SYSTEM_PROMPT
    async for event in stream_chat(
        messages, api_key, prompt,
        requested_model=model, images=images, request_id=request_id,
    ):
        yield event
