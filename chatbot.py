import os
import httpx
from dotenv import load_dotenv

load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
REQUEST_TIMEOUT = 60.0


def get_api_key():
    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        return api_key.strip()

    return None


def get_response(messages):
    """Send chat history to Groq and return {"ok": bool, "reply": str}."""

    if not isinstance(messages, list) or not messages:
        return {"ok": False, "reply": "Please enter a message."}

    api_key = get_api_key()

    if not api_key:
        return {
            "ok": False,
            "reply": "API key is missing. Add GROQ_API_KEY to your .env file "
                     "(locally) or to Render Environment Variables.",
        }

    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

    system_prompt = (
        "You are Astra AI, a friendly and helpful assistant. "
        "Always reply in the same language the user used "
        "(English, Hindi, Hinglish, Urdu etc.). "
        "If the user asks for code, give complete, runnable code first, "
        "followed by at most 2-3 short usage steps — do not explain concepts "
        "unless asked. Keep replies concise and conversational with minimal "
        "markdown: no headings, no bullet lists unless necessary. "
        "Put code only inside ``` code blocks. Never use LaTeX."
    )

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "max_tokens": 2048,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = httpx.post(
            GROQ_API_URL,
            json=payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 429:
            return {
                "ok": False,
                "reply": "Groq rate limit reached. Please wait a moment and try again.",
            }

        if response.status_code in (401, 403):
            try:
                detail = response.json()
                message = str(detail.get("error", {}) or "")
            except Exception:
                message = ""

            if "credit" in message.lower() or "license" in message.lower():
                return {
                    "ok": False,
                    "reply": "Your Groq account has no credits left. "
                             "Check console.groq.com, then try again.",
                }

            return {
                "ok": False,
                "reply": "The Groq API key was rejected. Check your GROQ_API_KEY configuration.",
            }

        if response.status_code == 404:
            return {
                "ok": False,
                "reply": f"The model '{model}' was not found. Check GROQ_MODEL in your environment.",
            }

        if response.status_code >= 500:
            return {
                "ok": False,
                "reply": "Groq service is temporarily unavailable. Please try again in a moment.",
            }

        if response.status_code != 200:
            try:
                detail = response.json()
                message = str(detail.get("error", {}).get("message", ""))[:200]
            except Exception:
                message = ""
            return {
                "ok": False,
                "reply": f"Groq API error ({response.status_code}): {message}",
            }

        data = response.json()
        answer = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        if answer and answer.strip():
            return {"ok": True, "reply": answer.strip()}

        return {"ok": False, "reply": "I couldn't generate a response. Please try again."}

    except httpx.TimeoutException:
        return {
            "ok": False,
            "reply": "The request took too long. Please try again.",
        }
    except Exception as error:
        print(f"Groq API error: {error}")
        return {
            "ok": False,
            "reply": "Something went wrong while contacting Groq. Please try again.",
        }
