"""Voice selection & speech-text sanitisation.

Separates TTS voice choice from endpoint business logic:

- TTS is Gemini TTS (generateContent + responseModalities=AUDIO); voices are
  Gemini prebuilt voice names (https://ai.google.dev/gemini-api/docs/speech).
- Voices are configurable via env (GEMINI_TTS_VOICE shared default;
  ASTRA_VOICE_EN / ASTRA_VOICE_HI / ASTRA_VOICE_UR per-language overrides).
- Explicit user preference ("en" / "hi") always wins over detection.
- Script detection ONLY recognises Unicode ranges (Devanagari, Arabic).
  Romanized Hinglish ("namaste", "kya haal hai") is deliberately NOT
  guessed — it would misfire on English text with borrowed words. When
  detection finds nothing, we fall back to the English/default voice.
- strip_for_speech() removes markdown, HTML tags and control chars so we
  never speak untrusted markup or hidden prompt text.
"""

import os
import re

# Default Gemini prebuilt voice — friendly, natural, multilingual
# (handles Hindi/English/Hinglish mixed text cleanly).
DEFAULT_VOICE = "Kore"

DEFAULT_VOICES = {
    "en": DEFAULT_VOICE,
    "hi": DEFAULT_VOICE,
    "ur": DEFAULT_VOICE,
}

# Languages a user may explicitly pick for TTS output
SUPPORTED_LANGS = ("auto", "en", "hi")


def get_voice(lang: str) -> str:
    """Resolve the Gemini voice name for a language code (en/hi/ur)."""
    code = (lang or "en").strip().lower()[:2]
    env_key = f"ASTRA_VOICE_{code.upper()}"
    configured = (os.getenv(env_key) or "").strip()
    if configured:
        return configured
    shared = (os.getenv("GEMINI_TTS_VOICE") or "").strip()
    if shared:
        return shared
    return DEFAULT_VOICES.get(code, DEFAULT_VOICE)


def detect_script(text: str) -> str | None:
    """Detect Hindi (Devanagari) or Urdu/Arabic script. Returns 'hi'/'ur'/None.

    Only Unicode ranges are used — Romanized Hinglish is intentionally
    undetectable here (see module docstring).
    """
    if any("ऀ" <= ch <= "ॿ" for ch in text):
        return "hi"
    if any("؀" <= ch <= "ۿ" for ch in text):
        return "ur"
    return None


def pick_voice(text: str, preference: str = "auto") -> str:
    """Choose the Gemini TTS voice.

    Priority: explicit user preference > script detection > default voice.
    `preference` is "auto" | "en" | "hi" (anything else = auto).
    """
    pref = (preference or "auto").strip().lower()
    if pref in ("en", "hi"):
        return get_voice(pref)
    detected = detect_script(text or "")
    if detected:
        return get_voice(detected)
    return get_voice("en")


def strip_for_speech(text: str) -> str:
    """Convert arbitrary chat text to plain spoken words.

    Removes: HTML tags, code fences + inline code, markdown decorations,
    URLs, control characters. Never returns markup.
    """
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # hidden/untrusted markup: drop script/style CONTENT, then any leftover tags
    text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
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
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"\|", " ", text)
    text = re.sub(r"https?://\S+", "", text)  # don't read raw URLs aloud
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.replace("*", "").replace("#", "").replace("`", "")
    return text.strip()
