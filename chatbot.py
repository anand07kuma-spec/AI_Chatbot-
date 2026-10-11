
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Read API key from environment or Streamlit Cloud Secrets
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    try:
        import streamlit as st
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = None

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing. Add it to your .env file "
        "or Streamlit Cloud Secrets."
    )

# Gemini client with a 20-second HTTP timeout
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=20000)
)

# Use one model to avoid waiting for model fallback
MODEL = "gemini-3.5-flash"


def get_response(user_input):
    """Generate a response from Gemini."""

    if not user_input or not user_input.strip():
        return "Please enter a question."
    
    
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=(
                "Answer the user's question in clear, simple English. "
                "Give a complete and useful explanation. "
                "For educational questions, include a definition, "
                "a simple example, and important points when relevant. "
                "For coding questions, provide complete working code "
                "and briefly explain it. Do not stop mid-sentence.\n\n"
                f"User question: {user_input.strip()}"
            ),
            config=types.GenerateContentConfig(
                max_output_tokens=2048,
                temperature=0.7
            )
        )

        answer = response.text

        if answer and answer.strip():
            return answer.strip()

        return "AI returned an empty response. Please try again."


  

    except Exception as e:
        error = str(e)
        error_upper = error.upper()

        if "429" in error_upper or "RESOURCE_EXHAUSTED" in error_upper:
            return (
                "Gemini API limit reached. "
                "Please try again later or check your API quota."
            )

        if (
            "504" in error_upper
            or "DEADLINE_EXCEEDED" in error_upper
            or "TIMEOUT" in error_upper
            or "TIMED OUT" in error_upper
        ):
            return (
                "Gemini is taking too long to respond. "
                "Please try again."
            )

        if (
            "503" in error_upper
            or "UNAVAILABLE" in error_upper
        ):
            return (
                "Gemini is temporarily busy. "
                "Please try again in a few seconds."
            )

        if (
            "404" in error_upper
            or "NOT_FOUND" in error_upper
            or "NOT SUPPORTED" in error_upper
        ):
            return (
                f"Model '{MODEL}' is unavailable for your API key. "
                "Please select a model supported by your Gemini API."
            )

        return f"Gemini API error: {error[:300]}"
