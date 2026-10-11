
import os
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing. Check your .env file."
    )

client = genai.Client(api_key=api_key)

# Try models in order.
# Availability depends on your Gemini API key.
MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
]


def get_response(user_input):
    last_error = ""

    for model in MODELS:
        # Retry temporary server errors up to 3 times
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=user_input
                )

                return response.text or "AI returned an empty response."

            except Exception as e:
                last_error = str(e)
                error_upper = last_error.upper()

                # Model unavailable: try the next model
                if (
                    "404" in error_upper
                    or "NOT_FOUND" in error_upper
                    or "NOT SUPPORTED" in error_upper
                ):
                    break

                # Server busy or temporarily unavailable
                if (
                    "503" in error_upper
                    or "UNAVAILABLE" in error_upper
                ):
                    if attempt < 2:
                        time.sleep(2)
                        continue

                    break

                # Other errors: show a helpful message
                if (
                    "429" in error_upper
                    or "RESOURCE_EXHAUSTED" in error_upper
                ):
                    return (
                        "⚠️ Gemini API quota or rate limit reached. "
                        "Please check your API quota and try again later."
                    )

                return f"⚠️ Gemini API error: {last_error}"

    return (
        "⚠️ AI is temporarily unavailable. "
        "Please try again later.\n\n"
        f"Last error: {last_error}"
    )
