import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Fallback models (try one by one)
MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
]

def get_response(user_input):
    last_error = ""

    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_input
            )
            return response.text

        except Exception as e:
            last_error = str(e)

            # Retry if server is busy
            if "503" in last_error:
                time.sleep(2)
                continue

            # Try next model if current one is unavailable
            if "404" in last_error or "NOT_FOUND" in last_error:
                continue

    return f"⚠️ AI is temporarily unavailable.\n\n{last_error}"