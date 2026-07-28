import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def get_response(user_input):
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_input
            )
            return response.text

        except Exception as e:
            if "503" in str(e) and attempt < 2:
                time.sleep(2)
                continue

            if "503" in str(e):
                return "⚠️ AI server is busy. Please try again in a few seconds."

            return f"Error: {e}"