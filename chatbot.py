import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def get_response(user_input):
    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=user_input
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"