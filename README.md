# 🤖 NOVA AI Chatbot

A fast AI Chatbot built using **Python**, **FastAPI**, and **Groq**, with a clean dark chat UI.

## 🚀 Features

- ⚡ Fast API backend (FastAPI + Uvicorn)
- 🤖 Groq API — `openai/gpt-oss-120b` model
- 💬 Multi-turn chat with context
- 🧠 Suggestion prompts for new chats
- 🔄 Retry on error
- 🗑️ New chat / clear history
- 📱 Responsive dark UI
- ❤️ Health check endpoint (`/health`)

## 🛠️ Technologies Used

- Python
- FastAPI
- Uvicorn
- Groq API
- httpx
- python-dotenv

## 📂 Project Structure

```
AI_Chatbot/
│── main.py            # FastAPI app + API routes
│── chatbot.py         # Groq API integration
│── static/
│   └── index.html     # Chat frontend (HTML/CSS/JS)
│── requirements.txt
│── render.yaml         # Render deploy config
│── .env                # GROQ_API_KEY (not committed)
│── .gitignore
│── README.md
```

## ⚙️ Local Installation

```bash
git clone <repository-url>
cd AI_Chatbot
pip install -r requirements.txt
```

Add your key to `.env`:

```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

> Get a key from https://console.groq.com

Run the server:

```bash
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

## ☁️ Deploy on Render (fast + free)

1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) → **New +** → **Web Service** → connect your repo.
3. Render will auto-detect `render.yaml` (build & start commands).
4. In **Environment**, add:
   - `GROQ_API_KEY` = your xAI API key
5. Deploy — you get a fast public URL like `https://nova-ai-chatbot.onrender.com`.

## 👨‍💻 Author

**Anand Kumar**
