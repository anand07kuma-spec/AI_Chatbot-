# Astra AI

A FastAPI-based AI assistant with streaming replies, smart model routing, live web search, document RAG, voice and vision — with **no login**: chats are scoped to your browser device. The frontend is plain HTML/CSS/JS (no framework, no build step).

## Key Features

- **Streaming chat (SSE)** — word-by-word replies with Stop and Regenerate
- **Smart model router** — Auto picks a fast model for simple chat and a reasoning model for complex/code questions; force Fast/Reasoning or an explicit model id via the API
- **Live web search** (optional, Tavily) — time-sensitive answers with `[1]` source citations; Auto/Always/Never toggle
- **Document RAG** — upload PDF/DOCX/TXT/MD/CSV/JSON/code per chat; answers cite `[D1]`-style sources
- **Vision** — image understanding (max 3 images per request)
- **Voice** — speech-to-text (Groq Whisper) and text-to-speech (Gemini TTS)
- **Coding assistant** — coding mode auto-detected from your message; isolated sandbox execution available via API (disabled unless configured)
- **Guest mode & memory** — device-scoped history and consent-gated long-term memory, stored in Supabase with an in-memory fallback
- **UI** — dark/light themes, markdown/tables/syntax-highlighted code, responsive and accessible
- **Hardening** — rate limiting, body-size caps, security headers, request IDs, bounded retries, pooled connections, secret-safe logs

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12+, FastAPI, uvicorn, httpx |
| LLM / vision / STT | Groq API |
| Web search | Tavily API (optional) |
| Text-to-speech | Google Gemini TTS (optional) |
| Embeddings | OpenAI API (optional; local hash fallback) |
| Persistence | Supabase (PostgREST) or in-memory store |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Tests | pytest |

## Project Structure

```
main.py                  FastAPI app, routes, middleware
router.py                model routing, retries, usage counters
chatbot.py               system prompts + chat facade
websearch.py             search policy, cache, citations
rag.py                   document extract → chunk → embed → retrieve
store.py                 Supabase store + in-memory fallback
memory.py                consent-gated, device-scoped memory
device.py                guest identity (X-Device-Id header)
execution.py             optional code-sandbox client
voice.py                 TTS/STT helpers
routes_conversations.py  chat history APIs
routes_memories.py       memory APIs
static/                  frontend (index.html, style.css, app.js)
tests/                   pytest suite (external calls mocked)
supabase/schemas/        SQL schema + RLS
scripts/                 env audit, latency benchmark
render.yaml              Render deploy config
.env.example             environment template (placeholders only)
```

## Installation

```powershell
git clone https://github.com/anand07kuma-spec/AI_Chatbot-.git
cd AI_Chatbot-

python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

Copy-Item .env.example .env
```

Then edit `.env` and add at least `GROQ_API_KEY` (from <https://console.groq.com>). Add the optional keys shown in `.env.example` only for the features you want. **Never commit `.env`** — it is gitignored.

Without Supabase credentials the app runs on the in-memory store (history resets on restart).

## Environment Variables

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | **Required** — chat, vision, speech-to-text |
| `GEMINI_API_KEY` | Enables text-to-speech output |
| `TAVILY_API_KEY` | Enables live web search |
| `OPENAI_API_KEY` | Real RAG embeddings (otherwise local hash fallback) |
| `SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY` | Durable history/memory (server-side only) |
| `ASTRA_STORE` | Force `supabase` or `memory` storage backend |
| `ASTRA_FAST_MODEL` / `ASTRA_REASONING_MODEL` | Fast-tier / reasoning-tier model ids |
| `ASTRA_VISION_MODEL` / `ASTRA_STT_MODEL` | Image and speech-to-text model ids |
| `ASTRA_ROUTER_MODE` | `auto` \| `fast` \| `reasoning` |
| `ASTRA_SANDBOX_URL` / `ASTRA_SANDBOX_KEY` | Piston-compatible code sandbox (optional) |
| `ASTRA_MAX_CONCURRENT_UPSTREAM` | Global cap on simultaneous model calls |
| `ASTRA_MODEL_TIMEOUT` / `ASTRA_STREAM_IDLE_TIMEOUT` | Upstream and stream idle timeouts (seconds) |
| `RATE_LIMIT_PER_MINUTE` | Per-IP API limit (`0` disables) |
| `ALLOWED_ORIGINS` | CORS origins (unset = same-origin only) |
| `GEMINI_TTS_MODEL` / `GEMINI_TTS_VOICE` | TTS model and default voice |

`.env.example` is the configuration template with placeholders and comments. Secrets are read from environment variables only — never committed, logged, or sent to the browser.

## Run Locally

```powershell
.\.venv\Scripts\activate
uvicorn main:app --reload --port 8010
```

Open <http://127.0.0.1:8010>.

If activation is blocked by execution policy, run directly:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8010
```

There is also a helper script: `.\start_server.bat`.

For durable history, create the Supabase tables once by running `supabase/schemas/001_create_tables.sql` in the Supabase SQL Editor.

## API Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness + token-usage counters |
| `GET /api/models` | Resolved model modes for this deployment |
| `POST /api/chat` | JSON chat (routing, search, RAG, memory) |
| `POST /api/chat/stream` | SSE streaming chat — events: `[docs?] [search*] → meta → delta* → (done \| error)` |
| `POST /api/docs` · `GET /api/docs` · `DELETE /api/docs/{id}` | Document upload / list / remove |
| `POST /api/tts` · `POST /api/transcribe` | Text-to-speech / speech-to-text |
| `GET /api/conversations/*` · `GET/DELETE /api/memories/*` | Device-scoped history and memory (send `X-Device-Id`) |
| `GET /api/sandbox` · `POST /api/execute` | Sandbox status / code execution (503 until configured) |
| `WS /ws/voice` | Voice session (rate-limited, device-scoped) |

Identity: the browser stores a random device id in `localStorage` and sends it as `X-Device-Id`; all history/memory queries are scoped to it. Every response carries `X-Request-ID` for log correlation.

## Testing

```powershell
.\.venv\Scripts\activate
python -m pytest
```

- Unit tests mock all external AI calls and force the in-memory store — no keys, network, or database needed.
- Integration tests (real API calls, cost quota) are deselected by default; run them explicitly with:
  `python -m pytest tests/integration -m integration -v`
- Latency benchmark (needs a running server): `python scripts/measure_latency.py`

## Security Note

- **Secrets**: environment variables only; `.env` is gitignored; keys are never logged, never sent to the client, and never appear in frontend JS.
- **No accounts, no user table**: a regex-validated random device id identifies each browser; every history/memory query is scoped to it, so other devices' rows are unreachable. The Supabase service-role key never leaves the server.
- **Transport hygiene**: security headers (nosniff, DENY framing, no-referrer), optional CORS allowlist, gzip for large bodies.
- **Abuse controls**: per-IP rate limit, body-size cap, image validation, upstream concurrency cap.
- **Prompt fencing**: web results and documents enter prompts as clearly-marked untrusted blocks; the UI escapes model output and allows only http(s) links.
- **Data control**: per-device export and delete for conversations and memories; memory is consent-gated — nothing is injected without approval.
