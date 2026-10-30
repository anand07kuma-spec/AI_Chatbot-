# Astra AI Chatbot

A production-minded AI assistant built with **Python**, **FastAPI** and **Groq** — streaming replies, smart model routing, live web search, document RAG, voice, vision, guest-mode chats with device-scoped history and user-controlled memory, and an optional code sandbox. Dark/light chat UI, no frontend framework, no build step.

## Features

- **Streaming replies (SSE)** — word-by-word output, Stop button, Regenerate, Retry on errors
- **Smart model router** — Auto picks a fast model (`openai/gpt-oss-20b`) for simple chat and a reasoning model (`openai/gpt-oss-120b`) for complex/code questions; force Fast/Reasoning from the UI or send an explicit Groq model id via the API
- **Live web search** (optional, Tavily) — time-sensitive questions get real sources with `[1]` citations; Auto/Always/Never toggle
- **Document intelligence & RAG** — upload PDF/DOCX/TXT/MD/CSV/JSON/code per chat; answers from your documents with `[D1]` citations and page numbers
- **Vision** — image understanding (`qwen/qwen3.8-27b`, max 3 images, 4MB each)
- **Voice** — speech-to-text (Groq Whisper) and text-to-speech (Gemini TTS, WAV output), auto language detect
- **Coding assistant** — auto-detected coding mode; optional isolated sandbox execution (Piston-compatible, disabled by default)
- **Guest mode & memory** — no login/signup; chats sync per browser device (`X-Device-Id` header, localStorage id), persisted in Supabase with an in-memory fallback, consent-gated long-term memory, export/delete
- **UI** — dark/light themes, settings modal, markdown/tables/syntax-highlighted code, accessibility (ARIA, focus-visible, keyboard nav), responsive
- **Hardening** — rate limiting, body-size caps, security headers, X-Request-ID on every response, bounded retries, concurrency caps, pooled connections, secret-safe logs

## Architecture

```
Browser (static/index.html + app.js, vanilla JS)
   │  SSE stream / JSON
   ▼
FastAPI (main.py)
   ├─ middleware: request-id → static-cache → security-headers → body-limit → rate-limit → gzip
   ├─ router.py       classify → resolve model → pooled Groq call → retry/fallback → usage counters
   ├─ websearch.py    Tavily provider → TTL cache → rate limit → safe failure
   ├─ rag.py          extract → chunk → embed → DocStore → retrieve → citations
   ├─ execution.py    optional Piston sandbox client (HTTP, isolated)
   ├─ device.py       X-Device-Id guest identity (validation, optional/required deps)
   └─ store.py        Supabase (PostgREST via httpx) with thread-safe memory fallback
```

Key design rules:

- **Streaming path** (`/api/chat/stream`) uses a per-event-loop pooled `httpx.AsyncClient` (keep-alive to Groq); **sync paths** (chat, search, embeddings) share one pooled `httpx.Client`. See `http_pool.py`.
- **Every upstream call is bounded** — timeouts everywhere, transient-only retries (1 retry, short backoff), and a global concurrency cap (`ASTRA_MAX_CONCURRENT_UPSTREAM`, default 8) that returns a clear "busy" error instead of piling up sockets.
- **Untrusted data is fenced** — web results and document chunks are injected into prompts as clearly-marked untrusted blocks; the UI renders only escaped markdown with an http(s)-only link allowlist.
- **Never crashes a chat** — search/RAG/memory failures degrade to a normal reply with a safe notice.

### Project layout

```
main.py                  FastAPI app, routes, middleware
router.py                model routing, retries, fallbacks, usage counters
http_pool.py             shared pooled httpx clients
websearch.py             Tavily search: policy, cache, citations
rag.py                   documents: extract → chunk → embed → store → retrieve
chatbot.py               chat facade (system prompts + router)
execution.py             sandbox client (optional)
device.py               guest identity (X-Device-Id header)
store.py                Supabase PostgREST store + in-memory fallback
memory.py               consent-gated memory (store-backed, device-scoped)
voice.py                Gemini TTS voice selection
routes_conversations.py chat history APIs (device-scoped CRUD/search/export)
routes_memories.py      memory APIs (list/preferences/suggest/approve/delete)
static/                  frontend (index.html, style.css, app.js)
tests/                   unit suite (mocked externals)
tests/integration/       live API tests (opt-in)
scripts/measure_latency.py   latency bench
render.yaml              Render deploy config
```

## Setup (local)

```bash
git clone <repository-url>
cd AI_Chatbot
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
copy .env.example .env        # Windows  (cp .env.example .env on macOS/Linux)
```

Edit `.env` — the only required key:

```
GROQ_API_KEY=your_groq_api_key_here     # https://console.groq.com
GEMINI_API_KEY=your_gemini_api_key_here # TTS — https://aistudio.google.com/apikey
```

Without `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` the app runs on the in-memory store (history lives only for the process lifetime).

Run:

```bash
uvicorn main:app --reload --port 8010
```

Open **http://127.0.0.1:8010**. For durable history, create the Supabase tables once via the SQL Editor (see `Supabase setup` below).

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — (required) | Chat / vision / STT provider key |
| `GEMINI_API_KEY` | unset (TTS disabled) | Gemini text-to-speech |
| `GEMINI_TTS_MODEL` | `gemini-2.5-flash-preview-tts` | TTS model id |
| `GEMINI_TTS_VOICE` | `Kore` | Default Gemini voice (per-lang: `ASTRA_VOICE_EN/HI/UR`) |
| `SUPABASE_URL` | unset | Supabase project URL (else in-memory store) |
| `SUPABASE_SERVICE_ROLE_KEY` | unset | Server-side PostgREST auth (never sent to the browser) |
| `SUPABASE_ANON_KEY` | unset | Anon key (read probes) |
| `ASTRA_STORE` | auto | `supabase` \| `memory` force-override |
| `TAVILY_API_KEY` | unset | Enables live web search |
| `OPENAI_API_KEY` | unset | Real RAG embeddings (else local hash fallback) |
| `GROQ_MODEL` / `ASTRA_REASONING_MODEL` | `openai/gpt-oss-120b` | Reasoning-tier model |
| `ASTRA_FAST_MODEL` | `openai/gpt-oss-20b` | Fast-tier model |
| `ASTRA_VISION_MODEL` | `qwen/qwen3.8-27b` | Image understanding |
| `ASTRA_STT_MODEL` | `whisper-large-v3-turbo` | Speech-to-text |
| `ASTRA_ROUTER_MODE` | `auto` | `auto` \| `fast` \| `reasoning` |
| `ASTRA_MODEL_TIMEOUT` | `60` | Seconds per provider call |
| `ASTRA_STREAM_IDLE_TIMEOUT` | `30` | Max silent seconds between stream chunks |
| `ASTRA_MAX_CONCURRENT_UPSTREAM` | `8` | Global cap on simultaneous model calls |
| `ASTRA_EMBEDDINGS_PROVIDER` | `auto` | `auto` \| `openai` \| `hash` |
| `ASTRA_SEARCH_*` | see `websearch.py` | Search timeout / cache TTL / max results |
| `ASTRA_SANDBOX_URL` / `ASTRA_SANDBOX_KEY` | unset | Piston-compatible code sandbox (optional) |
| `RATE_LIMIT_PER_MINUTE` | `30` | Per-IP API limit (`0` disables) |
| `ALLOWED_ORIGINS` | unset | CORS origins (disabled = same-origin only) |

Secrets are read from environment variables only — never committed, never sent to the frontend, never logged.

## API overview

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness + token-usage counters |
| `GET /api/models` | Resolved model modes for this deployment |
| `POST /api/chat` | JSON chat (routing, search, RAG, memory) |
| `POST /api/chat/stream` | SSE streaming chat (same body; optional `web_search`, `conversation_id`, `doc_ids`, `reply_lang`, `mode`) |
| `POST /api/tts` · `POST /api/transcribe` | Voice synthesis / speech-to-text |
| `GET /api/sandbox` · `POST /api/execute` | Sandbox status / code execution |
| `POST /api/docs` · `GET` · `DELETE /api/docs/{id}` | Document upload / list / remove |
| `/api/conversations/*` · `/api/memories/*` | Device-scoped chat history, memory (send `X-Device-Id`; `GET /api/memories/preferences` for the toggle) |

SSE event order: `[docs?] [search*] → meta → delta* → (done | error)`, with `memory` events (suggestions) on streams. Every stream and every response carries an `X-Request-ID` for log correlation. Identity: the browser generates a random device id (localStorage `astra_device_id`, regex-validated) and sends it as `X-Device-Id`; all history/memory queries are scoped to it.

## Testing

```bash
# Unit suite — all external AI calls mocked; no keys or network needed
python -m pytest

# Verbose
python -m pytest tests/ -v
```

- 301 unit tests cover routing, streaming, search, RAG, device identity, guest history/memory APIs, voice, sandbox, UI-facing APIs, security headers, rate limiting, malformed/oversized input, log redaction, request ids, usage counters, the concurrency cap and frontend code-block rendering (the JS render test skips if `node` is not on PATH).
- Tests force the in-memory store (`ASTRA_STORE=memory`) — no database or network needed.

**Integration tests** (real Groq/Tavily calls — opt-in, cost real quota):

```bash
# Windows
set ASTRA_INTEGRATION=1&& set GROQ_API_KEY=gsk_...&& python -m pytest tests/integration -m integration -v
# macOS/Linux
ASTRA_INTEGRATION=1 GROQ_API_KEY=gsk_... python -m pytest tests/integration -m integration -v
```

They are **deselected by default** (`pytest.ini`) so `pytest` never hits the network or spends your quota by accident.

**Latency bench** (real model calls; run against a local server):

```bash
uvicorn main:app --port 8010    # in one shell
python scripts/measure_latency.py
```

## Performance

Measured locally (Windows, Python 3.14, real Groq calls, n=5 chat / n=20 static; before/after Phase 9 pooling):

| Endpoint | p50 (before) | p50 (after) | Notes |
|---|---|---|---|
| `GET /health` | 5.2 ms | 6.1 ms | noise-level |
| `GET /` | 4.8 ms | 5.6 ms | noise-level |
| `GET /api/models` | 2.3 ms | 4.6 ms | noise-level |
| `POST /api/chat` (real) | 757 ms | 677 ms | dominated by Groq model latency |
| stream TTFT (real) | 691 ms | 349 ms (min) | keep-alive reuse helps repeat calls |

Honest caveats: n=5 real-model samples have high variance (one slow upstream call moves p95 by seconds); the structural wins are connection reuse (no repeated TCP+TLS handshake to Groq/Tavily/OpenAI), bounded concurrency (a burst cannot open unbounded sockets), and transient-only retries (1 retry, 0.3s backoff — never retries 4xx or "busy"). Token usage is counted in-process and exposed at `/health` (`calls`, `prompt_tokens`, `completion_tokens`, `total_tokens`).

## Supabase setup (one-time)

The server persists guest history/memory in Supabase via PostgREST (service-role key, server-side only). Run this once in **Supabase → SQL Editor** (safe to re-run):

```sql
create table if not exists devices (
  device_id text primary key,
  memory_enabled boolean not null default true,
  created_at timestamptz not null default now()
);

create table if not exists conversations (
  id text primary key,
  device_id text not null references devices(device_id) on delete cascade,
  title text not null default '',
  created_at timestamptz not null,
  updated_at timestamptz not null
);
create index if not exists conversations_device_idx on conversations (device_id, updated_at desc);

create table if not exists messages (
  id text primary key,
  conversation_id text not null references conversations(id) on delete cascade,
  role text not null,
  content text not null,
  created_at timestamptz not null
);
create index if not exists messages_conv_idx on messages (conversation_id, created_at);

create table if not exists memories (
  id text primary key,
  device_id text not null references devices(device_id) on delete cascade,
  content text not null,
  kind text not null default 'fact',
  status text not null default 'active',
  created_at timestamptz not null,
  updated_at timestamptz not null
);
create index if not exists memories_device_idx on memories (device_id, status);

alter table devices enable row level security;
alter table conversations enable row level security;
alter table messages enable row level security;
alter table memories enable row level security;
```

RLS stays enabled with **no policies**: the browser never talks to Supabase directly — only the server does, with the service-role key (which bypasses RLS), and every query is already scoped by `device_id` in application code.

## Deployment (Render)

`render.yaml` is ready: Python pinned via `PYTHON_VERSION`, `uvicorn main:app --host 0.0.0.0 --port $PORT`, `/health` health check, no migration step (schema lives in the SQL above).

1. Push the repo to GitHub.
2. [render.com](https://render.com) → **New +** → **Web Service** → connect the repo (Render reads `render.yaml` automatically).
3. Create the Supabase tables once (SQL above) if not already done.
4. Set environment variables (all secrets `sync: false` in `render.yaml` — set them in the Render dashboard):
   - `GROQ_API_KEY` (required, Secret)
   - `GEMINI_API_KEY` (Secret — enables TTS)
   - `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_ANON_KEY` (Secrets)
   - `TAVILY_API_KEY`, `OPENAI_API_KEY` (optional, Secret)
5. Deploy → public URL like `https://astra-ai-chatbot.onrender.com`.

**Operational notes:**

- **No local DB** — without Supabase keys the app serves from the in-memory store (wiped per restart/redeploy); with Supabase, history is durable. In-process caches (TTS clips, search cache, RAG DocStore) reset per deploy — by design.
- **Rollback** — Render keeps deploy history; roll back from the dashboard (previous image).
- **Logs** — request-id correlated, secret-free; tail from the Render dashboard.

This deployment configuration has **not** been live-verified from this machine — the app is verified locally (301 tests + live smoke + Playwright E2E + latency bench); Render behaviour is inferred from `render.yaml` and Render's documented Python/runtime behaviour.

## Security

- **Secrets**: env vars only; `.env` gitignored; keys never logged (verified by a redaction test), never sent to the client, never in frontend JS.
- **Identity without accounts**: there is no login and no user table — the browser generates a random device id (validated regex `^[A-Za-z0-9_-]{8,64}$`, 400 on malformed), every history/memory query is scoped to it, and other devices' rows are unreachable. The service-role key never leaves the server; RLS is enabled on all tables as defence in depth.
- **Transport hygiene**: security headers (nosniff, DENY framing, no-referrer), optional CORS locked to `ALLOWED_ORIGINS`, GZip for large bodies.
- **Abuse controls**: per-IP rate limit, 30MB body cap, image MIME/size/count validation, upstream concurrency cap, request-id correlated logs.
- **Prompt fencing**: web results and documents enter prompts as untrusted, clearly-marked blocks; the UI escapes all model output and allows only http(s) links.
- **Data control**: per-device export (markdown) and delete of single/all conversations and memories; memory is consent-gated — nothing is injected without approval.
- **Dependencies**: all pinned with `==`; `pip-audit` against `requirements.txt` reports **no known vulnerabilities** (Oct 2026).

## Limitations

- RAG DocStore is in-process (24h TTL, wiped on redeploy) — fine for the free tier, not a multi-instance database; pluggable for a later pgvector backend.
- No OCR — scanned/image-only PDFs are detected and reported, not parsed.
- Hash embeddings (no `OPENAI_API_KEY`) give keyword-adjacent retrieval only.
- Memory extractor is a deliberate lightweight heuristic — it suggests less rather than more; nothing is remembered without user approval.
- Code execution requires an external Piston-compatible sandbox (`ASTRA_SANDBOX_URL`); without it, the coding assistant works but cannot run code.
- Groq does not expose per-token pricing — token counts are tracked; cost math is left to your billing dashboard.

## Author

**Anand Kumar**
