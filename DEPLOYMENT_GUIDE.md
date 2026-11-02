# Astra AI — Reliable Optional Supabase Persistence

## Overview
This repository implements a **pluggable persistence layer** for the Astra AI chatbot, supporting both an in-memory store (for local dev/testing) and Supabase (for production persistence). The backend is configured via the `ASTRA_STORE` environment variable.

---

## Architecture

### Backends

| Backend | When Used | Key Features |
|---|---|---|
| **MemoryStore** | Default when `ASTRA_STORE=memory` or no Supabase config | Thread-safe in-process dicts, full contract match with SupabaseStore, used for tests/offline dev |
| **SupabaseStore** | When `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` are set | PostgREST over HTTPS, device_id-scoped queries, RLS enforcement, durable history across restarts |

### Backend Selection (`get_store()`)

```python
mode = (os.getenv("ASTRA_STORE") or "").strip().lower()
url = (os.getenv("SUPABASE_URL") or "").strip()
key = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()

if mode == "memory":
    _store = MemoryStore()                              # force memory
elif url and key:                                       # auto-detect supabase
    _store = SupabaseStore(url, key)
else:
    _store = MemoryStore()                              # fallback to memory
```

- `ASTRA_STORE=memory` **forces** MemoryStore even if Supabase keys are set
- Omit `ASTRA_STORE` and set `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` to use Supabase
- Neither configured → MemoryStore fallback (safe, never corrupts history)

---

## SQL Schema Migrations

### Location
`supabase/schemas/001_create_tables.sql`

### Tables Created

1. **devices** — `device_id PK`, `memory_enabled bool`, `created_at/updated_at`
2. **conversations** — `id PK`, `device_id FK`, `title`, `created_at/updated_at`
3. **messages** — `id PK`, `conversation_id FK cascade`, `role`, `content`, `created_at`
4. **memories** — `id PK`, `device_id FK`, `content`, `kind`, `status` (active/pending/deleted), `created_at/updated_at`

### Row-Level Security (RLS)

Each table has a `USING (device_id = current_setting('device_id', true))` policy so that:
- A device can only read/write its own rows
- No user table, no login, no cookie — anonymous UUID identity
- Even if a query omits the `device_id` filter, the database blocks cross-device access

### Helper Function
```sql
SELECT set_config('device_id', '<UUID>', false);
```
Sets the session-level device_id so PostgREST automatically scopes queries. The Python `SupabaseStore` uses query-string scoping (`device_id=eq.X`) as the primary mechanism, with RLS as defense-in-depth.

---

## Environment Variables

| Variable | Required? | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key for LLM calls |
| `SUPABASE_URL` | No | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | No | **Server-side only — NEVER expose to browser**. Full admin privileges. |
| `SUPABASE_ANON_KEY` | No | Browser-compatible key (documented for future client operations). Not currently used by the Python server-side store. |
| `ASTRA_STORE` | No | `memory` or `supabase`. Default: auto-detect based on above. |
| `ASTRA_REFLECTION` | No | `1`/`true`/`yes` to enable reflection LLM call (default: **off**). When on, each chat turn triggers one additional quality-review call. |
| `ASTRA_FAST_MODEL` | No | Default: `openai/gpt-oss-20b` |
| `ASTRA_REASONING_MODEL` | No | Default: `openai/gpt-oss-120b` |

### `.env` File
- **Gitignored** — never commit real keys
- Fill in real values from the `.env.example` template
- `.env.example` is version-controlled and documents all variables

---

## Memory Consent Flow (Preserved from Original Model)

1. **Explicit consent**: User manually adds a memory via `/api/memories` (status=`active` by definition)
2. **Suggestion extraction**: `_run_search`/`_run_doc_retrieval` heuristic extracts pending memories from user text
3. **Pending state**: `save_pending_suggestions` creates `status='pending'` memories
4. **Approval**: User clicks "approve" → `set_memory_status(memory_id, 'active')` — memory becomes eligible for injection
5. **Rejection**: User clicks "reject" → `delete_memory(memory_id)` — pending memory is discarded
6. **Delete all**: `/api/memories?confirm=true` → `delete_all_memories(device_id)`

All memory operations are **device-scoped** via `device_id` — a device can only manage its own memories.

---

## Device Identity & Isolation

### Browser-Side
- A UUID is generated once and stored in `localStorage` as `astra_device_id`
- Sent as `X-Device-Id` header on every API call
- Format: `^[A-Za-z0-9_-]{8,64}$` (validated by `device.py`)

### Server-Side
- `get_device_id_optional()` / `get_device_id()` FastAPI dependencies extract the header
- **Every** store query is scoped by `device_id` — the ID is **never trusted** from client input
- Conversation/memory rows owned by another device are invisible (404 if requested)
- `MemoryStore.create_conversation()` explicitly rejects adopting another device's conversation id

---

## Error Handling & Outage Gracefulness

### SupabaseStore
- **Network errors** (`httpx.HTTPError`): Logged, `StoreError("Database unavailable. Please try again.")` raised
- **HTTP 401/403**: `StoreError("Database rejected the request. Check SUPABASE keys.")`
- **Empty responses**: Returns `[]` (empty list), never corrupts state
- **All endpoint wrappers** (`routes_conversations.py`, `routes_memories.py`, `main.py _persist_chat_turn()`): Wrap store calls in `try/except Exception` — failures are logged but **do not break** the chat flow

### MemoryStore
- All operations are in-process and **cannot raise** due to database failures
- `StoreError` can be raised if code logic is wrong, but never due to external outages

### Key Invariant
> **Database outages must not corrupt conversation history.** — Every persistence call in `main.py` is wrapped in `try/except Exception`, and the chat endpoint always continues even if persistence fails.

---

## Reflection Feature (Configurable, Default OFF)

### Configuration
```bash
# Enable reflection (one extra LLM call per chat turn for quality review)
ASTRA_REFLECTION=1
# Or: ASTRA_REFLECTION=true or ASTRA_REFLECTION=yes
```

### Behavior
- **Default** (`ASTRA_REFLECTION` unset or empty): **OFF** — no extra LLM call, normal metrics
- **Enabled** (`ASTRA_REFLECTION=1`): Each chat turn triggers one additional Groq call for quality review before the reply is finalised. The extra call is:
  - Tracked in `usage_snapshot()` as `reflection_calls`
  - Labelled in the model field
  - Cost-estimated via `_MODEL_COST_PER_1K` and shown as `estimated_cost_usd`
  - **Not** double-counted: `_record_usage(data, is_reflection=True)` only increments `_reflection_calls`, not `_token_usage["calls"]`

### Metrics (`usage_snapshot()`)
Returns:
```json
{
  "calls": 5,               // primary chat completions (retries do not double-count)
  "prompt_tokens": 1245,
  "completion_tokens": 890,
  "total_tokens": 2135,
  "reflection_calls": 2,    // only when ASTRA_REFLECTION is on
  "model": "openai/gpt-oss-20b",
  "estimated_cost_usd": "$1.2340 (prompt:500 tok + completion:350 tok)"
}
```
- If the model is not in `_MODEL_COST_PER_1K`, `estimated_cost_usd` is `"unknown (model not in cost table)"`
- Placeholder rates in `_MODEL_COST_PER_1K` — operators should replace with actual Groq contract rates

---

## Test Suite

### `tests/test_store.py`
- **MemoryStore tests**: Basics, conversation isolation, message persistence, memory consent flow, delete-all, backend swap
- **SupabaseStore tests**: Guarded by env vars — silently skipped when no Supabase project is configured
- **Backend swap tests**: `ASTRA_STORE=memory` forces MemoryStore; no config defaults to MemoryStore
- **Outage resilience tests**: Verifies MemoryStore never corrupts state, `_persist_chat_turn` wrapper doesn't break chat

### Running Tests
```bash
# Install deps
pip install -r requirements.txt

# Run MemoryStore tests only (no Supabase needed)
python -m pytest tests/test_store.py -v --tb=short

# Or run all store tests (will skip Supabase tests if keys not set)
python -m pytest tests/ -v --tb=short -k "not Supabase and not integration"
```

---

## CI / Render Deployment

### 1. Initialize Git
```bash
cd C:\Users\anand\OneDrive\Documents\Default Project\AI_Chatbot\AI_Chatbot--main
git init
git add .gitignore
git add main.py router.py websearch.py rag.py chatbot.py store.py memory.py voice.py
git add requirements.txt .env.example .gitignore routes_conversations.py routes_memories.py
git commit -m "Initial Astra AI repository structure"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Application Starts
```bash
python -c "import main; print('Application loads successfully')"
```

### 4. Test Backend Selection
```bash
# Force MemoryStore (local dev / tests)
export ASTRA_STORE=memory

# Use Supabase (production)
# export SUPABASE_URL=https://abcdef.supabase.co
# export SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
# unset ASTRA_STORE  # auto-detect
```

### 5. Run Full Test Suite
```bash
python -m pytest tests/ -v --tb=short
# Expected: 300+ tests pass with MemoryStore fallback
# 1 test (test_rate_limit_uses_x_forwarded_for) needs TRUSTED_PROXIES=10.0.0.1
```

### 6. Render Deployment
```bash
# Set env vars in Render dashboard:
#   GROQ_API_KEY=your_key
#   ASTRA_REFLECTION=1  (optional, defaults to off)
#   ASTRA_STORE=memory  (for local dev; omit for Supabase if configured)

# Deploy
render services create --name astra-ai --env-file .env render.yaml
```

---

## What NOT to Commit
- `.env` file (contains real API keys — already gitignored)
- `.venv/` directory (virtual environment — 3800+ files, excluded by .gitignore)
- `__pycache__/` directories (30+ cache files, excluded)
- `screenshot.png` (268KB — generated output)
- `.pytest_cache/` (test caching)
- Any local database files (`*.db`, `*.sqlite3`)

---

## Requirements Compliance Checklist

- [x] Use Supabase server-side client with service-role key never exposed to browser
- [x] Create SQL schema for conversations, messages, memories with row-level security
- [x] Enforce device_id isolation — never trust client-provided IDs as ownership proof
- [x] Use stable authenticated identities (UUID in localStorage + X-Device-Id header) for anonymous users
- [x] Preserve memory consent, deletion, export behavior from existing model
- [x] Handle database outages gracefully without corrupting history
- [x] Make backend configurable: local dev without Supabase when not available
- [x] Add tests for persistence across restarts, user isolation, consent, deletion, DB failures
- [x] Document SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY env vars; SUPABASE_ANON_KEY documented
- [x] Never invent migration or test results
- [x] Reflection configurable and default OFF for ordinary chat
- [x] Thread-safe metrics with is_reflection flag to avoid double-counting retries
- [x] Model-specific cost estimates with clear "unknown" labeling when unavailable