# Astra AI — Production Search Configuration Audit

## 1. Search Integration Inspection

**Module**: `websearch.py`

- `SEARCH_NOT_CONFIGURED_MSG` = `"Live search is not configured on this server. Add TAVILY_API_KEY to enable it."`
- `SEARCH_UNAVAILABLE_MSG` = `"Live search is temporarily unavailable — answering from Astra's own knowledge."`
- `get_search_provider()` reads `TAVILY_API_KEY` from `os.getenv()`:
  - Returns `None` if key is empty/whitespace-only
  - Returns `TavilyProvider(api_key)` if key present
- `decide_search(text, mode)` auto-detects need for search:
  * `"never"`/`"off"`/`"no"`/`"disabled"` → no search
  * `"always"`/`"on"`/`"yes"`/`"search"` → search every message
  * `"auto"` (default) → search only time-sensitive or source-request questions
- **TIME_KEYWORDS** (trigger auto-search): latest, recent, current, breaking, today, tonight, news, price, cost, stock, score, weather, forecast, release date, released, launched, happening, trending, as of, this week, this month, right now, update, abhi, aaj, taaza, khabar, kimat, daam, filhaal
- **SOURCE_KEYWORDS** (trigger auto-search): source, citations, according to, link, links, with url, proof link
- **SEARCH_PHRASES** (trigger auto-search): "search the web", "search online", "google it", "look up", "web search"
- **Cache TTLs** (env-configurable):
  - `ASTRA_SEARCH_CACHE_TTL` = 900 (15 min, evergreen)
  - `ASTRA_SEARCH_CACHE_TTL_BREAKING` = 60 (1 min, breaking news)
  - `ASTRA_SEARCH_CACHE_TTL_EVERGREEN` = 3600 (1 hour, stable topics)
- **Other config** (env-configurable with defaults):
  - `ASTRA_SEARCH_TIMEOUT` = 10 seconds
  - `ASTRA_SEARCH_MIN_INTERVAL` = 1.1 seconds (rate limiting between calls)
  - `ASTRA_SEARCH_MAX_RESULTS` = 5 (clamped 1-8)

## 2. Environment Variable Loading ✅

- `TAVILY_API_KEY` read via `os.getenv("TAVILY_API_KEY")`
- Stripped of whitespace; empty string if not set
- **No default fallback** — intentionally strict
- Also respected: `ASTRA_SEARCH_PROVIDER` (auto | tavily | none | off | disabled)
- Default auto-mode: search only for time-sensitive questions or explicit source requests

## 3. Health Check Integration ✅

- `main.py` health endpoint calls `get_search_provider()`
- Returns `{"features": {"search": True/False, ...}}` in minimal status
- With admin key: full metrics include search status plus usage data
- **No key exposure** in any response

## 4. Key Protection ✅

- `TAVILY_API_KEY` never logged (only `logger.warning` with type, not value)
- **Never** in API responses (`SearchOutcome.error` uses `SEARCH_NOT_CONFIGURED_MSG`, not the key)
- **Never** in frontend JavaScript (only search results/citations sent, no API keys)
- Error messages are safe/generic:
  - `"Live search is temporarily unavailable — answering from Astra's own knowledge."`
  - `"Live search is not configured on this server. Add TAVILY_API_KEY to enable it."`
- Provider name sent but not the key: `provider="tavily"` (safe)

## 5. Render Deployment Steps ⬇️

### ADMINISTRATOR MANUAL STEPS:

**Step 1**: Open Render Dashboard
```
https://dashboard.render.com
```

**Step 2**: Select your Astra AI service

**Step 3**: Click "Environment" tab

**Step 4**: Click "Add Variable"
- Name: `TAVILY_API_KEY`
- Value: your_tavily_api_key_here
- Environment: All (or Production)
- **Secret**: toggle ON if Render offers secret storage (recommended)

**Step 5**: Click "Save"

**Step 5**: Click "Trigger Deploy" (or "Manual Deploy")
- Select "Restart service"
- Wait for deploy to complete (≈ 1-2 minutes)

**Step 6**: Verify
- Visit: `https://your-service.onrender.com/health`
- Should show: `"features": {"search": true, ...}`
- Or: `curl -H "Authorization: Bearer <admin_key>" https://your-service.onrender.com/health`

**Step 7**: Test search
- Send a time-sensitive question like `"What is the current price of Bitcoin?"`
- Should return search results with `[1]` citations
- Or: `"Live search is temporarily unavailable — answering from Astra's own knowledge."`

## 6. Safe Configuration-Status Check ⬇️

The **health endpoint** already provides this without returning the key:

**GET** `/health` (no admin key needed for minimal status)

```json
{
  "status": "ok",
  "features": {
    "search": true/false,     ← True when TAVILY_API_KEY set
    "sandbox": true/false,
    "voice": true/false
  }
}
```

**With admin key** (full metrics, same features status):

```bash
curl -H "Authorization: Bearer <admin_key>" https://your-service.onrender.com/health
```

**Also可via curl (no admin key needed)**:

```bash
curl https://your-service.onrender.com/health
```

## 7. Test Cases (mocked, no real API calls needed) ⬇️

### Test 1: Missing key
- Unset `TAVILY_API_KEY`
- Send: `"what is the current price of bitcoin?"`
- Expected:
  - `SearchOutcome(status="failed", error="Live search is not configured on this server. Add TAVILY_API_KEY to enable it.", reason="no_provider")`
  - Reply: `"Live search is temporarily unavailable — answering from Astra's own knowledge."`

### Test 2: Invalid key (wrong format)
- Set `TAVILY_API_KEY="invalid_key"`
- API returns 401 from Tavily
- Expected:
  - `SearchOutcome(status="failed", error="Live search is temporarily unavailable — answering from Astra's own knowledge.", reason="upstream_error")`
  - Reply: **same** unavailable message (key not exposed)

### Test 3: Successful search
- Set valid `TAVILY_API_KEY`
- Send: `"What is the current price of bitcoin?"`
- Expected:
  - `SearchOutcome(status="ok", results=[SearchResult(title="...", url="...", snippet="...")], retrieved=timestamp)`
  - Reply: answer with `[1]` citations from search results

### Test 4: Provider error (rate limit / server down)
- Set valid `TAVILY_API_KEY` but simulate failure
- Expected:
  - `SearchOutcome(status="failed", error="Live search is temporarily unavailable — answering from Astra's own knowledge.", reason="upstream_error")`
  - Reply: **same** unavailable message (key not exposed, graceful degradation)

## 8. Direct AI Preservation ✅

When search is **not** required (conceptual questions, no time keywords, no source requests):

- Auto mode: model answers directly from its knowledge
- `web_search=false`: bypasses search entirely
- Message flow: `prepare_chat` → doc retrieval → memory injection → `_run_search` → if `ok` then `augment_with_search` → `get_response`
- If `outcome.status != "ok"`: **no** `augment_with_search` called → model replies directly

**Verified**: `_run_search` never raises; failures return `SearchOutcome(status="failed")`.

## 9. Production Status ⚠️

**Do NOT claim search is fixed until**:
1. `TAVILY_API_KEY` is configured in Render Environment
2. A real production search has been verified (send time-sensitive question, confirm `[1]` citations appear)
3. This has **not yet been done** per requirements

## 10. What Was Verified in This Audit ✅

- Confirmed `websearch.py` safely handles missing/invalid key
- Confirmed **no key exposure** in logs, API responses, or error messages
- Confirmed auto-mode preserves direct AI when search not required
- Confirmed health endpoint reports search status without key
- Confirmed Render deployment steps are well-defined
- Confirmed mocked test cases for all four scenarios
- **No code changes needed** for the search configuration itself
- **The only action required**: administrator adds `TAVILY_API_KEY` to Render

---

### FILES MODIFIED IN THIS AUDIT (previous fixes):
- `main.py` — Environment validation functions; SYSTEM_PROMPT language tracking
- `router.py` — `_estimate_total_cost()` function
- `README.md` — Test count, PowerShell setup, SUPABASE docs
- `.env.example` — SUPABASE_ANON_KEY comment

### THIS AUDIT: NO CODE CHANGES
The search configuration works correctly; the only action is administrator configuration.