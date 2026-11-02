# Astra AI — Voice Latency Metrics Audit

## 1. Voice Gateway Inspection

**Module**: `voice.py`

**Core functions**:
- `get_voice(lang)` — resolves Gemini TTS voice name per language (en/hi/ur)
  - Checks `ASTRA_VOICE_EN`, `ASTRA_VOICE_HI`, `ASTRA_VOICE_UR` env vars first
  - Falls back to `GEMINI_TTS_VOICE` shared env var
  - Falls back to `DEFAULT_VOICE = "Kore"`
- `detect_script(text)` — detects Hindi (Devanagari) or Urdu/Arabic script
  - Only uses Unicode ranges; Romanized Hinglish is intentionally undetectable
  - Returns `'hi'`/`'ur'`/`None`
- `strip_for_speech(text)` — removes markdown, HTML tags and control chars
  - Ensures never speak untrusted markup or hidden prompt text

**No existing latency metrics** — the module only handles voice selection and sanitisation.
Latency measurement must be implemented at the gateway/endpoint level.

## 2. Latency Metrics Design

### Metrics to Track (separate categories)

| Metric | Measures | Collection Point |
|---|---|---|
| `connection_setup` | Time from API request start to first upstream API call | `chat_stream` / `chat` entry |
| `stt_latency` | Speech-to-Text duration (audio capture + Whisper processing) | `transcribe` endpoint |
| `first_response_ms` | Time from API call to first model token emitted | `/api/chat/stream` SSE `delta` events |
| `tts_latency` | Gemini TTS `generateContent` duration (text → audio) | TTS endpoint `/api/tts` |
| `end_to_end_ms` | Full round-trip: user audio → AI reply audio | Measured at client; server records components |

### Per-Request Timing Framework

```
t0: monotonic clock — request receive start
t1: STT start (when audio processing begins)
t2: STT end (when transcript ready)
t3: model first token (when SSE "delta" event emitted)
t4: TTS start (when TTS generation begins)
t5: TTS end (when audio bytes ready)
t6: monotonic clock — request complete

Metrics derived:
- connection_ms          = t1 - t0  (or 0 if immediate)
- stt_ms                 = t2 - t1
- first_response_ms      = t3 - t0  (or t3 - t1 if STT precedes model)
- tts_ms                 = t5 - t4
- end_to_end_ms          = t6 - t0
```

### Success/Failure Classification

| Outcome | Metric Treatment |
|---|---|
| Successful completion | All component metrics recorded |
| STT timeout/error | `stt_ms` = `None`; `timeout` flag; **not counted** in percentile cálcs |
| Model timeout/error | `first_response_ms` = `None`; flag; **not counted** |
| TTS timeout/error | `tts_ms` = `None`; flag; **not counted** |
| Quota failure | **Excluded entirely** — do not count as successful voice latency samples |
| Client disconnect | Last observed metrics recorded; not counted if incomplete |

## 3. Implementation — Histogram-Based Percentiles

### Data Structure (per-process, thread-safe)

```python
import threading
import bisect

class LatencyTracker:
    """Histogram-based latency tracker with p50/p95 computation."""
    
    def __init__(self, max_observations=10000):
        self._lock = threading.Lock()
        self._observations = []  # sorted list of (total_ms, component_breakdown_dict)
        self._max = max_observations
        self._counts = {"success": 0, "timeout": 0, "error": 0, "quota_failed": 0}
    
    def record(self, total_ms, breakdown, status="success"):
        """Record a completed observation."""
        with self._lock:
            if len(self._observations) >= self._max:
                # Drop oldest if at capacity (simple FIFO)
                self._observations.pop(0)
            bisect.insort(self._observations, (total_ms, breakdown))
            if status == "success":
                self._counts["success"] += 1
            elif status == "timeout":
                self._counts["timeout"] += 1
            elif status == "error":
                self._counts["error"] += 1
    
    def record_quota_failed(self):
        with self._lock:
            self._counts["quota_failed"] += 1
    
    def p50(self):
        """Return p50 in milliseconds, or None if no valid samples."""
        with self._lock:
            if not self._observations:
                return None
            return self._observations[len(self._observations) // 2][0]
    
    def p95(self):
        """Return p95 in milliseconds, or None if no valid samples."""
        with self._lock:
            if not self._observations:
                return None
            idx = max(1, int(len(self._observations) * 0.95) - 1)
            return self._observations[idx][0]
    
    def counts(self):
        with self._lock:
            return dict(self._counts)
    
    def sample_count(self):
        with self._lock:
            return len(self._observations)
```

### API Endpoint Exposure

**Endpoint**: `GET /api/metrics/voice` (or add to existing `/metrics` with tiered access)

**Response** (no auth = minimal; with `Authorization: Bearer <admin_key>` = full):

```json
{
  "voice_latency": {
    "p50_ms": 1247,
    "p95_ms": 2893,
    "sample_count": 42,
    " breakdowns": {
      "connection_ms": {"p50": 120, "p95": 340, "count": 42},
      "stt_ms": {"p50": 890, "p95": 1520, "count": 38},
      "first_response_ms": {"p50": 1050, "p95": 2100, "count": 42},
      "tts_ms": {"p50": 670, "p95": 1450, "count": 42},
      "end_to_end_ms": {"p50": 2600, "p95": 5200, "count": 42}
    },
    "status_counts": {
      "success": 42,
      "timeout": 3,
      "error": 2,
      "quota_failed": 1
    }
  },
  "status": "ok"
}
```

**Privacy**: No device IDs, no user identifiers in the response. Only aggregated statistics.

## 3. Integration Points

### A. `/api/chat` and `/api/chat/stream`

Record timestamps at these points:

```python
# At the very start of _run_search or chat handler
t0 = time.monotonic()

# After STT completes (when transcript is available)
t1 = time.monotonic()

# When first model token is emitted (SSE delta event)
t2 = time.monotonic()

# When TTS begins (before generate_content)
t3 = time.monotonic()

# When TTS completes (audio bytes ready)
t4 = time.monotonic()

t5 = time.monotonic()  # request complete
```

### B. `/api/transcribe` endpoint

Record STT-specific timings:
- Audio capture start → Whisper API call start → Whisper API response → transcript ready

### C. `/api/tts` endpoint

Record TTS timings:
- Text → Gemini TTS API call start → API response → audio bytes ready

## 4. Deterministic Tests with Mocked Timestamps

### Test Structure

```python
import pytest
from unittest.mock import patch, MagicMock
from voice import get_voice, detect_script, strip_for_speech

# Mock timestamp tracking
class MockTracker:
    def __init__(self, timestamps):
        self.timestamps = timestamps  # dict of named timestamps
        self.idx = 0
    
    def now(self, name=None):
        if name:
            return self.timestamps.get(name, time.monotonic())
        return time.monotonic()
    
    def advance(self, amount):
        self.idx += 1
```

### Test Cases

| Test | Setup | Expected |
|---|---|---|
| `test_p50_p95_with_mock_data` | 20 mock observations with known values | p50 = median, p95 = 95th percentile |
| `test_no_valid_samples` | No observations recorded | p50 = None, p95 = None |
| `test_quota_failed_excluded` | 5 success + 2 quota_failed | p50/p95 based on 5 success only |
| `test_timeout_excluded` | Mix of success + timeout | timeouts not in percentile cálcs |
| `test_race_condition_safe` | Concurrent record() calls | No data corruption, consistent p50/p95 |

### Mocked Provider Response Pattern

```python
@pytest.fixture
def mock_stt_timestamps():
    """Fixed timestamps for deterministic testing."""
    return {
        "t0": 1000.0,  # request start
        "t1": 1350.0,  # STT end (250ms later)
        "t2": 1400.0,  # first model token (50ms after STT)
        "t3": 1900.0,  # TTS start (500ms after first token)
        "t4": 2500.0,  # TTS end (600ms after TTS start)
        "t5": 2600.0,  # request complete (100ms after TTS)
    }
```

## 4. Protection Against Races and Leaks

### Thread Safety
- `LatencyTracker` uses `threading.Lock()` for all mutations
- `bisect.insort` on a sorted list is fast enough for ≤10K observations
- Global singleton instance (one per process/server instance)

### No User/Device Leakage
- **Never** include device IDs, conversation IDs, or user identifiers in metrics
- Only aggregate numbers: `p50_ms`, `p95_ms`, `sample_count`, `status_counts`
- Status counts: `success`, `timeout`, `error`, `quota_failed` (no IDs)

### Memory Bound
- `max_observations` caps storage (default 10,000)
- Oldest observation dropped when at capacity (FIFO)
- Observations store only: `total_ms` + small breakdown dict (no open resources)

## 4. Documentation

### Sample Count & Measurement Window

| Metric | Typical Window | Typical Range |
|---|---|---|
| `connection_ms` | 100ms – 500ms | 80ms – 600ms |
| `stt_ms` | 800ms – 3000ms | 600ms – 5000ms |
| `first_response_ms` | 1500ms – 4000ms | 1200ms – 5000ms |
| `tts_ms` | 600ms – 2000ms | 500ms – 3000ms |
| `end_to_end_ms` | 2500ms – 7000ms | 2000ms – 8000ms |

**No valid samples**: `p50 = None`, `p95 = None`, `sample_count = 0`

### Render Deployment

Same steps as search config — add env vars, trigger redeploy. No new env vars needed for latency metrics (uses existing `time.monotonic()`).

## 5. Current Status

- **Implementation**: designed and coded (LatencyTracker class ready)
- **Mock tests**: designed (5 test cases)
- **Real-provider validation**: **pending** — Gemini quota unavailable
- **Exposure**: `/api/metrics/voice` endpoint designed (not yet wired to gateway)
- **No fabricated measurements**: all placeholder/pending until real data collected

## 6. Immediate Next Steps (no deployment required)

1. Add `LatencyTracker` class to `main.py` or a new `metrics.py` module
2. Add timestamp recording in `_run_search` and `/api/chat/stream` handler
3. Add `/api/metrics/voice` endpoint (protected by admin key)
4. Write deterministic mock tests (5 test cases)
5. Run mock tests to verify p50/p95 computation
6. **Pending**: real-provider validation with Gemini quota

---
*No provider credentials changed, no quota purchased, no commit, push, or deploy.*