"""Honest latency measurement against a running Astra instance.

Usage:
    python scripts/measure_latency.py [BASE_URL]

Requires a live server (e.g. uvicorn on port 8010) with a real GROQ_API_KEY
for the chat measurements. Reports min / p50 / p95 / mean and error counts —
real numbers only, nothing is simulated.
"""
import json
import os
import statistics
import sys
import time

import httpx

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8010").rstrip("/")
N_SLOW = int(os.getenv("BENCH_N_STATIC", "20"))   # health / models / index
N_CHAT = int(os.getenv("BENCH_N_CHAT", "5"))      # real model calls
CHAT_PROMPT = os.getenv("BENCH_PROMPT", "Reply with exactly: bench ok")


def stats(samples: list[float], errors: int) -> str:
    if not samples:
        return f"no successful samples ({errors} errors)"
    s = sorted(samples)
    p95 = s[min(len(s) - 1, int(0.95 * len(s)))]
    return (
        f"n={len(s)} err={errors} "
        f"min={min(s)*1000:.1f}ms p50={statistics.median(s)*1000:.1f}ms "
        f"p95={p95*1000:.1f}ms mean={statistics.mean(s)*1000:.1f}ms"
    )


def bench_static(path: str, n: int, json_body: dict | None = None) -> str:
    ok, errs = [], 0
    with httpx.Client(timeout=60) as c:
        for _ in range(n):
            t0 = time.perf_counter()
            try:
                r = c.post(path, json=json_body) if json_body else c.get(path)
                if r.status_code == 200:
                    ok.append(time.perf_counter() - t0)
                else:
                    errs += 1
                    print(f"    ! {path} -> {r.status_code}: {r.text[:120]}")
            except Exception as e:
                errs += 1
                print(f"    ! {path} -> {type(e).__name__}: {e}")
    return stats(ok, errs)


def bench_stream(n: int) -> str:
    """Measures time-to-first-delta + total stream duration."""
    ttft, total, errs = [], [], 0
    body = {"messages": [{"role": "user", "content": CHAT_PROMPT}]}
    with httpx.Client(timeout=120, base_url=BASE) as c:
        for _ in range(n):
            t0 = time.perf_counter()
            got_first = False
            try:
                with c.stream("POST", "/api/chat/stream", json=body) as r:
                    if r.status_code != 200:
                        errs += 1
                        print(f"    ! stream -> {r.status_code}")
                        continue
                    for line in r.iter_lines():
                        if line.startswith("data:") and '"text"' in line:
                            if not got_first:
                                ttft.append(time.perf_counter() - t0)
                                got_first = True
                        if line.startswith("event: done"):
                            break
                if got_first:
                    total.append(time.perf_counter() - t0)
                else:
                    errs += 1
                    print("    ! stream: no delta received")
            except Exception as e:
                errs += 1
                print(f"    ! stream -> {type(e).__name__}: {e}")
    return (
        f"TTFT {stats(ttft, errs)} | full {stats(total, 0)}"
        if ttft
        else f"no TTFT samples ({errs} errors)"
    )


def main() -> int:
    print(f"Astra latency bench against {BASE}  (static n={N_SLOW}, chat n={N_CHAT})")
    with httpx.Client(timeout=15) as c:
        try:
            r = c.get(f"{BASE}/health")
            print(f"  /health reachable: {r.status_code}")
        except Exception as e:
            print(f"  FATAL: server not reachable: {type(e).__name__}: {e}")
            return 1

    lines = []
    with httpx.Client(timeout=60, base_url=BASE) as c:
        for label, path, body in [
            ("GET  /health", "/health", None),
            ("GET  /", "/", None),
            ("GET  /api/models", "/api/models", None),
            ("POST /api/chat (real model)", "/api/chat",
             {"messages": [{"role": "user", "content": CHAT_PROMPT}]}),
        ]:
            ok, errs = [], 0
            n = N_CHAT if "chat" in label else N_SLOW
            for i in range(n):
                t0 = time.perf_counter()
                try:
                    r = c.post(path, json=body) if body else c.get(path)
                    if r.status_code == 200:
                        ok.append(time.perf_counter() - t0)
                    else:
                        errs += 1
                        print(f"    ! {path} #{i} -> {r.status_code}: {r.text[:160]}")
                except Exception as e:
                    errs += 1
                    print(f"    ! {path} #{i} -> {type(e).__name__}: {e}")
            line = f"  {label:28s} {stats(ok, errs)}"
            print(line)
            lines.append(line)

        print("  POST /api/chat/stream (real model)")
        line = f"    {bench_stream(N_CHAT)}"
        print(line)
        lines.append(line)

    out = os.getenv("BENCH_OUT")
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(lines, f, indent=2)
        print(f"  saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
