"""Code-execution sandbox client (Phase 7).

Design rules (non-negotiable):
- NOTHING is exec()'d / eval()'d inside the FastAPI process. All execution
  happens in a separate, isolated service reached over HTTP.
- Execution is DISABLED unless a sandbox URL is explicitly configured
  (ASTRA_SANDBOX_URL). There is no in-process fallback.
- Source code is untrusted: we validate language allowlist, size limits,
  time limits and output limits before returning anything to the client.
- We never forward host env vars / secrets to the sandbox. The request body
  carries ONLY {language, version, files, stdin} — nothing else.
- The sandbox itself must run non-root, network-disabled, read-only base FS.
  We document those requirements; we cannot enforce them from this client,
  so the operator chooses a reputable provider (Piston, Judge0, a locked-down
  container service…). If none is configured, the Run button is hidden and
  /api/execute returns 503.

Supported provider today: any Piston-compatible HTTP API
(https://github.com/engineer-man/piston — POST /api/v2/execute).
"""

import asyncio
import logging
import os

import httpx

logger = logging.getLogger("astra.execution")

# Explicit allowlist — nothing outside this dict can be executed.
# version="*" lets the sandbox pick its installed version; operators can pin.
LANGUAGE_ALLOWLIST: dict[str, str] = {
    "python": "*",
    "c": "*",
    "cpp": "*",
    "java": "*",
    "javascript": "*",
    "typescript": "*",
    "go": "*",
    "rust": "*",
    "php": "*",
    "ruby": "*",
    "bash": "*",
}

CODE_MAX_BYTES = 64 * 1024        # 64 KB of source per run
STDIN_MAX_BYTES = 16 * 1024       # 16 KB of stdin
TIME_LIMIT_MS = int(os.getenv("ASTRA_SANDBOX_TIME_MS", "3000"))    # wall-clock run cap
OUTPUT_MAX_CHARS = 10_000         # truncate stdout/stderr per stream
SANDBOX_TIMEOUT_SEC = float(os.getenv("ASTRA_SANDBOX_TIMEOUT", "15"))  # HTTP timeout to sandbox


class SandboxError(Exception):
    """Sandbox infrastructure failure (network, bad response, …)."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def sandbox_enabled() -> bool:
    """True only when an operator has explicitly configured a sandbox URL."""
    return bool((os.getenv("ASTRA_SANDBOX_URL") or "").strip())


def allowed_languages() -> list[str]:
    return sorted(LANGUAGE_ALLOWLIST.keys())


def _validate(language: str, code: str, stdin: str) -> tuple[str, str, str]:
    lang = (language or "").strip().lower()
    if lang not in LANGUAGE_ALLOWLIST:
        raise SandboxError(
            f"Language '{lang}' is not allowed. Allowed: {', '.join(allowed_languages())}.",
            status=400,
        )
    if not isinstance(code, str) or not code.strip():
        raise SandboxError("No code provided.", status=400)
    raw = code.encode("utf-8", errors="replace")
    if len(raw) > CODE_MAX_BYTES:
        raise SandboxError(
            f"Code too large (max {CODE_MAX_BYTES // 1024}KB).", status=413
        )
    stdin_raw = (stdin or "").encode("utf-8", errors="replace")
    if len(stdin_raw) > STDIN_MAX_BYTES:
        raise SandboxError(
            f"stdin too large (max {STDIN_MAX_BYTES // 1024}KB).", status=413
        )
    return lang, code, stdin or ""


def _truncate(text: str) -> str:
    if len(text) <= OUTPUT_MAX_CHARS:
        return text
    return text[:OUTPUT_MAX_CHARS] + f"\n… [truncated, {len(text)} chars total]"


async def execute_code(language: str, code: str, stdin: str = "") -> dict:
    """Run `code` in the remote sandbox. Raises SandboxError on any failure.

    Returns {stdout, stderr, exit_code, language, version, timed_out}.
    """
    if not sandbox_enabled():
        raise SandboxError(
            "Code execution is not configured on this server. "
            "Coding help works without it — ask Astra to explain, debug, "
            "refactor or write tests instead.",
            status=503,
        )

    lang, code, stdin = _validate(language, code, stdin)
    url = (os.getenv("ASTRA_SANDBOX_URL") or "").strip().rstrip("/")
    api_key = (os.getenv("ASTRA_SANDBOX_KEY") or "").strip()

    payload = {
        "language": lang,
        "version": LANGUAGE_ALLOWLIST[lang],
        "files": [{"content": code}],
        "stdin": stdin,
        "compile_timeout": TIME_LIMIT_MS,
        "run_timeout": TIME_LIMIT_MS,
        "compile_memory_limit": -1,
        "run_memory_limit": -1,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        # only a provider-supplied key — never host env secrets
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=SANDBOX_TIMEOUT_SEC) as client:
            resp = await client.post(f"{url}/api/v2/execute", json=payload, headers=headers)
    except httpx.TimeoutException:
        logger.warning("sandbox timeout lang=%s", lang)
        raise SandboxError("Execution timed out.", status=504)
    except httpx.HTTPError as error:
        logger.warning("sandbox unreachable: %s", error)
        raise SandboxError("Sandbox service unreachable.", status=502)

    if resp.status_code >= 500:
        logger.warning("sandbox 5xx: %s", resp.status_code)
        raise SandboxError("Sandbox service error.", status=502)
    if resp.status_code == 429:
        raise SandboxError("Sandbox rate limit — try again shortly.", status=429)
    if resp.status_code >= 400:
        # e.g. unknown version, compile failure — surface as a clean client error
        try:
            detail = resp.json().get("message") or resp.json().get("error") or ""
        except Exception:
            detail = resp.text[:200]
        raise SandboxError(detail or "Sandbox rejected the request.", status=400)

    try:
        data = resp.json()
    except Exception:
        raise SandboxError("Sandbox returned an invalid response.", status=502)

    run = data.get("run") or {}
    compile_stage = data.get("compile") or {}
    # compile errors surface as non-zero compile.code with useful stderr
    if compile_stage.get("code") not in (None, 0):
        stderr = _truncate(str(compile_stage.get("stderr") or compile_stage.get("output") or ""))
        return {
            "stdout": "",
            "stderr": stderr or "Compilation failed.",
            "exit_code": int(compile_stage.get("code") or 1),
            "language": lang,
            "version": str(data.get("version", "")),
            "timed_out": False,
        }

    exit_code = run.get("code")
    if exit_code is None:
        exit_code = 0
    return {
        "stdout": _truncate(str(run.get("stdout") or "")),
        "stderr": _truncate(str(run.get("stderr") or "")),
        "exit_code": int(exit_code),
        "language": lang,
        "version": str(data.get("version", "")),
        "timed_out": bool(run.get("signal") == "SIGKILL"),
    }
