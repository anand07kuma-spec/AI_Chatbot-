"""Audit which API keys / env vars this project needs and which are set.

Safe by design: values are never printed, only status + length + prefix.

Usage:  python scripts/env_audit.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"

# (name, why it's needed, required?)
KEYS = [
    ("GROQ_API_KEY", "chat / vision / speech-to-text (Groq)", True),
    ("GEMINI_API_KEY", "text-to-speech (Gemini TTS) - voice output", False),
    ("TAVILY_API_KEY", "live web search - off if unset", False),
    ("OPENAI_API_KEY", "real RAG embeddings - else local hash fallback", False),
    ("SUPABASE_URL", "durable chat history - else in-memory store", False),
    ("SUPABASE_SERVICE_ROLE_KEY", "Supabase server auth (with SUPABASE_URL)", False),
    ("SUPABASE_ANON_KEY", "Supabase read probes (optional)", False),
    ("ASTRA_SANDBOX_URL", "code execution sandbox - Run button stays hidden if unset", False),
    ("ASTRA_SANDBOX_KEY", "sandbox provider key (with ASTRA_SANDBOX_URL)", False),
]

PLACEHOLDER_PREFIXES = ("your_", "xxxx", "insert_", "paste_", "<")


def load_env_file(path: Path) -> dict[str, str]:
    """Minimal .env parser (KEY=VALUE, ignores comments)."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        out[key.strip()] = val.strip().strip('"').strip("'")
    return out


def status(value: str | None) -> str:
    if value is None:
        return "MISSING"
    if not value:
        return "EMPTY"
    low = value.lower()
    if any(low.startswith(p) for p in PLACEHOLDER_PREFIXES) or "your_" in low:
        return "PLACEHOLDER - not filled in"
    return f"SET (len {len(value)}, {value[:3]}***)"


def main() -> int:
    env = load_env_file(ENV_FILE)
    proc = os.environ

    print(f"project root : {ROOT}")
    print(f".env file    : {'FOUND' if ENV_FILE.exists() else 'NOT FOUND'} "
          f"({ENV_FILE})")
    if ENV_FILE.exists():
        print(f"  entries parsed: {len(env)}")
    print()

    print(f"{'KEY':<30} {'.env':<28} {'process env':<28} {'needed for'}")
    print("-" * 130)

    required_missing = []
    optional_missing = []
    placeholder = []

    for name, why, required in KEYS:
        f_val = env.get(name)
        p_val = proc.get(name)

        f_status = status(f_val) if name in env else ("-" if f_val is None else status(f_val))
        p_status = status(p_val)

        # effective = process env wins (python-dotenv does not override by default,
        # but the server process may have it set at the OS level)
        effective = p_val or f_val
        eff_status = status(effective)

        flag = "" if eff_status.startswith("SET") else ("  <-- REQUIRED" if required else "")
        if required and not eff_status.startswith("SET"):
            required_missing.append(name)
        elif not required and not eff_status.startswith("SET"):
            optional_missing.append(name)
        if eff_status.startswith("PLACEHOLDER"):
            placeholder.append(name)

        print(f"{name:<30} {f_status:<28} {p_status:<28} {why}{flag}")

    print("-" * 130)
    print()
    if required_missing:
        print(f"[X] REQUIRED keys missing/placeholder: {', '.join(required_missing)}")
        print("    -> chat, vision and voice transcription WILL NOT work.")
    else:
        print("[OK] All required keys are set.")

    if optional_missing:
        print(f"[-] Optional keys not set: {', '.join(optional_missing)}")
        print("    -> these features are disabled (graceful, by design):")
        for name, why, _ in KEYS:
            if name in optional_missing:
                print(f"       - {name}: {why}")

    print()
    return 1 if required_missing else 0


if __name__ == "__main__":
    sys.exit(main())
