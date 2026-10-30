"""Coding assistance for Astra (Phase 7).

Coding MODE is prompt-level: a dedicated system prompt guides explanation,
debugging (code + error + context), refactoring, examples and test
generation across the languages the configured model supports.

CODE EXECUTION is a separate concern (see execution.py) — a remote isolated
sandbox. Nothing here ever exec()s or eval()s user/model code in-process.

Honesty rule (enforced in the system prompt): the assistant must NEVER claim
code was executed or tested unless a real sandbox result is shown to it.
"""

import re

# Languages we surface in the UI (assistance works for all of these).
ASSIST_LANGS = (
    ("python", "Python"),
    ("c", "C"),
    ("cpp", "C++"),
    ("java", "Java"),
    ("javascript", "JavaScript"),
    ("typescript", "TypeScript"),
    ("html", "HTML"),
    ("css", "CSS"),
    ("sql", "SQL"),
    ("bash", "Bash"),
    ("go", "Go"),
    ("rust", "Rust"),
    ("php", "PHP"),
    ("ruby", "Ruby"),
    ("swift", "Swift"),
    ("kotlin", "Kotlin"),
)

# Heuristic markers that the last user message is a coding request
_CODE_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\s*\n")
_CODE_EXT_RE = re.compile(
    r"\b[\w-]+\.(py|js|ts|jsx|tsx|c|cc|cpp|h|hpp|java|sql|html|css|sh|go|rs|php|rb)\b"
)
_CODE_PHRASE_RE = re.compile(
    r"\b("
    r"explain (this|the following|the above) (code|snippet|function|script)|"
    r"debug (this|the following|my|the above)|"
    r"fix (this|the following|my) (code|bug|error|exception)|"
    r"refactor (this|the following|the above|my)|"
    r"write (unit |integration )?tests? (for|to test)|"
    r"generate (unit |integration )?tests? (for|to test)|"
    r"why (is|does) (this|my) (code|script|program) (fail|error|not work)|"
    r"traceback|stack trace|exception (in|thrown by)|"
    r"syntaxerror|typeerror|valueerror|referenceerror|indexerror|keyerror|"
    r"segfault|segmentation fault|compilation error|compile error|runtime error|"
    r"code review|review (this|the following|my) code|"
    r"translate (this )?code|convert (this )?(code|snippet) (to|into)"
    r")\b",
    re.IGNORECASE,
)

CODING_SYSTEM_PROMPT = (
    "You are Astra AI, an expert coding assistant. "
    "ALWAYS detect the language of the user's message and reply in EXACTLY that language "
    "(English, Hindi, Hinglish, Urdu, etc.) — but code identifiers, comments and strings "
    "inside code blocks stay in the code's own language/idiom. "
    "You help with: explaining code, debugging (accept code + error message + context), "
    "refactoring, writing runnable examples, and generating tests. "
    "Supported languages include Python, C, C++, Java, JavaScript, TypeScript, HTML/CSS, "
    "SQL, Bash, Go, Rust, PHP, Ruby, Swift, Kotlin — and anything else the model knows. "
    "When the user submits code for debugging, ALWAYS analyse the code AND the error together; "
    "if the error message or relevant context is missing, ask for exactly that (do not guess). "
    "Preserve exact code formatting: every code snippet goes in a ``` fenced block "
    "with the language tag (```python, ```cpp, ```sql, …). "
    "For test generation, produce complete, runnable test code (pytest, JUnit, Jest, etc. "
    "matching the language). "
    "HONESTY RULE: you have NO execution environment. NEVER claim code was executed, run, "
    "tested, or verified unless the user explicitly tells you they ran it (or shows you "
    "the output). Prefer 'this should work because…' over 'this works'. "
    "Keep prose concise; put the full solution code first, then at most 3 short steps."
)


def is_coding_request(text: str) -> bool:
    """Heuristic: does this user message look like a coding request?"""
    if not text:
        return False
    if _CODE_FENCE_RE.search(text):
        return True
    if _CODE_EXT_RE.search(text):
        return True
    if _CODE_PHRASE_RE.search(text):
        return True
    # dense code-ish signal: many punctuation tokens typical of source code
    if text.count("(") >= 2 and text.count(")") >= 2 and ("=" in text or ";" in text):
        return True
    return False


def extract_first_code_block(text: str) -> tuple[str, str] | None:
    """Return (language, code) of the first fenced block, or None."""
    m = _CODE_FENCE_RE.search(text or "")
    if not m:
        return None
    lang = (m.group(1) or "text").lower()
    start = m.end()
    end = text.find("```", start)
    if end == -1:
        return lang, text[start:].strip()
    return lang, text[start:end].strip()


def detect_request_language(text: str) -> str | None:
    """Best-effort language slug for the code in a message (for the Run button)."""
    block = extract_first_code_block(text or "")
    if block:
        lang = block[0]
        if lang in ("py", "python3"):
            return "python"
        if lang in ("js", "node", "mjs"):
            return "javascript"
        if lang in ("ts",):
            return "typescript"
        if lang in ("c++", "cxx"):
            return "cpp"
        if lang in ("sh", "shell"):
            return "bash"
        if lang in ("htm",):
            return "html"
        if lang in ("golang",):
            return "go"
        if lang in ("rs",):
            return "rust"
        if lang in ("rb",):
            return "ruby"
        return lang
    m = _CODE_EXT_RE.search(text or "")
    if m:
        ext = m.group(1).lower()
        return {
            "py": "python", "js": "javascript", "ts": "typescript",
            "jsx": "javascript", "tsx": "typescript",
            "c": "c", "cc": "cpp", "cpp": "cpp", "h": "c", "hpp": "cpp",
            "java": "java", "sql": "sql", "html": "html", "css": "css",
            "sh": "bash", "go": "go", "rs": "rust", "php": "php", "rb": "ruby",
        }.get(ext)
    return None
