"""Guest device identity: X-Device-Id header dependency.

The browser generates a UUID once (localStorage) and sends it on every API
call. That id is the only identity in the system — no login, no cookies.
Format matches the conversation-id regex: ^[A-Za-z0-9_-]{8,64}$
"""

import re

from fastapi import Header, HTTPException

DEVICE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def normalize_device_id(value) -> str | None:
    """Return a valid device id or None."""
    if not value:
        return None
    value = str(value).strip()
    return value if DEVICE_ID_RE.match(value) else None


def get_device_id(x_device_id: str | None = Header(default=None)) -> str:
    """Required device id — 400 when missing/invalid."""
    device_id = normalize_device_id(x_device_id)
    if device_id is None:
        raise HTTPException(
            status_code=400,
            detail="Missing or invalid X-Device-Id header.",
        )
    return device_id


def get_device_id_optional(x_device_id: str | None = Header(default=None)) -> str | None:
    """Optional device id — persistence is simply skipped when absent."""
    return normalize_device_id(x_device_id)
