from __future__ import annotations

import secrets

from fastapi import Header, HTTPException

from app.config import get_api_key


def require_auth_if_configured(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    configured_key = get_api_key()
    if not configured_key:
        return

    token = ""
    if x_api_key:
        token = x_api_key.strip()
    elif authorization:
        auth = authorization.strip()
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()

    if token and secrets.compare_digest(token, configured_key):
        return

    raise HTTPException(status_code=401, detail="Unauthorized")

