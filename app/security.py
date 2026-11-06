"""One shared password for the whole site (HTTP Basic auth). Off when APP_PASSWORD is not set, so local use stays easy."""
from __future__ import annotations

import base64
import os
import secrets

from starlette.responses import PlainTextResponse, Response

OPEN_PATHS = {"/healthz"}   # the host's health check must work without a password


class PasswordGate:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        password = os.environ.get("APP_PASSWORD", "")
        if scope["type"] != "http" or not password or scope["path"] in OPEN_PATHS:
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        if _ok(headers.get(b"authorization", b"").decode("latin-1"), password):
            return await self.app(scope, receive, send)
        resp: Response = PlainTextResponse("Password needed.", status_code=401,
                                           headers={"WWW-Authenticate": 'Basic realm="Databoarder", charset="UTF-8"'})
        return await resp(scope, receive, send)


def _ok(header: str, password: str) -> bool:
    """Any username; only the password is checked (compared in constant time)."""
    if not header.lower().startswith("basic "):
        return False
    try:
        given = base64.b64decode(header[6:]).decode("utf-8").partition(":")[2]
    except Exception:
        return False
    return secrets.compare_digest(given.encode(), password.encode())
