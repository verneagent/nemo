"""Lark tenant token management with auto-refresh."""

from __future__ import annotations

import json
import threading
import time
import urllib.request

BASE_URL = "https://open.larksuite.com/open-apis"


class LarkAuth:
  """Manages tenant access token with caching and auto-refresh."""

  def __init__(self) -> None:
    self._token: str = ""
    self._expires_at: float = 0
    self._lock = threading.Lock()

  def get_token(self, app_id: str, app_secret: str) -> str:
    """Get a valid tenant access token, refreshing if needed."""
    with self._lock:
      if self._token and time.time() < self._expires_at - 60:
        return self._token

      url = f"{BASE_URL}/auth/v3/tenant_access_token/internal"
      payload = json.dumps({
        "app_id": app_id,
        "app_secret": app_secret,
      }).encode()
      req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
      )
      with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())

      if data.get("code") != 0:
        raise RuntimeError(f"Token error: {data}")

      self._token = data["tenant_access_token"]
      self._expires_at = time.time() + data.get("expire", 7200)
      return self._token

  def invalidate(self) -> None:
    """Force token refresh on next get_token() call."""
    with self._lock:
      self._expires_at = 0


# Module-level singleton
_auth = LarkAuth()


def get_token(app_id: str, app_secret: str) -> str:
  return _auth.get_token(app_id, app_secret)


def invalidate() -> None:
  _auth.invalidate()
