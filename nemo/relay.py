"""Relay client — heartbeat-based idle detection via the nemo relay server.

Replaces PID-based idle detection (which only works on same machine)
with relay heartbeat endpoints that work across devices.
"""

from __future__ import annotations

import json
import logging
import urllib.request

from .config import load_relay_config

log = logging.getLogger(__name__)


def _relay_request(method: str, path: str, data: dict | None = None) -> dict:
    """Make an authenticated request to the relay server."""
    relay_url, api_key = load_relay_config()
    if not relay_url:
        raise RuntimeError("Relay not configured (set relay_url in ~/.nemo/config.json)")

    url = f"{relay_url.rstrip('/')}{path}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def send_heartbeat(chat_id: str, pid: int = 0,
                   model: str = "", machine: str = "") -> None:
    """Send a heartbeat to the relay, marking this chat as occupied."""
    try:
        _relay_request("POST", f"/heartbeat/chat:{chat_id}", {
            "pid": pid,
            "model": model,
            "machine": machine,
        })
    except Exception as e:
        log.warning("Heartbeat send failed: %s", e)


def is_alive(chat_id: str) -> bool:
    """Check if a chat has an active agent (heartbeat not expired)."""
    try:
        resp = _relay_request("GET", f"/heartbeat/chat:{chat_id}")
        return resp.get("alive", False)
    except Exception as e:
        log.warning("Heartbeat check failed: %s", e)
        return False  # Can't reach relay → treat as idle


def release_heartbeat(chat_id: str) -> None:
    """Explicitly release the heartbeat, marking the chat as idle."""
    try:
        _relay_request("DELETE", f"/heartbeat/chat:{chat_id}")
    except Exception as e:
        log.warning("Heartbeat release failed: %s", e)


def register_message(message_id: str, chat_id: str) -> None:
    """Register a message→chat mapping for reaction routing."""
    try:
        _relay_request("POST", "/register-message", {
            "message_id": message_id,
            "chat_id": chat_id,
        })
    except Exception as e:
        log.debug("Register message failed: %s", e)
