"""Status tab — show agent/WebSocket status as a chat tab.

Uses a URL tab with emoji prefix to indicate state:
  🟢 Nemo (model) — connected, idle
  🟡 Nemo (model) — working
  🔴 Nemo (model) — disconnected / stopped

The tab is identified by its URL containing `?type=nemo-status`,
not by parsing the tab name.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

_BASE_URL = "https://github.com/verneagent/nemo"
_TAB_MARKER = "type=nemo-status"
TAB_URL = f"{_BASE_URL}?{_TAB_MARKER}"


def _find_nemo_tab(tabs: list[dict[str, Any]]) -> dict[str, Any] | None:
  """Find the existing Nemo status tab by URL marker."""
  for tab in tabs:
    url = (tab.get("tab_content") or {}).get("url", "")
    if _TAB_MARKER in url:
      return tab
  return None


def _tab_name(model: str, status: str) -> str:
  if status == "working":
    return f"🟡 Nemo ({model})"
  if status == "stopped":
    return f"🔴 Nemo ({model})"
  return f"🟢 Nemo ({model})"


def update_status(token: str, chat_id: str, model: str,
                  status: str = "idle") -> None:
  """Create or update the status tab. status: idle, working, stopped."""
  from .lark import api as lark_api

  name = _tab_name(model, status)
  try:
    tabs = lark_api.list_chat_tabs(token, chat_id)
    existing = _find_nemo_tab(tabs)
    if existing:
      tab_id = existing.get("tab_id", "")
      if existing.get("tab_name") != name and tab_id:
        lark_api.update_chat_tab(token, chat_id, tab_id, name, TAB_URL)
    else:
      lark_api.create_chat_tab(token, chat_id, name, TAB_URL)
  except Exception as e:
    log.warning("Status tab update failed: %s", e)


def remove_tab(token: str, chat_id: str) -> None:
  """Remove the Nemo status tab."""
  from .lark import api as lark_api

  try:
    tabs = lark_api.list_chat_tabs(token, chat_id)
    existing = _find_nemo_tab(tabs)
    if existing and existing.get("tab_id"):
      lark_api.delete_chat_tab(token, chat_id, [existing["tab_id"]])
  except Exception as e:
    log.warning("Status tab removal failed: %s", e)
