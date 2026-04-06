"""Status tab — show agent/WebSocket status as a chat tab.

Uses a URL tab with emoji prefix to indicate state:
  🟢 Nemo (model) — connected, idle
  🟡 Nemo (model) — working
  🔴 Nemo (model) — disconnected / stopped
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

TAB_URL = "https://github.com/verneagent/nemo"
_TAB_PREFIX = "Nemo"


def _find_nemo_tab(tabs: list[dict[str, Any]]) -> dict[str, Any] | None:
  """Find the existing Nemo status tab."""
  for tab in tabs:
    name = tab.get("tab_name", "")
    if name.startswith("🟢") or name.startswith("🟡") or name.startswith("🔴"):
      if _TAB_PREFIX in name:
        return tab
  return None


def _tab_name(model: str, status: str) -> str:
  if status == "working":
    return f"🟡 {_TAB_PREFIX} ({model})"
  if status == "stopped":
    return f"🔴 {_TAB_PREFIX} ({model})"
  return f"🟢 {_TAB_PREFIX} ({model})"


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
