"""Status tabs — show model and WebSocket status as separate chat tabs.

Two tabs:
  1. Model tab: shows the model name (e.g. "claude-opus-4-7")
  2. WS status tab: emoji only (🟢 idle, 🟡 working, 🔴 stopped)

Tabs are identified by URL markers (?type=nemo-status, ?type=nemo-model),
not by parsing the tab name.
"""

from __future__ import annotations

import logging

from .types import JsonObject

log = logging.getLogger(__name__)

_BASE_URL = "https://github.com/verneagent/nemo"
_STATUS_MARKER = "type=nemo-status"
_MODEL_MARKER = "type=nemo-model"
TAB_URL = f"{_BASE_URL}?{_STATUS_MARKER}"
MODEL_TAB_URL = f"{_BASE_URL}?{_MODEL_MARKER}"

_STATUS_EMOJI = {
  "idle": "🟢",
  "working": "🟡",
  "stopped": "🔴",
}


def _find_tab_by_marker(tabs: list[JsonObject], marker: str) -> JsonObject | None:
  """Find a tab by URL marker."""
  for tab in tabs:
    url = (tab.get("tab_content") or {}).get("url", "")
    if marker in url:
      return tab
  return None



def update_status(token: str, chat_id: str, model: str,
                  status: str = "idle") -> None:
  """Create or update both tabs. status: idle, working, stopped."""
  from .lark import api as lark_api

  emoji = _STATUS_EMOJI.get(status, "🟢")
  try:
    tabs = lark_api.list_chat_tabs(token, chat_id)

    # --- Status tab (emoji only) ---
    status_tab = _find_tab_by_marker(tabs, _STATUS_MARKER)
    if status_tab:
      tab_id = status_tab.get("tab_id", "")
      if status_tab.get("tab_name") != emoji and tab_id:
        lark_api.update_chat_tab(token, chat_id, tab_id, emoji, TAB_URL)
    else:
      lark_api.create_chat_tab(token, chat_id, emoji, TAB_URL)

    # --- Model tab (model name) ---
    model_tab = _find_tab_by_marker(tabs, _MODEL_MARKER)
    if not model_tab:
      lark_api.create_chat_tab(token, chat_id, model, MODEL_TAB_URL)
    elif model_tab.get("tab_name") != model:
      tab_id = model_tab.get("tab_id", "")
      if tab_id:
        lark_api.update_chat_tab(token, chat_id, tab_id, model, MODEL_TAB_URL)

    # Move pin tab to the end
    _sort_pin_last(token, chat_id)

  except Exception as e:
    log.warning("Status tab update failed: %s", e)


def _sort_pin_last(token: str, chat_id: str) -> None:
  """Re-order tabs so pin is last."""
  from .lark import api as lark_api

  tabs = lark_api.list_chat_tabs(token, chat_id)
  non_pin = [t["tab_id"] for t in tabs if t.get("tab_type") != "pin"]
  pin = [t["tab_id"] for t in tabs if t.get("tab_type") == "pin"]
  if pin and tabs[-1].get("tab_type") != "pin":
    lark_api.sort_chat_tabs(token, chat_id, non_pin + pin)


def remove_tab(token: str, chat_id: str) -> None:
  """Remove both Nemo tabs."""
  from .lark import api as lark_api

  try:
    tabs = lark_api.list_chat_tabs(token, chat_id)
    tab_ids = []
    for marker in (_STATUS_MARKER, _MODEL_MARKER):
      tab = _find_tab_by_marker(tabs, marker)
      if tab and tab.get("tab_id"):
        tab_ids.append(tab["tab_id"])
    if tab_ids:
      lark_api.delete_chat_tab(token, chat_id, tab_ids)
  except Exception as e:
    log.warning("Status tab removal failed: %s", e)
