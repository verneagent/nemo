"""Tests for nemo.status_tab — two-tab status management."""

from unittest import mock

from nemo.status_tab import (
  _find_tab_by_marker, update_status, remove_tab,
  TAB_URL, MODEL_TAB_URL, _STATUS_MARKER, _MODEL_MARKER,
)


def _make_tab(tab_id: str, name: str, url: str) -> dict:
  return {"tab_id": tab_id, "tab_name": name, "tab_content": {"url": url}}


# --- _find_tab_by_marker ---

def test_find_tab_by_status_marker():
  tabs = [
    _make_tab("t1", "Messages", "https://example.com"),
    _make_tab("t2", "🟢", TAB_URL),
  ]
  assert _find_tab_by_marker(tabs, _STATUS_MARKER)["tab_id"] == "t2"


def test_find_tab_by_model_marker():
  tabs = [
    _make_tab("t1", "claude-opus-4-6", MODEL_TAB_URL),
  ]
  assert _find_tab_by_marker(tabs, _MODEL_MARKER)["tab_id"] == "t1"


def test_find_tab_not_found():
  tabs = [_make_tab("t1", "Messages", "https://example.com")]
  assert _find_tab_by_marker(tabs, _STATUS_MARKER) is None


def test_find_tab_ignores_name_only():
  """Should NOT match by name if URL doesn't have the marker."""
  tabs = [_make_tab("t1", "🟢", "https://other.com")]
  assert _find_tab_by_marker(tabs, _STATUS_MARKER) is None


# --- update_status: creates both tabs when none exist ---

def test_update_creates_both_tabs():
  with mock.patch("nemo.lark.api.list_chat_tabs", return_value=[]):
    with mock.patch("nemo.lark.api.create_chat_tab") as mock_create:
      update_status("tok", "oc_1", "opus", "idle")
  assert mock_create.call_count == 2
  # First call: status tab (emoji)
  assert mock_create.call_args_list[0][0][2] == "🟢"
  assert _STATUS_MARKER in mock_create.call_args_list[0][0][3]
  # Second call: model tab
  assert mock_create.call_args_list[1][0][2] == "opus"
  assert _MODEL_MARKER in mock_create.call_args_list[1][0][3]


# --- update_status: updates status emoji ---

def test_update_status_emoji():
  tabs = [_make_tab("t1", "🟢", TAB_URL), _make_tab("t2", "opus", MODEL_TAB_URL)]
  with mock.patch("nemo.lark.api.list_chat_tabs", return_value=tabs):
    with mock.patch("nemo.lark.api.update_chat_tab") as mock_update:
      with mock.patch("nemo.lark.api.create_chat_tab") as mock_create:
        update_status("tok", "oc_1", "opus", "working")
  # Status tab updated to 🟡
  mock_update.assert_called_once()
  assert mock_update.call_args[0][3] == "🟡"
  # Model tab unchanged — no create or extra update
  mock_create.assert_not_called()


# --- update_status: skips if emoji unchanged ---

def test_update_status_skips_same_emoji():
  tabs = [_make_tab("t1", "🟢", TAB_URL), _make_tab("t2", "opus", MODEL_TAB_URL)]
  with mock.patch("nemo.lark.api.list_chat_tabs", return_value=tabs):
    with mock.patch("nemo.lark.api.update_chat_tab") as mock_update:
      update_status("tok", "oc_1", "opus", "idle")
  mock_update.assert_not_called()


# --- update_status: updates model name ---

def test_update_model_name():
  tabs = [_make_tab("t1", "🟢", TAB_URL), _make_tab("t2", "old-model", MODEL_TAB_URL)]
  with mock.patch("nemo.lark.api.list_chat_tabs", return_value=tabs):
    with mock.patch("nemo.lark.api.update_chat_tab") as mock_update:
      update_status("tok", "oc_1", "new-model", "idle")
  # Only model tab updated (status emoji same)
  mock_update.assert_called_once()
  assert mock_update.call_args[0][3] == "new-model"


# --- remove_tab: removes both tabs ---

def test_remove_both_tabs():
  tabs = [_make_tab("t1", "🟡", TAB_URL), _make_tab("t2", "opus", MODEL_TAB_URL)]
  with mock.patch("nemo.lark.api.list_chat_tabs", return_value=tabs):
    with mock.patch("nemo.lark.api.delete_chat_tab") as mock_delete:
      remove_tab("tok", "oc_1")
  mock_delete.assert_called_once_with("tok", "oc_1", ["t1", "t2"])


def test_remove_tab_only_status():
  tabs = [_make_tab("t1", "🟡", TAB_URL)]
  with mock.patch("nemo.lark.api.list_chat_tabs", return_value=tabs):
    with mock.patch("nemo.lark.api.delete_chat_tab") as mock_delete:
      remove_tab("tok", "oc_1")
  mock_delete.assert_called_once_with("tok", "oc_1", ["t1"])


def test_remove_tab_none():
  with mock.patch("nemo.lark.api.list_chat_tabs", return_value=[]):
    with mock.patch("nemo.lark.api.delete_chat_tab") as mock_delete:
      remove_tab("tok", "oc_1")
  mock_delete.assert_not_called()
