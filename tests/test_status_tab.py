"""Tests for nemo.status_tab — status tab management."""

from unittest import mock

from nemo.status_tab import (
  _tab_name, _find_nemo_tab, update_status, remove_tab, TAB_URL,
)


def test_tab_name_idle():
  assert _tab_name("claude-opus-4-6", "idle") == "🟢 Nemo (claude-opus-4-6)"


def test_tab_name_working():
  assert _tab_name("claude-opus-4-6", "working") == "🟡 Nemo (claude-opus-4-6)"


def test_tab_name_stopped():
  assert _tab_name("claude-opus-4-6", "stopped") == "🔴 Nemo (claude-opus-4-6)"


def test_find_nemo_tab_by_url():
  tabs = [
    {"tab_id": "t1", "tab_name": "Messages", "tab_content": {"url": "https://example.com"}},
    {"tab_id": "t2", "tab_name": "🟢 Nemo (opus)", "tab_content": {"url": TAB_URL}},
  ]
  assert _find_nemo_tab(tabs)["tab_id"] == "t2"


def test_find_nemo_tab_not_found():
  tabs = [{"tab_id": "t1", "tab_name": "Messages", "tab_content": {"url": "https://example.com"}}]
  assert _find_nemo_tab(tabs) is None


def test_find_nemo_tab_ignores_name_only():
  """Should NOT match by name if URL doesn't have the marker."""
  tabs = [{"tab_id": "t1", "tab_name": "🟢 Nemo (opus)", "tab_content": {"url": "https://other.com"}}]
  assert _find_nemo_tab(tabs) is None


def test_update_status_creates_tab():
  with mock.patch("nemo.lark.api.list_chat_tabs", return_value=[]):
    with mock.patch("nemo.lark.api.create_chat_tab") as mock_create:
      update_status("tok", "oc_1", "opus", "idle")
  mock_create.assert_called_once()
  assert "🟢" in mock_create.call_args[0][2]
  assert "nemo-status" in mock_create.call_args[0][3]


def test_update_status_updates_existing():
  tabs = [{"tab_id": "t1", "tab_name": "🟢 Nemo (opus)", "tab_content": {"url": TAB_URL}}]
  with mock.patch("nemo.lark.api.list_chat_tabs", return_value=tabs):
    with mock.patch("nemo.lark.api.update_chat_tab") as mock_update:
      update_status("tok", "oc_1", "opus", "working")
  mock_update.assert_called_once()
  assert "🟡" in mock_update.call_args[0][3]


def test_update_status_skips_if_same():
  tabs = [{"tab_id": "t1", "tab_name": "🟢 Nemo (opus)", "tab_content": {"url": TAB_URL}}]
  with mock.patch("nemo.lark.api.list_chat_tabs", return_value=tabs):
    with mock.patch("nemo.lark.api.update_chat_tab") as mock_update:
      update_status("tok", "oc_1", "opus", "idle")
  mock_update.assert_not_called()


def test_remove_tab():
  tabs = [{"tab_id": "t1", "tab_name": "🟡 Nemo (opus)", "tab_content": {"url": TAB_URL}}]
  with mock.patch("nemo.lark.api.list_chat_tabs", return_value=tabs):
    with mock.patch("nemo.lark.api.delete_chat_tab") as mock_delete:
      remove_tab("tok", "oc_1")
  mock_delete.assert_called_once_with("tok", "oc_1", ["t1"])


def test_remove_tab_no_tab():
  with mock.patch("nemo.lark.api.list_chat_tabs", return_value=[]):
    with mock.patch("nemo.lark.api.delete_chat_tab") as mock_delete:
      remove_tab("tok", "oc_1")
  mock_delete.assert_not_called()
