"""Tests for nemo.workspace — workspace discovery."""

from unittest import mock

from nemo.workspace import (
  get_machine_name, get_workspace_id,
  _workspace_tag_matches, discover_chat_id,
  ensure_workspace_tag,
)


def test_get_workspace_id():
  with mock.patch("nemo.workspace.get_machine_name", return_value="MyMac"):
    ws_id = get_workspace_id("/Users/alice/projects/myapp")
  assert ws_id == "MyMac-Users-alice-projects-myapp"


def test_workspace_tag_matches_exact():
  assert _workspace_tag_matches("workspace:A-B", "workspace:A-B")


def test_workspace_tag_matches_with_trailing():
  assert _workspace_tag_matches("workspace:A-B some notes", "workspace:A-B")


def test_workspace_tag_no_prefix_match():
  """workspace:A-B should NOT match workspace:A-B-C."""
  assert not _workspace_tag_matches("workspace:A-B-C", "workspace:A-B")


def test_workspace_tag_not_found():
  assert not _workspace_tag_matches("some random desc", "workspace:X")


def test_workspace_tag_newline():
  assert _workspace_tag_matches("workspace:A-B\nmore info", "workspace:A-B")


def test_discover_chat_id_found():
  with mock.patch("nemo.workspace.get_machine_name", return_value="Mac"):
    with mock.patch("nemo.lark.api.list_bot_chats", return_value=[
      {"chat_id": "oc_1"}, {"chat_id": "oc_2"},
    ]):
      with mock.patch("nemo.lark.api.get_chat_info", side_effect=[
        {"description": "unrelated"},
        {"description": "workspace:Mac-tmp-project"},
      ]):
        result = discover_chat_id("tok", "/tmp/project")
  assert result == "oc_2"


def test_discover_chat_id_not_found():
  with mock.patch("nemo.workspace.get_machine_name", return_value="Mac"):
    with mock.patch("nemo.lark.api.list_bot_chats", return_value=[
      {"chat_id": "oc_1"},
    ]):
      with mock.patch("nemo.lark.api.get_chat_info", return_value={"description": "nope"}):
        result = discover_chat_id("tok", "/tmp/project")
  assert result is None


def test_discover_chat_id_api_error():
  with mock.patch("nemo.workspace.get_machine_name", return_value="Mac"):
    with mock.patch("nemo.lark.api.list_bot_chats", side_effect=RuntimeError("fail")):
      result = discover_chat_id("tok", "/tmp/project")
  assert result is None


def test_ensure_workspace_tag_already_present():
  """Should not update if tag already exists."""
  with mock.patch("nemo.workspace.get_machine_name", return_value="Mac"):
    with mock.patch("nemo.lark.api.get_chat_info",
                    return_value={"description": "workspace:Mac-tmp-proj"}):
      with mock.patch("nemo.lark.api.update_chat_info") as mock_update:
        ensure_workspace_tag("tok", "oc_1", "/tmp/proj")
        mock_update.assert_not_called()


def test_ensure_workspace_tag_appends():
  """Should append tag to existing description."""
  with mock.patch("nemo.workspace.get_machine_name", return_value="Mac"):
    with mock.patch("nemo.lark.api.get_chat_info",
                    return_value={"description": "My project group"}):
      with mock.patch("nemo.lark.api.update_chat_info") as mock_update:
        ensure_workspace_tag("tok", "oc_1", "/tmp/proj")
        mock_update.assert_called_once()
        new_desc = mock_update.call_args[0][2]["description"]
        assert "workspace:Mac-tmp-proj" in new_desc
        assert "My project group" in new_desc


def test_ensure_workspace_tag_empty_description():
  """Should set tag as description when empty."""
  with mock.patch("nemo.workspace.get_machine_name", return_value="Mac"):
    with mock.patch("nemo.lark.api.get_chat_info",
                    return_value={"description": ""}):
      with mock.patch("nemo.lark.api.update_chat_info") as mock_update:
        ensure_workspace_tag("tok", "oc_1", "/tmp/proj")
        new_desc = mock_update.call_args[0][2]["description"]
        assert new_desc == "workspace:Mac-tmp-proj"
