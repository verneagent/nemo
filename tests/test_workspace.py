"""Tests for nemo.workspace — workspace discovery."""

from unittest import mock

from nemo.workspace import (
  get_machine_name, get_workspace_id,
  _workspace_tag_matches, discover_chat_id,
  ensure_workspace_tag, auto_create_chat,
  discover_or_create_chat, claim_group, release_group,
  _is_group_idle, _compute_group_name,
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


def test_discover_chat_id_found_idle():
  with mock.patch("nemo.workspace.get_machine_name", return_value="Mac"):
    with mock.patch("nemo.lark.api.list_bot_chats", return_value=[
      {"chat_id": "oc_1"}, {"chat_id": "oc_2"},
    ]):
      with mock.patch("nemo.lark.api.get_chat_info", side_effect=[
        {"description": "unrelated"},
        {"description": "workspace:Mac-tmp-project"},
      ]):
        with mock.patch("nemo.workspace._is_group_idle", return_value=True):
          result = discover_chat_id("tok", "/tmp/project")
  assert result == "oc_2"


def test_discover_chat_id_skips_occupied():
  """Should skip occupied groups and return None if all occupied."""
  with mock.patch("nemo.workspace.get_machine_name", return_value="Mac"):
    with mock.patch("nemo.lark.api.list_bot_chats", return_value=[
      {"chat_id": "oc_1"},
    ]):
      with mock.patch("nemo.lark.api.get_chat_info",
                      return_value={"description": "workspace:Mac-tmp-project"}):
        with mock.patch("nemo.workspace._is_group_idle", return_value=False):
          result = discover_chat_id("tok", "/tmp/project")
  assert result is None


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


# ---------------------------------------------------------------------------
# auto_create_chat tests
# ---------------------------------------------------------------------------

def test_auto_create_chat_success():
  """Should create group, resolve email, and pin config."""
  with mock.patch("nemo.workspace.get_machine_name", return_value="Mac"):
    with mock.patch("nemo.lark.api.lookup_open_id_by_email", return_value="ou_123"):
      with mock.patch("nemo.lark.api.create_chat", return_value="oc_new") as mock_create:
        with mock.patch("nemo.group_config.save_config") as mock_save:
          result = auto_create_chat("tok", "/tmp/proj", email="a@b.com")
  assert result == "oc_new"
  mock_create.assert_called_once()
  call_kwargs = mock_create.call_args
  assert call_kwargs[1]["user_ids"] == ["ou_123"]
  assert "workspace:Mac-tmp-proj" in call_kwargs[1]["description"]
  mock_save.assert_called_once()


def test_auto_create_chat_no_email():
  """Should create group without members when no email."""
  with mock.patch("nemo.workspace.get_machine_name", return_value="Mac"):
    with mock.patch("nemo.lark.api.create_chat", return_value="oc_new"):
      with mock.patch("nemo.group_config.save_config"):
        result = auto_create_chat("tok", "/tmp/proj")
  assert result == "oc_new"


def test_auto_create_chat_email_resolve_fails():
  """Should still create group if email resolution fails."""
  with mock.patch("nemo.workspace.get_machine_name", return_value="Mac"):
    with mock.patch("nemo.lark.api.lookup_open_id_by_email", return_value=None):
      with mock.patch("nemo.lark.api.create_chat", return_value="oc_new") as mock_create:
        with mock.patch("nemo.group_config.save_config"):
          result = auto_create_chat("tok", "/tmp/proj", email="a@b.com")
  assert result == "oc_new"
  assert mock_create.call_args[1]["user_ids"] == []


def test_auto_create_chat_create_fails():
  """Should return None when group creation fails."""
  with mock.patch("nemo.workspace.get_machine_name", return_value="Mac"):
    with mock.patch("nemo.lark.api.create_chat", side_effect=RuntimeError("fail")):
      result = auto_create_chat("tok", "/tmp/proj")
  assert result is None


def test_auto_create_chat_config_pin_fails():
  """Should still return chat_id if config pin fails."""
  with mock.patch("nemo.workspace.get_machine_name", return_value="Mac"):
    with mock.patch("nemo.lark.api.create_chat", return_value="oc_new"):
      with mock.patch("nemo.group_config.save_config", side_effect=RuntimeError("pin fail")):
        result = auto_create_chat("tok", "/tmp/proj")
  assert result == "oc_new"


# ---------------------------------------------------------------------------
# _is_group_idle tests
# ---------------------------------------------------------------------------

def test_is_group_idle_no_pid():
  """No active_pid → idle."""
  with mock.patch("nemo.group_config.load_config",
                  return_value={"autoapprove": False}):
    assert _is_group_idle("tok", "oc_1") is True


def test_is_group_idle_pid_zero():
  """active_pid=0 → idle."""
  with mock.patch("nemo.group_config.load_config",
                  return_value={"active_pid": 0}):
    assert _is_group_idle("tok", "oc_1") is True


def test_is_group_idle_dead_pid():
  """active_pid points to dead process → idle."""
  with mock.patch("nemo.group_config.load_config",
                  return_value={"active_pid": 99999999}):
    with mock.patch("nemo.workspace._is_pid_alive", return_value=False):
      assert _is_group_idle("tok", "oc_1") is True


def test_is_group_idle_live_pid():
  """active_pid points to live process → occupied."""
  with mock.patch("nemo.group_config.load_config",
                  return_value={"active_pid": 12345}):
    with mock.patch("nemo.workspace._is_pid_alive", return_value=True):
      assert _is_group_idle("tok", "oc_1") is False


def test_is_group_idle_config_error():
  """Config read error → treat as idle."""
  with mock.patch("nemo.group_config.load_config",
                  side_effect=RuntimeError("fail")):
    assert _is_group_idle("tok", "oc_1") is True


# ---------------------------------------------------------------------------
# _compute_group_name tests
# ---------------------------------------------------------------------------

def test_compute_group_name_first():
  assert _compute_group_name("myapp", "Mac", []) == "myapp@Mac"


def test_compute_group_name_second():
  assert _compute_group_name("myapp", "Mac", ["myapp@Mac"]) == "myapp2@Mac"


def test_compute_group_name_skip():
  existing = ["myapp@Mac", "myapp2@Mac"]
  assert _compute_group_name("myapp", "Mac", existing) == "myapp3@Mac"


# ---------------------------------------------------------------------------
# discover_or_create_chat tests
# ---------------------------------------------------------------------------

def test_discover_or_create_reuses_idle():
  """Should reuse first idle group."""
  with mock.patch("nemo.workspace._find_workspace_groups",
                  return_value=[{"chat_id": "oc_1", "name": "G1"}]):
    with mock.patch("nemo.workspace._is_group_idle", return_value=True):
      result = discover_or_create_chat("tok", "/tmp/proj")
  assert result == "oc_1"


def test_discover_or_create_skips_occupied_and_creates():
  """Should create new group when all are occupied."""
  with mock.patch("nemo.workspace._find_workspace_groups",
                  return_value=[{"chat_id": "oc_1", "name": "Nemo · proj"}]):
    with mock.patch("nemo.workspace._is_group_idle", return_value=False):
      with mock.patch("nemo.workspace.auto_create_chat",
                      return_value="oc_new") as mock_create:
        result = discover_or_create_chat("tok", "/tmp/proj", email="a@b.com")
  assert result == "oc_new"
  mock_create.assert_called_once_with(
    "tok", "/tmp/proj", email="a@b.com",
    existing_names=["Nemo · proj"],
  )


def test_discover_or_create_no_groups_creates():
  """Should create when no groups exist at all."""
  with mock.patch("nemo.workspace._find_workspace_groups", return_value=[]):
    with mock.patch("nemo.workspace.auto_create_chat",
                    return_value="oc_new") as mock_create:
      result = discover_or_create_chat("tok", "/tmp/proj")
  assert result == "oc_new"
  mock_create.assert_called_once()


# ---------------------------------------------------------------------------
# claim_group / release_group tests
# ---------------------------------------------------------------------------

def test_claim_group():
  with mock.patch("nemo.group_config.load_config",
                  return_value={"autoapprove": False}):
    with mock.patch("nemo.group_config.save_config") as mock_save:
      with mock.patch("os.getpid", return_value=42):
        claim_group("tok", "oc_1")
  saved_config = mock_save.call_args[0][2]
  assert saved_config["active_pid"] == 42


def test_release_group():
  with mock.patch("nemo.group_config.load_config",
                  return_value={"autoapprove": False, "active_pid": 42}):
    with mock.patch("nemo.group_config.save_config") as mock_save:
      release_group("tok", "oc_1")
  saved_config = mock_save.call_args[0][2]
  assert saved_config["active_pid"] == 0


def test_claim_group_error_tolerant():
  """Should not raise on failure."""
  with mock.patch("nemo.group_config.load_config",
                  side_effect=RuntimeError("fail")):
    claim_group("tok", "oc_1")  # Should not raise


def test_release_group_error_tolerant():
  """Should not raise on failure."""
  with mock.patch("nemo.group_config.load_config",
                  side_effect=RuntimeError("fail")):
    release_group("tok", "oc_1")  # Should not raise
