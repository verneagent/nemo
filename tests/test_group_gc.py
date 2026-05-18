from unittest import mock

from nemo.group_gc import (
  GcChat,
  collect_gc_chats,
  dissolve_chats,
  format_gc_table,
  gc_clean,
)


def test_collect_gc_chats_uses_workspace_tag_and_heartbeat():
  with mock.patch("nemo.lark.api.list_bot_chats", return_value=[
    {"chat_id": "oc_1", "name": "one"},
    {"chat_id": "oc_2", "name": "two"},
  ]), mock.patch("nemo.lark.api.get_chat_info", side_effect=[
    {"name": "one", "description": "workspace:Mac|/repo", "owner_id": "ou_bot"},
    {"name": "two", "description": "plain", "owner_id": "ou_bot"},
  ]), mock.patch("nemo.lark.api.get_bot_info", return_value={
    "open_id": "ou_bot",
  }), mock.patch("nemo.group_gc._has_nemo_config_pin", return_value=False), \
       mock.patch("nemo.relay.heartbeat_status",
                  return_value={"alive": True, "machine": "Mac", "model": "opus"}):
    rows = collect_gc_chats("tok")

  assert len(rows) == 1
  assert rows[0].chat_id == "oc_1"
  assert rows[0].alive is True
  assert rows[0].machine == "Mac"


def test_collect_gc_chats_accepts_config_pin_without_workspace_tag():
  with mock.patch("nemo.lark.api.list_bot_chats", return_value=[
    {"chat_id": "oc_1", "name": "one"},
  ]), mock.patch("nemo.lark.api.get_chat_info",
                  return_value={
                    "name": "one",
                    "description": "",
                    "owner_id": "ou_bot",
                  }), \
       mock.patch("nemo.lark.api.get_bot_info", return_value={
         "open_id": "ou_bot",
       }), \
       mock.patch("nemo.lark.api.list_pins",
                  return_value=[{"message_id": "om_cfg"}]), \
       mock.patch("nemo.lark.api.get_message", return_value={
         "msg_type": "text",
         "body": {"content": '{"text":"__nemo_config__\\nautoapprove: false"}'},
       }), mock.patch("nemo.relay.heartbeat_status",
                       return_value={"alive": False}):
    rows = collect_gc_chats("tok")

  assert len(rows) == 1
  assert rows[0].status == "IDLE"


def test_collect_gc_chats_skips_nemo_marked_chat_not_owned_by_bot():
  with mock.patch("nemo.lark.api.list_bot_chats", return_value=[
    {"chat_id": "oc_1", "name": "not-owned"},
  ]), mock.patch("nemo.lark.api.get_chat_info", return_value={
    "name": "not-owned",
    "description": "workspace:Mac|/repo",
    "owner_id": "ou_user",
  }), mock.patch("nemo.lark.api.get_bot_info", return_value={
    "open_id": "ou_bot",
  }), mock.patch("nemo.relay.heartbeat_status") as heartbeat:
    rows = collect_gc_chats("tok")

  assert rows == []
  heartbeat.assert_not_called()


def test_collect_gc_chats_skips_when_bot_owner_unknown():
  with mock.patch("nemo.lark.api.get_bot_info", return_value={}), \
       mock.patch("nemo.lark.api.list_bot_chats") as list_chats:
    rows = collect_gc_chats("tok")

  assert rows == []
  list_chats.assert_not_called()


def test_format_gc_table_marks_relay_errors_unknown():
  with mock.patch("nemo.lark.api.list_bot_chats", return_value=[
    {"chat_id": "oc_1", "name": "one"},
  ]), mock.patch("nemo.lark.api.get_chat_info",
                  return_value={
                    "name": "one",
                    "description": "workspace:x",
                    "owner_id": "ou_bot",
                  }), \
       mock.patch("nemo.lark.api.get_bot_info", return_value={
         "open_id": "ou_bot",
       }), \
       mock.patch("nemo.relay.heartbeat_status",
                  return_value={"alive": False, "error": "timeout"}):
    rows = collect_gc_chats("tok")

  rendered = format_gc_table(rows)

  assert "UNKNOWN" in rendered
  assert "timeout" in rendered


def test_dissolve_chats_refuses_alive_group():
  row = mock.Mock(
    chat_id="oc_1",
    name="one",
  )
  with mock.patch("nemo.relay.heartbeat_status", return_value={"alive": True}), \
       mock.patch("nemo.lark.api.dissolve_chat") as dissolve:
    dissolved, skipped = dissolve_chats("tok", [row])

  assert dissolved == []
  assert skipped
  dissolve.assert_not_called()


def test_dissolve_chats_refuses_unknown_heartbeat():
  row = mock.Mock(
    chat_id="oc_1",
    name="one",
  )
  with mock.patch("nemo.relay.heartbeat_status",
                  return_value={"alive": False, "error": "relay down"}), \
       mock.patch("nemo.lark.api.dissolve_chat") as dissolve:
    dissolved, skipped = dissolve_chats("tok", [row])

  assert dissolved == []
  assert "relay down" in skipped[0]
  dissolve.assert_not_called()


def test_gc_clean_interactive_selection_dissolves_selected(capsys):
  rows = [
    GcChat("oc_1", "one", "", True, False, "", "", ""),
    GcChat("oc_2", "two", "", True, False, "", "", ""),
  ]
  with mock.patch("nemo.group_gc.collect_gc_chats", return_value=rows), \
       mock.patch("nemo.relay.heartbeat_status", return_value={"alive": False}), \
       mock.patch("nemo.lark.api.dissolve_chat") as dissolve, \
       mock.patch("sys.stdin.readline", side_effect=["2\n", "dissolve\n"]):
    rc = gc_clean("tok")

  assert rc == 0
  dissolve.assert_called_once_with("tok", "oc_2")
  assert "Dissolved two" in capsys.readouterr().out
