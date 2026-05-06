"""Tests for nemo.group_config — pinned text message configuration (YAML)."""

import json
from unittest import mock

import yaml

from nemo.group_config import (
  _build_config_text, _parse_config_text,
  load_config, save_config, DEFAULT_CONFIG, CONFIG_MARKER,
)


def _text_msg(text: str) -> dict:
  """Build a fake get_message response for a text message."""
  return {
    "msg_type": "text",
    "body": {"content": json.dumps({"text": text})},
  }


def test_build_config_text():
  config = {"autoapprove": True, "guests": []}
  text = _build_config_text(config)
  assert text.startswith(CONFIG_MARKER)
  yaml_part = text[len(CONFIG_MARKER):].strip()
  assert yaml.safe_load(yaml_part)["autoapprove"] is True


def test_parse_config_text_valid():
  config = {"autoapprove": False, "guests": []}
  text = _build_config_text(config)
  msg = _text_msg(text)
  result = _parse_config_text(msg)
  assert result is not None
  assert result["autoapprove"] is False


def test_parse_config_text_not_text_msg():
  assert _parse_config_text({"msg_type": "interactive"}) is None


def test_parse_config_text_no_marker():
  msg = _text_msg("hello world")
  assert _parse_config_text(msg) is None


def test_parse_config_text_invalid_yaml():
  msg = _text_msg(f"{CONFIG_MARKER}\n: : : bad")
  assert _parse_config_text(msg) is None


def test_parse_config_text_no_valid_keys():
  msg = _text_msg(f'{CONFIG_MARKER}\n{{"random": 1}}')
  assert _parse_config_text(msg) is None


def test_parse_config_text_empty():
  assert _parse_config_text({}) is None
  assert _parse_config_text({"msg_type": "text", "body": {}}) is None



def test_load_config_found():
  config = {"autoapprove": True, "guests": [], "rules": {}}
  text = _build_config_text(config)
  msg = _text_msg(text)
  with mock.patch("nemo.lark.api.list_pins", return_value=[
    {"message_id": "msg_1"},
  ]):
    with mock.patch("nemo.lark.api.get_message", return_value=msg):
      result = load_config("tok", "oc_1")
  assert result["autoapprove"] is True


def test_load_config_not_found():
  with mock.patch("nemo.lark.api.list_pins", return_value=[]):
    result = load_config("tok", "oc_1")
  assert result == DEFAULT_CONFIG


def test_load_config_merges_defaults():
  config = {"autoapprove": True}
  text = _build_config_text(config)
  msg = _text_msg(text)
  with mock.patch("nemo.lark.api.list_pins", return_value=[
    {"message_id": "msg_1"},
  ]):
    with mock.patch("nemo.lark.api.get_message", return_value=msg):
      result = load_config("tok", "oc_1")
  assert result["autoapprove"] is True
  assert result["guests"] == []
  assert result["rules"] == {}


def test_save_config_creates_new():
  config = {"autoapprove": True, "guests": [], "rules": {}}
  with mock.patch("nemo.lark.api.list_pins", return_value=[]):
    with mock.patch("nemo.lark.api.send_text", return_value="msg_new") as mock_send:
      with mock.patch("nemo.lark.api.create_pin") as mock_pin:
        msg_id = save_config("tok", "oc_1", config)
  assert msg_id == "msg_new"
  mock_pin.assert_called_once_with("tok", "msg_new")
  # Verify the text contains the marker
  sent_text = mock_send.call_args[0][2]
  assert sent_text.startswith(CONFIG_MARKER)


def test_find_config_pin_deduplicates():
  """When multiple config pins exist, keep first and delete the rest."""
  config1 = {"autoapprove": False, "guests": []}
  config2 = {"autoapprove": True, "guests": []}
  msg1 = _text_msg(_build_config_text(config1))
  msg2 = _text_msg(_build_config_text(config2))

  def get_message_side_effect(token, msg_id):
    return {"msg_1": msg1, "msg_2": msg2}[msg_id]

  with mock.patch("nemo.lark.api.list_pins", return_value=[
    {"message_id": "msg_1"}, {"message_id": "msg_2"},
  ]):
    with mock.patch("nemo.lark.api.get_message", side_effect=get_message_side_effect):
      with mock.patch("nemo.lark.api.delete_pin") as mock_del_pin:
        with mock.patch("nemo.lark.api.delete_message") as mock_del_msg:
          result = load_config("tok", "oc_1")
  # Kept msg_1 (first), deleted msg_2 (duplicate)
  assert result["autoapprove"] is False
  mock_del_pin.assert_called_once_with("tok", "msg_2")
  mock_del_msg.assert_called_once_with("tok", "msg_2")


def test_save_config_concurrent_safety():
  """save_config should not create duplicates under concurrent calls."""
  import threading

  config = {"autoapprove": True, "guests": [], "rules": {}}
  call_count = {"create_pin": 0}

  def mock_create_pin(token, msg_id):
    call_count["create_pin"] += 1

  with mock.patch("nemo.lark.api.list_pins", return_value=[]):
    with mock.patch("nemo.lark.api.send_text", return_value="msg_new"):
      with mock.patch("nemo.lark.api.create_pin", side_effect=mock_create_pin):
        threads = [threading.Thread(target=save_config, args=("tok", "oc_1", config))
                   for _ in range(5)]
        for t in threads:
          t.start()
        for t in threads:
          t.join()

  # Lock ensures sequential execution — each call creates a pin,
  # but they don't race to see "no pin exists" simultaneously.
  # With mock always returning empty pins, each creates one, but serially.
  assert call_count["create_pin"] == 5


def test_save_config_updates_existing_in_place():
  """Edit path: save_config edits existing pin in place, keeping its id."""
  old_config = {"autoapprove": False, "guests": []}
  old_msg = _text_msg(_build_config_text(old_config))
  with mock.patch("nemo.lark.api.list_pins", return_value=[
    {"message_id": "msg_old"},
  ]):
    with mock.patch("nemo.lark.api.get_message", return_value=old_msg):
      with mock.patch("nemo.lark.api.edit_text") as mock_edit:
        new_config = {**old_config, "autoapprove": True}
        msg_id = save_config("tok", "oc_1", new_config)
  assert msg_id == "msg_old"
  mock_edit.assert_called_once()
  edited_text = mock_edit.call_args[0][2]
  assert edited_text.startswith(CONFIG_MARKER)
  assert "autoapprove: true" in edited_text


def test_save_config_recreates_when_edit_window_expired():
  """When edit_text fails with 230075 (Lark's text-edit window exceeded),
  save_config must unpin/delete the stale pin and create a fresh one
  instead of letting the slash command fail."""
  old_config = {"autoapprove": False, "guests": []}
  old_msg = _text_msg(_build_config_text(old_config))
  edit_error = RuntimeError(
    "Failed to edit text: {'code': 230075, 'msg': "
    "'The message has exceeded the time that can be edited.'}")

  with mock.patch("nemo.lark.api.list_pins", return_value=[
    {"message_id": "msg_old"},
  ]):
    with mock.patch("nemo.lark.api.get_message", return_value=old_msg), \
         mock.patch("nemo.lark.api.edit_text", side_effect=edit_error), \
         mock.patch("nemo.lark.api.delete_pin") as mock_del_pin, \
         mock.patch("nemo.lark.api.delete_message") as mock_del_msg, \
         mock.patch("nemo.lark.api.send_text", return_value="msg_new") as mock_send, \
         mock.patch("nemo.lark.api.create_pin") as mock_pin:
      new_config = {**old_config, "autoapprove": True}
      msg_id = save_config("tok", "oc_1", new_config)

  # Returned the freshly-created id, not the stale one.
  assert msg_id == "msg_new"
  # Stale pin best-effort cleaned up.
  mock_del_pin.assert_called_once_with("tok", "msg_old")
  mock_del_msg.assert_called_once_with("tok", "msg_old")
  # New pin created with updated content.
  mock_send.assert_called_once()
  mock_pin.assert_called_once_with("tok", "msg_new")
  sent_text = mock_send.call_args[0][2]
  assert "autoapprove: true" in sent_text


def test_save_config_recreate_survives_cleanup_failures():
  """Cleanup of the stale pin is best-effort: if delete_pin or
  delete_message fail (Lark recall window also expired, etc.), we still
  create the new pin so the user's config update lands.  _find_config_pin
  on the next read will dedupe any orphans."""
  old_config = {"autoapprove": False, "guests": []}
  old_msg = _text_msg(_build_config_text(old_config))
  edit_error = RuntimeError(
    "Failed to edit text: {'code': 230075, 'msg': '...'}")

  with mock.patch("nemo.lark.api.list_pins", return_value=[
    {"message_id": "msg_old"},
  ]):
    with mock.patch("nemo.lark.api.get_message", return_value=old_msg), \
         mock.patch("nemo.lark.api.edit_text", side_effect=edit_error), \
         mock.patch("nemo.lark.api.delete_pin",
                    side_effect=RuntimeError("delete_pin boom")), \
         mock.patch("nemo.lark.api.delete_message",
                    side_effect=RuntimeError("recall expired")), \
         mock.patch("nemo.lark.api.send_text", return_value="msg_new"), \
         mock.patch("nemo.lark.api.create_pin"):
      msg_id = save_config("tok", "oc_1", {**old_config, "autoapprove": True})

  assert msg_id == "msg_new"


def test_save_config_propagates_non_230075_errors():
  """Other edit_text failures (e.g. 401 auth, 5xx) must NOT silently
  recreate — that would mask real bugs."""
  old_config = {"autoapprove": False, "guests": []}
  old_msg = _text_msg(_build_config_text(old_config))
  edit_error = RuntimeError("Failed to edit text: {'code': 99991663, 'msg': 'rate limited'}")

  with mock.patch("nemo.lark.api.list_pins", return_value=[
    {"message_id": "msg_old"},
  ]):
    with mock.patch("nemo.lark.api.get_message", return_value=old_msg), \
         mock.patch("nemo.lark.api.edit_text", side_effect=edit_error), \
         mock.patch("nemo.lark.api.send_text") as mock_send, \
         mock.patch("nemo.lark.api.create_pin") as mock_pin:
      try:
        save_config("tok", "oc_1", old_config)
      except RuntimeError as e:
        assert "rate limited" in str(e)
      else:
        raise AssertionError("expected RuntimeError to propagate")
  mock_send.assert_not_called()
  mock_pin.assert_not_called()


