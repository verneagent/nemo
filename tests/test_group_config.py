"""Tests for nemo.group_config — pinned text message configuration."""

import json
from unittest import mock

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
  json_part = text[len(CONFIG_MARKER):].strip()
  assert json.loads(json_part)["autoapprove"] is True


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


def test_parse_config_text_invalid_json():
  msg = _text_msg(f"{CONFIG_MARKER}\nnot json")
  assert _parse_config_text(msg) is None


def test_parse_config_text_no_valid_keys():
  msg = _text_msg(f'{CONFIG_MARKER}\n{{"random": 1}}')
  assert _parse_config_text(msg) is None


def test_parse_config_text_empty():
  assert _parse_config_text({}) is None
  assert _parse_config_text({"msg_type": "text", "body": {}}) is None


def test_load_config_found():
  config = {"autoapprove": True, "guests": [], "filter": "verbose", "rules": {}}
  text = _build_config_text(config)
  msg = _text_msg(text)
  with mock.patch("nemo.lark.api.list_pins", return_value=[
    {"message_id": "msg_1"},
  ]):
    with mock.patch("nemo.lark.api.get_message", return_value=msg):
      result = load_config("tok", "oc_1")
  assert result["autoapprove"] is True
  assert result["filter"] == "verbose"


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
  assert result["filter"] == "concise"


def test_save_config_creates_new():
  config = {"autoapprove": True, "guests": [], "filter": "concise", "rules": {}}
  with mock.patch("nemo.lark.api.list_pins", return_value=[]):
    with mock.patch("nemo.lark.api.send_text", return_value="msg_new") as mock_send:
      with mock.patch("nemo.lark.api.create_pin") as mock_pin:
        msg_id = save_config("tok", "oc_1", config)
  assert msg_id == "msg_new"
  mock_pin.assert_called_once_with("tok", "msg_new")
  # Verify the text contains the marker
  sent_text = mock_send.call_args[0][2]
  assert sent_text.startswith(CONFIG_MARKER)


def test_save_config_updates_existing():
  old_config = {"autoapprove": False, "guests": []}
  old_text = _build_config_text(old_config)
  old_msg = _text_msg(old_text)
  with mock.patch("nemo.lark.api.list_pins", return_value=[
    {"message_id": "msg_old"},
  ]):
    with mock.patch("nemo.lark.api.get_message", return_value=old_msg):
      with mock.patch("nemo.lark.api.delete_pin"):
        with mock.patch("nemo.lark.api.delete_message"):
          with mock.patch("nemo.lark.api.send_text", return_value="msg_new"):
            with mock.patch("nemo.lark.api.create_pin"):
              new_config = {**old_config, "autoapprove": True}
              msg_id = save_config("tok", "oc_1", new_config)
  assert msg_id == "msg_new"
