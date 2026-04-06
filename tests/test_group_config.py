"""Tests for nemo.group_config — pinned card configuration."""

import json
from unittest import mock

from nemo.group_config import (
  _build_config_card, _parse_config_from_card,
  load_config, save_config, DEFAULT_CONFIG, CONFIG_TITLE,
)


def test_build_config_card_structure():
  config = {"autoapprove": True, "guests": [], "filter": "concise", "rules": {}}
  card = _build_config_card(config)
  assert card["header"]["title"]["content"] == CONFIG_TITLE
  body = card["body"]["elements"][0]["content"]
  assert "```json" in body
  parsed = json.loads(body.strip().split("\n", 1)[1].rsplit("```", 1)[0])
  assert parsed["autoapprove"] is True


def test_parse_config_from_card_direct():
  """Parse config from a card structure (as built)."""
  config = {"autoapprove": False, "guests": []}
  card = _build_config_card(config)
  result = _parse_config_from_card(card)
  assert result is not None
  assert result["autoapprove"] is False


def test_parse_config_from_get_message():
  """Parse config from get_message API response (content as JSON string)."""
  config = {"autoapprove": True, "guests": [], "filter": "concise", "rules": {}}
  card = _build_config_card(config)
  # get_message returns content as a JSON string
  msg = {"content": json.dumps(card)}
  result = _parse_config_from_card(msg)
  assert result is not None
  assert result["autoapprove"] is True


def test_parse_config_invalid():
  assert _parse_config_from_card({}) is None
  assert _parse_config_from_card({"body": {"elements": []}}) is None


def test_parse_config_not_config_card():
  """A card without valid config keys should return None."""
  msg = {"body": {"elements": [{"content": '```json\n{"random": 1}\n```'}]}}
  assert _parse_config_from_card(msg) is None


def test_load_config_found():
  config = {"autoapprove": True, "guests": [], "filter": "verbose", "rules": {}}
  card = _build_config_card(config)
  with mock.patch("nemo.lark.api.list_pins", return_value=[
    {"pin": {"message_id": "msg_1"}},
  ]):
    with mock.patch("nemo.lark.api.get_message", return_value={
      "content": json.dumps(card),
    }):
      result = load_config("tok", "oc_1")
  assert result["autoapprove"] is True
  assert result["filter"] == "verbose"


def test_load_config_not_found():
  with mock.patch("nemo.lark.api.list_pins", return_value=[]):
    result = load_config("tok", "oc_1")
  assert result == DEFAULT_CONFIG


def test_load_config_merges_defaults():
  """Missing keys should be filled from defaults."""
  config = {"autoapprove": True}
  card = _build_config_card(config)
  with mock.patch("nemo.lark.api.list_pins", return_value=[
    {"pin": {"message_id": "msg_1"}},
  ]):
    with mock.patch("nemo.lark.api.get_message", return_value={
      "content": json.dumps(card),
    }):
      result = load_config("tok", "oc_1")
  assert result["autoapprove"] is True
  assert result["guests"] == []  # from default
  assert result["filter"] == "concise"  # from default


def test_save_config_creates_new():
  config = {"autoapprove": True, "guests": [], "filter": "concise", "rules": {}}
  with mock.patch("nemo.lark.api.list_pins", return_value=[]):
    with mock.patch("nemo.lark.api.send_card", return_value="msg_new"):
      with mock.patch("nemo.lark.api.create_pin") as mock_pin:
        msg_id = save_config("tok", "oc_1", config)
  assert msg_id == "msg_new"
  mock_pin.assert_called_once_with("tok", "msg_new")


def test_save_config_updates_existing():
  old_config = {"autoapprove": False, "guests": [], "filter": "concise", "rules": {}}
  old_card = _build_config_card(old_config)
  with mock.patch("nemo.lark.api.list_pins", return_value=[
    {"pin": {"message_id": "msg_old"}},
  ]):
    with mock.patch("nemo.lark.api.get_message", return_value={
      "content": json.dumps(old_card),
    }):
      with mock.patch("nemo.lark.api.update_card") as mock_update:
        new_config = {**old_config, "autoapprove": True}
        msg_id = save_config("tok", "oc_1", new_config)
  assert msg_id == "msg_old"
  mock_update.assert_called_once()


def test_save_config_recreates_on_expiry():
  """When PATCH fails (card expired), should recreate."""
  old_config = {"autoapprove": False, "guests": [], "filter": "concise", "rules": {}}
  old_card = _build_config_card(old_config)
  with mock.patch("nemo.lark.api.list_pins", return_value=[
    {"pin": {"message_id": "msg_old"}},
  ]):
    with mock.patch("nemo.lark.api.get_message", return_value={
      "content": json.dumps(old_card),
    }):
      with mock.patch("nemo.lark.api.update_card", side_effect=RuntimeError("expired")):
        with mock.patch("nemo.lark.api.delete_pin"):
          with mock.patch("nemo.lark.api.delete_message"):
            with mock.patch("nemo.lark.api.send_card", return_value="msg_new"):
              with mock.patch("nemo.lark.api.create_pin"):
                msg_id = save_config("tok", "oc_1", {"autoapprove": True})
  assert msg_id == "msg_new"
