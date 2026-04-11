"""Tests for nemo.guests — guest management."""

import copy
from unittest import mock

import pytest

from nemo.group_config import DEFAULT_CONFIG


def _fresh_config(**overrides):
  """Return a deep copy of DEFAULT_CONFIG with overrides applied."""
  cfg = copy.deepcopy(DEFAULT_CONFIG)
  cfg.update(overrides)
  return cfg


from nemo.guests import (
  add_guest, get_member_roles, is_authorized_sender,
  list_guests, remove_guest,
)


# ---------------------------------------------------------------------------
# list_guests
# ---------------------------------------------------------------------------

def test_list_guests_empty():
  with mock.patch("nemo.group_config.load_config", return_value=_fresh_config()):
    result = list_guests("tok", "oc_1")
  assert result == []


def test_list_guests_with_entries():
  config = _fresh_config(guests=[
    {"open_id": "ou_a", "name": "Alice", "role": "coowner"},
    {"open_id": "ou_b", "name": "Bob", "role": "guest"},
  ])
  with mock.patch("nemo.group_config.load_config", return_value=config):
    result = list_guests("tok", "oc_1")
  assert len(result) == 2
  assert result[0]["name"] == "Alice"


# ---------------------------------------------------------------------------
# add_guest
# ---------------------------------------------------------------------------

def test_add_guest_new():
  saved = {}

  def fake_save(token, chat_id, cfg):
    saved.update(cfg)
    return "msg_new"

  with mock.patch("nemo.group_config.load_config", return_value=_fresh_config()):
    with mock.patch("nemo.group_config.save_config", side_effect=fake_save):
      add_guest("tok", "oc_1", "ou_a", name="Alice", role="guest")

  assert len(saved["guests"]) == 1
  assert saved["guests"][0]["open_id"] == "ou_a"
  assert saved["guests"][0]["role"] == "guest"


def test_add_guest_updates_existing():
  config = _fresh_config(guests=[
    {"open_id": "ou_a", "name": "Alice", "role": "guest"},
  ])
  saved = {}

  def fake_save(token, chat_id, cfg):
    saved.update(cfg)
    return "msg_new"

  with mock.patch("nemo.group_config.load_config", return_value=config):
    with mock.patch("nemo.group_config.save_config", side_effect=fake_save):
      add_guest("tok", "oc_1", "ou_a", name="Alice", role="coowner")

  assert len(saved["guests"]) == 1
  assert saved["guests"][0]["role"] == "coowner"


def test_add_guest_invalid_role():
  with pytest.raises(ValueError, match="Invalid role"):
    add_guest("tok", "oc_1", "ou_a", role="admin")


# ---------------------------------------------------------------------------
# remove_guest
# ---------------------------------------------------------------------------

def test_remove_guest():
  config = _fresh_config(guests=[
    {"open_id": "ou_a", "name": "Alice", "role": "coowner"},
    {"open_id": "ou_b", "name": "Bob", "role": "guest"},
  ])
  saved = {}

  def fake_save(token, chat_id, cfg):
    saved.update(cfg)
    return "msg_new"

  with mock.patch("nemo.group_config.load_config", return_value=config):
    with mock.patch("nemo.group_config.save_config", side_effect=fake_save):
      remove_guest("tok", "oc_1", "ou_a")

  assert len(saved["guests"]) == 1
  assert saved["guests"][0]["open_id"] == "ou_b"


def test_remove_guest_not_found():
  """Removing a non-existent guest should be a no-op (no error)."""
  saved = {}

  def fake_save(token, chat_id, cfg):
    saved.update(cfg)
    return "msg_new"

  with mock.patch("nemo.group_config.load_config", return_value=_fresh_config()):
    with mock.patch("nemo.group_config.save_config", side_effect=fake_save):
      remove_guest("tok", "oc_1", "ou_nonexistent")

  assert saved["guests"] == []


# ---------------------------------------------------------------------------
# get_member_roles
# ---------------------------------------------------------------------------

def test_get_member_roles():
  config = _fresh_config(guests=[
    {"open_id": "ou_a", "name": "Alice", "role": "coowner"},
    {"open_id": "ou_b", "name": "Bob", "role": "guest"},
  ])
  with mock.patch("nemo.group_config.load_config", return_value=config):
    roles = get_member_roles("tok", "oc_1")
  assert roles == {"ou_a": "coowner", "ou_b": "guest"}


def test_get_member_roles_empty():
  with mock.patch("nemo.group_config.load_config", return_value=_fresh_config()):
    roles = get_member_roles("tok", "oc_1")
  assert roles == {}


# ---------------------------------------------------------------------------
# is_authorized_sender
# ---------------------------------------------------------------------------

def test_is_authorized_operator():
  assert is_authorized_sender("ou_op", "ou_op", {}) is True


def test_is_authorized_coowner():
  roles = {"ou_a": "coowner"}
  assert is_authorized_sender("ou_a", "ou_op", roles) is True


def test_is_authorized_guest():
  roles = {"ou_b": "guest"}
  assert is_authorized_sender("ou_b", "ou_op", roles) is True


def test_is_unauthorized():
  roles = {"ou_a": "coowner"}
  assert is_authorized_sender("ou_unknown", "ou_op", roles) is False


# ---------------------------------------------------------------------------
# Command dispatch for /guest
# ---------------------------------------------------------------------------

from nemo.commands import try_dispatch, AgentContext


def _ctx():
  return AgentContext(model="opus", project_dir="/tmp/test", start_time=0)


def test_guest_list_command():
  handled, resp = try_dispatch("/guest list", _ctx())
  assert handled
  assert resp == "__guest_list__"


def test_guest_add_command():
  handled, resp = try_dispatch("/guest add Alice", _ctx())
  assert handled
  assert resp == "__guest_add__:guest:Alice"


def test_guest_remove_command():
  handled, resp = try_dispatch("/guest remove Bob", _ctx())
  assert handled
  assert resp == "__guest_remove__:Bob"


def test_guest_no_subcommand():
  handled, resp = try_dispatch("/guest", _ctx())
  assert handled
  assert "Guest Commands" in resp


def test_guest_invalid_subcommand():
  handled, resp = try_dispatch("/guest foo", _ctx())
  assert handled
  assert "Usage" in resp


def test_guest_add_missing_name():
  handled, resp = try_dispatch("/guest add", _ctx())
  assert handled
  assert "Usage" in resp


# ---------------------------------------------------------------------------
# /guest add with coowner role
# ---------------------------------------------------------------------------

def test_guest_add_coowner():
  """/guest add Alice coowner produces __guest_add__:coowner:Alice."""
  handled, resp = try_dispatch("/guest add Alice coowner", _ctx())
  assert handled
  assert resp == "__guest_add__:coowner:Alice"


def test_guest_add_coowner_case_insensitive():
  """/guest add Alice Coowner (mixed case) still resolves to coowner."""
  handled, resp = try_dispatch("/guest add Alice Coowner", _ctx())
  assert handled
  assert resp == "__guest_add__:coowner:Alice"


def test_guest_add_default_role():
  """/guest add Alice (no role) defaults to guest."""
  handled, resp = try_dispatch("/guest add Alice", _ctx())
  assert handled
  assert resp == "__guest_add__:guest:Alice"


def test_guest_add_unknown_role_ignored():
  """/guest add Alice admin — unknown role word is ignored, defaults to guest."""
  handled, resp = try_dispatch("/guest add Alice admin", _ctx())
  assert handled
  assert resp == "__guest_add__:guest:Alice"


# ---------------------------------------------------------------------------
# Token round-trip: parsing tokens back into role + name
# ---------------------------------------------------------------------------

def test_guest_add_token_roundtrip_guest():
  """Token from /guest add Alice can be split back into role='guest', name='Alice'."""
  _, token = try_dispatch("/guest add Alice", _ctx())
  assert token.startswith("__guest_add__:")
  _, rest = token.split(":", 1)
  role, name = rest.split(":", 1)
  assert role == "guest"
  assert name == "Alice"


def test_guest_add_all_as_guest():
  """/guest add all produces __guest_add_all__:guest."""
  handled, resp = try_dispatch("/guest add all", _ctx())
  assert handled
  assert resp == "__guest_add_all__:guest"


def test_guest_add_all_as_coowner():
  """/guest add all coowner produces __guest_add_all__:coowner."""
  handled, resp = try_dispatch("/guest add all coowner", _ctx())
  assert handled
  assert resp == "__guest_add_all__:coowner"


def test_guest_add_all_case_insensitive():
  """/guest add ALL should work."""
  handled, resp = try_dispatch("/guest add ALL", _ctx())
  assert handled
  assert resp == "__guest_add_all__:guest"


def test_guest_add_token_roundtrip_coowner():
  """Token from /guest add Alice coowner can be split back into role='coowner', name='Alice'."""
  _, token = try_dispatch("/guest add Alice coowner", _ctx())
  assert token.startswith("__guest_add__:")
  _, rest = token.split(":", 1)
  role, name = rest.split(":", 1)
  assert role == "coowner"
  assert name == "Alice"


def test_guest_remove_token_roundtrip():
  """Token from /guest remove Bob can be split back into name='Bob'."""
  _, token = try_dispatch("/guest remove Bob", _ctx())
  assert token.startswith("__guest_remove__:")
  name = token.split(":", 1)[1]
  assert name == "Bob"


def test_guest_add_name_with_no_extra_colons():
  """Ensure the token format has exactly the expected structure (no stray colons)."""
  _, token = try_dispatch("/guest add Charlie coowner", _ctx())
  # Format: __guest_add__:coowner:Charlie — split on first ':' gives prefix + rest
  prefix, rest = token.split(":", 1)
  assert prefix == "__guest_add__"
  parts = rest.split(":", 1)
  assert len(parts) == 2
  assert parts[0] == "coowner"
  assert parts[1] == "Charlie"
