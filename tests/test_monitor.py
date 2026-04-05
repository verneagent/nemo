"""Tests for nemo.monitor — signal detection."""

from nemo.monitor import is_esc, is_handback, is_permission_reply, is_authorized


def test_is_esc():
  assert is_esc("/esc")
  assert is_esc("esc")
  assert is_esc("cancel")
  assert is_esc("取消")
  assert not is_esc("hello")
  assert not is_esc("escape")


def test_is_esc_with_mentions():
  assert is_esc("@bot /esc", [{"key": "@bot"}])
  assert is_esc("@bot esc", [{"key": "@bot"}])


def test_is_handback():
  assert is_handback("handback")
  assert is_handback("hand back")
  assert is_handback("handback dissolve")
  assert not is_handback("hello")


def test_is_permission_reply():
  assert is_permission_reply("y") == "allow"
  assert is_permission_reply("yes") == "allow"
  assert is_permission_reply("ok") == "allow"
  assert is_permission_reply("允许") == "allow"
  assert is_permission_reply("always") == "always"
  assert is_permission_reply("全部允许") == "always"
  assert is_permission_reply("n") == "deny"
  assert is_permission_reply("no") == "deny"
  assert is_permission_reply("deny") == "deny"
  assert is_permission_reply("拒绝") == "deny"
  assert is_permission_reply("hello") is None
  assert is_permission_reply("") is None


def test_is_authorized():
  assert is_authorized("ou_1", "ou_1")
  assert not is_authorized("ou_2", "ou_1")
  assert is_authorized("ou_2", "ou_1", {"ou_2": "coowner"})
  assert not is_authorized("ou_3", "ou_1", {"ou_2": "coowner"})
  # No operator = everyone authorized
  assert is_authorized("anyone", "")
