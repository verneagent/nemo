"""Tests for nemo.monitor — signal detection."""

from nemo.monitor import (
  is_esc, is_exit, is_dissolve, is_permission_reply, is_privileged, parse_esc,
)


def test_is_esc():
  assert is_esc("/esc")
  assert not is_esc("esc")
  assert not is_esc("cancel")
  assert not is_esc("取消")
  assert not is_esc("hello")
  assert not is_esc("escape")


def test_is_esc_with_mentions():
  assert is_esc("@bot /esc", [{"key": "@bot"}])
  assert not is_esc("@bot esc", [{"key": "@bot"}])


def test_parse_esc_bare():
  assert parse_esc("/esc") == ""
  assert parse_esc("esc") is None
  assert parse_esc("cancel") is None
  assert parse_esc("取消") is None


def test_parse_esc_not_esc():
  assert parse_esc("hello") is None
  assert parse_esc("escape") is None
  assert parse_esc("") is None


def test_parse_esc_with_follow_up():
  assert parse_esc("/esc fix the bug") == "fix the bug"
  assert parse_esc("/esc Use TypeScript") == "Use TypeScript"
  # Preserves multi-line / multi-space content
  assert parse_esc("/esc do A\nthen B") == "do A then B"


def test_parse_esc_with_mentions_and_text():
  assert parse_esc(
    "@bot /esc continue", [{"key": "@bot"}]
  ) == "continue"
  # Mention key in the middle should be stripped without losing case
  assert parse_esc(
    "@bot /esc Use API", [{"key": "@bot"}]
  ) == "Use API"


def test_parse_esc_is_case_insensitive_prefix():
  assert parse_esc("/ESC continue") == "continue"
  assert parse_esc("Cancel keep going") is None


def test_is_exit():
  assert is_exit("/exit")
  assert not is_exit("exit")
  assert not is_exit("hello")
  assert not is_exit("/dissolve")


def test_is_dissolve():
  assert is_dissolve("/dissolve")
  assert not is_dissolve("dissolve")
  assert not is_dissolve("exit")
  assert not is_dissolve("hello")


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


def test_is_privileged():
  assert is_privileged("ou_1", "ou_1")
  assert not is_privileged("ou_2", "ou_1")
  assert is_privileged("ou_2", "ou_1", {"ou_2": "coowner"})
  assert not is_privileged("ou_3", "ou_1", {"ou_2": "coowner"})
  # Guests are not privileged — only coowner is.
  assert not is_privileged("ou_2", "ou_1", {"ou_2": "guest"})
  # No operator = everyone authorized
  assert is_privileged("anyone", "")
