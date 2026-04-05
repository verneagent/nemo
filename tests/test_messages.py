"""Tests for nemo.messages — message filtering and prompt building."""

from dataclasses import dataclass, field

from nemo.messages import (
  build_prompt, strip_mentions, filter_self_bot,
  filter_by_operator, filter_by_allowed_senders,
  filter_bot_interactions,
)


# Works with both dicts and objects
@dataclass
class FakeMsg:
  text: str = ""
  sender_id: str = ""
  image_key: str = ""
  file_key: str = ""
  msg_type: str = "text"
  parent_id: str = ""
  mentions: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

def test_build_prompt_text_only():
  replies = [{"text": "hello"}, {"text": "world"}]
  assert build_prompt(replies) == "hello\nworld"


def test_build_prompt_with_media():
  replies = [{"text": "see this", "image_key": "img_1", "msg_type": "image"}]
  result = build_prompt(replies)
  assert "img_1" in result  # JSON output


def test_build_prompt_with_parent_id():
  replies = [{"text": "reply", "parent_id": "om_parent"}]
  result = build_prompt(replies)
  assert "parent_id" in result  # JSON output


def test_build_prompt_with_objects():
  replies = [FakeMsg(text="hello"), FakeMsg(text="world")]
  assert build_prompt(replies) == "hello\nworld"


def test_build_prompt_objects_with_media():
  replies = [FakeMsg(text="see", image_key="img_1", msg_type="image")]
  result = build_prompt(replies)
  assert "img_1" in result


# ---------------------------------------------------------------------------
# strip_mentions
# ---------------------------------------------------------------------------

def test_strip_mentions_dict():
  text = "@bot hello there"
  replies = [{"mentions": [{"key": "@bot"}]}]
  assert strip_mentions(text, replies) == "hello there"


def test_strip_mentions_object():
  text = "@bot hello there"
  replies = [FakeMsg(mentions=[{"key": "@bot"}])]
  assert strip_mentions(text, replies) == "hello there"


def test_strip_mentions_no_mentions():
  assert strip_mentions("hello", [{"mentions": []}]) == "hello"


# ---------------------------------------------------------------------------
# filter_self_bot
# ---------------------------------------------------------------------------

def test_filter_self_bot_dict():
  replies = [{"sender_id": "ou_bot"}, {"sender_id": "ou_user"}]
  filtered = filter_self_bot(replies, "ou_bot")
  assert len(filtered) == 1
  assert filtered[0]["sender_id"] == "ou_user"


def test_filter_self_bot_object():
  replies = [FakeMsg(sender_id="ou_bot"), FakeMsg(sender_id="ou_user")]
  filtered = filter_self_bot(replies, "ou_bot")
  assert len(filtered) == 1


def test_filter_self_bot_empty_id():
  replies = [{"sender_id": "ou_bot"}]
  assert len(filter_self_bot(replies, "")) == 1


# ---------------------------------------------------------------------------
# filter_by_operator
# ---------------------------------------------------------------------------

def test_filter_by_operator():
  replies = [{"sender_id": "ou_op"}, {"sender_id": "ou_other"}]
  filtered = filter_by_operator(replies, "ou_op")
  assert len(filtered) == 1


def test_filter_by_operator_object():
  replies = [FakeMsg(sender_id="ou_op"), FakeMsg(sender_id="ou_other")]
  filtered = filter_by_operator(replies, "ou_op")
  assert len(filtered) == 1


# ---------------------------------------------------------------------------
# filter_by_allowed_senders
# ---------------------------------------------------------------------------

def test_filter_by_allowed_senders():
  replies = [
    {"sender_id": "ou_op"},
    {"sender_id": "ou_coowner"},
    {"sender_id": "ou_stranger"},
  ]
  roles = {"ou_coowner": "coowner"}
  filtered = filter_by_allowed_senders(replies, "ou_op", roles)
  assert len(filtered) == 2


# ---------------------------------------------------------------------------
# filter_bot_interactions
# ---------------------------------------------------------------------------

def test_filter_bot_interactions_mention():
  replies = [
    {"mentions": [{"id": "ou_bot"}], "msg_type": "text"},
    {"mentions": [], "msg_type": "text"},
  ]
  filtered = filter_bot_interactions(replies, "ou_bot")
  assert len(filtered) == 1


def test_filter_bot_interactions_reply():
  replies = [{"parent_id": "om_1", "msg_type": "text", "mentions": []}]
  filtered = filter_bot_interactions(replies, "ou_bot")
  assert len(filtered) == 1


def test_filter_bot_interactions_reaction():
  replies = [{"msg_type": "reaction", "mentions": []}]
  filtered = filter_bot_interactions(replies, "ou_bot")
  assert len(filtered) == 1


def test_filter_bot_interactions_object():
  replies = [
    FakeMsg(mentions=[{"id": "ou_bot"}]),
    FakeMsg(mentions=[]),
  ]
  filtered = filter_bot_interactions(replies, "ou_bot")
  assert len(filtered) == 1
