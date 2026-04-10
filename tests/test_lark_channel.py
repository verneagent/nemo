"""Tests for _to_incoming enrichment and helpers in nemo.lark_channel."""

import json
from unittest import mock

from nemo.lark.events import LarkEvent
from nemo.lark_channel import _to_incoming, _extract_message_text


TOKEN = "t-test_token"


def _make_event(**overrides) -> LarkEvent:
  defaults = dict(
    event_type="im.message.receive_v1",
    chat_id="oc_123",
    chat_type="group",
    sender_id="ou_user1",
    message_id="om_msg1",
    msg_type="text",
    text="hello",
  )
  defaults.update(overrides)
  return LarkEvent(**defaults)


# ---------------------------------------------------------------------------
# File enrichment
# ---------------------------------------------------------------------------

@mock.patch("nemo.lark_channel.lark_api.download_file", return_value="/tmp/a.pdf")
def test_file_event_embeds_path(mock_dl):
  ev = _make_event(msg_type="file", file_key="fk_1", file_name="a.pdf", text="")
  msg = _to_incoming(ev, token=TOKEN)
  mock_dl.assert_called_once_with(TOKEN, "om_msg1", "fk_1", "a.pdf")
  assert "/tmp/a.pdf" in msg.text
  assert "a.pdf" in msg.text


@mock.patch("nemo.lark_channel.lark_api.download_file", return_value="/tmp/a.pdf")
def test_file_event_appends_when_text_present(mock_dl):
  ev = _make_event(msg_type="file", file_key="fk_1", file_name="a.pdf", text="see attached")
  msg = _to_incoming(ev, token=TOKEN)
  assert msg.text.startswith("see attached")
  assert "/tmp/a.pdf" in msg.text


@mock.patch("nemo.lark_channel.lark_api.download_file", side_effect=RuntimeError("network"))
def test_file_download_failure_graceful(mock_dl):
  ev = _make_event(msg_type="file", file_key="fk_1", file_name="a.pdf", text="")
  msg = _to_incoming(ev, token=TOKEN)
  assert "failed" in msg.text.lower() or "a.pdf" in msg.text
  # Should not raise


# ---------------------------------------------------------------------------
# Image enrichment
# ---------------------------------------------------------------------------

@mock.patch("nemo.lark_channel.lark_api.download_image", return_value="/tmp/img.png")
def test_image_event_embeds_path(mock_dl):
  ev = _make_event(msg_type="image", image_key="ik_1", text="")
  msg = _to_incoming(ev, token=TOKEN)
  mock_dl.assert_called_once_with(TOKEN, "om_msg1", "ik_1")
  assert "/tmp/img.png" in msg.text


@mock.patch("nemo.lark_channel.lark_api.download_image", return_value="/tmp/img.png")
def test_image_event_appends_when_text_present(mock_dl):
  ev = _make_event(msg_type="image", image_key="ik_1", text="check this")
  msg = _to_incoming(ev, token=TOKEN)
  assert msg.text.startswith("check this")
  assert "/tmp/img.png" in msg.text


@mock.patch("nemo.lark_channel.lark_api.download_image", side_effect=RuntimeError("fail"))
def test_image_download_failure_graceful(mock_dl):
  """Image download failure should not crash; original text preserved."""
  ev = _make_event(msg_type="image", image_key="ik_1", text="caption")
  msg = _to_incoming(ev, token=TOKEN)
  # Text should still be usable (original or with warning)
  assert msg.text  # not empty


@mock.patch("nemo.lark_channel.lark_api.download_image", return_value="/tmp/img.png")
def test_post_with_image_placeholder_replaced(mock_dl):
  """Post messages with [image] placeholder should replace it with path."""
  ev = _make_event(msg_type="post", image_key="ik_1", text="[image]\nlook at this")
  msg = _to_incoming(ev, token=TOKEN)
  assert "[image: /tmp/img.png]" in msg.text
  assert "[image]\n" not in msg.text  # placeholder replaced


@mock.patch("nemo.lark_channel.lark_api.download_image", return_value="/tmp/img.png")
def test_post_with_image_no_msg_type_check(mock_dl):
  """Image key triggers download regardless of msg_type."""
  ev = _make_event(msg_type="post", image_key="ik_1", text="see this")
  msg = _to_incoming(ev, token=TOKEN)
  assert "/tmp/img.png" in msg.text


@mock.patch("nemo.lark_channel.lark_api.download_image",
            side_effect=["/tmp/a.png", "/tmp/b.png"])
def test_multiple_image_keys(mock_dl):
  """Comma-separated image keys should all be downloaded."""
  ev = _make_event(msg_type="post", image_key="ik_1,ik_2",
                   text="[image] and [image]")
  msg = _to_incoming(ev, token=TOKEN)
  assert "/tmp/a.png" in msg.text
  assert "/tmp/b.png" in msg.text
  assert mock_dl.call_count == 2


# ---------------------------------------------------------------------------
# Reply (parent_id) enrichment
# ---------------------------------------------------------------------------

@mock.patch("nemo.lark_channel.lark_api.get_message", return_value={
  "msg_type": "text",
  "body": {"content": json.dumps({"text": "original question"})},
})
def test_reply_prepends_parent_context(mock_get):
  ev = _make_event(parent_id="om_parent1")
  msg = _to_incoming(ev, token=TOKEN)
  mock_get.assert_called_once_with(TOKEN, "om_parent1")
  assert "original question" in msg.text
  assert msg.text.index("replying to") < msg.text.index("hello")


@mock.patch("nemo.lark_channel.lark_api.get_message", side_effect=RuntimeError("not found"))
def test_reply_parent_fetch_failure_graceful(mock_get):
  ev = _make_event(parent_id="om_parent1")
  msg = _to_incoming(ev, token=TOKEN)
  # Should fall back to original text without crashing
  assert "hello" in msg.text


# ---------------------------------------------------------------------------
# Plain text — no enrichment
# ---------------------------------------------------------------------------

def test_plain_text_no_enrichment():
  """Plain text event with no file/image/parent should pass through unchanged."""
  ev = _make_event(msg_type="text", text="just a message")
  msg = _to_incoming(ev, token=TOKEN)
  assert msg.text == "just a message"
  assert msg.chat_id == "oc_123"
  assert msg.sender_id == "ou_user1"


# ---------------------------------------------------------------------------
# No token — no enrichment attempted
# ---------------------------------------------------------------------------

@mock.patch("nemo.lark_channel.lark_api.download_file")
@mock.patch("nemo.lark_channel.lark_api.download_image")
@mock.patch("nemo.lark_channel.lark_api.get_message")
def test_no_token_skips_enrichment(mock_get, mock_img, mock_file):
  ev = _make_event(
    msg_type="file", file_key="fk_1", file_name="a.pdf",
    image_key="ik_1", parent_id="om_parent1", text="raw",
  )
  msg = _to_incoming(ev, token="")
  mock_file.assert_not_called()
  mock_img.assert_not_called()
  mock_get.assert_not_called()
  assert msg.text == "raw"


# ---------------------------------------------------------------------------
# _extract_message_text helper
# ---------------------------------------------------------------------------

def test_extract_text_message():
  msg = {"msg_type": "text", "body": {"content": json.dumps({"text": "hi"})}}
  assert _extract_message_text(msg) == "hi"


def test_extract_file_message():
  msg = {"msg_type": "file", "body": {"content": json.dumps({"file_name": "doc.pdf"})}}
  result = _extract_message_text(msg)
  assert "doc.pdf" in result


def test_extract_image_message():
  msg = {"msg_type": "image", "body": {"content": json.dumps({})}}
  assert _extract_message_text(msg) == "[image]"


def test_extract_post_message():
  content = {"zh_cn": {"title": "Hello", "content": [
    [{"text": "line one"}, {"tag": "img", "image_key": "ik_1"}],
    [{"text": "line two"}],
  ]}}
  msg = {"msg_type": "post", "body": {"content": json.dumps(content)}}
  result = _extract_message_text(msg)
  assert "line one" in result
  assert "[image]" in result
  assert "line two" in result
  assert "Hello" in result


def test_extract_post_direct_content():
  """Post with content directly (not locale-keyed)."""
  content = {"content": [[{"text": "direct text"}]]}
  msg = {"msg_type": "post", "body": {"content": json.dumps(content)}}
  result = _extract_message_text(msg)
  assert "direct text" in result


def test_extract_unknown_type():
  msg = {"msg_type": "sticker", "body": {"content": "{}"}}
  assert _extract_message_text(msg) == "[sticker]"


def test_extract_empty_msg_type():
  msg = {"msg_type": "", "body": {"content": "{}"}}
  assert _extract_message_text(msg) == ""


def test_extract_missing_body():
  msg = {"msg_type": "text"}
  result = _extract_message_text(msg)
  # Should not crash; body defaults to {}
  assert isinstance(result, str)
