"""Tests for _to_incoming enrichment and helpers in nemo.lark_channel."""

import asyncio
import json
import urllib.error
from unittest import mock

from nemo.channel import IncomingMessage
from nemo.lark.events import LarkEvent, LarkEventStream
from nemo.lark_channel import LarkChannel, _to_incoming, _extract_message_text


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


@mock.patch("nemo.status_tab.update_status")
def test_update_status_refreshes_vision_capability(mock_status):
  """update_status is the chokepoint that keeps _model_sees_images current as
  the model changes (startup, /model, /agent)."""
  ch = LarkChannel.__new__(LarkChannel)
  ch.chat_id = "oc_x"
  ch.credentials = {"app_id": "a", "app_secret": "s"}
  with mock.patch("nemo.lark_channel.lark_auth.get_token", return_value="t"):
    asyncio.run(ch.update_status("claude-opus-4-7", "idle", "claude"))
    assert ch._model_sees_images is True
    asyncio.run(ch.update_status("deepseek-v4-pro", "idle", "claude"))
    assert ch._model_sees_images is False


@mock.patch("nemo.lark_channel.lark_api.download_image", return_value="/tmp/img.png")
def test_image_no_vision_adds_nemo_vision_hint(mock_dl):
  """A model without native vision gets pointed at nemo-vision (and away from
  Read) for the image it just received."""
  ev = _make_event(msg_type="image", image_key="ik_1", text="")
  msg = _to_incoming(ev, token=TOKEN, model_sees_images=False)
  assert "[image: /tmp/img.png]" in msg.text
  assert "nemo-vision" in msg.text
  assert "Read tool" in msg.text


@mock.patch("nemo.lark_channel.lark_api.download_image", return_value="/tmp/img.png")
def test_image_with_vision_has_no_hint(mock_dl):
  """A vision-capable model (default) sees the image itself — no hint."""
  ev = _make_event(msg_type="image", image_key="ik_1", text="")
  msg = _to_incoming(ev, token=TOKEN, model_sees_images=True)
  assert "[image: /tmp/img.png]" in msg.text
  assert "nemo-vision" not in msg.text


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
# Video ("media") enrichment
# ---------------------------------------------------------------------------

@mock.patch("nemo.lark_channel.lark_api.download_file", return_value="/tmp/v.mp4")
def test_media_event_emits_video_marker_and_hint(mock_dl):
  """A Lark video (media) message downloads the file and emits a [video: path]
  marker plus a hint to run nemo-vision (no agent sees video natively)."""
  ev = _make_event(msg_type="media", file_key="fk_v", file_name="clip.mp4",
                   text="")
  msg = _to_incoming(ev, token=TOKEN)
  mock_dl.assert_called_once_with(TOKEN, "om_msg1", "fk_v", "clip.mp4")
  assert "[video: /tmp/v.mp4]" in msg.text
  assert "nemo-vision" in msg.text


@mock.patch("nemo.lark_channel.lark_api.download_file", return_value="/tmp/v.mp4")
def test_media_event_appends_when_text_present(mock_dl):
  ev = _make_event(msg_type="media", file_key="fk_v", file_name="clip.mp4",
                   text="看一下这个视频")
  msg = _to_incoming(ev, token=TOKEN)
  assert msg.text.startswith("看一下这个视频")
  assert "[video: /tmp/v.mp4]" in msg.text


@mock.patch("nemo.lark_channel.lark_api.download_image", return_value="/tmp/thumb.png")
@mock.patch("nemo.lark_channel.lark_api.download_file", return_value="/tmp/v.mp4")
def test_media_skips_thumbnail_image(mock_file, mock_img):
  """A media message carries image_key (the video thumbnail). Only the video
  is downloaded; the thumbnail image must not produce a stray [image:] marker."""
  ev = _make_event(msg_type="media", file_key="fk_v", file_name="clip.mp4",
                   image_key="thumb_key", text="")
  msg = _to_incoming(ev, token=TOKEN)
  mock_file.assert_called_once()
  mock_img.assert_not_called()
  assert "[image:" not in msg.text
  assert "[video: /tmp/v.mp4]" in msg.text


@mock.patch("nemo.lark_channel.lark_api.download_file", return_value="/tmp/fk_v.mp4")
def test_media_without_filename_defaults_mp4(mock_dl):
  ev = _make_event(msg_type="media", file_key="fk_v", file_name="", text="")
  _to_incoming(ev, token=TOKEN)
  mock_dl.assert_called_once_with(TOKEN, "om_msg1", "fk_v", "fk_v.mp4")


@mock.patch("nemo.lark_channel.lark_api.download_file",
            side_effect=RuntimeError("network"))
def test_media_download_failure_graceful(mock_dl):
  ev = _make_event(msg_type="media", file_key="fk_v", file_name="clip.mp4",
                   text="")
  msg = _to_incoming(ev, token=TOKEN)
  assert "failed" in msg.text.lower() or "clip.mp4" in msg.text
  # Should not raise.


# ---------------------------------------------------------------------------
# Reply (parent_id) enrichment
# ---------------------------------------------------------------------------

@mock.patch("nemo.lark_channel.lark_api.get_message", return_value={
  "msg_type": "text",
  "body": {"content": json.dumps({"text": "original question"})},
})
def test_reply_includes_parent_as_secondary_context(mock_get):
  """User's own text must come first; the quoted parent is secondary
  context. Otherwise the model anchors on the quote and ignores the
  user's actual instruction."""
  ev = _make_event(parent_id="om_parent1")
  msg = _to_incoming(ev, token=TOKEN)
  mock_get.assert_called_once_with(TOKEN, "om_parent1")
  assert "original question" in msg.text
  # User text "hello" leads; the quoted context appears afterward.
  assert msg.text.index("hello") < msg.text.index("original question")
  # Framing makes it clear the quote is reference, not an instruction.
  assert "reference context" in msg.text or "earlier message" in msg.text


@mock.patch("nemo.lark_channel.lark_api.get_message", side_effect=RuntimeError("not found"))
def test_reply_parent_fetch_failure_graceful(mock_get):
  ev = _make_event(parent_id="om_parent1")
  msg = _to_incoming(ev, token=TOKEN)
  # Should fall back to original text without crashing
  assert "hello" in msg.text


@mock.patch("nemo.lark_channel.lark_api.get_message")
def test_reply_parent_lookup_preferred_over_api(mock_get):
  """parent_lookup (DB) wins over get_message — needed because Lark's
  get_message returns no body for interactive cards, so quoting a bot
  card previously yielded just '[interactive]'."""
  ev = _make_event(parent_id="om_card1")
  msg = _to_incoming(
    ev, token=TOKEN,
    parent_lookup=lambda mid: "Here is the delete account flow…",
  )
  mock_get.assert_not_called()
  assert "delete account flow" in msg.text
  # User text leads; quote is appended as secondary context.
  assert msg.text.index("hello") < msg.text.index("delete account flow")


@mock.patch("nemo.lark_channel.lark_api.get_message", return_value={
  "msg_type": "text",
  "body": {"content": json.dumps({"text": "from api"})},
})
def test_reply_parent_lookup_falls_through_to_api(mock_get):
  """When parent_lookup returns None, fall back to get_message API."""
  ev = _make_event(parent_id="om_parent2")
  msg = _to_incoming(
    ev, token=TOKEN,
    parent_lookup=lambda mid: None,
  )
  mock_get.assert_called_once_with(TOKEN, "om_parent2")
  assert "from api" in msg.text


# Jenkins bot CI card — real-world payload shape from list_messages
_JENKINS_CARD = {
  "msg_type": "interactive",
  "body": {"content": json.dumps({
    "title": "FiveD iOS Build (preview)",
    "elements": [
      [{"tag": "text", "text": "✅ Checkout\n⏳ EAS Build (local)"}],
      [{"tag": "hr"}],
      [{"tag": "a", "href": "https://x", "text": "79f8b32"},
       {"tag": "text", "text": " Merge main into HEAD — CI"}],
    ],
  })},
}


def test_extract_interactive_card_text():
  """Interactive cards (e.g. from other bots) should yield readable text
  instead of the opaque '[interactive]' placeholder."""
  text = _extract_message_text(_JENKINS_CARD)
  assert "FiveD iOS Build" in text
  assert "✅ Checkout" in text
  assert "79f8b32" in text
  assert "Merge main into HEAD" in text
  assert "[interactive]" not in text


@mock.patch("nemo.lark_channel.lark_api.get_message", return_value=_JENKINS_CARD)
def test_reply_to_other_bot_card_includes_card_text(mock_get):
  """User replies to a Jenkins/other bot card — nemo should see the
  card's actual text in the prompt, not just '[interactive]'."""
  ev = _make_event(parent_id="om_jenkins1")
  msg = _to_incoming(ev, token=TOKEN)
  assert "FiveD iOS Build" in msg.text
  assert "[interactive]" not in msg.text


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
# is_internal round trip — /session recall injects a synthetic message with
# is_internal=True so the main loop skips its auth/mention filters. The flag
# must survive push_back -> next_event -> _to_incoming or the injected recall
# message gets dropped and the user never gets a summary.
# ---------------------------------------------------------------------------

def test_to_incoming_preserves_is_internal():
  ev = _make_event(text="recall me", is_internal=True)
  msg = _to_incoming(ev, token="")
  assert msg.is_internal is True


def test_push_back_round_trip_preserves_is_internal():
  ch = LarkChannel.__new__(LarkChannel)
  ch.chat_id = "oc_recall"
  ch.credentials = {"app_id": "cli_x", "app_secret": "s"}
  ch.parent_lookup = None
  ch._chat_mode = ""
  ch._reply_anchor = ""
  ch._events = LarkEventStream("cli_x", "s")  # queue is ready without start()
  injected = IncomingMessage(
    event_type="im.message.receive_v1",
    chat_id="oc_recall",
    sender_id="",  # synthetic — no real user
    message_id="recall_abc",
    msg_type="text",
    text="[Nemo recall] summarize this transcript",
    create_time="1",
    is_internal=True,
  )
  ch.push_back(injected)
  with mock.patch(
      "nemo.lark_channel.lark_auth.get_token", return_value="t-fake"):
    received = asyncio.run(ch.receive(timeout=1))
  assert received is not None
  assert received.is_internal is True
  assert received.text == "[Nemo recall] summarize this transcript"
  assert received.message_id == "recall_abc"


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


# ---------------------------------------------------------------------------
# Topic-chat routing (chat_mode="topic")
# ---------------------------------------------------------------------------

def _build_topic_channel() -> LarkChannel:
  """Construct a LarkChannel without running start() so unit tests
  can exercise topic routing without touching real credentials."""
  ch = LarkChannel.__new__(LarkChannel)
  ch.chat_id = "oc_topic"
  ch.credentials = {"app_id": "cli_x", "app_secret": "s"}
  ch._events = None
  ch.parent_lookup = None
  ch._chat_mode = "topic"
  ch._reply_anchor = ""
  return ch


def test_send_card_routes_to_reply_in_topic():
  """In topic mode with an anchor, send_card must use reply_card with
  reply_in_thread=True so the card lands in the current topic."""
  ch = _build_topic_channel()
  ch._reply_anchor = "om_user_msg"
  card = {"title": "hi"}
  with mock.patch("nemo.lark_channel.lark_auth.get_token", return_value="t"), \
       mock.patch("nemo.lark_channel.lark_api.reply_card",
                  return_value="om_bot_msg") as mock_reply, \
       mock.patch("nemo.lark_channel.lark_api.send_card") as mock_send:
    result = asyncio.run(ch.send_card("oc_topic", card))
  assert result == "om_bot_msg"
  mock_reply.assert_called_once_with("t", "om_user_msg", card, True)
  mock_send.assert_not_called()


def test_send_text_routes_to_reply_in_topic():
  ch = _build_topic_channel()
  ch._reply_anchor = "om_user_msg"
  with mock.patch("nemo.lark_channel.lark_auth.get_token", return_value="t"), \
       mock.patch("nemo.lark_channel.lark_api.reply_message",
                  return_value="om_bot_msg") as mock_reply, \
       mock.patch("nemo.lark_channel.lark_api.send_text") as mock_send:
    result = asyncio.run(ch.send_text("oc_topic", "hello"))
  assert result == "om_bot_msg"
  mock_reply.assert_called_once_with("t", "om_user_msg", "hello", True)
  mock_send.assert_not_called()


def test_send_card_plain_in_topic_without_anchor():
  """Before any user message arrives (e.g. startup card), send_card
  must not try to reply — it falls back to the plain send endpoint so
  the card creates a fresh topic instead of crashing."""
  ch = _build_topic_channel()
  card = {"title": "nemo up"}
  with mock.patch("nemo.lark_channel.lark_auth.get_token", return_value="t"), \
       mock.patch("nemo.lark_channel.lark_api.send_card",
                  return_value="om_start") as mock_send, \
       mock.patch("nemo.lark_channel.lark_api.reply_card") as mock_reply:
    result = asyncio.run(ch.send_card("oc_topic", card))
  assert result == "om_start"
  mock_send.assert_called_once_with("t", "oc_topic", card)
  mock_reply.assert_not_called()


def _make_http_error(code: int) -> urllib.error.HTTPError:
  return urllib.error.HTTPError(
    "https://test/url", code, "Forbidden", hdrs=None, fp=None)  # type: ignore[arg-type]


def test_update_card_retries_on_http_403():
  """HTTP 403 should invalidate token and retry once with fresh token."""
  ch = _build_topic_channel()
  ch._chat_mode = "group"
  call_count = 0

  def fake_update(_token, _msg_id, _card):
    nonlocal call_count
    call_count += 1
    if call_count == 1:
      raise _make_http_error(403)

  with mock.patch("nemo.lark_channel.lark_auth.get_token", return_value="t"), \
       mock.patch("nemo.lark_channel.lark_auth.invalidate") as mock_invalidate, \
       mock.patch("nemo.lark_channel.lark_api.update_card", side_effect=fake_update):
    asyncio.run(ch.update_card("om_1", {"title": "x"}))
  assert call_count == 2
  mock_invalidate.assert_called_once()


def test_update_card_falls_back_to_new_card_when_refresh_fails():
  """If token refresh doesn't fix 403, send as a new card instead of
  surfacing the error — the user still sees the response. The new
  message_id is returned so callers can retarget subsequent updates."""
  ch = _build_topic_channel()
  ch._chat_mode = "group"

  with mock.patch("nemo.lark_channel.lark_auth.get_token", return_value="t"), \
       mock.patch("nemo.lark_channel.lark_auth.invalidate"), \
       mock.patch("nemo.lark_channel.lark_api.update_card",
                  side_effect=_make_http_error(403)), \
       mock.patch("nemo.lark_channel.lark_api.send_card",
                  return_value="om_new") as mock_send:
    new_id = asyncio.run(ch.update_card("om_old", {"title": "x"}))
  mock_send.assert_called_once()
  assert new_id == "om_new"


def test_update_card_topic_fallback_replies_to_failed_card():
  """In topic chats the fallback card must reply to the failed message,
  not to self._reply_anchor (which may have drifted)."""
  ch = _build_topic_channel()
  ch._reply_anchor = "om_newer_inbound"  # drifted since old card was posted

  with mock.patch("nemo.lark_channel.lark_auth.get_token", return_value="t"), \
       mock.patch("nemo.lark_channel.lark_auth.invalidate"), \
       mock.patch("nemo.lark_channel.lark_api.update_card",
                  side_effect=_make_http_error(403)), \
       mock.patch("nemo.lark_channel.lark_api.reply_card",
                  return_value="om_new") as mock_reply, \
       mock.patch("nemo.lark_channel.lark_api.send_card") as mock_send:
    new_id = asyncio.run(ch.update_card("om_original", {"title": "x"}))
  assert new_id == "om_new"
  # Anchored to the FAILED card, not the drifted _reply_anchor.
  mock_reply.assert_called_once_with("t", "om_original", {"title": "x"}, True)
  mock_send.assert_not_called()


def test_update_card_returns_input_id_on_success():
  """When update succeeds in-place, the input message_id is returned."""
  ch = _build_topic_channel()
  ch._chat_mode = "group"

  with mock.patch("nemo.lark_channel.lark_auth.get_token", return_value="t"), \
       mock.patch("nemo.lark_channel.lark_api.update_card"):
    result = asyncio.run(ch.update_card("om_1", {"title": "x"}))
  assert result == "om_1"


def test_update_card_propagates_non_auth_http_errors():
  """Non-auth HTTP errors (500 etc.) must propagate — no fallback."""
  ch = _build_topic_channel()
  ch._chat_mode = "group"

  with mock.patch("nemo.lark_channel.lark_auth.get_token", return_value="t"), \
       mock.patch("nemo.lark_channel.lark_api.update_card",
                  side_effect=_make_http_error(500)), \
       mock.patch("nemo.lark_channel.lark_api.send_card") as mock_send:
    try:
      asyncio.run(ch.update_card("om_1", {"title": "x"}))
      assert False, "expected HTTPError to propagate"
    except urllib.error.HTTPError as e:
      assert e.code == 500
  mock_send.assert_not_called()


def test_send_card_plain_in_non_topic_chat():
  """Non-topic chats keep the existing behavior even when an anchor
  happens to be set — other code paths rely on plain sends."""
  ch = _build_topic_channel()
  ch._chat_mode = "group"
  ch._reply_anchor = "om_whatever"
  card = {"title": "x"}
  with mock.patch("nemo.lark_channel.lark_auth.get_token", return_value="t"), \
       mock.patch("nemo.lark_channel.lark_api.send_card",
                  return_value="om_sent") as mock_send, \
       mock.patch("nemo.lark_channel.lark_api.reply_card") as mock_reply:
    result = asyncio.run(ch.send_card("oc_topic", card))
  assert result == "om_sent"
  mock_send.assert_called_once()
  mock_reply.assert_not_called()
