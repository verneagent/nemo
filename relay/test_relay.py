"""Tests for Nemo Relay Server — behavioral equivalence with CF Worker.

Each test documents which CF Worker behavior it verifies.
"""

import asyncio
import json
import os
import threading
import time
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import aiohttp

# Set test config before importing relay
os.environ["RELAY_PORT"] = "19801"
os.environ["RELAY_DB"] = "/private/tmp/claude/test_relay.db"
os.environ["RELAY_API_KEY"] = "test-key"
os.environ["VERIFY_TOKENS"] = "tok1,tok2"

import importlib
import relay
importlib.reload(relay)  # Re-read env vars in case another test changed them

BASE = "http://127.0.0.1:19801"
AUTH = {"Authorization": "Bearer test-key"}


class RelayTestCase(unittest.TestCase):
  runner = None
  site = None
  loop = None
  loop_thread = None

  @classmethod
  def setUpClass(cls):
    # Ensure VERIFY_TOKENS is correct (may have been changed by other tests)
    relay.VERIFY_TOKENS = {"tok1", "tok2"}
    if os.path.exists("/private/tmp/claude/test_relay.db"):
      os.remove("/private/tmp/claude/test_relay.db")
    relay._init_db()

    cls.loop = asyncio.new_event_loop()
    app = relay.create_app()

    async def start():
      cls.runner = relay.web.AppRunner(app)
      await cls.runner.setup()
      cls.site = relay.web.TCPSite(cls.runner, "127.0.0.1", 19801)
      await cls.site.start()

    cls.loop.run_until_complete(start())
    cls.loop_thread = threading.Thread(target=cls.loop.run_forever, daemon=True)
    cls.loop_thread.start()

  @classmethod
  def tearDownClass(cls):
    if cls.runner:
      asyncio.run_coroutine_threadsafe(cls.runner.cleanup(), cls.loop).result(5)
    if cls.loop:
      cls.loop.call_soon_threadsafe(cls.loop.stop)
      if cls.loop_thread:
        cls.loop_thread.join(5)
    if os.path.exists("/private/tmp/claude/test_relay.db"):
      os.remove("/private/tmp/claude/test_relay.db")

  # --- helpers ---

  def _url(self, path):
    return f"{BASE}{path}"

  def _get(self, path, auth=True):
    req = Request(self._url(path))
    if auth:
      req.add_header("Authorization", "Bearer test-key")
    return json.loads(urlopen(req, timeout=5).read())

  def _post(self, path, data, auth=True):
    body = json.dumps(data).encode()
    req = Request(self._url(path), data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if auth:
      req.add_header("Authorization", "Bearer test-key")
    return json.loads(urlopen(req, timeout=5).read())

  def _webhook(self, data):
    """POST to /webhook without auth (like Lark)."""
    return self._post("/webhook", data, auth=False)

  def _push_text(self, chat_id, text, create_time, evt_id=None, **extra):
    """Push a text message through the webhook."""
    msg = {
      "chat_id": chat_id,
      "message_type": "text",
      "content": json.dumps({"text": text}),
      "create_time": create_time,
      "message_id": f"msg_{evt_id or create_time}",
      **extra,
    }
    return self._webhook({
      "header": {"token": "tok1", "event_type": "im.message.receive_v1",
                 "event_id": evt_id or f"evt_{create_time}"},
      "event": {
        "message": msg,
        "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_test"}},
      },
    })

  def _run_async(self, coro):
    """Run an async coroutine on the server's event loop."""
    fut = asyncio.run_coroutine_threadsafe(coro, self.__class__.loop)
    return fut.result(10)

  # =====================================================
  #  1. Health & Auth — matches Worker auth behavior
  # =====================================================

  def test_health_no_auth(self):
    """Worker: /health requires API_KEY. Relay: /health is public (simpler).
    Both return verify_tokens count."""
    result = self._get("/health", auth=False)
    self.assertTrue(result["ok"])
    self.assertEqual(result["verify_tokens"], 2)

  def test_poll_requires_auth(self):
    """Worker: checkApiKey returns 401 if no Bearer token."""
    with self.assertRaises(HTTPError) as ctx:
      self._get("/poll/chat:test", auth=False)
    self.assertEqual(ctx.exception.code, 401)

  def test_replies_requires_auth(self):
    with self.assertRaises(HTTPError) as ctx:
      self._get("/replies/chat:test", auth=False)
    self.assertEqual(ctx.exception.code, 401)

  def test_takeover_requires_auth(self):
    with self.assertRaises(HTTPError) as ctx:
      self._post("/takeover/chat:test", {}, auth=False)
    self.assertEqual(ctx.exception.code, 401)

  def test_relay_requires_auth(self):
    with self.assertRaises(HTTPError) as ctx:
      self._post("/relay", {"to_chat_id": "x", "message": "y"}, auth=False)
    self.assertEqual(ctx.exception.code, 401)

  def test_stop_requires_auth(self):
    with self.assertRaises(HTTPError) as ctx:
      self._get("/stop/chat:test", auth=False)
    self.assertEqual(ctx.exception.code, 401)

  def test_status_requires_auth(self):
    with self.assertRaises(HTTPError) as ctx:
      self._get("/status/chat:test", auth=False)
    self.assertEqual(ctx.exception.code, 401)

  # =====================================================
  #  2. Webhook: challenge, token, idempotency
  # =====================================================

  def test_url_verification(self):
    """Worker: if data.type === 'url_verification', return challenge.
    Happens BEFORE token check."""
    result = self._webhook({"type": "url_verification", "challenge": "abc123"})
    self.assertEqual(result["challenge"], "abc123")

  def test_card_action_also_handles_challenge(self):
    """/card-action routes to same handleWebhook in Worker."""
    result = self._post("/card-action", {
      "type": "url_verification", "challenge": "xyz",
    }, auth=False)
    self.assertEqual(result["challenge"], "xyz")

  def test_webhook_rejects_bad_token(self):
    """Worker: returns 403 'Forbidden' on token mismatch."""
    with self.assertRaises(HTTPError) as ctx:
      self._webhook({
        "header": {"token": "bad-token", "event_type": "im.message.receive_v1"},
        "event": {},
      })
    self.assertEqual(ctx.exception.code, 403)

  def test_webhook_accepts_valid_token(self):
    result = self._webhook({
      "header": {"token": "tok1", "event_type": "im.message.receive_v1",
                 "event_id": "evt_accept_eq"},
      "event": {"message": {}, "sender": {}},
    })
    self.assertTrue(result.get("ok"))

  def test_duplicate_event(self):
    """Worker: KV idempotency — seen:eventId check."""
    data = {
      "header": {"token": "tok1", "event_type": "im.message.receive_v1",
                 "event_id": "evt_dup_eq"},
      "event": {
        "message": {"chat_id": "oc_dup_eq", "message_type": "text",
                     "content": '{"text":"x"}', "create_time": "100",
                     "message_id": "msg_dup_eq"},
        "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_x"}},
      },
    }
    r1 = self._webhook(data)
    self.assertTrue(r1.get("ok"))
    r2 = self._webhook(data)
    self.assertTrue(r2.get("duplicate"))

  # =====================================================
  #  3. Message parsing — reply shape matches Worker
  # =====================================================

  def test_text_message_reply_shape(self):
    """Worker: reply has text, msg_type, sender_type, sender_id, create_time,
    message_id. Optional fields (image_key, file_key, parent_id, mentions)
    are OMITTED when empty (Worker uses undefined → not in JSON)."""
    self._push_text("oc_shape", "hello", "20000", evt_id="evt_shape1")
    result = self._get("/replies/chat:oc_shape?since=")
    r = result["replies"][0]
    # Required fields
    self.assertEqual(r["text"], "hello")
    self.assertEqual(r["msg_type"], "text")
    self.assertEqual(r["sender_type"], "user")
    self.assertEqual(r["sender_id"], "ou_test")
    self.assertEqual(r["create_time"], "20000")
    self.assertEqual(r["message_id"], "msg_evt_shape1")
    # Optional fields should NOT be present
    self.assertNotIn("image_key", r)
    self.assertNotIn("file_key", r)
    self.assertNotIn("file_name", r)
    self.assertNotIn("parent_id", r)
    self.assertNotIn("mentions", r)

  def test_image_message(self):
    """Worker: image → text='[image]', image_key set."""
    self._webhook({
      "header": {"token": "tok1", "event_type": "im.message.receive_v1",
                 "event_id": "evt_img_eq"},
      "event": {
        "message": {"chat_id": "oc_img_eq", "message_type": "image",
                     "content": '{"image_key":"img_xxx"}',
                     "create_time": "21000", "message_id": "msg_img_eq"},
        "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_i"}},
      },
    })
    r = self._get("/replies/chat:oc_img_eq?since=")["replies"][0]
    self.assertEqual(r["text"], "[image]")
    self.assertEqual(r["image_key"], "img_xxx")
    self.assertEqual(r["msg_type"], "image")

  def test_file_message(self):
    """Worker: file → text='[file: name]', file_key + file_name set."""
    self._webhook({
      "header": {"token": "tok1", "event_type": "im.message.receive_v1",
                 "event_id": "evt_file_eq"},
      "event": {
        "message": {"chat_id": "oc_file_eq", "message_type": "file",
                     "content": '{"file_key":"fk_abc","file_name":"doc.pdf"}',
                     "create_time": "22000", "message_id": "msg_file_eq"},
        "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_f"}},
      },
    })
    r = self._get("/replies/chat:oc_file_eq?since=")["replies"][0]
    self.assertEqual(r["text"], "[file: doc.pdf]")
    self.assertEqual(r["file_key"], "fk_abc")
    self.assertEqual(r["file_name"], "doc.pdf")

  def test_media_message(self):
    """Video (media) → file_key + file_name forwarded so the daemon can
    download it. Regression: previously fell to the unknown-type else and
    dropped file_key, so the daemon never saw the video."""
    self._webhook({
      "header": {"token": "tok1", "event_type": "im.message.receive_v1",
                 "event_id": "evt_media_eq"},
      "event": {
        "message": {"chat_id": "oc_media_eq", "message_type": "media",
                     "content": '{"file_key":"fk_vid","file_name":"clip.mp4","image_key":"thumb_xyz"}',
                     "create_time": "22500", "message_id": "msg_media_eq"},
        "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_m"}},
      },
    })
    r = self._get("/replies/chat:oc_media_eq?since=")["replies"][0]
    self.assertEqual(r["msg_type"], "media")
    self.assertEqual(r["file_key"], "fk_vid")
    self.assertEqual(r["file_name"], "clip.mp4")
    self.assertEqual(r["text"], "[video: clip.mp4]")
    # The thumbnail image_key is deliberately NOT forwarded (daemon ignores it).
    self.assertNotIn("image_key", r)

  def test_sticker_message(self):
    """Worker: sticker → text='[sticker]', file_key set."""
    self._webhook({
      "header": {"token": "tok1", "event_type": "im.message.receive_v1",
                 "event_id": "evt_stk_eq"},
      "event": {
        "message": {"chat_id": "oc_stk_eq", "message_type": "sticker",
                     "content": '{"file_key":"stk_123"}',
                     "create_time": "23000", "message_id": "msg_stk_eq"},
        "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_s"}},
      },
    })
    r = self._get("/replies/chat:oc_stk_eq?since=")["replies"][0]
    self.assertEqual(r["text"], "[sticker]")
    self.assertEqual(r["file_key"], "stk_123")

  def test_merge_forward_message(self):
    self._webhook({
      "header": {"token": "tok1", "event_type": "im.message.receive_v1",
                 "event_id": "evt_mf_eq"},
      "event": {
        "message": {"chat_id": "oc_mf_eq", "message_type": "merge_forward",
                     "content": "{}", "create_time": "24000",
                     "message_id": "msg_mf_eq"},
        "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_mf"}},
      },
    })
    r = self._get("/replies/chat:oc_mf_eq?since=")["replies"][0]
    self.assertEqual(r["text"], "[merge_forward]")
    self.assertEqual(r["msg_type"], "merge_forward")

  def test_unknown_message_type(self):
    """Worker: unknown type → text='[{type} message]'."""
    self._webhook({
      "header": {"token": "tok1", "event_type": "im.message.receive_v1",
                 "event_id": "evt_unk_eq"},
      "event": {
        "message": {"chat_id": "oc_unk_eq", "message_type": "audio",
                     "content": "{}", "create_time": "25000",
                     "message_id": "msg_unk_eq"},
        "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_u"}},
      },
    })
    r = self._get("/replies/chat:oc_unk_eq?since=")["replies"][0]
    self.assertEqual(r["text"], "[audio message]")

  def test_post_message_with_images(self):
    """Worker: post with mixed text + img → text joined by newline,
    image_key = comma-separated keys."""
    post_content = json.dumps({
      "content": [
        [{"text": "line one"}, {"tag": "img", "image_key": "img_a"}],
        [{"text": "line two"}, {"tag": "img", "image_key": "img_b"}],
      ]
    })
    self._webhook({
      "header": {"token": "tok1", "event_type": "im.message.receive_v1",
                 "event_id": "evt_post_eq"},
      "event": {
        "message": {"chat_id": "oc_post_eq", "message_type": "post",
                     "content": post_content, "create_time": "26000",
                     "message_id": "msg_post_eq"},
        "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_p"}},
      },
    })
    r = self._get("/replies/chat:oc_post_eq?since=")["replies"][0]
    self.assertEqual(r["text"], "line one\n[image]\nline two\n[image]")
    self.assertEqual(r["image_key"], "img_a,img_b")

  def test_post_with_embedded_video(self):
    """A video embedded in a post (the common 'video + caption' send): the
    parser faithfully keeps msg_type=post, marks the video's position with a
    [video] placeholder, and forwards file_key — mirroring [image]/image_key.
    The daemon resolves the placeholder + downloads. Regression: the parser
    used to drop the media element and forward only the caption."""
    post_content = json.dumps({
      "content": [
        [{"tag": "media", "file_key": "fk_postvid", "file_name": "clip.mp4"},
         {"text": "看看这个视频"}],
      ]
    })
    self._webhook({
      "header": {"token": "tok1", "event_type": "im.message.receive_v1",
                 "event_id": "evt_postvid_eq"},
      "event": {
        "message": {"chat_id": "oc_postvid_eq", "message_type": "post",
                     "content": post_content, "create_time": "26500",
                     "message_id": "msg_postvid_eq"},
        "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_pv"}},
      },
    })
    r = self._get("/replies/chat:oc_postvid_eq?since=")["replies"][0]
    self.assertEqual(r["msg_type"], "post")       # faithful — not rewritten
    self.assertEqual(r["file_key"], "fk_postvid")
    self.assertEqual(r["file_name"], "clip.mp4")
    self.assertEqual(r["text"], "[video]\n看看这个视频")  # placeholder + caption
    self.assertNotIn("image_key", r)

  def test_post_locale_keyed(self):
    """Worker: locale-keyed post {en_us: {content: [...]}}."""
    post_content = json.dumps({
      "en_us": {"title": "Title", "content": [[{"text": "english"}]]}
    })
    self._webhook({
      "header": {"token": "tok1", "event_type": "im.message.receive_v1",
                 "event_id": "evt_post_loc"},
      "event": {
        "message": {"chat_id": "oc_post_loc", "message_type": "post",
                     "content": post_content, "create_time": "26500",
                     "message_id": "msg_post_loc"},
        "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_p"}},
      },
    })
    r = self._get("/replies/chat:oc_post_loc?since=")["replies"][0]
    self.assertEqual(r["text"], "english")

  def test_parent_id_present(self):
    """Worker: parent_id included when present on message."""
    self._push_text("oc_pid", "reply", "27000", evt_id="evt_pid",
                    parent_id="om_parent")
    r = self._get("/replies/chat:oc_pid?since=")["replies"][0]
    self.assertEqual(r["parent_id"], "om_parent")

  def test_thread_id_present(self):
    """thread_id included when the message belongs to a Lark thread —
    nemo routes /fork sub-threads by it, so the relay must not drop it."""
    self._push_text("oc_tid", "in thread", "27500", evt_id="evt_tid",
                    thread_id="omt_fork_xyz")
    r = self._get("/replies/chat:oc_tid?since=")["replies"][0]
    self.assertEqual(r["thread_id"], "omt_fork_xyz")

  def test_thread_id_omitted_when_absent(self):
    """Top-level messages carry no thread_id key."""
    self._push_text("oc_no_tid", "top level", "27600", evt_id="evt_no_tid")
    r = self._get("/replies/chat:oc_no_tid?since=")["replies"][0]
    self.assertNotIn("thread_id", r)

  def test_mentions_structure(self):
    """Worker: mentions = [{key, id: openid, name}]."""
    self._webhook({
      "header": {"token": "tok1", "event_type": "im.message.receive_v1",
                 "event_id": "evt_ment_eq"},
      "event": {
        "message": {
          "chat_id": "oc_ment_eq", "message_type": "text",
          "content": '{"text":"@bot hi"}', "create_time": "28000",
          "message_id": "msg_ment_eq",
          "mentions": [
            {"key": "@_user_1", "id": {"open_id": "ou_bot"}, "name": "Bot"},
            {"key": "@_user_2", "id": {"open_id": "ou_other"}, "name": "Other"},
          ],
        },
        "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_m"}},
      },
    })
    r = self._get("/replies/chat:oc_ment_eq?since=")["replies"][0]
    self.assertEqual(len(r["mentions"]), 2)
    self.assertEqual(r["mentions"][0], {"key": "@_user_1", "id": "ou_bot", "name": "Bot"})
    self.assertEqual(r["mentions"][1], {"key": "@_user_2", "id": "ou_other", "name": "Other"})

  # =====================================================
  #  4. Routing — chat key, root key, nonce key
  # =====================================================

  def test_chat_and_root_dual_push(self):
    """Worker: pushes to both chat:chatId AND rootId when root_id present."""
    self._webhook({
      "header": {"token": "tok1", "event_type": "im.message.receive_v1",
                 "event_id": "evt_dual_eq"},
      "event": {
        "message": {"chat_id": "oc_dual", "root_id": "om_root_dual",
                     "message_type": "text", "content": '{"text":"threaded"}',
                     "create_time": "30000", "message_id": "msg_dual"},
        "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_d"}},
      },
    })
    # Appears in BOTH keys
    chat = self._get("/replies/chat:oc_dual?since=")
    self.assertEqual(chat["count"], 1)
    self.assertEqual(chat["replies"][0]["text"], "threaded")

    root = self._get("/replies/om_root_dual?since=")
    self.assertEqual(root["count"], 1)
    self.assertEqual(root["replies"][0]["text"], "threaded")

  def test_card_action_triple_push(self):
    """Worker: card action pushes to chat + root + nonce when all present."""
    self._webhook({
      "header": {"token": "tok1", "event_type": "card.action.trigger"},
      "event": {
        "action": {"value": {"action": "y", "chat_id": "oc_tri",
                              "root_id": "om_tri_root", "nonce": "nonce_tri"}},
        "operator": {"open_id": "ou_op"},
      },
    })
    for key in ("chat:oc_tri", "om_tri_root", "nonce_tri"):
      result = self._get(f"/replies/{key}?since=")
      self.assertTrue(
        any(r["text"] == "y" for r in result["replies"]),
        f"Expected 'y' in {key}",
      )

  def test_no_push_without_chat_or_root(self):
    """Worker: if (!rootId && !chatId) return — no push happens."""
    self._webhook({
      "header": {"token": "tok1", "event_type": "im.message.receive_v1",
                 "event_id": "evt_nopush"},
      "event": {
        "message": {"message_type": "text", "content": '{"text":"orphan"}',
                     "create_time": "31000", "message_id": "msg_nopush"},
        "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_n"}},
      },
    })
    # No way to verify absence easily, but at least no crash
    # (Worker silently returns)

  # =====================================================
  #  5. since filter — string comparison (Worker behavior)
  # =====================================================

  def test_since_is_string_comparison(self):
    """Worker DO: this.replies.filter(r => r.create_time > since).
    String comparison: '2' > '10000' is TRUE (string '2' > '1').
    Both poll and replies use string comparison."""
    self._push_text("oc_cmp", "first", "10000", evt_id="evt_cmp1")
    self._push_text("oc_cmp", "second", "20000", evt_id="evt_cmp2")
    self._push_text("oc_cmp", "third", "9", evt_id="evt_cmp3")

    # since="15000" → keeps "20000" and "9" (string "9" > "15000")
    result = self._get("/replies/chat:oc_cmp?since=15000")
    texts = [r["text"] for r in result["replies"]]
    self.assertIn("second", texts)  # "20000" > "15000"
    self.assertIn("third", texts)   # "9" > "15000" (string comparison!)
    self.assertNotIn("first", texts)  # "10000" < "15000"

  def test_since_empty_returns_all(self):
    """Worker: since='' → no filter, return all."""
    self._push_text("oc_all", "a", "40000", evt_id="evt_all1")
    self._push_text("oc_all", "b", "40001", evt_id="evt_all2")
    result = self._get("/replies/chat:oc_all?since=")
    self.assertEqual(result["count"], 2)

  def test_poll_since_string_comparison(self):
    """Poll uses same string comparison as /replies/."""
    self._push_text("oc_pcmp", "x", "50000", evt_id="evt_pcmp1")
    result = self._get("/poll/chat:oc_pcmp?since=50000&timeout=1")
    self.assertEqual(result["count"], 0)  # "50000" NOT > "50000"

    result2 = self._get("/poll/chat:oc_pcmp?since=49999&timeout=1")
    self.assertEqual(result2["count"], 1)  # "50000" > "49999"

  # =====================================================
  #  6. Ack — remove messages with create_time <= before
  # =====================================================

  def test_ack_removes_by_create_time(self):
    """Worker DO: this.replies.filter(r => r.create_time > before).
    Messages with create_time <= before are removed."""
    self._push_text("oc_ack_eq", "a", "60000", evt_id="evt_ack_eq1")
    self._push_text("oc_ack_eq", "b", "60001", evt_id="evt_ack_eq2")
    self._push_text("oc_ack_eq", "c", "60002", evt_id="evt_ack_eq3")

    result = self._post("/replies/chat:oc_ack_eq/ack?before=60001", {})
    self.assertEqual(result["removed"], 2)  # a (60000) + b (60001)
    self.assertEqual(result["remaining"], 1)  # c (60002)

    replies = self._get("/replies/chat:oc_ack_eq?since=")
    self.assertEqual(replies["count"], 1)
    self.assertEqual(replies["replies"][0]["text"], "c")

  def test_ack_string_comparison(self):
    """Ack also uses string comparison like the Worker.
    Worker: this.replies.filter(r => r.create_time > before)
    JS string: "10000" > "5" → false ("1" < "5"), "9" > "5" → true.
    SQL: create_time <= "5" for "10000" → true ("1" < "5"), for "9" → false."""
    self._push_text("oc_ack_str", "x", "9", evt_id="evt_ack_str1")
    self._push_text("oc_ack_str", "y", "10000", evt_id="evt_ack_str2")

    # before="5": "10000" <= "5" (string: "1"<"5") → removed; "9" > "5" → kept
    result = self._post("/replies/chat:oc_ack_str/ack?before=5", {})
    self.assertEqual(result["removed"], 1)  # "10000" removed
    self.assertEqual(result["remaining"], 1)  # "9" kept

    replies = self._get("/replies/chat:oc_ack_str?since=")
    self.assertEqual(replies["replies"][0]["text"], "x")  # "9" survived

  def test_ack_missing_before(self):
    """Worker: returns 400 if before param missing."""
    try:
      self._post("/replies/chat:oc_ack_miss/ack", {})
      self.fail("Expected error")
    except HTTPError as e:
      self.assertEqual(e.code, 400)

  # =====================================================
  #  7. Takeover — consumed-once semantics
  # =====================================================

  def test_takeover_consumed_by_poll(self):
    """Worker DO: takeover flag reset to false after first poll read."""
    self._post("/takeover/chat:oc_take_poll", {})
    r1 = self._get("/poll/chat:oc_take_poll?since=&timeout=1")
    self.assertTrue(r1.get("takeover"))
    self.assertEqual(r1["count"], 0)
    self.assertEqual(r1["replies"], [])

    r2 = self._get("/poll/chat:oc_take_poll?since=&timeout=1")
    self.assertFalse(r2.get("takeover", False))

  def test_takeover_consumed_by_replies(self):
    """Worker DO handleGet: takeover checked and consumed."""
    self._post("/takeover/chat:oc_take_rep", {})
    r1 = self._get("/replies/chat:oc_take_rep?since=")
    self.assertTrue(r1.get("takeover"))

    r2 = self._get("/replies/chat:oc_take_rep?since=")
    self.assertFalse(r2.get("takeover", False))

  def test_takeover_wakes_poll(self):
    """Worker DO: takeover resolves waiting polls immediately."""
    def delayed_takeover():
      time.sleep(0.5)
      self._post("/takeover/chat:oc_take_wake", {})

    threading.Thread(target=delayed_takeover, daemon=True).start()
    start = time.time()
    result = self._get("/poll/chat:oc_take_wake?since=&timeout=10")
    elapsed = time.time() - start
    self.assertTrue(result.get("takeover"))
    self.assertLess(elapsed, 3)  # Should return quickly, not wait 10s

  # =====================================================
  #  8. Stop — KV flag + poll integration
  # =====================================================

  def test_post_stop_sets_flag(self):
    """Worker: POST /stop sets KV flag, GET /stop consumes it."""
    self._post("/stop/chat:oc_stop_eq", {})
    r1 = self._get("/stop/chat:oc_stop_eq")
    self.assertTrue(r1["stop"])

    r2 = self._get("/stop/chat:oc_stop_eq")
    self.assertFalse(r2["stop"])  # Consumed

  def test_stop_via_card_button(self):
    """Worker: __stop__ card action sets KV stop + pushes stop_signal."""
    result = self._webhook({
      "header": {"token": "tok1", "event_type": "card.action.trigger"},
      "event": {
        "action": {"value": {"action": "__stop__", "chat_id": "oc_stop_card"}},
        "operator": {"open_id": "ou_op"},
      },
    })
    self.assertEqual(result["toast"]["type"], "warning")
    self.assertEqual(result["toast"]["content"], "Stopping...")
    self.assertNotIn("card", result)

    # Stop flag consumed via GET
    stop = self._get("/stop/chat:oc_stop_card")
    self.assertTrue(stop["stop"])

  def test_stop_checked_in_poll(self):
    """Worker: poll checks stop flag before AND after wait."""
    self._post("/stop/chat:oc_stop_poll", {})
    result = self._get("/poll/chat:oc_stop_poll?since=&timeout=1")
    self.assertTrue(result.get("stop"))
    self.assertEqual(result["count"], 0)

  def test_stop_after_wait(self):
    """Worker: if stop is set during wait, poll returns stop=true."""
    def delayed_stop():
      time.sleep(0.5)
      self._post("/stop/chat:oc_stop_wait", {})

    threading.Thread(target=delayed_stop, daemon=True).start()
    start = time.time()
    result = self._get("/poll/chat:oc_stop_wait?since=&timeout=10")
    elapsed = time.time() - start
    self.assertTrue(result.get("stop"))
    self.assertLess(elapsed, 3)

  def test_stop_authorization_allowed(self):
    """Worker: stop button respects approvers list — allowed user."""
    result = self._webhook({
      "header": {"token": "tok1", "event_type": "card.action.trigger"},
      "event": {
        "action": {"value": {"action": "__stop__", "chat_id": "oc_stop_auth_ok",
                              "approvers": ["ou_owner"]}},
        "operator": {"open_id": "ou_owner"},
      },
    })
    self.assertEqual(result["toast"]["type"], "warning")

  def test_stop_authorization_denied(self):
    """Worker: stop button rejects non-approver."""
    result = self._webhook({
      "header": {"token": "tok1", "event_type": "card.action.trigger"},
      "event": {
        "action": {"value": {"action": "__stop__", "chat_id": "oc_stop_auth_bad",
                              "approvers": ["ou_owner"]}},
        "operator": {"open_id": "ou_hacker"},
      },
    })
    self.assertEqual(result["toast"]["type"], "error")

  # =====================================================
  #  9. Card actions — response shape
  # =====================================================

  def test_card_approve_response(self):
    """Worker: approve → green card, success toast."""
    result = self._webhook({
      "header": {"token": "tok1", "event_type": "card.action.trigger"},
      "event": {
        "action": {"value": {"action": "y", "chat_id": "oc_card_app",
                              "title": "Permission", "body": "Allow write?"}},
        "operator": {"open_id": "ou_op"},
      },
    })
    self.assertEqual(result["toast"]["type"], "success")
    self.assertIn("y", result["toast"]["content"])
    card = result["card"]["data"]
    self.assertEqual(card["header"]["template"], "green")
    self.assertEqual(card["header"]["title"]["content"], "Permission")
    # Body + selected text
    self.assertEqual(len(card["elements"]), 2)
    self.assertIn("Allow write?", card["elements"][0]["text"]["content"])
    self.assertIn("**y**", card["elements"][1]["text"]["content"])

  def test_card_deny_response(self):
    """Worker: deny actions (n/no/deny/reject/0) → red card, warning toast."""
    for deny_text in ("n", "no", "deny", "reject", "0"):
      result = self._webhook({
        "header": {"token": "tok1", "event_type": "card.action.trigger"},
        "event": {
          "action": {"value": {"action": deny_text, "chat_id": f"oc_deny_{deny_text}"}},
          "operator": {"open_id": "ou_op"},
        },
      })
      self.assertEqual(result["toast"]["type"], "warning", f"Failed for {deny_text}")
      self.assertEqual(result["card"]["data"]["header"]["template"], "red",
                       f"Failed for {deny_text}")

  def test_card_authorization(self):
    """Worker: approvers list on regular action → unauthorized toast."""
    result = self._webhook({
      "header": {"token": "tok1", "event_type": "card.action.trigger"},
      "event": {
        "action": {"value": {"action": "y", "chat_id": "oc_card_auth",
                              "approvers": ["ou_allowed"]}},
        "operator": {"open_id": "ou_stranger"},
      },
    })
    self.assertEqual(result["toast"]["type"], "error")
    # Card should NOT be present (unauthorized, keep card unchanged)
    # Worker returns only toast for unauthorized
    self.assertNotIn("card", result)

  def test_card_no_body(self):
    """Worker: when body is empty, only 'Selected' element in card."""
    result = self._webhook({
      "header": {"token": "tok1", "event_type": "card.action.trigger"},
      "event": {
        "action": {"value": {"action": "ok", "chat_id": "oc_card_nobody"}},
        "operator": {"open_id": "ou_op"},
      },
    })
    card = result["card"]["data"]
    self.assertEqual(len(card["elements"]), 1)
    self.assertIn("**ok**", card["elements"][0]["text"]["content"])

  # =====================================================
  #  10. Card action types: form, select, input
  # =====================================================

  def test_form_action_single_field(self):
    """Worker extractActionInfo: single form field → value directly."""
    result = self._webhook({
      "header": {"token": "tok1", "event_type": "card.action.trigger"},
      "event": {
        "action": {
          "value": {"chat_id": "oc_form_single"},
          "form_value": {"input": "typed text"},
        },
        "operator": {"open_id": "ou_op"},
      },
    })
    self.assertIn("toast", result)
    poll = self._get("/replies/chat:oc_form_single?since=")
    r = [x for x in poll["replies"] if x["msg_type"] == "form_action"]
    self.assertEqual(len(r), 1)
    self.assertEqual(r[0]["text"], "typed text")

  def test_form_action_multi_field(self):
    """Worker extractActionInfo: multi field → JSON string."""
    self._webhook({
      "header": {"token": "tok1", "event_type": "card.action.trigger"},
      "event": {
        "action": {
          "value": {"chat_id": "oc_form_multi"},
          "form_value": {"name": "Alice", "age": "30"},
        },
        "operator": {"open_id": "ou_op"},
      },
    })
    poll = self._get("/replies/chat:oc_form_multi?since=")
    r = [x for x in poll["replies"] if x["msg_type"] == "form_action"]
    self.assertEqual(len(r), 1)
    parsed = json.loads(r[0]["text"])
    self.assertEqual(parsed["name"], "Alice")
    self.assertEqual(parsed["age"], "30")

  def test_form_action_falls_back_to_context_chat_id(self):
    """Lark V2 form submissions (form_action_type='submit') sometimes
    drop the button's ``value`` field — only ``form_value`` survives.
    Without ``value.chat_id`` the original push branch was a silent
    no-op, so the daemon never saw the submission. ``event.context``
    always carries ``open_chat_id`` for card actions; the relay must
    fall back to it so the form click still reaches the bot."""
    self._webhook({
      "header": {"token": "tok1", "event_type": "card.action.trigger"},
      "event": {
        "action": {
          # No value at all — like Lark's form_action_type=submit
          # behaviour where the button's value is sometimes dropped.
          "form_value": {"model": "model_switch:claude-sonnet-4-6"},
          "tag": "button",
        },
        "operator": {"open_id": "ou_op"},
        "context": {
          "open_chat_id": "oc_ctx_fallback",
          "open_message_id": "om_picker_xyz",
        },
      },
    })
    poll = self._get("/replies/chat:oc_ctx_fallback?since=")
    r = [x for x in poll["replies"] if x["msg_type"] == "form_action"]
    self.assertEqual(len(r), 1, poll)
    self.assertEqual(r[0]["text"], "model_switch:claude-sonnet-4-6")
    # And the card's own message_id flows through so the daemon can
    # PATCH the picker into its post-submit confirmation state.
    self.assertEqual(r[0]["message_id"], "om_picker_xyz")

  def test_card_action_skips_confirm_card_for_model_picker_prefixes(self):
    """``model_switch:*`` is a bot-owned action: the daemon PATCHes
    the picker into a confirmation state shortly after the click
    lands. Without this suppression Lark briefly flashes a generic
    ``Confirmed / Selected: model_switch:claude-opus-4-7`` card over
    the picker. Same pattern as ``askq:``."""
    result = self._webhook({
      "header": {"token": "tok1", "event_type": "card.action.trigger"},
      "event": {
        "action": {
          "value": {"chat_id": "oc_model_pref"},
          "form_value": {"model": "model_switch:claude-opus-4-7"},
        },
        "operator": {"open_id": "ou_op"},
      },
    })
    # Toast-only response — no ``card`` key, so Lark leaves the
    # picker visible until the bot PATCHes it.
    self.assertIn("toast", result)
    self.assertNotIn("card", result)

  def test_form_action_session_recall_with_and_without_value(self):
    """The /session recall picker submits a single-field form whose value
    is ``session_recall:<uuid>``. Cover both Lark V2 form-submit shapes:
    with the button ``value`` present, and with it DROPPED (only
    ``form_value`` + ``context`` survive) — the relay must route both via
    the ``open_chat_id`` / ``open_message_id`` context fallback."""
    # (a) value present
    self._webhook({
      "header": {"token": "tok1", "event_type": "card.action.trigger"},
      "event": {
        "action": {
          "value": {"chat_id": "oc_recall_a"},
          "form_value": {"session": "session_recall:uuid-aaaa"},
        },
        "operator": {"open_id": "ou_op"},
      },
    })
    r = [x for x in self._get("/replies/chat:oc_recall_a?since=")["replies"]
         if x["msg_type"] == "form_action"]
    self.assertEqual(len(r), 1)
    self.assertEqual(r[0]["text"], "session_recall:uuid-aaaa")

    # (b) button value dropped — context fallback carries chat + message id
    self._webhook({
      "header": {"token": "tok1", "event_type": "card.action.trigger"},
      "event": {
        "action": {
          "form_value": {"session": "session_recall:uuid-bbbb"},
          "tag": "button",
        },
        "operator": {"open_id": "ou_op"},
        "context": {
          "open_chat_id": "oc_recall_b",
          "open_message_id": "om_session_picker",
        },
      },
    })
    r = [x for x in self._get("/replies/chat:oc_recall_b?since=")["replies"]
         if x["msg_type"] == "form_action"]
    self.assertEqual(len(r), 1)
    self.assertEqual(r[0]["text"], "session_recall:uuid-bbbb")
    self.assertEqual(r[0]["message_id"], "om_session_picker")

  def test_card_action_skips_confirm_card_for_session_recall_prefix(self):
    """``session_recall:*`` is bot-owned: the daemon PATCHes the picker
    into its locked state, so the relay must suppress the generic
    Confirmed/Selected card (toast only)."""
    result = self._webhook({
      "header": {"token": "tok1", "event_type": "card.action.trigger"},
      "event": {
        "action": {
          "value": {"chat_id": "oc_recall_pref"},
          "form_value": {"session": "session_recall:uuid-cccc"},
        },
        "operator": {"open_id": "ou_op"},
      },
    })
    self.assertIn("toast", result)
    self.assertNotIn("card", result)

  def test_select_action(self):
    """Worker extractActionInfo: action.option → select_action."""
    self._webhook({
      "header": {"token": "tok1", "event_type": "card.action.trigger"},
      "event": {
        "action": {
          "value": {"chat_id": "oc_select"},
          "option": "option_b",
        },
        "operator": {"open_id": "ou_op"},
      },
    })
    poll = self._get("/replies/chat:oc_select?since=")
    r = [x for x in poll["replies"] if x["msg_type"] == "select_action"]
    self.assertEqual(len(r), 1)
    self.assertEqual(r[0]["text"], "option_b")

  def test_input_action(self):
    """Worker extractActionInfo: action.input_value → input_action."""
    self._webhook({
      "header": {"token": "tok1", "event_type": "card.action.trigger"},
      "event": {
        "action": {
          "value": {"chat_id": "oc_input"},
          "input_value": "user input text",
        },
        "operator": {"open_id": "ou_op"},
      },
    })
    poll = self._get("/replies/chat:oc_input?since=")
    r = [x for x in poll["replies"] if x["msg_type"] == "input_action"]
    self.assertEqual(len(r), 1)
    self.assertEqual(r[0]["text"], "user input text")

  # =====================================================
  #  11. Reaction routing
  # =====================================================

  def test_reaction_routing(self):
    """Worker: register-message stores msgchat:msgId→chatId in KV,
    reaction event looks up chatId and pushes reply."""
    self._post("/register-message", {
      "message_id": "msg_react_eq", "chat_id": "oc_react_eq",
    })
    self._webhook({
      "header": {"token": "tok1", "event_type": "im.message.reaction.created_v1",
                 "event_id": "evt_react_eq"},
      "event": {
        "message_id": "msg_react_eq",
        "reaction_type": {"emoji_type": "THUMBSUP"},
        "user_id": {"open_id": "ou_reactor"},
        "action_time": "70000",
      },
    })
    result = self._get("/replies/chat:oc_react_eq?since=")
    reactions = [r for r in result["replies"] if r["msg_type"] == "reaction"]
    self.assertEqual(len(reactions), 1)
    r = reactions[0]
    self.assertEqual(r["text"], "THUMBSUP")
    self.assertEqual(r["target_message_id"], "msg_react_eq")
    self.assertEqual(r["sender_id"], "ou_reactor")
    self.assertEqual(r["create_time"], "70000")

  def test_reaction_unregistered_message_ignored(self):
    """Worker: if chatId lookup returns null, reaction is silently dropped."""
    self._webhook({
      "header": {"token": "tok1", "event_type": "im.message.reaction.created_v1",
                 "event_id": "evt_react_unreg"},
      "event": {
        "message_id": "msg_not_registered",
        "reaction_type": {"emoji_type": "SMILE"},
        "user_id": {"open_id": "ou_r"},
        "action_time": "71000",
      },
    })
    # No crash — silently ignored

  # =====================================================
  #  11b. Recall forwarding — daemon's in-turn watcher needs these
  # =====================================================

  def test_recall_forwarded_as_msg_type_recall(self):
    """Lark recall webhook → relay pushes msg_type='recall' to the chat,
    so the daemon's watcher can drop the recalled message from its
    pending queue. Pre-fix, the relay swallowed recall webhooks entirely
    and recall never worked end-to-end on the relay path."""
    self._webhook({
      "header": {"token": "tok1", "event_type": "im.message.recalled_v1",
                 "event_id": "evt_recall_eq"},
      "event": {
        "message_id": "om_recalled_eq",
        "chat_id": "oc_recall_eq",
        "notify_time": "80000",
      },
    })
    result = self._get("/replies/chat:oc_recall_eq?since=")
    recalls = [r for r in result["replies"] if r["msg_type"] == "recall"]
    self.assertEqual(len(recalls), 1)
    r = recalls[0]
    self.assertEqual(r["message_id"], "om_recalled_eq")
    self.assertEqual(r["create_time"], "80000")

  def test_recall_without_chat_id_dropped(self):
    """Recall webhook missing chat_id has nowhere to route — drop silently."""
    self._webhook({
      "header": {"token": "tok1", "event_type": "im.message.recalled_v1",
                 "event_id": "evt_recall_no_chat"},
      "event": {"message_id": "om_recalled_no_chat"},
    })
    # No crash; no chat to assert against.

  # =====================================================
  #  12. Relay — cross-group messaging
  # =====================================================

  def test_relay_message_shape(self):
    """Worker: relay reply has msg_type='relay', from_* fields, sender_type='relay'."""
    result = self._post("/relay", {
      "to_chat_id": "oc_relay_eq",
      "message": "cross-group hello",
      "from_chat_id": "oc_source",
      "from_chat_name": "Source Group",
      "from_workspace": "/code/project",
    })
    self.assertTrue(result["ok"])

    poll = self._get("/replies/chat:oc_relay_eq?since=")
    self.assertEqual(poll["count"], 1)
    r = poll["replies"][0]
    self.assertEqual(r["text"], "cross-group hello")
    self.assertEqual(r["msg_type"], "relay")
    self.assertEqual(r["from_chat_id"], "oc_source")
    self.assertEqual(r["from_chat_name"], "Source Group")
    self.assertEqual(r["from_workspace"], "/code/project")
    self.assertEqual(r["sender_type"], "relay")
    self.assertEqual(r["sender_id"], "")
    self.assertEqual(r["message_id"], "")

  def test_relay_missing_fields(self):
    """Worker: returns 400 if to_chat_id or message missing."""
    try:
      self._post("/relay", {"to_chat_id": "x"})
      self.fail("Expected 400")
    except HTTPError as e:
      self.assertEqual(e.code, 400)

  # =====================================================
  #  13. Status endpoint
  # =====================================================

  def test_status_always_ok(self):
    """Worker: reads do_quota_exhausted from KV. Relay: always ok."""
    result = self._get("/status/chat:oc_any")
    self.assertFalse(result["do_quota_exhausted"])
    self.assertIsNone(result["exhausted_at"])

  # =====================================================
  #  14. Long-poll behavior
  # =====================================================

  def test_poll_timeout(self):
    """Worker DO: waits up to timeout seconds then returns empty."""
    start = time.time()
    result = self._get("/poll/chat:oc_empty_eq?since=&timeout=1")
    elapsed = time.time() - start
    self.assertEqual(result["count"], 0)
    self.assertGreaterEqual(elapsed, 0.9)
    self.assertLess(elapsed, 3)

  def test_poll_max_timeout_capped(self):
    """Worker DO: timeout capped at 55 seconds."""
    # We test with timeout=100, should be capped at 55
    # Just verify it doesn't error (we don't wait 55s)
    self._push_text("oc_cap", "cap", "80000", evt_id="evt_cap")
    result = self._get("/poll/chat:oc_cap?since=&timeout=100")
    self.assertGreater(result["count"], 0)

  def test_poll_notification(self):
    """Worker DO: push wakes waiting poll immediately."""
    def delayed_push():
      time.sleep(0.5)
      self._run_async(relay._push_message("chat:oc_notify_eq", {
        "text": "delayed",
        "msg_type": "text",
        "create_time": "90000",
      }))

    threading.Thread(target=delayed_push, daemon=True).start()
    start = time.time()
    result = self._get("/poll/chat:oc_notify_eq?since=&timeout=10")
    elapsed = time.time() - start
    self.assertEqual(result["count"], 1)
    self.assertLess(elapsed, 3)

  # =====================================================
  #  15. WebSocket — real-time push, ack, ping, takeover
  # =====================================================

  def test_ws_connect_and_initial_messages(self):
    """Worker DO handleWebSocketUpgrade: sends initial messages on connect,
    filtered by since param."""
    self._push_text("oc_ws1", "a", "100000", evt_id="evt_ws1a")
    self._push_text("oc_ws1", "b", "100001", evt_id="evt_ws1b")

    async def run():
      async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
          f"{BASE}/ws/chat:oc_ws1?since=100000",
          headers=AUTH,
        ) as ws:
          msg = await asyncio.wait_for(ws.receive_json(), timeout=3)
          # Only "b" should be sent (create_time "100001" > "100000")
          self.assertEqual(msg["count"], 1)
          self.assertEqual(msg["replies"][0]["text"], "b")

    self._run_async(run())

  def test_ws_connect_empty_since_returns_all(self):
    """Worker: since='' → all messages sent on connect."""
    self._push_text("oc_ws2", "x", "101000", evt_id="evt_ws2")

    async def run():
      async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
          f"{BASE}/ws/chat:oc_ws2?since=",
          headers=AUTH,
        ) as ws:
          msg = await asyncio.wait_for(ws.receive_json(), timeout=3)
          self.assertGreaterEqual(msg["count"], 1)

    self._run_async(run())

  def test_ws_receives_push(self):
    """Worker DO broadcastWebSocket: push broadcasts to WS clients."""
    async def run():
      async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
          f"{BASE}/ws/chat:oc_ws_push?since=",
          headers=AUTH,
        ) as ws:
          # Push a message after connecting
          await relay._push_message("chat:oc_ws_push", {
            "text": "ws-pushed",
            "msg_type": "text",
            "create_time": "102000",
          })
          msg = await asyncio.wait_for(ws.receive_json(), timeout=3)
          self.assertEqual(msg["count"], 1)
          self.assertEqual(msg["replies"][0]["text"], "ws-pushed")

    self._run_async(run())

  def test_ws_ping_pong(self):
    """Worker DO webSocketMessage: {ping: true} → {pong: true}."""
    async def run():
      async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
          f"{BASE}/ws/chat:oc_ws_ping?since=",
          headers=AUTH,
        ) as ws:
          await ws.send_json({"ping": True})
          msg = await asyncio.wait_for(ws.receive_json(), timeout=3)
          self.assertTrue(msg.get("pong"))

    self._run_async(run())

  def test_ws_ack(self):
    """Worker DO webSocketMessage: {ack: timestamp} removes older messages."""
    self._push_text("oc_ws_ack", "old", "103000", evt_id="evt_ws_ack1")
    self._push_text("oc_ws_ack", "new", "103001", evt_id="evt_ws_ack2")

    async def run():
      async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
          f"{BASE}/ws/chat:oc_ws_ack?since=",
          headers=AUTH,
        ) as ws:
          msg = await asyncio.wait_for(ws.receive_json(), timeout=3)
          self.assertEqual(msg["count"], 2)

          # Ack up to 103000
          await ws.send_json({"ack": "103000"})
          await asyncio.sleep(0.3)

        # Verify via HTTP (use aiohttp, not sync urllib)
        async with session.get(
          f"{BASE}/replies/chat:oc_ws_ack?since=",
          headers=AUTH,
        ) as resp:
          replies = await resp.json()
          self.assertEqual(replies["count"], 1)
          self.assertEqual(replies["replies"][0]["text"], "new")

    self._run_async(run())

  def test_ws_takeover_on_connect(self):
    """Worker DO handleWebSocketUpgrade: if takeover flag set, sends
    {replies:[], count:0, takeover:true} instead of initial messages."""
    self._push_text("oc_ws_take", "msg", "104000", evt_id="evt_ws_take")
    self._post("/takeover/chat:oc_ws_take", {})

    async def run():
      async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
          f"{BASE}/ws/chat:oc_ws_take?since=",
          headers=AUTH,
        ) as ws:
          msg = await asyncio.wait_for(ws.receive_json(), timeout=3)
          self.assertTrue(msg.get("takeover"))
          self.assertEqual(msg["count"], 0)
          self.assertEqual(msg["replies"], [])

    self._run_async(run())

  def test_ws_takeover_broadcast(self):
    """Worker DO: takeover broadcast to connected WS clients."""
    async def run():
      async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
          f"{BASE}/ws/chat:oc_ws_take_bc?since=",
          headers=AUTH,
        ) as ws:
          # Trigger takeover while connected (use aiohttp, not sync urllib)
          async with session.post(
            f"{BASE}/takeover/chat:oc_ws_take_bc",
            headers=AUTH,
          ) as resp:
            self.assertEqual(resp.status, 200)
          msg = await asyncio.wait_for(ws.receive_json(), timeout=3)
          self.assertTrue(msg.get("takeover"))

    self._run_async(run())

  def test_ws_requires_auth(self):
    """Worker: checkApiKey on /ws/ endpoint."""
    async def run():
      async with aiohttp.ClientSession() as session:
        # No auth header → should get non-101 response
        try:
          async with session.ws_connect(f"{BASE}/ws/chat:test") as ws:
            self.fail("Expected auth failure")
        except aiohttp.WSServerHandshakeError as e:
          self.assertEqual(e.status, 401)

    self._run_async(run())


  # =====================================================
  #  16. Error handling — malformed input
  # =====================================================

  def test_poll_non_numeric_timeout(self):
    """Poll should not crash when timeout query param is non-numeric.
    Server should default to a valid timeout instead of returning 500."""
    self._push_text("oc_bad_timeout", "x", "110000", evt_id="evt_bad_to")
    result = self._get("/poll/chat:oc_bad_timeout?timeout=abc&since=")
    self.assertGreaterEqual(result["count"], 1)

  def test_webhook_malformed_json(self):
    """POST /webhook with non-JSON body should return 400."""
    body = b"this is not json"
    req = Request(self._url("/webhook"), data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with self.assertRaises(HTTPError) as ctx:
      urlopen(req, timeout=5)
    self.assertEqual(ctx.exception.code, 400)

  def test_card_action_malformed_json(self):
    """POST /card-action with non-JSON body should return 400."""
    body = b"this is not json"
    req = Request(self._url("/card-action"), data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with self.assertRaises(HTTPError) as ctx:
      urlopen(req, timeout=5)
    self.assertEqual(ctx.exception.code, 400)


if __name__ == "__main__":
  unittest.main()
