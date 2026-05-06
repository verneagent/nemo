"""Tests for nemo.db."""

from unittest import mock

from nemo.db import Database


def test_activate_and_get_session(tmp_path):
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    db.activate("sess1", "chat1", "opus", operator_open_id="op1", bot_open_id="bot1")
    s = db.get_session("sess1")
    assert s is not None
    assert s["chat_id"] == "chat1"
    assert s["session_model"] == "opus"
    assert s["operator_open_id"] == "op1"
    assert s["bot_open_id"] == "bot1"
    assert s["need_mention"] is False
    assert s["autoapprove"] is False
    db.close()


def test_get_session_nonexistent(tmp_path):
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    assert db.get_session("nope") is None
    db.close()


def test_deactivate_keeps_row_for_resume(tmp_path):
  # deactivate must preserve the sessions row so the per-provider resume
  # ids survive clean shutdown and the next boot can resume the
  # coding-agent thread. Ownership fields get overwritten by the next
  # activate().
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    db.activate("sess1", "chat1", "opus")
    db.set_sdk_session_id("chat1", "thread-abc", "claude")
    chat_id = db.deactivate("sess1")
    assert chat_id == "chat1"
    # Row survives — resume state is still queryable by chat_id.
    assert db.get_sdk_session_id("chat1", "claude") == "thread-abc"
    assert db.get_chat_owner("chat1") == "sess1"
    db.close()


def test_activate_after_deactivate_preserves_sdk_session_id(tmp_path):
  # Simulates the real restart flow: old daemon deactivates, new daemon
  # activates with a fresh session_id. The INSERT OR REPLACE on the
  # chat_id UNIQUE conflict must carry the per-provider resume ids over.
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    db.activate("sess1", "chat1", "opus")
    db.set_sdk_session_id("chat1", "thread-abc", "claude")
    db.deactivate("sess1")
    db.activate("sess2", "chat1", "opus")
    assert db.get_sdk_session_id("chat1", "claude") == "thread-abc"
    assert db.get_chat_owner("chat1") == "sess2"
    db.close()


def test_per_provider_session_ids_isolated(tmp_path):
  # Switching provider must NOT cross-pollute resume targets — Codex
  # rejects Claude UUIDs and vice versa, the resume-fallback would mask
  # the failure but waste a turn.
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    db.activate("sess1", "chat1", "opus")
    db.set_sdk_session_id("chat1", "claude-uuid", "claude")
    db.set_sdk_session_id("chat1", "codex-thread", "codex")
    db.set_sdk_session_id("chat1", "opencode-sess", "opencode")
    assert db.get_sdk_session_id("chat1", "claude") == "claude-uuid"
    assert db.get_sdk_session_id("chat1", "codex") == "codex-thread"
    assert db.get_sdk_session_id("chat1", "opencode") == "opencode-sess"
    # Each provider's slot survives a deactivate + reactivate.
    db.deactivate("sess1")
    db.activate("sess2", "chat1", "opus")
    assert db.get_sdk_session_id("chat1", "claude") == "claude-uuid"
    assert db.get_sdk_session_id("chat1", "codex") == "codex-thread"
    assert db.get_sdk_session_id("chat1", "opencode") == "opencode-sess"
    # Unknown provider is a no-op rather than a crash.
    assert db.get_sdk_session_id("chat1", "bogus") == ""
    db.set_sdk_session_id("chat1", "ignored", "bogus")  # silent
    assert db.get_sdk_session_id("chat1", "claude") == "claude-uuid"  # unchanged
    db.close()


def test_legacy_sdk_session_id_migrates_to_claude(tmp_path):
  # First-time upgrade path: a pre-0.3.87 DB has the legacy
  # sdk_session_id populated and the new per-provider columns missing.
  # On open, the new claude_session_id column should be backfilled with
  # the legacy id so an in-flight Claude session keeps resuming.
  import sqlite3

  from nemo.db import _db_path

  proj = str(tmp_path / "project")
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    legacy_path = _db_path(proj)  # same hash-derived path Database opens
  # Hand-build a legacy schema row at that exact path.
  conn = sqlite3.connect(legacy_path)
  conn.execute("""
    CREATE TABLE sessions (
      session_id TEXT PRIMARY KEY,
      chat_id TEXT UNIQUE,
      session_model TEXT DEFAULT '',
      last_checked TEXT DEFAULT '',
      activated_at TEXT DEFAULT '',
      operator_open_id TEXT DEFAULT '',
      bot_open_id TEXT DEFAULT '',
      need_mention INTEGER DEFAULT 0,
      autoapprove INTEGER DEFAULT 0,
      sdk_session_id TEXT DEFAULT ''
    )
  """)
  conn.execute(
    "INSERT INTO sessions (session_id, chat_id, sdk_session_id) "
    "VALUES ('legacy', 'chat-old', 'old-claude-uuid')"
  )
  conn.commit()
  conn.close()

  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(proj)
    # New per-provider column has been backfilled from legacy.
    assert db.get_sdk_session_id("chat-old", "claude") == "old-claude-uuid"
    # Other providers stay empty — we don't know which provider the
    # legacy id was for, claude is the historical default.
    assert db.get_sdk_session_id("chat-old", "codex") == ""
    assert db.get_sdk_session_id("chat-old", "opencode") == ""
    db.close()


def test_deactivate_nonexistent(tmp_path):
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    assert db.deactivate("nope") is None
    db.close()


def test_get_chat_owner(tmp_path):
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    db.activate("sess1", "chat1", "opus")
    assert db.get_chat_owner("chat1") == "sess1"
    assert db.get_chat_owner("chat999") is None
    db.close()


def test_set_autoapprove(tmp_path):
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    db.activate("sess1", "chat1", "opus")
    db.set_autoapprove("chat1", True)
    s = db.get_session("sess1")
    assert s["autoapprove"] is True
    db.set_autoapprove("chat1", False)
    s = db.get_session("sess1")
    assert s["autoapprove"] is False
    db.close()


def test_record_and_lookup_messages(tmp_path):
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    db.record_received("chat1", text="hello", source_message_id="om_1")
    db.record_sent("om_2", text="hi back", chat_id="chat1")
    parent = db.lookup_parent_message("om_2")
    assert parent is not None
    assert parent["text"] == "hi back"
    # Lookup also finds received messages by source_message_id —
    # needed so quoted user messages can be recovered from our own DB.
    received = db.lookup_parent_message("om_1")
    assert received is not None
    assert received["text"] == "hello"
    assert db.lookup_parent_message("om_999") is None
    db.close()


def test_working_state(tmp_path):
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    db.activate("sess1", "chat1", "opus")
    assert db.get_working("sess1") is None
    db.set_working("sess1", "msg_123")
    assert db.get_working("sess1") == "msg_123"
    db.clear_working("sess1")
    assert db.get_working("sess1") is None
    db.close()


def test_deactivate_clears_working(tmp_path):
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    db.activate("sess1", "chat1", "opus")
    db.set_working("sess1", "msg_123")
    db.deactivate("sess1")
    assert db.get_working("sess1") is None
    db.close()


def test_need_mention(tmp_path):
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    db.activate("sess1", "chat1", "opus", need_mention=True)
    s = db.get_session("sess1")
    assert s["need_mention"] is True
    db.close()


def test_get_current_session(tmp_path):
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    assert db.get_current_session() is None
    db.activate("sess1", "chat1", "opus")
    s = db.get_current_session()
    assert s is not None
    assert s["session_id"] == "sess1"
    db.close()
