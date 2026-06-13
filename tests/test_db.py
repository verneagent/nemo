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
  # deactivate must preserve the sessions row so the per-agent resume
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
  # chat_id UNIQUE conflict must carry the per-agent resume ids over.
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    db.activate("sess1", "chat1", "opus")
    db.set_sdk_session_id("chat1", "thread-abc", "claude")
    db.deactivate("sess1")
    db.activate("sess2", "chat1", "opus")
    assert db.get_sdk_session_id("chat1", "claude") == "thread-abc"
    assert db.get_chat_owner("chat1") == "sess2"
    db.close()


def test_per_agent_session_ids_isolated(tmp_path):
  # Switching agent must NOT cross-pollute resume targets — Codex
  # rejects Claude UUIDs and vice versa, the resume-fallback would mask
  # the failure but waste a turn.
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    db.activate("sess1", "chat1", "opus")
    db.set_sdk_session_id("chat1", "claude-uuid", "claude")
    db.set_sdk_session_id("chat1", "codex-thread", "codex")
    db.set_sdk_session_id("chat1", "opencode-sess", "opencode")
    # claude-cli (pty TUI) keeps a SEPARATE slot from the SDK claude — without
    # its own column this round-trip silently no-op'd and daemon restart could
    # never resume a claude-cli conversation.
    db.set_sdk_session_id("chat1", "clicli-uuid", "claude-cli")
    assert db.get_sdk_session_id("chat1", "claude") == "claude-uuid"
    assert db.get_sdk_session_id("chat1", "codex") == "codex-thread"
    assert db.get_sdk_session_id("chat1", "opencode") == "opencode-sess"
    assert db.get_sdk_session_id("chat1", "claude-cli") == "clicli-uuid"
    # Each agent's slot survives a deactivate + reactivate.
    db.deactivate("sess1")
    db.activate("sess2", "chat1", "opus")
    assert db.get_sdk_session_id("chat1", "claude") == "claude-uuid"
    assert db.get_sdk_session_id("chat1", "codex") == "codex-thread"
    assert db.get_sdk_session_id("chat1", "opencode") == "opencode-sess"
    assert db.get_sdk_session_id("chat1", "claude-cli") == "clicli-uuid"
    # Unknown agent is a no-op rather than a crash.
    assert db.get_sdk_session_id("chat1", "bogus") == ""
    db.set_sdk_session_id("chat1", "ignored", "bogus")  # silent
    assert db.get_sdk_session_id("chat1", "claude") == "claude-uuid"  # unchanged
    db.close()


def test_per_endpoint_sessions_isolated_within_agent(tmp_path):
  # Within one agent, different upstream endpoints (default Anthropic
  # vs DeepSeek's Anthropic-compatible gateway) must keep separate
  # session ids. Otherwise resuming a DeepSeek-produced transcript
  # against real Anthropic surfaces as
  # ``400 Invalid signature in thinking block`` — the thinking blocks
  # were signed by DeepSeek and the Anthropic API rejects them.
  #
  # endpoint_key is the upstream URL — same URL = same signing
  # authority = safe to share a session.
  ds_url = "https://api.deepseek.com/anthropic"
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    db.activate("sess1", "chat1", "claude-opus-4-7")
    # Default endpoint (real Anthropic).
    db.set_sdk_session_id("chat1", "anthropic-uuid", "claude")
    # Preset endpoint (DeepSeek's anthropic-compat gateway).
    db.set_sdk_session_id("chat1", "deepseek-uuid", "claude", ds_url)
    # Each endpoint's slot is independent. Switching endpoints must
    # not bleed across.
    assert db.get_sdk_session_id("chat1", "claude") == "anthropic-uuid"
    assert db.get_sdk_session_id("chat1", "claude", "") == "anthropic-uuid"
    assert db.get_sdk_session_id("chat1", "claude", ds_url) == "deepseek-uuid"
    # Unknown endpoint_key for a known agent → empty (start fresh)
    # rather than falling back to another endpoint's id.
    assert db.get_sdk_session_id(
      "chat1", "claude", "https://other-vendor.example/anthropic") == ""
    # Survives deactivate + reactivate just like the legacy columns do.
    db.deactivate("sess1")
    db.activate("sess2", "chat1", "claude-opus-4-7")
    assert db.get_sdk_session_id("chat1", "claude") == "anthropic-uuid"
    assert db.get_sdk_session_id("chat1", "claude", ds_url) == "deepseek-uuid"
    # Overwriting one endpoint leaves the other intact.
    db.set_sdk_session_id("chat1", "deepseek-uuid-2", "claude", ds_url)
    assert db.get_sdk_session_id("chat1", "claude", ds_url) == "deepseek-uuid-2"
    assert db.get_sdk_session_id("chat1", "claude") == "anthropic-uuid"
    db.close()


def test_legacy_sdk_session_id_does_not_backfill(tmp_path):
  # Pre-0.3.87 DB has sdk_session_id populated and per-agent columns
  # missing. We deliberately do NOT copy the legacy id into any new
  # column on first upgrade: the old column was agent-blind, so a
  # codex thread id could end up in there. Feeding that to claude on
  # the next daemon spawn would make the SDK subprocess silently exit
  # 1 (Claude has no lazy-throw resume fallback). Better to lose one
  # resume than to wedge the daemon.
  import sqlite3

  from nemo.db import _db_path

  proj = str(tmp_path / "project")
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    legacy_path = _db_path(proj)
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
    "VALUES ('legacy', 'chat-old', 'opaque-id-of-unknown-agent')"
  )
  conn.commit()
  conn.close()

  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(proj)
    # All per-agent columns must be empty. The legacy column is
    # left alone (legacy readers can still see it), but nothing flows
    # into the new columns.
    for agent in ("claude", "codex", "opencode"):
      assert db.get_sdk_session_id("chat-old", agent) == "", agent
    # Legacy column itself untouched.
    legacy = db._conn.execute(
      "SELECT sdk_session_id FROM sessions WHERE chat_id = ?", ("chat-old",)
    ).fetchone()
    assert legacy[0] == "opaque-id-of-unknown-agent"
    db.close()


def test_migrates_legacy_provider_sessions_table(tmp_path):
  # Existing user DBs from before the agent rename have a
  # provider_sessions table with a `provider` column. _ensure_tables
  # must rename both in place so resume data isn't lost on upgrade.
  # Schema strings here are the literal LEGACY shape — do not "fix"
  # them to the new names, that's the whole point.
  import sqlite3

  from nemo.db import _db_path

  proj = str(tmp_path / "project")
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    legacy_path = _db_path(proj)
  conn = sqlite3.connect(legacy_path)
  conn.execute("""
    CREATE TABLE provider_sessions (
      chat_id TEXT NOT NULL,
      provider TEXT NOT NULL,
      endpoint_key TEXT NOT NULL DEFAULT '',
      sdk_session_id TEXT NOT NULL DEFAULT '',
      PRIMARY KEY (chat_id, provider, endpoint_key)
    )
  """)
  conn.execute(
    "INSERT INTO provider_sessions (chat_id, provider, endpoint_key, sdk_session_id) "
    "VALUES ('chat1', 'claude', 'https://api.deepseek.com/anthropic', 'preserved-uuid')"
  )
  conn.commit()
  conn.close()

  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(proj)
    # Old table gone, new table exists with the data.
    tables = {
      row[0] for row in db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
      )
    }
    assert "provider_sessions" not in tables
    assert "agent_sessions" in tables
    # Column was also renamed.
    cols = {row[1] for row in db._conn.execute("PRAGMA table_info(agent_sessions)")}
    assert "agent" in cols
    assert "provider" not in cols
    # Resume id survived.
    assert db.get_sdk_session_id(
      "chat1", "claude", "https://api.deepseek.com/anthropic"
    ) == "preserved-uuid"
    db.close()


def test_migration_is_idempotent(tmp_path):
  # _ensure_tables runs every connect(). On a DB that already migrated,
  # the migration must be a no-op — otherwise repeated daemon restarts
  # would crash with "table agent_sessions already exists" or worse.
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    db.activate("sess1", "chat1", "opus")
    db.set_sdk_session_id("chat1", "uuid-1", "claude",
                          "https://api.deepseek.com/anthropic")
    db.close()
    # Second open re-runs _ensure_tables on a DB that has agent_sessions
    # already. Nothing should break and the data should persist.
    db2 = Database(str(tmp_path / "project"))
    assert db2.get_sdk_session_id(
      "chat1", "claude", "https://api.deepseek.com/anthropic"
    ) == "uuid-1"
    db2.close()


def test_migration_recovers_from_partial_state(tmp_path):
  # Pathological state: an interrupted previous migration left both
  # tables — provider_sessions has real data, agent_sessions exists but
  # is empty (e.g. created by CREATE TABLE IF NOT EXISTS before the
  # RENAME ran). _migrate_provider_to_agent must drop the empty new
  # table and let the rename complete, not bail out silently.
  import sqlite3

  from nemo.db import _db_path

  proj = str(tmp_path / "project")
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    legacy_path = _db_path(proj)
  conn = sqlite3.connect(legacy_path)
  # Legacy table with data.
  conn.execute("""
    CREATE TABLE provider_sessions (
      chat_id TEXT NOT NULL,
      provider TEXT NOT NULL,
      endpoint_key TEXT NOT NULL DEFAULT '',
      sdk_session_id TEXT NOT NULL DEFAULT '',
      PRIMARY KEY (chat_id, provider, endpoint_key)
    )
  """)
  conn.execute(
    "INSERT INTO provider_sessions (chat_id, provider, endpoint_key, sdk_session_id) "
    "VALUES ('chat1', 'claude', 'https://example.com', 'survived')"
  )
  # New table from a half-completed prior migration (empty, with the
  # final schema shape — what CREATE TABLE IF NOT EXISTS would produce).
  conn.execute("""
    CREATE TABLE agent_sessions (
      chat_id TEXT NOT NULL,
      agent TEXT NOT NULL,
      endpoint_key TEXT NOT NULL DEFAULT '',
      sdk_session_id TEXT NOT NULL DEFAULT '',
      PRIMARY KEY (chat_id, agent, endpoint_key)
    )
  """)
  conn.commit()
  conn.close()

  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(proj)
    # Real data from provider_sessions made it across.
    assert db.get_sdk_session_id(
      "chat1", "claude", "https://example.com"
    ) == "survived"
    tables = {
      row[0] for row in db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
      )
    }
    assert "provider_sessions" not in tables
    assert "agent_sessions" in tables
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


def test_set_autoesc(tmp_path):
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    db.activate("sess1", "chat1", "opus")
    s = db.get_session("sess1")
    assert s["autoesc"] is False
    db.set_autoesc("chat1", True)
    s = db.get_session("sess1")
    assert s["autoesc"] is True
    db.set_autoesc("chat1", False)
    s = db.get_session("sess1")
    assert s["autoesc"] is False
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
