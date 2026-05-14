"""SQLite database for sessions, messages, and working state."""

from __future__ import annotations

import os
import sqlite3
import time

from .config import DB_BASE


def _db_path(project_dir: str) -> str:
  """Compute the DB path for a project directory."""
  import hashlib
  import platform
  machine = platform.node().split(".")[0]
  folder = project_dir.replace("/", "-").strip("-")
  workspace = f"{machine}-{folder}"
  project_hash = hashlib.md5(workspace.encode()).hexdigest()[:12]
  db_dir = os.path.join(DB_BASE, project_hash)
  os.makedirs(db_dir, exist_ok=True)
  return os.path.join(db_dir, "nemo.db")


def _connect(project_dir: str) -> sqlite3.Connection:
  path = _db_path(project_dir)
  conn = sqlite3.connect(path, timeout=10, check_same_thread=False)
  conn.execute("PRAGMA journal_mode=WAL")
  conn.execute("PRAGMA busy_timeout=5000")
  conn.row_factory = sqlite3.Row
  return conn


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  chat_id TEXT UNIQUE,
  session_model TEXT DEFAULT '',
  last_checked TEXT DEFAULT '',
  activated_at TEXT DEFAULT '',
  operator_open_id TEXT DEFAULT '',
  bot_open_id TEXT DEFAULT '',
  need_mention INTEGER DEFAULT 0,
  autoapprove INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  direction TEXT NOT NULL,
  message_id TEXT DEFAULT '',
  source_message_id TEXT DEFAULT '',
  chat_id TEXT DEFAULT '',
  message_time TEXT DEFAULT '',
  text TEXT DEFAULT '',
  sent_at REAL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS working_state (
  session_id TEXT PRIMARY KEY,
  message_id TEXT DEFAULT '',
  created_at REAL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS agent_sessions (
  chat_id TEXT NOT NULL,
  agent TEXT NOT NULL,
  endpoint_key TEXT NOT NULL DEFAULT '',
  sdk_session_id TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (chat_id, agent, endpoint_key)
);
"""


_AGENT_SESSION_COLUMNS: dict[str, str] = {
  "claude": "claude_session_id",
  "codex": "codex_session_id",
  "opencode": "opencode_session_id",
}


def _migrate_provider_to_agent(conn: sqlite3.Connection) -> None:
  """Rename the legacy ``provider_sessions`` table (and its ``provider``
  column) to the agent terminology. Idempotent: only runs when the old
  table is present and the new one isn't.
  """
  has_old = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name='provider_sessions'"
  ).fetchone() is not None
  has_new = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_sessions'"
  ).fetchone() is not None
  if not has_old:
    return
  if has_new:
    # Both present is a pathological state from an interrupted migration;
    # drop the freshly-created empty new table and let the rename retry.
    conn.execute("DROP TABLE agent_sessions")
  conn.execute("ALTER TABLE provider_sessions RENAME TO agent_sessions")
  conn.execute("ALTER TABLE agent_sessions RENAME COLUMN provider TO agent")


def _ensure_tables(conn: sqlite3.Connection) -> None:
  # Migration runs before CREATE TABLE IF NOT EXISTS so the new table
  # isn't accidentally created empty while the old one still holds data.
  _migrate_provider_to_agent(conn)
  conn.executescript(_SCHEMA)
  cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
  if "sdk_session_id" not in cols:
    # Legacy column from before per-agent session storage. New
    # captain-nemo no longer writes here, but the column is still added
    # to brand-new DBs so older captain-nemo can read them.
    conn.execute("ALTER TABLE sessions ADD COLUMN sdk_session_id TEXT DEFAULT ''")
  for col in _AGENT_SESSION_COLUMNS.values():
    if col not in cols:
      conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} TEXT DEFAULT ''")
  # Deliberately NO backfill from the legacy column. The historical
  # `sdk_session_id` held whatever the most-recent daemon wrote
  # regardless of agent — a codex thread id can sit in there from
  # a previous codex run, and copying it into claude_session_id would
  # make the next claude daemon try to resume a codex thread.  The
  # claude SDK doesn't have the lazy-throw fallback the codex sidecar
  # does, so that surfaces as a silent subprocess exit-1 loop. Cheap
  # to lose one resume on upgrade; expensive to debug a wedged daemon.
  conn.commit()


class Database:
  """Session-scoped database handle."""

  def __init__(self, project_dir: str):
    self._project_dir = project_dir
    self._conn = _connect(project_dir)
    _ensure_tables(self._conn)
    self._session_id: str | None = None

  @property
  def path(self) -> str:
    return _db_path(self._project_dir)

  def close(self) -> None:
    self._conn.close()

  # --- Sessions ---

  def activate(
    self,
    session_id: str,
    chat_id: str,
    model: str,
    *,
    operator_open_id: str = "",
    bot_open_id: str = "",
    need_mention: bool = False,
  ) -> None:
    self._session_id = session_id
    # Preserve resume state from the previous row for this chat — both
    # the legacy sdk_session_id (still read by old captain-nemo) and the
    # per-agent columns introduced in 0.3.87. INSERT OR REPLACE
    # rewrites every column to its DEFAULT, so we have to fish out the
    # current values and pass them through.
    agent_cols = list(_AGENT_SESSION_COLUMNS.values())
    old_cols = ["sdk_session_id", *agent_cols]
    select_cols = ", ".join(old_cols)
    old = self._conn.execute(
      f"SELECT {select_cols} FROM sessions WHERE chat_id = ?", (chat_id,)
    ).fetchone()
    preserved = {col: ((old[col] or "") if old else "") for col in old_cols}

    insert_cols = [
      "session_id", "chat_id", "session_model", "activated_at",
      "operator_open_id", "bot_open_id", "need_mention",
    ] + old_cols
    placeholders = ", ".join(["?"] * len(insert_cols))
    self._conn.execute(
      f"""INSERT OR REPLACE INTO sessions ({", ".join(insert_cols)})
          VALUES ({placeholders})""",
      (session_id, chat_id, model, str(int(time.time() * 1000)),
       operator_open_id, bot_open_id, int(need_mention),
       *(preserved[col] for col in old_cols)),
    )
    self._conn.commit()

  def deactivate(self, session_id: str) -> str | None:
    # Keep the sessions row so sdk_session_id (codex thread / Claude SDK
    # session) survives clean shutdown and the next daemon's boot can
    # resume. The row's ownership fields are naturally overwritten by the
    # next activate() via INSERT OR REPLACE on the chat_id UNIQUE conflict.
    # Only working_state (per-session UI state) is ephemeral.
    row = self._conn.execute(
      "SELECT chat_id FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    if not row:
      return None
    chat_id = row["chat_id"]
    self._conn.execute("DELETE FROM working_state WHERE session_id = ?", (session_id,))
    self._conn.commit()
    return chat_id

  def get_session(self, session_id: str) -> dict[str, object] | None:
    row = self._conn.execute(
      "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    if not row:
      return None
    d = dict(row)
    d["need_mention"] = bool(d.get("need_mention"))
    d["autoapprove"] = bool(d.get("autoapprove"))
    return d

  def get_current_session(self) -> dict[str, object] | None:
    if not self._session_id:
      return None
    return self.get_session(self._session_id)

  def get_chat_owner(self, chat_id: str) -> str | None:
    row = self._conn.execute(
      "SELECT session_id FROM sessions WHERE chat_id = ?", (chat_id,)
    ).fetchone()
    return row["session_id"] if row else None

  def get_sdk_session_id(
    self, chat_id: str, agent: str, endpoint_key: str = "",
  ) -> str:
    """Look up the resume id for ``agent`` on ``chat_id``.

    Each agent kind (claude / codex / opencode) keeps its own slot so
    switching agents on a chat doesn't try to feed a Claude UUID into
    Codex (or vice versa). Within one agent, ``endpoint_key`` further
    isolates sessions by upstream endpoint — switching e.g. from real
    Anthropic to DeepSeek's Anthropic-compatible gateway gives the new
    endpoint its own fresh session rather than replaying a transcript
    whose ``thinking`` blocks were signed by a different vendor (the
    Anthropic API rejects those with HTTP 400).

    ``endpoint_key`` is the upstream URL (``EndpointConfig.base_url``);
    ``""`` means the agent's default endpoint. Two presets pointing
    at the same gateway (e.g. ``deepseek-v4-pro`` and
    ``deepseek-v4-flash`` both at ``api.deepseek.com/anthropic``)
    deliberately share a session — they share signing keys so the
    history is replayable. Returns ``""`` if no resume target.
    """
    if _AGENT_SESSION_COLUMNS.get(agent) is None:
      return ""
    if endpoint_key == "":
      # Default endpoint stays in the per-agent column so older
      # captain-nemo readers (which predate agent_sessions) keep
      # working — they only ever knew the default endpoint anyway.
      col = _AGENT_SESSION_COLUMNS[agent]
      row = self._conn.execute(
        f"SELECT {col} FROM sessions WHERE chat_id = ?", (chat_id,)
      ).fetchone()
      return row[col] if row and row[col] else ""
    row = self._conn.execute(
      "SELECT sdk_session_id FROM agent_sessions "
      "WHERE chat_id = ? AND agent = ? AND endpoint_key = ?",
      (chat_id, agent, endpoint_key),
    ).fetchone()
    return row["sdk_session_id"] if row and row["sdk_session_id"] else ""

  def set_sdk_session_id(
    self, chat_id: str, sdk_session_id: str, agent: str,
    endpoint_key: str = "",
  ) -> None:
    """Persist the most recent SDK session id for ``agent`` on ``chat_id``.

    ``endpoint_key`` scopes the id to a specific upstream endpoint —
    see ``get_sdk_session_id`` for why. Unknown agent kinds are a silent
    no-op (matches pre-existing semantics).
    """
    if _AGENT_SESSION_COLUMNS.get(agent) is None:
      return
    if endpoint_key == "":
      col = _AGENT_SESSION_COLUMNS[agent]
      self._conn.execute(
        f"UPDATE sessions SET {col} = ? WHERE chat_id = ?",
        (sdk_session_id, chat_id),
      )
    else:
      self._conn.execute(
        "INSERT INTO agent_sessions "
        "(chat_id, agent, endpoint_key, sdk_session_id) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(chat_id, agent, endpoint_key) "
        "DO UPDATE SET sdk_session_id = excluded.sdk_session_id",
        (chat_id, agent, endpoint_key, sdk_session_id),
      )
    self._conn.commit()

  def set_autoapprove(self, chat_id: str, enabled: bool) -> None:
    self._conn.execute(
      "UPDATE sessions SET autoapprove = ? WHERE chat_id = ?",
      (int(enabled), chat_id),
    )
    self._conn.commit()

  # --- Messages ---

  def record_received(
    self, chat_id: str, text: str = "",
    source_message_id: str = "", message_time: str = "",
  ) -> None:
    self._conn.execute(
      """INSERT INTO messages (direction, chat_id, text,
         source_message_id, message_time, sent_at)
         VALUES ('received', ?, ?, ?, ?, ?)""",
      (chat_id, text, source_message_id, message_time, time.time()),
    )
    self._conn.commit()

  def record_sent(
    self, message_id: str, text: str = "", chat_id: str = "",
  ) -> None:
    self._conn.execute(
      """INSERT INTO messages (direction, message_id, chat_id, text, sent_at)
         VALUES ('sent', ?, ?, ?, ?)""",
      (message_id, chat_id, text, time.time()),
    )
    self._conn.commit()

  def lookup_parent_message(self, message_id: str) -> dict[str, object] | None:
    """Look up a past message by ID — matches both sent (message_id) and
    received (source_message_id) messages."""
    row = self._conn.execute(
      """SELECT * FROM messages
         WHERE message_id = ? OR source_message_id = ?
         ORDER BY id DESC LIMIT 1""",
      (message_id, message_id),
    ).fetchone()
    return dict(row) if row else None

  # --- Working state ---

  def set_working(self, session_id: str, message_id: str) -> None:
    self._conn.execute(
      """INSERT OR REPLACE INTO working_state (session_id, message_id, created_at)
         VALUES (?, ?, ?)""",
      (session_id, message_id, time.time()),
    )
    self._conn.commit()

  def clear_working(self, session_id: str) -> None:
    self._conn.execute(
      "DELETE FROM working_state WHERE session_id = ?", (session_id,)
    )
    self._conn.commit()

  def get_working(self, session_id: str) -> str | None:
    row = self._conn.execute(
      "SELECT message_id FROM working_state WHERE session_id = ?", (session_id,)
    ).fetchone()
    return row["message_id"] if row else None
