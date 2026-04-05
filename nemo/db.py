"""SQLite database for sessions, messages, and working state."""

from __future__ import annotations

import json
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
  project = hashlib.md5(workspace.encode()).hexdigest()[:12]
  db_dir = os.path.join(DB_BASE, project)
  os.makedirs(db_dir, exist_ok=True)
  return os.path.join(db_dir, "handoff-data.db")


def _connect(project_dir: str) -> sqlite3.Connection:
  path = _db_path(project_dir)
  conn = sqlite3.connect(path, timeout=10)
  conn.execute("PRAGMA journal_mode=WAL")
  conn.execute("PRAGMA busy_timeout=5000")
  conn.row_factory = sqlite3.Row
  return conn


def _ensure_tables(conn: sqlite3.Connection) -> None:
  conn.executescript("""
    CREATE TABLE IF NOT EXISTS sessions (
      session_id TEXT PRIMARY KEY,
      chat_id TEXT UNIQUE,
      session_tool TEXT DEFAULT '',
      session_model TEXT DEFAULT '',
      last_checked TEXT DEFAULT '',
      activated_at TEXT DEFAULT '',
      operator_open_id TEXT DEFAULT '',
      bot_open_id TEXT DEFAULT '',
      need_mention INTEGER DEFAULT 0,
      guests TEXT DEFAULT '[]',
      config_profile TEXT DEFAULT 'default',
      autoapprove INTEGER DEFAULT 0,
      message_filter TEXT DEFAULT 'concise'
    );
    CREATE TABLE IF NOT EXISTS messages (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      direction TEXT NOT NULL,
      message_id TEXT DEFAULT '',
      source_message_id TEXT DEFAULT '',
      chat_id TEXT DEFAULT '',
      message_time TEXT DEFAULT '',
      text TEXT DEFAULT '',
      title TEXT DEFAULT '',
      sent_at REAL DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS working_state (
      session_id TEXT PRIMARY KEY,
      message_id TEXT DEFAULT '',
      created_at REAL DEFAULT 0,
      counter INTEGER DEFAULT 0
    );
  """)


class Database:
  """Session-scoped database handle."""

  def __init__(self, project_dir: str):
    self._project_dir = project_dir
    self._conn = _connect(project_dir)
    _ensure_tables(self._conn)
    self._session_id: str | None = None

  def close(self):
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
    config_profile: str = "default",
  ) -> None:
    self._session_id = session_id
    self._conn.execute(
      """INSERT OR REPLACE INTO sessions
         (session_id, chat_id, session_tool, session_model, activated_at,
          operator_open_id, bot_open_id, need_mention, config_profile)
         VALUES (?, ?, 'Claude Agent SDK', ?, ?, ?, ?, ?, ?)""",
      (session_id, chat_id, model, str(int(time.time() * 1000)),
       operator_open_id, bot_open_id, int(need_mention), config_profile),
    )
    self._conn.commit()

  def deactivate(self, session_id: str) -> str | None:
    row = self._conn.execute(
      "SELECT chat_id FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    if not row:
      return None
    chat_id = row["chat_id"]
    self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    self._conn.execute("DELETE FROM working_state WHERE session_id = ?", (session_id,))
    self._conn.commit()
    return chat_id

  def get_session(self, session_id: str) -> dict | None:
    row = self._conn.execute(
      "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    if not row:
      return None
    d = dict(row)
    d["need_mention"] = bool(d.get("need_mention"))
    d["autoapprove"] = bool(d.get("autoapprove"))
    try:
      d["guests"] = json.loads(d.get("guests", "[]"))
    except Exception:
      d["guests"] = []
    return d

  def get_current_session(self) -> dict | None:
    if self._session_id:
      return self.get_session(self._session_id)
    return None

  def get_chat_owner(self, chat_id: str) -> str | None:
    row = self._conn.execute(
      "SELECT session_id FROM sessions WHERE chat_id = ?", (chat_id,)
    ).fetchone()
    return row["session_id"] if row else None

  def set_last_checked(self, session_id: str, ts: str) -> None:
    self._conn.execute(
      "UPDATE sessions SET last_checked = ? WHERE session_id = ?", (ts, session_id)
    )
    self._conn.commit()

  def set_guests(self, session_id: str, guests: list) -> None:
    self._conn.execute(
      "UPDATE sessions SET guests = ? WHERE session_id = ?",
      (json.dumps(guests), session_id),
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
    self, chat_id: str, text: str = "", title: str = "",
    source_message_id: str = "", message_time: str = "",
  ) -> None:
    self._conn.execute(
      """INSERT INTO messages (direction, chat_id, text, title,
         source_message_id, message_time, sent_at)
         VALUES ('received', ?, ?, ?, ?, ?, ?)""",
      (chat_id, text, title, source_message_id, message_time, time.time()),
    )
    self._conn.commit()

  def record_sent(
    self, message_id: str, text: str = "", title: str = "",
    chat_id: str = "",
  ) -> None:
    self._conn.execute(
      """INSERT INTO messages (direction, message_id, chat_id, text, title, sent_at)
         VALUES ('sent', ?, ?, ?, ?, ?)""",
      (message_id, chat_id, text, title, time.time()),
    )
    self._conn.commit()

  def get_latest_sent(self, chat_id: str) -> dict | None:
    row = self._conn.execute(
      """SELECT * FROM messages WHERE direction='sent' AND chat_id=?
         ORDER BY sent_at DESC LIMIT 1""",
      (chat_id,),
    ).fetchone()
    return dict(row) if row else None

  # --- Working state ---

  def set_working(self, session_id: str, message_id: str) -> None:
    self._conn.execute(
      """INSERT OR REPLACE INTO working_state (session_id, message_id, created_at, counter)
         VALUES (?, ?, ?, 0)""",
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
