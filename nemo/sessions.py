"""Read past coding-agent sessions stored on disk by the underlying CLIs.

Both Claude (``~/.claude/projects/<encoded-cwd>/<uuid>.jsonl``) and Codex
(``~/.codex/sessions/<yyyy>/<mm>/<dd>/rollout-<ts>-<uuid>.jsonl``) keep
the per-turn transcript locally as JSONL. ``/session list`` and
``/session recall`` walk those files so the user can browse what's been
discussed in this project across agents/endpoints and pull a past
session's contents back into the current conversation as memory (no
real SDK resume — see the per-endpoint isolation note in db.py).
"""

from __future__ import annotations

import glob as _glob
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable


@dataclass
class SessionInfo:
  """One past session discovered on disk."""
  uuid: str            # bare uuid (no rollout- prefix etc.)
  agent: str           # "claude" | "codex"
  path: str            # absolute path to the JSONL
  mtime: float         # last-modified epoch seconds
  first_user_text: str        # cleaned preview of the first real user prompt
  model: str                  # last-seen model id (e.g. "claude-opus-4-7", may be empty)
  last_user_texts: list[str] = field(default_factory=list)
  # ↑ up to ~3 most recent user prompts, oldest first; default empty
  # so the SessionInfo constructor remains callable from tests/helpers
  # that only care about the identity fields.


@dataclass
class SessionDeleteFailure:
  """A session file that could not be removed."""
  session: SessionInfo
  error: str


@dataclass
class SessionDeleteResult:
  """Outcome for a session deletion request."""
  deleted: list[SessionInfo] = field(default_factory=list)
  failures: list[SessionDeleteFailure] = field(default_factory=list)
  ambiguous: list[SessionInfo] = field(default_factory=list)
  not_found: str = ""


@dataclass
class SessionMessage:
  """A user/assistant message extracted from a session transcript."""
  role: str
  text: str
  timestamp: str


@dataclass
class SessionDetail:
  """Detailed transcript summary for one session."""
  session: SessionInfo
  size_bytes: int
  message_count: int
  first_message: SessionMessage | None = None
  last_messages: list[SessionMessage] = field(default_factory=list)


@dataclass
class SessionDetailResult:
  """Resolution result for /session info."""
  detail: SessionDetail | None = None
  ambiguous: list[SessionInfo] = field(default_factory=list)
  not_found: str = ""


# ---------------------------------------------------------------------------
# Path encoding
# ---------------------------------------------------------------------------

def claude_project_slug(project_dir: str) -> str:
  """Replicate the Claude CLI's project-dir → transcript-dir slug.

  The CLI resolves the realpath (so ``/var`` → ``/private/var`` on macOS,
  and symlinked worktrees resolve to their real location) and replaces
  EVERY non-alphanumeric char with ``-`` — ``/``, ``.``, ``_``, and spaces
  (e.g. macOS "Application Support") all collapse to ``-``. Getting this
  exactly right matters: glob below only finds a project's sessions if the
  slug matches the folder the CLI created byte-for-byte. A naive
  ``replace("/", "-")`` silently broke listing for any path containing a
  space, ``_``, ``.``, or a symlink.
  """
  return re.sub(r"[^a-zA-Z0-9]", "-", os.path.realpath(project_dir))


def _claude_project_dir(project_dir: str) -> str:
  """Map a project_dir to the Claude CLI's local storage folder."""
  return os.path.expanduser(
    f"~/.claude/projects/{claude_project_slug(project_dir)}")


_NOISE_TAG_RE = re.compile(
  r"<(local-command-caveat|command-name|command-message|command-args|"
  r"system-reminder|user-prompt-submit-hook)[\s\S]*?</\1>",
  re.IGNORECASE,
)
_OPEN_TAG_NEWLINES_RE = re.compile(r"\n{3,}")


def _clean_preview(text: str) -> str:
  """Strip noise tags so the first-user preview reads like a real prompt."""
  if not text:
    return ""
  text = _NOISE_TAG_RE.sub("", text)
  text = _OPEN_TAG_NEWLINES_RE.sub("\n\n", text).strip()
  return text


def _extract_text_blocks(content: object) -> str:
  """Flatten a Claude/Codex ``content`` value into plain text.

  Accepts a raw string or a list of content blocks. Recognised block
  shapes:
    - ``{"type": "text", "text": "..."}``                 (Claude)
    - ``{"type": "input_text", "text": "..."}``            (Codex)
    - ``{"type": "tool_use", ...}``                        (skipped)
  Unknown shapes contribute nothing rather than crashing.
  """
  if isinstance(content, str):
    return content
  if not isinstance(content, list):
    return ""
  parts: list[str] = []
  for block in content:
    if not isinstance(block, dict):
      continue
    btype = block.get("type")
    if btype in ("text", "input_text", "output_text"):
      t = block.get("text", "")
      if isinstance(t, str) and t:
        parts.append(t)
  return "\n".join(parts)


_TAIL_BYTES_DEFAULT = 256 * 1024
_PREVIEW_LIMIT = 120
_RECENT_USER_COUNT = 3


def _read_tail_lines(path: str, max_bytes: int = _TAIL_BYTES_DEFAULT) -> list[str]:
  """Return decoded lines from the tail of ``path`` without loading the
  whole file. Discards the first (possibly partial) line if the file
  exceeds ``max_bytes`` — the trade-off is missing the very oldest
  events when scanning a multi-MB session, which is exactly what tail
  is for.
  """
  try:
    size = os.path.getsize(path)
  except OSError:
    return []
  if size == 0:
    return []
  try:
    with open(path, "rb") as f:
      if size > max_bytes:
        f.seek(size - max_bytes)
        f.readline()  # discard the partial line we landed mid-way through
      raw = f.read()
  except OSError:
    return []
  return raw.decode("utf-8", errors="replace").splitlines()


def _claude_user_text(ev: dict) -> str:
  """Extract a user prompt from a Claude JSONL event, or ``""``."""
  if ev.get("type") != "user":
    return ""
  msg = ev.get("message")
  if not isinstance(msg, dict):
    return ""
  return _clean_preview(_extract_text_blocks(msg.get("content")))


def _codex_user_text(ev: dict) -> str:
  """Extract a user prompt from a Codex rollout event, or ``""``."""
  if ev.get("type") != "response_item":
    return ""
  p = ev.get("payload")
  if not isinstance(p, dict):
    return ""
  if p.get("type") != "message" or p.get("role") != "user":
    return ""
  return _clean_preview(_extract_text_blocks(p.get("content")))


def _event_timestamp(ev: dict, fallback_mtime: float) -> str:
  raw = ev.get("timestamp")
  if isinstance(raw, str) and raw:
    return raw
  return datetime.fromtimestamp(
    fallback_mtime, tz=timezone.utc).isoformat(timespec="seconds")


def _claude_message(ev: dict, fallback_mtime: float) -> SessionMessage | None:
  etype = ev.get("type")
  if etype not in ("user", "assistant"):
    return None
  msg = ev.get("message")
  if not isinstance(msg, dict):
    return None
  role = msg.get("role")
  if not isinstance(role, str) or role not in ("user", "assistant"):
    role = str(etype)
  text = _clean_preview(_extract_text_blocks(msg.get("content")))
  if not text:
    return None
  return SessionMessage(
    role=role,
    text=text,
    timestamp=_event_timestamp(ev, fallback_mtime),
  )


def _codex_message(ev: dict, fallback_mtime: float) -> SessionMessage | None:
  if ev.get("type") != "response_item":
    return None
  p = ev.get("payload")
  if not isinstance(p, dict):
    return None
  if p.get("type") != "message":
    return None
  role = p.get("role")
  if not isinstance(role, str) or role not in ("user", "assistant"):
    return None
  text = _clean_preview(_extract_text_blocks(p.get("content")))
  if not text:
    return None
  if role == "user" and _looks_like_injected_context(text):
    return None
  return SessionMessage(
    role=role,
    text=text,
    timestamp=_event_timestamp(ev, fallback_mtime),
  )


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------

def _scan_claude_session(path: str) -> SessionInfo | None:
  """Read a Claude JSONL just far enough to fill SessionInfo.

  Forward pass with early-exit gets the first user prompt + model id
  (cheap even on multi-MB files). A separate tail read picks up the
  last few user prompts so the listing shows "what was being discussed
  recently" without forcing a full scan.
  """
  try:
    st = os.stat(path)
  except OSError:
    return None
  uuid = os.path.basename(path).removesuffix(".jsonl")
  first_user = ""
  model = ""
  try:
    with open(path, encoding="utf-8") as f:
      for line in f:
        try:
          ev = json.loads(line)
        except json.JSONDecodeError:
          continue
        if not first_user:
          text = _claude_user_text(ev)
          if text:
            first_user = text[:_PREVIEW_LIMIT]
        if ev.get("type") == "assistant":
          msg = ev.get("message")
          if isinstance(msg, dict):
            m = msg.get("model")
            if isinstance(m, str) and m:
              model = m
        # Heuristic stop: once we have a first user message and at least
        # one assistant model id, no need to scan further — files can be
        # hundreds of MB.
        if first_user and model:
          break
  except OSError:
    return None

  # Tail pass for recent user prompts. Keeps the last N seen, in order.
  recent: list[str] = []
  for line in _read_tail_lines(path):
    try:
      ev = json.loads(line)
    except json.JSONDecodeError:
      continue
    text = _claude_user_text(ev)
    if text:
      recent.append(text[:_PREVIEW_LIMIT])
      if len(recent) > _RECENT_USER_COUNT:
        recent.pop(0)

  return SessionInfo(
    uuid=uuid, agent="claude", path=path, mtime=st.st_mtime,
    first_user_text=first_user, last_user_texts=recent, model=model,
  )


def list_claude_sessions(project_dir: str) -> list[SessionInfo]:
  """All Claude session JSONLs the CLI stored for ``project_dir``."""
  folder = _claude_project_dir(project_dir)
  if not os.path.isdir(folder):
    return []
  out: list[SessionInfo] = []
  for path in _glob.glob(os.path.join(folder, "*.jsonl")):
    info = _scan_claude_session(path)
    if info is not None:
      out.append(info)
  return out


# ---------------------------------------------------------------------------
# Codex
# ---------------------------------------------------------------------------

# Codex filenames look like rollout-2026-02-09T09-02-51-019c3fec-3890-78f0-988c-cdb3802197b8.jsonl
_CODEX_FILENAME_RE = re.compile(
  r"^rollout-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-([0-9a-f-]+)\.jsonl$"
)


# Codex always injects the project AGENTS.md (and similar boilerplate)
# as the first "user" message. That's not a real prompt; skip past it
# when looking for the preview.
_CODEX_INJECTED_PREFIXES = (
  "# AGENTS.md",
  "# Instructions",
  "<INSTRUCTIONS>",
)


def _looks_like_injected_context(text: str) -> bool:
  stripped = text.lstrip()
  return stripped.startswith(_CODEX_INJECTED_PREFIXES)


def _scan_codex_session(path: str, want_cwd: str) -> SessionInfo | None:
  """Read a Codex rollout JSONL just far enough to fill SessionInfo.

  Skips early (returns None) when the session's recorded cwd doesn't
  match ``want_cwd`` — Codex scatters files by date rather than by cwd,
  so this filter is the only way to scope to one project_dir.
  """
  try:
    st = os.stat(path)
  except OSError:
    return None
  uuid = ""
  cwd = ""
  first_user = ""
  model = ""
  try:
    with open(path, encoding="utf-8") as f:
      for line in f:
        try:
          ev = json.loads(line)
        except json.JSONDecodeError:
          continue
        etype = ev.get("type")
        if etype == "session_meta":
          p = ev.get("payload", {})
          if isinstance(p, dict):
            uuid = str(p.get("id") or uuid)
            cwd = str(p.get("cwd") or cwd)
            if cwd and cwd != want_cwd:
              return None  # not ours
            m = p.get("model") or p.get("model_name")
            if isinstance(m, str) and m:
              model = m
        elif etype == "turn_context" and not model:
          p = ev.get("payload", {})
          if isinstance(p, dict):
            m = p.get("model")
            if isinstance(m, str) and m:
              model = m
        elif etype == "response_item" and not first_user:
          text = _codex_user_text(ev)
          if text and not _looks_like_injected_context(text):
            first_user = text[:_PREVIEW_LIMIT]
        if uuid and first_user and (model or cwd):
          break
  except OSError:
    return None
  if cwd != want_cwd or not uuid:
    return None

  # Tail pass for recent user prompts. Skip the AGENTS.md auto-
  # injection so the recent-prompts column also lands on real prompts.
  recent: list[str] = []
  for line in _read_tail_lines(path):
    try:
      ev = json.loads(line)
    except json.JSONDecodeError:
      continue
    text = _codex_user_text(ev)
    if text and not _looks_like_injected_context(text):
      recent.append(text[:_PREVIEW_LIMIT])
      if len(recent) > _RECENT_USER_COUNT:
        recent.pop(0)

  return SessionInfo(
    uuid=uuid, agent="codex", path=path, mtime=st.st_mtime,
    first_user_text=first_user, last_user_texts=recent, model=model,
  )


def list_codex_sessions(project_dir: str) -> list[SessionInfo]:
  """All Codex rollout JSONLs whose recorded cwd is ``project_dir``."""
  base = os.path.expanduser("~/.codex/sessions")
  if not os.path.isdir(base):
    return []
  want = os.path.abspath(project_dir)
  out: list[SessionInfo] = []
  for path in _glob.glob(os.path.join(base, "**", "rollout-*.jsonl"),
                          recursive=True):
    if not _CODEX_FILENAME_RE.match(os.path.basename(path)):
      continue
    info = _scan_codex_session(path, want)
    if info is not None:
      out.append(info)
  return out


# ---------------------------------------------------------------------------
# Combined surface
# ---------------------------------------------------------------------------

def list_sessions(project_dir: str) -> list[SessionInfo]:
  """All sessions for ``project_dir`` across agents, newest first."""
  sessions = list_claude_sessions(project_dir) + list_codex_sessions(project_dir)
  sessions.sort(key=lambda s: s.mtime, reverse=True)
  return sessions


def find_session(
  uuid_or_prefix: str, sessions: Iterable[SessionInfo],
) -> list[SessionInfo]:
  """Return all sessions whose uuid matches ``uuid_or_prefix``.

  Exact-match wins over prefix matches; callers should treat a multi-
  element result with no exact match as ambiguous and ask the user to
  use more characters.
  """
  needle = uuid_or_prefix.strip().lower()
  if not needle:
    return []
  matches: list[SessionInfo] = []
  exact: list[SessionInfo] = []
  for s in sessions:
    u = s.uuid.lower()
    if u == needle:
      exact.append(s)
    elif u.startswith(needle):
      matches.append(s)
  return exact or matches


def _read_session_detail(info: SessionInfo) -> SessionDetail:
  try:
    size = os.path.getsize(info.path)
  except OSError:
    size = 0
  first: SessionMessage | None = None
  last: list[SessionMessage] = []
  count = 0
  extractor = _claude_message if info.agent == "claude" else _codex_message
  try:
    with open(info.path, encoding="utf-8") as f:
      for line in f:
        try:
          ev = json.loads(line)
        except json.JSONDecodeError:
          continue
        if not isinstance(ev, dict):
          continue
        message = extractor(ev, info.mtime)
        if message is None:
          continue
        count += 1
        if first is None:
          first = message
        last.append(message)
        if len(last) > 3:
          last.pop(0)
  except OSError:
    pass
  return SessionDetail(
    session=info,
    size_bytes=size,
    message_count=count,
    first_message=first,
    last_messages=last,
  )


def session_detail(
  project_dir: str,
  uuid_or_prefix: str,
  current_uuid: str = "",
) -> SessionDetailResult:
  """Return detailed info for one session.

  Empty ``uuid_or_prefix`` means "current session". If no current SDK
  session id exists yet, return not_found="" so callers can explain that
  the daemon has not produced a transcript yet.
  """
  target = uuid_or_prefix.strip() or current_uuid.strip()
  if not target:
    return SessionDetailResult(not_found="")
  all_sessions = list_sessions(project_dir)
  matches = find_session(target, all_sessions)
  if not matches:
    return SessionDetailResult(not_found=target)
  if len(matches) > 1:
    return SessionDetailResult(ambiguous=matches)
  return SessionDetailResult(detail=_read_session_detail(matches[0]))


def _delete_session_files(candidates: Iterable[SessionInfo]) -> SessionDeleteResult:
  result = SessionDeleteResult()
  for session in candidates:
    try:
      os.remove(session.path)
    except OSError as e:
      result.failures.append(SessionDeleteFailure(session=session, error=str(e)))
    else:
      result.deleted.append(session)
  return result


def remove_session(project_dir: str, uuid_or_prefix: str) -> SessionDeleteResult:
  """Remove one session matching ``uuid_or_prefix`` for ``project_dir``."""
  needle = uuid_or_prefix.strip()
  if not needle:
    return SessionDeleteResult(not_found=uuid_or_prefix)
  all_sessions = list_sessions(project_dir)
  matches = find_session(needle, all_sessions)
  if not matches:
    return SessionDeleteResult(not_found=needle)
  if len(matches) > 1:
    return SessionDeleteResult(ambiguous=matches)
  return _delete_session_files(matches)


def purge_sessions(
  project_dir: str,
  older_than_uuid_or_prefix: str = "",
  current_uuid: str = "",
) -> SessionDeleteResult:
  """Remove old sessions for ``project_dir``.

  With ``older_than_uuid_or_prefix``, removes sessions whose mtime is older
  than the matched session, excluding the matched session itself. Without it,
  removes every session except ``current_uuid``.
  """
  all_sessions = list_sessions(project_dir)
  target = older_than_uuid_or_prefix.strip()
  if target:
    matches = find_session(target, all_sessions)
    if not matches:
      return SessionDeleteResult(not_found=target)
    if len(matches) > 1:
      return SessionDeleteResult(ambiguous=matches)
    pivot = matches[0]
    candidates = [
      s for s in all_sessions
      if s.uuid != pivot.uuid and s.mtime < pivot.mtime
    ]
    return _delete_session_files(candidates)

  current = current_uuid.strip()
  candidates = [s for s in all_sessions if not current or s.uuid != current]
  return _delete_session_files(candidates)
