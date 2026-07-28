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
import shutil
import subprocess
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
  skipped_active: list[SessionInfo] = field(default_factory=list)
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

# Claude's SDK stamps synthetic assistant messages (interrupt/continue stubs
# from a resumed session) with this literal model id. It's uninformative and,
# worse, the angle brackets open a stray tag when rendered in a Lark card, so
# the scanner ignores it and keeps looking for a real model id.
_SYNTHETIC_MODEL = "<synthetic>"

# A bare greeting as the opening prompt carries no topic — a session that
# starts with "hi" and only later gets to the real request would otherwise
# preview as "hi" and be unrecognisable in the recall picker. Skip these when
# choosing the first-prompt preview (falling back to the greeting only if the
# whole session is nothing but greetings).
_GREETINGS = frozenset({
  "hi", "hii", "hihi", "hi hi", "hello", "helo", "hey", "heya", "hiya",
  "yo", "sup", "hola", "在", "在吗", "在么", "在不在", "你好", "您好",
  "哈喽", "嗨", "早", "早上好", "晚上好", "hello?", "hi?",
})
_GREETING_STRIP = " \t\r\n!?.,~，。！？、…"


def _is_greeting(text: str) -> bool:
  """True for a content-free opening greeting (case/punctuation-insensitive)."""
  return text.strip().strip(_GREETING_STRIP).lower() in _GREETINGS


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
  first_any = ""  # fallback: the literal first prompt, even if a greeting
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
            snippet = text[:_PREVIEW_LIMIT]
            if not first_any:
              first_any = snippet
            if not _is_greeting(text):
              first_user = snippet
        if ev.get("type") == "assistant":
          msg = ev.get("message")
          if isinstance(msg, dict):
            m = msg.get("model")
            if isinstance(m, str) and m and m != _SYNTHETIC_MODEL:
              model = m
        # Heuristic stop: once we have a first user message and at least
        # one assistant model id, no need to scan further — files can be
        # hundreds of MB.
        if first_user and model:
          break
  except OSError:
    return None
  if not first_user:
    first_user = first_any

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
  "<environment_context>",
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
  first_any = ""  # fallback: the literal first prompt, even if a greeting
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
            snippet = text[:_PREVIEW_LIMIT]
            if not first_any:
              first_any = snippet
            if not _is_greeting(text):
              first_user = snippet
        if uuid and first_user and (model or cwd):
          break
  except OSError:
    return None
  if cwd != want_cwd or not uuid:
    return None
  if not first_user:
    first_user = first_any

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

# `/session recall` produces its own JSONL sessions in the same project dir:
# a throwaway read-only digest sub-session (whose first user prompt is the
# digest task, starting with DIGEST_TASK_INTRO) and, on completion, a
# recall-injection turn in the main agent (starting with RECALL_PROMPT_PREFIX).
# Neither is real user work, so the recall picker filters them out via
# `is_recallable`. These are the canonical marker strings — the producers
# (agent._RECALL_PROMPT_PREFIX, claude_agent._DIGEST_TASK) reference them here
# so the filter can't drift out of sync.
RECALL_PROMPT_PREFIX = "[Nemo recall] "
DIGEST_TASK_INTRO = (
  "A past coding session in this project was recorded as a JSONL transcript:"
)
_SYNTHETIC_FIRST_USER_PREFIXES = (RECALL_PROMPT_PREFIX, DIGEST_TASK_INTRO)


def is_recallable(info: SessionInfo) -> bool:
  """False for internal recall/digest sub-sessions that must not be offered
  as recall targets (they'd otherwise eat picker slots and confuse the user).
  """
  return not info.first_user_text.lstrip().startswith(
    _SYNTHETIC_FIRST_USER_PREFIXES)


def list_sessions(project_dir: str) -> list[SessionInfo]:
  """All sessions for ``project_dir`` across agents, newest first."""
  sessions = list_claude_sessions(project_dir) + list_codex_sessions(project_dir)
  sessions.sort(key=lambda s: s.mtime, reverse=True)
  return sessions


_SEARCH_CHUNK = 1 << 20


def _file_contains(path: str, needle: str) -> bool:
  """Case-insensitive substring search over a transcript, streamed.

  Read in chunks with a needle-sized overlap so a match straddling a chunk
  boundary is still found without holding a whole (possibly huge) transcript
  in memory. An unreadable file is simply not a match.
  """
  overlap = max(0, len(needle) - 1)
  tail = ""
  try:
    with open(path, "rb") as f:
      while chunk := f.read(_SEARCH_CHUNK):
        text = tail + chunk.decode("utf-8", errors="replace").lower()
        if needle in text:
          return True
        tail = text[-overlap:] if overlap else ""
  except OSError:
    return False
  return False


def search_sessions(
  sessions: Iterable[SessionInfo], query: str,
) -> list[SessionInfo]:
  """Filter ``sessions`` to those whose transcript mentions ``query``.

  Case-insensitive substring grep over the raw JSONL, so it matches user
  prompts, assistant replies and tool output alike — the point of
  `/session recall <keyword>` is "find the session where we talked about
  X", which the picker's short prompt previews can't answer. Input order
  (newest first) is preserved; an empty query matches everything.
  """
  needle = query.strip().lower()
  if not needle:
    return list(sessions)
  return [s for s in sessions if _file_contains(s.path, needle)]


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


# ---------------------------------------------------------------------------
# Recall digest cache
# ---------------------------------------------------------------------------
#
# `/session recall` summarises a past transcript via a throwaway read-only
# session (see CodingAgent.digest_transcript). A finished session's JSONL is
# immutable, so the summary is a pure function of (path, mtime, size): cache
# it keyed by uuid and re-validate against the live file. Repeat recalls of
# the same session then cost nothing (no second sub-session). The first line
# of the cache file is a ``<!-- mtime=… size=… -->`` stamp; a mismatch (the
# session was appended to, e.g. it's still the current one) invalidates it.

_DIGEST_STAMP_RE = re.compile(r"^<!-- mtime=([\d.]+) size=(\d+) -->\n")


def _digest_cache_dir() -> str:
  return os.path.expanduser("~/.nemo/digests")


def _digest_cache_path(uuid: str) -> str:
  return os.path.join(_digest_cache_dir(), f"{uuid}.md")


def _current_stat(path: str) -> tuple[float, int]:
  """Live (mtime, size) for ``path``; (0.0, 0) if it can't be stat'd."""
  try:
    st = os.stat(path)
  except OSError:
    return (0.0, 0)
  return (st.st_mtime, st.st_size)


def read_cached_digest(info: SessionInfo) -> str:
  """Return the cached recall digest for ``info`` if still fresh, else "".

  Fresh = the cache file's ``mtime``/``size`` stamp matches the transcript
  on disk right now. Any mismatch / missing file / parse failure returns ""
  so the caller recomputes.
  """
  path = _digest_cache_path(info.uuid)
  try:
    with open(path, encoding="utf-8") as f:
      blob = f.read()
  except OSError:
    return ""
  m = _DIGEST_STAMP_RE.match(blob)
  if not m:
    return ""
  cached_mtime, cached_size = float(m.group(1)), int(m.group(2))
  live_mtime, live_size = _current_stat(info.path)
  if live_size == 0 or cached_size != live_size:
    return ""
  # mtime can wobble at sub-second precision across filesystems; compare
  # with a small tolerance and lean on size as the real guard.
  if abs(cached_mtime - live_mtime) > 1.0:
    return ""
  return blob[m.end():]


def write_cached_digest(info: SessionInfo, digest: str) -> None:
  """Persist ``digest`` for ``info``, stamped with the transcript's current
  mtime/size. Best-effort — a cache write failure is never fatal to recall.
  """
  if not digest.strip():
    return
  mtime, size = _current_stat(info.path)
  if size == 0:
    return
  path = _digest_cache_path(info.uuid)
  try:
    os.makedirs(_digest_cache_dir(), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
      f.write(f"<!-- mtime={mtime} size={size} -->\n")
      f.write(digest)
  except OSError:
    pass


def _session_is_active(path: str) -> bool:
  """True if a live process currently holds the transcript open.

  The running Claude CLI / Codex SDK keeps its session JSONL open for
  append while a daemon is alive, so an open file handle is ground truth
  that *some* daemon is using this session right now — far more reliable
  than ``~/.nemo/pids`` (which goes stale on crash/kill and is subject to
  PID reuse) and independent of chat_id/DB. We use it to refuse deleting a
  session another daemon sharing this workspace is live on.

  If lsof isn't installed we can't probe, so return False (preserve the
  old "delete anything" behavior rather than refusing every delete). If
  lsof is present but the probe itself errors, assume active to be safe.
  """
  if shutil.which("lsof") is None:
    return False
  try:
    proc = subprocess.run(
      ["lsof", "-t", "--", path],
      capture_output=True, text=True, timeout=5,
    )
  except (OSError, subprocess.SubprocessError):
    return True
  return bool(proc.stdout.strip())


def _delete_session_files(candidates: Iterable[SessionInfo]) -> SessionDeleteResult:
  result = SessionDeleteResult()
  for session in candidates:
    if _session_is_active(session.path):
      result.skipped_active.append(session)
      continue
    try:
      os.remove(session.path)
    except OSError as e:
      result.failures.append(SessionDeleteFailure(session=session, error=str(e)))
    else:
      result.deleted.append(session)
      # Drop any cached recall digest so it can't outlive its transcript.
      try:
        os.remove(_digest_cache_path(session.uuid))
      except OSError:
        pass
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
