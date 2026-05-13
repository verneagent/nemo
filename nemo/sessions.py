"""Read past coding-agent sessions stored on disk by the underlying CLIs.

Both Claude (``~/.claude/projects/<encoded-cwd>/<uuid>.jsonl``) and Codex
(``~/.codex/sessions/<yyyy>/<mm>/<dd>/rollout-<ts>-<uuid>.jsonl``) keep
the per-turn transcript locally as JSONL. ``/session list`` and
``/session recall`` walk those files so the user can browse what's been
discussed in this project across providers/endpoints and pull a past
session's contents back into the current conversation as memory (no
real SDK resume — see the per-endpoint isolation note in db.py).
"""

from __future__ import annotations

import glob as _glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class SessionInfo:
  """One past session discovered on disk."""
  uuid: str            # bare uuid (no rollout- prefix etc.)
  provider: str        # "claude" | "codex"
  path: str            # absolute path to the JSONL
  mtime: float         # last-modified epoch seconds
  first_user_text: str        # cleaned preview of the first real user prompt
  model: str                  # last-seen model id (e.g. "claude-opus-4-7", may be empty)
  last_user_texts: list[str] = field(default_factory=list)
  # ↑ up to ~3 most recent user prompts, oldest first; default empty
  # so the SessionInfo constructor remains callable from tests/helpers
  # that only care about the identity fields.


# ---------------------------------------------------------------------------
# Path encoding
# ---------------------------------------------------------------------------

def _claude_project_dir(project_dir: str) -> str:
  """Map a project_dir to the Claude CLI's storage folder.

  Claude CLI's heuristic: replace every ``/`` in the absolute cwd with
  ``-``, which for absolute paths starts the result with ``-``.
  """
  abs_dir = os.path.abspath(project_dir)
  encoded = abs_dir.replace("/", "-")
  return os.path.expanduser(f"~/.claude/projects/{encoded}")


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
    uuid=uuid, provider="claude", path=path, mtime=st.st_mtime,
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
    uuid=uuid, provider="codex", path=path, mtime=st.st_mtime,
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
  """All sessions for ``project_dir`` across providers, newest first."""
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


# ---------------------------------------------------------------------------
# Digest extraction (for /session recall)
# ---------------------------------------------------------------------------

def _digest_claude(path: str, max_turns: int, max_chars: int) -> str:
  turns: list[tuple[str, str]] = []
  try:
    with open(path, encoding="utf-8") as f:
      for line in f:
        try:
          ev = json.loads(line)
        except json.JSONDecodeError:
          continue
        etype = ev.get("type")
        if etype not in ("user", "assistant"):
          continue
        msg = ev.get("message")
        if not isinstance(msg, dict):
          continue
        text = _clean_preview(_extract_text_blocks(msg.get("content")))
        if not text:
          continue
        role = "user" if etype == "user" else "assistant"
        turns.append((role, text))
  except OSError:
    return ""
  return _format_turns(turns, max_turns, max_chars)


def _digest_codex(path: str, max_turns: int, max_chars: int) -> str:
  turns: list[tuple[str, str]] = []
  try:
    with open(path, encoding="utf-8") as f:
      for line in f:
        try:
          ev = json.loads(line)
        except json.JSONDecodeError:
          continue
        if ev.get("type") != "response_item":
          continue
        p = ev.get("payload")
        if not isinstance(p, dict) or p.get("type") != "message":
          continue
        role = p.get("role")
        if role not in ("user", "assistant"):
          continue
        text = _clean_preview(_extract_text_blocks(p.get("content")))
        if not text:
          continue
        turns.append((str(role), text))
  except OSError:
    return ""
  return _format_turns(turns, max_turns, max_chars)


def _format_turns(
  turns: list[tuple[str, str]], max_turns: int, max_chars: int,
) -> str:
  """Take the LAST ``max_turns`` turns and join, then truncate from the
  front to ``max_chars``. Recent turns are the most useful for recall;
  the earliest setup messages tend to be project-AGENTS.md boilerplate.
  """
  tail = turns[-max_turns:] if max_turns > 0 else turns
  parts = [f"[{role}] {text}" for role, text in tail]
  digest = "\n\n".join(parts)
  if max_chars > 0 and len(digest) > max_chars:
    digest = "...(truncated)...\n\n" + digest[-max_chars:]
  return digest


def session_digest(
  info: SessionInfo, *, max_turns: int = 20, max_chars: int = 6000,
) -> str:
  """Plain-text digest of ``info`` suitable for injecting as recall context."""
  if info.provider == "claude":
    return _digest_claude(info.path, max_turns, max_chars)
  if info.provider == "codex":
    return _digest_codex(info.path, max_turns, max_chars)
  return ""
