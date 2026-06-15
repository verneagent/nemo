"""Main agent loop — event-driven message processing.

Wires together:
- Lark event stream (lark.events) — receives messages via WebSocket 长连接
- Command dispatch (commands) — built-in /clear, /model, etc.
- SDK turn execution (turn) — Claude Agent SDK
- Signal monitoring (monitor) — /esc, handback detection
- Card presentation (cards) — unified turn card
- Permission bridge (permissions) — button card + reaction approval
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import logging
import os
import re
import signal
import time
import urllib.error
import uuid
from typing import TYPE_CHECKING, Awaitable, Callable

from . import cards, commands, messages, monitor, shell_command
from .agent_factory import AGENT_KINDS, AgentKind, build_coding_agent, is_model_compatible
from .coding_agent import CodingAgent, EndpointConfig
from .channel import IncomingMessage, TurnCardCtx
from .fork import ForkManager
from .config import load_credentials
from .db import Database
from .lark_channel import LarkChannel
from .turn import (
  AnswerEvent, CompactNoticeEvent, CompactStartedEvent, DoneEvent,
  ProgressEvent, RateLimitNoticeEvent, StaleLeakNoticeEvent,
)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
  from .sessions import SessionDeleteResult

_USER_MESSAGE_EVENT_TYPES = {"", "message", "im.message.receive_v1"}


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]\n]+\]\([^)]+\)")
_TABLE_LINE_RE = re.compile(r"^\s*\|(?:[^|\n]+\|){1,}.*$")
_NUMBERED_LIST_RE = re.compile(r"\s*\d+[.)]\s+")
_BULLET_LIST_RE = re.compile(r"\s*[-*+]\s+")
_HEADING_RE = re.compile(r"\s{0,3}#{1,6}\s+\S")
_SETEXT_HEADING_UNDERLINE_RE = re.compile(r"\s{0,3}(?:=+|-+)\s*$")
_CONTROL_TAG_RE = re.compile(r"</?[a-zA-Z][^>\n]*>")
_EMPHASIS_RES = (
  re.compile(r"(?<!\*)\*\*[^*\n]+\*\*(?!\*)"),
  re.compile(r"(?<!\*)\*[^*\n]+\*(?!\*)"),
  re.compile(r"(?<!_)__[^_\n]+__(?!_)"),
  re.compile(r"(?<![\w_])_[^_\n]+_(?![\w_])"),
  re.compile(r"~~[^~\n]+~~"),
)


def _should_send_plain_text(text: str) -> bool:
  """Return True only for short, unformatted natural-language replies."""
  stripped = text.strip()
  if not stripped:
    return True

  if len(stripped) > 280:
    return False
  if "```" in stripped:
    return False
  if _INLINE_CODE_RE.search(stripped):
    return False
  if any(regex.search(stripped) for regex in _EMPHASIS_RES):
    return False
  if _MARKDOWN_LINK_RE.search(stripped):
    return False
  if _CONTROL_TAG_RE.search(stripped):
    return False

  lines = stripped.splitlines()
  if len(lines) > 6:
    return False

  non_empty_lines = [line.strip() for line in lines if line.strip()]
  if not non_empty_lines:
    return True

  if any(_HEADING_RE.match(line) for line in non_empty_lines):
    return False
  for prev_line, line in zip(non_empty_lines, non_empty_lines[1:]):
    if prev_line and _SETEXT_HEADING_UNDERLINE_RE.match(line):
      return False
  if any(line.startswith("> ") for line in non_empty_lines):
    return False
  if any(_TABLE_LINE_RE.match(line) for line in non_empty_lines):
    return False
  if "\n\n" in stripped:
    return False

  for line in non_empty_lines:
    if _BULLET_LIST_RE.match(line) or _NUMBERED_LIST_RE.match(line):
      return False

  if len(non_empty_lines) >= 4:
    return False

  return True


def _is_user_message_event(msg: IncomingMessage) -> bool:
  """Return True for events that represent user-authored chat messages."""
  return msg.event_type in _USER_MESSAGE_EVENT_TYPES


def _merge_pending(pending: list[IncomingMessage]) -> IncomingMessage | None:
  """Merge multiple pending messages into a single IncomingMessage.

  Messages collected during a turn are combined so that the coding agent
  receives one turn instead of N separate turns.  A short header tells
  the model how many messages were merged.
  """
  if not pending:
    return None
  if len(pending) == 1:
    return pending[0]

  # Separate regular text messages from non-text (commands, card actions, etc.)
  # Real events from the relay/Lark stream carry event_type
  # "im.message.receive_v1"; older code paths and tests use "message" or "".
  text_msgs: list[IncomingMessage] = []
  other_msgs: list[IncomingMessage] = []
  for msg in pending:
    if _is_user_message_event(msg) and msg.text.strip():
      text_msgs.append(msg)
    else:
      other_msgs.append(msg)

  merged: IncomingMessage | None = None
  if text_msgs:
    if len(text_msgs) == 1:
      merged = text_msgs[0]
    else:
      lines = [f"[用户在上一轮工作期间发送了 {len(text_msgs)} 条消息]"]
      for m in text_msgs:
        lines.append(m.text.strip())
      base = text_msgs[0]
      merged = IncomingMessage(
        event_type=base.event_type,
        chat_id=base.chat_id,
        chat_type=base.chat_type,
        sender_id=base.sender_id,
        message_id=text_msgs[-1].message_id,
        msg_type="text",
        text="\n".join(lines),
        mentions=base.mentions,
        create_time=text_msgs[-1].create_time,
        raw=base.raw,
      )

  # Return merged text + any non-text messages that need separate handling
  if other_msgs:
    # Non-text messages (card actions, commands) stay individual
    return merged, other_msgs  # type: ignore[return-value]
  return merged


def _requeue_pending(
  pending: list[IncomingMessage],
  channel: LarkChannel,
) -> None:
  """Merge pending messages and push back into the channel queue."""
  if not pending:
    return
  result = _merge_pending(pending)
  if result is None:
    return
  # _merge_pending returns a tuple when there are non-text messages
  if isinstance(result, tuple):
    merged, others = result
    if merged is not None:
      channel.push_back(merged)
    for msg in others:
      channel.push_back(msg)
  else:
    channel.push_back(result)


def _in_turn_filtered_out(
  msg: IncomingMessage,
  bot_open_id: str,
  is_own_message: Callable[[str], bool],
) -> bool:
  """Return True iff a regular in-turn message should be ignored under
  ``need_mention=True`` because it isn't bot-directed.

  Recall events and card.action.trigger events are system-level signals
  and must NOT pass through this filter — callers handle those branches
  before invoking this helper.
  """
  kept = messages.filter_bot_interactions(
    [msg], bot_open_id, is_own_message=is_own_message)
  return not kept


async def _send_response(
  channel: LarkChannel, chat_id: str, text: str, db: Database,
) -> str | None:
  text = text.strip()
  if not text:
    return None
  try:
    if _should_send_plain_text(text):
      msg_id = await channel.send_text(chat_id, text)
      log.info("Response sent transport=text chat=%s msg=%s", chat_id, msg_id)
    else:
      card = cards.build_markdown_card(text)
      msg_id = await channel.send_card(chat_id, card)
      log.info("Response sent transport=card chat=%s msg=%s", chat_id, msg_id)
    db.record_sent(msg_id, text=text[:500], chat_id=chat_id)
    _register_msg(msg_id, chat_id)
    return msg_id
  except Exception as e:
    log.error("Send error: %s", e)
    return None


def _register_msg(msg_id: str, chat_id: str) -> None:
  """Register message for reaction routing (relay only, best-effort)."""
  from .config import load_relay_config
  relay_url, _ = load_relay_config()
  if relay_url and msg_id:
    from . import relay as relay_client
    relay_client.register_message(msg_id, chat_id)


def _truncate_for_preview(text: str, limit: int = 2000) -> str:
  """Return text clipped to ``limit`` chars at a line boundary."""
  if len(text) <= limit:
    return text
  return text[:limit].rsplit("\n", 1)[0] + "\n\n_…truncated_"


def _cancel_emoji(elapsed: float) -> str:
  """Pick an emoji based on how long the turn ran before cancellation."""
  if elapsed < 3:
    return "👋"
  if elapsed < 30:
    return "🛑"
  if elapsed < 120:
    return "😮‍💨"
  if elapsed < 300:
    return "💨"
  return "🫠"


def _update_done_card_with_fallback(
  *,
  channel: LarkChannel,
  chat_id: str,
  turn_card_id: str,
  final_text: str,
  thinking,
  elapsed: int,
  usage,
  session_id: str,
  await_channel,
  register_msg,
  compact_notice: str = "",
  answered_questions=None,
) -> str:
  """Update the done card with tiered fallback; return the resulting id.

  1. Try the full-body card.
  2. On failure (other than auth-class HTTPError), retry with a truncated
     preview — this recovers transport blips and card-body-too-large alike.
  3. If the preview retry also fails, upload the full text as a .md file
     and update the card with a short preview + "sent as file" note.
  """
  full_card = cards.build_turn_card(
    "done", body=final_text, steps=thinking,
    elapsed=elapsed, usage=usage, session_id=session_id,
    compact_notice=compact_notice,
    answered_questions=answered_questions,
  )
  try:
    prev_id = turn_card_id
    turn_card_id = await_channel(channel.update_card(turn_card_id, full_card))
    if turn_card_id != prev_id:
      register_msg(turn_card_id, chat_id)
    return turn_card_id
  except Exception as e:
    log.warning("Failed to update done card: %s", e)
    first_err = e

  # Auth-class HTTP errors won't be fixed by shrinking or uploading a file.
  if (
    isinstance(first_err, urllib.error.HTTPError)
    and first_err.code in (401, 403, 404)
  ) or not final_text:
    return turn_card_id

  preview_body = _truncate_for_preview(final_text)
  preview_card = cards.build_turn_card(
    "done", body=preview_body, steps=thinking,
    elapsed=elapsed, usage=usage, session_id=session_id,
    compact_notice=compact_notice,
    answered_questions=answered_questions,
  )
  try:
    prev_id = turn_card_id
    turn_card_id = await_channel(
      channel.update_card(turn_card_id, preview_card))
    if turn_card_id != prev_id:
      register_msg(turn_card_id, chat_id)
    return turn_card_id
  except Exception as e:
    log.warning("Preview card update also failed: %s", e)

  try:
    import tempfile
    file_dir = os.path.join("/tmp/nemo", "nemo-files")
    os.makedirs(file_dir, exist_ok=True)
    fd, overflow_path = tempfile.mkstemp(
      suffix=".md", prefix="nemo-response-", dir=file_dir)
    with os.fdopen(fd, "w") as f:
      f.write(final_text)
    from .lark import api as lark_api
    file_key = lark_api.upload_file(channel.token, overflow_path)
    lark_api.send_file(channel.token, chat_id, file_key)
    log.info("Sent overflow response as file: %s", overflow_path)
    preview = final_text[:500].rsplit("\n", 1)[0]
    preview += f"\n\n_…full response ({len(final_text)} chars) sent as file_"
    fallback_card = cards.build_turn_card(
      "done", body=preview, steps=thinking,
      elapsed=elapsed, usage=usage, session_id=session_id,
      compact_notice=compact_notice,
      answered_questions=answered_questions,
    )
    prev_id = turn_card_id
    turn_card_id = await_channel(
      channel.update_card(turn_card_id, fallback_card))
    if turn_card_id != prev_id:
      register_msg(turn_card_id, chat_id)
  except Exception as e:
    log.warning("Failed to send overflow fallback: %s", e)
  return turn_card_id


async def _handle_turn_error(
  message: str,
  exc: Exception,
  channel: LarkChannel,
  chat_id: str,
  db: Database,
  session_id: str,
  card_id: str | None,
  steps: list,
  turn_start: float,
) -> None:
  """Display a red error card for SDK turn errors (timeout, rate limit, etc.)."""
  log.error("Turn error: %s", exc)
  try:
    if card_id:
      elapsed = int(time.time() - turn_start)
      err_card = cards.build_turn_card(
        "error", body=f"**{message}**",
        steps=steps, elapsed=elapsed,
      )
      await channel.update_card(card_id, err_card)
      db.clear_working(session_id)
    else:
      err_card = cards.build_card("Error", body=f"**{message}**", color="red")
      msg_id = await channel.send_card(chat_id, err_card)
      db.record_sent(msg_id, text=message[:500], chat_id=chat_id)
  except Exception as e:
    log.warning("Failed to send error card: %s", e)


async def _handle_diag(
  channel: LarkChannel,
  chat_id: str,
  project_dir: str,
  db: Database,
) -> None:
  """Run diagnostics and send results as a card."""
  results: list[str] = []

  # Check token
  try:
    _ = channel.token  # triggers auto-refresh if expired
    results.append("Token: OK")
  except Exception as e:
    results.append(f"Token: FAIL ({e})")

  # Check send/receive
  try:
    test_card = cards.build_card("Diag", body="test", color="grey")
    msg_id = await channel.send_card(chat_id, test_card)
    if msg_id:
      results.append("Send card: OK")
      try:
        await channel.delete_message(msg_id)
      except Exception as e:
        log.debug("Failed to delete diag test message: %s", e)
    else:
      results.append("Send card: FAIL (no msg_id)")
  except Exception as e:
    results.append(f"Send card: FAIL ({e})")

  # Check workspace tag
  try:
    from .workspace import get_workspace_id
    ws_id = get_workspace_id(project_dir)
    info = await channel.get_chat_info(chat_id)
    desc = info.get("description", "")
    tag = f"workspace:{ws_id}"
    if tag in desc:
      results.append(f"Workspace tag: OK ({ws_id})")
    else:
      results.append(f"Workspace tag: MISSING ({ws_id})")
  except Exception as e:
    results.append(f"Workspace tag: FAIL ({e})")

  body = "\n".join(f"- {r}" for r in results)
  diag_card = cards.build_card("Diagnostics", body=body, color="blue")
  try:
    await channel.send_card(chat_id, diag_card)
  except Exception as e:
    log.error("Failed to send diag card: %s", e)


async def _handle_session_list(
  channel: LarkChannel,
  chat_id: str,
  project_dir: str,
  db: Database,
  current_sdk_session_id: str,
) -> None:
  """Send a markdown card listing all known sessions for ``project_dir``.

  Sessions are pulled from both Claude CLI's storage (`~/.claude/...`)
  and Codex's (`~/.codex/...`) and merged into one mtime-desc list.
  Each row shows uuid prefix, agent, model, age, and a short user-
  prompt preview so the operator can identify which one to recall.
  """
  from . import sessions as _sessions
  sessions = _sessions.list_sessions(project_dir)
  if not sessions:
    await _send_response(
      channel, chat_id,
      f"No past sessions found in `{project_dir}`.",
      db,
    )
    return
  now = time.time()
  lines: list[str] = []
  for s in sessions[:30]:
    age = max(0, int(now - s.mtime))
    if age < 3600:
      when = f"{age // 60}m ago"
    elif age < 86400:
      when = f"{age // 3600}h ago"
    else:
      when = f"{age // 86400}d ago"
    marker = " ← current" if s.uuid == current_sdk_session_id else ""
    model = f" `{s.model}`" if s.model else ""
    bullets = _format_session_previews(s)
    lines.append(
      f"- `{s.uuid[:8]}` · **{s.agent}**{model} · {when}{marker}\n"
      f"{bullets}"
    )
  more = ""
  if len(sessions) > 30:
    more = f"\n\n(+{len(sessions) - 30} older sessions not shown)"
  body = (
    f"📂 `{project_dir}` — {len(sessions)} session(s)\n\n"
    + "\n".join(lines)
    + more
    + "\n\nRecall one with `/session recall <uuid prefix>`."
  )
  await _send_response(channel, chat_id, body, db)


def _format_session_message(msg, label: str) -> str:
  text = msg.text.replace("\n", " ").strip()
  if len(text) > 240:
    text = text[:237] + "..."
  return f"- **{label}** `{msg.timestamp}` · {msg.role}: {text}"


async def _handle_session_info(
  channel: LarkChannel,
  chat_id: str,
  project_dir: str,
  target: str,
  current_sdk_session_id: str,
  db: Database,
) -> None:
  """Send basic metadata plus first and last messages for one session."""
  from . import sessions as _sessions
  result = _sessions.session_detail(
    project_dir, target, current_uuid=current_sdk_session_id)
  if result.not_found or (not target.strip() and not current_sdk_session_id):
    if not target.strip() and not current_sdk_session_id:
      body = (
        "Current session has no SDK session id yet. "
        "If this daemon just started or no LLM turn has completed, "
        "there is no transcript to inspect."
      )
    else:
      body = (
        f"No session matches `{result.not_found}` in this project. "
        f"Run `/session list` to see what's available."
      )
    await _send_response(channel, chat_id, body, db)
    return
  if result.ambiguous:
    ambig = ", ".join(f"`{m.uuid[:8]}`" for m in result.ambiguous[:5])
    await _send_response(
      channel, chat_id,
      f"`{target}` is ambiguous — matches {ambig}. "
      f"Use more characters from the uuid.",
      db,
    )
    return
  if result.detail is None:
    await _send_response(channel, chat_id, "Session info unavailable.", db)
    return

  detail = result.detail
  s = detail.session
  import datetime as _dt
  modified = _dt.datetime.fromtimestamp(s.mtime).strftime("%Y-%m-%d %H:%M:%S")
  size_kb = max(1, detail.size_bytes // 1024) if detail.size_bytes else 0
  current = "yes" if s.uuid == current_sdk_session_id else "no"
  lines = [
    "**Session Info**",
    "",
    f"- UUID: `{s.uuid}`",
    f"- Agent: **{s.agent}**",
    f"- Model: `{s.model or 'unknown'}`",
    f"- Current: **{current}**",
    f"- Messages: **{detail.message_count}**",
    f"- Last modified: `{modified}`",
    f"- File: `{s.path}`",
    f"- Size: ~{size_kb}KB",
  ]
  if detail.message_count == 0:
    lines.extend([
      "",
      "No user/assistant messages recorded yet. "
      "This can happen for a newly-created session before the first "
      "LLM turn writes its transcript.",
    ])
  else:
    lines.extend(["", "**First Message**"])
    if detail.first_message is not None:
      lines.append(_format_session_message(detail.first_message, "first"))
    lines.extend(["", "**Last Messages**"])
    for idx, msg in enumerate(detail.last_messages, start=1):
      lines.append(_format_session_message(msg, f"last {idx}"))
  await _send_response(channel, chat_id, "\n".join(lines), db)


def _format_session_previews(s) -> str:
  """Render the per-session preview lines: first prompt + the last few.

  Deduplicates so a session with ≤4 user prompts doesn't repeat the
  first one in the "recent" bullets, and tags ages so the operator can
  tell at a glance whether a session was short or long.
  """
  def _flatten(text: str, limit: int = 80) -> str:
    return text.replace("\n", " ").strip()[:limit] or "(no user message)"
  parts: list[str] = []
  first = _flatten(s.first_user_text)
  parts.append(f"   ▸ {first}")
  recent = list(getattr(s, "last_user_texts", []) or [])
  # Drop the first prompt from the "recent" tail if it's already shown.
  if recent and recent[0].strip() == s.first_user_text.strip():
    recent = recent[1:]
  for text in recent:
    parts.append(f"   · {_flatten(text)}")
  return "\n".join(parts)


def _transcript_format_hint(agent: str) -> str:
  """The on-disk JSONL event shape for a given agent's transcript.

  Shared by the digest sub-session (``digest_transcript``) and the inline
  fallback recall prompt so both describe the file the same way.
  """
  return {
    "claude": (
      "One JSON event per line. ``type:\"user\"`` carries the user "
      "prompt at ``message.content`` (string or list of "
      "``{type:\"text\",text:...}`` blocks). ``type:\"assistant\"`` "
      "carries the model reply with ``message.model`` and the same "
      "content shape. Tool uses appear as ``tool_use`` / "
      "``tool_result`` blocks — skim them, don't quote verbatim."
    ),
    "codex": (
      "One JSON event per line. The first event is ``session_meta`` "
      "(payload.cwd, payload.model). Real turns are "
      "``type:\"response_item\"`` with ``payload.role`` of user/"
      "assistant and ``payload.content[].text`` or ``input_text``. "
      "The first user message is usually injected AGENTS.md "
      "boilerplate — skip past it."
    ),
  }.get(agent, "")


async def _handle_session_recall(
  channel: LarkChannel,
  chat_id: str,
  project_dir: str,
  target: str,
  coding_agent: CodingAgent,
  db: Database,
) -> str:
  """Resolve ``target`` to a past session and recall it into context.

  Two paths, picked at runtime:

  - **Digest** (preferred): a throwaway, context-free read-only session
    reads the full transcript and returns a compact summary
    (``coding_agent.digest_transcript``). Only that summary — plus a
    pointer to the transcript for on-demand detail — is injected into the
    live agent's context, so recall costs the working session a few
    hundred tokens instead of a pile of raw JSONL. The summary is cached
    by session uuid (transcripts are immutable once ended), so repeat
    recalls are free.
  - **Inline fallback**: when the agent can't run a blank side session
    (Codex / OpenCode, or the digest came back empty) the live agent is
    handed the path and asked to Read it itself — the original behavior.

  Returns an ERROR/usage string for the caller to send (no match,
  ambiguous, …), or "" on success (this function sends its own progress
  ack and injects the recall turn). Never touches ``_sdk_session_id`` —
  SDK resume across endpoints replays thinking-block signatures and 400s
  (see db.py).
  """
  from . import sessions as _sessions
  from .channel import IncomingMessage
  if not target.strip():
    return "Usage: `/session recall <uuid prefix>`"
  all_sessions = _sessions.list_sessions(project_dir)
  matches = _sessions.find_session(target, all_sessions)
  if not matches:
    return (f"No session matches `{target}` in this project. "
            f"Run `/session list` to see what's available.")
  if len(matches) > 1:
    ambig = ", ".join(f"`{m.uuid[:8]}`" for m in matches[:5])
    return (f"`{target}` is ambiguous — matches {ambig}. "
            f"Use more characters from the uuid.")
  info = matches[0]
  try:
    size = os.path.getsize(info.path)
  except OSError:
    size = 0
  size_kb = max(1, size // 1024)
  import datetime as _dt
  when = _dt.datetime.fromtimestamp(info.mtime).strftime("%Y-%m-%d %H:%M")
  fmt_hint = _transcript_format_hint(info.agent)

  # Progress ack up front: producing the digest can take a while on a
  # large transcript (the sub-session reads it in chunks), so don't leave
  # the user staring at silence between the click/command and the recall
  # turn's placeholder card.
  await _send_response(
    channel, chat_id,
    f"📖 Recalling session `{info.uuid[:8]}` ({info.agent}, ~{size_kb}KB)…",
    db,
  )

  # Digest path: cache → blank sub-session. A "" digest (unsupported agent,
  # empty/failed run) drops us to the inline fallback below.
  summary = _sessions.read_cached_digest(info)
  if not summary:
    try:
      summary = await coding_agent.digest_transcript(info.path, fmt_hint)
    except Exception as exc:  # never let a recall digest crash the loop
      log.warning("digest_transcript raised during recall: %s", exc)
      summary = ""
    if summary:
      _sessions.write_cached_digest(info, summary)

  meta = (
    f"agent `{info.agent}`, uuid `{info.uuid[:8]}`, model "
    f"`{info.model or 'unknown'}`, last activity {when}, ~{size_kb}KB"
  )
  if summary:
    prompt = (
      _RECALL_PROMPT_PREFIX
      + "The user asked you to recall a past coding session in this "
      f"project ({meta}). A separate read-only pass already read the full "
      "transcript and produced this summary:\n\n"
      "---\n"
      f"{summary}\n"
      "---\n\n"
      f"The full transcript is at {info.path} if you need a specific "
      "detail the summary doesn't cover — Read just the relevant slice "
      "(prefer the tail), don't load it all. Hold the gist in working "
      "memory; the user may refer back to it. Reply with a short "
      "confirmation of what you recovered (lean on the summary above). "
      "Do NOT re-execute any actions described in the past session."
    )
  else:
    prompt = (
      _RECALL_PROMPT_PREFIX
      + "The user asked you to recall a past coding session in this "
      "project. Its JSONL transcript lives at:\n\n"
      f"  {info.path}\n\n"
      f"Session metadata: {meta}.\n\n"
      f"Format: {fmt_hint}\n\n"
      "Use your Read tool to skim it — for large files prefer the tail "
      "(most recent turns are usually more relevant than the opening "
      "setup). Figure out: what was being worked on, what got decided, "
      "and any pending threads. Hold the gist in working memory; the user "
      "may refer back to it. Reply with a short summary (a few bullet "
      "points) of what you recovered. Do NOT re-execute any actions "
      "described in the past session."
    )
  recall_msg = IncomingMessage(
    event_type="im.message.receive_v1",
    chat_id=chat_id,
    sender_id="",  # synthetic — not from a real user
    message_id=f"recall_{info.uuid[:8]}_{int(time.time())}",
    msg_type="text",
    text=prompt,
    create_time=str(int(time.time() * 1000)),
    is_internal=True,
  )
  channel.push_back(recall_msg)
  return ""


async def _handle_btw(
  channel: LarkChannel,
  chat_id: str,
  coding_agent: CodingAgent,
  sdk_session_id: str,
  question: str,
) -> None:
  """Answer a `/btw` side question and post it as an ephemeral card.

  Ephemerality is enforced on Nemo's side here: the answer card is sent
  WITHOUT ``db.record_sent`` / ``_register_msg``, so it never lands in
  the SQLite ``messages`` table the agent reads back as chat history.
  Combined with the adapter forking (or, pre-first-turn, never
  persisting) the SDK session, the answer can never re-enter the
  agent's context. (The inbound ``/btw`` line itself
  is still recorded like any other command — only the answer and any
  reasoning are kept out of history.)
  """
  answer = ""
  try:
    answer = await coding_agent.side_question(question, sdk_session_id)
  except Exception as exc:  # never let a side question crash the loop
    log.warning("side_question raised: %s", exc)
    answer = f"⚠️ btw failed: {exc}"
  if not answer:
    answer = (
      "btw isn't supported by the current agent. Side questions are "
      "Claude-only — switch with `/agent claude`."
    )
  body = (
    f"{answer}\n\n"
    f"<font color='grey'>Ephemeral — this side answer is not saved to "
    f"the conversation.</font>"
  )
  card = cards.build_markdown_card(body, title="💬 btw", color="indigo")
  try:
    await channel.send_card(chat_id, card)
  except Exception as exc:
    log.warning("Failed to send btw card: %s", exc)


def _format_session_delete_result(
  action: str, result: SessionDeleteResult,
) -> str:
  if result.not_found:
    return (
      f"No session matches `{result.not_found}` in this project. "
      f"Run `/session list` to see what's available."
    )
  if result.ambiguous:
    ambig = ", ".join(f"`{m.uuid[:8]}`" for m in result.ambiguous[:5])
    return (
      f"`{action}` target is ambiguous — matches {ambig}. "
      f"Use more characters from the uuid."
    )

  parts: list[str] = []
  if result.deleted:
    by_agent: dict[str, int] = {}
    for s in result.deleted:
      by_agent[s.agent] = by_agent.get(s.agent, 0) + 1
    detail = ", ".join(f"{count} {agent}" for agent, count in sorted(by_agent.items()))
    parts.append(f"Removed {len(result.deleted)} session(s) ({detail}).")
  else:
    parts.append("No sessions matched the purge criteria.")

  if result.failures:
    failures = ", ".join(
      f"`{f.session.uuid[:8]}`: {f.error}" for f in result.failures[:3]
    )
    more = "" if len(result.failures) <= 3 else f" (+{len(result.failures) - 3} more)"
    parts.append(f"Failed to remove {len(result.failures)} session(s): {failures}{more}")

  return " ".join(parts)


async def _handle_session_rm(
  channel: LarkChannel,
  chat_id: str,
  project_dir: str,
  target: str,
  db: Database,
) -> None:
  from . import sessions as _sessions
  if not target.strip():
    await _send_response(channel, chat_id, "Usage: `/session rm <uuid prefix>`", db)
    return
  result = _sessions.remove_session(project_dir, target)
  await _send_response(channel, chat_id, _format_session_delete_result("rm", result), db)


async def _handle_session_purge(
  channel: LarkChannel,
  chat_id: str,
  project_dir: str,
  target: str,
  current_sdk_session_id: str,
  db: Database,
) -> None:
  from . import sessions as _sessions
  result = _sessions.purge_sessions(
    project_dir,
    older_than_uuid_or_prefix=target,
    current_uuid=current_sdk_session_id,
  )
  await _send_response(
    channel, chat_id, _format_session_delete_result("purge", result), db)


# Marker prefixing the synthetic prompt that `/session recall` injects.
# Shared so the turn loop can recognise a recall turn (its first SDK token
# can take a while — resume + reading the past transcript) and surface a
# placeholder working card up front instead of leaving the user staring at
# the recall ack during the silence.
_RECALL_PROMPT_PREFIX = "[Nemo recall] "


# When an SDK turn times out the underlying CLI/agent is usually choking on a
# heavy context that's been asked to do too much at once. The next user turn
# gets this preamble so the agent paces itself instead of repeating the
# overrun. One-shot per timeout — cleared on /clear or after one use.
_PACING_HINT_PREFIX = (
  "[Nemo 系统提示] 上一回合超时了——通常是上下文较重、一次想做太多事造成的。"
  "这一回合请放慢节奏：把工作拆成多个小步，每回合只做一步，做完等用户确认再继续，"
  "不要试图一次完成所有事。\n\n"
  "（以下是用户的新消息）\n"
)


def _format_rate_limit_notice(event: RateLimitNoticeEvent) -> str:
  """Render a RateLimitNoticeEvent as a one-line banner string.

  Returns "" when the status is "allowed" (limit cleared) so callers can use
  truthiness to know whether to show or hide the banner.
  """
  status = (event.status or "").lower()
  if status == "allowed":
    return ""

  if status == "rejected":
    head = "⛔ Rate limit hit"
  elif status == "allowed_warning":
    head = "⚠️ Rate limit warning"
  else:
    head = f"Rate limit: {status}" if status else "Rate limit"

  bits = [head]
  if event.rate_limit_type:
    bits.append(f"({event.rate_limit_type})")
  if event.utilization is not None:
    try:
      bits.append(f"{event.utilization * 100:.0f}% used")
    except Exception:
      pass
  if event.resets_at:
    delta = int(event.resets_at - time.time())
    if delta > 0:
      if delta < 60:
        bits.append(f"resets in {delta}s")
      elif delta < 3600:
        # Ceiling so "resets in N min" doesn't visibly tick down by a minute
        # within the first second of being shown.
        mins = -(-delta // 60)
        bits.append(f"resets in {mins}m")
      else:
        h, rem = divmod(delta, 3600)
        mins = -(-rem // 60)
        if mins == 60:
          h += 1
          mins = 0
        bits.append(f"resets in {h}h {mins}m" if mins else f"resets in {h}h")
  return " ".join(bits)


_COMPACT_TRIGGER_LABEL = {"auto": "自动", "manual": "手动"}


def _format_compact_started(event: CompactStartedEvent) -> str:
  """Render a CompactStartedEvent as a one-line timeline step.

  Fires from the SDK's PreCompact hook just before the CLI begins
  summarising the conversation. We only know the trigger here — tokens
  and duration are not known until the matching CompactNoticeEvent.
  """
  label = _COMPACT_TRIGGER_LABEL.get(event.trigger, event.trigger or "")
  if label:
    return f"🗜 上下文{label}压缩中…"
  return "🗜 上下文压缩中…"


def _format_compact_notice(event: CompactNoticeEvent) -> str:
  """Render a CompactNoticeEvent as a one-line timeline step.

  Emitted on the matching SystemMessage(subtype="compact_boundary") after
  the CLI finishes compacting; carries pre/post tokens and duration. The
  paired CompactStartedEvent step (if present) lives just above this one
  in the timeline.
  """
  label = _COMPACT_TRIGGER_LABEL.get(event.trigger, event.trigger or "")
  parts: list[str] = []
  if event.pre_tokens and event.post_tokens:
    parts.append(f"{event.pre_tokens:,} → {event.post_tokens:,} tokens")
  elif event.pre_tokens:
    parts.append(f"{event.pre_tokens:,} tokens")
  if event.duration_ms:
    parts.append(f"{event.duration_ms / 1000:.1f}s")
  detail = " · ".join(parts)
  head = f"🗜 上下文已{label}压缩" if label else "🗜 上下文已压缩"
  return f"{head}（{detail}）" if detail else head


def _endpoint_change_note(
  old_endpoint_key: str, new_endpoint_key: str, sdk_session_id: str,
) -> str:
  """One-line trailing note shown after ``/model`` when the upstream
  endpoint actually changed.

  Returns ``""`` when the endpoint did not change (e.g. opus↔sonnet on
  default Anthropic) so a routine model swap stays a one-line confirm.
  When the endpoint flipped, the per-endpoint session isolation makes
  the new model blind to the other endpoint's transcript — surface
  that explicitly so users don't think the bot "forgot" anything.
  """
  if old_endpoint_key == new_endpoint_key:
    return ""
  if sdk_session_id:
    return (f" Resuming this endpoint's prior conversation "
            f"(session `{sdk_session_id[:8]}`); the other endpoint's "
            f"history is kept separately.")
  return (" Fresh conversation on this endpoint — the previous "
          "endpoint's history is preserved, switch back to continue it.")


def _model_picker_options(
  agent: AgentKind, project_dir: str,
) -> list[tuple[str, str]]:
  """Flatten the model catalog into ``(display_label, model_name)``
  pairs for the picker dropdown.

  Visible models come first (the everyday picks), then API-only slugs
  with a `(API-only)` suffix, then aliases pointing at their canonical
  name. ``hidden`` legacy entries are intentionally omitted — they stay
  reachable via ``/model <name>`` for muscle memory but don't deserve a
  slot in the dropdown.
  """
  from .agent_factory import model_catalog_for_agent
  catalog = model_catalog_for_agent(agent, project_dir)
  options: list[tuple[str, str]] = []
  seen: set[str] = set()
  for name in catalog.visible:
    options.append((name, name))
    seen.add(name)
  for name in getattr(catalog, "api_only", ()):
    label = f"{name} (API-only)"
    options.append((label, name))
    seen.add(name)
  for alias, full in catalog.aliases.items():
    if alias in seen:
      continue
    options.append((f"{alias} → {full}", alias))
    seen.add(alias)
  return options


async def _send_model_picker(
  channel: LarkChannel,
  chat_id: str,
  project_dir: str,
  ctx: commands.AgentContext,
  db: Database,
) -> None:
  """Send the interactive `/model` picker card.

  Falls back to a plain-text listing when there are no models to pick
  (e.g. opencode with no configured catalog) so the user still gets a
  helpful response instead of an empty dropdown.
  """
  from .agent_factory import model_catalog_for_agent
  options = _model_picker_options(ctx.agent, project_dir)
  catalog = model_catalog_for_agent(ctx.agent, project_dir)
  if not options:
    listing = commands._format_model_catalog(catalog)
    await _send_response(
      channel, chat_id,
      f"Current model: **{ctx.model}** (agent **{ctx.agent}**)\n\n"
      f"{listing}\n\nUsage: `/model <name>`",
      db,
    )
    return
  # The catalog listing (multi-line: Available / API-only / aliases /
  # opencode dynamic-models note) renders as plain markdown so its
  # nuances stay visible — the bare dropdown labels can't carry them.
  # The one-line usage hint renders as a small grey footer note. They
  # MUST stay separate: a <font>-wrapped note can't span a \n\n
  # paragraph break without leaking a bare </font> (see
  # build_model_picker_card). The hint also uses `/model NAME` (no
  # angle brackets) so a literal `<name>` can't open a stray tag.
  listing = commands._format_model_catalog(catalog)
  hint = "Pick a model and click Submit. Or type `/model NAME` directly."
  card = cards.build_model_picker_card(
    options,
    current_model=ctx.model,
    current_agent=ctx.agent,
    chat_id=chat_id,
    info=listing,
    hint=hint,
  )
  try:
    msg_id = await channel.send_card(chat_id, card)
    db.record_sent(msg_id, text="Switch Model", chat_id=chat_id)
    _register_msg(msg_id, chat_id)
  except Exception as e:
    log.error("Model picker send failed: %s", e)
    listing = commands._format_model_catalog(catalog)
    await _send_response(
      channel, chat_id,
      f"Current model: **{ctx.model}** (agent **{ctx.agent}**)\n\n"
      f"{listing}\n\nUsage: `/model NAME`",
      db,
    )


async def _lock_model_picker(
  channel: LarkChannel,
  picker_msg_id: str,
  *,
  agent: str,
  model: str,
  project_dir: str,
  ok: bool = True,
  attempted: str = "",
  reason: str = "",
) -> None:
  """PATCH a submitted /model picker into its locked confirmation state.

  Removes the dropdown + Submit button (the card is rebuilt without a
  form) so the picker can't be re-submitted with a now-stale model
  list, and prominently shows the current agent + model. Keeps the
  available-model catalog visible so the locked card still tells the user
  what else they can switch to (via ``/model NAME``) instead of wiping it
  down to bare agent+model. No-op when there is no picker card to lock
  (e.g. the submit came from the standalone fallback card, or the card id
  wasn't propagated).
  """
  if not picker_msg_id:
    return
  try:
    from .agent_factory import model_catalog_for_agent
    listing = commands._format_model_catalog(
      model_catalog_for_agent(agent, project_dir))
    card = cards.build_model_switched_card(
      agent=agent, model=model, ok=ok, attempted=attempted, reason=reason,
      info=listing)
    await channel.update_card(picker_msg_id, card)
    log.info("Locked /model picker %s (ok=%s agent=%s model=%s)",
             picker_msg_id, ok, agent, model)
  except Exception as exc:
    log.warning("Picker confirm-lock update failed: %s", exc)


def _agent_picker_options() -> list[tuple[str, str]]:
  """Flatten the CodingAgent kinds into ``(display_label, agent_name)``
  pairs for the picker dropdown.

  The set is fixed by ``agent_factory.AgentKind``; we hard-code the order
  (claude, claude-cli, codex, opencode) so the picker is stable across
  daemons and doesn't depend on dict iteration order. Labels carry each
  agent's default model so the user can see what the switch will land on
  without reading the catalog separately.
  """
  from .agent_factory import default_model_for_agent
  agents: tuple[AgentKind, ...] = ("claude", "claude-cli", "codex", "opencode")
  return [(f"{name} (default: {default_model_for_agent(name)})", name)
          for name in agents]


async def _send_agent_picker(
  channel: LarkChannel,
  chat_id: str,
  ctx: commands.AgentContext,
  db: Database,
) -> None:
  """Send the interactive `/agent` picker card — mirrors `_send_model_picker`."""
  options = _agent_picker_options()
  info = (
    "Switching resets the model to that agent's default and keeps each "
    "agent's last session id separately, so flipping back resumes the "
    "prior conversation."
  )
  hint = "Pick an agent and click Submit. Or type `/agent NAME` directly."
  card = cards.build_agent_picker_card(
    options,
    current_agent=ctx.agent,
    current_model=ctx.model,
    chat_id=chat_id,
    info=info,
    hint=hint,
  )
  try:
    msg_id = await channel.send_card(chat_id, card)
    db.record_sent(msg_id, text="Switch Agent", chat_id=chat_id)
    _register_msg(msg_id, chat_id)
  except Exception as e:
    log.error("Agent picker send failed: %s", e)
    await _send_response(
      channel, chat_id,
      f"Current agent: **{ctx.agent}** (model **{ctx.model}**)\n\n"
      f"Available: {', '.join(f'`{k}`' for k in sorted(AGENT_KINDS))}. {info}\n\n"
      f"Usage: `/agent NAME`",
      db,
    )


async def _lock_agent_picker(
  channel: LarkChannel,
  picker_msg_id: str,
  *,
  agent: str,
  model: str,
  ok: bool = True,
  attempted: str = "",
  reason: str = "",
) -> None:
  """PATCH a submitted /agent picker into its locked confirmation state.

  Same role as ``_lock_model_picker``: removes the dropdown + Submit
  button so the picker can't be re-submitted, and shows the resulting
  agent + model. No-op when there is no picker card id to lock (e.g.
  the submit came from /agent <name> typed directly).
  """
  if not picker_msg_id:
    return
  try:
    card = cards.build_agent_switched_card(
      agent=agent, model=model, ok=ok, attempted=attempted, reason=reason)
    await channel.update_card(picker_msg_id, card)
    log.info("Locked /agent picker %s (ok=%s agent=%s model=%s)",
             picker_msg_id, ok, agent, model)
  except Exception as exc:
    log.warning("Agent picker confirm-lock update failed: %s", exc)


# Each session is its own card block (markdown + Recall button), so cap the
# count to keep the card from getting absurdly tall on mobile.
_SESSION_PICKER_LIMIT = 12


def _session_picker_options(
  project_dir: str, current_sdk_session_id: str = "",
) -> list[tuple[str, str]]:
  """Flatten past sessions into ``(description_markdown, uuid)`` pairs,
  newest first, for the recall picker.

  Each description is a two-line markdown block — a bold meta line
  (``<uuid8> · <agent> · <model> · <age>`` + a current-session tag) and
  the first user prompt as a preview — so the operator can recognise a
  session at a glance without `/session list`. The card renders one
  Recall button under each block.
  """
  from . import sessions as _sessions
  now = time.time()
  out: list[tuple[str, str]] = []
  for s in _sessions.list_sessions(project_dir)[:_SESSION_PICKER_LIMIT]:
    age = max(0, int(now - s.mtime))
    if age < 3600:
      when = f"{age // 60}m ago"
    elif age < 86400:
      when = f"{age // 3600}h ago"
    else:
      when = f"{age // 86400}d ago"
    bits = [f"`{s.uuid[:8]}`", s.agent]
    if s.model:
      bits.append(s.model)
    bits.append(when)
    meta = " · ".join(bits)
    if s.uuid == current_sdk_session_id:
      meta += " · _(current)_"
    preview = (s.first_user_text or "").replace("\n", " ").strip()[:120]
    description = f"**{meta}**"
    if preview:
      description = f"{description}\n{preview}"
    out.append((description, s.uuid))
  return out


async def _send_session_picker(
  channel: LarkChannel,
  chat_id: str,
  project_dir: str,
  db: Database,
  current_sdk_session_id: str = "",
) -> None:
  """Send the interactive `/session recall` picker card.

  Falls back to the plain-text `/session list` when there are no sessions
  to pick (so the user still gets a useful answer instead of an empty
  list)."""
  options = _session_picker_options(project_dir, current_sdk_session_id)
  if not options:
    await _send_response(
      channel, chat_id,
      f"No past sessions found in `{project_dir}`.",
      db,
    )
    return
  hint = "Pick a session and click Recall. Or type `/session recall UUID`."
  card = cards.build_session_picker_card(options, chat_id=chat_id, hint=hint)
  try:
    msg_id = await channel.send_card(chat_id, card)
    db.record_sent(msg_id, text="Recall Session", chat_id=chat_id)
    _register_msg(msg_id, chat_id)
  except Exception as e:
    log.error("Session picker send failed: %s", e)
    await _send_response(
      channel, chat_id,
      "Couldn't render the session picker. Use `/session list` then "
      "`/session recall UUID`.",
      db,
    )


async def _lock_session_picker(
  channel: LarkChannel,
  picker_msg_id: str,
  *,
  uuid: str,
  agent: str = "",
  model: str = "",
) -> None:
  """PATCH a submitted /session recall picker into its locked state.

  Collapses the per-session list + Recall buttons to a single confirmation
  so the same pick can't be re-clicked. No-op without a card id (e.g. the
  recall was typed directly)."""
  if not picker_msg_id:
    return
  try:
    card = cards.build_session_recalled_card(uuid=uuid, agent=agent, model=model)
    await channel.update_card(picker_msg_id, card)
    log.info("Locked /session recall picker %s (uuid=%s)", picker_msg_id, uuid[:8])
  except Exception as exc:
    log.warning("Session picker confirm-lock update failed: %s", exc)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _restart_model_arg(model: str, endpoint_key: str, agent: str) -> str:
  """The value to pass as ``--model`` when relaunching the daemon.

  Once a preset is active ``model`` is the resolved *remote id* (e.g.
  ``deepseek-v4-pro[1m]``) and ``endpoint_key`` is its base URL — but
  ``--model`` at startup only routes preset *names* through the endpoint
  registry. Reverse-resolve so /restart and /upgrade relaunch with a name
  the next boot can route; otherwise the new daemon sends the remote id to
  the default endpoint and every turn fails "model not found" (including
  the forked /btw CLI exiting 1). Plain models on the default endpoint
  (no ``endpoint_key``) pass through unchanged.
  """
  if endpoint_key:
    from .presets import preset_name_for_endpoint
    name = preset_name_for_endpoint(model, endpoint_key, agent)
    if name:
      return name
  return model


async def _interrupt_and_drain(
  coding_agent: CodingAgent,
  sdk_task: "asyncio.Task[object]",
  timeout: float = 10.0,
) -> str:
  """Interrupt the running turn for a stop/esc and wait for it to wind down.

  Returns a short status for logging: ``"clean"`` (turn ended on its own),
  ``"aborted"`` (the turn cancelled itself in response to the interrupt),
  or ``"forced"`` (interrupt raised, task force-cancelled).

  The subtle case this exists for: stop pressed while the turn is inside a
  reconnect loop (e.g. SDK #788 stale-leak-resume). ``interrupt()`` calls
  ``cancel()``, so the turn raises ``asyncio.CancelledError`` to abort the
  reconnect — exactly what stop wants. But ``CancelledError`` is a
  ``BaseException``, so the old ``except Exception`` let it escape to the
  main loop's loop-level ``except asyncio.CancelledError``, which set
  ``running = False`` and tore the whole daemon down. Stop must only
  interrupt the turn (AGENTS.md), so swallow the turn task's own
  cancellation here — while still re-raising if *this* caller is the one
  genuinely being cancelled (a real shutdown).
  """
  try:
    await coding_agent.interrupt()
    await asyncio.wait_for(sdk_task, timeout=timeout)
    return "clean"
  except asyncio.CancelledError:
    current = asyncio.current_task()
    if current is not None and current.cancelling():
      raise  # we ourselves are being cancelled → genuine shutdown
    return "aborted"
  except Exception as exc:
    log.warning("SDK interrupt failed (%s), cancelling task", exc)
    sdk_task.cancel()
    return "forced"


async def main_loop(
  chat_id: str,
  project_dir: str,
  model: str,
  agent: AgentKind = "claude",
  permission_mode: str = "bypassPermissions",
  effort: str = "",
  system_prompt: str = "",
  endpoint: EndpointConfig | None = None,
  endpoint_key: str = "",
) -> int:
  """Run the agent main loop."""
  session_id = str(uuid.uuid4())

  credentials = load_credentials()
  if not credentials:
    log.error("No credentials configured")
    return 1

  channel = LarkChannel(chat_id)
  await channel.start()

  # Resolve operator & bot
  operator_open_id = ""
  bot_open_id = ""
  try:
    email = credentials.get("email", "")
    operator_open_id, bot_open_id = await channel.resolve_operator_and_bot(email)
  except Exception as e:
    log.warning("Operator/bot lookup failed: %s", e)

  # Database
  db = Database(project_dir)

  # Let LarkChannel recover quoted-message text from our own DB when the
  # Lark API can't (e.g. interactive cards lose body content on get_message).
  def _db_parent_lookup(mid: str) -> str | None:
    row = db.lookup_parent_message(mid)
    if not row:
      return None
    text = row.get("text")
    return str(text) if text else None

  channel.parent_lookup = _db_parent_lookup

  # Clean stale sessions (preserve sdk_session_id for resume).
  # Per-(agent, endpoint) lookup so switching agents or endpoints
  # on a chat doesn't feed a Claude UUID into Codex (or vice versa), and
  # doesn't replay a transcript whose thinking blocks were signed by a
  # different upstream — the right answer is always to start that slot
  # fresh while leaving other slots' stored ids intact for when the user
  # switches back.
  _resume_sdk_id = ""
  try:
    old_owner = db.get_chat_owner(chat_id)
    if old_owner:
      _resume_sdk_id = db.get_sdk_session_id(chat_id, agent, endpoint_key)
      log.info("Cleaning stale session %s (sdk[%s/%s]=%s)", old_owner,
               agent, endpoint_key or "default",
               _resume_sdk_id[:8] if _resume_sdk_id else "none")
      db.deactivate(old_owner)
  except Exception as e:
    log.warning("Stale cleanup error: %s", e)

  # Ensure workspace tag and claim group
  await channel.ensure_workspace_claimed(project_dir, model)

  # Detect need_mention: nemo-managed groups (with workspace tag) or
  # 1-on-1 groups default to False. Other multi-human groups default to True.
  # Can be overridden via group config.
  from . import group_config as gcfg
  gc = gcfg.load_config(channel.token, chat_id)
  if "need_mention" in gc:
    need_mention = bool(gc["need_mention"])
  else:
    need_mention = True
    try:
      info = await channel.get_chat_info(chat_id)
      desc = str(info.get("description", "") or "")
      if "workspace:" in desc:
        # Nemo-managed group (created by auto_create_chat)
        need_mention = False
      else:
        members = await channel.get_chat_members(chat_id)
        human_count = sum(1 for m in members if m.get("member_id") != bot_open_id)
        if human_count <= 1:
          need_mention = False
    except Exception as e:
      log.warning("Failed to detect need_mention, defaulting to True: %s", e)

  # Activate session
  db.activate(
    session_id, chat_id, model,
    operator_open_id=operator_open_id,
    bot_open_id=bot_open_id,
    need_mention=need_mention,
  )
  log.info("Session %s activated (chat=%s bot=%s operator=%s need_mention=%s)",
           session_id, chat_id, bot_open_id[:16] if bot_open_id else "?",
           operator_open_id[:16] if operator_open_id else "?", need_mention)

  # autoesc: when True, a new user message arriving during an in-flight
  # turn auto-cancels the running turn before being processed.
  autoesc = bool((db.get_session(session_id) or {}).get("autoesc"))

  # Send start card
  log.info("Sending start card to %s", chat_id)
  from nemo import __version__
  folder = os.path.basename(project_dir) or project_dir
  start_lines = [f"📂 `{folder}`  ·  pid `{os.getpid()}`"]
  if _resume_sdk_id:
    start_lines.append(f"Session `{_resume_sdk_id[:8]}` resumed")
  start_note = project_dir
  start_card = cards.build_card(
    f"Nemo v{__version__} ({agent} · {model})",
    body="\n".join(start_lines),
    color="blue",
    note=start_note,
  )
  try:
    msg_id = await channel.send_card(chat_id, start_card)
    log.info("Start card sent: %s", msg_id)
  except Exception as e:
    log.error("Start card failed: %s", e)
    err_msg = str(e)
    if "230002" in err_msg or "NOT be out of the chat" in err_msg:
      return 1

  # Status tab — green idle, with agent name next to the dot.
  await channel.update_status(model, "idle", agent)

  # Periodic heartbeat (relay-based idle detection)
  _heartbeat_task: asyncio.Task | None = None
  from .config import load_relay_config
  relay_url, _ = load_relay_config()
  if relay_url:
    async def _heartbeat_loop():
      while True:
        await asyncio.sleep(30)
        await channel.send_heartbeat(model)

    _heartbeat_task = asyncio.create_task(_heartbeat_loop())

  coding_agent = build_coding_agent(
    agent,
    credentials, chat_id, db, channel,
    permission_mode=permission_mode,
    system_prompt=system_prompt,
    endpoint=endpoint,
  )
  if effort:
    coding_agent.set_effort(effort)
  # Resume previous SDK session if available
  _sdk_session_id: str = _resume_sdk_id
  # Track which upstream endpoint the running session belongs to so we
  # never resume a transcript across endpoints — see the comment on
  # ``db.get_sdk_session_id`` for why.
  _endpoint_key: str = endpoint_key
  # Snapshot saved at each /clear so /undo-clear can restore the
  # session id that was active just before the user reset. Held in
  # process memory only — does not survive a daemon restart.
  _prev_sdk_session_id: str = ""
  if _sdk_session_id:
    log.info("Resuming SDK session %s", _sdk_session_id[:8])
  try:
    await coding_agent.start(project_dir, model, resume=_sdk_session_id)
  except Exception as e:
    log.error("SDK startup failed: %s", e)
    err_card = cards.build_card(
      "Error", body=f"Startup failed:\n```\n{e}\n```", color="red"
    )
    try:
      await channel.send_card(chat_id, err_card)
    except Exception as send_err:
      log.warning("Failed to send SDK startup error card: %s", send_err)
    # Structural startup failures (missing CLI / sidecar / credentials) do
    # not recover by retrying later turns. Exit so a supervisor can
    # restart us cleanly rather than stay in a half-alive zombie state
    # where every user message produces another obscure error.
    raise

  # Context
  ctx = commands.AgentContext(model, project_dir, time.time())
  ctx.agent = agent
  ctx.effort = effort
  main_loop_ref = asyncio.get_running_loop()
  running = True
  _dissolve_on_exit = False
  # In-flight `/btw` side questions spawned during a running turn. Kept
  # daemon-scoped (not turn-scoped) so an answer started in turn N can
  # still arrive after turn N ends without its task being GC'd.
  _btw_tasks: set[asyncio.Task[None]] = set()
  # In-flight `/fork` open/close tasks — spawned fire-and-forget so opening a
  # fork (which starts a separate SDK subprocess) never blocks the main loop
  # or the in-turn signal watcher. Daemon-scoped so they survive turn ends.
  _fork_tasks: set[asyncio.Task[None]] = set()
  _pending_shell_contexts: list[str] = []

  def _queue_shell_context(context: str) -> None:
    _pending_shell_contexts.append(context)
    del _pending_shell_contexts[:-shell_command.MAX_PENDING_CONTEXTS]

  shell_manager = shell_command.ShellJobManager(
    channel,
    chat_id=chat_id,
    project_dir=project_dir,
    on_context=_queue_shell_context,
  )

  # /fork — read-only multi-turn sub-threads. The manager owns all fork
  # lifecycle + card rendering; the main loop only routes thread-scoped
  # messages to it (see fork-routing below) and spawns open/close tasks.
  async def _fork_notify(text: str) -> None:
    await _send_response(channel, chat_id, text, db)

  fork_mgr = ForkManager(channel, chat_id, _fork_notify)

  def _spawn_fork(coro: Awaitable[None]) -> None:
    t = asyncio.create_task(coro)
    _fork_tasks.add(t)
    t.add_done_callback(_fork_tasks.discard)

  def _route_fork_message(msg: IncomingMessage) -> bool:
    """If `msg` lands in a live fork's sub-thread, route it to that fork
    (concurrent with the main conversation) and return True. `/fork close`
    inside the thread closes it. Fork messages never enter main-chat
    history and bypass the @mention requirement (being in the thread is
    intent enough)."""
    sess = fork_mgr.get(msg.thread_id)
    if sess is None:
      return False
    ftext = (msg.text or "").strip()
    if ftext:
      fp = messages.strip_parent_quote(
        messages.strip_mentions(ftext, [msg], bot_open_id=bot_open_id) or ftext)
      if commands.is_fork_close(fp):
        _spawn_fork(fork_mgr.close(msg.thread_id))
      else:
        fork_mgr.route(msg.thread_id, fp)
    return True
  # One-shot flag: prepend a pacing hint to the next user prompt after an
  # SDK timeout. Cleared after use or on /clear.
  _pending_pacing_hint = False
  # Message id of the /model picker card awaiting a confirmation PATCH.
  # Set when a picker submit (model_switch:*) is accepted; the __model__
  # handler reads it after the switch lands and rewrites that card into
  # a locked "now on agent/model" state (dropdown + Submit removed) so
  # the picker can't be re-submitted with a now-stale model list.
  _pending_picker_msg_id = ""
  # Same role for the /agent picker — stored on form-submit, consumed by
  # the __agent__: handler after the switch lands (success or failure).
  _pending_agent_picker_msg_id = ""

  def handle_sig(_sig, _frame):
    nonlocal running
    running = False
    channel.push_back(IncomingMessage())

  signal.signal(signal.SIGINT, handle_sig)
  signal.signal(signal.SIGTERM, handle_sig)

  async def _restart_client(resume: str = ""):
    await coding_agent.reset(project_dir, model, resume=resume)

  def _is_own_message(mid: str) -> bool:
    """True iff `mid` was sent by this bot (recorded as direction='sent')."""
    row = db.lookup_parent_message(mid)
    return bool(row and row.get("direction") == "sent")

  # Load guest/coowner roles for authorization
  from .guests import get_member_roles
  _member_roles: dict[str, str] = {}
  try:
    _member_roles = get_member_roles(channel.token, chat_id)
  except Exception as e:
    log.warning("Failed to load member roles: %s", e)

  # ---- Main loop: event-driven ----
  while running:
    try:
      reply = await channel.receive(timeout=300)
      if reply is None:
        continue  # Timeout, keep waiting

      log.info("Event: type=%s chat=%s sender=%s text=%r",
               reply.event_type, reply.chat_id[:16] if reply.chat_id else "?",
               reply.sender_id[:16] if reply.sender_id else "?",
               reply.text[:60] if reply.text else "")

      # Skip card action events at top level (handled during turns)
      if reply.event_type == "card.action.trigger":
        if reply.action_value.get("action") == "shell_abort":
          job_id = str(reply.action_value.get("job_id", ""))
          if monitor.is_privileged(
              reply.operator_id, operator_open_id, _member_roles):
            await shell_manager.abort(job_id)
          else:
            log.info("Ignoring unauthorized shell abort by %s", reply.operator_id)
          continue
        action_str = str(reply.action_value.get("action", ""))
        if action_str == "model_picker_submit":
          # Submit clicked but the dropdown was empty — Lark falls back
          # to the button's own action value when form_value carries no
          # selection. Tell the user instead of silently dropping it.
          await _send_response(
            channel, chat_id,
            "Pick a model from the dropdown before submitting.", db,
          )
          continue
        if action_str.startswith("model_switch:"):
          # `/model` picker form submitted. The select_static's value
          # carries the ``model_switch:<name>`` prefix; the relay passes
          # the single-field form_value through as the action string.
          model_name = action_str.split(":", 1)[1].strip()
          if not model_name:
            await _send_response(
              channel, chat_id,
              "Pick a model from the dropdown before submitting.", db,
            )
            continue
          from .guests import is_authorized_sender
          if operator_open_id and not is_authorized_sender(
              reply.operator_id, operator_open_id, _member_roles):
            log.info(
              "Ignoring unauthorized model picker submit by %s",
              reply.operator_id,
            )
            continue
          # Pre-validate against the CURRENT agent. A picker built under
          # one agent can be submitted after the user has since switched
          # agents (the card stays live until submitted), at which point
          # its model list is stale. Reject up front and lock the card to
          # an error state so the now-incompatible model never reaches
          # the switch logic — and the stale dropdown can't be retried.
          if not is_model_compatible(agent, model_name):
            await _lock_model_picker(
              channel, reply.message_id, agent=agent, model=model,
              project_dir=project_dir, ok=False, attempted=model_name,
              reason=(
                f"`{model_name}` isn't available for agent **{agent}** "
                f"(the picker may be from before an /agent switch)."
              ),
            )
            await _send_response(
              channel, chat_id,
              f"**{model_name}** isn't available for agent **{agent}**. "
              f"Run `/model` for the current list.",
              db,
            )
            continue
          # Compatible — synthesise an internal ``/model <name>`` so the
          # existing /model dispatch handles the real switch (preset
          # resolution, SDK restart, response card). is_internal=True
          # bypasses sender/mention filters since the click already came
          # from an authorized operator. Stash the picker card id so the
          # __model__ handler can lock it to a confirmation state once
          # the switch lands.
          _pending_picker_msg_id = reply.message_id
          synthetic = dataclasses.replace(
            reply,
            event_type="im.message.receive_v1",
            msg_type="text",
            text=f"/model {model_name}",
            sender_id=reply.operator_id or operator_open_id,
            is_internal=True,
            action_value={},
            action_tag="",
          )
          channel.push_back(synthetic)
          continue
        if action_str == "agent_picker_submit":
          # Same shape as model_picker_submit — Lark falls back to the
          # button's own action value when form_value carries no selection.
          await _send_response(
            channel, chat_id,
            "Pick an agent from the dropdown before submitting.", db,
          )
          continue
        if action_str.startswith("agent_switch:"):
          # `/agent` picker form submitted. The select_static's value carries
          # the ``agent_switch:<name>`` prefix; the relay passes the single-
          # field form_value through as the action string (same path as
          # model_switch).
          agent_name = action_str.split(":", 1)[1].strip()
          if not agent_name:
            await _send_response(
              channel, chat_id,
              "Pick an agent from the dropdown before submitting.", db,
            )
            continue
          from .guests import is_authorized_sender
          if operator_open_id and not is_authorized_sender(
              reply.operator_id, operator_open_id, _member_roles):
            log.info(
              "Ignoring unauthorized agent picker submit by %s",
              reply.operator_id,
            )
            continue
          if agent_name not in AGENT_KINDS:
            # Should be unreachable (the dropdown only offers valid kinds),
            # but defend in case the wire delivers something weird.
            await _lock_agent_picker(
              channel, reply.message_id, agent=agent, model=model,
              ok=False, attempted=agent_name,
              reason=f"Unknown agent `{agent_name}`.",
            )
            await _send_response(
              channel, chat_id,
              f"Unknown agent **{agent_name}**.", db,
            )
            continue
          if agent_name == agent:
            # No-op switch: lock the card to its confirmation state and tell
            # the user, instead of routing through the __agent__: handler
            # (which is skipped by commands.py when arg == ctx.agent and
            # would leave the picker live).
            await _lock_agent_picker(
              channel, reply.message_id, agent=agent, model=model, ok=True)
            await _send_response(
              channel, chat_id, f"Already on agent **{agent}**.", db)
            continue
          # Synthesise an internal ``/agent <name>`` so the existing /agent
          # dispatch handles the real switch (SDK rebuild, default-model
          # reset, context-note response). is_internal=True bypasses
          # sender/mention filters since the click already came from an
          # authorized operator. Stash the picker card id so the __agent__
          # handler can lock the card once the switch lands.
          _pending_agent_picker_msg_id = reply.message_id
          synthetic = dataclasses.replace(
            reply,
            event_type="im.message.receive_v1",
            msg_type="text",
            text=f"/agent {agent_name}",
            sender_id=reply.operator_id or operator_open_id,
            is_internal=True,
            action_value={},
            action_tag="",
          )
          channel.push_back(synthetic)
          continue
        if action_str.startswith("session_recall:"):
          # A `/session recall` picker row's Recall button was clicked. The
          # button value carries the ``session_recall:<uuid>`` discriminator.
          # Recall is read-only (it summarises a local transcript and injects
          # a turn) so — like the `/session recall <uuid>` text command — it
          # is NOT privilege-gated.
          session_uuid = action_str.split(":", 1)[1].strip()
          if session_uuid:
            # Lock the picker (collapse the session list to a confirmation),
            # then run the recall — the uuid came from our own card so it
            # resolves.
            await _lock_session_picker(
              channel, reply.message_id, uuid=session_uuid)
            err = await _handle_session_recall(
              channel, chat_id, project_dir, session_uuid, coding_agent, db)
            if err:
              await _send_response(channel, chat_id, err, db)
          continue
        if action_str.startswith("fork_stop:"):
          # Fork-scoped Stop button — interrupt that fork's in-flight turn
          # only (the main turn / other forks are untouched). Spawned so it
          # never blocks the loop.
          from .guests import is_authorized_sender
          if operator_open_id and not is_authorized_sender(
              reply.operator_id, operator_open_id, _member_roles):
            log.info("Ignoring unauthorized fork stop by %s", reply.operator_id)
            continue
          _spawn_fork(fork_mgr.interrupt(action_str.split(":", 1)[1]))
          continue
        log.info("Ignoring card action outside turn: %s", reply.action_value)
        continue

      # Skip recall events at top level (handled during turns)
      if reply.event_type == "im.message.recalled_v1":
        continue

      if not _is_user_message_event(reply):
        log.info("Ignoring unsupported event outside turn: %s", reply.event_type)
        continue

      # Scope to this session's chat (WebSocket receives all chats)
      if reply.chat_id and reply.chat_id != chat_id:
        log.debug("Skipping: wrong chat %s (expected %s)", reply.chat_id, chat_id)
        continue

      # Ignore sticker messages
      if getattr(reply, "msg_type", "") == "sticker":
        continue

      # Filter
      sender = reply.sender_id
      if bot_open_id and sender == bot_open_id:
        log.debug("Skipping: own message from bot %s", sender)
        continue  # Skip own messages
      # Internal messages (e.g. /session recall injection) bypass the
      # human-auth checks — Nemo synthesised them on behalf of an
      # already-authorized command invocation.
      if not reply.is_internal:
        from .guests import is_authorized_sender
        if operator_open_id and not is_authorized_sender(
            sender, operator_open_id, _member_roles):
          log.info("Skipping: unauthorized sender %s (operator=%s)", sender, operator_open_id)
          continue

      # /fork sub-thread routing — a message inside a live fork's thread goes
      # to that fork (concurrent with the main conversation), never the main
      # session. Checked before need_mention since being in the thread is
      # intent enough; after auth so only authorized members drive a fork.
      if not reply.is_internal and _route_fork_message(reply):
        continue

      # need_mention mode: only respond to @mentions and interactions
      # directed at nemo's own messages (text reply or emoji reaction).
      # Slash commands must also be @-directed. Replying to *other*
      # people's messages (quoting a teammate's card while @-ing
      # another teammate) is not considered bot-directed.
      if need_mention and bot_open_id and not reply.is_internal:
        kept = messages.filter_bot_interactions(
          [reply], bot_open_id, is_own_message=_is_own_message)
        if not kept:
          log.info("Skipping: need_mention on, not bot-directed "
                   "(event=%s parent=%s mentions=%s)",
                   reply.event_type,
                   reply.parent_id[:16] if reply.parent_id else "",
                   [m.get("id","")[:10] for m in reply.mentions])
          continue

      text = reply.text.strip()
      if not text:
        log.debug("Skipping: empty text")
        continue

      # Strip @-mention markers
      user_message = messages.strip_mentions(text, [reply], bot_open_id=bot_open_id)
      if not user_message:
        log.debug("Skipping: empty after stripping mentions")
        continue

      # Acknowledge receipt with THINKING reaction + persist to
      # messages table — but only for real user messages. Nemo-
      # synthesised internal messages (e.g. /session recall) have a
      # fabricated message_id that Lark would reject if we tried to
      # react to it, and their text is a system prompt to the agent,
      # not user-authored content that belongs in chat history.
      ack_msg_id = "" if reply.is_internal else reply.message_id
      ack_reaction_id = ""
      if not reply.is_internal:
        ack_reaction_id = await channel.add_reaction(ack_msg_id, "THINKING")
        db.record_received(
          chat_id=chat_id, text=text,
          source_message_id=reply.message_id,
          message_time=reply.create_time,
        )

      async def _clear_ack():
        nonlocal ack_reaction_id
        if ack_reaction_id and ack_msg_id:
          await channel.remove_reaction(ack_msg_id, ack_reaction_id)
          ack_reaction_id = ""

      # Command dispatch — strip the parent-quote tail first, otherwise
      # slash commands in topic chats (where every message carries a
      # root_id) never match because the enriched text looks like
      # "/mention\n\n(The user is replying…)".
      handled, response = commands.try_dispatch(
        messages.strip_parent_quote(user_message), ctx)
      if handled:
        if response == "__clear__":
          # Snapshot the active session id BEFORE clearing so /undo-clear
          # can restore it. The DB still has the old id at this point —
          # it gets overwritten on the next turn's first event.
          _prev_sdk_session_id = _sdk_session_id
          t = datetime.datetime.now().strftime("%H:%M")
          card = cards.build_card("🔄 Session Cleared",
                                  body=f"Context reset at {t}.", color="orange")
          msg_id = await channel.send_card(chat_id, card)
          db.record_sent(msg_id, text="Session Cleared", chat_id=chat_id)
          _register_msg(msg_id, chat_id)
          log.info("Session cleared card sent: %s (saved prev=%s)",
                   msg_id,
                   _prev_sdk_session_id[:8] if _prev_sdk_session_id else "none")
          ctx.clear_context_usage()
          await _restart_client()
          _pending_pacing_hint = False
        elif response == "__undo_clear__":
          if not _prev_sdk_session_id:
            await _send_response(
              channel, chat_id,
              "Nothing to undo — no `/clear` recorded since this daemon started.",
              db,
            )
          else:
            restored = _prev_sdk_session_id
            _sdk_session_id = restored
            db.set_sdk_session_id(chat_id, restored, agent, _endpoint_key)
            log.info("Restoring SDK session %s after /undo-clear", restored[:8])
            await _restart_client(resume=restored)
            await _send_response(
              channel, chat_id,
              f"↩️ Restored session `{restored[:8]}`. "
              f"The next message continues that conversation.",
              db,
            )
            # Single-shot: clear so a second /undo-clear errors out.
            _prev_sdk_session_id = ""
        elif response == "__esc__" or (response and response.startswith("__esc__:")):
          # `/esc <text>`: between turns there's nothing to cancel, but
          # the user expects `<text>` to run next — push it back as a
          # fresh message so the loop picks it up on the next iteration.
          if response.startswith("__esc__:"):
            follow_up = response.split(":", 1)[1]
            follow_msg = dataclasses.replace(reply, text=follow_up)
            channel.push_back(follow_msg)
        elif response == "__model_picker__":
          # `/model` (no args) — send the interactive picker card. The
          # actual model switch happens later when the user submits the
          # form (routed through the model_switch:<name> card action
          # handler at the top of this loop).
          await _send_model_picker(
            channel, chat_id, project_dir, ctx, db,
          )
        elif response == "__agent_picker__":
          # `/agent` (no args) — same shape as the model picker, routed
          # through agent_switch:<name> on submit.
          await _send_agent_picker(channel, chat_id, ctx, db)
        elif response == "__session_picker__":
          # `/session recall` (no uuid) — one Recall button per past
          # session; the recall fires when a button is clicked
          # (session_recall:<uuid> card action handled at the top of this loop).
          await _send_session_picker(
            channel, chat_id, project_dir, db, _sdk_session_id)
        elif response and response.startswith("__model__:"):
          new_model = response.split(":", 1)[1]
          # Resolve against the preset registry first. A preset name
          # expands to (endpoint, remote model id) — switching to one
          # also flips the SDK env vars on the next reconnect, so the
          # daemon can move between e.g. claude-opus-4-7 (real
          # Anthropic) and deepseek-v4-pro (DeepSeek's Anthropic
          # endpoint) on the same chat without restart.
          from .presets import resolve_preset
          preset = resolve_preset(new_model)
          if preset is not None:
            if not preset.supports(agent):
              await _send_response(
                channel, chat_id,
                f"Model **{new_model}** has no endpoint for "
                f"agent **{agent}**.",
                db,
              )
              await _lock_model_picker(
                channel, _pending_picker_msg_id,
                agent=agent, model=model, project_dir=project_dir,
              ok=False, attempted=new_model,
                reason=f"Preset **{new_model}** has no endpoint for "
                       f"agent **{agent}**.")
              _pending_picker_msg_id = ""
              await _clear_ack()
              continue
            if preset.api_key_env and not os.environ.get(preset.api_key_env):
              await _send_response(
                channel, chat_id,
                f"Preset **{new_model}** requires `${preset.api_key_env}` "
                f"in the daemon's environment.",
                db,
              )
              await _lock_model_picker(
                channel, _pending_picker_msg_id,
                agent=agent, model=model, project_dir=project_dir,
              ok=False, attempted=new_model,
                reason=f"Preset **{new_model}** needs "
                       f"`${preset.api_key_env}` in the daemon env.")
              _pending_picker_msg_id = ""
              await _clear_ack()
              continue
            old_endpoint_key = _endpoint_key
            new_endpoint = preset.endpoint_for(agent)
            coding_agent.set_endpoint(new_endpoint)
            switched_to = preset.remote_for(agent)
            # Each upstream endpoint (preset vs default) keeps its own
            # SDK session so we never replay one vendor's signed
            # ``thinking`` blocks against another vendor's API. Key by
            # the upstream URL — same URL = same signing authority =
            # safe to share a session across e.g. opus↔sonnet (both at
            # api.anthropic.com) or two DeepSeek model variants.
            _endpoint_key = new_endpoint.base_url
            _sdk_session_id = db.get_sdk_session_id(
              chat_id, agent, _endpoint_key)
            log.info("Model switch to preset %s → %s (endpoint=%s resume=%s)",
                     preset.name, switched_to, _endpoint_key,
                     _sdk_session_id[:8] if _sdk_session_id else "none")
            model = switched_to
            ctx.model = model
            await _restart_client(resume=_sdk_session_id)
            await channel.update_status(model, "idle", agent)
            note = _endpoint_change_note(
              old_endpoint_key, _endpoint_key, _sdk_session_id)
            await _send_response(
              channel, chat_id,
              f"Model switched to preset **{preset.name}** "
              f"(remote: `{switched_to}`).{note}",
              db,
            )
            await _lock_model_picker(
              channel, _pending_picker_msg_id,
              agent=agent, model=model, project_dir=project_dir, ok=True)
            _pending_picker_msg_id = ""
            continue
          if not is_model_compatible(agent, new_model):
            await _send_response(
              channel,
              chat_id,
              f"Model **{new_model}** is not supported by agent **{agent}**.",
              db,
            )
            await _lock_model_picker(
              channel, _pending_picker_msg_id,
              agent=agent, model=model, project_dir=project_dir,
              ok=False, attempted=new_model,
              reason=f"`{new_model}` isn't available for agent **{agent}**.")
            _pending_picker_msg_id = ""
            await _clear_ack()
            continue
          # Plain model swap. Clear any preset endpoint that was active
          # so we go back to the agent's default auth path. The
          # default endpoint has its own SDK session — fetch it instead
          # of replaying the preset endpoint's transcript whose
          # ``thinking`` blocks would 400 against real Anthropic.
          old_endpoint_key = _endpoint_key
          coding_agent.set_endpoint(EndpointConfig())
          _endpoint_key = ""
          _sdk_session_id = db.get_sdk_session_id(
            chat_id, agent, _endpoint_key)
          model = new_model
          ctx.model = model
          log.info("Model switch to %s (endpoint=default resume=%s)",
                   model, _sdk_session_id[:8] if _sdk_session_id else "none")
          await _restart_client(resume=_sdk_session_id)
          await channel.update_status(model, "idle", agent)
          note = _endpoint_change_note(
            old_endpoint_key, _endpoint_key, _sdk_session_id)
          await _send_response(
            channel, chat_id,
            f"Model switched to **{model}**.{note}", db,
          )
          await _lock_model_picker(
            channel, _pending_picker_msg_id,
            agent=agent, model=model, project_dir=project_dir, ok=True)
          _pending_picker_msg_id = ""
        elif response and response.startswith("__agent__:"):
          # Format: "__agent__:<name>:<default_model>"
          _, new_agent, default_model = response.split(":", 2)
          # Stop the old adapter cleanly so its subprocesses / threads
          # release before we build the replacement.
          try:
            await coding_agent.stop()
          except Exception as exc:
            log.warning("Stopping %s agent on /agent switch: %s",
                        agent, exc)
          # Reset state to the new agent's defaults. Endpoint goes
          # back to empty (no preset assumed) and model is the new
          # agent's default — the user can /model afterwards if they
          # want a non-default. Each agent's resume id lives in its
          # own DB column (per-agent session storage), so switching
          # back later resumes that agent's last thread.
          agent = new_agent  # type: ignore[assignment]
          model = default_model
          endpoint = EndpointConfig()
          _endpoint_key = ""
          ctx.agent = agent
          ctx.model = model
          _sdk_session_id = db.get_sdk_session_id(chat_id, agent, _endpoint_key)
          coding_agent = build_coding_agent(
            agent, credentials, chat_id, db, channel,
            permission_mode=permission_mode,
            system_prompt=system_prompt,
            endpoint=endpoint,
          )
          if ctx.effort:
            coding_agent.set_effort(ctx.effort)
          log.info("Agent switch to %s (model=%s, resume=%s)",
                   agent, model,
                   _sdk_session_id[:8] if _sdk_session_id else "none")
          try:
            await coding_agent.start(project_dir, model, resume=_sdk_session_id)
          except Exception as exc:
            log.error("Failed to start %s agent: %s", agent, exc, exc_info=True)
            await _send_response(
              channel, chat_id,
              f"Agent switch to **{agent}** failed: `{exc}`. "
              f"Daemon is in a broken state — restart it.",
              db,
            )
            await _lock_agent_picker(
              channel, _pending_agent_picker_msg_id,
              agent=agent, model=model, ok=False, attempted=new_agent,
              reason=f"Starting **{new_agent}** failed: {exc}")
            _pending_agent_picker_msg_id = ""
            await _clear_ack()
            continue
          await channel.update_status(model, "idle", agent)
          # Tell the user explicitly what happened to their context.
          # Each agent keeps its own session id (per-agent DB columns),
          # so the new agent either resumes its OWN prior history on
          # this chat or starts fresh — it never sees the previous
          # agent's transcript. Without this note users hit "the bot
          # forgot what we just talked about" surprises.
          if _sdk_session_id:
            context_note = (
              f"Resuming **{agent}**'s prior conversation on this "
              f"chat (session `{_sdk_session_id[:8]}`). It does not "
              f"see what other agents said here."
            )
          else:
            context_note = (
              f"Fresh **{agent}** conversation — no prior history "
              f"on this chat. Other agents' transcripts are kept "
              f"separately and reachable by switching back."
            )
          await _send_response(
            channel, chat_id,
            f"Switched to agent **{agent}** "
            f"(default model `{model}`). {context_note} "
            f"Use `/model <name>` to pick a different one.",
            db,
          )
          await _lock_agent_picker(
            channel, _pending_agent_picker_msg_id,
            agent=agent, model=model, ok=True)
          _pending_agent_picker_msg_id = ""
        elif response and response.startswith("__effort__:"):
          new_effort = response.split(":", 1)[1]
          ctx.effort = new_effort
          coding_agent.set_effort(new_effort)
          label = new_effort if new_effort else "default"
          detail_map = commands._EFFORT_DETAIL.get(agent, {})
          detail = detail_map.get(new_effort, "")
          hint = f" — {detail}" if detail else ""
          log.info("Reasoning effort set to %s — restarting client (resume=%s)",
                   label, _sdk_session_id[:8] if _sdk_session_id else "none")
          # Effort lives on SDK options for native-effort backends (Claude),
          # so we must rebuild options + reconnect for the change to apply.
          # Codex/OpenCode reset is essentially free.
          await _restart_client(resume=_sdk_session_id)
          await _send_response(channel, chat_id, f"Reasoning effort: **{label}**{hint}.", db)
        elif response and response.startswith("__cd__:"):
          new_dir = response.split(":", 1)[1]
          project_dir = new_dir
          ctx.project_dir = project_dir
          shell_manager.set_project_dir(project_dir)
          await _restart_client()
          await channel.update_workspace_tag(project_dir)
          await _send_response(channel, chat_id, f"Working directory: **{project_dir}**", db)
        elif response and response.startswith("__forward__:"):
          # Forward a CLI-native slash command (/compact, /usage) to the agent's
          # underlying CLI and post its rendered result. Only claude-cli
          # implements forward_native_command; others returned a static reply
          # above and never reach this branch.
          native_cmd = response.split(":", 1)[1]
          try:
            result = await coding_agent.forward_native_command(native_cmd)
          except Exception as exc:
            log.warning("forward_native_command(%s) raised: %s", native_cmd, exc)
            result = ""
          await _send_response(
            channel, chat_id,
            result.strip() or f"`{native_cmd}` produced no output.", db)
        elif response == "__autoapprove_toggle__":
          sess = db.get_session(session_id) or {}
          enabled = not bool(sess.get("autoapprove"))
          db.set_autoapprove(chat_id, enabled)
          await _send_response(
            channel, chat_id,
            f"Auto-approve **{'enabled' if enabled else 'disabled'}**.", db)
        elif response and response.startswith("__autoapprove__:"):
          enabled = response.endswith(":on")
          db.set_autoapprove(chat_id, enabled)
          await _send_response(
            channel, chat_id,
            f"Auto-approve **{'enabled' if enabled else 'disabled'}**.", db)
        elif response == "__autoesc_toggle__":
          autoesc = not autoesc
          db.set_autoesc(chat_id, autoesc)
          await _send_response(
            channel, chat_id,
            f"Auto-esc **{'on' if autoesc else 'off'}** "
            f"— new messages {'will' if autoesc else 'will not'} "
            f"cancel the running turn.", db)
        elif response and response.startswith("__autoesc__:"):
          autoesc = response.endswith(":on")
          db.set_autoesc(chat_id, autoesc)
          await _send_response(
            channel, chat_id,
            f"Auto-esc **{'on' if autoesc else 'off'}** "
            f"— new messages {'will' if autoesc else 'will not'} "
            f"cancel the running turn.", db)
        elif response == "__mention_toggle__":
          need_mention = not need_mention
          _gc = gcfg.load_config(channel.token, chat_id)
          _gc["need_mention"] = need_mention
          gcfg.save_config(channel.token, chat_id, _gc)
          await _send_response(
            channel, chat_id,
            f"@mention requirement **{'on' if need_mention else 'off'}**.", db)
        elif response and response.startswith("__mention__:"):
          need_mention = response.endswith(":on")
          _gc = gcfg.load_config(channel.token, chat_id)
          _gc["need_mention"] = need_mention
          gcfg.save_config(channel.token, chat_id, _gc)
          await _send_response(
            channel, chat_id,
            f"@mention requirement **{'on' if need_mention else 'off'}**.", db)
        elif response == "__norm_list__":
          from .norms import get_norms, format_norms_prompt
          norms = get_norms(channel.token, chat_id)
          if norms:
            lines = [f"**Group Norms**\n"]
            for name, text in norms.items():
              lines.append(f"- **{name}**: {text}")
            await _send_response(channel, chat_id, "\n".join(lines), db)
          else:
            await _send_response(channel, chat_id, "No norms configured.", db)
        elif response and response.startswith("__norm_add__:"):
          from .norms import add_norm
          _, rest = response.split(":", 1)
          name, text = rest.split(":", 1)
          add_norm(channel.token, chat_id, name, text)
          await _send_response(channel, chat_id, f"Norm **{name}** added.", db)
        elif response and response.startswith("__norm_remove__:"):
          from .norms import remove_norm
          name = response.split(":", 1)[1]
          if remove_norm(channel.token, chat_id, name):
            await _send_response(channel, chat_id, f"Norm **{name}** removed.", db)
          else:
            await _send_response(channel, chat_id, f"Norm **{name}** not found.", db)
        elif response == "__guest_list__":
          from .guests import list_guests
          guests = list_guests(channel.token, chat_id)
          if guests:
            lines = ["**Guests**\n"]
            for g in guests:
              role = g.get("role", "guest")
              name = g.get("name", g.get("open_id", "?")[:16])
              lines.append(f"- **{name}** ({role})")
            await _send_response(channel, chat_id, "\n".join(lines), db)
          else:
            await _send_response(channel, chat_id, "No guests configured.", db)
        elif response and response.startswith("__guest_add_all__:"):
          from .guests import add_guest
          role = response.split(":", 1)[1]
          added: list[str] = []
          try:
            members = await channel.get_chat_members(chat_id)
            for m in members:
              mid = str(m.get("member_id", ""))
              mname = str(m.get("name", "")) or mid[:16]
              if not mid:
                continue
              if mid == operator_open_id:
                continue  # skip the owner/operator
              if mid == bot_open_id:
                continue  # skip the bot
              add_guest(channel.token, chat_id, mid, name=mname, role=role)
              added.append(mname)
            _member_roles = get_member_roles(channel.token, chat_id)
          except Exception as e:
            log.warning("Failed to batch-add guests: %s", e)
            await _send_response(channel, chat_id, f"Batch add failed: {e}", db)
          if added:
            lines = [f"Added **{len(added)}** members as **{role}**:"]
            lines.extend(f"- {n}" for n in added)
            await _send_response(channel, chat_id, "\n".join(lines), db)
          else:
            await _send_response(channel, chat_id, "No members to add.", db)
        elif response and response.startswith("__guest_add__:"):
          from .guests import add_guest
          _, rest = response.split(":", 1)
          role, name = rest.split(":", 1)
          # Resolve name to open_id by searching chat members
          open_id = ""
          try:
            members = await channel.get_chat_members(chat_id)
            for m in members:
              mname = str(m.get("name", ""))
              if mname.lower() == name.lower():
                open_id = str(m.get("member_id", ""))
                name = mname  # Use canonical name
                break
          except Exception as e:
            log.warning("Failed to get chat members for guest add: %s", e)
          if open_id:
            add_guest(channel.token, chat_id, open_id, name=name, role=role)
            _member_roles = get_member_roles(channel.token, chat_id)
            await _send_response(channel, chat_id, f"Added **{name}** as **{role}**.", db)
          else:
            await _send_response(channel, chat_id, f"Could not find **{name}** in this group.", db)
        elif response and response.startswith("__guest_remove__:"):
          from .guests import remove_guest, list_guests as _lg
          name = response.split(":", 1)[1]
          # Find open_id by name
          guests = _lg(channel.token, chat_id)
          target = next((g for g in guests if g.get("name", "").lower() == name.lower()), None)
          if target:
            remove_guest(channel.token, chat_id, target["open_id"])
            _member_roles = get_member_roles(channel.token, chat_id)
            await _send_response(channel, chat_id, f"Removed **{name}**.", db)
          else:
            await _send_response(channel, chat_id, f"Guest **{name}** not found.", db)
        elif response and response.startswith("__name__:"):
          new_name = response.split(":", 1)[1]
          try:
            from .lark import api as lark_api
            lark_api.update_chat_info(channel.token, chat_id, {"name": new_name})
            await _send_response(channel, chat_id, f"Renamed to **{new_name}**.", db)
          except Exception as e:
            await _send_response(channel, chat_id, f"Rename failed: {e}", db)
        elif response == "__diag__":
          await _handle_diag(channel, chat_id, project_dir, db)
        elif response == "__restart__":
          from .lifecycle import RestartSpec, spawn_lifecycle_helper

          spec = RestartSpec(
            chat_id=chat_id,
            project_dir=project_dir,
            agent=agent,
            model=_restart_model_arg(model, _endpoint_key, agent),
            permission_mode=permission_mode,
            effort=ctx.effort,
          )
          try:
            log_path = spawn_lifecycle_helper(spec)
          except Exception as exc:
            log.error("Failed to spawn restart helper: %s", exc, exc_info=True)
            await _send_response(
              channel, chat_id,
              f"Restart failed before handoff: `{exc}`.",
              db,
            )
            await _clear_ack()
            continue
          await _send_response(
            channel, chat_id,
            f"Restarting Nemo. Lifecycle log: `{log_path}`",
            db,
          )
          running = False
          await _clear_ack()
          break
        elif response == "__upgrade__":
          from .lifecycle import (
            RestartSpec, is_editable_install, run_pipx_upgrade,
            spawn_lifecycle_helper,
          )

          if is_editable_install():
            await _send_response(
              channel, chat_id,
              "Current install is editable. Update the source checkout, then "
              "use `/restart` to reload Nemo.",
              db,
            )
            await _clear_ack()
            continue
          await _send_response(
            channel, chat_id,
            "Running `pipx upgrade captain-nemo`...",
            db,
          )
          result = await asyncio.to_thread(run_pipx_upgrade)
          if result.returncode != 0:
            output = result.output.strip()[-1500:] or "no output"
            await _send_response(
              channel, chat_id,
              "Upgrade failed; current daemon is still running.\n\n"
              f"```\n{output}\n```",
              db,
            )
            await _clear_ack()
            continue
          spec = RestartSpec(
            chat_id=chat_id,
            project_dir=project_dir,
            agent=agent,
            model=_restart_model_arg(model, _endpoint_key, agent),
            permission_mode=permission_mode,
            effort=ctx.effort,
          )
          try:
            log_path = spawn_lifecycle_helper(spec)
          except Exception as exc:
            log.error("Upgrade succeeded but restart helper failed: %s",
                      exc, exc_info=True)
            await _send_response(
              channel, chat_id,
              "Upgrade succeeded, but restart handoff failed. "
              f"Start Nemo manually. Error: `{exc}`",
              db,
            )
            await _clear_ack()
            continue
          await _send_response(
            channel, chat_id,
            f"Upgrade succeeded. Restarting Nemo. Lifecycle log: `{log_path}`",
            db,
          )
          running = False
          await _clear_ack()
          break
        elif response == "__upgrade_check__":
          from .lifecycle import check_pypi_upgrade

          result = await asyncio.to_thread(check_pypi_upgrade)
          await _send_response(channel, chat_id, result.output, db)
          await _clear_ack()
          continue
        elif response and response.startswith("__btw__:"):
          # Fire-and-forget: a side question must never block the main
          # loop (it can take many seconds) and its failures must never
          # reach main_loop. _handle_btw self-contains all errors.
          _idle_btw = asyncio.create_task(
            _handle_btw(
              channel, chat_id, coding_agent,
              _sdk_session_id, response.split(":", 1)[1]))
          _btw_tasks.add(_idle_btw)
          _idle_btw.add_done_callback(_btw_tasks.discard)
        elif response and response.startswith("__fork__:"):
          # Fire-and-forget: opening a fork starts a separate SDK subprocess
          # (seconds) and must not block the main loop. ForkManager.open is
          # self-contained — it posts its own decline/status messages.
          _spawn_fork(fork_mgr.open(
            main_agent=coding_agent, anchor_msg_id=reply.message_id,
            parent_sid=_sdk_session_id, project_dir=project_dir,
            model=model, prompt=response.split(":", 1)[1]))
        elif response == "__fork_close__":
          # `/fork close` is meant to be sent INSIDE a fork thread (handled by
          # fork routing before dispatch). Reaching here = typed in main chat.
          await _send_response(
            channel, chat_id,
            "Send `/fork close` inside the fork's own thread to close it.", db)
        elif response == "__session_list__":
          await _handle_session_list(
            channel, chat_id, project_dir, db, _sdk_session_id)
        elif response and response.startswith("__session_info__:"):
          target = response.split(":", 1)[1]
          await _handle_session_info(
            channel, chat_id, project_dir, target, _sdk_session_id, db)
        elif response and response.startswith("__session_recall__:"):
          target = response.split(":", 1)[1]
          # Recall summarises the past session (digest sub-session, or the
          # agent reads it inline) and injects the result as a synthetic
          # user message processed on the next turn — no SDK resume (would
          # 400 across endpoints, see the per-endpoint isolation comment in
          # db.py). Sends its own progress ack; returns only error text.
          err = await _handle_session_recall(
            channel, chat_id, project_dir, target, coding_agent, db)
          if err:
            await _send_response(channel, chat_id, err, db)
        elif response and response.startswith("__session_rm__:"):
          target = response.split(":", 1)[1]
          await _handle_session_rm(channel, chat_id, project_dir, target, db)
        elif response and response.startswith("__session_purge__:"):
          target = response.split(":", 1)[1]
          await _handle_session_purge(
            channel, chat_id, project_dir, target, _sdk_session_id, db)
        elif response == "__exit__":
          end_card = cards.build_card("Nemo — Stopped", body="Agent stopped.", color="blue")
          await channel.send_card(chat_id, end_card)
          running = False
          break
        elif response == "__dissolve__":
          end_card = cards.build_card("Nemo — Dissolved", body="Agent stopped. Group will be dissolved.", color="red")
          await channel.send_card(chat_id, end_card)
          _dissolve_on_exit = True
          running = False
          break
        elif response:
          await _send_response(channel, chat_id, response, db)
        await _clear_ack()
        continue

      shell_shortcut = shell_command.parse_shell_shortcut(
        messages.strip_parent_quote(
          messages.strip_mentions_preserve_newlines(
            text, [reply], bot_open_id=bot_open_id)))
      if shell_shortcut is not None:
        response_text = await shell_manager.start(shell_shortcut)
        log.info("Shell shortcut handled: %s", response_text)
        if response_text:
          await _send_response(channel, chat_id, response_text, db)
        await _clear_ack()
        continue

      # --- Run SDK turn ---
      log.info("Processing: %s", user_message[:80])
      ctx.msg_count += 1

      # Turn state
      _turn_card_id: str | None = None
      _turn_steps: list[cards.ThinkingStep] = []
      _turn_start = time.time()
      _turn_current_tool = ""
      _turn_interrupt_phase: str | None = None
      # Latest rendered rate-limit notice for this turn. Persists on the
      # working card and is appended to the timeout error if the turn dies.
      _turn_rate_limit_notice = ""
      # Latest rendered compact-notice banner for this turn. Set when the
      # SDK fires PreCompact (CompactStartedEvent) and replaced with the
      # post-fact summary on CompactNoticeEvent. Lives outside the
      # collapsible thinking panel so a 10–60s silent compaction is
      # explained as it happens rather than buried after the user expands
      # thinking.
      _turn_compact_notice = ""
      # Live "what's happening now" banner for a silent pre-first-token
      # stretch (currently: a recall turn reading the past transcript).
      # Set up front before the SDK starts streaming and cleared on the
      # first real progress event so it doesn't linger as stale.
      _turn_status_notice = ""

      async def _update_interrupt_card(phase: str) -> None:
        nonlocal _turn_card_id, _turn_interrupt_phase
        if not _turn_card_id:
          return
        _turn_interrupt_phase = phase
        try:
          card = cards.build_turn_card(
            phase,
            steps=_turn_steps,
            current_tool=_turn_current_tool,
            elapsed=int(time.time() - _turn_start),
            rate_limit_notice=_turn_rate_limit_notice,
            compact_notice=_turn_compact_notice,
            answered_questions=list(channel.turn_ctx.answered_questions),
          )
          prev_id = _turn_card_id
          _turn_card_id = await channel.update_card(_turn_card_id, card)
          if _turn_card_id != prev_id:
            _register_msg(_turn_card_id, chat_id)
        except Exception as e:
          log.warning("Failed to update interrupt card: %s", e)

      def _await_channel(coro):
        return asyncio.run_coroutine_threadsafe(coro, main_loop_ref).result()

      # _ensure_card and _update_working are lifted out of _on_event so the
      # askq handler can call them via channel.turn_ctx.redraw to embed an
      # in-flight AskUserQuestion into the working card (no separate card).
      # Both run on the SDK thread (either via _on_event or via the
      # can_use_tool callback) and use _await_channel to marshal IO to the
      # main loop — same threading guarantee as before.

      def _ensure_card():
        """Create the working card if it doesn't exist yet.

        Bake in any current turn_ctx state (pending AskUserQuestion,
        answered history) so the very first card the user sees already
        has the question buttons. Without this the send_card → PATCH
        sequence flashes an empty grey "Working..." card for one frame
        before the question elements land.
        """
        nonlocal _turn_card_id
        if _turn_card_id:
          return
        _await_channel(_clear_ack())
        card = cards.build_turn_card(
          "working",
          chat_id=chat_id,
          answered_questions=list(channel.turn_ctx.answered_questions),
          pending_question=channel.turn_ctx.pending_question,
        )
        try:
          _turn_card_id = _await_channel(channel.send_card(chat_id, card))
          db.set_working(session_id, _turn_card_id)
          _register_msg(_turn_card_id, chat_id)
          channel.turn_ctx.turn_card_id = _turn_card_id
        except Exception as e:
          log.error("Working card error: %s", e)

      def _update_working(**kwargs):
        """Update the working card with current state."""
        nonlocal _turn_card_id
        if _turn_interrupt_phase:
          return
        if not _turn_card_id:
          # The card-create call on the first progress event failed
          # (transient Lark connection drop, rate-limit blip, etc).
          # Retry now instead of staying silent for the rest of the
          # turn — if the second attempt also fails, _ensure_card
          # logs and leaves _turn_card_id None and we exit silently
          # below, same as before.
          _ensure_card()
          if not _turn_card_id:
            return
        elapsed = int(time.time() - _turn_start)
        # Merge in the AskUserQuestion state so the working card keeps
        # showing both the picks the user already made and any in-flight
        # question buttons. Callers may still pass `current_tool` etc as
        # kwargs.
        ctx_kwargs = {
          "answered_questions": list(channel.turn_ctx.answered_questions),
          "pending_question": channel.turn_ctx.pending_question,
        }
        ctx_kwargs.update(kwargs)
        card = cards.build_turn_card(
          "working",
          steps=_turn_steps,
          elapsed=elapsed,
          chat_id=chat_id,
          status_notice=_turn_status_notice,
          rate_limit_notice=_turn_rate_limit_notice,
          compact_notice=_turn_compact_notice,
          **ctx_kwargs,
        )
        try:
          prev_id = _turn_card_id
          _turn_card_id = _await_channel(channel.update_card(_turn_card_id, card))
          if _turn_card_id != prev_id:
            _register_msg(_turn_card_id, chat_id)
            channel.turn_ctx.turn_card_id = _turn_card_id
        except Exception as e:
          log.debug("Failed to update working card: %s", e)

      # Wire the askq handler's redraw signal to _update_working. The
      # handler mutates channel.turn_ctx.pending_question / answered_questions
      # and calls redraw() after each click; _update_working ensures the
      # working card exists and PATCHes it.
      def _turn_ctx_redraw():
        try:
          _update_working()
        except Exception as e:
          log.warning("turn_ctx redraw failed: %s", e)

      channel.turn_ctx = TurnCardCtx(redraw=_turn_ctx_redraw)

      def _on_event(event):
        # Thread safety: this runs on the SDK thread. It mutates _turn_card_id,
        # _turn_steps. The main loop only reads these AFTER
        # asyncio.wait({sdk_task, ...}) completes, which guarantees all
        # _on_event calls have finished. No lock needed.
        nonlocal _turn_card_id, _sdk_session_id, _turn_current_tool
        nonlocal _turn_rate_limit_notice, _turn_compact_notice
        nonlocal _turn_status_notice

        if isinstance(event, ProgressEvent):
          _turn_steps.append(cards.ThinkingStep(event.kind, event.summary))
          _turn_current_tool = event.summary if event.kind == "tool" else _turn_current_tool
          if event.first:
            # Real output is streaming now — drop the pre-first-token
            # placeholder banner (e.g. the recall "reading transcript" hint)
            # so it doesn't linger behind the live progress.
            _turn_status_notice = ""
            _ensure_card()
          _update_working(current_tool=_turn_current_tool if event.kind == "tool" else None)

        elif isinstance(event, AnswerEvent):
          _turn_steps.append(cards.ThinkingStep("answer", event.text))
          # Don't create card for text-only responses — let them go as
          # plain text messages. Only update if card already exists.
          _update_working()

        elif isinstance(event, RateLimitNoticeEvent):
          new_notice = _format_rate_limit_notice(event)
          if new_notice == _turn_rate_limit_notice:
            return
          _turn_rate_limit_notice = new_notice
          # Surface upstream rate-limit pressure even when the turn hasn't
          # produced any visible work yet — that silence is exactly what we
          # want to explain.
          _ensure_card()
          _update_working()

        elif isinstance(event, StaleLeakNoticeEvent):
          # SDK #788: a stale background-task notification from a prior
          # turn leaked in. The turn is being transparently recovered
          # (reconnect with resume + retry the same prompt). Leave a
          # timeline breadcrumb so the brief delay is explained and the
          # user can correlate it with the daemon log if needed.
          _turn_steps.append(cards.ThinkingStep(
            "reasoning",
            "♻️ Recovered a stale background-task notification "
            f"(SDK #788, task {event.task_id}) — reconnected with resume "
            "and retried this prompt; conversation context preserved."))
          _ensure_card()
          _update_working()

        elif isinstance(event, CompactStartedEvent):
          # PreCompact hook fired — compaction may take 10-60s during which
          # no other SDK messages arrive. Surface a banner above the
          # working state so the silence is explained as it happens.
          # CompactNoticeEvent below replaces this with the post-fact
          # summary (tokens, duration) once compaction finishes.
          _turn_compact_notice = _format_compact_started(event)
          _ensure_card()
          _update_working()

        elif isinstance(event, CompactNoticeEvent):
          # SystemMessage(subtype=compact_boundary) arrived — compaction
          # finished. Swap the "compressing…" banner for a summary
          # banner with tokens + duration so the user can see at a
          # glance what happened during the silent stretch above.
          _turn_compact_notice = _format_compact_notice(event)
          ctx.record_compact(event.pre_tokens, event.post_tokens)
          _ensure_card()
          _update_working()

        elif isinstance(event, DoneEvent):
          _await_channel(_clear_ack())
          if event.session_id:
            _sdk_session_id = event.session_id
            db.set_sdk_session_id(
              chat_id, _sdk_session_id, agent, _endpoint_key)
          if _turn_interrupt_phase:
            if _turn_card_id:
              db.clear_working(session_id)
            return
          elapsed = int(time.time() - _turn_start)
          # Final response = last answer step (if any)
          answer_steps = [s for s in _turn_steps if s.kind == "answer"]
          final_text = answer_steps[-1].content if answer_steps else ""
          # Ask the active coding agent for any agent-specific note to tack
          # onto the response (e.g. Claude warns when the session jsonl is
          # getting large so the user remembers to /clear).
          if final_text:
            trailing = coding_agent.trailing_note(_sdk_session_id)
            if trailing:
              final_text = final_text + trailing
          # Thinking timeline = all non-answer steps
          thinking = [s for s in _turn_steps if s.kind != "answer"]
          if _turn_card_id:
            _turn_card_id = _update_done_card_with_fallback(
              channel=channel,
              chat_id=chat_id,
              turn_card_id=_turn_card_id,
              final_text=final_text,
              thinking=thinking,
              elapsed=elapsed,
              usage=event.usage,
              session_id=_sdk_session_id,
              compact_notice=_turn_compact_notice,
              await_channel=_await_channel,
              register_msg=_register_msg,
              answered_questions=list(channel.turn_ctx.answered_questions),
            )
            db.clear_working(session_id)
            if final_text:
              db.record_sent(_turn_card_id, text=final_text[:500], chat_id=chat_id)
          else:
            # Pure text response with no tools and no card created
            if final_text:
              _await_channel(_send_response(channel, chat_id, final_text, db))
          ctx.total_cost += event.cost
          ctx.record_context_usage(event.usage)

      # A recall turn's first SDK token can take a long time (resume +
      # reading the past transcript); until then only non-progress
      # SystemMessages arrive, so the working card would otherwise not
      # appear until first-token — ~80s in practice — leaving the user on
      # the recall ack. Send the working card up front with a blue status
      # banner so the silence is explained as it happens. The first real
      # ProgressEvent clears the banner (see _on_event) and reuses this
      # card (_ensure_card no-ops once _turn_card_id is set). Done on the
      # main loop with a direct await — unlike _ensure_card (built for the
      # SDK thread, which would deadlock on run_coroutine_threadsafe here).
      if user_message.startswith(_RECALL_PROMPT_PREFIX) and not _turn_card_id:
        _turn_status_notice = (
          "📖 Reading the recalled session transcript… the first turn can "
          "take a while before output starts.")
        try:
          await _clear_ack()
          card = cards.build_turn_card(
            "working",
            elapsed=int(time.time() - _turn_start),
            chat_id=chat_id,
            status_notice=_turn_status_notice,
            answered_questions=list(channel.turn_ctx.answered_questions),
            pending_question=channel.turn_ctx.pending_question,
          )
          _turn_card_id = await channel.send_card(chat_id, card)
          db.set_working(session_id, _turn_card_id)
          _register_msg(_turn_card_id, chat_id)
          channel.turn_ctx.turn_card_id = _turn_card_id
        except Exception as e:
          log.warning("Failed to send recall placeholder card: %s", e)

      if _pending_shell_contexts:
        log.info(
          "Injecting %d shell context(s) into next turn",
          len(_pending_shell_contexts),
        )
        shell_context = "\n\n".join(_pending_shell_contexts)
        _pending_shell_contexts.clear()
        user_message = shell_context + "\n\n[User message]\n" + user_message

      if _pending_pacing_hint:
        log.info("Prepending pacing hint after prior timeout")
        prompt_for_agent = _PACING_HINT_PREFIX + user_message
        _pending_pacing_hint = False
      else:
        prompt_for_agent = user_message
      sdk_task = asyncio.create_task(
        coding_agent.run_turn(prompt_for_agent, _on_event)
      )

      # Concurrent signal watcher: read events during SDK execution
      signal_detected = None

      _pending_msgs: list = []
      _pending_ack_msg_id: str = ""
      _pending_ack_reaction_id: str = ""

      async def _dispatch_inline(response: str | None, msg: IncomingMessage) -> None:
        """Handle an inline-safe command during an active turn."""
        nonlocal need_mention, autoesc
        try:
          # Remove THINKING reaction from the command message
          if msg.message_id:
            await channel.add_reaction(msg.message_id, "DONE")

          if response == "__autoapprove_toggle__":
            cur = db.get_session(db._session_id) or {}
            enabled = not bool(cur.get("autoapprove"))
            db.set_autoapprove(chat_id, enabled)
            await _send_response(
              channel, chat_id,
              f"Auto-approve **{'enabled' if enabled else 'disabled'}**.", db)
          elif response and response.startswith("__autoapprove__:"):
            enabled = response.endswith(":on")
            db.set_autoapprove(chat_id, enabled)
            await _send_response(
              channel, chat_id,
              f"Auto-approve **{'enabled' if enabled else 'disabled'}**.", db)
          elif response == "__autoesc_toggle__":
            autoesc = not autoesc
            db.set_autoesc(chat_id, autoesc)
            await _send_response(
              channel, chat_id,
              f"Auto-esc **{'on' if autoesc else 'off'}** "
              f"— new messages {'will' if autoesc else 'will not'} "
              f"cancel the running turn.", db)
          elif response and response.startswith("__autoesc__:"):
            autoesc = response.endswith(":on")
            db.set_autoesc(chat_id, autoesc)
            await _send_response(
              channel, chat_id,
              f"Auto-esc **{'on' if autoesc else 'off'}** "
              f"— new messages {'will' if autoesc else 'will not'} "
              f"cancel the running turn.", db)
          elif response == "__mention_toggle__":
            need_mention = not need_mention
            _gc = gcfg.load_config(channel.token, chat_id)
            _gc["need_mention"] = need_mention
            gcfg.save_config(channel.token, chat_id, _gc)
            await _send_response(
              channel, chat_id,
              f"@mention requirement **{'on' if need_mention else 'off'}**.", db)
          elif response and response.startswith("__mention__:"):
            need_mention = response.endswith(":on")
            _gc = gcfg.load_config(channel.token, chat_id)
            _gc["need_mention"] = need_mention
            gcfg.save_config(channel.token, chat_id, _gc)
            await _send_response(
              channel, chat_id,
              f"@mention requirement **{'on' if need_mention else 'off'}**.", db)
          elif response == "__norm_list__":
            from .norms import get_norms
            norms = get_norms(channel.token, chat_id)
            if norms:
              lines = ["**Group Norms**\n"]
              for name, text in norms.items():
                lines.append(f"- **{name}**: {text}")
              await _send_response(channel, chat_id, "\n".join(lines), db)
            else:
              await _send_response(channel, chat_id, "No norms configured.", db)
          elif response and response.startswith("__norm_add__:"):
            from .norms import add_norm
            _, rest = response.split(":", 1)
            name, text = rest.split(":", 1)
            add_norm(channel.token, chat_id, name, text)
            await _send_response(channel, chat_id, f"Norm **{name}** added.", db)
          elif response and response.startswith("__norm_remove__:"):
            from .norms import remove_norm
            name = response.split(":", 1)[1]
            if remove_norm(channel.token, chat_id, name):
              await _send_response(channel, chat_id, f"Norm **{name}** removed.", db)
            else:
              await _send_response(channel, chat_id, f"Norm **{name}** not found.", db)
          elif response and response.startswith("__name__:"):
            new_name = response.split(":", 1)[1]
            try:
              from .lark import api as lark_api
              lark_api.update_chat_info(channel.token, chat_id, {"name": new_name})
              await _send_response(channel, chat_id, f"Renamed to **{new_name}**.", db)
            except Exception as e:
              await _send_response(channel, chat_id, f"Rename failed: {e}", db)
          elif response == "__diag__":
            await _handle_diag(channel, chat_id, project_dir, db)
          elif response == "__model_picker__":
            # Sending the picker card during an active turn is fine —
            # it only displays the dropdown; the submit click is what
            # actually switches the model and that path is queued
            # behind any in-flight turn by the top-level card.action
            # handler (it push_back's an internal /model <name>).
            await _send_model_picker(
              channel, chat_id, project_dir, ctx, db,
            )
          elif response == "__agent_picker__":
            # Same reasoning as __model_picker__: only displays the dropdown;
            # submit's actual switch is queued behind the in-flight turn.
            await _send_agent_picker(channel, chat_id, ctx, db)
          elif response == "__session_picker__":
            # Same reasoning: only displays the session list; the recall fires
            # when a Recall button is clicked (queued behind the in-flight turn
            # by the in-turn card.action handler below).
            await _send_session_picker(
              channel, chat_id, project_dir, db, _sdk_session_id)
          elif response and response.startswith("__btw__:"):
            # During a running turn, run the side question in the
            # background so the signal watcher keeps reacting to
            # esc/stop and the main turn is never blocked or touched.
            btw_q = response.split(":", 1)[1]
            btw_task = asyncio.create_task(
              _handle_btw(
                channel, chat_id, coding_agent, _sdk_session_id, btw_q))
            _btw_tasks.add(btw_task)
            btw_task.add_done_callback(_btw_tasks.discard)
          elif response and response.startswith("__fork__:"):
            # Open a fork mid-turn: spawned so the signal watcher keeps
            # reacting to esc/stop and the main turn is never touched. The
            # fork runs concurrently on its own SDK subprocess.
            _spawn_fork(fork_mgr.open(
              main_agent=coding_agent, anchor_msg_id=msg.message_id,
              parent_sid=_sdk_session_id, project_dir=project_dir,
              model=model, prompt=response.split(":", 1)[1]))
          elif response == "__fork_close__":
            await _send_response(
              channel, chat_id,
              "Send `/fork close` inside the fork's own thread to close it.",
              db)
          elif response and response.startswith("__effort__:"):
            new_effort = response.split(":", 1)[1]
            ctx.effort = new_effort
            coding_agent.set_effort(new_effort)
            label = new_effort if new_effort else "default"
            detail_map = commands._EFFORT_DETAIL.get(agent, {})
            detail = detail_map.get(new_effort, "")
            hint = f" — {detail}" if detail else ""
            await _restart_client(resume=_sdk_session_id)
            await _send_response(channel, chat_id, f"Reasoning effort: **{label}**{hint}.", db)
          elif response:
            # Text responses: /ping, /cost, /help, /usage, /guest help, /norm help
            await _send_response(channel, chat_id, response, db)
        except Exception as e:
          log.warning("Inline command error: %s", e)

      async def _ack_pending(msg: IncomingMessage) -> None:
        """Move the OneSecond reaction to the latest pending message."""
        nonlocal _pending_ack_msg_id, _pending_ack_reaction_id
        # Remove from previous message
        if _pending_ack_msg_id and _pending_ack_reaction_id:
          try:
            await channel.remove_reaction(
              _pending_ack_msg_id, _pending_ack_reaction_id)
          except Exception:
            pass
          _pending_ack_msg_id = ""
          _pending_ack_reaction_id = ""
        # Add to new message
        if msg.message_id:
          try:
            _pending_ack_reaction_id = await channel.add_reaction(
              msg.message_id, "OneSecond")
            _pending_ack_msg_id = msg.message_id
          except Exception as exc:
            log.warning("Failed to ack pending message: %s", exc)

      async def _clear_pending_ack() -> None:
        """Remove the OneSecond reaction (before THINKING replaces it)."""
        nonlocal _pending_ack_msg_id, _pending_ack_reaction_id
        if _pending_ack_msg_id and _pending_ack_reaction_id:
          try:
            await channel.remove_reaction(
              _pending_ack_msg_id, _pending_ack_reaction_id)
          except Exception:
            pass
          _pending_ack_msg_id = ""
          _pending_ack_reaction_id = ""

      async def _watch_signals():
        nonlocal signal_detected
        while not sdk_task.done():
          # If permission handler is reading the queue, yield to it
          if channel.permission_active:
            await asyncio.sleep(0.2)
            continue
          msg = await channel.receive(timeout=5)
          if msg is None:
            continue
          # Double-check: if permission became active while we waited,
          # push back the message so permission handler can read it
          if channel.permission_active:
            channel.push_back(msg)
            await asyncio.sleep(0.1)
            continue
          # Scope to this session's chat
          if msg.chat_id and msg.chat_id != chat_id:
            continue
          # Handle message recall: remove from pending queue
          if msg.event_type == "im.message.recalled_v1":
            recalled_id = msg.message_id
            before = len(_pending_msgs)
            _pending_msgs[:] = [
              m for m in _pending_msgs if m.message_id != recalled_id
            ]
            if len(_pending_msgs) < before:
              log.info("Recalled message %s removed from pending queue", recalled_id)
              # Update OneSecond ack to the new last pending message
              if _pending_msgs:
                await _ack_pending(_pending_msgs[-1])
              else:
                await _clear_pending_ack()
            continue
          # Handle Stop button card action (check authorization)
          if msg.event_type == "card.action.trigger":
            action = msg.action_value.get("action", "")
            if action == "shell_abort":
              job_id = str(msg.action_value.get("job_id", ""))
              if monitor.is_privileged(
                  msg.operator_id, operator_open_id, _member_roles):
                await shell_manager.abort(job_id)
              else:
                log.info("Ignoring unauthorized shell abort by %s", msg.operator_id)
              continue
            # Relay-originated stop signals are already authenticated by the relay.
            # Raw "__stop__" actions should still require operator authorization.
            if action == "stop":
              signal_detected = "stop"
              return
            if action == "__stop__" and monitor.is_privileged(
                msg.operator_id, operator_open_id, _member_roles):
              signal_detected = "stop"
              return
            if action.startswith("fork_stop:"):
              # Fork-scoped Stop during a main turn — interrupt that fork's
              # turn only; the main turn (this watcher) keeps running, so we do
              # NOT set signal_detected. Spawned so it never blocks the watcher.
              from .guests import is_authorized_sender
              if operator_open_id and not is_authorized_sender(
                  msg.operator_id, operator_open_id, _member_roles):
                log.info("Ignoring unauthorized in-turn fork stop by %s",
                         msg.operator_id)
                continue
              _spawn_fork(fork_mgr.interrupt(action.split(":", 1)[1]))
              continue
            if action == "model_picker_submit":
              # Empty Submit click mid-turn — tell the user to pick a
              # model first so they're not left wondering why nothing
              # happened. Same response as outside the turn.
              await _send_response(
                channel, chat_id,
                "Pick a model from the dropdown before submitting.", db,
              )
              continue
            if action.startswith("model_switch:"):
              # Picker submitted mid-turn. Model switch needs an SDK
              # restart so it can't run inline. Append the original
              # card.action.trigger event (not a synthesised /model
              # text message) so _merge_pending routes it into
              # ``other_msgs`` and it survives the post-turn requeue
              # individually — a text synthesis here would get glued
              # together with any concurrent user replies and the
              # ``/model`` prefix would be lost.
              model_name = action.split(":", 1)[1].strip()
              if not model_name:
                continue
              from .guests import is_authorized_sender
              if operator_open_id and not is_authorized_sender(
                  msg.operator_id, operator_open_id, _member_roles):
                log.info(
                  "Ignoring unauthorized in-turn model picker submit by %s",
                  msg.operator_id,
                )
                continue
              _pending_msgs.append(msg)
              if autoesc:
                signal_detected = "autoesc"
                return
              continue
            if action.startswith("session_recall:"):
              # Recall button clicked mid-turn. Defer it: append the original
              # card.action event so _merge_pending requeues it and the
              # top-level handler runs the recall after this turn (recall
              # needs no SDK restart — it just injects a fresh turn).
              if action.split(":", 1)[1].strip():
                _pending_msgs.append(msg)
              continue
            continue
          if not _is_user_message_event(msg):
            log.info("Skipping during turn: unsupported event=%s", msg.event_type)
            continue
          # /fork sub-thread routing during a main turn — route the message
          # to its fork (concurrent, separate SDK client) instead of buffering
          # it as a main-turn pending message. Mirrors the top-level loop.
          if not msg.is_internal and _route_fork_message(msg):
            continue
          # Apply @mention filter so non-bot-directed chat doesn't get
          # pulled into the in-turn pending queue (mirrors the top-level
          # loop's filter). Recall + card.action.trigger handled above.
          if need_mention and bot_open_id and _in_turn_filtered_out(
              msg, bot_open_id, _is_own_message):
            log.info("Skipping during turn: not bot-directed (event=%s)",
                     msg.event_type)
            continue
          msg_text = msg.text
          mentions = msg.mentions
          esc_follow_up = monitor.parse_esc(msg_text, mentions)
          if esc_follow_up is not None:
            if esc_follow_up:
              # `/esc <text>`: queue `<text>` as the next turn's input so
              # the requeue at the end of the esc handler picks it up.
              follow_msg = dataclasses.replace(msg, text=esc_follow_up)
              _pending_msgs.append(follow_msg)
              await _ack_pending(follow_msg)
            signal_detected = "esc"
            return
          if monitor.is_dissolve(msg_text, mentions):
            signal_detected = "dissolve"
            return
          if monitor.is_exit(msg_text, mentions):
            signal_detected = "exit"
            return
          # Inline-safe commands: execute during turn without waiting
          stripped = messages.strip_mentions(msg_text, [msg], bot_open_id=bot_open_id)
          if stripped:
            handled, response = commands.try_dispatch(
              messages.strip_parent_quote(stripped), ctx)
            if handled and commands.is_inline_safe(response):
              await _dispatch_inline(response, msg)
              continue
            elif handled:
              # Needs SDK restart — re-queue for after turn
              _pending_msgs.append(msg)
              await _ack_pending(msg)
              if autoesc:
                signal_detected = "autoesc"
                return
              continue
            shell_shortcut = shell_command.parse_shell_shortcut(
              messages.strip_parent_quote(
                messages.strip_mentions_preserve_newlines(
                  msg_text, [msg], bot_open_id=bot_open_id)))
            if shell_shortcut is not None:
              try:
                response_text = await shell_manager.start(shell_shortcut)
                if response_text:
                  await _send_response(channel, chat_id, response_text, db)
                if msg.message_id:
                  await channel.add_reaction(msg.message_id, "DONE")
              except Exception as exc:
                log.warning("Shell shortcut during turn failed: %s", exc)
                if msg.message_id:
                  await channel.add_reaction(msg.message_id, "CROSS_MARK")
              continue
          # Re-queue non-signal messages so they aren't lost
          _pending_msgs.append(msg)
          await _ack_pending(msg)
          if autoesc:
            signal_detected = "autoesc"
            return

      watcher = asyncio.create_task(_watch_signals())

      done_tasks, _ = await asyncio.wait(
        {sdk_task, watcher},
        return_when=asyncio.FIRST_COMPLETED,
      )

      if watcher in done_tasks and signal_detected:
        if signal_detected in ("esc", "stop", "autoesc"):
          if signal_detected == "autoesc":
            log.info("Auto-esc — interrupting SDK for new incoming message")
          else:
            log.info("Stop signal received — interrupting SDK")
          await _clear_ack()
          await _update_interrupt_card("stopping")
          status = await _interrupt_and_drain(coding_agent, sdk_task)
          if status == "clean":
            log.info("SDK turn interrupted cleanly")
          elif status == "aborted":
            # Turn cancelled its own reconnect loop (e.g. stop during a
            # #788 stale-leak-resume). Interrupt succeeded; daemon lives.
            log.info("SDK turn aborted by interrupt")
          await _update_interrupt_card("stopped")

        elif signal_detected in ("exit", "dissolve"):
          await _clear_ack()
          try:
            await coding_agent.interrupt()
            await asyncio.wait_for(sdk_task, timeout=10)
          except Exception:
            sdk_task.cancel()
          if signal_detected == "dissolve":
            end_card = cards.build_card("Nemo — Dissolved",
                                        body="Agent stopped. Group will be dissolved.", color="red")
            _dissolve_on_exit = True
          else:
            end_card = cards.build_card("Nemo — Stopped", body="Agent stopped.", color="blue")
          await channel.send_card(chat_id, end_card)
          running = False
          break
      else:
        # SDK finished, cancel watcher
        watcher.cancel()
        try:
          await watcher
        except asyncio.CancelledError:
          pass  # expected on watcher cancel
        # Check for errors from run_turn (timeout, rate limit, SDK errors)
        try:
          sdk_task.result()
        except TimeoutError as exc:
          msg = "Timed out — SDK stopped responding. Context preserved, send another message to continue."
          if _turn_rate_limit_notice:
            msg = f"{msg}\n\n{_turn_rate_limit_notice}"
          await _handle_turn_error(
            msg, exc, channel, chat_id, db, session_id,
            _turn_card_id, _turn_steps, _turn_start,
          )
          _pending_pacing_hint = True
          await _clear_ack()
          await _clear_pending_ack()
          _requeue_pending(_pending_msgs, channel)
          continue
        except Exception as exc:
          try:
            await coding_agent.interrupt()
          except Exception as interrupt_exc:
            log.warning("SDK cleanup after turn error failed: %s", interrupt_exc)
          await _handle_turn_error(
            str(exc), exc, channel, chat_id, db, session_id,
            _turn_card_id, _turn_steps, _turn_start,
          )
          await _clear_ack()
          await _clear_pending_ack()
          _requeue_pending(_pending_msgs, channel)
          continue

      # Re-queue any messages consumed during the turn
      await _clear_pending_ack()
      _requeue_pending(_pending_msgs, channel)

    except KeyboardInterrupt:
      running = False
    except asyncio.CancelledError:
      log.warning("Loop cancelled (CancelledError)")
      running = False
    except Exception as e:
      log.error("Loop error: %s", e)
      try:
        err_card = cards.build_card("Error", body=f"```\n{str(e)[:500]}\n```", color="red")
        msg_id = await channel.send_card(chat_id, err_card)
        db.record_sent(msg_id, text=str(e)[:500], chat_id=chat_id)
      except Exception as e2:
        log.warning("Failed to send loop error card: %s", e2)
      await asyncio.sleep(5)

  # Cleanup — all threads are daemon, so fire-and-forget is safe.
  if _heartbeat_task:
    _heartbeat_task.cancel()
    try:
      await _heartbeat_task
    except asyncio.CancelledError:
      pass  # expected on shutdown cancel
  # Stop any live forks first (frees their SDK subprocesses + scratch dirs).
  await fork_mgr.shutdown()
  # Close SDK, event stream, and Lark API calls all concurrently
  loop = asyncio.get_event_loop()
  cleanup: list = [shell_manager.close(), coding_agent.stop(), channel.stop()]
  cleanup.append(channel.release_workspace())
  cleanup.append(channel.update_status(model, "stopped", agent))
  await asyncio.gather(*cleanup, return_exceptions=True)
  db.deactivate(session_id)
  db.close()
  if _dissolve_on_exit:
    try:
      await channel.dissolve_chat()
      log.info("Dissolved group %s", chat_id)
    except Exception as e:
      log.warning("Failed to dissolve group: %s", e)
  log.info("Agent stopped.")
  return 0
