"""Main agent loop — event-driven message processing.

Wires together:
- Lark event stream (lark.events) — receives messages via WebSocket 长连接
- Command dispatch (commands) — built-in /clear, /model, etc.
- SDK turn execution (turn) — Claude Agent SDK
- Signal monitoring (monitor) — /esc, handback detection
- Card presentation (cards) — unified turn card
- Permission bridge (permissions) — text-based approval
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import signal
import time
import uuid

from . import cards, commands, messages, monitor
from .cards import ToolRecord
from .config import load_credentials
from .db import Database
from .lark import api as lark_api
from .lark import auth as lark_auth
from .lark.events import LarkEvent, LarkEventStream
from .permissions import build_permission_handler
from .turn import (
  DoneEvent, TextEvent, ToolProgressEvent, ToolStartEvent, run_turn,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _send_response(token: str, chat_id: str, text: str, db: Database) -> str | None:
  """Send a markdown response card. Returns message_id."""
  card = cards.build_markdown_card(text)
  try:
    msg_id = lark_api.send_card(token, chat_id, card)
    db.record_sent(msg_id, text=text[:500], chat_id=chat_id)
    return msg_id
  except Exception as e:
    log.error("Send error: %s", e)
    return None


def _handle_diag(
  token: str, chat_id: str,
  credentials: dict, project_dir: str,
  db: Database,
) -> None:
  """Run diagnostics and send results as a card."""
  results: list[str] = []

  # Check token refresh
  try:
    new_token = lark_auth.get_token(credentials["app_id"], credentials["app_secret"])
    results.append("Token refresh: OK")
  except Exception as e:
    results.append(f"Token refresh: FAIL ({e})")
    new_token = token

  # Check send/receive
  try:
    test_card = cards.build_card("Diag", body="test", color="grey")
    msg_id = lark_api.send_card(new_token, chat_id, test_card)
    if msg_id:
      results.append("Send card: OK")
      try:
        lark_api.delete_message(new_token, msg_id)
      except Exception:
        pass
    else:
      results.append("Send card: FAIL (no msg_id)")
  except Exception as e:
    results.append(f"Send card: FAIL ({e})")

  # Check workspace tag
  try:
    from .workspace import get_workspace_id
    ws_id = get_workspace_id(project_dir)
    info = lark_api.get_chat_info(new_token, chat_id)
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
    lark_api.send_card(new_token, chat_id, diag_card)
  except Exception as e:
    log.error("Failed to send diag card: %s", e)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def main_loop(
  chat_id: str,
  project_dir: str,
  model: str,
  sidecar: bool = False,
) -> int:
  """Run the agent main loop."""
  session_id = str(uuid.uuid4())
  os.environ["HANDOFF_SESSION_ID"] = session_id
  os.environ["HANDOFF_PROJECT_DIR"] = project_dir
  os.environ["HANDOFF_SESSION_TOOL"] = "Claude Agent SDK"

  credentials = load_credentials()
  if not credentials:
    log.error("No credentials configured")
    return 1

  token = lark_auth.get_token(credentials["app_id"], credentials["app_secret"])

  # Resolve operator & bot
  operator_open_id = ""
  bot_open_id = ""
  try:
    email = credentials.get("email", "")
    if email:
      operator_open_id = lark_api.lookup_open_id_by_email(token, email) or ""
    bot_info = lark_api.get_bot_info(token)
    bot_open_id = bot_info.get("open_id", "")
  except Exception as e:
    log.warning("Operator/bot lookup failed: %s", e)

  # Database
  db = Database(project_dir)

  # Clean stale sessions
  try:
    old_owner = db.get_chat_owner(chat_id)
    if old_owner:
      log.info("Cleaning stale session %s", old_owner)
      db.deactivate(old_owner)
  except Exception as e:
    log.warning("Stale cleanup error: %s", e)

  # Ensure workspace tag and claim group (skip in sidecar mode)
  if not sidecar:
    from .workspace import ensure_workspace_tag, claim_group
    ensure_workspace_tag(token, chat_id, project_dir)
    claim_group(token, chat_id)

  # Detect need_mention
  if sidecar:
    need_mention = True
  else:
    need_mention = True
    try:
      members = lark_api.get_chat_members(token, chat_id)
      human_count = sum(1 for m in members if m.get("member_id") != bot_open_id)
      need_mention = human_count > 1
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

  # Connect to Lark event stream
  events = LarkEventStream(credentials["app_id"], credentials["app_secret"])
  await events.connect()

  # Send start card
  log.info("Sending start card to %s", chat_id)
  start_card = cards.build_card(
    f"Nemo ({model})",
    body="Agent ready. Send a message to begin.",
    color="blue",
  )
  try:
    msg_id = lark_api.send_card(token, chat_id, start_card)
    log.info("Start card sent: %s", msg_id)
  except Exception as e:
    log.warning("Start card failed: %s", e)

  # Status tab — green idle
  from . import status_tab
  status_tab.update_status(token, chat_id, model, "idle")

  # Init SDK client
  from claude_agent_sdk import ClaudeSDKClient
  session_id_ref = [session_id]
  sdk_options = _build_sdk_options(
    project_dir, model, credentials, chat_id, session_id_ref, db, events)
  client = ClaudeSDKClient(options=sdk_options)
  await client.__aenter__()

  # Context
  ctx = commands.AgentContext(model, project_dir, time.time())
  running = True
  _dissolve_on_exit = False
  _stale_tasks: set[str] = set()

  def handle_sig(_sig, _frame):
    nonlocal running
    running = False
    events.push_back(LarkEvent())

  signal.signal(signal.SIGINT, handle_sig)
  signal.signal(signal.SIGTERM, handle_sig)

  async def _restart_client():
    nonlocal client, sdk_options
    try:
      await client.__aexit__(None, None, None)
    except Exception:
      pass
    sdk_options = _build_sdk_options(
      project_dir, model, credentials, chat_id, session_id_ref, db, events)
    client = ClaudeSDKClient(options=sdk_options)
    await client.__aenter__()

  # ---- Main loop: event-driven ----
  while running:
    try:
      # Wait for next message from Lark event stream
      reply = await events.next_message(timeout=300)
      if reply is None:
        continue  # Timeout, keep waiting

      log.debug("Event: type=%s chat=%s sender=%s msg_type=%s text=%r",
                reply.event_type, reply.chat_id, reply.sender_id,
                reply.msg_type, reply.text[:80] if reply.text else "")

      # Skip card action events at top level (handled during turns)
      if reply.event_type == "card.action.trigger":
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
      if not monitor.is_authorized(sender, operator_open_id):
        log.debug("Skipping: unauthorized sender %s (operator=%s)", sender, operator_open_id)
        continue

      # Sidecar mode: only respond to bot interactions
      if sidecar and bot_open_id:
        kept = messages.filter_bot_interactions([reply], bot_open_id)
        if not kept:
          continue

      text = reply.text.strip()
      if not text:
        log.debug("Skipping: empty text")
        continue

      # Strip @-mention markers
      user_message = messages.strip_mentions(text, [reply])
      if not user_message:
        log.debug("Skipping: empty after stripping mentions")
        continue

      # Acknowledge receipt with THINKING reaction
      ack_msg_id = reply.message_id
      ack_reaction_id = lark_api.add_reaction(token, ack_msg_id, "THINKING")
      db.record_received(
        chat_id=chat_id, text=text,
        source_message_id=reply.message_id,
        message_time=reply.create_time,
      )

      def _clear_ack():
        nonlocal ack_reaction_id
        if ack_reaction_id and ack_msg_id:
          lark_api.remove_reaction(token, ack_msg_id, ack_reaction_id)
          ack_reaction_id = ""

      # Command dispatch
      handled, response = commands.try_dispatch(user_message, ctx)
      if handled:
        token = lark_auth.get_token(credentials["app_id"], credentials["app_secret"])
        if response == "__clear__":
          await _restart_client()
          t = datetime.datetime.now().strftime("%H:%M")
          card = cards.build_card("🔄 Session Cleared",
                                  body=f"Context reset at {t}.", color="orange")
          lark_api.send_card(token, chat_id, card)
        elif response == "__esc__":
          _send_response(token, chat_id, "Operation cancelled.", db)
        elif response and response.startswith("__model__:"):
          new_model = response.split(":", 1)[1]
          model = new_model
          ctx.model = model
          await _restart_client()
          _send_response(token, chat_id, f"Model switched to **{model}**.", db)
        elif response and response.startswith("__cd__:"):
          new_dir = response.split(":", 1)[1]
          project_dir = new_dir
          ctx.project_dir = project_dir
          os.environ["HANDOFF_PROJECT_DIR"] = project_dir
          await _restart_client()
          _send_response(token, chat_id, f"Working directory: **{project_dir}**", db)
        elif response and response.startswith("__autoapprove__:"):
          enabled = response.endswith(":on")
          db.set_autoapprove(chat_id, enabled)
          _send_response(token, chat_id,
                         f"Auto-approve **{'enabled' if enabled else 'disabled'}**.", db)
        elif response == "__norm_list__":
          from .norms import get_norms, format_norms_prompt
          norms = get_norms(token, chat_id)
          if norms:
            lines = [f"**Group Norms**\n"]
            for name, text in norms.items():
              lines.append(f"- **{name}**: {text}")
            _send_response(token, chat_id, "\n".join(lines), db)
          else:
            _send_response(token, chat_id, "No norms configured.", db)
        elif response and response.startswith("__norm_add__:"):
          from .norms import add_norm
          _, rest = response.split(":", 1)
          name, text = rest.split(":", 1)
          add_norm(token, chat_id, name, text)
          _send_response(token, chat_id, f"Norm **{name}** added.", db)
        elif response and response.startswith("__norm_remove__:"):
          from .norms import remove_norm
          name = response.split(":", 1)[1]
          if remove_norm(token, chat_id, name):
            _send_response(token, chat_id, f"Norm **{name}** removed.", db)
          else:
            _send_response(token, chat_id, f"Norm **{name}** not found.", db)
        elif response == "__diag__":
          _handle_diag(token, chat_id, credentials, project_dir, db)
        elif response == "__exit__":
          end_card = cards.build_card("Nemo — Stopped", body="Agent stopped.", color="blue")
          lark_api.send_card(token, chat_id, end_card)
          running = False
          break
        elif response == "__dissolve__":
          end_card = cards.build_card("Nemo — Dissolved", body="Agent stopped. Group will be dissolved.", color="red")
          lark_api.send_card(token, chat_id, end_card)
          _dissolve_on_exit = True
          running = False
          break
        elif response:
          _send_response(token, chat_id, response, db)
        _clear_ack()
        continue

      # --- Run SDK turn ---
      log.info("Processing: %s", user_message[:80])
      ctx.msg_count += 1

      # Turn state
      _turn_card_id: str | None = None
      _turn_tools: list[ToolRecord] = []
      _turn_texts: list[str] = []
      _turn_start = time.time()

      def _on_event(event):
        nonlocal _turn_card_id
        token = lark_auth.get_token(credentials["app_id"], credentials["app_secret"])

        def _ensure_card():
          """Create working card if it doesn't exist yet."""
          nonlocal _turn_card_id
          if _turn_card_id:
            return
          _clear_ack()
          card = cards.build_turn_card("working", chat_id=chat_id)
          try:
            _turn_card_id = lark_api.send_card(token, chat_id, card)
            db.set_working(session_id, _turn_card_id)
          except Exception as e:
            log.error("Working card error: %s", e)

        def _update_working(**kwargs):
          """Update the working card with current state."""
          if not _turn_card_id:
            return
          elapsed = int(time.time() - _turn_start)
          latest_text = _turn_texts[-1] if _turn_texts else ""
          card = cards.build_turn_card(
            "working",
            body=latest_text,
            tools=_turn_tools,
            elapsed=elapsed,
            chat_id=chat_id,
            **kwargs,
          )
          try:
            lark_api.update_card(token, _turn_card_id, card)
          except Exception:
            pass

        if isinstance(event, (ToolStartEvent, ToolProgressEvent)):
          _turn_tools.append(event.tool)
          _ensure_card()
          _update_working(current_tool=event.tool.summary)

        elif isinstance(event, TextEvent):
          _turn_texts.append(event.text)
          _ensure_card()
          _update_working()

        elif isinstance(event, DoneEvent):
          elapsed = int(time.time() - _turn_start)
          merged = "\n\n---\n\n".join(_turn_texts) if _turn_texts else ""
          if _turn_card_id:
            card = cards.build_turn_card(
              "done", body=merged, tools=_turn_tools,
              elapsed=elapsed, usage=event.usage,
            )
            try:
              lark_api.update_card(token, _turn_card_id, card)
            except Exception:
              pass
            db.clear_working(session_id)
          else:
            # Pure text response with no tools and no card created
            if merged:
              _send_response(token, chat_id, merged, db)
          ctx.total_cost += event.cost

      # Run SDK turn as async task, watch event stream for signals
      sdk_task = asyncio.create_task(
        run_turn(client, user_message, _on_event, stale_tasks=_stale_tasks))

      # Concurrent signal watcher: read events during SDK execution
      signal_detected = None

      _pending_msgs: list = []

      async def _watch_signals():
        nonlocal signal_detected
        while not sdk_task.done():
          # If permission handler is reading the queue, yield to it
          if events.permission_active:
            await asyncio.sleep(0.2)
            continue
          msg = await events.next_message(timeout=5)
          if msg is None:
            continue
          # Double-check: if permission became active while we waited,
          # push back the message so permission handler can read it
          if events.permission_active:
            events.push_back(msg)
            await asyncio.sleep(0.1)
            continue
          # Scope to this session's chat
          if msg.chat_id and msg.chat_id != chat_id:
            continue
          # Handle Stop button card action (check authorization)
          if msg.event_type == "card.action.trigger":
            action = msg.action_value.get("action", "")
            if action == "stop" and monitor.is_authorized(
                msg.operator_id, operator_open_id):
              signal_detected = "esc"
              return
            continue
          msg_text = msg.text
          mentions = msg.mentions
          if monitor.is_esc(msg_text, mentions):
            signal_detected = "esc"
            return
          if monitor.is_dissolve(msg_text, mentions):
            signal_detected = "dissolve"
            return
          if monitor.is_exit(msg_text, mentions):
            signal_detected = "exit"
            return
          # Re-queue non-signal messages so they aren't lost
          _pending_msgs.append(msg)

      watcher = asyncio.create_task(_watch_signals())

      done_tasks, _ = await asyncio.wait(
        {sdk_task, watcher},
        return_when=asyncio.FIRST_COMPLETED,
      )

      if watcher in done_tasks and signal_detected:
        if signal_detected == "esc":
          try:
            await client.interrupt()
            await asyncio.wait_for(sdk_task, timeout=30)
          except Exception:
            sdk_task.cancel()
          token = lark_auth.get_token(credentials["app_id"], credentials["app_secret"])
          _send_response(token, chat_id, "Operation cancelled.", db)

        elif signal_detected in ("exit", "dissolve"):
          try:
            await client.interrupt()
            await asyncio.wait_for(sdk_task, timeout=10)
          except Exception:
            sdk_task.cancel()
          token = lark_auth.get_token(credentials["app_id"], credentials["app_secret"])
          if signal_detected == "dissolve":
            end_card = cards.build_card("Nemo — Dissolved",
                                        body="Agent stopped. Group will be dissolved.", color="red")
            _dissolve_on_exit = True
          else:
            end_card = cards.build_card("Nemo — Stopped", body="Agent stopped.", color="blue")
          lark_api.send_card(token, chat_id, end_card)
          running = False
          break
      else:
        # SDK finished, cancel watcher
        watcher.cancel()
        try:
          await watcher
        except asyncio.CancelledError:
          pass
        sdk_task.result()

      # Re-queue any messages consumed during the turn
      for pending in _pending_msgs:
        events.push_back(pending)

    except KeyboardInterrupt:
      running = False
    except Exception as e:
      log.error("Loop error: %s", e)
      try:
        token = lark_auth.get_token(credentials["app_id"], credentials["app_secret"])
        _send_response(token, chat_id, f"**Error:**\n```\n{str(e)[:500]}\n```", db)
      except Exception:
        pass
      await asyncio.sleep(5)

  # Cleanup
  try:
    await client.__aexit__(None, None, None)
  except Exception:
    pass
  await events.close()
  db.deactivate(session_id)
  db.close()
  if not sidecar:
    from .workspace import release_group
    release_group(token, chat_id)
    status_tab.update_status(token, chat_id, model, "stopped")
  if _dissolve_on_exit:
    try:
      token = lark_auth.get_token(credentials["app_id"], credentials["app_secret"])
      lark_api.dissolve_chat(token, chat_id)
      log.info("Dissolved group %s", chat_id)
    except Exception as e:
      log.warning("Failed to dissolve group: %s", e)
  log.info("Agent stopped.")
  return 0


# ---------------------------------------------------------------------------
# SDK options builder
# ---------------------------------------------------------------------------

def _build_sdk_options(
  project_dir: str,
  model: str,
  credentials: dict,
  chat_id: str,
  session_id_ref: list[str],
  db: Database,
  events: LarkEventStream,
):
  from claude_agent_sdk import ClaudeAgentOptions

  # System prompt: tell the SDK client it's running inside Nemo
  agent_prompt = (
    "You are running inside Nemo, a Lark-connected coding agent daemon. "
    "Users interact with you through Lark mobile app. "
    "Process one message at a time. Return your response as text — "
    "the agent process sends it to Lark for you.\n\n"
    "Keep responses concise (mobile reading). Use 2-space indentation in code blocks."
  )

  env = {
    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    "HOME": os.environ.get("HOME", ""),
    "USER": os.environ.get("USER", ""),
    "HANDOFF_SESSION_ID": session_id_ref[0],
    "HANDOFF_PROJECT_DIR": project_dir,
    "HANDOFF_SESSION_TOOL": "Claude Agent SDK",
  }
  for key in ("HANDOFF_TMP_DIR", "http_proxy", "https_proxy", "all_proxy"):
    val = os.environ.get(key)
    if val:
      env[key] = val

  perm_handler = build_permission_handler(credentials, chat_id, db, events)

  def _stderr_handler(line: str) -> None:
    log.debug("[sdk-stderr] %s", line.rstrip())

  return ClaudeAgentOptions(
    allowed_tools=["Skill", "Read", "Write", "Edit", "Bash", "Glob", "Grep"],
    setting_sources=["user", "project"],
    permission_mode="default",
    system_prompt={
      "type": "preset",
      "preset": "claude_code",
      "append": agent_prompt,
    },
    cwd=project_dir,
    model=model,
    env=env,
    stderr=_stderr_handler,
    can_use_tool=perm_handler,
    hooks={},
  )
