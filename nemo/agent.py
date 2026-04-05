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
from .config import load_credentials, resolve_profile
from .db import Database
from .lark import api as lark_api
from .lark import auth as lark_auth
from .lark.events import LarkEventStream
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


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def main_loop(
  chat_id: str,
  project_dir: str,
  model: str,
  profile: str | None = None,
) -> int:
  """Run the agent main loop."""
  session_id = str(uuid.uuid4())
  os.environ["HANDOFF_SESSION_ID"] = session_id
  os.environ["HANDOFF_PROJECT_DIR"] = project_dir
  os.environ["HANDOFF_SESSION_TOOL"] = "Claude Agent SDK"

  resolved_profile = resolve_profile(explicit=profile)
  credentials = load_credentials(resolved_profile)
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

  # Detect need_mention
  need_mention = False
  try:
    members = lark_api.get_chat_members(token, chat_id)
    human_count = sum(1 for m in members if m.get("member_id") != bot_open_id)
    need_mention = human_count > 1
  except Exception:
    pass

  # Activate session
  db.activate(
    session_id, chat_id, model,
    operator_open_id=operator_open_id,
    bot_open_id=bot_open_id,
    need_mention=need_mention,
    config_profile=resolved_profile,
  )
  log.info("Session %s activated for chat %s", session_id, chat_id)

  # Connect to Lark event stream
  events = LarkEventStream(credentials["app_id"], credentials["app_secret"])
  await events.connect()

  # Send start card
  start_card = cards.build_card(
    f"Nemo ({model})",
    body="Agent ready. Send a message to begin.",
    color="blue",
  )
  try:
    lark_api.send_card(token, chat_id, start_card)
  except Exception as e:
    log.warning("Start card failed: %s", e)

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
  _stale_tasks: set[str] = set()

  def handle_sig(_sig, _frame):
    nonlocal running
    running = False

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

      # Filter
      sender = reply.get("sender_id", "")
      if bot_open_id and sender == bot_open_id:
        continue  # Skip own messages
      if not monitor.is_authorized(sender, operator_open_id):
        continue

      text = reply.get("text", "").strip()
      if not text:
        continue

      # Strip @-mention markers
      user_message = messages.strip_mentions(text, [reply])
      if not user_message:
        continue

      # Acknowledge receipt
      try:
        lark_api.add_reaction(token, reply.get("message_id", ""), "EYES")
      except Exception:
        pass
      db.record_received(
        chat_id=chat_id, text=text,
        source_message_id=reply.get("message_id", ""),
        message_time=reply.get("create_time", ""),
      )

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
          await _restart_client()
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
        elif response and response.startswith("__handback__:"):
          body = "Agent stopped."
          end_card = cards.build_card("Nemo — Stopped", body=body, color="blue")
          lark_api.send_card(token, chat_id, end_card)
          running = False
          break
        elif response:
          _send_response(token, chat_id, response, db)
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

        if isinstance(event, ToolStartEvent):
          _turn_tools.append(event.tool)
          card = cards.build_turn_card(
            "working",
            current_tool=event.tool.summary,
            tools=_turn_tools,
            chat_id=chat_id,
          )
          try:
            _turn_card_id = lark_api.send_card(token, chat_id, card)
            db.set_working(session_id, _turn_card_id)
          except Exception as e:
            log.error("Working card error: %s", e)

        elif isinstance(event, ToolProgressEvent):
          _turn_tools.append(event.tool)
          if _turn_card_id:
            elapsed = int(time.time() - _turn_start)
            card = cards.build_turn_card(
              "working",
              current_tool=event.tool.summary,
              tools=_turn_tools,
              elapsed=elapsed,
              chat_id=chat_id,
            )
            try:
              lark_api.update_card(token, _turn_card_id, card)
            except Exception:
              pass

        elif isinstance(event, TextEvent):
          _turn_texts.append(event.text)
          merged = "\n\n---\n\n".join(_turn_texts)
          if _turn_card_id:
            card = cards.build_turn_card(
              "response", body=merged, tools=_turn_tools)
            try:
              lark_api.update_card(token, _turn_card_id, card)
            except Exception:
              _send_response(token, chat_id, event.text, db)
          else:
            _turn_card_id = _send_response(token, chat_id, merged, db)

        elif isinstance(event, DoneEvent):
          elapsed = int(time.time() - _turn_start)
          if _turn_card_id:
            merged = "\n\n---\n\n".join(_turn_texts) if _turn_texts else ""
            card = cards.build_turn_card(
              "done", body=merged, tools=_turn_tools,
              elapsed=elapsed, usage=event.usage,
            )
            try:
              lark_api.update_card(token, _turn_card_id, card)
            except Exception:
              pass
            db.clear_working(session_id)
          ctx.total_cost += event.cost

      # Run SDK turn as async task, watch event stream for signals
      sdk_task = asyncio.create_task(
        run_turn(client, user_message, _on_event, stale_tasks=_stale_tasks))

      # Concurrent signal watcher: read events during SDK execution
      signal_detected = None

      async def _watch_signals():
        nonlocal signal_detected
        while not sdk_task.done():
          msg = await events.next_message(timeout=5)
          if msg is None:
            continue
          msg_text = msg.get("text", "")
          mentions = msg.get("mentions")
          if monitor.is_esc(msg_text, mentions):
            signal_detected = "esc"
            return
          if monitor.is_handback(msg_text, mentions):
            signal_detected = "handback"
            return

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
            await _restart_client()
          token = lark_auth.get_token(credentials["app_id"], credentials["app_secret"])
          _send_response(token, chat_id, "Operation cancelled.", db)

        elif signal_detected == "handback":
          try:
            await client.interrupt()
            await asyncio.wait_for(sdk_task, timeout=10)
          except Exception:
            sdk_task.cancel()
          token = lark_auth.get_token(credentials["app_id"], credentials["app_secret"])
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
        cost, usage = sdk_task.result()

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
    can_use_tool=perm_handler,
    hooks={},
  )
