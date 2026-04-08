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
import datetime
import logging
import os
import signal
import time
import uuid

from . import cards, commands, messages, monitor
from .config import load_credentials
from .db import Database
from .lark import api as lark_api
from .lark import auth as lark_auth
from .lark.events import LarkEvent, LarkEventStream
from .permissions import build_permission_handler
from .turn import (
  DoneEvent, TextEvent, ToolProgressEvent, ToolStartEvent,
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


def _handle_turn_error(
  message: str,
  exc: Exception,
  credentials: dict,
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
    token = lark_auth.get_token(credentials["app_id"], credentials["app_secret"])
    if card_id:
      elapsed = int(time.time() - turn_start)
      err_card = cards.build_turn_card(
        "error", body=f"**{message}**",
        steps=steps, elapsed=elapsed,
      )
      lark_api.update_card(token, card_id, err_card)
      db.clear_working(session_id)
    else:
      err_card = cards.build_card("Error", body=f"**{message}**", color="red")
      msg_id = lark_api.send_card(token, chat_id, err_card)
      db.record_sent(msg_id, text=message[:500], chat_id=chat_id)
  except Exception:
    pass


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
  permission_mode: str = "bypassPermissions",
) -> int:
  """Run the agent main loop."""
  session_id = str(uuid.uuid4())

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

  # Ensure workspace tag and claim group
  from .workspace import ensure_workspace_tag, evict_existing, claim_group
  ensure_workspace_tag(token, chat_id, project_dir)
  evict_existing(token, chat_id)
  claim_group(token, chat_id, model=model)

  # Detect need_mention: bot-owned groups or 1-on-1 groups default to False,
  # multi-human groups default to True. Can be overridden via group config.
  from . import group_config as gcfg
  gc = gcfg.load_config(token, chat_id)
  if "need_mention" in gc:
    need_mention = bool(gc["need_mention"])
  else:
    need_mention = True
    try:
      info = lark_api.get_chat_info(token, chat_id)
      owner_id = info.get("owner_id", "")
      if owner_id and owner_id == bot_open_id:
        need_mention = False
      else:
        members = lark_api.get_chat_members(token, chat_id)
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

  # Connect to event stream (relay if configured, otherwise lark-oapi WS)
  from .config import load_relay_config
  relay_url, relay_api_key = load_relay_config()
  if relay_url:
    from .relay_events import RelayEventStream
    events = RelayEventStream(relay_url, relay_api_key, chat_id)
  else:
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

  # Periodic heartbeat (relay-based idle detection)
  _heartbeat_task: asyncio.Task | None = None
  from .config import load_relay_config
  relay_url, _ = load_relay_config()
  if relay_url:
    from . import relay as relay_client
    from .workspace import get_machine_name

    async def _heartbeat_loop():
      while True:
        await asyncio.sleep(30)
        relay_client.send_heartbeat(
          chat_id, pid=os.getpid(), model=model,
          machine=get_machine_name())

    _heartbeat_task = asyncio.create_task(_heartbeat_loop())

  # Init SDK client — runs in a dedicated thread to isolate anyio from our asyncio
  from .sdk_thread import SDKThread

  sdk_options = _build_sdk_options(
    project_dir, model, credentials, chat_id, db, events,
    permission_mode=permission_mode)

  sdk = SDKThread()
  sdk.start()
  await sdk.create_client(sdk_options)

  # Context
  ctx = commands.AgentContext(model, project_dir, time.time())
  running = True
  _dissolve_on_exit = False
  _stale_tasks: set[str] = set()
  _sdk_session_id: str = ""  # CLI session UUID for --resume on model switch

  def handle_sig(_sig, _frame):
    nonlocal running
    running = False
    events.push_back(LarkEvent())

  signal.signal(signal.SIGINT, handle_sig)
  signal.signal(signal.SIGTERM, handle_sig)

  async def _restart_client(resume: str = ""):
    nonlocal sdk_options
    sdk_options = _build_sdk_options(
      project_dir, model, credentials, chat_id, db, events,
      permission_mode=permission_mode, resume=resume)
    await sdk.reconnect(sdk_options)

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

      # need_mention mode: only respond to @mentions, replies, reactions
      if need_mention and bot_open_id:
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
          log.info("Model switch to %s (resume=%s)", model, _sdk_session_id[:8] if _sdk_session_id else "none")
          await _restart_client(resume=_sdk_session_id)
          _send_response(token, chat_id, f"Model switched to **{model}**.", db)
        elif response and response.startswith("__cd__:"):
          new_dir = response.split(":", 1)[1]
          project_dir = new_dir
          ctx.project_dir = project_dir
          await _restart_client()
          _send_response(token, chat_id, f"Working directory: **{project_dir}**", db)
        elif response == "__autoapprove_toggle__":
          sess = db.get_session(session_id) or {}
          enabled = not bool(sess.get("autoapprove"))
          db.set_autoapprove(chat_id, enabled)
          _send_response(token, chat_id,
                         f"Auto-approve **{'enabled' if enabled else 'disabled'}**.", db)
        elif response and response.startswith("__autoapprove__:"):
          enabled = response.endswith(":on")
          db.set_autoapprove(chat_id, enabled)
          _send_response(token, chat_id,
                         f"Auto-approve **{'enabled' if enabled else 'disabled'}**.", db)
        elif response == "__mention_toggle__":
          need_mention = not need_mention
          _gc = gcfg.load_config(token, chat_id)
          _gc["need_mention"] = need_mention
          gcfg.save_config(token, chat_id, _gc)
          _send_response(token, chat_id,
                         f"@mention requirement **{'on' if need_mention else 'off'}**.", db)
        elif response and response.startswith("__mention__:"):
          need_mention = response.endswith(":on")
          _gc = gcfg.load_config(token, chat_id)
          _gc["need_mention"] = need_mention
          gcfg.save_config(token, chat_id, _gc)
          _send_response(token, chat_id,
                         f"@mention requirement **{'on' if need_mention else 'off'}**.", db)
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
      _turn_steps: list[cards.ThinkingStep] = []
      _turn_start = time.time()

      def _on_event(event):
        # Thread safety: this runs on the SDK thread. It mutates _turn_card_id,
        # _turn_steps. The main loop only reads these AFTER
        # asyncio.wait({sdk_task, ...}) completes, which guarantees all
        # _on_event calls have finished. No lock needed.
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
            _register_msg(_turn_card_id, chat_id)
          except Exception as e:
            log.error("Working card error: %s", e)

        def _update_working(**kwargs):
          """Update the working card with current state."""
          if not _turn_card_id:
            return
          elapsed = int(time.time() - _turn_start)
          card = cards.build_turn_card(
            "working",
            steps=_turn_steps,
            elapsed=elapsed,
            chat_id=chat_id,
            **kwargs,
          )
          try:
            lark_api.update_card(token, _turn_card_id, card)
          except Exception:
            pass

        if isinstance(event, (ToolStartEvent, ToolProgressEvent)):
          _turn_steps.append(cards.ThinkingStep("tool", event.tool.summary))
          _ensure_card()
          _update_working(current_tool=event.tool.summary)

        elif isinstance(event, TextEvent):
          _turn_steps.append(cards.ThinkingStep("text", event.text))
          _ensure_card()
          _update_working()

        elif isinstance(event, DoneEvent):
          if event.session_id:
            _sdk_session_id = event.session_id
          elapsed = int(time.time() - _turn_start)
          # Final response = last text step (if any)
          text_steps = [s for s in _turn_steps if s.kind == "text"]
          final_text = text_steps[-1].content if text_steps else ""
          # Thinking timeline = all steps except the last text
          thinking = _turn_steps[:-1] if text_steps and _turn_steps and _turn_steps[-1].kind == "text" else list(_turn_steps)
          if _turn_card_id:
            card = cards.build_turn_card(
              "done", body=final_text, steps=thinking,
              elapsed=elapsed, usage=event.usage,
            )
            try:
              lark_api.update_card(token, _turn_card_id, card)
            except Exception:
              pass
            db.clear_working(session_id)
          else:
            # Pure text response with no tools and no card created
            if final_text:
              _send_response(token, chat_id, final_text, db)
          ctx.total_cost += event.cost

      # Run SDK turn on dedicated thread (isolates anyio from our asyncio)
      sdk_task = asyncio.create_task(
        sdk.run_turn_with_reconnect(
          user_message, _on_event,
          stale_tasks=_stale_tasks, options=sdk_options,
        )
      )

      # Concurrent signal watcher: read events during SDK execution
      signal_detected = None

      _pending_msgs: list = []

      def _dispatch_inline(response: str | None, msg: LarkEvent) -> None:
        """Handle an inline-safe command during an active turn."""
        nonlocal need_mention
        try:
          _tok = lark_auth.get_token(credentials["app_id"], credentials["app_secret"])
          # Remove THINKING reaction from the command message
          if msg.message_id:
            lark_api.add_reaction(_tok, msg.message_id, "DONE")

          if response == "__autoapprove_toggle__":
            cur = db.get_session(db._session_id) or {}
            enabled = not bool(cur.get("autoapprove"))
            db.set_autoapprove(chat_id, enabled)
            _send_response(_tok, chat_id,
                           f"Auto-approve **{'enabled' if enabled else 'disabled'}**.", db)
          elif response and response.startswith("__autoapprove__:"):
            enabled = response.endswith(":on")
            db.set_autoapprove(chat_id, enabled)
            _send_response(_tok, chat_id,
                           f"Auto-approve **{'enabled' if enabled else 'disabled'}**.", db)
          elif response == "__mention_toggle__":
            need_mention = not need_mention
            _gc = gcfg.load_config(_tok, chat_id)
            _gc["need_mention"] = need_mention
            gcfg.save_config(_tok, chat_id, _gc)
            _send_response(_tok, chat_id,
                           f"@mention requirement **{'on' if need_mention else 'off'}**.", db)
          elif response and response.startswith("__mention__:"):
            need_mention = response.endswith(":on")
            _gc = gcfg.load_config(_tok, chat_id)
            _gc["need_mention"] = need_mention
            gcfg.save_config(_tok, chat_id, _gc)
            _send_response(_tok, chat_id,
                           f"@mention requirement **{'on' if need_mention else 'off'}**.", db)
          elif response == "__norm_list__":
            from .norms import get_norms
            norms = get_norms(_tok, chat_id)
            if norms:
              lines = ["**Group Norms**\n"]
              for name, text in norms.items():
                lines.append(f"- **{name}**: {text}")
              _send_response(_tok, chat_id, "\n".join(lines), db)
            else:
              _send_response(_tok, chat_id, "No norms configured.", db)
          elif response and response.startswith("__norm_add__:"):
            from .norms import add_norm
            _, rest = response.split(":", 1)
            name, text = rest.split(":", 1)
            add_norm(_tok, chat_id, name, text)
            _send_response(_tok, chat_id, f"Norm **{name}** added.", db)
          elif response and response.startswith("__norm_remove__:"):
            from .norms import remove_norm
            name = response.split(":", 1)[1]
            if remove_norm(_tok, chat_id, name):
              _send_response(_tok, chat_id, f"Norm **{name}** removed.", db)
            else:
              _send_response(_tok, chat_id, f"Norm **{name}** not found.", db)
          elif response == "__diag__":
            _handle_diag(_tok, chat_id, credentials, project_dir, db)
          elif response:
            # Text responses: /ping, /cost, /help, /usage, /guest help, /norm help
            _send_response(_tok, chat_id, response, db)
        except Exception as e:
          log.warning("Inline command error: %s", e)

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
            if action in ("stop", "__stop__") and monitor.is_authorized(
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
          # Inline-safe commands: execute during turn without waiting
          stripped = messages.strip_mentions(msg_text, [msg])
          if stripped:
            handled, response = commands.try_dispatch(stripped, ctx)
            if handled and commands.is_inline_safe(response):
              _dispatch_inline(response, msg)
              continue
            elif handled:
              # Needs SDK restart — re-queue for after turn
              _pending_msgs.append(msg)
              continue
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
            await sdk.interrupt()
            await asyncio.wait_for(sdk_task, timeout=30)
          except Exception:
            sdk_task.cancel()
          token = lark_auth.get_token(credentials["app_id"], credentials["app_secret"])
          _send_response(token, chat_id, "Operation cancelled.", db)

        elif signal_detected in ("exit", "dissolve"):
          try:
            await sdk.interrupt()
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
        # Check for errors from run_turn (timeout, rate limit, SDK errors)
        try:
          sdk_task.result()
        except TimeoutError as exc:
          _handle_turn_error(
            "Timed out — SDK stopped responding. Context preserved, send another message to continue.",
            exc, credentials, chat_id, db, session_id,
            _turn_card_id, _turn_steps, _turn_start,
          )
          _clear_ack()
          for pending in _pending_msgs:
            events.push_back(pending)
          continue
        except Exception as exc:
          _handle_turn_error(
            str(exc), exc, credentials, chat_id, db, session_id,
            _turn_card_id, _turn_steps, _turn_start,
          )
          _clear_ack()
          for pending in _pending_msgs:
            events.push_back(pending)
          continue

      # Re-queue any messages consumed during the turn
      for pending in _pending_msgs:
        events.push_back(pending)

    except KeyboardInterrupt:
      running = False
    except Exception as e:
      log.error("Loop error: %s", e)
      try:
        token = lark_auth.get_token(credentials["app_id"], credentials["app_secret"])
        err_card = cards.build_card("Error", body=f"```\n{str(e)[:500]}\n```", color="red")
        msg_id = lark_api.send_card(token, chat_id, err_card)
        db.record_sent(msg_id, text=str(e)[:500], chat_id=chat_id)
      except Exception:
        pass
      await asyncio.sleep(5)

  # Cleanup — all threads are daemon, so fire-and-forget is safe.
  if _heartbeat_task:
    _heartbeat_task.cancel()
    try:
      await _heartbeat_task
    except asyncio.CancelledError:
      pass
  # Close SDK, event stream, and Lark API calls all concurrently
  loop = asyncio.get_event_loop()
  cleanup: list = [sdk.close_client(), events.close()]
  from .workspace import release_group
  cleanup.append(loop.run_in_executor(None, release_group, token, chat_id))
  cleanup.append(loop.run_in_executor(None, status_tab.update_status, token, chat_id, model, "stopped"))
  await asyncio.gather(*cleanup, return_exceptions=True)
  sdk.stop()
  db.deactivate(session_id)
  db.close()
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
  db: Database,
  events: LarkEventStream,
  permission_mode: str = "bypassPermissions",
  resume: str = "",
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
    # Enable CLI's built-in stream watchdog — aborts stalled API streams
    "CLAUDE_ENABLE_STREAM_WATCHDOG": "1",
    "CLAUDE_STREAM_IDLE_TIMEOUT_MS": "90000",  # 90s, CLI default
  }
  for key in ("http_proxy", "https_proxy", "all_proxy"):
    val = os.environ.get(key)
    if val:
      env[key] = val

  perm_handler = None
  if permission_mode != "bypassPermissions":
    perm_handler = build_permission_handler(credentials, chat_id, db, events)

  def _stderr_handler(line: str) -> None:
    log.info("[sdk-stderr] %s", line.rstrip())

  # Cast str to the SDK's Literal type
  from typing import cast
  from claude_agent_sdk.types import PermissionMode
  perm_mode = cast(PermissionMode, permission_mode)

  opts: dict = dict(
    allowed_tools=["Agent", "Skill", "Read", "Write", "Edit", "Bash", "Glob", "Grep"],
    setting_sources=["user", "project"],
    permission_mode=perm_mode,
    system_prompt={
      "type": "preset",
      "preset": "claude_code",
      "append": agent_prompt,
    },
    cwd=project_dir,
    model=model,
    env=env,
    stderr=_stderr_handler,
    hooks={},
  )
  if perm_handler is not None:
    opts["can_use_tool"] = perm_handler
  if resume:
    opts["resume"] = resume

  return ClaudeAgentOptions(**opts)
