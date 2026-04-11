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
from .agent_factory import AgentProvider, build_coding_agent, is_model_compatible
from .channel import IncomingMessage
from .config import load_credentials
from .db import Database
from .lark_channel import LarkChannel
from .turn import (
  DoneEvent, TextEvent, ThinkingEvent, ToolProgressEvent, ToolStartEvent,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

async def _send_response(
  channel: LarkChannel, chat_id: str, text: str, db: Database,
) -> str | None:
  """Send a markdown response card. Returns message_id."""
  card = cards.build_markdown_card(text)
  try:
    msg_id = await channel.send_card(chat_id, card)
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


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def main_loop(
  chat_id: str,
  project_dir: str,
  model: str,
  provider: AgentProvider = "claude",
  permission_mode: str = "bypassPermissions",
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

  # Clean stale sessions (preserve sdk_session_id for resume)
  _resume_sdk_id = ""
  try:
    old_owner = db.get_chat_owner(chat_id)
    if old_owner:
      _resume_sdk_id = db.get_sdk_session_id(chat_id)
      log.info("Cleaning stale session %s (sdk=%s)", old_owner,
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

  # Send start card
  log.info("Sending start card to %s", chat_id)
  from nemo import __version__
  folder = os.path.basename(project_dir) or project_dir
  start_lines = [f"📂 `{folder}`  ·  pid `{os.getpid()}`"]
  if _resume_sdk_id:
    start_lines.append(f"Session `{_resume_sdk_id[:8]}` resumed")
  start_note = project_dir
  start_card = cards.build_card(
    f"Nemo v{__version__} ({model})",
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

  # Status tab — green idle
  await channel.update_status(model, "idle")

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

  agent = build_coding_agent(
    provider,
    credentials, chat_id, db, channel,
    permission_mode=permission_mode,
  )
  # Resume previous SDK session if available
  _sdk_session_id: str = _resume_sdk_id
  if _sdk_session_id:
    log.info("Resuming SDK session %s", _sdk_session_id[:8])
  try:
    await agent.start(project_dir, model, resume=_sdk_session_id)
  except Exception as e:
    log.error("SDK startup failed: %s", e)
    err_card = cards.build_card("Error", body=f"```\n{e}\n```", color="red")
    try:
      await channel.send_card(chat_id, err_card)
    except Exception as e:
      log.warning("Failed to send SDK startup error card: %s", e)
    # Continue into the main loop — run_turn_with_reconnect will
    # attempt to reconnect when the user sends a message.

  # Context
  ctx = commands.AgentContext(model, project_dir, time.time())
  main_loop_ref = asyncio.get_running_loop()
  running = True
  _dissolve_on_exit = False
  _stale_tasks: set[str] = set()

  def handle_sig(_sig, _frame):
    nonlocal running
    running = False
    channel.push_back(IncomingMessage())

  signal.signal(signal.SIGINT, handle_sig)
  signal.signal(signal.SIGTERM, handle_sig)

  async def _restart_client(resume: str = ""):
    await agent.reset(project_dir, model, resume=resume)

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
        log.info("Ignoring card action outside turn: %s", reply.action_value)
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
      if not monitor.is_authorized(sender, operator_open_id, _member_roles):
        log.debug("Skipping: unauthorized sender %s (operator=%s)", sender, operator_open_id)
        continue

      # need_mention mode: only respond to @mentions, replies to nemo's
      # own messages, and reactions. Slash commands always pass through
      # — otherwise users can't send /mention to toggle the requirement
      # off (catch-22). Replying to *other* people's messages (e.g.
      # quoting a teammate's card while @-ing another teammate) is not
      # considered bot-directed.
      if need_mention and bot_open_id:
        stripped = reply.text.strip() if reply.text else ""
        if not stripped.startswith("/"):
          def _is_own_parent(mid: str) -> bool:
            row = db.lookup_parent_message(mid)
            return bool(row and row.get("direction") == "sent")
          kept = messages.filter_bot_interactions(
            [reply], bot_open_id, is_own_parent=_is_own_parent)
          if not kept:
            log.info("Skipping: need_mention on, not bot-directed "
                     "(parent=%s mentions=%s)",
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

      # Acknowledge receipt with THINKING reaction
      ack_msg_id = reply.message_id
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

      # Command dispatch
      handled, response = commands.try_dispatch(user_message, ctx)
      if handled:
        if response == "__clear__":
          t = datetime.datetime.now().strftime("%H:%M")
          card = cards.build_card("🔄 Session Cleared",
                                  body=f"Context reset at {t}.", color="orange")
          msg_id = await channel.send_card(chat_id, card)
          db.record_sent(msg_id, text="Session Cleared", chat_id=chat_id)
          _register_msg(msg_id, chat_id)
          log.info("Session cleared card sent: %s", msg_id)
          await _restart_client()
        elif response == "__esc__":
          await _send_response(channel, chat_id, "Operation cancelled.", db)
        elif response and response.startswith("__model__:"):
          new_model = response.split(":", 1)[1]
          if not is_model_compatible(provider, new_model):
            await _send_response(
              channel,
              chat_id,
              f"Model **{new_model}** is not supported by provider **{provider}**.",
              db,
            )
            await _clear_ack()
            continue
          model = new_model
          ctx.model = model
          log.info("Model switch to %s (resume=%s)", model, _sdk_session_id[:8] if _sdk_session_id else "none")
          await _restart_client(resume=_sdk_session_id)
          await _send_response(channel, chat_id, f"Model switched to **{model}**.", db)
        elif response and response.startswith("__cd__:"):
          new_dir = response.split(":", 1)[1]
          project_dir = new_dir
          ctx.project_dir = project_dir
          await _restart_client()
          await channel.update_workspace_tag(project_dir)
          await _send_response(channel, chat_id, f"Working directory: **{project_dir}**", db)
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

      # --- Run SDK turn ---
      log.info("Processing: %s", user_message[:80])
      ctx.msg_count += 1

      # Turn state
      _turn_card_id: str | None = None
      _turn_steps: list[cards.ThinkingStep] = []
      _turn_start = time.time()
      _turn_current_tool = ""
      _turn_interrupt_phase: str | None = None

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
          )
          await channel.update_card(_turn_card_id, card)
        except Exception as e:
          log.warning("Failed to update interrupt card: %s", e)

      def _await_channel(coro):
        return asyncio.run_coroutine_threadsafe(coro, main_loop_ref).result()

      def _on_event(event):
        # Thread safety: this runs on the SDK thread. It mutates _turn_card_id,
        # _turn_steps. The main loop only reads these AFTER
        # asyncio.wait({sdk_task, ...}) completes, which guarantees all
        # _on_event calls have finished. No lock needed.
        nonlocal _turn_card_id, _sdk_session_id, _turn_current_tool

        def _ensure_card():
          """Create working card if it doesn't exist yet."""
          nonlocal _turn_card_id
          if _turn_card_id:
            return
          _await_channel(_clear_ack())
          card = cards.build_turn_card("working", chat_id=chat_id)
          try:
            _turn_card_id = _await_channel(channel.send_card(chat_id, card))
            db.set_working(session_id, _turn_card_id)
            _register_msg(_turn_card_id, chat_id)
          except Exception as e:
            log.error("Working card error: %s", e)

        def _update_working(**kwargs):
          """Update the working card with current state."""
          if not _turn_card_id or _turn_interrupt_phase:
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
            _await_channel(channel.update_card(_turn_card_id, card))
          except Exception as e:
            log.debug("Failed to update working card: %s", e)

        if isinstance(event, (ToolStartEvent, ToolProgressEvent)):
          _turn_steps.append(cards.ThinkingStep("tool", event.tool.summary))
          _turn_current_tool = event.tool.summary
          _ensure_card()
          _update_working(current_tool=event.tool.summary)

        elif isinstance(event, ThinkingEvent):
          _turn_steps.append(cards.ThinkingStep("thinking", event.text))
          _update_working()

        elif isinstance(event, TextEvent):
          _turn_steps.append(cards.ThinkingStep("text", event.text))
          # Don't create card for text-only responses — let them go as
          # plain text messages. Only update if card already exists.
          _update_working()

        elif isinstance(event, DoneEvent):
          _await_channel(_clear_ack())
          if event.session_id:
            _sdk_session_id = event.session_id
            db.set_sdk_session_id(chat_id, _sdk_session_id)
          if _turn_interrupt_phase:
            if _turn_card_id:
              db.clear_working(session_id)
            return
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
              session_id=_sdk_session_id,
            )
            _card_update_failed = False
            try:
              _await_channel(channel.update_card(_turn_card_id, card))
            except Exception as e:
              log.warning("Failed to update done card: %s", e)
              _card_update_failed = True
            # Fallback: if card update failed (typically table limit), send
            # the full response as a markdown file and retry with preview.
            if _card_update_failed and final_text:
              try:
                import tempfile
                file_dir = os.path.join("/tmp/nemo", "nemo-files")
                os.makedirs(file_dir, exist_ok=True)
                fd, overflow_path = tempfile.mkstemp(
                  suffix=".md", prefix="nemo-response-", dir=file_dir)
                with os.fdopen(fd, "w") as f:
                  f.write(final_text)
                # Send as file
                from .lark import api as lark_api
                file_key = lark_api.upload_file(channel.token, overflow_path)
                lark_api.send_file(channel.token, chat_id, file_key)
                log.info("Sent overflow response as file: %s", overflow_path)
                # Retry card with a short preview
                preview = final_text[:500].rsplit("\n", 1)[0]
                preview += f"\n\n_…full response ({len(final_text)} chars) sent as file_"
                fallback_card = cards.build_turn_card(
                  "done", body=preview, steps=thinking,
                  elapsed=elapsed, usage=event.usage,
                  session_id=_sdk_session_id,
                )
                _await_channel(channel.update_card(_turn_card_id, fallback_card))
              except Exception as e:
                log.warning("Failed to send overflow fallback: %s", e)
            db.clear_working(session_id)
            if final_text:
              db.record_sent(_turn_card_id, text=final_text[:500], chat_id=chat_id)
          else:
            # Pure text response with no tools and no card created
            if final_text:
              _await_channel(_send_response(channel, chat_id, final_text, db))
          ctx.total_cost += event.cost

      sdk_task = asyncio.create_task(
        agent.run_turn(user_message, _on_event, stale_tasks=_stale_tasks)
      )

      # Concurrent signal watcher: read events during SDK execution
      signal_detected = None

      _pending_msgs: list = []

      async def _dispatch_inline(response: str | None, msg: IncomingMessage) -> None:
        """Handle an inline-safe command during an active turn."""
        nonlocal need_mention
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
          elif response == "__diag__":
            await _handle_diag(channel, chat_id, project_dir, db)
          elif response:
            # Text responses: /ping, /cost, /help, /usage, /guest help, /norm help
            await _send_response(channel, chat_id, response, db)
        except Exception as e:
          log.warning("Inline command error: %s", e)

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
          # Handle Stop button card action (check authorization)
          if msg.event_type == "card.action.trigger":
            action = msg.action_value.get("action", "")
            # Relay-originated stop signals are already authenticated by the relay.
            # Raw "__stop__" actions should still require operator authorization.
            if action == "stop":
              signal_detected = "stop"
              return
            if action == "__stop__" and monitor.is_authorized(
                msg.operator_id, operator_open_id, _member_roles):
              signal_detected = "stop"
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
          stripped = messages.strip_mentions(msg_text, [msg], bot_open_id=bot_open_id)
          if stripped:
            handled, response = commands.try_dispatch(stripped, ctx)
            if handled and commands.is_inline_safe(response):
              await _dispatch_inline(response, msg)
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
        if signal_detected in ("esc", "stop"):
          log.info("Stop signal received — interrupting SDK")
          await _clear_ack()
          await _update_interrupt_card("stopping")
          try:
            await agent.interrupt()
            await asyncio.wait_for(sdk_task, timeout=10)
            log.info("SDK turn interrupted cleanly")
          except Exception as exc:
            log.warning("SDK interrupt failed (%s), cancelling task", exc)
            sdk_task.cancel()
          await _send_response(channel, chat_id, "Operation cancelled.", db)
          await _update_interrupt_card("stopped")

        elif signal_detected in ("exit", "dissolve"):
          await _clear_ack()
          try:
            await agent.interrupt()
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
          await _handle_turn_error(
            "Timed out — SDK stopped responding. Context preserved, send another message to continue.",
            exc, channel, chat_id, db, session_id,
            _turn_card_id, _turn_steps, _turn_start,
          )
          await _clear_ack()
          for pending in _pending_msgs:
            channel.push_back(pending)
          continue
        except Exception as exc:
          await _handle_turn_error(
            str(exc), exc, channel, chat_id, db, session_id,
            _turn_card_id, _turn_steps, _turn_start,
          )
          await _clear_ack()
          for pending in _pending_msgs:
            channel.push_back(pending)
          continue

      # Re-queue any messages consumed during the turn
      for pending in _pending_msgs:
        channel.push_back(pending)

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
  # Close SDK, event stream, and Lark API calls all concurrently
  loop = asyncio.get_event_loop()
  cleanup: list = [agent.stop(), channel.stop()]
  cleanup.append(channel.release_workspace())
  cleanup.append(channel.update_status(model, "stopped"))
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
