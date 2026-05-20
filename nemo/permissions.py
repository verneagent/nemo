"""Button-based permission bridge — Approve / Approve All / Deny.

When the SDK wants to run a tool that needs approval:
1. Send a card with 3 buttons (Approve, Approve All, Deny)
2. Wait for: button click (card action), text reply, or THUMBSUP reaction
3. PATCH card to show decision
4. Return Allow/Deny to the SDK

Supports:
- Card action callbacks (button clicks via relay)
- Text replies (y/n/always — backward compatible)
- THUMBSUP reaction on the permission card = approve
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Callable

from .channel import Channel
from .db import Database
from .types import JsonObject

log = logging.getLogger(__name__)

# Tools that are always auto-approved (internal operations)
AUTO_APPROVE_PATTERNS: set[str] = set()

# Reaction emoji types that mean "approve"
APPROVE_REACTIONS = {"THUMBSUP", "OK", "YES", "APPROVE", "LIKESMILEY"}


def is_auto_approve(tool_name: str, tool_input: JsonObject) -> bool:
  """Check if a tool call should be auto-approved."""
  if tool_name != "Bash":
    return False
  cmd = tool_input.get("command", "")
  if not isinstance(cmd, str):
    return False
  return any(pat in cmd for pat in AUTO_APPROVE_PATTERNS)


def format_tool(tool_name: str, tool_input: JsonObject) -> str:
  """Format tool for permission card body."""
  if tool_name == "Bash":
    desc = tool_input.get("description", "")
    cmd = tool_input.get("command", "")
    desc_s = desc if isinstance(desc, str) else ""
    cmd_s = cmd if isinstance(cmd, str) else ""
    label = desc_s or cmd_s
    if len(label) > 200:
      label = label[:197] + "..."
    return f"**Bash**: `{label}`"
  if tool_name in ("Edit", "Write", "Read"):
    fp = tool_input.get("file_path", "")
    fp_s = fp if isinstance(fp, str) else ""
    name = os.path.basename(fp_s) if fp_s else "file"
    return f"**{tool_name}**: `{name}`"
  return f"**{tool_name}**"


def _build_permission_card(
  body: str,
  chat_id: str,
  nonce: str,
) -> JsonObject:
  """Build a permission request card with Approve/Approve All/Deny buttons."""
  from .cards import build_card

  buttons = [
    ("Approve", f"perm_approve:{nonce}", "primary"),
    ("Approve All", f"perm_always:{nonce}", "default"),
    ("Deny", f"perm_deny:{nonce}", "danger"),
  ]
  return build_card(
    "Permission Request",
    body=body,
    color="yellow",
    buttons=buttons,
    chat_id=chat_id,
  )


def _classify_action(action_value: JsonObject, nonce: str) -> str | None:
  """Classify a card action event as a permission decision.

  Returns "allow", "always", "deny", or None if not a permission action.
  """
  action = action_value.get("action", "")
  if not isinstance(action, str):
    return None
  if action == f"perm_approve:{nonce}":
    return "allow"
  if action == f"perm_always:{nonce}":
    return "always"
  if action == f"perm_deny:{nonce}":
    return "deny"
  return None


def _classify_reaction(emoji_type: str) -> str | None:
  """Classify a reaction emoji as a permission decision."""
  if emoji_type.upper() in APPROVE_REACTIONS:
    return "allow"
  return None


def build_permission_handler(
  credentials: dict[str, str],
  chat_id: str,
  db: Database,
  events_source: Channel,
) -> Callable[[str, JsonObject, object], object]:
  """Build an async can_use_tool handler for the SDK.

  events_source: a Channel that returns the next operator event via receive().

  IMPORTANT: can_use_tool runs on the SDK thread's event loop (not the main
  loop). The events_source queue is bound to the main loop, so we must
  bridge calls via run_coroutine_threadsafe to avoid cross-loop hangs.
  """
  import asyncio as _asyncio

  from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

  # Capture the main loop at build time — this is called from the main loop.
  _main_loop = _asyncio.get_event_loop()

  async def _read_from_main_loop(timeout: float) -> object:
    """Read next message from events_source on the main loop."""
    return await events_source.receive(timeout=timeout)

  def _set_permission_flag(active: bool) -> None:
    events_source.permission_active = active

  def _push_back_on_main(msg: object) -> None:
    from .channel import IncomingMessage

    if isinstance(msg, IncomingMessage):
      events_source.push_back(msg)

  async def can_use_tool(
    tool_name: str,
    tool_input: JsonObject,
    _context: object,
  ) -> object:
    log.debug("can_use_tool: %s %s", tool_name,
              {k: str(v)[:80] for k, v in tool_input.items()})

    # Auto-approve internals
    if is_auto_approve(tool_name, tool_input):
      return PermissionResultAllow()

    # Check autoapprove setting (db uses check_same_thread=False, safe here)
    session = db.get_current_session()
    if session and session.get("autoapprove"):
      return PermissionResultAllow()

    # Generate nonce for this permission request
    nonce = uuid.uuid4().hex[:12]

    # Send permission card with buttons
    from .lark.auth import get_token
    from .lark.api import send_card, update_card
    from .cards import build_card

    token = get_token(credentials["app_id"], credentials["app_secret"])
    body = format_tool(tool_name, tool_input)

    card = _build_permission_card(body, chat_id, nonce)
    msg_id = send_card(token, chat_id, card)
    log.info("Permission request: %s (card=%s, nonce=%s)", tool_name, msg_id, nonce)

    # Wait for button click, text reply, or reaction
    import time as _time
    from .monitor import is_permission_reply
    decision = None
    deadline = _time.time() + 300
    _pending: list[object] = []

    # Detect if we're on the main loop (tests) vs SDK thread (production)
    try:
      _current_loop = _asyncio.get_running_loop()
    except RuntimeError:
      _current_loop = None
    _on_main_loop = _current_loop is _main_loop

    if _on_main_loop:
      _set_permission_flag(True)
    else:
      _main_loop.call_soon_threadsafe(_set_permission_flag, True)

    try:
      while decision is None:
        remaining = deadline - _time.time()
        if remaining <= 0:
          break
        timeout = min(remaining, 30)
        try:
          if _on_main_loop:
            reply = await _read_from_main_loop(timeout)
          else:
            future = _asyncio.run_coroutine_threadsafe(
              _read_from_main_loop(timeout), _main_loop)
            reply = await _asyncio.wrap_future(future)
        except Exception:
          break
        if reply is None:
          if remaining <= 30:
            break
          continue

        event_type = getattr(reply, "event_type", "")
        reply_chat = getattr(reply, "chat_id", "")

        # Scope to this session's chat
        if reply_chat and reply_chat != chat_id:
          _pending.append(reply)
          continue

        # 1. Card action callback (button click)
        if event_type == "card.action.trigger":
          action_value = getattr(reply, "action_value", {})
          decision = _classify_action(action_value, nonce)
          if decision is None:
            # Not our permission card — re-queue
            _pending.append(reply)
          continue

        # 2. Reaction on the permission card
        if event_type == "im.message.reaction.created_v1":
          reaction_target = getattr(reply, "message_id", "")
          emoji = getattr(reply, "text", "")
          if reaction_target == msg_id:
            decision = _classify_reaction(emoji)
          if decision is None:
            _pending.append(reply)
          continue

        # 3. Text reply (backward compatible)
        reply_text = getattr(reply, "text", "")
        decision = is_permission_reply(reply_text)
        if decision is None:
          # Not a permission reply — re-queue so it isn't lost
          _pending.append(reply)

    finally:
      if _on_main_loop:
        _set_permission_flag(False)
      else:
        _main_loop.call_soon_threadsafe(_set_permission_flag, False)

    # Re-queue any consumed non-permission messages
    for msg in _pending:
      if _on_main_loop:
        _push_back_on_main(msg)
      else:
        _main_loop.call_soon_threadsafe(_push_back_on_main, msg)

    if decision is None:
      decision = "deny"  # Timeout or unrecognized = deny

    log.info("Permission decision: %s for %s", decision, tool_name)

    # Update card with decision
    try:
      token = get_token(credentials["app_id"], credentials["app_secret"])
      if decision in ("allow", "always"):
        update_card(token, msg_id, build_card(
          "Approved ✓", body=body, color="green"))
        if decision == "always":
          db.set_autoapprove(chat_id, True)
      else:
        update_card(token, msg_id, build_card(
          "Denied ✗", body=body, color="red"))
    except Exception as e:
      log.warning("Failed to update permission card: %s", e)

    if decision in ("allow", "always"):
      return PermissionResultAllow()
    return PermissionResultDeny()

  return can_use_tool


# ----------------------------------------------------------------------------
# AskUserQuestion handler
# ----------------------------------------------------------------------------
#
# AskUserQuestion is a Claude built-in tool registered with shouldDefer=true,
# requiresUserInteraction=true. In headless SDK mode the CLI has no UI to
# fill in the `answers` field, so without a can_use_tool hook the tool's
# call() runs with empty answers and the model gets back an empty response —
# which is exactly the "skill silently exits without asking" symptom.
#
# This handler intercepts AskUserQuestion via can_use_tool, renders the
# questions as a Lark card with one button per option, waits for the user
# to click (or type a free-text answer), and returns
# PermissionResultAllow(updated_input={..., answers: {q: label}}) so the
# tool's call() formats a proper tool_result for the model.
#
# Same event-loop bridging as build_permission_handler. Same permission_active
# flag so card actions get routed to this handler instead of being dropped
# at the top level of agent.py.


def _parse_askq_action(action_value: JsonObject, nonce: str) -> tuple[str, int, str] | None:
  """Parse an askq action string into (kind, qidx, payload).

  kind ∈ {"option", "other", "done"}.
  Returns None if the action doesn't belong to this handler/nonce.
  """
  action = action_value.get("action", "")
  if not isinstance(action, str) or not action.startswith(f"askq:{nonce}:"):
    return None
  parts = action.split(":", 3)
  # parts: ["askq", nonce, qidx, payload]
  if len(parts) < 4:
    return None
  try:
    qidx = int(parts[2])
  except ValueError:
    return None
  payload = parts[3]
  if payload == "other":
    return ("other", qidx, "")
  if payload == "done":
    return ("done", qidx, "")
  try:
    int(payload)
  except ValueError:
    return None
  return ("option", qidx, payload)


def build_ask_user_question_handler(
  credentials: dict[str, str],
  chat_id: str,
  events_source: Channel,
  max_wait_seconds: float = 600,
) -> Callable[[str, JsonObject, object], object]:
  """Build a can_use_tool handler that ONLY handles AskUserQuestion calls.

  Should be composed with the regular permission handler in claude_agent.py.
  Other tool calls passed to this function will be allowed without prompting.

  `max_wait_seconds` is the wall-clock deadline for collecting all answers
  (default 10 minutes). Exists mainly so tests can pass a smaller value.
  """
  import asyncio as _asyncio

  from claude_agent_sdk import PermissionResultAllow

  _main_loop = _asyncio.get_event_loop()

  async def _read_from_main_loop(timeout: float) -> object:
    return await events_source.receive(timeout=timeout)

  def _set_permission_flag(active: bool) -> None:
    events_source.permission_active = active

  def _push_back_on_main(msg: object) -> None:
    from .channel import IncomingMessage

    if isinstance(msg, IncomingMessage):
      events_source.push_back(msg)

  async def can_use_ask_user_question(
    tool_name: str,
    tool_input: JsonObject,
    _context: object,
  ) -> object:
    if tool_name != "AskUserQuestion":
      return PermissionResultAllow()

    raw_questions = tool_input.get("questions", [])
    if not isinstance(raw_questions, list) or not raw_questions:
      # Malformed call — let the model see an empty answers dict so it
      # doesn't loop. Same shape as Claude Code's CLI fallback.
      return PermissionResultAllow(updated_input={
        "questions": [],
        "answers": {},
        "metadata": {"source": "nemo", "error": "no_questions"},
      })
    questions: list[JsonObject] = [q for q in raw_questions if isinstance(q, dict)]
    if not questions:
      return PermissionResultAllow(updated_input={
        "questions": [],
        "answers": {},
        "metadata": {"source": "nemo", "error": "no_questions"},
      })

    nonce = uuid.uuid4().hex[:12]

    from .lark.auth import get_token
    from .lark.api import send_card, update_card
    from .cards import build_ask_user_question_card, build_turn_card
    from .channel import AnsweredQuestion, PendingQuestion

    # answers[qidx] = str (single-select) or list[str] (multi-select)
    answers: dict[int, object] = {}
    # multi_done tracks which multiSelect questions the user has finalized
    # with the "Submit" button. A multiSelect question counts as "answered"
    # only after it lands in multi_done — otherwise toggling a single option
    # would prematurely close the loop before the user picks the rest.
    multi_done: set[int] = set()

    # Publish the in-flight question on the channel's turn context so the
    # agent's working-card builder can render it inline. Mutating this
    # object plus calling turn_ctx.redraw() is the only way we update the
    # card — there is no separate card in the embedded path.
    turn_ctx = getattr(events_source, "turn_ctx", None)
    pending = PendingQuestion(
      questions=questions,
      answers=answers,
      nonce=nonce,
      multi_done=multi_done,
    )
    if turn_ctx is not None:
      turn_ctx.pending_question = pending

    def _redraw_via_turn_ctx() -> None:
      """Repaint the working turn card after a click. Returns True if the
      working card actually exists (so callers can decide to fall back to
      a standalone card)."""
      if turn_ctx is None:
        return
      try:
        turn_ctx.redraw()
      except Exception as e:
        log.warning("turn_ctx redraw raised: %s", e)

    # Initial paint into the working card. If the working card doesn't
    # exist yet (e.g., AskUserQuestion is the very first event of the
    # turn before any progress event), redraw() will run _ensure_card via
    # the agent loop. We re-check turn_card_id afterwards and fall back
    # to a standalone card if the working card still couldn't be created.
    _redraw_via_turn_ctx()

    fallback_msg_id: str = ""
    use_standalone = turn_ctx is None or not turn_ctx.turn_card_id

    if use_standalone:
      log.info(
        "AskUserQuestion: no working card available; "
        "falling back to standalone card")
      try:
        token = get_token(credentials["app_id"], credentials["app_secret"])
        card = build_ask_user_question_card(questions, chat_id, nonce)
        fallback_msg_id = send_card(token, chat_id, card)
      except Exception as e:
        log.error("Failed to send fallback askq card: %s", e)

    log.info(
      "AskUserQuestion: %d question(s) (nonce=%s, embedded=%s, fallback=%s)",
      len(questions), nonce, not use_standalone, fallback_msg_id or "-")

    awaiting_other_for: int | None = None

    import time as _time
    deadline = _time.time() + max_wait_seconds
    _pending: list[object] = []

    try:
      _current_loop = _asyncio.get_running_loop()
    except RuntimeError:
      _current_loop = None
    _on_main_loop = _current_loop is _main_loop

    if _on_main_loop:
      _set_permission_flag(True)
    else:
      _main_loop.call_soon_threadsafe(_set_permission_flag, True)

    def _all_answered() -> bool:
      for qidx, question in enumerate(questions):
        if question.get("multiSelect"):
          # Must wait for explicit Submit click (see multi_done comment).
          if qidx not in multi_done:
            return False
        else:
          if qidx not in answers:
            return False
      return True

    def _redraw_card() -> None:
      """Repaint after a user action. Embedded path goes through the
      turn-card redraw; standalone path PATCHes its own card."""
      if not use_standalone:
        _redraw_via_turn_ctx()
        return
      if not fallback_msg_id:
        return
      try:
        new_token = get_token(credentials["app_id"], credentials["app_secret"])
        update_card(new_token, fallback_msg_id, build_ask_user_question_card(
          questions, chat_id, nonce, answers=answers))
      except Exception as e:
        log.warning("Failed to redraw askq card: %s", e)

    try:
      while not _all_answered():
        remaining = deadline - _time.time()
        if remaining <= 0:
          break
        timeout = min(remaining, 30)
        try:
          if _on_main_loop:
            reply = await _read_from_main_loop(timeout)
          else:
            future = _asyncio.run_coroutine_threadsafe(
              _read_from_main_loop(timeout), _main_loop)
            reply = await _asyncio.wrap_future(future)
        except Exception:
          break
        if reply is None:
          continue

        event_type = getattr(reply, "event_type", "")
        reply_chat = getattr(reply, "chat_id", "")

        if reply_chat and reply_chat != chat_id:
          _pending.append(reply)
          continue

        # 1. Card button click
        if event_type == "card.action.trigger":
          parsed = _parse_askq_action(getattr(reply, "action_value", {}), nonce)
          if parsed is None:
            _pending.append(reply)
            continue
          kind, qidx, payload = parsed
          if qidx >= len(questions):
            continue
          question = questions[qidx]
          raw_options = question.get("options", [])
          options: list[JsonObject] = (
            [o for o in raw_options if isinstance(o, dict)]
            if isinstance(raw_options, list) else []
          )
          multi = bool(question.get("multiSelect"))

          if kind == "option":
            try:
              oidx = int(payload)
            except ValueError:
              continue
            if oidx >= len(options):
              continue
            label = str(options[oidx].get("label", ""))
            if multi:
              cur = answers.get(qidx)
              cur_list: list[str] = list(cur) if isinstance(cur, list) else []
              if label in cur_list:
                cur_list.remove(label)
              else:
                cur_list.append(label)
              answers[qidx] = cur_list
            else:
              answers[qidx] = label
            _redraw_card()
            continue

          if kind == "done":
            # Multi-select submission — finalize whatever was toggled.
            if qidx not in answers:
              answers[qidx] = []
            multi_done.add(qidx)
            _redraw_card()
            continue

          if kind == "other":
            awaiting_other_for = qidx
            try:
              from .lark.api import send_text
              send_text(
                get_token(credentials["app_id"], credentials["app_secret"]),
                chat_id,
                f'Type your answer for "{question.get("header", "")}":',
              )
            except Exception as e:
              log.warning("Failed to send 'Other' prompt: %s", e)
            continue

        # 2. Free-text reply
        if event_type == "im.message.receive_v1":
          text = (getattr(reply, "text", "") or "").strip()
          if not text:
            _pending.append(reply)
            continue
          # Route to whichever question we're waiting on. If user typed
          # without clicking "Other", route to the first unanswered question.
          target_qidx = awaiting_other_for
          if target_qidx is None:
            for qidx, _q in enumerate(questions):
              if qidx not in answers:
                target_qidx = qidx
                break
          if target_qidx is None:
            _pending.append(reply)
            continue
          question = questions[target_qidx]
          if question.get("multiSelect"):
            answers[target_qidx] = [text]
          else:
            answers[target_qidx] = text
          awaiting_other_for = None
          _redraw_card()
          continue

        _pending.append(reply)

    finally:
      if _on_main_loop:
        _set_permission_flag(False)
      else:
        _main_loop.call_soon_threadsafe(_set_permission_flag, False)

    for msg in _pending:
      if _on_main_loop:
        _push_back_on_main(msg)
      else:
        _main_loop.call_soon_threadsafe(_push_back_on_main, msg)

    if not _all_answered():
      log.info("AskUserQuestion: timed out without all answers; using partials")
      # Even on timeout, return whatever we got — the model can decide
      # to file with defaults rather than retry forever. The standalone
      # fallback card (if any) is repainted below alongside the embedded
      # path's history flush.

    # Build the answers map keyed by question text (the format the SDK
    # tool's call() echoes back into the tool_result message).
    answers_by_text: dict[str, object] = {}
    for qidx, question in enumerate(questions):
      key = str(question.get("question", f"question_{qidx}"))
      if qidx in answers:
        answers_by_text[key] = answers[qidx]
      else:
        answers_by_text[key] = ""

    log.info("AskUserQuestion answered: %s", {k: str(v)[:60] for k, v in answers_by_text.items()})

    # Move the resolved questions out of `pending_question` into the
    # turn's `answered_questions` history so the working card keeps
    # showing them for the rest of the turn — and into the final done
    # card. Then drop the pending banner and repaint.
    if turn_ctx is not None:
      for qidx, question in enumerate(questions):
        turn_ctx.answered_questions.append(AnsweredQuestion(
          header=str(question.get("header") or question.get("question") or ""),
          question=str(question.get("question", "")),
          answer=answers.get(qidx, ""),
        ))
      turn_ctx.pending_question = None
      _redraw_via_turn_ctx()

    # Finalize standalone fallback card, if we ever sent one. Its
    # purpose ends now that the model has the answers; refresh it with
    # the final selections so the user has a record.
    if use_standalone and fallback_msg_id:
      try:
        token = get_token(credentials["app_id"], credentials["app_secret"])
        update_card(token, fallback_msg_id, build_ask_user_question_card(
          questions, chat_id, nonce, answers=answers))
      except Exception:
        pass

    metadata: JsonObject = {"source": "nemo"}
    if not _all_answered():
      # Partial answers due to timeout — signal to the model so it can
      # decide whether to proceed with defaults or retry.
      metadata["timeout"] = True
    return PermissionResultAllow(updated_input={
      "questions": questions,
      "answers": answers_by_text,
      "metadata": metadata,
    })

  return can_use_ask_user_question
