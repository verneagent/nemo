"""Tests for nemo.agent main loop behavior."""

import asyncio
import os
import shlex
import sys
import urllib.error
from unittest import mock

import pytest

from nemo.agent import (
  _RECALL_PROMPT_PREFIX,
  _format_rate_limit_notice,
  _in_turn_filtered_out,
  _interrupt_and_drain,
  _is_user_message_event,
  _merge_pending,
  _requeue_pending,
  _restart_model_arg,
  _send_response,
  _should_send_plain_text,
  _update_done_card_with_fallback,
  main_loop,
)
from nemo.channel import IncomingMessage
from nemo.turn import (
  AnswerEvent, CompactNoticeEvent, CompactStartedEvent, DoneEvent,
  ProgressEvent, RateLimitNoticeEvent,
)


class _FakeDB:
  def __init__(self, _project_dir):
    self._session_id = "sess_1"

  def get_chat_owner(self, _chat_id):
    return None

  def activate(self, session_id, *_args, **_kwargs):
    self._session_id = session_id

  def deactivate(self, _session_id):
    pass

  def close(self):
    pass

  def record_received(self, **_kwargs):
    pass

  def record_sent(self, *_args, **_kwargs):
    pass

  def clear_working(self, _session_id):
    pass

  def set_working(self, _session_id, _card_id):
    pass

  def get_session(self, _session_id):
    return {}

  def get_sdk_session_id(self, _chat_id, _agent, _endpoint_key=""):
    return ""

  def set_sdk_session_id(self, _chat_id, _sdk_session_id, _agent, _endpoint_key=""):
    pass


class _FakeChannel:
  def __init__(self, _chat_id):
    self.token = "tok"
    self._count = 0
    self._permission_active = False

  async def start(self):
    pass

  async def stop(self):
    pass

  async def resolve_operator_and_bot(self, _email):
    return "ou_user", "ou_bot"

  async def ensure_workspace_claimed(self, _project_dir, _model):
    pass

  async def get_chat_info(self, _chat_id):
    return {"owner_id": "ou_bot"}

  async def get_chat_members(self, _chat_id):
    return []

  async def send_card(self, _chat_id, _card):
    return "om_reply"

  async def send_text(self, _chat_id, _text):
    return "om_text"

  async def update_status(self, _model, _state, _agent=""):
    pass

  async def receive(self, timeout=300):
    del timeout
    if self._count == 0:
      self._count += 1
      from nemo.channel import IncomingMessage
      return IncomingMessage(
        event_type="im.message.receive_v1",
        chat_id="oc_test",
        sender_id="ou_user",
        message_id="om_src",
        msg_type="text",
        text="hi",
        create_time="1",
      )
    if self._count == 1:
      self._count += 1
      from nemo.channel import IncomingMessage
      return IncomingMessage(
        event_type="im.message.receive_v1",
        chat_id="oc_test",
        sender_id="ou_user",
        message_id="om_exit",
        msg_type="text",
        text="/exit",
        create_time="2",
      )
    return None

  def push_back(self, _message):
    pass

  async def add_reaction(self, _message_id, _emoji_type):
    return "r_thinking"

  async def remove_reaction(self, _message_id, _reaction_id):
    pass

  async def release_workspace(self):
    pass

  @property
  def permission_active(self):
    return self._permission_active

  @permission_active.setter
  def permission_active(self, active):
    self._permission_active = active


class _QueuedChannel(_FakeChannel):
  def __init__(self, _chat_id, messages):
    super().__init__(_chat_id)
    self._messages = list(messages)

  async def receive(self, timeout=300):
    del timeout
    if self._messages:
      return self._messages.pop(0)
    return None


class _FakeAgent:
  def __init__(self, *_args, **_kwargs):
    pass

  async def start(self, _project_dir, _model, resume=""):
    del resume
    pass

  async def reset(self, _project_dir, _model, resume=""):
    del resume
    pass

  def set_effort(self, _effort):
    pass

  async def interrupt(self):
    pass

  async def stop(self):
    pass

  async def run_turn(self, _prompt, on_event):
    def _emit():
      on_event(AnswerEvent("hi! how can i help?"))
      on_event(DoneEvent(cost=0.01, usage={"input_tokens": 1}))

    await asyncio.to_thread(_emit)
    return 0.01, {"input_tokens": 1}


class _RecordingChannel(_FakeChannel):
  def __init__(self, _chat_id):
    super().__init__(_chat_id)
    self.sent_cards = []
    self.sent_texts = []

  async def send_card(self, chat_id, card):
    self.sent_cards.append((chat_id, card))
    return "om_card"

  async def send_text(self, chat_id, text):
    self.sent_texts.append((chat_id, text))
    return "om_text"


def test_text_only_turn_clears_thinking_reaction(tmp_path):
  remove_reaction = mock.AsyncMock()

  with mock.patch("nemo.agent.load_credentials", return_value={
    "app_id": "app_id",
    "app_secret": "app_secret",
    "email": "user@example.com",
  }), \
       mock.patch("nemo.agent.Database", _FakeDB), \
       mock.patch("nemo.agent.LarkChannel", _FakeChannel), \
       mock.patch("nemo.agent.build_coding_agent", return_value=_FakeAgent()), \
       mock.patch.object(_FakeChannel, "add_reaction", return_value="r_thinking") as add_reaction, \
       mock.patch.object(_FakeChannel, "remove_reaction", remove_reaction), \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    result = asyncio.run(main_loop("oc_test", str(tmp_path), "claude-opus-4-6"))

  assert result == 0
  add_reaction.assert_awaited_once_with("om_src", "THINKING")
  remove_reaction.assert_awaited_once_with("om_src", "r_thinking")


def test_idle_esc_is_silent(tmp_path):
  from nemo.channel import IncomingMessage

  channel = _QueuedChannel("oc_test", [
    IncomingMessage(
      event_type="im.message.receive_v1",
      chat_id="oc_test",
      sender_id="ou_user",
      message_id="om_esc",
      msg_type="text",
      text="/esc",
      create_time="1",
    ),
    IncomingMessage(
      event_type="im.message.receive_v1",
      chat_id="oc_test",
      sender_id="ou_user",
      message_id="om_exit",
      msg_type="text",
      text="/exit",
      create_time="2",
    ),
  ])
  send_response = mock.AsyncMock()

  with mock.patch("nemo.agent.load_credentials", return_value={
    "app_id": "app_id",
    "app_secret": "app_secret",
    "email": "user@example.com",
  }), \
       mock.patch("nemo.agent.Database", _FakeDB), \
       mock.patch("nemo.agent.LarkChannel", return_value=channel), \
       mock.patch("nemo.agent.build_coding_agent", return_value=_FakeAgent()), \
       mock.patch("nemo.agent._send_response", new=send_response), \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    result = asyncio.run(main_loop("oc_test", str(tmp_path), "claude-opus-4-6"))

  assert result == 0
  send_response.assert_not_awaited()


def test_restart_command_spawns_lifecycle_helper_and_exits(tmp_path):
  channel = _QueuedChannel("oc_test", [
    IncomingMessage(
      event_type="im.message.receive_v1",
      chat_id="oc_test",
      sender_id="ou_user",
      message_id="om_restart",
      msg_type="text",
      text="/restart",
      create_time="1",
    ),
  ])
  send_response = mock.AsyncMock()

  with mock.patch("nemo.agent.load_credentials", return_value={
    "app_id": "app_id",
    "app_secret": "app_secret",
    "email": "user@example.com",
  }), \
       mock.patch("nemo.agent.Database", _FakeDB), \
       mock.patch("nemo.agent.LarkChannel", return_value=channel), \
       mock.patch("nemo.agent.build_coding_agent", return_value=_FakeAgent()), \
       mock.patch("nemo.agent._send_response", new=send_response), \
       mock.patch("nemo.lifecycle.spawn_lifecycle_helper", return_value="/tmp/life.log") as spawn, \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    result = asyncio.run(main_loop(
      "oc_test", str(tmp_path), "claude-opus-4-6",
      agent="claude", permission_mode="acceptEdits", effort="high",
    ))

  assert result == 0
  spawn.assert_called_once()
  spec = spawn.call_args.args[0]
  assert spec.chat_id == "oc_test"
  assert spec.project_dir == str(tmp_path)
  assert spec.agent == "claude"
  assert spec.model == "claude-opus-4-6"
  assert spec.permission_mode == "acceptEdits"
  assert spec.effort == "high"
  assert "Restarting Nemo" in send_response.await_args.args[2]


def test_upgrade_editable_does_not_restart(tmp_path):
  channel = _QueuedChannel("oc_test", [
    IncomingMessage(
      event_type="im.message.receive_v1",
      chat_id="oc_test",
      sender_id="ou_user",
      message_id="om_upgrade",
      msg_type="text",
      text="/upgrade",
      create_time="1",
    ),
    IncomingMessage(
      event_type="im.message.receive_v1",
      chat_id="oc_test",
      sender_id="ou_user",
      message_id="om_exit",
      msg_type="text",
      text="/exit",
      create_time="2",
    ),
  ])
  send_response = mock.AsyncMock()

  with mock.patch("nemo.agent.load_credentials", return_value={
    "app_id": "app_id",
    "app_secret": "app_secret",
    "email": "user@example.com",
  }), \
       mock.patch("nemo.agent.Database", _FakeDB), \
       mock.patch("nemo.agent.LarkChannel", return_value=channel), \
       mock.patch("nemo.agent.build_coding_agent", return_value=_FakeAgent()), \
       mock.patch("nemo.agent._send_response", new=send_response), \
       mock.patch("nemo.lifecycle.is_editable_install", return_value=True), \
       mock.patch("nemo.lifecycle.spawn_lifecycle_helper") as spawn, \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    result = asyncio.run(main_loop("oc_test", str(tmp_path), "claude-opus-4-6"))

  assert result == 0
  spawn.assert_not_called()
  assert "editable" in send_response.await_args_list[0].args[2]


def test_upgrade_failure_keeps_daemon_running(tmp_path):
  from nemo.lifecycle import CommandResult

  channel = _QueuedChannel("oc_test", [
    IncomingMessage(
      event_type="im.message.receive_v1",
      chat_id="oc_test",
      sender_id="ou_user",
      message_id="om_upgrade",
      msg_type="text",
      text="/upgrade",
      create_time="1",
    ),
    IncomingMessage(
      event_type="im.message.receive_v1",
      chat_id="oc_test",
      sender_id="ou_user",
      message_id="om_exit",
      msg_type="text",
      text="/exit",
      create_time="2",
    ),
  ])
  send_response = mock.AsyncMock()

  with mock.patch("nemo.agent.load_credentials", return_value={
    "app_id": "app_id",
    "app_secret": "app_secret",
    "email": "user@example.com",
  }), \
       mock.patch("nemo.agent.Database", _FakeDB), \
       mock.patch("nemo.agent.LarkChannel", return_value=channel), \
       mock.patch("nemo.agent.build_coding_agent", return_value=_FakeAgent()), \
       mock.patch("nemo.agent._send_response", new=send_response), \
       mock.patch("nemo.lifecycle.is_editable_install", return_value=False), \
       mock.patch("nemo.lifecycle.run_pipx_upgrade", return_value=CommandResult(1, "bad")), \
       mock.patch("nemo.lifecycle.spawn_lifecycle_helper") as spawn, \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    result = asyncio.run(main_loop("oc_test", str(tmp_path), "claude-opus-4-6"))

  assert result == 0
  spawn.assert_not_called()
  assert any(
    "Upgrade failed" in call.args[2]
    for call in send_response.await_args_list
  )


def test_upgrade_success_spawns_restart_helper_and_exits(tmp_path):
  from nemo.lifecycle import CommandResult

  channel = _QueuedChannel("oc_test", [
    IncomingMessage(
      event_type="im.message.receive_v1",
      chat_id="oc_test",
      sender_id="ou_user",
      message_id="om_upgrade",
      msg_type="text",
      text="/upgrade",
      create_time="1",
    ),
  ])
  send_response = mock.AsyncMock()

  with mock.patch("nemo.agent.load_credentials", return_value={
    "app_id": "app_id",
    "app_secret": "app_secret",
    "email": "user@example.com",
  }), \
       mock.patch("nemo.agent.Database", _FakeDB), \
       mock.patch("nemo.agent.LarkChannel", return_value=channel), \
       mock.patch("nemo.agent.build_coding_agent", return_value=_FakeAgent()), \
       mock.patch("nemo.agent._send_response", new=send_response), \
       mock.patch("nemo.lifecycle.is_editable_install", return_value=False), \
       mock.patch("nemo.lifecycle.run_pipx_upgrade", return_value=CommandResult(0, "ok")), \
       mock.patch("nemo.lifecycle.spawn_lifecycle_helper", return_value="/tmp/life.log") as spawn, \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    result = asyncio.run(main_loop("oc_test", str(tmp_path), "claude-opus-4-6"))

  assert result == 0
  spawn.assert_called_once()
  assert any(
    "Upgrade succeeded" in call.args[2]
    for call in send_response.await_args_list
  )


def test_active_esc_updates_card_without_cancel_message(tmp_path):
  from nemo.channel import IncomingMessage

  class _InterruptChannel(_QueuedChannel):
    def __init__(self):
      super().__init__("oc_test", [
        IncomingMessage(
          event_type="im.message.receive_v1",
          chat_id="oc_test",
          sender_id="ou_user",
          message_id="om_work",
          msg_type="text",
          text="work",
          create_time="1",
        ),
        IncomingMessage(
          event_type="im.message.receive_v1",
          chat_id="oc_test",
          sender_id="ou_user",
          message_id="om_esc",
          msg_type="text",
          text="/esc",
          create_time="2",
        ),
        IncomingMessage(
          event_type="im.message.receive_v1",
          chat_id="oc_test",
          sender_id="ou_user",
          message_id="om_exit",
          msg_type="text",
          text="/exit",
          create_time="3",
        ),
      ])
      self._receive_count = 0
      self.updated_cards: list[object] = []

    async def receive(self, timeout=300):
      self._receive_count += 1
      if self._receive_count == 2:
        await asyncio.sleep(0.05)
      return await super().receive(timeout)

    async def send_card(self, _chat_id, _card):
      return f"om_card_{self._receive_count}"

    async def update_card(self, card_id, card):
      self.updated_cards.append(card)
      return card_id

  class _InterruptibleAgent(_FakeAgent):
    def __init__(self):
      self._interrupted = asyncio.Event()

    async def interrupt(self):
      self._interrupted.set()

    async def run_turn(self, _prompt, on_event):
      await asyncio.to_thread(
        lambda: on_event(ProgressEvent(kind="tool", summary="Read", first=True))
      )
      await self._interrupted.wait()
      await asyncio.to_thread(
        lambda: on_event(DoneEvent(cost=0.0, usage={"input_tokens": 1}))
      )
      return 0.0, {"input_tokens": 1}

  channel = _InterruptChannel()
  send_response = mock.AsyncMock()

  with mock.patch("nemo.agent.load_credentials", return_value={
    "app_id": "app_id",
    "app_secret": "app_secret",
    "email": "user@example.com",
  }), \
       mock.patch("nemo.agent.Database", _FakeDB), \
       mock.patch("nemo.agent.LarkChannel", return_value=channel), \
       mock.patch("nemo.agent.build_coding_agent", return_value=_InterruptibleAgent()), \
       mock.patch("nemo.agent._send_response", new=send_response), \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    result = asyncio.run(main_loop("oc_test", str(tmp_path), "claude-opus-4-6"))

  assert result == 0
  send_response.assert_not_awaited()
  assert any(
    card.get("header", {}).get("title", {}).get("content") == "Stopped"
    for card in channel.updated_cards
  )


def test_fork_command_during_turn_opens_fork(tmp_path):
  """A `/fork <prompt>` sent during a running turn is inline-dispatched to
  ForkManager.open (fire-and-forget) — without restarting the main client or
  blocking the turn. Validates the agent.py → ForkManager wiring."""
  from nemo.channel import IncomingMessage

  class _SpyForkManager:
    def __init__(self, *_a, **_k):
      self.opened = []
      self.routed = []

    def get(self, _thread_id):
      return None  # no live fork → thread routing is a no-op here

    def count(self):
      return 0

    async def open(self, **kwargs):
      self.opened.append(kwargs)

    def route(self, thread_id, prompt):
      self.routed.append((thread_id, prompt))
      return False

    async def close(self, _thread_id):
      return False

    async def shutdown(self):
      pass

  spy = _SpyForkManager()

  class _ForkCmdChannel(_QueuedChannel):
    def __init__(self):
      super().__init__("oc_test", [
        IncomingMessage(event_type="im.message.receive_v1", chat_id="oc_test",
                        sender_id="ou_user", message_id="om_work",
                        msg_type="text", text="work on it", create_time="1"),
        IncomingMessage(event_type="im.message.receive_v1", chat_id="oc_test",
                        sender_id="ou_user", message_id="om_fork",
                        msg_type="text", text="/fork investigate the bug",
                        create_time="2"),
        IncomingMessage(event_type="im.message.receive_v1", chat_id="oc_test",
                        sender_id="ou_user", message_id="om_exit",
                        msg_type="text", text="/exit", create_time="3"),
      ])
      self._n = 0

    async def receive(self, timeout=300):
      self._n += 1
      if self._n == 2:  # let the main turn establish before /fork
        await asyncio.sleep(0.05)
      return await super().receive(timeout)

    async def send_card(self, _chat_id, _card):
      return "om_card"

    async def update_card(self, card_id, _card):
      return card_id

  class _BlockingAgent(_FakeAgent):
    def __init__(self):
      self._interrupted = asyncio.Event()

    async def interrupt(self):
      self._interrupted.set()

    async def run_turn(self, _prompt, on_event):
      await asyncio.to_thread(
        lambda: on_event(ProgressEvent(kind="tool", summary="Read", first=True)))
      await self._interrupted.wait()
      await asyncio.to_thread(
        lambda: on_event(DoneEvent(cost=0.0, usage={"input_tokens": 1})))
      return 0.0, {"input_tokens": 1}

  channel = _ForkCmdChannel()

  with mock.patch("nemo.agent.load_credentials", return_value={
    "app_id": "app_id", "app_secret": "app_secret",
    "email": "user@example.com",
  }), \
       mock.patch("nemo.agent.Database", _FakeDB), \
       mock.patch("nemo.agent.LarkChannel", return_value=channel), \
       mock.patch("nemo.agent.build_coding_agent", return_value=_BlockingAgent()), \
       mock.patch("nemo.agent.ForkManager", return_value=spy), \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    result = asyncio.run(main_loop("oc_test", str(tmp_path), "claude-opus-4-6"))

  assert result == 0
  assert len(spy.opened) == 1, f"expected one fork open, got {spy.opened}"
  assert spy.opened[0]["prompt"] == "investigate the bug"
  assert spy.opened[0]["anchor_msg_id"] == "om_fork"


def test_pacing_hint_prepended_after_timeout(tmp_path):
  """After a TimeoutError the next turn's prompt is prefixed with a pacing hint;
  the hint is not applied to subsequent turns."""
  from nemo.channel import IncomingMessage

  class _TimeoutThenOkAgent(_FakeAgent):
    def __init__(self):
      self.prompts: list[str] = []
      self._calls = 0

    async def run_turn(self, prompt, on_event):
      self.prompts.append(prompt)
      self._calls += 1
      if self._calls == 1:
        raise TimeoutError("simulated SDK heartbeat timeout")
      def _emit():
        on_event(AnswerEvent("ok"))
        on_event(DoneEvent(cost=0.0, usage={}))
      await asyncio.to_thread(_emit)
      return 0.0, {}

  class _PushBackChannel(_QueuedChannel):
    """_QueuedChannel that actually re-inserts pushed-back messages so the
    timeout requeue path is observable in tests."""
    def push_back(self, message):
      self._messages.insert(0, message)

  msg_user = lambda mid, text, ts: IncomingMessage(
    event_type="im.message.receive_v1", chat_id="oc_test",
    sender_id="ou_user", message_id=mid, msg_type="text",
    text=text, create_time=ts,
  )

  agent = _TimeoutThenOkAgent()
  # /clear between turns 2 and 3 ensures /exit is consumed at top-of-loop
  # rather than caught mid-turn by the signal watcher.
  queued = _PushBackChannel("oc_test", [
    msg_user("om1", "do the big task", "1"),
    msg_user("om2", "continue", "2"),
    msg_user("om3", "and then this", "3"),
    msg_user("om_exit", "/exit", "4"),
  ])

  with mock.patch("nemo.agent.load_credentials", return_value={
    "app_id": "app_id",
    "app_secret": "app_secret",
    "email": "user@example.com",
  }), \
       mock.patch("nemo.agent.Database", _FakeDB), \
       mock.patch("nemo.agent.LarkChannel", return_value=queued), \
       mock.patch("nemo.agent.build_coding_agent", return_value=agent), \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    result = asyncio.run(main_loop("oc_test", str(tmp_path), "claude-opus-4-6"))

  assert result == 0
  # We only assert that the second turn (post-timeout) has the hint and the
  # first turn does not. Whether a third user turn lands before /exit's signal
  # depends on watcher scheduling, which is out of scope for this test.
  assert len(agent.prompts) >= 2
  assert agent.prompts[0] == "do the big task"
  assert agent.prompts[1].startswith("[Nemo 系统提示]")
  assert "continue" in agent.prompts[1]
  # Any later prompt must not carry the hint — one-shot semantics.
  for later in agent.prompts[2:]:
    assert not later.startswith("[Nemo 系统提示]"), later


def test_recall_turn_shows_placeholder_card_before_first_token(tmp_path):
  """A recall turn surfaces a working card with a blue status banner up
  front, before the SDK streams its first token, so the (often long)
  resume + transcript-read silence is explained instead of leaving the
  user on the recall ack. The banner clears once real progress streams."""
  import json

  banner_marker = "Reading the recalled session transcript"

  class _RecallChannel(_QueuedChannel):
    def __init__(self, chat_id, messages):
      super().__init__(chat_id, messages)
      self.sent_cards: list[object] = []
      self.updated_cards: list[object] = []

    async def send_card(self, _chat_id, card):
      self.sent_cards.append(card)
      return "om_card"

    async def update_card(self, card_id, card):
      self.updated_cards.append(card)
      return card_id

  class _RecallAgent(_FakeAgent):
    def __init__(self):
      self.sent_at_entry: int | None = None

    def trailing_note(self, _sdk_session_id):
      return ""

    async def run_turn(self, _prompt, on_event):
      # Snapshot what the user can already see at the instant the SDK turn
      # begins — i.e. before any progress streams. The placeholder must be
      # among these.
      self.sent_at_entry = len(channel.sent_cards)

      def _emit():
        # First real progress only after the (here-simulated) wait.
        on_event(ProgressEvent(kind="tool", summary="Read: past.jsonl", first=True))
        on_event(AnswerEvent("Recovered: was fixing the parser."))
        on_event(DoneEvent(cost=0.0, usage={}))
      await asyncio.to_thread(_emit)
      return 0.0, {}

  agent = _RecallAgent()
  channel = _RecallChannel("oc_test", [
    IncomingMessage(
      event_type="im.message.receive_v1", chat_id="oc_test",
      sender_id="", message_id="recall_abc_1", msg_type="text",
      text=_RECALL_PROMPT_PREFIX + "The user asked you to recall a past "
      "coding session in this project.", create_time="1", is_internal=True,
    ),
    IncomingMessage(
      event_type="im.message.receive_v1", chat_id="oc_test",
      sender_id="ou_user", message_id="om_exit", msg_type="text",
      text="/exit", create_time="2",
    ),
  ])

  with mock.patch("nemo.agent.load_credentials", return_value={
    "app_id": "app_id",
    "app_secret": "app_secret",
    "email": "user@example.com",
  }), \
       mock.patch("nemo.agent.Database", _FakeDB), \
       mock.patch("nemo.agent.LarkChannel", return_value=channel), \
       mock.patch("nemo.agent.build_coding_agent", return_value=agent), \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    result = asyncio.run(main_loop("oc_test", str(tmp_path), "claude-opus-4-6"))

  assert result == 0
  # The placeholder card was visible BEFORE the SDK produced any event.
  assert agent.sent_at_entry is not None and agent.sent_at_entry >= 1
  pre_turn = channel.sent_cards[:agent.sent_at_entry]
  assert any(
    banner_marker in json.dumps(c, ensure_ascii=False) for c in pre_turn), pre_turn
  # Banner uses the blue informational colour.
  placeholder = next(
    c for c in pre_turn if banner_marker in json.dumps(c, ensure_ascii=False))
  assert "blue" in json.dumps(placeholder, ensure_ascii=False)
  # Once real progress streamed, the banner is gone from every later update
  # (working refresh + done card) — it doesn't linger as stale.
  for card in channel.updated_cards:
    assert banner_marker not in json.dumps(card, ensure_ascii=False), card


def test_shell_bang_context_injected_next_turn_double_bang_not_injected(tmp_path):
  from nemo.channel import IncomingMessage

  class _CapturingAgent(_FakeAgent):
    def __init__(self):
      self.prompts: list[str] = []

    def trailing_note(self, _sdk_session_id):
      return ""

    async def run_turn(self, prompt, on_event):
      self.prompts.append(prompt)
      def _emit():
        on_event(AnswerEvent("ok"))
        on_event(DoneEvent(cost=0.0, usage={}))
      await asyncio.to_thread(_emit)
      return 0.0, {}

  class _FakeShellManager:
    def __init__(self, _channel, *, chat_id, project_dir, on_context, timeout=60.0):
      del chat_id, project_dir, timeout
      self._on_context = on_context
      self.started: list[tuple[str, bool]] = []

    async def start(self, shortcut):
      self.started.append((shortcut.command, shortcut.inject_context))
      if shortcut.inject_context:
        self._on_context(
          "[Nemo shell context]\n"
          f"Command:\n{shortcut.command}\n\n"
          "Output:\ninjected-output")
      return "started"

    async def abort(self, _job_id):
      return True

    async def close(self):
      pass

    def set_project_dir(self, _project_dir):
      pass

  msg_user = lambda mid, text, ts: IncomingMessage(
    event_type="im.message.receive_v1", chat_id="oc_test",
    sender_id="ou_user", message_id=mid, msg_type="text",
    text=text, create_time=ts,
  )

  agent = _CapturingAgent()
  channel = _QueuedChannel("oc_test", [
    msg_user("om_shell1", "!echo injected-output", "1"),
    msg_user("om_shell2", "!!echo hidden-output", "2"),
    msg_user("om_prompt", "what happened?", "3"),
    msg_user("om_exit", "/exit", "4"),
  ])

  with mock.patch("nemo.agent.load_credentials", return_value={
    "app_id": "app_id",
    "app_secret": "app_secret",
    "email": "user@example.com",
  }), \
       mock.patch("nemo.agent.Database", _FakeDB), \
       mock.patch("nemo.agent.LarkChannel", return_value=channel), \
       mock.patch("nemo.agent.build_coding_agent", return_value=agent), \
       mock.patch("nemo.agent.shell_command.ShellJobManager", _FakeShellManager), \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    result = asyncio.run(main_loop("oc_test", str(tmp_path), "claude-opus-4-6"))

  assert result == 0
  assert agent.prompts
  assert "[Nemo shell context]" in agent.prompts[0]
  assert "echo injected-output" in agent.prompts[0]
  assert "injected-output" in agent.prompts[0]
  assert "hidden-output" not in agent.prompts[0]
  assert agent.prompts[0].endswith("[User message]\nwhat happened?")


def test_shell_complex_command_e2e_injects_real_subprocess_result(tmp_path):
  from nemo.channel import IncomingMessage

  class _CapturingAgent(_FakeAgent):
    def __init__(self):
      self.prompts: list[str] = []

    def trailing_note(self, _sdk_session_id):
      return ""

    async def run_turn(self, prompt, on_event):
      self.prompts.append(prompt)
      def _emit():
        on_event(AnswerEvent("ok"))
        on_event(DoneEvent(cost=0.0, usage={}))
      await asyncio.to_thread(_emit)
      return 0.0, {}

  class _ShellE2EChannel(_FakeChannel):
    def __init__(self, _chat_id, queued):
      super().__init__(_chat_id)
      self._messages = list(queued)
      self.sent_cards: list[object] = []
      self.updated_cards: list[object] = []
      self._shell_done = asyncio.Event()
      self._card_count = 0

    async def receive(self, timeout=300):
      if not self._messages:
        return None
      # Do not let the in-turn watcher consume /exit; leave it for the
      # top-level loop after the agent turn completes.
      if self._messages[0].text == "/exit" and timeout < 100:
        await asyncio.sleep(0.01)
        return None
      # Hold the user's follow-up until the real shell job has completed
      # and queued its context.
      if self._messages[0].text == "summarize shell result":
        try:
          await asyncio.wait_for(self._shell_done.wait(), timeout=5)
        except TimeoutError:
          pass
      return self._messages.pop(0)

    async def send_card(self, chat_id, card):
      self.sent_cards.append(card)
      self._card_count += 1
      return f"om_card_{self._card_count}"

    async def update_card(self, card_id, card):
      self.updated_cards.append(card)
      title = card.get("header", {}).get("title", {}).get("content", "")
      if title.startswith("Shell ") and title != "Shell running":
        self._shell_done.set()
      return card_id

    def push_back(self, message):
      self._messages.insert(0, message)

  msg_user = lambda mid, text, ts: IncomingMessage(
    event_type="im.message.receive_v1", chat_id="oc_test",
    sender_id="ou_user", message_id=mid, msg_type="text",
    text=text, create_time=ts,
  )
  marker = "nemo-shell-e2e-marker"
  out_path = tmp_path / "shell-e2e.txt"
  code = (
    "import json, pathlib, sys, time\n"
    f"path = pathlib.Path({str(out_path)!r})\n"
    "path.write_text('file-content', encoding='utf-8')\n"
    "print('phase=begin', flush=True)\n"
    "time.sleep(0.2)\n"
    f"print(json.dumps({{'marker': {marker!r}, 'numbers': [1, 2, 3]}}), flush=True)\n"
    "print('warn=stderr-line', file=sys.stderr, flush=True)\n"
    "print('phase=end', flush=True)\n"
  )
  command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
  channel = _ShellE2EChannel("oc_test", [
    msg_user("om_shell", f"!{command}", "1"),
    msg_user("om_prompt", "summarize shell result", "2"),
    msg_user("om_exit", "/exit", "3"),
  ])
  agent = _CapturingAgent()

  with mock.patch("nemo.agent.load_credentials", return_value={
    "app_id": "app_id",
    "app_secret": "app_secret",
    "email": "user@example.com",
  }), \
       mock.patch("nemo.agent.Database", _FakeDB), \
       mock.patch("nemo.agent.LarkChannel", return_value=channel), \
       mock.patch("nemo.agent.build_coding_agent", return_value=agent), \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    result = asyncio.run(main_loop("oc_test", str(tmp_path), "claude-opus-4-6"))

  assert result == 0
  rendered_cards = "\n".join(str(card) for card in channel.updated_cards)
  assert out_path.exists(), rendered_cards
  assert out_path.read_text(encoding="utf-8") == "file-content", rendered_cards
  assert agent.prompts
  prompt = agent.prompts[0]
  assert "[Nemo shell context]" in prompt
  assert marker in prompt
  assert "phase=begin" in prompt
  assert "phase=end" in prompt
  assert "warn=stderr-line" in prompt
  assert prompt.endswith("[User message]\nsummarize shell result")
  assert any(
    card.get("header", {}).get("title", {}).get("content") == "Shell done"
    for card in channel.updated_cards
  )


def test_main_loop_threads_agent_through_db_calls(tmp_path):
  """Per-agent session storage requires the daemon's --agent to
  reach both db.get_sdk_session_id (resume lookup at startup) and
  db.set_sdk_session_id (write back from DoneEvent.session_id). A
  typo or a missed kwarg silently degrades back to the old
  agent-blind behavior."""
  from nemo.channel import IncomingMessage

  recorded_get: list[tuple[str, str, str]] = []
  recorded_set: list[tuple[str, str, str, str]] = []

  class _SpyDB(_FakeDB):
    def get_chat_owner(self, _chat_id):
      # Return non-None so main_loop bothers calling get_sdk_session_id —
      # otherwise the early-out would skip the lookup entirely.
      return "old_session"

    def get_sdk_session_id(self, chat_id, agent, endpoint_key=""):
      recorded_get.append((chat_id, agent, endpoint_key))
      return ""

    def set_sdk_session_id(self, chat_id, sdk_session_id, agent,
                           endpoint_key=""):
      recorded_set.append((chat_id, sdk_session_id, agent, endpoint_key))

  class _SessionEmittingAgent(_FakeAgent):
    async def run_turn(self, _prompt, on_event):
      def _emit():
        on_event(AnswerEvent("ok"))
        # Non-empty session_id is what triggers set_sdk_session_id.
        on_event(DoneEvent(cost=0.01, usage={"input_tokens": 1},
                           session_id="codex-thread-xyz"))
      await asyncio.to_thread(_emit)
      return 0.01, {"input_tokens": 1}

  agent = _SessionEmittingAgent()
  queued = _QueuedChannel("oc_test", [
    IncomingMessage(
      event_type="im.message.receive_v1",
      chat_id="oc_test", sender_id="ou_user",
      message_id="om_msg", msg_type="text",
      text="hello", create_time="1",
    ),
    IncomingMessage(
      event_type="im.message.receive_v1",
      chat_id="oc_test", sender_id="ou_user",
      message_id="om_exit", msg_type="text",
      text="/exit", create_time="2",
    ),
  ])

  with mock.patch("nemo.agent.load_credentials", return_value={
    "app_id": "app_id", "app_secret": "app_secret", "email": "u@e.com",
  }), \
       mock.patch("nemo.agent.Database", _SpyDB), \
       mock.patch("nemo.agent.LarkChannel", return_value=queued), \
       mock.patch("nemo.agent.build_coding_agent", return_value=agent), \
       mock.patch("nemo.agent._send_response", new=mock.AsyncMock()), \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    rc = asyncio.run(
      main_loop("oc_test", str(tmp_path), "gpt-5.5", agent="codex")
    )
  assert rc == 0

  # Read path: agent is correctly threaded into get_sdk_session_id.
  # Default endpoint at startup → endpoint_key="".
  assert recorded_get == [("oc_test", "codex", "")], recorded_get
  # Write path: DoneEvent.session_id reaches the codex column, not a
  # agent-blind one. Default endpoint stays under endpoint_key="".
  assert ("oc_test", "codex-thread-xyz", "codex", "") in recorded_set, recorded_set


def test_model_switch_to_preset_sets_endpoint_and_remote_name(tmp_path):
  """`/model deepseek-v4-pro` must (a) flip the agent's EndpointConfig
  to the preset's anthropic_url + api key, (b) feed the protocol-
  specific remote name (e.g. ``deepseek-v4-pro[1m]``) into the next
  reset(), and (c) NOT crash on the legacy is_model_compatible path."""
  import os as _os
  from nemo.channel import IncomingMessage

  endpoint_history: list[tuple[str, str]] = []  # (base_url, api_key)
  reset_history: list[str] = []                  # model passed to reset

  class _SpyAgent(_FakeAgent):
    def set_endpoint(self, endpoint):
      endpoint_history.append((endpoint.base_url, endpoint.api_key))

    async def reset(self, _project_dir, model, resume=""):
      del resume
      reset_history.append(model)

  agent = _SpyAgent()
  queued = _QueuedChannel("oc_test", [
    IncomingMessage(
      event_type="im.message.receive_v1", chat_id="oc_test",
      sender_id="ou_user", message_id="om_switch", msg_type="text",
      text="/model deepseek-v4-pro", create_time="1",
    ),
    IncomingMessage(
      event_type="im.message.receive_v1", chat_id="oc_test",
      sender_id="ou_user", message_id="om_exit", msg_type="text",
      text="/exit", create_time="2",
    ),
  ])

  with mock.patch.dict(_os.environ, {"DEEPSEEK_API_KEY": "sk-test"}), \
       mock.patch("nemo.agent.load_credentials", return_value={
         "app_id": "a", "app_secret": "s", "email": "u@e.com",
       }), \
       mock.patch("nemo.agent.Database", _FakeDB), \
       mock.patch("nemo.agent.LarkChannel", return_value=queued), \
       mock.patch("nemo.agent.build_coding_agent", return_value=agent), \
       mock.patch("nemo.agent._send_response", new=mock.AsyncMock()), \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    rc = asyncio.run(
      main_loop("oc_test", str(tmp_path), "claude-opus-4-7", agent="claude")
    )
  assert rc == 0
  # Endpoint gets flipped exactly once with the DeepSeek anthropic URL
  # + key from the env var.
  assert endpoint_history == [
    ("https://api.deepseek.com/anthropic", "sk-test"),
  ], endpoint_history
  # Reset is called with the protocol-specific remote name (the [1m]
  # variant for the Anthropic side).
  assert "deepseek-v4-pro[1m]" in reset_history, reset_history


def test_model_switch_isolates_session_per_endpoint(tmp_path):
  """Regression: `/model deepseek-v4-pro` then `/model claude-opus-4-7`
  must NOT feed the DeepSeek session id back into real Anthropic.

  The DeepSeek Anthropic-compat gateway emits ``thinking`` blocks whose
  signatures only verify at DeepSeek; replaying that transcript against
  api.anthropic.com yields ``400 Invalid signature in thinking block``
  and wedges the session. Each upstream endpoint must keep its own
  resume id.
  """
  import os as _os
  from nemo.channel import IncomingMessage

  reset_calls: list[tuple[str, str, str]] = []  # (model, resume, endpoint_url)

  class _SpyAgent(_FakeAgent):
    def __init__(self):
      super().__init__()
      self._endpoint_url = ""

    def set_endpoint(self, endpoint):
      self._endpoint_url = endpoint.base_url

    async def reset(self, _project_dir, model, resume=""):
      reset_calls.append((model, resume, self._endpoint_url))

  agent = _SpyAgent()

  class _SpyDB(_FakeDB):
    """Default endpoint has a stored session from previous turns; the
    DeepSeek preset slot is empty. After switching to DeepSeek and back,
    the daemon should restore the default-endpoint session — NOT
    whatever DeepSeek left behind."""

    def __init__(self, project_dir):
      super().__init__(project_dir)
      # Persisted store: {(chat, agent, endpoint_key): sdk_session_id}
      self._store: dict[tuple[str, str, str], str] = {
        ("oc_test", "claude", ""): "default-uuid",
      }

    def get_sdk_session_id(self, chat_id, agent, endpoint_key=""):
      return self._store.get((chat_id, agent, endpoint_key), "")

    def set_sdk_session_id(self, chat_id, sdk_session_id, agent,
                           endpoint_key=""):
      self._store[(chat_id, agent, endpoint_key)] = sdk_session_id

  queued = _QueuedChannel("oc_test", [
    IncomingMessage(
      event_type="im.message.receive_v1", chat_id="oc_test",
      sender_id="ou_user", message_id="om_to_deepseek", msg_type="text",
      text="/model deepseek-v4-pro", create_time="1",
    ),
    IncomingMessage(
      event_type="im.message.receive_v1", chat_id="oc_test",
      sender_id="ou_user", message_id="om_back_to_opus", msg_type="text",
      text="/model claude-opus-4-7", create_time="2",
    ),
    IncomingMessage(
      event_type="im.message.receive_v1", chat_id="oc_test",
      sender_id="ou_user", message_id="om_exit", msg_type="text",
      text="/exit", create_time="3",
    ),
  ])

  with mock.patch.dict(_os.environ, {"DEEPSEEK_API_KEY": "sk-test"}), \
       mock.patch("nemo.agent.load_credentials", return_value={
         "app_id": "a", "app_secret": "s", "email": "u@e.com",
       }), \
       mock.patch("nemo.agent.Database", _SpyDB), \
       mock.patch("nemo.agent.LarkChannel", return_value=queued), \
       mock.patch("nemo.agent.build_coding_agent", return_value=agent), \
       mock.patch("nemo.agent._send_response", new=mock.AsyncMock()), \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    rc = asyncio.run(
      main_loop("oc_test", str(tmp_path), "claude-opus-4-7", agent="claude")
    )
  assert rc == 0

  # Two reset() calls: one for each /model switch.
  assert len(reset_calls) == 2, reset_calls
  # First switch: opus default → DeepSeek preset. DeepSeek slot is empty
  # in this DB, so the daemon must start a fresh session (resume=""),
  # not replay the default-endpoint transcript against DeepSeek.
  model1, resume1, endpoint1 = reset_calls[0]
  assert model1 == "deepseek-v4-pro[1m]", reset_calls
  assert resume1 == "", reset_calls
  assert endpoint1 == "https://api.deepseek.com/anthropic", reset_calls
  # Second switch: DeepSeek preset → opus default. THIS is the bug
  # path. Resume MUST come from the default-endpoint slot
  # ("default-uuid"), not from the DeepSeek session (whatever the
  # daemon ended up holding while running against the gateway).
  model2, resume2, endpoint2 = reset_calls[1]
  assert model2 == "claude-opus-4-7", reset_calls
  assert resume2 == "default-uuid", reset_calls
  assert endpoint2 == "", reset_calls


def test_model_swap_within_same_endpoint_keeps_session(tmp_path):
  """Swapping between two models that share an upstream (e.g.
  opus↔sonnet, both on api.anthropic.com) must keep the same SDK
  session. The bug fix isolates by endpoint URL, not by model name —
  routine model swaps should NOT segment context."""
  from nemo.channel import IncomingMessage

  reset_calls: list[tuple[str, str]] = []  # (model, resume)

  class _SpyAgent(_FakeAgent):
    def set_endpoint(self, _endpoint):
      pass

    async def reset(self, _project_dir, model, resume=""):
      reset_calls.append((model, resume))

  agent = _SpyAgent()

  class _SpyDB(_FakeDB):
    def __init__(self, project_dir):
      super().__init__(project_dir)
      self._store: dict[tuple[str, str, str], str] = {
        ("oc_test", "claude", ""): "default-uuid",
      }

    def get_sdk_session_id(self, chat_id, agent, endpoint_key=""):
      return self._store.get((chat_id, agent, endpoint_key), "")

    def set_sdk_session_id(self, chat_id, sdk_session_id, agent,
                           endpoint_key=""):
      self._store[(chat_id, agent, endpoint_key)] = sdk_session_id

  queued = _QueuedChannel("oc_test", [
    # Start on claude-opus-4-7 (default endpoint). Switch to sonnet,
    # then haiku — both also on default endpoint. Session continuity
    # is what we're checking.
    IncomingMessage(
      event_type="im.message.receive_v1", chat_id="oc_test",
      sender_id="ou_user", message_id="om_to_sonnet", msg_type="text",
      text="/model claude-sonnet-4-6", create_time="1",
    ),
    IncomingMessage(
      event_type="im.message.receive_v1", chat_id="oc_test",
      sender_id="ou_user", message_id="om_to_haiku", msg_type="text",
      text="/model claude-haiku-4-5", create_time="2",
    ),
    IncomingMessage(
      event_type="im.message.receive_v1", chat_id="oc_test",
      sender_id="ou_user", message_id="om_exit", msg_type="text",
      text="/exit", create_time="3",
    ),
  ])

  with mock.patch("nemo.agent.load_credentials", return_value={
         "app_id": "a", "app_secret": "s", "email": "u@e.com",
       }), \
       mock.patch("nemo.agent.Database", _SpyDB), \
       mock.patch("nemo.agent.LarkChannel", return_value=queued), \
       mock.patch("nemo.agent.build_coding_agent", return_value=agent), \
       mock.patch("nemo.agent._send_response", new=mock.AsyncMock()), \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    rc = asyncio.run(
      main_loop("oc_test", str(tmp_path), "claude-opus-4-7", agent="claude")
    )
  assert rc == 0

  # Both switches stay on the default endpoint and so must resume the
  # same session id. Losing continuity here would break the user-
  # facing promise that "swap model = pick a different brain for the
  # same conversation".
  assert len(reset_calls) == 2, reset_calls
  for model, resume in reset_calls:
    assert resume == "default-uuid", (model, resume, reset_calls)
  assert reset_calls[0][0] == "claude-sonnet-4-6", reset_calls
  assert reset_calls[1][0] == "claude-haiku-4-5", reset_calls


def _run_agent_switch(tmp_path, *, prior_codex_session: str = ""):
  """Helper: drive main_loop through /agent codex (then /exit).

  Returns (start_calls, send_calls). prior_codex_session controls
  whether the spy DB advertises a stored codex session id for this
  chat (resume path) or none (fresh-start path).
  """
  from nemo.channel import IncomingMessage

  start_calls: list[tuple[str, str]] = []
  send_calls: list[str] = []

  class _SwitchAgent(_FakeAgent):
    async def start(self, _project_dir, model, resume=""):
      start_calls.append((model, resume))

  class _SpyDB(_FakeDB):
    def get_chat_owner(self, _chat_id):
      return None

    def get_sdk_session_id(self, chat_id, agent, endpoint_key=""):
      del chat_id, endpoint_key
      if agent == "codex":
        return prior_codex_session
      return ""

    def set_sdk_session_id(self, *_args, **_kwargs):
      pass

  queued = _QueuedChannel("oc_test", [
    IncomingMessage(
      event_type="im.message.receive_v1", chat_id="oc_test",
      sender_id="ou_user", message_id="om_switch", msg_type="text",
      text="/agent codex", create_time="1",
    ),
    IncomingMessage(
      event_type="im.message.receive_v1", chat_id="oc_test",
      sender_id="ou_user", message_id="om_exit", msg_type="text",
      text="/exit", create_time="2",
    ),
  ])

  send_response = mock.AsyncMock()

  def _capture_send(_channel, _chat, body, _db, *_args, **_kwargs):
    send_calls.append(body)

  send_response.side_effect = _capture_send

  with mock.patch("nemo.agent.load_credentials", return_value={
    "app_id": "a", "app_secret": "s", "email": "u@e.com",
  }), \
       mock.patch("nemo.agent.Database", _SpyDB), \
       mock.patch("nemo.agent.LarkChannel", return_value=queued), \
       mock.patch("nemo.agent.build_coding_agent",
                  side_effect=lambda _p, *_, **__: _SwitchAgent()), \
       mock.patch("nemo.agent._send_response", new=send_response), \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    rc = asyncio.run(
      main_loop("oc_test", str(tmp_path), "claude-opus-4-7", agent="claude")
    )
  assert rc == 0
  return start_calls, send_calls


def test_agent_switch_rebuilds_agent_with_new_default(tmp_path):
  """`/agent codex` from a Claude daemon must (a) stop the old
  agent, (b) build a fresh agent for codex, (c) reset model to
  gpt-5.5 (codex default), (d) load the per-agent session id, (e)
  call agent.start with the new model + that session id."""
  start_calls, _ = _run_agent_switch(
    tmp_path, prior_codex_session="codex-thread-prev")
  # Two start()s: the original at boot (claude default model) and the
  # post-switch one (codex default + per-agent resume id).
  assert start_calls[0] == ("claude-opus-4-7", ""), start_calls
  assert start_calls[1] == ("gpt-5.5", "codex-thread-prev"), start_calls


def test_agent_switch_message_says_resumed_when_prior_session_exists(tmp_path):
  """Confirmation card after /agent must tell the user their
  context is the new agent's *own* prior history — not the
  previous agent's. Otherwise users hit "did the bot forget?"
  surprises when each agent has its own session column."""
  _, send_calls = _run_agent_switch(
    tmp_path, prior_codex_session="codex-thread-9876ab")
  # Find the switch confirmation among the send_response calls.
  switch_msg = next(
    (m for m in send_calls if "Switched to agent" in m), None)
  assert switch_msg is not None, send_calls
  # Mentions resume + the codex session prefix + the cross-agent
  # isolation note.
  assert "Resuming" in switch_msg
  assert "codex" in switch_msg
  assert "codex-th" in switch_msg  # first 8 chars of stored id
  assert "does not see" in switch_msg


def test_agent_switch_message_says_fresh_when_no_prior_session(tmp_path):
  """First-time switch: new agent has no stored session id for
  this chat → the card should say "Fresh" and explain the other
  agents' transcripts are still around when you switch back."""
  _, send_calls = _run_agent_switch(tmp_path, prior_codex_session="")
  switch_msg = next(
    (m for m in send_calls if "Switched to agent" in m), None)
  assert switch_msg is not None, send_calls
  assert "Fresh" in switch_msg
  assert "codex" in switch_msg
  assert "no prior history" in switch_msg.lower()
  assert "switching back" in switch_msg.lower()


def test_model_picker_submit_switches_model(tmp_path):
  """Clicking Submit on the `/model` picker fires a card.action.trigger
  with action='model_switch:<name>'. The main loop must synthesise an
  internal /model <name>, push it back to itself, and run the regular
  /model dispatch on the next iteration so the SDK is restarted with
  the picked model."""
  from nemo.channel import IncomingMessage

  class _ReplayChannel(_QueuedChannel):
    """_QueuedChannel + a working push_back so the daemon's synthetic
    /model message actually gets replayed on the next receive(). Also
    records update_card calls so we can assert the picker gets locked."""

    def __init__(self, *a, **kw):
      super().__init__(*a, **kw)
      self.update_card_calls: list[tuple[str, object]] = []

    def push_back(self, message):
      self._messages.insert(0, message)

    async def update_card(self, card_id, card):
      self.update_card_calls.append((card_id, card))
      return card_id

  class _SpyAgent(_FakeAgent):
    def __init__(self):
      super().__init__()
      self.reset_calls: list[str] = []

    def set_endpoint(self, _endpoint):
      pass

    async def reset(self, _project_dir, model, resume=""):
      del resume
      self.reset_calls.append(model)

  agent = _SpyAgent()
  queued = _ReplayChannel("oc_test", [
    # The picker submit arrives as a card.action.trigger from the
    # operator with the model_switch:<name> action discriminator.
    # message_id is the picker card's id (relay propagates it from
    # event.context.open_message_id) — the daemon locks THIS card.
    IncomingMessage(
      event_type="card.action.trigger",
      chat_id="oc_test",
      operator_id="ou_user",
      sender_id="ou_user",
      message_id="om_picker_card",
      action_value={"action": "model_switch:claude-sonnet-4-6",
                    "chat_id": "oc_test"},
      create_time="1",
    ),
    IncomingMessage(
      event_type="im.message.receive_v1", chat_id="oc_test",
      sender_id="ou_user", message_id="om_exit", msg_type="text",
      text="/exit", create_time="2",
    ),
  ])

  with mock.patch("nemo.agent.load_credentials", return_value={
    "app_id": "app_id", "app_secret": "app_secret",
    "email": "user@example.com",
  }), \
       mock.patch("nemo.agent.Database", _FakeDB), \
       mock.patch("nemo.agent.LarkChannel", return_value=queued), \
       mock.patch("nemo.agent.build_coding_agent", return_value=agent), \
       mock.patch("nemo.agent._send_response", new=mock.AsyncMock()), \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    rc = asyncio.run(
      main_loop("oc_test", str(tmp_path), "claude-opus-4-7", agent="claude"))

  assert rc == 0
  # The synthesised /model claude-sonnet-4-6 must have driven a reset
  # to sonnet — bare "switch happened" is the contract here.
  assert "claude-sonnet-4-6" in agent.reset_calls, agent.reset_calls
  # And the picker card must have been locked to a confirmation state
  # (no form). The lock targets the picker's own message_id.
  locks = [c for c in queued.update_card_calls if c[0] == "om_picker_card"]
  assert locks, queued.update_card_calls
  locked_card = locks[-1][1]
  import json as _json
  blob = _json.dumps(locked_card, ensure_ascii=False)
  assert "select_static" not in blob and "form_action_type" not in blob
  assert "claude-sonnet-4-6" in blob


def test_model_picker_submit_unauthorized_is_ignored(tmp_path):
  """A submit from a sender who is not the operator/coowner/guest must
  NOT swap the model — otherwise any chat member could hijack /model
  by clicking the picker card."""
  from nemo.channel import IncomingMessage

  class _ReplayChannel(_QueuedChannel):
    def push_back(self, message):
      self._messages.insert(0, message)

  class _SpyAgent(_FakeAgent):
    def __init__(self):
      super().__init__()
      self.reset_calls: list[str] = []

    def set_endpoint(self, _endpoint):
      pass

    async def reset(self, _project_dir, model, resume=""):
      del resume
      self.reset_calls.append(model)

  agent = _SpyAgent()
  queued = _ReplayChannel("oc_test", [
    IncomingMessage(
      event_type="card.action.trigger",
      chat_id="oc_test",
      # operator_id is some random user, not "ou_user" (the daemon's
      # operator per _FakeChannel.resolve_operator_and_bot).
      operator_id="ou_stranger",
      sender_id="ou_stranger",
      action_value={"action": "model_switch:claude-sonnet-4-6",
                    "chat_id": "oc_test"},
      create_time="1",
    ),
    IncomingMessage(
      event_type="im.message.receive_v1", chat_id="oc_test",
      sender_id="ou_user", message_id="om_exit", msg_type="text",
      text="/exit", create_time="2",
    ),
  ])

  with mock.patch("nemo.agent.load_credentials", return_value={
    "app_id": "app_id", "app_secret": "app_secret",
    "email": "user@example.com",
  }), \
       mock.patch("nemo.agent.Database", _FakeDB), \
       mock.patch("nemo.agent.LarkChannel", return_value=queued), \
       mock.patch("nemo.agent.build_coding_agent", return_value=agent), \
       mock.patch("nemo.agent._send_response", new=mock.AsyncMock()), \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    rc = asyncio.run(
      main_loop("oc_test", str(tmp_path), "claude-opus-4-7", agent="claude"))
  assert rc == 0
  assert agent.reset_calls == [], agent.reset_calls


def test_agent_picker_submit_switches_agent(tmp_path):
  """Clicking Submit on the `/agent` picker fires a card.action.trigger
  with action='agent_switch:<name>'. The main loop must synthesise an
  internal /agent <name>, push it back, and run the regular /agent
  dispatch on the next iteration so build_coding_agent is rebuilt with
  the picked kind and the picker card is locked."""
  from nemo.channel import IncomingMessage

  class _ReplayChannel(_QueuedChannel):
    def __init__(self, *a, **kw):
      super().__init__(*a, **kw)
      self.update_card_calls: list[tuple[str, object]] = []

    def push_back(self, message):
      self._messages.insert(0, message)

    async def update_card(self, card_id, card):
      self.update_card_calls.append((card_id, card))
      return card_id

  builds: list[str] = []

  def _build(kind, *a, **kw):
    del a, kw
    builds.append(kind)
    return _FakeAgent()

  queued = _ReplayChannel("oc_test", [
    IncomingMessage(
      event_type="card.action.trigger",
      chat_id="oc_test",
      operator_id="ou_user",
      sender_id="ou_user",
      message_id="om_agent_picker_card",
      action_value={"action": "agent_switch:codex", "chat_id": "oc_test"},
      create_time="1",
    ),
    IncomingMessage(
      event_type="im.message.receive_v1", chat_id="oc_test",
      sender_id="ou_user", message_id="om_exit", msg_type="text",
      text="/exit", create_time="2",
    ),
  ])

  with mock.patch("nemo.agent.load_credentials", return_value={
    "app_id": "app_id", "app_secret": "app_secret",
    "email": "user@example.com",
  }), \
       mock.patch("nemo.agent.Database", _FakeDB), \
       mock.patch("nemo.agent.LarkChannel", return_value=queued), \
       mock.patch("nemo.agent.build_coding_agent", side_effect=_build), \
       mock.patch("nemo.agent._send_response", new=mock.AsyncMock()), \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    rc = asyncio.run(
      main_loop("oc_test", str(tmp_path), "claude-opus-4-7", agent="claude"))

  assert rc == 0
  # build_coding_agent should have been called once at startup for claude
  # AND again for the synthesised /agent codex switch.
  assert "claude" in builds and "codex" in builds, builds
  # The picker card must have been locked to a confirmation state — same
  # message_id the click came from, no form survives the lock.
  locks = [c for c in queued.update_card_calls if c[0] == "om_agent_picker_card"]
  assert locks, queued.update_card_calls
  locked_card = locks[-1][1]
  import json as _json
  blob = _json.dumps(locked_card, ensure_ascii=False)
  assert "select_static" not in blob and "form_action_type" not in blob
  assert "codex" in blob


def test_codex_agent_rejects_claude_model_switch(tmp_path):
  from nemo.channel import IncomingMessage

  class _TrackResetAgent(_FakeAgent):
    def __init__(self):
      self.reset = mock.AsyncMock()

  agent = _TrackResetAgent()
  queued = _QueuedChannel("oc_test", [
    IncomingMessage(
      event_type="im.message.receive_v1",
      chat_id="oc_test",
      sender_id="ou_user",
      message_id="om_model",
      msg_type="text",
      text="/model claude-sonnet-4-6",
      create_time="1",
    ),
    IncomingMessage(
      event_type="im.message.receive_v1",
      chat_id="oc_test",
      sender_id="ou_user",
      message_id="om_exit",
      msg_type="text",
      text="/exit",
      create_time="2",
    ),
  ])

  with mock.patch("nemo.agent.load_credentials", return_value={
    "app_id": "app_id",
    "app_secret": "app_secret",
    "email": "user@example.com",
  }), \
       mock.patch("nemo.agent.Database", _FakeDB), \
       mock.patch("nemo.agent.LarkChannel", return_value=queued), \
       mock.patch("nemo.agent.build_coding_agent", return_value=agent), \
       mock.patch("nemo.agent._send_response", new=mock.AsyncMock()) as send_response, \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    result = asyncio.run(
      main_loop("oc_test", str(tmp_path), "gpt-5.5", agent="codex")
    )

  assert result == 0
  agent.reset.assert_not_awaited()
  call = next(
    c for c in send_response.await_args_list
    if "claude-sonnet-4-6" in c.args[2]
  )
  body = call.args[2]
  assert "Unknown model" in body
  assert "codex" in body
  assert "gpt-5.5" in body  # available list shown


# ---------------------------------------------------------------------------
# Done-card tiered fallback
# ---------------------------------------------------------------------------


class _FakeUpdateChannel:
  """Channel stub exposing a controllable synchronous update_card."""

  def __init__(self, outcomes):
    # outcomes: list where each item is either an Exception (raise) or a
    # string (return as new card id). Consumed in order.
    self._outcomes = list(outcomes)
    self.calls = []  # captured (card_id, card) tuples
    self.token = "tok"

  def update_card(self, card_id, card):
    self.calls.append((card_id, card))
    outcome = self._outcomes.pop(0)
    if isinstance(outcome, Exception):
      raise outcome
    return outcome


def _run_fallback(channel, final_text="answer body", compact_notice="", thinking=None):
  """Drive _update_done_card_with_fallback with identity await_channel."""
  register_calls = []
  return _update_done_card_with_fallback(
    channel=channel,
    chat_id="oc_test",
    turn_card_id="om_card0",
    final_text=final_text,
    thinking=thinking or [],
    elapsed=1,
    usage={"input_tokens": 1},
    session_id="sess_x",
    await_channel=lambda x: x,
    register_msg=lambda msg_id, chat_id: register_calls.append((msg_id, chat_id)),
    compact_notice=compact_notice,
  )


def test_done_card_full_body_success_single_update():
  channel = _FakeUpdateChannel(outcomes=["om_card1"])
  result = _run_fallback(channel)
  assert result == "om_card1"
  assert len(channel.calls) == 1


def test_done_card_fallback_preserves_compact_notice():
  channel = _FakeUpdateChannel(outcomes=["om_card1"])
  result = _run_fallback(
    channel,
    compact_notice="Context compacted: 12k -> 4k tokens",
  )
  assert result == "om_card1"
  _, card = channel.calls[0]
  elements = card["body"]["elements"]
  assert "Context compacted" in elements[0]["content"]
  assert elements[1]["content"] == "answer body"


def test_done_card_transport_error_recovers_via_preview_retry():
  # Tier 1 fails with RemoteDisconnected (transport), tier 2 (preview)
  # succeeds. No file upload should happen.
  transport_err = ConnectionError("Remote end closed connection without response")
  channel = _FakeUpdateChannel(outcomes=[transport_err, "om_card1"])

  # Even a tiny body must still go through preview retry, not file upload,
  # since the first failure was transport (not content-size).
  with mock.patch("nemo.lark.api.upload_file") as upload, \
       mock.patch("nemo.lark.api.send_file") as send_file:
    result = _run_fallback(channel, final_text="short answer")

  assert result == "om_card1"
  assert len(channel.calls) == 2
  upload.assert_not_called()
  send_file.assert_not_called()


def test_done_card_falls_back_to_file_when_preview_also_fails():
  # Both tier 1 and tier 2 fail — tier 3 uploads file and updates card.
  err1 = ConnectionError("closed")
  err2 = ConnectionError("closed again")
  channel = _FakeUpdateChannel(outcomes=[err1, err2, "om_card_final"])

  with mock.patch("nemo.lark.api.upload_file", return_value="file_key_1") as upload, \
       mock.patch("nemo.lark.api.send_file") as send_file:
    result = _run_fallback(channel, final_text="body " * 50)

  assert result == "om_card_final"
  assert len(channel.calls) == 3
  upload.assert_called_once()
  send_file.assert_called_once()
  # Third card body should include the "sent as file" note.
  _, final_card = channel.calls[2]
  serialized = repr(final_card)
  assert "sent as file" in serialized


def test_done_card_auth_error_skips_fallback_chain():
  # HTTPError 403 → not retriable; no tier 2/3 attempts.
  import http.client
  http_err = urllib.error.HTTPError(
    url="http://x", code=403, msg="forbidden",
    hdrs=http.client.HTTPMessage(), fp=None,
  )
  channel = _FakeUpdateChannel(outcomes=[http_err])

  with mock.patch("nemo.lark.api.upload_file") as upload, \
       mock.patch("nemo.lark.api.send_file") as send_file:
    result = _run_fallback(channel)

  # Returns the original card id; only one update_card attempt made.
  assert result == "om_card0"
  assert len(channel.calls) == 1
  upload.assert_not_called()
  send_file.assert_not_called()


def test_done_card_oversized_body_uploads_file_without_doomed_patch():
  # The empty-card bug: Lark 200s an oversized card but renders it EMPTY, so
  # we must detect oversize proactively and never PATCH the doomed card.
  from nemo.cards import LARK_CARD_BYTE_LIMIT
  channel = _FakeUpdateChannel(outcomes=["om_card_file"])
  huge_body = "x" * (LARK_CARD_BYTE_LIMIT + 5000)

  with mock.patch("nemo.lark.api.upload_file", return_value="fk") as upload, \
       mock.patch("nemo.lark.api.send_file") as send_file:
    result = _run_fallback(channel, final_text=huge_body)

  # No oversized full_card was ever sent — the single update_card call carries
  # only the short preview, and the body went out as a file.
  upload.assert_called_once()
  send_file.assert_called_once()
  assert len(channel.calls) == 1
  _, preview_card = channel.calls[0]
  assert "sent as file" in repr(preview_card)
  assert result == "om_card_file"


def test_done_card_huge_timeline_attaches_file_and_updates_normally():
  # A huge timeline must be preserved as a .md file while the (now bounded)
  # done card still PATCHes normally — the inline panel is trimmed, not empty.
  from nemo.cards import ThinkingStep, timeline_overflows
  huge_thinking = []
  for i in range(1500):
    huge_thinking.append(
      ThinkingStep("thinking", f"reasoning block {i} " + "x" * 120))
    huge_thinking.append(ThinkingStep("tool", f"Read: file_{i}.py"))
  assert timeline_overflows(huge_thinking)
  channel = _FakeUpdateChannel(outcomes=["om_card1"])

  with mock.patch("nemo.lark.api.upload_file", return_value="fk") as upload, \
       mock.patch("nemo.lark.api.send_file") as send_file:
    result = _run_fallback(channel, thinking=huge_thinking)

  # Timeline went out as a file…
  upload.assert_called_once()
  send_file.assert_called_once()
  # …and the done card still updated in a single normal PATCH.
  assert result == "om_card1"
  assert len(channel.calls) == 1


def test_should_send_plain_text_for_short_natural_language():
  assert _should_send_plain_text("收到，我现在去看发送链路。")


def test_should_send_plain_text_for_simple_two_sentence_reply():
  text = "我已经看过了。接下来会把策略收口到 Nemo。"
  assert _should_send_plain_text(text)


def test_should_not_send_plain_text_for_code_fence():
  assert not _should_send_plain_text("这里有代码：\n```python\nprint('hi')\n```")
  assert not _should_send_plain_text("运行 `nemo --help` 看一下。")


def test_should_not_send_plain_text_for_list_link_and_emphasis():
  assert not _should_send_plain_text("- one")
  assert not _should_send_plain_text("- one\n- two")
  assert not _should_send_plain_text("+ one")
  assert not _should_send_plain_text("1. one")
  assert not _should_send_plain_text("1) one")
  assert not _should_send_plain_text("详情见 [文档](https://example.com)")
  assert not _should_send_plain_text("这是 **重点**。")
  assert not _should_send_plain_text("这是 *强调*。")
  assert not _should_send_plain_text("这是 __重点__。")
  assert not _should_send_plain_text("这是 ~~删除线~~。")


def test_should_not_send_plain_text_for_control_tags():
  assert not _should_send_plain_text('<at user_id="all">everyone</at> 请看这里')


def test_should_not_send_plain_text_for_structured_multiline_layout():
  assert not _should_send_plain_text("第一段。\n\n第二段。")
  assert not _should_send_plain_text("一\n二\n三\n四")
  assert not _should_send_plain_text("# 标题\n内容")
  assert not _should_send_plain_text("> 引用")
  assert not _should_send_plain_text("| a | b |\n| - | - |")
  assert not _should_send_plain_text("标题\n---")


def test_send_response_uses_plain_text_for_short_reply():
  channel = _RecordingChannel("oc_test")
  db = mock.Mock()

  with mock.patch("nemo.agent._register_msg"):
    msg_id = asyncio.run(
      _send_response(channel, "oc_test", "收到，我现在去处理。", db)
    )

  assert msg_id == "om_text"
  assert channel.sent_texts == [("oc_test", "收到，我现在去处理。")]
  assert channel.sent_cards == []
  db.record_sent.assert_called_once_with(
    "om_text", text="收到，我现在去处理。", chat_id="oc_test")


def test_send_response_uses_card_for_structured_markdown():
  channel = _RecordingChannel("oc_test")
  db = mock.Mock()
  text = "- first\n- second"

  with mock.patch("nemo.agent._register_msg"):
    msg_id = asyncio.run(_send_response(channel, "oc_test", text, db))

  assert msg_id == "om_card"
  assert channel.sent_texts == []
  assert len(channel.sent_cards) == 1
  db.record_sent.assert_called_once_with("om_card", text=text, chat_id="oc_test")


def test_send_response_uses_card_for_inline_code():
  channel = _RecordingChannel("oc_test")
  db = mock.Mock()
  text = "运行 `nemo --help` 看一下。"

  with mock.patch("nemo.agent._register_msg"):
    msg_id = asyncio.run(_send_response(channel, "oc_test", text, db))

  assert msg_id == "om_card"
  assert channel.sent_texts == []
  assert len(channel.sent_cards) == 1
  db.record_sent.assert_called_once_with("om_card", text=text, chat_id="oc_test")


def test_send_response_uses_card_for_control_tags():
  channel = _RecordingChannel("oc_test")
  db = mock.Mock()
  text = '<at user_id="all">everyone</at> 请看这里'

  with mock.patch("nemo.agent._register_msg"):
    msg_id = asyncio.run(_send_response(channel, "oc_test", text, db))

  assert msg_id == "om_card"
  assert channel.sent_texts == []
  assert len(channel.sent_cards) == 1
  db.record_sent.assert_called_once_with("om_card", text=text, chat_id="oc_test")


def test_send_response_uses_card_for_structured_multiline_layout():
  channel = _RecordingChannel("oc_test")
  db = mock.Mock()
  text = "第一段。\n\n第二段。"

  with mock.patch("nemo.agent._register_msg"):
    msg_id = asyncio.run(_send_response(channel, "oc_test", text, db))

  assert msg_id == "om_card"
  assert channel.sent_texts == []
  assert len(channel.sent_cards) == 1
  db.record_sent.assert_called_once_with("om_card", text=text, chat_id="oc_test")


# ---------------------------------------------------------------------------
# Pending message merging
# ---------------------------------------------------------------------------

from nemo.channel import IncomingMessage as _IM


def _msg(text: str, **kw) -> _IM:
  return _IM(event_type="message", chat_id="oc_test",
             sender_id="ou_user", msg_type="text", text=text,
             message_id=kw.get("message_id", "om_1"),
             create_time=kw.get("create_time", "1"), **{
               k: v for k, v in kw.items()
               if k not in ("message_id", "create_time")
             })


def test_merge_pending_empty():
  assert _merge_pending([]) is None


def test_merge_pending_single():
  m = _msg("hello")
  result = _merge_pending([m])
  assert result is m


def test_merge_pending_multiple_text():
  msgs = [
    _msg("帮我改一下登录页", message_id="om_1", create_time="1"),
    _msg("就是那个颜色太丑了", message_id="om_2", create_time="2"),
    _msg("改成蓝色的", message_id="om_3", create_time="3"),
  ]
  result = _merge_pending(msgs)
  assert isinstance(result, _IM)
  assert "[用户在上一轮工作期间发送了 3 条消息]" in result.text
  assert "帮我改一下登录页" in result.text
  assert "改成蓝色的" in result.text
  # Uses last message's id and create_time
  assert result.message_id == "om_3"
  assert result.create_time == "3"


def test_merge_pending_with_non_text():
  """Non-text messages (card actions) are returned separately."""
  text_msg = _msg("hello", message_id="om_1")
  card_msg = _IM(event_type="card.action.trigger", chat_id="oc_test",
                 action_value={"action": "approve"})
  result = _merge_pending([text_msg, card_msg])
  # Returns tuple: (merged_text, [non_text_msgs])
  assert isinstance(result, tuple)
  merged, others = result
  assert merged.text == "hello"  # single text msg, no header
  assert len(others) == 1
  assert others[0].event_type == "card.action.trigger"


def test_requeue_pending_merges():
  """_requeue_pending pushes a single merged message back."""
  pushed: list[_IM] = []

  class _FakeCh:
    def push_back(self, msg):
      pushed.append(msg)

  msgs = [
    _msg("first", message_id="om_1"),
    _msg("second", message_id="om_2"),
  ]
  _requeue_pending(msgs, _FakeCh())  # type: ignore[arg-type]
  assert len(pushed) == 1
  assert "[用户在上一轮工作期间发送了 2 条消息]" in pushed[0].text
  assert "first" in pushed[0].text
  assert "second" in pushed[0].text


def test_merge_pending_real_event_type():
  """Real Lark/relay messages carry event_type='im.message.receive_v1';
  they must still be recognized as text and merged. Regression for a bug
  where the filter accepted only 'message'/'' so production messages got
  re-pushed one-by-one instead of merged."""
  msgs = [
    _IM(event_type="im.message.receive_v1", chat_id="oc_test",
        sender_id="ou_user", msg_type="text", text="第一条",
        message_id="om_1", create_time="1"),
    _IM(event_type="im.message.receive_v1", chat_id="oc_test",
        sender_id="ou_user", msg_type="text", text="第二条",
        message_id="om_2", create_time="2"),
    _IM(event_type="im.message.receive_v1", chat_id="oc_test",
        sender_id="ou_user", msg_type="text", text="第三条",
        message_id="om_3", create_time="3"),
  ]
  result = _merge_pending(msgs)
  assert isinstance(result, _IM), \
    f"expected single merged IncomingMessage, got {type(result).__name__}"
  assert "[用户在上一轮工作期间发送了 3 条消息]" in result.text
  assert "第一条" in result.text and "第三条" in result.text
  assert result.message_id == "om_3"


def test_user_message_event_accepts_lark_receive_and_legacy_types():
  assert _is_user_message_event(_IM(event_type="im.message.receive_v1")) is True
  assert _is_user_message_event(_IM(event_type="message")) is True
  assert _is_user_message_event(_IM(event_type="")) is True


def test_reaction_event_is_not_user_message_event():
  msg = _IM(
    event_type="im.message.reaction.created_v1",
    chat_id="oc_test",
    sender_id="ou_user",
    message_id="om_target",
    text="Yes",
  )
  assert _is_user_message_event(msg) is False


def test_merge_pending_after_recall():
  """Recalling a message should leave it out of the merge."""
  msgs = [
    _msg("first", message_id="om_1"),
    _msg("second", message_id="om_2"),
    _msg("third", message_id="om_3"),
  ]
  # Simulate recall of om_2
  recalled_id = "om_2"
  msgs[:] = [m for m in msgs if m.message_id != recalled_id]

  result = _merge_pending(msgs)
  assert isinstance(result, _IM)
  assert "[用户在上一轮工作期间发送了 2 条消息]" in result.text
  assert "first" in result.text
  assert "second" not in result.text
  assert "third" in result.text


def test_merge_pending_all_recalled():
  """If all messages are recalled, nothing to merge."""
  msgs = [_msg("only", message_id="om_1")]
  msgs[:] = [m for m in msgs if m.message_id != "om_1"]
  assert _merge_pending(msgs) is None


# ---------------------------------------------------------------------------
# _in_turn_filtered_out — bug fix: in-turn watcher must apply mention filter
# so non-bot-directed chat doesn't get pulled into the pending queue when
# need_mention is on. Pre-fix, OneSecond reactions were attached to messages
# that weren't even directed at the bot, and those messages were re-queued
# back into the channel after the turn (causing nemo to "respond" to chatter
# between teammates).
# ---------------------------------------------------------------------------


def _never_own(_mid: str) -> bool:
  return False


def test_in_turn_filtered_out_skips_unmentioned_message():
  msg = IncomingMessage(
    event_type="message", chat_id="oc_x",
    text="hey teammate", mentions=[],
  )
  assert _in_turn_filtered_out(msg, "ou_bot", _never_own) is True


def test_in_turn_filtered_out_keeps_mentioned_message():
  msg = IncomingMessage(
    event_type="message", chat_id="oc_x",
    text="@nemo do thing",
    mentions=[{"id": "ou_bot", "name": "nemo"}],
  )
  assert _in_turn_filtered_out(msg, "ou_bot", _never_own) is False


def test_in_turn_filtered_out_keeps_reply_to_own_card():
  msg = IncomingMessage(
    event_type="message", chat_id="oc_x",
    text="follow up", mentions=[], parent_id="om_bot_card",
  )
  is_own = lambda mid: mid == "om_bot_card"  # noqa: E731
  assert _in_turn_filtered_out(msg, "ou_bot", is_own) is False


def test_in_turn_filtered_out_skips_reply_to_other_user():
  msg = IncomingMessage(
    event_type="message", chat_id="oc_x",
    text="agreeing with you", mentions=[], parent_id="om_other_user",
  )
  is_own = lambda mid: mid == "om_bot_card"  # noqa: E731
  assert _in_turn_filtered_out(msg, "ou_bot", is_own) is True


# ---------------------------------------------------------------------------
# _format_rate_limit_notice
# ---------------------------------------------------------------------------

def test_format_rate_limit_notice_rejected_with_resets_in_minutes():
  """Rejected status renders with red marker, type, utilization, and ETA."""
  import time as _time
  ev = RateLimitNoticeEvent(
    status="rejected",
    rate_limit_type="five_hour",
    resets_at=int(_time.time()) + 12 * 60,
    utilization=0.99,
  )
  out = _format_rate_limit_notice(ev)
  assert out.startswith("⛔ Rate limit hit")
  assert "(five_hour)" in out
  assert "99% used" in out
  assert "resets in 12m" in out


def test_format_rate_limit_notice_warning_renders_yellow_marker():
  ev = RateLimitNoticeEvent(status="allowed_warning", utilization=0.9)
  out = _format_rate_limit_notice(ev)
  assert out.startswith("⚠️ Rate limit warning")
  assert "90% used" in out


def test_format_rate_limit_notice_allowed_clears_to_empty():
  """Status returning to 'allowed' clears the banner — caller treats '' as hide."""
  ev = RateLimitNoticeEvent(status="allowed", rate_limit_type="five_hour")
  assert _format_rate_limit_notice(ev) == ""


def test_format_rate_limit_notice_resets_in_hours():
  import time as _time
  ev = RateLimitNoticeEvent(
    status="rejected",
    resets_at=int(_time.time()) + 3 * 3600 + 25 * 60,
  )
  out = _format_rate_limit_notice(ev)
  assert "resets in 3h 25m" in out


def test_format_rate_limit_notice_past_reset_omits_eta():
  """If resets_at is already in the past, drop the misleading 'resets in -Ns'."""
  import time as _time
  ev = RateLimitNoticeEvent(
    status="rejected",
    resets_at=int(_time.time()) - 30,
  )
  out = _format_rate_limit_notice(ev)
  assert "resets in" not in out


def test_working_card_retries_on_first_create_failure(tmp_path):
  """Regression: 0.4.8 and earlier — when the first send_card for a
  turn's working card fails (transient `Remote end closed connection
  without response` from Lark), _ensure_card logged the error and left
  _turn_card_id None. Every subsequent progress event silently no-op'd
  the update, so the user saw no working indicator for the rest of the
  turn — sometimes minutes of dead air before the final answer card.

  Fix: _update_working retries _ensure_card when the card is still
  missing. This test drives a turn with multiple ProgressEvents and a
  send_card that fails once then succeeds, and asserts the second
  progress event causes a retry that actually produces a card.
  """
  from nemo.channel import IncomingMessage

  send_card_calls: list[object] = []
  update_card_calls: list[object] = []
  # Distinguish the start card (always sent first by main_loop) from
  # the working card created on the first ProgressEvent.
  _state = {"working_create_attempts": 0}

  class _FlakyChannel(_FakeChannel):
    async def send_card(self, _chat_id, card):
      send_card_calls.append(card)
      # First send_card is the daemon's start card — let it succeed so
      # we're testing the working-card failure path specifically.
      if len(send_card_calls) == 1:
        return "om_start"
      _state["working_create_attempts"] += 1
      # The 2nd send_card is _ensure_card on the first ProgressEvent.
      # Fail it the way Lark did at 15:04:36 — the connection drops
      # before a response comes back. Later attempts (retry on
      # subsequent _update_working calls) succeed.
      if _state["working_create_attempts"] == 1:
        raise ConnectionError("Remote end closed connection without response")
      return "om_working"

    async def update_card(self, card_id, card):
      update_card_calls.append((card_id, card))
      return card_id

  class _MultiProgressAgent(_FakeAgent):
    async def run_turn(self, _prompt, on_event):
      def _emit():
        # `first=True` triggers _ensure_card — the failure we're
        # regressing against. Subsequent first=False events drive
        # _update_working, which is where the retry must happen.
        on_event(ProgressEvent(kind="tool", summary="Read", first=True))
        on_event(ProgressEvent(kind="tool", summary="Grep", first=False))
        on_event(ProgressEvent(kind="tool", summary="Edit", first=False))
        on_event(AnswerEvent("done"))
        on_event(DoneEvent(cost=0.0, usage={"input_tokens": 1}))
      await asyncio.to_thread(_emit)
      return 0.0, {"input_tokens": 1}

  channel = _FlakyChannel("oc_test")
  channel._messages = [
    IncomingMessage(
      event_type="im.message.receive_v1",
      chat_id="oc_test", sender_id="ou_user",
      message_id="om_msg", msg_type="text",
      text="hi", create_time="1",
    ),
    IncomingMessage(
      event_type="im.message.receive_v1",
      chat_id="oc_test", sender_id="ou_user",
      message_id="om_exit", msg_type="text",
      text="/exit", create_time="2",
    ),
  ]
  async def _recv(timeout=300):
    del timeout
    if channel._messages:
      return channel._messages.pop(0)
    return None
  channel.receive = _recv

  with mock.patch("nemo.agent.load_credentials", return_value={
    "app_id": "app_id", "app_secret": "app_secret", "email": "u@e.com",
  }), \
       mock.patch("nemo.agent.Database", _FakeDB), \
       mock.patch("nemo.agent.LarkChannel", return_value=channel), \
       mock.patch("nemo.agent.build_coding_agent", return_value=_MultiProgressAgent()), \
       mock.patch("nemo.agent._send_response", new=mock.AsyncMock()), \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    rc = asyncio.run(main_loop("oc_test", str(tmp_path), "claude-opus-4-6"))

  assert rc == 0
  # Before the fix: _ensure_card failed once on the first ProgressEvent
  # and _turn_card_id stayed None, so every subsequent _update_working
  # short-circuited at `if not _turn_card_id: return`. update_card was
  # never called for the rest of the turn.
  #
  # After the fix: the very next _update_working in the same first
  # event retries _ensure_card, the second send_card succeeds, and
  # later progress events PATCH that card via update_card.
  #
  # update_card is the right oracle here — send_card alone is noisy
  # because main_loop's `else` branch in the DoneEvent handler also
  # calls send_card (via _send_response) when the working card never
  # materialised, masking the bug if you only count send_card.
  assert len(update_card_calls) >= 1, (
    f"working card never recovered: 0 update_card calls "
    f"(send_card attempts={len(send_card_calls)}, "
    f"working_creates={_state['working_create_attempts']})"
  )


def test_working_card_retry_also_triggered_by_answer_event(tmp_path):
  """Companion to test_working_card_retries_on_first_create_failure:
  the retry path lives in `_update_working`, which is called from the
  ProgressEvent handler *and* from the AnswerEvent handler (and from
  RateLimit / Compact handlers). A turn whose only events are
  (first=True ProgressEvent that fails to create the card) + an
  AnswerEvent must still produce a working card via the AnswerEvent's
  retry — otherwise text-only turns (no follow-up tool calls) would
  stay silent after one transient blip.

  Without the fix: 0 successful card creates, 0 update_card calls.
  With the fix: AnswerEvent's _update_working calls _ensure_card again,
  the second send_card succeeds, and the AnswerEvent's update_card
  fires.
  """
  from nemo.channel import IncomingMessage

  send_card_calls: list[object] = []
  update_card_calls: list[object] = []
  _state = {"working_create_attempts": 0}

  class _FlakyChannel(_FakeChannel):
    async def send_card(self, _chat_id, card):
      send_card_calls.append(card)
      if len(send_card_calls) == 1:
        return "om_start"  # start card unrelated to the working card
      _state["working_create_attempts"] += 1
      if _state["working_create_attempts"] == 1:
        raise ConnectionError("Remote end closed connection without response")
      return "om_working"

    async def update_card(self, card_id, card):
      update_card_calls.append((card_id, card))
      return card_id

  class _OneToolThenAnswerAgent(_FakeAgent):
    async def run_turn(self, _prompt, on_event):
      def _emit():
        # Only ONE ProgressEvent — its _ensure_card fails. If the retry
        # only lived in subsequent ProgressEvents, this turn would
        # never recover. AnswerEvent's _update_working must pick it up.
        on_event(ProgressEvent(kind="tool", summary="Read", first=True))
        on_event(AnswerEvent("here is the answer"))
        on_event(DoneEvent(cost=0.0, usage={"input_tokens": 1}))
      await asyncio.to_thread(_emit)
      return 0.0, {"input_tokens": 1}

  channel = _FlakyChannel("oc_test")
  channel._messages = [
    IncomingMessage(
      event_type="im.message.receive_v1",
      chat_id="oc_test", sender_id="ou_user",
      message_id="om_msg", msg_type="text",
      text="hi", create_time="1",
    ),
    IncomingMessage(
      event_type="im.message.receive_v1",
      chat_id="oc_test", sender_id="ou_user",
      message_id="om_exit", msg_type="text",
      text="/exit", create_time="2",
    ),
  ]
  async def _recv(timeout=300):
    del timeout
    if channel._messages:
      return channel._messages.pop(0)
    return None
  channel.receive = _recv

  with mock.patch("nemo.agent.load_credentials", return_value={
    "app_id": "app_id", "app_secret": "app_secret", "email": "u@e.com",
  }), \
       mock.patch("nemo.agent.Database", _FakeDB), \
       mock.patch("nemo.agent.LarkChannel", return_value=channel), \
       mock.patch("nemo.agent.build_coding_agent", return_value=_OneToolThenAnswerAgent()), \
       mock.patch("nemo.agent._send_response", new=mock.AsyncMock()), \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    rc = asyncio.run(main_loop("oc_test", str(tmp_path), "claude-opus-4-6"))

  assert rc == 0
  # Either the ProgressEvent's in-line retry or the AnswerEvent's
  # retry must have produced an update_card call. Before the fix the
  # ProgressEvent has nothing to retry off of (first=True triggers
  # _ensure_card directly, which fails and never gets a second
  # chance) and the AnswerEvent's `if not _turn_card_id: return`
  # short-circuit kills the AnswerEvent path too — so update_card
  # stays at zero.
  assert len(update_card_calls) >= 1, (
    f"AnswerEvent didn't recover the working card after _ensure_card "
    f"failure: send_card attempts={len(send_card_calls)}, "
    f"working_creates={_state['working_create_attempts']}, "
    f"update_card={len(update_card_calls)}"
  )


def test_compact_events_set_banner_not_thinking_steps(tmp_path):
  """CompactStartedEvent / CompactNoticeEvent surface as a banner on the
  working card, not as ThinkingSteps inside the collapsible thinking
  timeline.

  Before the refactor: compaction was appended to ``_turn_steps`` as
  ``ThinkingStep("compact", …)`` and got grouped into the collapsible
  thinking panel — so a 10–60s silent compaction was invisible until
  the user expanded thinking. After: the latest compact message lives
  in a banner above the thinking panel (same slot as rate-limit) and
  ``_turn_steps`` carries no "compact" entries.

  Oracle: every update_card payload emitted during the turn must
  contain the compact-notice banner (grey markdown above the
  collapsible) and no ThinkingStep with the compact glyph inside the
  thinking panel.
  """
  from nemo.channel import IncomingMessage

  update_card_payloads: list[object] = []

  class _CapturingChannel(_FakeChannel):
    async def send_card(self, _chat_id, _card):
      return "om_working"

    async def update_card(self, card_id, card):
      update_card_payloads.append(card)
      return card_id

  class _CompactingAgent(_FakeAgent):
    async def run_turn(self, _prompt, on_event):
      def _emit():
        # first=True so the working card exists before the compact events
        # arrive — _update_working updates the existing card rather than
        # creating one mid-event.
        on_event(ProgressEvent(kind="tool", summary="Read", first=True))
        on_event(CompactStartedEvent(trigger="auto"))
        on_event(CompactNoticeEvent(
          trigger="auto", pre_tokens=180_000, post_tokens=60_000,
          duration_ms=15_000,
        ))
        on_event(AnswerEvent("after compact"))
        on_event(DoneEvent(cost=0.0, usage={"input_tokens": 1}))
      await asyncio.to_thread(_emit)
      return 0.0, {"input_tokens": 1}

  channel = _CapturingChannel("oc_test")
  channel._messages = [
    IncomingMessage(
      event_type="im.message.receive_v1",
      chat_id="oc_test", sender_id="ou_user",
      message_id="om_msg", msg_type="text",
      text="hi", create_time="1",
    ),
    IncomingMessage(
      event_type="im.message.receive_v1",
      chat_id="oc_test", sender_id="ou_user",
      message_id="om_exit", msg_type="text",
      text="/exit", create_time="2",
    ),
  ]
  async def _recv(timeout=300):
    del timeout
    if channel._messages:
      return channel._messages.pop(0)
    return None
  channel.receive = _recv

  with mock.patch("nemo.agent.load_credentials", return_value={
    "app_id": "app_id", "app_secret": "app_secret", "email": "u@e.com",
  }), \
       mock.patch("nemo.agent.Database", _FakeDB), \
       mock.patch("nemo.agent.LarkChannel", return_value=channel), \
       mock.patch("nemo.agent.build_coding_agent", return_value=_CompactingAgent()), \
       mock.patch("nemo.agent._send_response", new=mock.AsyncMock()), \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    rc = asyncio.run(main_loop("oc_test", str(tmp_path), "claude-opus-4-6"))

  assert rc == 0
  # At least one update_card after CompactStartedEvent must carry the
  # grey banner. After CompactNoticeEvent it gets replaced with the
  # post-fact summary that mentions tokens or duration. The very last
  # working-phase update should carry the summary banner.
  working_updates = [
    c for c in update_card_payloads
    if c.get("header", {}).get("title", {}).get("content", "").startswith(
      ("Working", "Stopping", "Stopped"))
  ]
  assert working_updates, "no working-phase update_card calls"
  banners_seen = []
  for card in working_updates:
    for el in card["body"]["elements"]:
      if el.get("tag") == "markdown" and "<font color='grey'>" in el.get("content", ""):
        banners_seen.append(el["content"])
  assert banners_seen, (
    f"no grey compact-notice banner in any of {len(working_updates)} "
    f"working_updates"
  )
  # Last banner is the post-fact summary — it must include either a
  # token count or a duration that came from CompactNoticeEvent.
  last_banner = banners_seen[-1]
  assert ("180" in last_banner or "60" in last_banner or "15" in last_banner), (
    f"last compact banner missing post-fact metadata: {last_banner!r}"
  )
  # No collapsible_thinking panel anywhere may contain the compaction
  # glyph — the banner must NOT also leak into the thinking timeline.
  for card in update_card_payloads:
    for el in card["body"]["elements"]:
      if el.get("tag") == "collapsible_panel":
        assert "🗜" not in repr(el), (
          f"compact glyph leaked into collapsible thinking: {el!r}"
        )


def test_stale_leak_notice_leaves_visible_breadcrumb(tmp_path):
  """SDK #788 recovery must NOT be silent: a StaleLeakNoticeEvent emitted
  mid-turn (right before the reconnect-with-resume) must surface as a
  visible breadcrumb in the turn card so the user can see what happened.
  """
  from nemo.channel import IncomingMessage
  from nemo.turn import StaleLeakNoticeEvent

  card_payloads: list[object] = []

  class _CapturingChannel(_FakeChannel):
    async def send_card(self, _chat_id, card):
      card_payloads.append(card)
      return "om_working"

    async def update_card(self, card_id, card):
      card_payloads.append(card)
      return card_id

  class _StaleLeakAgent(_FakeAgent):
    async def run_turn(self, _prompt, on_event):
      def _emit():
        on_event(ProgressEvent(kind="tool", summary="Read", first=True))
        # Leak detected -> breadcrumb -> (reconnect+retry happens in the
        # reconnect layer; here we just continue with the recovered answer).
        on_event(StaleLeakNoticeEvent(task_id="task_abc123"))
        on_event(AnswerEvent("recovered answer"))
        on_event(DoneEvent(cost=0.0, usage={"input_tokens": 1}))
      await asyncio.to_thread(_emit)
      return 0.0, {"input_tokens": 1}

  channel = _CapturingChannel("sl_test")
  channel._messages = [
    IncomingMessage(
      event_type="im.message.receive_v1",
      chat_id="sl_test", sender_id="ou_user",
      message_id="om_msg", msg_type="text",
      text="hi", create_time="1",
    ),
    IncomingMessage(
      event_type="im.message.receive_v1",
      chat_id="sl_test", sender_id="ou_user",
      message_id="om_exit", msg_type="text",
      text="/exit", create_time="2",
    ),
  ]
  async def _recv(timeout=300):
    del timeout
    if channel._messages:
      return channel._messages.pop(0)
    return None
  channel.receive = _recv

  with mock.patch("nemo.agent.load_credentials", return_value={
    "app_id": "app_id", "app_secret": "app_secret", "email": "u@e.com",
  }), \
       mock.patch("nemo.agent.Database", _FakeDB), \
       mock.patch("nemo.agent.LarkChannel", return_value=channel), \
       mock.patch("nemo.agent.build_coding_agent", return_value=_StaleLeakAgent()), \
       mock.patch("nemo.agent._send_response", new=mock.AsyncMock()), \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    rc = asyncio.run(main_loop("sl_test", str(tmp_path), "claude-opus-4-6"))

  assert rc == 0
  assert any(
    "SDK #788" in repr(card) and "task_abc123" in repr(card)
    for card in card_payloads
  ), "stale-leak breadcrumb (SDK #788 + task id) not visible in any card"
  # Sanity: the StaleLeakNoticeEvent type itself is part of TurnEvent.
  assert StaleLeakNoticeEvent(task_id="x").task_id == "x"


# ---------------------------------------------------------------------------
# /btw — ephemeral side question delivery
# ---------------------------------------------------------------------------


class _FakeBtwChannel:
  def __init__(self):
    self.token = "tok"
    self.cards = []

  async def send_card(self, chat_id, card):
    self.cards.append((chat_id, card))
    return "msg-id"


class _FakeBtwAgent:
  def __init__(self, answer="", exc=None):
    self._answer = answer
    self._exc = exc
    self.calls = []

  async def side_question(self, question, sdk_session_id):
    self.calls.append((question, sdk_session_id))
    if self._exc is not None:
      raise self._exc
    return self._answer


def test_handle_btw_posts_ephemeral_card_without_db_record():
  """The answer card must NOT be persisted (no db arg is even threaded
  in) so it can never re-enter the agent-readable chat history."""
  from nemo.agent import _handle_btw

  ch = _FakeBtwChannel()
  ag = _FakeBtwAgent(answer="It was config.toml.")
  asyncio.run(_handle_btw(ch, "chat-1", ag, "live-sess", "which file?"))

  assert ag.calls == [("which file?", "live-sess")]
  assert len(ch.cards) == 1
  _chat, card = ch.cards[0]
  blob = repr(card)
  assert "It was config.toml." in blob
  assert "Ephemeral" in blob
  assert "btw" in blob


def test_handle_btw_unsupported_falls_back_to_note():
  from nemo.agent import _handle_btw

  ch = _FakeBtwChannel()
  ag = _FakeBtwAgent(answer="")  # adapter doesn't support side questions
  asyncio.run(_handle_btw(ch, "c", ag, "", "q"))

  assert len(ch.cards) == 1
  assert "isn't supported" in repr(ch.cards[0][1])


def test_handle_btw_swallows_adapter_exception():
  from nemo.agent import _handle_btw

  ch = _FakeBtwChannel()
  ag = _FakeBtwAgent(exc=RuntimeError("kaboom"))
  asyncio.run(_handle_btw(ch, "c", ag, "s", "q"))

  assert len(ch.cards) == 1
  assert "btw failed" in repr(ch.cards[0][1])


# ---------------------------------------------------------------------------
# /restart and /upgrade relaunch with a routable --model
# ---------------------------------------------------------------------------

def test_restart_model_arg_reverses_preset_remote_to_name():
  """Regression: a preset switch leaves `model` holding the resolved remote
  id (deepseek-v4-pro[1m]) and `_endpoint_key` the preset URL. /restart fed
  that remote id straight to --model, which startup can't resolve as a
  preset → no endpoint → every turn (and the forked /btw CLI) failed
  "model not found". The relaunch arg must reverse to the preset NAME."""
  arg = _restart_model_arg(
    "deepseek-v4-pro[1m]", "https://api.deepseek.com/anthropic", "claude")
  assert arg == "deepseek-v4-pro"


def test_restart_model_arg_passes_plain_default_model_through():
  """No preset active (default endpoint, empty key) → the plain model id is
  already a valid --model, pass it through untouched."""
  assert _restart_model_arg("claude-opus-4-7", "", "claude") == "claude-opus-4-7"


# ---------------------------------------------------------------------------
# Stop/esc must interrupt the turn, never kill the daemon
# ---------------------------------------------------------------------------

class _InterruptAgent:
  """Mirrors ClaudeCodingAgent.interrupt → cancel() then client.interrupt()."""

  def __init__(self, raises: Exception | None = None):
    self._raises = raises

  async def interrupt(self) -> None:
    if self._raises is not None:
      raise self._raises


def test_interrupt_and_drain_swallows_turn_self_cancellation():
  """Regression for the oc_03b7c9b… daemon death: a Stop pressed while the
  turn was in a #788 stale-leak-resume reconnect made run_turn_with_reconnect
  raise CancelledError (it aborts the reconnect loop when _cancelled is set).
  The old `except Exception` couldn't catch CancelledError, so it bubbled to
  the loop-level `except asyncio.CancelledError` and shut the daemon down.
  The helper must swallow the turn task's own cancellation → 'aborted', and
  must NOT re-raise (which would propagate to the loop and stop the daemon)."""
  async def _scenario():
    async def _turn():
      raise asyncio.CancelledError("SDK turn cancelled")
    sdk_task = asyncio.ensure_future(_turn())
    return await _interrupt_and_drain(_InterruptAgent(), sdk_task, timeout=5)

  # Must return normally (not raise) — the daemon stays alive.
  assert asyncio.run(_scenario()) == "aborted"


def test_interrupt_and_drain_reports_clean_on_normal_finish():
  """Stop during an ordinary turn: client.interrupt() lets the turn end on
  its own, the task completes normally → 'clean', daemon lives."""
  async def _scenario():
    async def _turn():
      return (1.0, {})
    sdk_task = asyncio.ensure_future(_turn())
    return await _interrupt_and_drain(_InterruptAgent(), sdk_task, timeout=5)

  assert asyncio.run(_scenario()) == "clean"


def test_interrupt_and_drain_forces_cancel_when_interrupt_raises():
  """If interrupt() itself raises, the helper force-cancels the turn task
  and reports 'forced' — still no propagation that could kill the daemon."""
  async def _scenario():
    ev = asyncio.Event()

    async def _turn():
      await ev.wait()  # hang until force-cancelled

    sdk_task = asyncio.ensure_future(_turn())
    await asyncio.sleep(0)  # let the turn task start and block
    status = await _interrupt_and_drain(
      _InterruptAgent(raises=RuntimeError("boom")), sdk_task, timeout=5)
    try:
      await sdk_task
    except asyncio.CancelledError:
      pass
    return status

  assert asyncio.run(_scenario()) == "forced"


def _recall_creds():
  return {"app_id": "app_id", "app_secret": "app_secret",
          "email": "user@example.com"}


class _RecallReplayChannel(_QueuedChannel):
  """Records (without replaying) push_back'd messages and update_card calls,
  so a recall submit can be asserted without spinning the injected turn."""

  def __init__(self, *a, **kw):
    super().__init__(*a, **kw)
    self.pushed = []
    self.update_card_calls = []

  def push_back(self, message):
    self.pushed.append(message)

  async def update_card(self, card_id, card):
    self.update_card_calls.append((card_id, card))
    return card_id


def _recall_submit_msgs(uuid):
  return [
    IncomingMessage(
      event_type="card.action.trigger", chat_id="oc_test",
      operator_id="ou_user", sender_id="ou_user",
      message_id="om_session_picker",
      action_value={"action": f"session_recall:{uuid}", "chat_id": "oc_test"},
      create_time="1"),
    IncomingMessage(
      event_type="im.message.receive_v1", chat_id="oc_test",
      sender_id="ou_user", message_id="om_exit", msg_type="text",
      text="/exit", create_time="2"),
  ]


def _run_recall_loop(tmp_path, agent, queued):
  with mock.patch.dict(os.environ, {"HOME": str(tmp_path / "home")}), \
       mock.patch("nemo.agent.load_credentials", return_value=_recall_creds()), \
       mock.patch("nemo.agent.Database", _FakeDB), \
       mock.patch("nemo.agent.LarkChannel", return_value=queued), \
       mock.patch("nemo.agent.build_coding_agent", return_value=agent), \
       mock.patch("nemo.agent._send_response", new=mock.AsyncMock()), \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    return asyncio.run(
      main_loop("oc_test", str(tmp_path), "claude-opus-4-7", agent="claude"))


def test_session_picker_submit_recalls_with_digest(tmp_path):
  """Submitting the /session recall picker fires a card.action.trigger with
  action='session_recall:<uuid>'. The daemon must run the digest sub-session,
  inject ONLY the compact summary (+ a pointer to the transcript) into the
  live agent's context, and lock the picker card."""
  from nemo import sessions

  transcript = tmp_path / "rollout.jsonl"
  transcript.write_text(
    '{"type":"user","message":{"role":"user","content":"hi"}}\n')
  uuid = "01fe69c7-5793-4ad7-9ba6-7d1aa1e01f90"
  info = sessions.SessionInfo(
    uuid=uuid, agent="claude", path=str(transcript),
    mtime=transcript.stat().st_mtime, first_user_text="hi",
    model="claude-opus-4-7")

  class _DigestAgent(_FakeAgent):
    def __init__(self):
      super().__init__()
      self.digest_calls = []

    async def digest_transcript(self, path, fmt_hint):
      self.digest_calls.append((path, fmt_hint))
      return "### Working on\n- fixing the parser"

  agent = _DigestAgent()
  queued = _RecallReplayChannel("oc_test", _recall_submit_msgs(uuid))

  with mock.patch("nemo.sessions.list_sessions", return_value=[info]):
    rc = _run_recall_loop(tmp_path, agent, queued)

  assert rc == 0
  assert agent.digest_calls == [(str(transcript), mock.ANY)]
  assert queued.pushed, "recall must inject an internal turn"
  injected = queued.pushed[0]
  assert injected.is_internal
  assert injected.text.startswith(_RECALL_PROMPT_PREFIX)
  # The compact summary is carried in (NOT raw JSONL), plus the on-demand
  # read pointer to the transcript path.
  assert "### Working on" in injected.text
  assert str(transcript) in injected.text
  # And the picker card is locked to a no-form confirmation.
  locks = [c for c in queued.update_card_calls if c[0] == "om_session_picker"]
  assert locks, queued.update_card_calls
  import json as _json
  assert "select_static" not in _json.dumps(locks[-1][1], ensure_ascii=False)


def test_session_recall_falls_back_to_inline_when_no_digest(tmp_path):
  """When the agent can't digest (returns ""), recall falls back to handing
  the live agent the path to read itself — the original behavior."""
  from nemo import sessions

  transcript = tmp_path / "rollout.jsonl"
  transcript.write_text(
    '{"type":"user","message":{"role":"user","content":"hi"}}\n')
  uuid = "01fe69c7-5793-4ad7-9ba6-7d1aa1e01f90"
  info = sessions.SessionInfo(
    uuid=uuid, agent="claude", path=str(transcript),
    mtime=transcript.stat().st_mtime, first_user_text="hi", model="m")

  class _NoDigestAgent(_FakeAgent):
    async def digest_transcript(self, path, fmt_hint):
      return ""

  agent = _NoDigestAgent()
  queued = _RecallReplayChannel("oc_test", _recall_submit_msgs(uuid))

  with mock.patch("nemo.sessions.list_sessions", return_value=[info]):
    rc = _run_recall_loop(tmp_path, agent, queued)

  assert rc == 0
  assert queued.pushed
  injected = queued.pushed[0]
  assert injected.text.startswith(_RECALL_PROMPT_PREFIX)
  assert str(transcript) in injected.text
  # Inline prompt tells the agent to read the file itself.
  assert "Use your Read tool" in injected.text
  assert "### Working on" not in injected.text


def test_session_picker_no_arg_sends_picker_card(tmp_path):
  """`/session recall` with no uuid emits the dropdown picker card."""
  from nemo import sessions

  transcript = tmp_path / "rollout.jsonl"
  transcript.write_text(
    '{"type":"user","message":{"role":"user","content":"hi"}}\n')
  info = sessions.SessionInfo(
    uuid="01fe69c7-5793-4ad7-9ba6-7d1aa1e01f90", agent="claude",
    path=str(transcript), mtime=transcript.stat().st_mtime,
    first_user_text="fix the bug", model="m")

  sent_cards = []

  class _CardChannel(_RecallReplayChannel):
    async def send_card(self, chat_id, card):
      sent_cards.append(card)
      return "om_picker_sent"

  agent = _FakeAgent()
  queued = _CardChannel("oc_test", [
    IncomingMessage(
      event_type="im.message.receive_v1", chat_id="oc_test",
      sender_id="ou_user", message_id="om1", msg_type="text",
      text="/session recall", create_time="1"),
    IncomingMessage(
      event_type="im.message.receive_v1", chat_id="oc_test",
      sender_id="ou_user", message_id="om_exit", msg_type="text",
      text="/exit", create_time="2"),
  ])

  with mock.patch("nemo.sessions.list_sessions", return_value=[info]):
    rc = _run_recall_loop(tmp_path, agent, queued)

  assert rc == 0
  import json as _json
  picker = next(
    (c for c in sent_cards
     if c.get("header", {}).get("title", {}).get("content") == "Recall Session"),
    None)
  assert picker is not None, sent_cards
  blob = _json.dumps(picker, ensure_ascii=False)
  # Per-session Recall button carrying the uuid discriminator; the full
  # first-prompt preview is shown (radio-style rows, no dropdown/form).
  assert "session_recall:01fe69c7-5793-4ad7-9ba6-7d1aa1e01f90" in blob
  assert "fix the bug" in blob
  assert "select_static" not in blob


# ---------------------------------------------------------------------------
# Lark-free integration: real claude-cli adapter through the real main_loop.
#
# The full daemon e2e (scripts/e2e_test.py) needs Lark (the daemon is a Lark
# bot — preflight token, group, cards). But the adapter is channel-agnostic, so
# we exercise the WHOLE turn path (run_turn → on_event → card rendering →
# session store) with a fake channel + fake DB and NO Lark. Gated behind
# NEMO_REAL_SDK=1 because it spawns the real claude CLI and bills a real turn.
# ---------------------------------------------------------------------------

class _RecordingDB(_FakeDB):
  def __init__(self, _project_dir):
    super().__init__(_project_dir)
    self.stored_sessions = []

  def set_sdk_session_id(self, _chat_id, sdk_session_id, _agent, _endpoint_key=""):
    self.stored_sessions.append(sdk_session_id)


class _CapturingChannel(_FakeChannel):
  """Fake channel: feeds one real prompt, then holds /exit until the turn's
  answer is actually rendered (a real turn takes seconds, so returning /exit
  eagerly would stop the loop mid-turn). Records cards/texts."""
  def __init__(self, _chat_id):
    super().__init__(_chat_id)
    self.cards = []
    self.texts = []
    self.prompt = ""
    self._sent_prompt = False
    self._exited = False
    self._answered = False

  def _msg(self, text, mid):
    from nemo.channel import IncomingMessage
    return IncomingMessage(
      event_type="im.message.receive_v1", chat_id="oc_test",
      sender_id="ou_user", message_id=mid, msg_type="text",
      text=text, create_time="1")

  async def receive(self, timeout=300):
    del timeout
    if not self._sent_prompt:
      self._sent_prompt = True
      return self._msg(self.prompt, "om_prompt")
    if not self._exited:
      # Wait for the turn to render an answer before ending the loop.
      for _ in range(300):  # up to ~60s
        if self._answered:
          break
        await asyncio.sleep(0.2)
      self._exited = True
      return self._msg("/exit", "om_exit")
    return None

  async def send_card(self, chat_id, card):
    self.cards.append(card)
    return f"om_card_{len(self.cards)}"

  async def update_card(self, message_id, card):
    self.cards.append(card)
    return message_id

  async def send_text(self, chat_id, text):
    self.texts.append(text)
    if text.strip():
      self._answered = True   # the turn's answer reached the channel
    return f"om_text_{len(self.texts)}"


@pytest.mark.realsdk
@pytest.mark.skipif(
  os.environ.get("NEMO_REAL_SDK") != "1",
  reason="real-SDK integration: set NEMO_REAL_SDK=1 (needs claude CLI + auth)",
)
def test_claude_cli_turn_through_main_loop_no_lark(tmp_path):
  """One real claude-cli turn driven through the real orchestration, Lark-free:
  asserts the answer reaches the channel and the session id is persisted."""
  import subprocess
  from nemo.claude_cli_agent import ClaudeCliCodingAgent
  subprocess.run(["git", "init", "-q"], cwd=str(tmp_path))  # real project dir
  marker = "pineapple7392"
  db = _RecordingDB(str(tmp_path))
  channel = _CapturingChannel("oc_test")
  channel.prompt = f"Reply with exactly this token and nothing else: {marker}"

  def _build(*_a, **_k):
    return ClaudeCliCodingAgent(
      {}, "oc_test", db, channel, permission_mode="bypassPermissions")

  with mock.patch("nemo.agent.load_credentials", return_value={
        "app_id": "a", "app_secret": "s", "email": "u@example.com"}), \
       mock.patch("nemo.agent.Database", return_value=db), \
       mock.patch("nemo.agent.LarkChannel", return_value=channel), \
       mock.patch("nemo.agent.build_coding_agent", side_effect=_build), \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("signal.signal"):
    rc = asyncio.run(main_loop(
      "oc_test", str(tmp_path),
      os.environ.get("NEMO_REAL_SDK_MODEL", "claude-haiku-4-5"),
      agent="claude-cli"))

  assert rc == 0
  rendered = " ".join(_card_json(c) for c in channel.cards) + " ".join(channel.texts)
  assert marker in rendered, f"answer not rendered to channel; got: {rendered[:300]}"
  # Session id persisted for cross-restart resume.
  assert any(s for s in db.stored_sessions), "no sdk session id stored"


def _card_json(card):
  import json
  try:
    return json.dumps(card, ensure_ascii=False)
  except Exception:
    return str(card)
