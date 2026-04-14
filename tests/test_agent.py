"""Tests for nemo.agent main loop behavior."""

import asyncio
from unittest import mock

from nemo.agent import main_loop
from nemo.turn import AnswerEvent, DoneEvent


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

  async def update_status(self, _model, _state):
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


def test_codex_provider_rejects_claude_model_switch(tmp_path):
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
      main_loop("oc_test", str(tmp_path), "gpt-5-codex", provider="codex")
    )

  assert result == 0
  agent.reset.assert_not_awaited()
  send_response.assert_any_await(
    queued,
    "oc_test",
    "Model **claude-sonnet-4-6** is not supported by provider **codex**.",
    mock.ANY,
  )
