"""Tests for nemo.agent main loop behavior."""

import asyncio
import urllib.error
from unittest import mock

from nemo.agent import _update_done_card_with_fallback, main_loop
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
  call = next(
    c for c in send_response.await_args_list
    if "claude-sonnet-4-6" in c.args[2]
  )
  body = call.args[2]
  assert "Unknown model" in body
  assert "codex" in body
  assert "gpt-5-codex" in body  # available list shown


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


def _run_fallback(channel, final_text="answer body"):
  """Drive _update_done_card_with_fallback with identity await_channel."""
  register_calls = []
  return _update_done_card_with_fallback(
    channel=channel,
    chat_id="oc_test",
    turn_card_id="om_card0",
    final_text=final_text,
    thinking=[],
    elapsed=1,
    usage={"input_tokens": 1},
    session_id="sess_x",
    await_channel=lambda x: x,
    register_msg=lambda msg_id, chat_id: register_calls.append((msg_id, chat_id)),
  )


def test_done_card_full_body_success_single_update():
  channel = _FakeUpdateChannel(outcomes=["om_card1"])
  result = _run_fallback(channel)
  assert result == "om_card1"
  assert len(channel.calls) == 1


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
