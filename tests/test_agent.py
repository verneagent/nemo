"""Tests for nemo.agent main loop behavior."""

import asyncio
from unittest import mock

from nemo.agent import main_loop
from nemo.lark.events import LarkEvent
from nemo.turn import DoneEvent, TextEvent


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


class _FakeEvents:
  permission_active = False

  def __init__(self, *_args, **_kwargs):
    self._count = 0

  async def connect(self):
    pass

  async def close(self):
    pass

  def push_back(self, _event):
    pass

  async def next_message(self, timeout=300):
    del timeout
    if self._count == 0:
      self._count += 1
      return LarkEvent(
        event_type="im.message.receive_v1",
        chat_id="oc_test",
        sender_id="ou_user",
        message_id="om_src",
        msg_type="text",
        text="hi",
        create_time="1",
      )
    raise KeyboardInterrupt()


class _FakeSDKThread:
  def start(self):
    pass

  async def create_client(self, _options):
    pass

  async def close_client(self):
    pass

  async def reconnect(self, _options):
    pass

  async def interrupt(self):
    pass

  def stop(self):
    pass

  async def run_turn_with_reconnect(
    self, _prompt, on_event, stale_tasks=None, options=None, max_attempts=3,
  ):
    del stale_tasks, options, max_attempts
    on_event(TextEvent("hi! how can i help?"))
    on_event(DoneEvent(cost=0.01, usage={"input_tokens": 1}))
    return 0.01, {"input_tokens": 1}


def test_text_only_turn_clears_thinking_reaction(tmp_path):
  remove_reaction = mock.Mock()

  with mock.patch("nemo.agent.load_credentials", return_value={
    "app_id": "app_id",
    "app_secret": "app_secret",
    "email": "user@example.com",
  }), \
       mock.patch("nemo.agent.Database", _FakeDB), \
       mock.patch("nemo.agent.LarkEventStream", _FakeEvents), \
       mock.patch("nemo.agent._build_sdk_options", return_value={}), \
       mock.patch("nemo.sdk_thread.SDKThread", _FakeSDKThread), \
       mock.patch("nemo.lark.auth.get_token", return_value="tok"), \
       mock.patch("nemo.lark.api.lookup_open_id_by_email", return_value="ou_user"), \
       mock.patch("nemo.lark.api.get_bot_info", return_value={"open_id": "ou_bot"}), \
       mock.patch("nemo.lark.api.get_chat_info", return_value={"owner_id": "ou_bot"}), \
       mock.patch("nemo.lark.api.get_chat_members", return_value=[]), \
       mock.patch("nemo.lark.api.send_card", side_effect=["om_start", "om_reply"]), \
       mock.patch("nemo.lark.api.add_reaction", return_value="r_thinking") as add_reaction, \
       mock.patch("nemo.lark.api.remove_reaction", remove_reaction), \
       mock.patch("nemo.workspace.ensure_workspace_tag"), \
       mock.patch("nemo.workspace.evict_existing"), \
       mock.patch("nemo.workspace.claim_group"), \
       mock.patch("nemo.workspace.release_group"), \
       mock.patch("nemo.group_config.load_config", return_value={}), \
       mock.patch("nemo.config.load_relay_config", return_value=("", "")), \
       mock.patch("nemo.status_tab.update_status"), \
       mock.patch("signal.signal"):
    result = asyncio.run(main_loop("oc_test", str(tmp_path), "claude-opus-4-6"))

  assert result == 0
  add_reaction.assert_called_once_with("tok", "om_src", "THINKING")
  remove_reaction.assert_called_once_with("tok", "om_src", "r_thinking")
