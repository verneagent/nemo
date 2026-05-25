"""Tests for abstract Channel and CodingAgent interfaces."""

import asyncio

from nemo.channel import Channel, IncomingMessage
from nemo.coding_agent import CodingAgent
from nemo.turn import ProgressEvent, AnswerEvent, DoneEvent


# ---------------------------------------------------------------------------
# IncomingMessage dataclass
# ---------------------------------------------------------------------------

def test_incoming_message_defaults():
  msg = IncomingMessage()
  assert msg.event_type == ""
  assert msg.chat_id == ""
  assert msg.text == ""
  assert msg.mentions == []
  assert msg.action_value == {}
  assert msg.raw == {}


def test_incoming_message_fields():
  msg = IncomingMessage(
    event_type="message",
    chat_id="oc_1",
    sender_id="ou_1",
    text="hello",
    msg_type="text",
  )
  assert msg.event_type == "message"
  assert msg.chat_id == "oc_1"
  assert msg.text == "hello"


# ---------------------------------------------------------------------------
# Concrete Channel implementation for testing
# ---------------------------------------------------------------------------

class FakeChannel(Channel):
  def __init__(self):
    self.messages = []
    self.sent_cards = []
    self.updated_cards = []
    self.sent_texts = []
    self._permission_active = False

  async def receive(self, timeout=300):
    if self.messages:
      return self.messages.pop(0)
    return None

  async def send_card(self, chat_id, card):
    self.sent_cards.append((chat_id, card))
    return f"om_{len(self.sent_cards)}"

  def push_back(self, message):
    self.messages.insert(0, message)

  async def update_card(self, message_id, card):
    self.updated_cards.append((message_id, card))
    return message_id

  async def send_text(self, chat_id, text):
    self.sent_texts.append((chat_id, text))
    return f"om_text_{len(self.sent_texts)}"

  async def download_image(self, message_id, image_key):
    return f"/tmp/{image_key}.png"

  async def download_file(self, message_id, file_key, file_name=""):
    return f"/tmp/{file_name or file_key}"

  async def add_reaction(self, message_id, emoji_type):
    pass

  async def start(self):
    pass

  async def stop(self):
    pass

  async def get_bot_id(self):
    return "ou_bot"

  async def get_chat_members(self, chat_id):
    return [{"member_id": "ou_1"}, {"member_id": "ou_bot"}]

  @property
  def permission_active(self):
    return self._permission_active

  @permission_active.setter
  def permission_active(self, active):
    self._permission_active = active


def test_fake_channel_implements_interface():
  """FakeChannel should be a valid Channel subclass."""
  ch = FakeChannel()
  assert isinstance(ch, Channel)


def test_channel_send_card():
  async def _run():
    ch = FakeChannel()
    msg_id = await ch.send_card("oc_1", {"schema": "2.0"})
    assert msg_id.startswith("om_")
    assert len(ch.sent_cards) == 1
  asyncio.run(_run())


def test_channel_receive_empty():
  async def _run():
    ch = FakeChannel()
    result = await ch.receive(timeout=0.01)
    assert result is None
  asyncio.run(_run())


def test_channel_receive_message():
  async def _run():
    ch = FakeChannel()
    ch.messages.append(IncomingMessage(text="hello"))
    msg = await ch.receive()
    assert msg.text == "hello"
  asyncio.run(_run())


def test_channel_update_card():
  async def _run():
    ch = FakeChannel()
    await ch.update_card("om_1", {"body": {}})
    assert len(ch.updated_cards) == 1
  asyncio.run(_run())


def test_channel_send_text():
  async def _run():
    ch = FakeChannel()
    msg_id = await ch.send_text("oc_1", "hi")
    assert msg_id.startswith("om_text_")
  asyncio.run(_run())


def test_channel_get_bot_id():
  async def _run():
    ch = FakeChannel()
    bot_id = await ch.get_bot_id()
    assert bot_id == "ou_bot"
  asyncio.run(_run())


def test_channel_get_chat_members():
  async def _run():
    ch = FakeChannel()
    members = await ch.get_chat_members("oc_1")
    assert len(members) == 2
  asyncio.run(_run())


# ---------------------------------------------------------------------------
# Concrete CodingAgent implementation for testing
# ---------------------------------------------------------------------------

class FakeAgent(CodingAgent):
  def __init__(self):
    self.turns = []
    self._interrupted = False

  async def run_turn(self, prompt, on_event):
    self.turns.append(prompt)
    on_event(ProgressEvent(kind="tool", summary="Read /a/b.py", first=True))
    on_event(AnswerEvent(text="response"))
    on_event(DoneEvent(cost=0.01, usage={"input_tokens": 100}))
    return 0.01, {"input_tokens": 100}

  async def interrupt(self):
    self._interrupted = True

  async def reset(self, project_dir, model, resume=""):
    self.turns.clear()

  async def start(self, project_dir, model, resume=""):
    pass

  async def stop(self):
    pass


def test_fake_agent_implements_interface():
  agent = FakeAgent()
  assert isinstance(agent, CodingAgent)


def test_agent_run_turn():
  async def _run():
    agent = FakeAgent()
    events = []
    await agent.run_turn("fix the bug", lambda e: events.append(e))
    assert len(events) == 3
    assert isinstance(events[0], ProgressEvent)
    assert isinstance(events[1], AnswerEvent)
    assert events[1].text == "response"
    assert isinstance(events[2], DoneEvent)
    assert agent.turns == ["fix the bug"]
  asyncio.run(_run())


def test_agent_interrupt():
  async def _run():
    agent = FakeAgent()
    assert not agent._interrupted
    await agent.interrupt()
    assert agent._interrupted
  asyncio.run(_run())


def test_agent_reset():
  async def _run():
    agent = FakeAgent()
    agent.turns.append("old")
    await agent.reset("/tmp", "claude")
    assert agent.turns == []
  asyncio.run(_run())


# ---------------------------------------------------------------------------
# fork() — read-only forked sub-session (Claude-only)
# ---------------------------------------------------------------------------

def test_project_slug_matches_cli_rule():
  """The CLI slug is realpath + every non-alphanumeric char → '-' (NOT a
  naive '/'→'-'). _seed_fork_transcript and trailing_note depend on matching
  it byte-for-byte. Pinned against a real observed slug."""
  from nemo.claude_agent import _project_slug
  # '.' and '_' both collapse to '-'; '/.' becomes '--'.
  assert _project_slug("/Users/me/.foo_bar/baz") == "-Users-me--foo-bar-baz"
  # Mirrors an actually-observed ~/.claude/projects entry.
  assert (_project_slug("/Users/dinghaozeng/.supacode/repos/fived/legal-doc")
          == "-Users-dinghaozeng--supacode-repos-fived-legal-doc")


def test_base_agent_does_not_support_fork():
  """Non-Claude adapters inherit the unsupported defaults."""
  agent = FakeAgent()
  assert agent.supports_fork() is False
  async def _run():
    assert await agent.fork("sess", "/proj", "claude") is None
  asyncio.run(_run())


def _make_fork_agent():
  from nemo.claude_agent import ClaudeCodingAgent
  a = ClaudeCodingAgent({}, "oc_test", None, FakeChannel(), read_only=True)
  a._scratch_dir = "/tmp/nemo_fork_scratch_test"  # normally tempfile.mkdtemp
  return a


def test_claude_supports_fork():
  from nemo.claude_agent import ClaudeCodingAgent
  a = ClaudeCodingAgent({}, "oc_test", None, FakeChannel())
  assert a.supports_fork() is True


def test_fork_options_disallow_file_mutators_keep_investigative():
  a = _make_fork_agent()
  opts = a._build_fork_options("/Users/me/proj", "claude-x")
  for t in ("Write", "Edit", "NotebookEdit", "Agent", "Skill"):
    assert t in opts.disallowed_tools, f"{t} must be disallowed in a fork"
  assert "Write" not in opts.allowed_tools
  # Investigative tools stay so the fork can actually do work.
  assert "Bash" in opts.allowed_tools
  assert "Read" in opts.allowed_tools
  assert "Grep" in opts.allowed_tools


def test_fork_options_sandbox_and_scratch_cwd():
  """The project must NOT be cwd (cwd is sandbox-writable); it lives outside
  the workspace so the sandbox blocks writes to it while reads still work."""
  a = _make_fork_agent()
  opts = a._build_fork_options("/Users/me/proj", "claude-x")
  assert opts.sandbox == {"enabled": True, "autoAllowBashIfSandboxed": True}
  assert opts.cwd == a._scratch_dir
  assert opts.cwd != "/Users/me/proj"
  assert opts.permission_mode == "bypassPermissions"


def test_fork_options_fork_session_only_from_parent():
  """fork_session=True only while resuming the parent transcript; once the
  fork's own session id is live we must not re-fork each turn/reconnect."""
  a = _make_fork_agent()
  a._fork_parent_id = "parent_sess"
  forked = a._build_fork_options("/p", "m", resume="parent_sess")
  assert forked.fork_session is True
  assert forked.resume == "parent_sess"
  cont = a._build_fork_options("/p", "m", resume="fork_own_sess")
  assert cont.fork_session is False
  assert cont.resume == "fork_own_sess"
