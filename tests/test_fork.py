"""Tests for nemo.fork — the /fork ForkManager (open / route / close).

Deterministic: fakes stand in for the channel and the read-only fork agent,
so these cover the manager's logic without a live SDK. End-to-end behaviour
(real fork_session branch + sandbox read-only) is validated separately.
"""

import asyncio

from nemo.fork import ForkManager
from nemo.turn import AnswerEvent, DoneEvent, ProgressEvent


class _FakeForkAgent:
  def __init__(self):
    self.turns = []
    self.stopped = False

  async def run_turn(self, prompt, on_event):
    self.turns.append(prompt)
    # Emit from a worker thread — mirrors the real adapter (the renderer
    # marshals channel calls back to the loop via run_coroutine_threadsafe,
    # which would deadlock if events fired on the loop thread itself).
    def _emit():
      on_event(ProgressEvent(kind="tool", summary="Read foo.py", first=True))
      on_event(AnswerEvent("fork says hi"))
      on_event(DoneEvent(cost=0.0, usage={}, session_id="fsid"))
    await asyncio.to_thread(_emit)
    return 0.0, {}

  async def stop(self):
    self.stopped = True


class _FakeMainAgent:
  def __init__(self, supports=True):
    self._supports = supports
    self.fork_agents = []
    self.fork_calls = []

  def supports_fork(self):
    return self._supports

  async def fork(self, parent_sid, project_dir, model):
    self.fork_calls.append((parent_sid, project_dir, model))
    if not self._supports:
      return None
    a = _FakeForkAgent()
    self.fork_agents.append(a)
    return a


class _ThreadChannel:
  def __init__(self, threads=True):
    self._threads = threads
    self.thread_cards = []   # (anchor, card)
    self.updates = []        # (mid, card)

  def supports_threads(self):
    return self._threads

  async def send_card_in_thread(self, anchor, card):
    self.thread_cards.append((anchor, card))
    n = len(self.thread_cards)
    return f"om_t{n}", f"omt_{n}"

  async def update_card(self, mid, card):
    self.updates.append((mid, card))
    return mid


def _mgr(ch, notes, **kw):
  async def notify(t):
    notes.append(t)
  return ForkManager(ch, "oc_test", notify, **kw)


def test_open_creates_thread_and_runs_first_turn():
  async def _run():
    ch = _ThreadChannel()
    main = _FakeMainAgent()
    notes = []
    mgr = _mgr(ch, notes)
    await mgr.open(main_agent=main, anchor_msg_id="om_fork",
                   parent_sid="psid", project_dir="/p", model="m",
                   prompt="investigate the auth flow")
    # Forked from the live parent session, in the right project.
    assert main.fork_calls == [("psid", "/p", "m")]
    # Root card opened as a threaded reply to the /fork message.
    assert ch.thread_cards and ch.thread_cards[0][0] == "om_fork"
    # Registered by the thread_id Lark returned (omt_1).
    assert mgr.get("omt_1") is not None
    assert mgr.count() == 1
    # First turn ran with the opening prompt.
    await asyncio.gather(*list(mgr._tasks))
    assert main.fork_agents[0].turns == ["investigate the auth flow"]
    assert notes == []  # success → no main-chat decline message
  asyncio.run(_run())


def test_route_runs_followup_turn_on_same_fork():
  async def _run():
    ch = _ThreadChannel()
    main = _FakeMainAgent()
    mgr = _mgr(ch, [])
    await mgr.open(main_agent=main, anchor_msg_id="om_fork",
                   parent_sid="psid", project_dir="/p", model="m",
                   prompt="first")
    await asyncio.gather(*list(mgr._tasks))
    # A follow-up in the fork thread runs another turn on the SAME agent.
    assert mgr.route("omt_1", "now check the tests") is True
    await asyncio.gather(*list(mgr._tasks))
    assert main.fork_agents[0].turns == ["first", "now check the tests"]
    # Routing an unknown thread is a no-op.
    assert mgr.route("omt_unknown", "x") is False
  asyncio.run(_run())


def test_close_stops_agent_and_unregisters():
  async def _run():
    ch = _ThreadChannel()
    main = _FakeMainAgent()
    mgr = _mgr(ch, [])
    await mgr.open(main_agent=main, anchor_msg_id="om_fork",
                   parent_sid="psid", project_dir="/p", model="m",
                   prompt="first")
    await asyncio.gather(*list(mgr._tasks))
    assert await mgr.close("omt_1") is True
    assert main.fork_agents[0].stopped is True
    assert mgr.get("omt_1") is None
    assert mgr.count() == 0
    # Closing again / unknown thread is a no-op.
    assert await mgr.close("omt_1") is False
  asyncio.run(_run())


def test_decline_when_agent_cannot_fork():
  async def _run():
    ch = _ThreadChannel()
    main = _FakeMainAgent(supports=False)
    notes = []
    mgr = _mgr(ch, notes)
    await mgr.open(main_agent=main, anchor_msg_id="om_fork",
                   parent_sid="", project_dir="/p", model="m", prompt="x")
    assert mgr.count() == 0
    assert ch.thread_cards == []
    assert notes and "Claude or Codex" in notes[0]
  asyncio.run(_run())


def test_decline_when_channel_has_no_threads():
  async def _run():
    ch = _ThreadChannel(threads=False)
    main = _FakeMainAgent()
    notes = []
    mgr = _mgr(ch, notes)
    await mgr.open(main_agent=main, anchor_msg_id="om_fork",
                   parent_sid="", project_dir="/p", model="m", prompt="x")
    assert mgr.count() == 0
    assert main.fork_calls == []  # bailed before spinning up an agent
    assert notes and "sub-threads" in notes[0]
  asyncio.run(_run())


def test_decline_when_max_forks_reached():
  async def _run():
    ch = _ThreadChannel()
    main = _FakeMainAgent()
    notes = []
    mgr = _mgr(ch, notes, max_forks=2)
    for i in range(2):
      await mgr.open(main_agent=main, anchor_msg_id=f"om_{i}",
                     parent_sid="", project_dir="/p", model="m", prompt="x")
    assert mgr.count() == 2
    await mgr.open(main_agent=main, anchor_msg_id="om_3",
                   parent_sid="", project_dir="/p", model="m", prompt="x")
    assert mgr.count() == 2  # rejected
    assert notes and "Too many open forks" in notes[-1]
  asyncio.run(_run())


def test_shutdown_stops_all_forks():
  async def _run():
    ch = _ThreadChannel()
    main = _FakeMainAgent()
    mgr = _mgr(ch, [])
    for i in range(2):
      await mgr.open(main_agent=main, anchor_msg_id=f"om_{i}",
                     parent_sid="", project_dir="/p", model="m", prompt="x")
    await asyncio.gather(*list(mgr._tasks))
    await mgr.shutdown()
    assert mgr.count() == 0
    assert all(a.stopped for a in main.fork_agents)
  asyncio.run(_run())
