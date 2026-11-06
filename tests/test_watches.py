"""Daemon-owned watches (nemo/watches.py).

What these pin, and why: every bug the CLI-owned Monitor ever had came from
the daemon RECONSTRUCTING which watch a report belonged to and whether it was
still alive (see the module docstring). Here a report can only travel along
the pipe of the process that produced it, and a watch is over exactly when
that process ends — so the tests below assert the two properties directly:
each line reaches its OWN watch's callback, and status comes from the
process, not from a message.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from nemo.watches import Watch, WatchManager


class _Rig:
  def __init__(self, tmp_path: Path, max_live: int = 3) -> None:
    self.started: list[Watch] = []
    self.reports: list[tuple[str, str]] = []
    self.ended: list[tuple[str, str]] = []
    self.finished: list[Watch] = []
    self._max_live = max_live
    self._tmp = tmp_path

  def manager(self) -> WatchManager:
    async def on_start(watch: Watch) -> None:
      self.started.append(watch)

    async def on_report(watch: Watch, text: str) -> None:
      self.reports.append((watch.watch_id, text))

    async def on_end(watch: Watch) -> None:
      self.ended.append((watch.watch_id, watch.status))

    def on_finished(watch: Watch) -> None:
      self.finished.append(watch)

    return WatchManager(
      on_start=on_start, on_report=on_report, on_end=on_end,
      on_finished=on_finished, spawn_cwd=str(self._tmp),
      stop_dir=self._tmp / "watches", max_live=self._max_live)


async def _wait_for(predicate, timeout: float = 5.0) -> None:
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if predicate():
      return
    await asyncio.sleep(0.05)
  raise AssertionError("condition not reached in time")


def test_a_watch_streams_every_output_line_to_its_own_card(tmp_path):
  rig = _Rig(tmp_path)
  mgr = rig.manager()

  async def main() -> None:
    ok, reason = await mgr.arm(
      chat_id="oc_test", description="t", command="printf 'alpha\\nbeta\\n'",
      timeout_ms=0)
    assert ok, reason
    await _wait_for(lambda: len(rig.ended) == 1)
    await asyncio.wait_for(mgr.close(), 10)

  asyncio.run(main())
  assert [text for _, text in rig.reports] == ["alpha", "beta"]
  assert rig.started[0].reports == 2
  assert rig.ended == [(rig.started[0].watch_id, "completed")]
  assert [w.watch_id for w in rig.finished] == [rig.started[0].watch_id]


def test_two_watches_never_share_a_report(tmp_path):
  """The 2026-11-06 incident, inverted: identity is the pipe, not a guess.

  Both watches print a distinct marker at the same time; every report must
  arrive tagged with the watch whose process printed it.
  """
  rig = _Rig(tmp_path)
  mgr = rig.manager()

  async def main() -> None:
    a, _ = await mgr.arm(
      chat_id="oc_test", description="a",
      command="for i in 1 2 3; do echo A$i; sleep 0.05; done", timeout_ms=0)
    b, _ = await mgr.arm(
      chat_id="oc_test", description="b",
      command="for i in 1 2 3; do echo B$i; sleep 0.05; done", timeout_ms=0)
    assert a and b
    await _wait_for(lambda: len(rig.ended) == 2)
    await asyncio.wait_for(mgr.close(), 10)

  asyncio.run(main())
  by_watch: dict[str, list[str]] = {}
  for watch_id, text in rig.reports:
    by_watch.setdefault(watch_id, []).append(text)
  assert len(by_watch) == 2
  for texts in by_watch.values():
    assert all(t.startswith("A") for t in texts) or all(
      t.startswith("B") for t in texts)


def test_the_cap_refuses_a_fourth_watch_and_says_how_to_free_one(tmp_path):
  rig = _Rig(tmp_path, max_live=2)
  mgr = rig.manager()

  async def main() -> tuple[bool, str]:
    for i in range(2):
      ok, reason = await mgr.arm(
        chat_id="oc_test", description=f"w{i}", command="sleep 30",
        timeout_ms=0)
      assert ok, reason
    verdict = await mgr.arm(
      chat_id="oc_test", description="extra", command="sleep 30",
      timeout_ms=0)
    await asyncio.wait_for(mgr.close(), 10)
    return verdict

  ok, reason = asyncio.run(main())
  assert not ok
  assert ".stop" in reason and "2" in reason
  assert len(rig.started) == 2


def test_a_failed_marker_card_refuses_the_arm(tmp_path):
  """A watch nobody can see is not a watch — and must not start a process."""
  calls: list[Watch] = []

  async def on_start(watch: Watch) -> None:
    calls.append(watch)
    raise RuntimeError("card send failed")

  async def on_report(watch: Watch, text: str) -> None:
    raise AssertionError("must not report")

  async def on_end(watch: Watch) -> None:
    raise AssertionError("must not end")

  mgr = WatchManager(
    on_start=on_start, on_report=on_report, on_end=on_end,
    on_finished=lambda watch: None, spawn_cwd=str(tmp_path),
    stop_dir=tmp_path / "watches")

  async def main() -> tuple[bool, str]:
    verdict = await mgr.arm(
      chat_id="oc_test", description="x", command="sleep 30", timeout_ms=0)
    await asyncio.wait_for(mgr.close(), 10)
    return verdict

  ok, reason = asyncio.run(main())
  assert not ok
  assert "card send failed" in reason
  assert len(calls) == 1
  assert mgr.live_ids() == []


def test_the_stop_file_ends_the_watch(tmp_path):
  rig = _Rig(tmp_path)
  mgr = rig.manager()

  async def main() -> None:
    ok, reason = await mgr.arm(
      chat_id="oc_test", description="t",
      command="while true; do echo tick; sleep 0.2; done", timeout_ms=0)
    assert ok, reason
    watch = rig.started[0]
    await _wait_for(lambda: len(rig.reports) >= 1)
    assert watch.stop_file is not None
    watch.stop_file.write_text("stop")
    await _wait_for(lambda: len(rig.ended) == 1)
    await asyncio.wait_for(mgr.close(), 10)

  asyncio.run(main())
  assert rig.ended == [(rig.started[0].watch_id, "stopped")]
  assert not rig.started[0].stop_file.exists()


def test_the_timeout_cap_ends_the_watch(tmp_path):
  rig = _Rig(tmp_path)
  mgr = rig.manager()

  async def main() -> float:
    began = time.monotonic()
    ok, reason = await mgr.arm(
      chat_id="oc_test", description="t", command="sleep 30", timeout_ms=300)
    assert ok, reason
    await _wait_for(lambda: len(rig.ended) == 1)
    await asyncio.wait_for(mgr.close(), 10)
    return time.monotonic() - began

  elapsed = asyncio.run(main())
  assert rig.ended == [(rig.started[0].watch_id, "timeout")]
  assert elapsed < 8


def test_an_ended_watch_is_dropped_from_the_live_set(tmp_path):
  rig = _Rig(tmp_path)
  mgr = rig.manager()

  async def main() -> list[str]:
    ok, reason = await mgr.arm(
      chat_id="oc_test", description="t", command="echo once", timeout_ms=0)
    assert ok, reason
    watch_id = rig.started[0].watch_id
    assert mgr.live_ids() == [watch_id]
    await _wait_for(lambda: len(rig.ended) == 1)
    await asyncio.wait_for(mgr.close(), 10)
    return mgr.live_ids()

  assert asyncio.run(main()) == []
