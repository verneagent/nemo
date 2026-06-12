#!/usr/bin/env python3
"""Smoke test for ClaudeCliCodingAgent: multi-turn context, a tool-using turn
(deliverable A), and ESC interrupt — driven through the real adapter against a
real account.

    python3 scripts/cli_adapter_smoke.py
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time

from nemo.claude_cli_agent import ClaudeCliCodingAgent
from nemo.turn import AnswerEvent, DoneEvent, ErrorEvent, ProgressEvent, TurnEvent


def _collect() -> tuple[list[TurnEvent], callable]:
  events: list[TurnEvent] = []

  def on_event(ev: TurnEvent) -> None:
    events.append(ev)
    if isinstance(ev, ProgressEvent):
      print(f"    · tool: {ev.summary}")
    elif isinstance(ev, AnswerEvent):
      print(f"    ⏺ answer: {ev.text!r}")
    elif isinstance(ev, ErrorEvent):
      print(f"    ✗ error: {ev.message}")
    elif isinstance(ev, DoneEvent):
      print(f"    ✓ done (cost={ev.cost})")
  return events, on_event


def _answer(events: list[TurnEvent]) -> str:
  for ev in reversed(events):
    if isinstance(ev, AnswerEvent):
      return ev.text
  return ""


async def main() -> int:
  scratch = tempfile.mkdtemp(prefix="nemo_clicli_")
  print(f"scratch project: {scratch}")
  agent = ClaudeCliCodingAgent(
    credentials={}, chat_id="smoke", db=None, channel=None,  # type: ignore[arg-type]
    permission_mode="bypassPermissions",
  )
  ok = True
  try:
    await agent.start(scratch, model="")

    print("\n[turn 1] context seed")
    ev1, cb1 = _collect()
    await agent.run_turn("Remember the secret word is banana. Reply with just: ok", cb1)

    print("\n[turn 2] tool use (deliverable A)")
    ev2, cb2 = _collect()
    await agent.run_turn(
      "create a file hello.txt in the current directory with the text hi, then reply: done", cb2)
    hello = os.path.join(scratch, "hello.txt")
    file_ok = os.path.exists(hello) and open(hello).read().strip() == "hi"
    print(f"    hello.txt created with 'hi': {file_ok}")
    ok = ok and file_ok

    print("\n[turn 3] context recall (multi-turn)")
    ev3, cb3 = _collect()
    await agent.run_turn("What was the secret word? Reply with just the word.", cb3)
    recalled = "banana" in _answer(ev3).lower()
    print(f"    recalled 'banana': {recalled}")
    ok = ok and recalled

    print("\n[turn 4] ESC interrupt")
    ev4, cb4 = _collect()
    task = asyncio.create_task(agent.run_turn(
      "Count slowly from 1 to 30, one number per line, pausing between each.", cb4))
    await asyncio.sleep(6)
    print("    sending ESC...")
    await agent.interrupt()
    t0 = time.monotonic()
    try:
      await asyncio.wait_for(task, timeout=30)
      print(f"    turn returned {time.monotonic()-t0:.1f}s after ESC")
    except asyncio.TimeoutError:
      print("    ✗ turn did not return within 30s of ESC")
      ok = False
    # Session still usable after interrupt?
    print("\n[turn 5] session alive after interrupt")
    ev5, cb5 = _collect()
    await agent.run_turn("reply with exactly: alive", cb5)
    alive = "alive" in _answer(ev5).lower()
    print(f"    session responsive post-interrupt: {alive}")
    ok = ok and alive
  finally:
    await agent.stop()

  print(f"\n=== smoke {'PASS' if ok else 'FAIL'} ===")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(asyncio.run(main()))
