#!/usr/bin/env python3
"""Reproduce claude-agent-sdk-python#788: stale TaskNotification leak.

This script documents two symptoms:

1. SINGLE-TASK LEAK (original bug)
   One background task spawned in turn 1, allowed to complete between turns.
   Turn 2's receive_response() yields the stale TaskNotification before the
   new turn's init, and the model sees it in context and responds to it
   instead of the new user prompt.

2. RETRY AMPLIFICATION (discovered in production)
   Naive consumer-side workaround: detect stale, discard the turn, re-query
   the original prompt. This fails in practice when the original prompt is
   non-trivial: the model calls the Agent tool again on each retry, spawning
   NEW background tasks. Each retry drains one stale but spawns one or more
   new stales, so the retry budget is consumed without ever delivering a
   clean answer.

Usage:
  python3 scripts/repro_sdk_788.py              # run both symptoms
  python3 scripts/repro_sdk_788.py --single     # only symptom 1
  python3 scripts/repro_sdk_788.py --amplify    # only symptom 2

Requires: claude-agent-sdk, ANTHROPIC_API_KEY (or equivalent), a working
`claude` CLI on PATH.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Set

import anyio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TaskNotificationMessage,
    TaskStartedMessage,
    TextBlock,
)


def msg_text(msg: object) -> str:
  if isinstance(msg, AssistantMessage):
    return "".join(
      b.text for b in msg.content if isinstance(b, TextBlock) and b.text
    )
  return ""


async def _spawn_tasks(client: ClaudeSDKClient, n: int, sleep_s: int = 90) -> Set[str]:
  """Turn 1: ask model to spawn N background tasks and end turn immediately."""
  prompt = (
    f"Use the Agent tool with run_in_background=true to spawn exactly {n} "
    f"independent background subagents. Each subagent should run "
    f"'sleep {sleep_s} && echo done'. IMPORTANT: after dispatching, DO NOT "
    f"poll, wait, or check status. Reply with exactly the word 'Spawned.' "
    f"and end your turn IMMEDIATELY. The whole point is that these tasks "
    f"must still be running when your turn ends."
  )
  await client.query(prompt)
  pending: Set[str] = set()
  async for msg in client.receive_response():
    if isinstance(msg, TaskStartedMessage):
      pending.add(msg.task_id)
      print(f"  TaskStarted task={msg.task_id}")
    elif isinstance(msg, AssistantMessage):
      text = msg_text(msg)
      if text:
        print(f"  AssistantMessage text={text[:60]!r}")
    elif isinstance(msg, ResultMessage):
      print(f"  ResultMessage cost=${getattr(msg, 'total_cost_usd', 0):.4f}")
  return pending


async def run_single_leak() -> None:
  """Symptom 1: single stale TaskNotification leaks into turn 2 context."""
  print("\n======================================================")
  print(" SYMPTOM 1: single-task stale leak")
  print("======================================================")

  options = ClaudeAgentOptions(
    allowed_tools=["Agent", "Read", "Glob", "Grep", "Bash"],
    permission_mode="bypassPermissions",
    cwd="/tmp",
  )
  async with ClaudeSDKClient(options=options) as client:
    print("\n=== Turn 1: spawn 1 background task (sleep 60) ===")
    pending = await _spawn_tasks(client, n=1, sleep_s=60)
    print(f"Pending after turn 1: {pending}")
    if not pending:
      print("!! Model waited for task. Re-run.")
      return

    print("\nWaiting 70s for background task to complete...")
    await asyncio.sleep(70)

    print("\n=== Turn 2: simple unrelated question ===")
    await client.query("What is 7*8? Reply with only the number.")
    leaked = []
    answer = ""
    async for msg in client.receive_response():
      if isinstance(msg, TaskNotificationMessage):
        print(f"  *** LEAK: TaskNotification task={msg.task_id} in turn 2 ***")
        leaked.append(msg.task_id)
      elif isinstance(msg, AssistantMessage):
        text = msg_text(msg)
        if text:
          answer = text
          print(f"  AssistantMessage: {text[:80]!r}")
      elif isinstance(msg, ResultMessage):
        print(f"  ResultMessage cost=${getattr(msg, 'total_cost_usd', 0):.4f}")

    print(f"\nFinal answer: {answer!r}")
    print(f"Leaked ids: {leaked}")
    if leaked and "56" not in answer:
      print("✗ BUG CONFIRMED — stale notification leaked AND model gave wrong answer.")
    elif leaked:
      print("~ Partial: stale leaked but model still got the right answer.")
    else:
      print("? No leak this run (timing-dependent). Re-run to reproduce.")


async def run_amplification() -> None:
  """Symptom 2: naive retry with original prompt spawns new tasks each retry."""
  print("\n======================================================")
  print(" SYMPTOM 2: retry amplification (spawn-more feedback loop)")
  print("======================================================")

  options = ClaudeAgentOptions(
    allowed_tools=["Agent", "Read", "Glob", "Grep", "Bash"],
    permission_mode="bypassPermissions",
    cwd="/tmp",
  )
  async with ClaudeSDKClient(options=options) as client:
    print("\n=== Turn 1: spawn 3 background tasks (sleep 20) ===")
    initial_pending = await _spawn_tasks(client, n=3, sleep_s=20)
    print(f"Pending after turn 1: {initial_pending}")
    if not initial_pending:
      print("!! Model waited. Re-run.")
      return

    print("\nWaiting 30s for background tasks to complete...")
    await asyncio.sleep(30)

    # The key: turn 2's user prompt is NON-TRIVIAL and likely to make the
    # model use the Agent tool again. Each retry re-sends this prompt →
    # new subagents → new pending tasks → new stales next retry.
    user_prompt = (
      "Spawn a background Agent (run_in_background=true) to compute 7*8 "
      "and report the result. After dispatching, say 'Working.' and end turn."
    )

    stale_ids: Set[str] = set(initial_pending)
    max_retries = 5
    retry_log: list[dict] = []

    for attempt in range(max_retries):
      label = f"retry-{attempt}"
      print(f"\n--- {label}: query() ---")
      await client.query(user_prompt)

      found_stale = False
      stale_drained_this_turn: list[str] = []
      new_tasks_spawned: list[str] = []

      async for msg in client.receive_response():
        if isinstance(msg, TaskNotificationMessage):
          if msg.task_id in stale_ids:
            stale_ids.discard(msg.task_id)
            stale_drained_this_turn.append(msg.task_id)
            found_stale = True
            print(f"  [{label}] stale drained task={msg.task_id}")
          else:
            print(f"  [{label}] TaskNotification task={msg.task_id} (not stale)")
        elif isinstance(msg, TaskStartedMessage):
          new_tasks_spawned.append(msg.task_id)
          print(f"  [{label}] !! new TaskStarted task={msg.task_id} "
                f"(will become stale if turn ends before it completes)")
        elif isinstance(msg, AssistantMessage):
          text = msg_text(msg)
          if text:
            tag = "DROPPED" if found_stale else "kept"
            print(f"  [{label}] AssistantMessage [{tag}]: {text[:60]!r}")
        elif isinstance(msg, ResultMessage):
          print(f"  [{label}] ResultMessage")

      # At end of turn, any still-pending tasks become stale for next turn.
      for tid in new_tasks_spawned:
        stale_ids.add(tid)

      retry_log.append({
        "retry": attempt,
        "drained": stale_drained_this_turn,
        "spawned": new_tasks_spawned,
        "stale_remaining": len(stale_ids),
      })

      if not found_stale:
        print(f"\n[{label}] clean turn — real answer delivered.")
        break

    print("\n===================== SUMMARY =====================")
    total_drained = sum(len(r["drained"]) for r in retry_log)
    total_spawned = sum(len(r["spawned"]) for r in retry_log)
    print(f"Retries used:          {len(retry_log)} / {max_retries}")
    print(f"Total stales drained:  {total_drained}")
    print(f"Total new tasks spawned during retries: {total_spawned}")
    print(f"Stales remaining at end: {len(stale_ids)}")
    for r in retry_log:
      print(f"  retry {r['retry']}: drained={len(r['drained'])} "
            f"spawned={len(r['spawned'])} remaining={r['stale_remaining']}")

    if total_spawned > 0:
      print("\n✗ AMPLIFICATION CONFIRMED — retries spawn new background tasks,"
            "\n  which become future stales. Naive re-query cannot converge.")
    elif stale_ids:
      print("\n~ Retries did not amplify, but stales still remain.")
    else:
      print("\n✓ All stales drained without amplification "
            "(model avoided Agent tool this run; re-run to reproduce).")


async def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--single", action="store_true",
                      help="Only run symptom 1 (single-task leak)")
  parser.add_argument("--amplify", action="store_true",
                      help="Only run symptom 2 (retry amplification)")
  args = parser.parse_args()

  run_single = args.single or not args.amplify
  run_amp = args.amplify or not args.single

  if run_single:
    await run_single_leak()
  if run_amp:
    await run_amplification()
  return 0


if __name__ == "__main__":
  sys.exit(anyio.run(main))
