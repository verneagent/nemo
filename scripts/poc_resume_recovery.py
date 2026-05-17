#!/usr/bin/env python3
"""PoC: deterministic SDK#788 repro + reconnect-with-resume recovery, N runs.

Why deterministic now
---------------------
The earlier version was timing-dependent: it *asked* the model to end turn 1
quickly, but the model sometimes polled the background task to completion
inside turn 1, so nothing went stale. Here we remove that variable entirely:

  Turn 1: spawn a background task + seed a codeword. The MOMENT the task is
  launched (first TaskStartedMessage), we call client.interrupt() to end the
  turn with the task still pending. This is also the *faithful* production
  trigger — nemo's esc/stop path calls interrupt() with a task in flight.

  Wait > task duration so the task completes and its TaskNotification gets
  queued in the (now-wedged) CLI.

  Turn 2 (same subprocess): the queued stale TaskNotification leaks at the
  front of the stream — SDK#788 — deterministically.

Recovery under test (per leak): hard-close (NO interrupt — the wedged CLI's
control channel is dead; __aexit__ + SIGKILL, exactly nemo/sdk_thread.py:
_do_close) then reconnect with ClaudeAgentOptions(resume=<session_id>).

Health check on the resumed session uses a BRAND-NEW prompt (not a re-send)
that requires BOTH pre-leak history recall AND fresh computation, so a model
"still stuck on the previous turn" cannot pass.

Aggregate over N runs:
  * setup ok        (task launched + interrupted in turn 1)
  * leak reproduced (stale TaskNotification leaked into turn 2)
  * recovery passed (resumed turn: no leak, codeword recalled, fresh compute,
                     no stale '56' regurgitation)

Usage:
  python3 scripts/poc_resume_recovery.py [--runs N] [--wait S]

Requires: claude-agent-sdk, ANTHROPIC_API_KEY (or ~/.claude creds), `claude`.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import signal
import sys

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

CODEWORD = "BANANA-7"  # N=7; R1 expects 7*6=42


def msg_text(msg: object) -> str:
  if isinstance(msg, AssistantMessage):
    return "".join(
      b.text for b in msg.content if isinstance(b, TextBlock) and b.text
    )
  return ""


def _cli_pid(client: ClaudeSDKClient) -> int | None:
  transport = getattr(client, "_transport", None)
  proc = getattr(transport, "_process", None) if transport else None
  return getattr(proc, "pid", None) if proc else None


async def _hard_close(client: ClaudeSDKClient) -> None:
  """Mirror nemo/sdk_thread.py:_do_close — __aexit__ then SIGKILL. NO interrupt:
  the subprocess just leaked a stale stream; its control channel is the dead
  path. Destroying the process is strictly more thorough than interrupt()."""
  pid = _cli_pid(client)
  try:
    await asyncio.wait_for(client.__aexit__(None, None, None), timeout=5)
  except Exception as e:  # noqa: BLE001 — PoC: report, don't mask
    print(f"    (__aexit__ raised: {e!r})")
  if pid:
    try:
      os.kill(pid, 0)
      os.kill(pid, signal.SIGKILL)
      print(f"    CLI {pid} SIGKILLed")
    except OSError:
      print(f"    CLI {pid} already dead — good")


class PolledError(RuntimeError):
  """Turn 1 did not end within the cap — the model is polling the task to
  completion (the leak-suppressing branch). Kept for reference; no longer
  raised — we classify by the deterministic leak check instead."""


async def turn1_spawn(client: ClaudeSDKClient, gate: str,
                      safety_s: float = 120.0) -> tuple[set[str], str]:
  """Spawn a GATED bg task + seed codeword; let turn 1 end naturally.

  The subagent blocks until we ``touch <gate>``. We release the gate ONLY
  after turn 1 has returned — so the task is *guaranteed* still pending at
  turn-1 end regardless of model behaviour, then completes in the
  inter-turn gap. That is exactly SDK#788 symptom 1 ("task completes
  between turns"), made deterministic (near-100% leak per setup-ok run)
  instead of racing a fixed `sleep` against a variable turn-1 duration.

  ``safety_s`` guards the one residual non-determinism: a model that
  polls the (stuck) gated task instead of ending the turn. That surfaces
  as TimeoutError -> caller marks 'error' and retries. We do NOT
  interrupt (interrupt empirically suppresses the leak).

  Returns (all turn-1 task ids, session_id).
  """
  sub = (f"for i in $(seq 1 300); do [ -f {gate} ] && exit 0; sleep 1; "
         f"done")
  prompt = (
    f"Two instructions. (1) Use the Agent tool with run_in_background=true "
    f"to spawn ONE background subagent whose only job is to run this exact "
    f"bash command: {sub!r}. (2) Remember this codeword for later: "
    f"{CODEWORD}. After dispatching, DO NOT poll, wait, or check status — "
    f"reply with exactly 'Spawned.' and end your turn IMMEDIATELY."
  )
  await client.query(prompt)
  ids: set[str] = set()
  session_id = ""

  async def _drain() -> None:
    nonlocal session_id
    async for msg in client.receive_response():
      if isinstance(msg, TaskStartedMessage):
        ids.add(msg.task_id)
        print(f"    TaskStarted task={msg.task_id}", flush=True)
      elif isinstance(msg, AssistantMessage):
        t = msg_text(msg)
        if t:
          print(f"    turn1 answer={t[:50]!r}", flush=True)
      elif isinstance(msg, ResultMessage):
        session_id = getattr(msg, "session_id", "") or ""

  await asyncio.wait_for(_drain(), timeout=safety_s)
  return ids, session_id


async def read_turn(client: ClaudeSDKClient, prompt: str, stale: set[str],
                    label: str, timeout: float = 90.0
                    ) -> tuple[bool, str, str]:
  await client.query(prompt)
  leaked = False
  answer = ""
  session_id = ""

  async def _drain() -> None:
    nonlocal leaked, answer, session_id
    async for msg in client.receive_response():
      if isinstance(msg, TaskNotificationMessage):
        if msg.task_id in stale:
          leaked = True
          print(f"    [{label}] *** STALE LEAK task={msg.task_id} ***",
                flush=True)
      elif isinstance(msg, AssistantMessage):
        t = msg_text(msg)
        if t:
          answer += t
          print(f"    [{label}] answer={t[:70]!r}", flush=True)
      elif isinstance(msg, ResultMessage):
        session_id = getattr(msg, "session_id", "") or ""

  try:
    await asyncio.wait_for(_drain(), timeout=timeout)
  except asyncio.TimeoutError:
    print(f"    [{label}] read timed out after {timeout}s "
          f"(wedged CLI) — treating as leaked", flush=True)
    leaked = True  # a hung post-leak turn is itself the #788 failure mode
  return leaked, answer, session_id


async def resumed_health_ok(client: ClaudeSDKClient, stale: set[str]) -> bool:
  """Brand-new prompt: recall codeword (history) + compute N*6 (fresh),
  strict format, and must NOT contain '56' (q2/leak regurgitation)."""
  r1 = ("Use NO tools. Earlier in THIS conversation I gave you a codeword "
        "shaped WORD-N. Take N, multiply by 6, reply EXACTLY: "
        "CHECK <codeword> <product>")
  leaked, ans, _ = await read_turn(client, r1, stale, "resumed")
  ok = ((not leaked) and (CODEWORD in ans) and ("42" in ans)
        and ("56" not in ans))
  print(f"    health: leaked={leaked} codeword={CODEWORD in ans} "
        f"compute42={'42' in ans} no56={'56' not in ans} "
        f"-> {'PASS' if ok else 'FAIL'}")
  return ok


def dump_session_jsonl(session_id: str, tail: int = 8) -> None:
  hits = glob.glob(
    os.path.expanduser(f"~/.claude/projects/**/{session_id}*.jsonl"),
    recursive=True)
  if not hits:
    print(f"    (no jsonl for {session_id[:8]})")
    return
  with open(hits[0]) as f:
    lines = f.readlines()
  print(f"    persisted history tail ({len(lines)} recs): {hits[0]}")
  for ln in lines[-tail:]:
    try:
      rec = json.loads(ln)
    except Exception:
      continue
    msg = rec.get("message", {})
    role = msg.get("role", "")
    content = msg.get("content", "")
    if isinstance(content, list):
      content = " ".join(
        c.get("text", f"<{c.get('type')}>")
        for c in content if isinstance(c, dict))
    print(f"      [{rec.get('type','?')}/{role}] {str(content)[:80]!r}")


async def one_leak_run(attempt: int, base_opts: dict,
                       wait_s: int, show_jsonl: bool) -> dict:
  """One attempt. status:
       'setupfail' -> model never launched Agent; discard, retry
       'noleak'  -> setup ok, normal end, task finished in turn 1; retry
       'leak'    -> #788 reproduced; 'recovery' True/False set
       'error'   -> infinite hang / unexpected; retry
  """
  r: dict = {"status": "setupfail", "recovery": None}
  print(f"\n----- attempt {attempt} -----", flush=True)
  gate = f"/tmp/poc788_gate_{os.getpid()}_{attempt}"
  try:
    os.unlink(gate)
  except OSError:
    pass
  client = ClaudeSDKClient(options=ClaudeAgentOptions(**base_opts))
  await client.__aenter__()
  try:
    try:
      ids, sid1 = await turn1_spawn(client, gate)
    except asyncio.TimeoutError:
      print("    turn1 safety timeout (model polled stuck gated task) "
            "— retry (no interrupt)")
      r["status"] = "error"
      await _hard_close(client)
      return r
    if not ids:
      print("    setup FAILED (model never launched Agent) — retry")
      await _hard_close(client)
      return r
    stale = set(ids)

    # Turn 1 ended with the gated task GUARANTEED still pending. Release
    # the gate now so the task completes in the inter-turn gap -> its
    # TaskNotification queues and leaks into turn 2 (SDK#788 symptom 1).
    open(gate, "w").close()
    print(f"    turn1 ended; gate released; waiting {wait_s}s for task "
          f"to complete (-> stale)...")
    await asyncio.sleep(wait_s)

    q2 = ("What is 7*8? Also what codeword did I ask you to remember? "
          "Reply concisely.")
    leaked, _, sid2 = await read_turn(client, q2, stale, "buggy")
    session_id = sid2 or sid1
    if not leaked:
      print("    no leak this attempt")
      r["status"] = "noleak"
      await _hard_close(client)
      return r

    r["status"] = "leak"
    print("    RECOVERY: hard-close (NO interrupt) + reconnect(resume)")
    await _hard_close(client)
    if not session_id:
      print("    no session_id — cannot resume; recovery=FAIL")
      r["recovery"] = False
      return r
    resumed = ClaudeSDKClient(
      options=ClaudeAgentOptions(resume=session_id, **base_opts))
    await resumed.__aenter__()
    r["recovery"] = await resumed_health_ok(resumed, stale)
    if show_jsonl:
      dump_session_jsonl(session_id)
    await _hard_close(resumed)
  except Exception as e:  # noqa: BLE001 — record, keep the batch going
    print(f"    attempt errored: {type(e).__name__}: {e}")
    r["status"] = "error"
    try:
      await _hard_close(client)
    except Exception:
      pass
  finally:
    open(gate, "w").close()  # ensure any orphan subagent unblocks + exits
    try:
      os.unlink(gate)
    except OSError:
      pass
  return r


async def main() -> int:
  try:
    sys.stdout.reconfigure(line_buffering=True)  # survive timeout kills
  except Exception:
    pass
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--runs", type=int, default=5,
                  help="target number of ACTUAL #788 leak reproductions")
  ap.add_argument("--wait", type=int, default=15,
                  help="post-gate-release wait for the task to complete")
  ap.add_argument("--max-attempts", type=int, default=15,
                  help="hard cap on attempts to collect --runs leaks")
  args = ap.parse_args()

  base_opts = dict(
    allowed_tools=["Agent", "Read", "Glob", "Grep", "Bash"],
    permission_mode="bypassPermissions",
    cwd="/tmp",
  )

  print("=" * 64)
  print(f" SDK#788 repro + resume recovery — collect {args.runs} leaks")
  print(f" (Bernoulli sample-to-N: no-leak/setupfail attempts retried)")
  print("=" * 64)

  leaks: list[dict] = []
  tally = {"setupfail": 0, "noleak": 0, "error": 0, "leak": 0}
  attempt = 0
  while len(leaks) < args.runs and attempt < args.max_attempts:
    attempt += 1
    res = await one_leak_run(
      attempt, base_opts, args.wait,
      show_jsonl=(len(leaks) == args.runs - 1))
    tally[res["status"]] = tally.get(res["status"], 0) + 1
    if res["status"] == "leak":
      leaks.append(res)

  rec_pass = sum(1 for x in leaks if x["recovery"] is True)
  rec_fail = sum(1 for x in leaks if x["recovery"] is not True)

  print("\n" + "=" * 64)
  print(" SUMMARY")
  print("=" * 64)
  print(f"  attempts:              {attempt}/{args.max_attempts}")
  print(f"  discarded (setupfail): {tally['setupfail']}")
  print(f"  discarded (error):     {tally['error']}")
  print(f"  setup-ok / no leak:    {tally['noleak']}")
  print(f"  #788 LEAK reproduced:  {len(leaks)}")
  print(f"  recovery PASS:         {rec_pass}/{len(leaks)}")
  print(f"  recovery FAIL:         {rec_fail}/{len(leaks)}")
  for i, x in enumerate(leaks, 1):
    print(f"   leak {i}: recovery={x['recovery']}")

  proven = len(leaks) >= args.runs and rec_fail == 0
  print("\n" + (
    f"✓ PROVEN: {len(leaks)}/{len(leaks)} reproduced #788 leaks were "
    f"recovered by reconnect-with-resume (no interrupt)."
    if proven else
    "✗ NOT proven — too few leaks collected or a recovery FAILED."))
  print("=" * 64)
  return 0 if proven else 1


if __name__ == "__main__":
  sys.exit(anyio.run(main))
