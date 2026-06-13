# `claude-cli` — pty-driven interactive TUI adapter (feasibility experiment)

**Question:** can Nemo bill turns at the Claude **subscription** rate by driving
the *unmodified interactive* `claude` TUI under a pseudo-terminal, instead of the
SDK/headless path that bills against the metered Agent-SDK credit pool?

**Answer: yes, and it's built to a usable bar** — confirmed end-to-end through the
real Nemo daemon, with reliability hardening (worker-thread turns, readiness
detection, idle-gating, process-death handling) **and** real per-turn token usage
+ cross-restart resume read from the session transcript. The only residual
caveats are screen-scrape coupling to the TUI layout and the ToS gray area —
ship it opt-in.

> **Correction (important).** An earlier version of this doc claimed "no per-turn
> usage" and "no resume" were *inherent* limits. That was wrong — an artifact of
> testing from *inside* a Claude Code session. The parent session exports
> `CLAUDE_CODE_CHILD_SESSION=1` / `CLAUDECODE=1`; passing those through to the
> spawned `claude` made it behave as a nested child and skip persisting its own
> transcript jsonl. `_build_env` now strips all `CLAUDE_CODE_*` / `CLAUDECODE` /
> `AI_AGENT` markers, so the spawned CLI persists per-turn like any normal
> top-level session — which gives us both usage and resume.

---

## Background (the billing split)

Since 2026-06-15 Anthropic buckets Claude subscription usage by **interactive vs
non-interactive**, and the bucket is decided by a surface tag the CLI puts in its
own `User-Agent`:

| Path | User-Agent surface | Bills against |
|---|---|---|
| Interactive terminal TUI | `claude-cli/<ver> (external, cli)` | subscription quota |
| `claude -p` / Agent SDK / stream-json | `claude-cli/<ver> (external, sdk-cli)` | metered Agent-SDK credits |

Every SDK wrapper (including Nemo's existing `ClaudeCodingAgent`) drives the
headless stream-json path → `sdk-cli` → metered. The hypothesis: spawn the real
TUI under a pty and the *same binary* self-reports `cli` → subscription.

## Deliverable B + C — billing surface (decisive evidence)

`scripts/cli_billing_probe.py` points `ANTHROPIC_BASE_URL` at a local capture
server (dummy key → every call 401s, **zero quota spent**) and reads the
`User-Agent` the CLI sends on `POST /v1/messages`:

```
=== all captured requests ===
  /v1/messages?beta=true  <-  claude-cli/2.1.175 (external, cli)       ← Path A: pty TUI
  /v1/messages?beta=true  <-  claude-cli/2.1.175 (external, cli)
  /v1/messages?beta=true  <-  claude-cli/2.1.175 (external, cli)
  /v1/messages?beta=true  <-  claude-cli/2.1.175 (external, sdk-cli)   ← Path C: claude -p stream-json
  /v1/messages?beta=true  <-  claude-cli/2.1.175 (external, sdk-cli)
  ...
=== verdict ===
  ✅ FEASIBLE: interactive=cli, headless=sdk-cli — surfaces differ as hoped
```

The pty-driven TUI reports **`(external, cli)`**; the headless SDK path reports
**`(external, sdk-cli)`**. The header is the authoritative bucketing signal, so
this is the proof the experiment hinges on.

> `/usage` cross-check (recommended manual follow-up): the smoke turns below ran
> on a real `Claude Max` subscription and therefore on the `cli` surface; a
> before/after `/usage` read will show the deduction landing on the subscription
> pool rather than Agent-SDK credits. The header is already decisive; `/usage`
> is corroboration.

## Deliverable A — functional prototype

`nemo/claude_cli_agent.py` implements the `CodingAgent` interface by spawning
`claude` on a pty (stdlib `pty` + `subprocess`), feeding the byte stream into a
`pyte` terminal emulator, and scraping the rendered screen. Wired into
`agent_factory` and the `--agent` / `/agent` selectors as `claude-cli`.

`scripts/cli_adapter_smoke.py` (real account) exercises it:

```
[turn 1] context seed              ⏺ 'ok'
[turn 2] tool use (deliverable A)  · Write(hello.txt)  ⏺ 'done'   hello.txt == "hi" ✓
[turn 3] context recall            ⏺ 'banana'   (multi-turn context preserved) ✓
[turn 4] ESC interrupt             turn returned 4.1s after ESC ✓
[turn 5] session alive post-ESC    ⏺ 'alive' ✓
=== smoke PASS ===
```

So: a prompt runs a real tool call, returns the assistant's final text, keeps
context across turns, and ESC interrupts without killing the session — all the
properties Nemo needs from a `CodingAgent`.

### How it works
- **Spawn:** `claude --dangerously-skip-permissions` (for `bypassPermissions`)
  on a 200×160 pty, `TERM=xterm-256color`. A reader thread feeds pty bytes into
  a `pyte.HistoryScreen` under a lock.
- **Submit:** type the prompt, then send `\r` **after a short gap** (a combined
  `text\r` write is silently dropped by the TUI).
- **Sync:** wait for the TUI to be idle before submitting (else prompts queue and
  answers desync from turns — the first bug found), then wait for the turn to
  start and finish.
- **Completion:** the `esc to interrupt` spinner is present while working and
  gone when idle; done = spinner gone + empty `❯` input box + screen stable.
- **Answer extraction:** scrape the region after this turn's `❯ <prompt>` echo;
  the answer is the last `⏺` block that is **not** a tool call (tool calls render
  `⏺ Name(args)` and are followed by a `⎿` result line; prose answers are not).

### Reliability hardening (what makes it product-grade, not a toy)
- **Worker-thread turns:** the whole turn (pty polling + every `on_event` call)
  runs on a worker thread via `asyncio.to_thread`, like the SDK path. Mandatory,
  not stylistic — the host's `on_event` marshals card sends to the main loop
  with a blocking `run_coroutine_threadsafe(...).result()`, so calling it from
  the main loop deadlocks. (Found via the live daemon test: the first daemon
  turn hung until this was fixed.)
- **Readiness detection:** wait for the TUI footer (`shift+tab to cycle`) before
  the first prompt instead of a blind fixed sleep, then nudge past any first-run
  trust/theme dialog.
- **Idle-gating:** never submit into a busy TUI — wait until it's idle and
  stable, else prompts queue and answers desync from turns (the first bug found).
- **Process-death handling:** if the TUI exits mid-turn, surface an `ErrorEvent`
  instead of hanging the chat.

Pure scraping logic is pinned by `tests/test_claude_cli_agent.py` (9 tests).

### End-to-end daemon verification
Driven through the real `nemo --agent claude-cli` daemon (relay-injected Lark
message → turn → card):
- `scripts/e2e_test.py --agent claude-cli --skip-sdk`: daemon boots, all 11
  command tests pass.
- `scripts/e2e_test.py --agent claude-cli --workflow`: a full realistic coding
  session — one-line prompt → discussion → finalize plan → implement with file
  tools → skill review → bug-fix-and-verify — driven over Lark, with
  deterministic assertions (files written, the None-input bug actually fixed).
  This is the repeatable regression case for the adapter.

## Token usage + resume (work — read from the session transcript)

The spawned CLI persists each turn's transcript to
`~/.claude/projects/<slug>/<session_id>.jsonl` (the same file `--resume` uses),
**provided** the parent's `CLAUDE_CODE_*` markers are stripped (see the
Correction above). `_SessionLog` tails that file after each turn:

- **Per-turn token usage** — summed across the turn's assistant messages into
  the canonical usage schema, surfaced on `DoneEvent.usage`. Verified live, e.g.
  `{input: 12280, cache_read: 18073, cache_creation: 9509, output: 4,
  total: 39866}`. Cost in USD is still not reported (the transcript records
  tokens, not a per-turn cost figure).
- **Cross-restart resume** — the session id is captured from the transcript and
  carried on `DoneEvent.session_id`; `reset(resume=<id>)` respawns
  `claude --resume <id>`, falling back to a fresh session if the id can't be
  materialised. Verified live: after a `reset(resume)` (simulated daemon
  restart) the agent still recalled a secret word set before the restart.

## Remaining limitations

- **Scraping is coupled to TUI markers** (`⏺` / `⎿` / `❯` / `esc to interrupt`)
  for completion detection + answer extraction; a major `claude` TUI reflow
  could break turn boundaries or answer extraction. (Usage/resume come from the
  structured JSONL and are not affected by layout.)
- **No cost in USD** — only token counts (the transcript has no per-turn cost).
- **Weaker observability than the SDK** — tool calls are scraped `⏺` lines, not
  structured events; sub-agent (`Task`) lifecycle, rate-limit notices, and
  compaction banners are not surfaced.
- **ToS gray area → account risk.** This runs the unmodified official binary on
  the user's own account and automates their own terminal input (softer than
  forging the `User-Agent`), but it can still be read as circumventing usage
  metering. **Account-suspension risk is real.** Not advisable as a default.

## Verdict

| Dimension | Assessment |
|---|---|
| Does pty TUI bill as `(external, cli)`? | **Yes** — confirmed via header capture (B) and headless contrast (C). |
| Functional + reliable as a Nemo agent? | **Yes** — multi-turn, tool calls, interrupt, readiness/idle-gating/crash handling; verified through the daemon + repeatable workflow e2e. |
| Per-turn token usage | **Works** — summed from the session transcript onto `DoneEvent.usage` (USD cost still not reported). |
| Cross-restart resume | **Works** — `--resume <session_id>` from the persisted transcript; verified context survives a simulated restart. |
| Maintenance cost | **Medium-high** — coupled to TUI markers; expect occasional breakage on `claude` updates, guarded by the scraping unit tests + workflow e2e. |
| ToS risk | **Material** — possible account suspension; ship behind an explicit opt-in. |

**Recommendation:** usable for real work where subscription billing matters —
multi-turn context, tool calls, interrupt, per-turn token usage, and
cross-restart resume all work; the reliability hardening means it behaves like a
real agent (no hangs, clean turn boundaries, interruptible), not a demo. The
remaining trade-offs vs the SDK adapter are screen-scrape fragility, no USD cost,
and weaker tool/sub-agent observability. Keep it an opt-in `--agent claude-cli`
provider with the ToS caveat surfaced.

## Artifacts

- `nemo/claude_cli_agent.py` — the adapter.
- `tests/test_claude_cli_agent.py` — scraping unit tests (9).
- `scripts/e2e_test.py --workflow` — repeatable full-flow daemon regression case.
- `scripts/cli_billing_probe.py` — B/C header capture (zero-cost).
- `scripts/cli_adapter_smoke.py` — multi-turn + tool + interrupt smoke (real account).
- `scripts/cli_tui_explore.py` — screen-layout exploration harness.
