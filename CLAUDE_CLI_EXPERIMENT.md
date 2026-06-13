# `claude-cli` — pty-driven interactive TUI adapter (feasibility experiment)

**Question:** can Nemo bill turns at the Claude **subscription** rate by driving
the *unmodified interactive* `claude` TUI under a pseudo-terminal, instead of the
SDK/headless path that bills against the metered Agent-SDK credit pool?

**Answer: yes, and it's built to a usable bar** — confirmed end-to-end through the
real Nemo daemon, with reliability hardening (worker-thread turns, readiness
detection, idle-gating, process-death handling). Two hard limits are inherent to
driving a human TUI and cannot be engineered away (no per-turn token usage, no
cross-restart resume — see below); everything else is solid. Ship it opt-in,
because the billing approach is a ToS gray area.

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

## Inherent limitations (cannot be engineered away — they're properties of the TUI)

- **No per-turn token usage / cost.** `DoneEvent` carries empty usage / 0 cost.
  Investigated thoroughly: the *spawned interactive* CLI does not persist
  conversation to its session JSONL — a live, idle session for 30 s leaves only
  metadata (`ai-title`/`mode`) on disk, no assistant messages or usage. (The SDK
  path's structured usage comes from stream-json, which is exactly the
  `sdk-cli`-billed channel we're avoiding.) So Nemo's token cards / `/context`
  are blank under this adapter.
- **No cross-restart resume.** Context lives in the running TUI process; since
  the JSONL isn't persisted, `--continue` has nothing to resume (it errors and
  exits). `reset()` therefore respawns a fresh session and **loses context**.
  Within a *live* daemon, multi-turn context IS preserved (the process stays up)
  — only a restart/crash loses it.
- **Scraping is coupled to TUI markers** (`⏺` / `⎿` / `❯` / `esc to interrupt`);
  a major `claude` TUI reflow could break turn boundaries or answer extraction.
- **Weaker observability.** Tool calls are scraped `⏺` lines, not structured
  events; sub-agent (`Task`) lifecycle, rate-limit notices, and compaction
  banners that the SDK adapter surfaces are not available here.
- **ToS gray area → account risk.** This runs the unmodified official binary on
  the user's own account and automates their own terminal input (softer than
  forging the `User-Agent`), but it can still be read as circumventing usage
  metering. **Account-suspension risk is real.** Not advisable as a default.

## Verdict

| Dimension | Assessment |
|---|---|
| Does pty TUI bill as `(external, cli)`? | **Yes** — confirmed via header capture (B) and headless contrast (C). |
| Functional + reliable as a Nemo agent? | **Yes** — multi-turn, tool calls, interrupt, readiness/idle-gating/crash handling; verified through the daemon + repeatable workflow e2e. |
| Per-turn usage / cost | **Unavailable** — inherent to the TUI; token cards blank. |
| Cross-restart resume | **Unavailable** — inherent (no persisted JSONL); live-session context preserved. |
| Maintenance cost | **Medium-high** — coupled to TUI markers; expect occasional breakage on `claude` updates, guarded by the scraping unit tests + workflow e2e. |
| ToS risk | **Material** — possible account suspension; ship behind an explicit opt-in. |

**Recommendation:** usable for real work where subscription billing matters and
the two inherent gaps (no usage display, no resume across restart) are
acceptable. It is **not** a drop-in replacement for the SDK adapter — keep it an
opt-in `--agent claude-cli` provider with the ToS caveat surfaced. The
reliability work means it behaves like a real agent (no hangs, clean turn
boundaries, interruptible), not a demo.

## Artifacts

- `nemo/claude_cli_agent.py` — the adapter.
- `tests/test_claude_cli_agent.py` — scraping unit tests (9).
- `scripts/e2e_test.py --workflow` — repeatable full-flow daemon regression case.
- `scripts/cli_billing_probe.py` — B/C header capture (zero-cost).
- `scripts/cli_adapter_smoke.py` — multi-turn + tool + interrupt smoke (real account).
- `scripts/cli_tui_explore.py` — screen-layout exploration harness.
