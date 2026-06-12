# `claude-cli` — pty-driven interactive TUI adapter (feasibility experiment)

**Question:** can Nemo bill turns at the Claude **subscription** rate by driving
the *unmodified interactive* `claude` TUI under a pseudo-terminal, instead of the
SDK/headless path that bills against the metered Agent-SDK credit pool?

**Answer: yes, it works** — feasibility confirmed end-to-end, with real caveats.
This is a prototype, not production.

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

- **Threading:** the whole turn (pty polling + every `on_event` call) runs on a
  worker thread via `asyncio.to_thread`, exactly like the SDK path. This is
  mandatory, not stylistic: the host's `on_event` marshals card sends to the
  main loop with a blocking `run_coroutine_threadsafe(...).result()`, so calling
  it from the main loop would deadlock. (Found via the live daemon test below —
  the first daemon turn hung until this was fixed.)

Pure scraping logic is pinned by `tests/test_claude_cli_agent.py` (8 tests).

### End-to-end daemon verification
Driven through the real `nemo --agent claude-cli` daemon (relay-injected Lark
message → turn → card):
- `scripts/e2e_test.py --agent claude-cli --skip-sdk`: daemon boots, all 11
  command tests pass.
- Single real turn: a Lark **`Done ✓` card** comes back with the scraped answer;
  daemon log shows `run_turn: done (began=True timed_out=False answer=9 chars,
  8s)`. So the adapter is usable as a real Nemo agent, not just in isolation.

## Fragility / limitations (do not hide these)

- **Screen-scraping is the only answer channel.** The interactive TUI does **not**
  flush its transcript jsonl per turn (verified: it stayed at 118 bytes,
  metadata only, while alive) — so the clean structured-data shortcut the SDK
  adapter enjoys is unavailable. Everything rides on the `⏺` / `⎿` / `❯` /
  `esc to interrupt` markers; any TUI reflow or version bump can break it.
- **No structured usage / cost.** `DoneEvent` reports empty usage and `0` cost;
  the TUI exposes no per-turn token counts on a machine channel. Nemo's token
  cards and `/context` would be blank under this adapter.
- **No resume across restarts.** Because the jsonl isn't flushed live, `reset()`
  respawns a fresh session and **loses conversation context**. The SDK adapter's
  resume-with-session-id recovery has no equivalent here.
- **Weaker observability.** Tool calls are best-effort scraped `⏺` lines, not
  structured events; permissions/errors/compaction are invisible compared to the
  SDK's typed stream. Sub-agent (`Task`) lifecycle, rate-limit notices, and
  compaction banners are all lost.
- **ToS gray area → account risk.** This runs the unmodified official binary on
  the user's own account and automates their own terminal input (softer than
  forging the `User-Agent`), but it can still be read as circumventing usage
  metering. **Account-suspension risk is real.** Not advisable as a default.

## Verdict

| Dimension | Assessment |
|---|---|
| Does pty TUI bill as `(external, cli)`? | **Yes** — confirmed via header capture (B) and headless contrast (C). |
| Functional as a Nemo agent? | **Yes** — multi-turn, tool calls, interrupt all work (A). |
| Production-ready? | **No** — no usage/cost, no resume, screen-scraping fragility, weaker observability. |
| Maintenance cost | **High** — coupled to exact TUI rendering; expect breakage on `claude` updates. |
| ToS risk | **Material** — possible account suspension; ship behind an explicit opt-in with a loud warning, if at all. |

**Recommendation:** the mechanism is proven and the prototype is usable for
experiments or low-stakes personal automation where subscription billing matters
more than robustness. It is **not** a drop-in replacement for the SDK adapter:
the loss of structured usage, resume, and observability — plus the ToS exposure —
make it unsuitable as Nemo's default path. Keep it as an opt-in `--agent
claude-cli` provider with the caveats above surfaced to the user.

## Artifacts

- `nemo/claude_cli_agent.py` — the adapter.
- `tests/test_claude_cli_agent.py` — scraping unit tests.
- `scripts/cli_billing_probe.py` — B/C header capture (zero-cost).
- `scripts/cli_adapter_smoke.py` — A end-to-end smoke (real account).
- `scripts/cli_tui_explore.py` — screen-layout exploration harness.
