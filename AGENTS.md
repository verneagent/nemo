# Nemo

Lark-connected coding agent daemon. Repo focus:
- Lark/relay-facing orchestration in Python
- coding-agent runtime behind `CodingAgent` (Claude or Codex)
- channel I/O behind `Channel`

## Working Model

- Keep `nemo/agent.py` as orchestration only. Push agent-specific logic into concrete adapters such as `LarkChannel`, `ClaudeCodingAgent`, and `CodexCodingAgent`.
- `agent.py` is channel-agnostic and agent-agnostic. It only sees `Channel` and `CodingAgent` abstractions. Lark-specific logic (file download, message enrichment, API calls) belongs in `LarkChannel`. SDK-specific logic belongs in the concrete `CodingAgent` adapters.
- The coding agent is selected by `--agent claude|codex|opencode` (default `claude`). `nemo/agent_factory.py` maps the agent kind to its adapter and enforces agent/model compatibility. ("Provider" in this repo means a *model* provider — DeepSeek / Kimi / Anthropic — and only appears in `nemo/models.json`'s top-level `providers` grouping.)
- Reasoning effort is a shared `low/medium/high/max` knob (`--effort` at startup, `/effort` at runtime) exposed on `CodingAgent.set_effort`. Each adapter translates: `ClaudeCodingAgent` passes the value through the SDK's native `ClaudeAgentOptions.effort` parameter (claude-agent-sdk ≥ 0.1.50); `CodexCodingAgent` passes `--effort` to the sidecar, which sets `ThreadOptions.modelReasoningEffort` (clamps `max` → `high`, since the Codex SDK has no `max` tier); `OpenCodeCodingAgent` injects a prompt prefix (also clamps `max` → `high`). Because Claude's effort lives on SDK options rather than per-turn input, the host reconnects with `resume=<sdk_session_id>` after `/effort` so the new value takes effect on the next turn — session context is preserved across the reconnect.
- Prefer relay-backed event delivery. Direct Lark 长连接 is only a fallback when relay is not configured.
- Preserve the one-card-per-turn model: turn cards evolve through PATCH instead of emitting a new card for each phase.
- Keep turn execution event-driven. `run_turn()` should emit typed events and the main loop should react to them.
- Stop/esc only interrupts the current turn — do not reset or restart the SDK client. Session and conversation context must be preserved (match CLI Escape behavior).
- `/fork` (read-only, multi-turn sub-thread) is exposed via `CodingAgent.supports_fork()` / `fork()` and orchestrated by `nemo/fork.py`'s `ForkManager` (one read-only sub-agent + SDK subprocess per fork, routed by Lark `thread_id`). Supported on **Claude** (resume + `fork_session=True`, branching the transcript; read-only via bash sandbox with a scratch cwd so the project sits outside the writable workspace) and **Codex** (copy the parent rollout jsonl to a new thread id then resume the copy — Codex resume *appends*, so the fork needs a private copy; read-only via native `sandboxMode=read-only`, so cwd can stay the project). OpenCode is unsupported. Both keep full parent context but cannot modify project files.

## Runtime Notes

- Dev install:
  ```bash
  pip install -e .
  nemo --chat-id <ID> --project-dir <DIR>
  ```
- Do not use `pipx install captain-nemo` on the dev machine. `pipx` is only for end-user installs.
- Profile config lives in `~/.nemo/<profile>.json`.
- Relay config can come from config or `NEMO_RELAY_URL` / `NEMO_RELAY_API_KEY`.

## Architecture

- `nemo/channel.py`: abstract user/channel boundary
- `nemo/coding_agent.py`: abstract coding-agent boundary
- `nemo/agent_factory.py`: agent kind → `CodingAgent` adapter, agent/model compatibility
- `nemo/lark_channel.py`: Lark-backed channel implementation
- `nemo/claude_agent.py`: Claude Agent SDK adapter (in-process Python SDK via `SDKThread`)
- `nemo/codex_agent.py`: Codex adapter that spawns the node sidecar per turn
- `codex_sidecar/run_turn.mjs`: node sidecar around `@openai/codex-sdk` — streams JSON events on stdout, reads prompt from stdin. Requires `node` and the `codex` CLI on `PATH`.
- `nemo/turn.py`: typed turn events and streaming turn runner
- `nemo/relay_events.py`: relay WebSocket / poll event source
- `nemo/lark/`: Lark API/auth/direct-event plumbing

## Lark Constraints

- Lark 长连接 is single-consumer per app. Prefer the relay server for real usage.
- Card V2 constraints matter here:
  - no `action` wrapper
  - no `note` tag
  - `collapsible_panel` headers must use `plain_text`
- `get_message` loses original card body content. Persistent config/state must not depend on reading interactive card bodies back.
- `<font color=…>…</font>` (and any inline HTML) cannot span a markdown paragraph break (`\n\n`): the open lands in one block, the close in the next, and Lark leaks a bare `</font>` into the rendered card. Keep a grey `_note_element` to a single line; render multi-line content as plain markdown. A literal `<name>`-style token in card text also opens a stray tag — write `NAME`, not `<name>`.
- Form submit (`form_action_type: "submit"`): Lark puts every *named* form child into `action.form_value` and may DROP the button's `action.value`. So (a) leave the submit button nameless or the single-field `form_value` becomes multi-field and the relay JSON-encodes it (breaking a `startswith(prefix)` route), and (b) the relay must fall back to `event.context.open_chat_id` / `open_message_id` for routing since `value.chat_id` can be missing. Encode the routing discriminator in the *select option value* (e.g. `model_switch:<name>`), not in the button.

## Debugging

- Pulling a chat's messages when analysing a bug — three sources, most-useful first:
  - **Daemon log (primary / forensic).** Map the chat to its daemon, then read the log:
    ```bash
    pid=$(cat ~/.nemo/pids/<chat_id>.pid)
    less ~/.nemo/logs/nemo-$pid.log
    ```
    Every inbound event is logged as `Event: type=… chat=… sender=… text=…` and every reply as `Response sent …`. This is the source of truth for what the daemon actually saw and did (including synthesised internal messages and card actions).
  - **Lark itself (the real conversation).** Use `lark-cli` (lark-im skill) to read what users actually sent, including interactive card bodies that `get_message` strips: `lark-cli im +chat-search` (find chat_id by group name), `+chat-messages-list --chat-id <cid>` (history, time range / pagination), `+messages-search` (keyword / sender / time), `+messages-mget` (batch by `om_` ids). Why lark-cli and not nemo's own code: `nemo/lark/api.py` only wraps the runtime primitives the daemon needs — single-message `get_message` (which also drops card bodies), plus send / update / download / members. It has **no** history-list (`GET /im/v1/messages?container_id=…`) or message-search wrapper, and Lark message *search* is user-identity-only (the daemon's bot token can't do it). lark-cli already wraps list + search across both identities, so it's the pragmatic forensic tool rather than adding a runtime-unused wrapper to nemo.
  - **Relay queue (live only).** `GET <relay_url>/replies/chat:<chat_id>?since=` (header `Authorization: Bearer <relay_api_key>`) returns only UN-consumed messages — an active daemon has already drained them, so this reflects the live queue / injected test events, not history.

## Error Handling

- Never `except Exception: pass` — always log the exception. Silent swallowing hides bugs (e.g. the zombie CLI subprocess bug was invisible because `__aexit__` failures were silently passed).

## Typing Rule

- Do not introduce `Any` in repo Python code.
- For opaque runtime handles, use `object` or a narrow `Protocol`.
- For SDK/channel boundaries, define the minimal protocol the caller actually needs.
- For JSON-like payloads, use shared aliases from `nemo/types.py` instead of `dict[str, Any]`.

## Validation

- Bug fixes must include a test case that covers the fix.
- Minimum regression pass after core changes:
  ```bash
  pytest tests/test_main.py tests/test_interfaces.py tests/test_permissions.py tests/test_turn.py -q
  ```
- Run relay-injection e2e when touching channel/event/permission flow:
  ```bash
  python3 -u scripts/e2e_test.py --skip-sdk
  python3 -u scripts/e2e_test.py --perm
  ```
  The e2e injects events via the relay and uses the bot/tenant token (app_id +
  app_secret) to send/read cards — it does NOT need a Lark **user** token. An
  "User token expired / refresh failed" line is expected and auto-falls back to
  relay; never stop to refresh it. Run with `-u` (stdout is block-buffered when
  piped — without it the run looks frozen until exit; don't kill it, SDK phases
  take minutes). See the `scripts/e2e_test.py` header for full details.
- Interactive card features (forms / dropdowns / buttons) must be tested at all three layers, not just the daemon. A daemon test that hand-builds `action_value={"action": …}` stubs away the wire format and will miss bugs in how Lark/the relay actually deliver the action (this is exactly how a `/model` picker form submit shipped broken — the relay dropped it because `value.chat_id` was empty). Cover:
  - `relay/test_relay.py` — POST a realistic webhook (try with AND without `action.value`, since Lark V2 form submits are flaky about preserving it).
  - `tests/test_relay_events.py` — round-trip the relay reply dict through `_relay_msg_to_event`.
  - `tests/test_agent.py` — the daemon main-loop handler.
  - `python3 scripts/e2e_test.py --picker` for the live `/model` picker chain.
  - `python3 scripts/e2e_test.py --recall-picker` for the `/session recall` picker. NB: the relay's `BOT_OWNED_CARD_PREFIXES` "Selected:"-flash suppression only takes effect once the **remote** relay (`/opt/nemo-relay/relay.py` on the configured `relay_url` host) is redeployed — repo `relay.py` is verified by `relay/test_relay.py`, but the live host can lag (same caveat as `--fork`). A stale remote relay shows as a SKIP, not a failure.
- `/fork` (read-only sub-thread) is thread-routed, so it has the same wire-format risk as the picker — the relay must forward `thread_id` or follow-ups never reach the fork. `python3 scripts/e2e_test.py --fork` covers the live round-trip; because the configured remote relay predates the `thread_id` forwarding fix, this phase spins up a LOCAL relay (repo `relay/relay.py`) for inbound events while outbound card sends still hit real Lark. The `/fork` message must anchor on a REAL Lark message id (Lark rejects reply-in-thread to a fabricated id); the phase reuses the daemon's start-card id. No user token needed — relay injection + the tenant token suffice.
