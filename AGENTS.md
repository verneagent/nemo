# Nemo

Lark-connected coding agent daemon. Repo focus:
- Lark/relay-facing orchestration in Python
- coding-agent runtime behind `CodingAgent` (Claude or Codex)
- channel I/O behind `Channel`

## Working Model

- Keep `nemo/agent.py` as orchestration only. Push provider-specific logic into concrete adapters such as `LarkChannel`, `ClaudeCodingAgent`, and `CodexCodingAgent`.
- `agent.py` is channel-agnostic and agent-agnostic. It only sees `Channel` and `CodingAgent` abstractions. Lark-specific logic (file download, message enrichment, API calls) belongs in `LarkChannel`. SDK-specific logic belongs in the concrete `CodingAgent` adapters.
- The coding agent is selected by `--provider claude|codex` (default `claude`). `nemo/agent_factory.py` maps provider → adapter and enforces provider/model compatibility.
- Reasoning effort is a shared `low/medium/high/max` knob (`--effort` at startup, `/effort` at runtime) exposed on `CodingAgent.set_effort`. Each adapter translates: `ClaudeCodingAgent` passes the value through the SDK's native `ClaudeAgentOptions.effort` parameter (claude-agent-sdk ≥ 0.1.50); `CodexCodingAgent` passes `--effort` to the sidecar, which sets `ThreadOptions.modelReasoningEffort` (clamps `max` → `high`, since the Codex SDK has no `max` tier); `OpenCodeCodingAgent` injects a prompt prefix (also clamps `max` → `high`). Because Claude's effort lives on SDK options rather than per-turn input, the host reconnects with `resume=<sdk_session_id>` after `/effort` so the new value takes effect on the next turn — session context is preserved across the reconnect.
- Prefer relay-backed event delivery. Direct Lark 长连接 is only a fallback when relay is not configured.
- Preserve the one-card-per-turn model: turn cards evolve through PATCH instead of emitting a new card for each phase.
- Keep turn execution event-driven. `run_turn()` should emit typed events and the main loop should react to them.
- Stop/esc only interrupts the current turn — do not reset or restart the SDK client. Session and conversation context must be preserved (match CLI Escape behavior).

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
- `nemo/agent_factory.py`: provider → `CodingAgent` adapter, provider/model compatibility
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
  python3 scripts/e2e_test.py --skip-sdk
  python3 scripts/e2e_test.py --perm
  ```
