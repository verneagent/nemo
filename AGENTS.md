# Nemo

Lark-connected coding agent daemon. Repo focus:
- Lark/relay-facing orchestration in Python
- coding-agent runtime behind `CodingAgent`
- channel I/O behind `Channel`

## Working Model

- Keep `nemo/agent.py` as orchestration only. Push provider-specific logic into concrete adapters such as `LarkChannel` and `ClaudeCodingAgent`.
- Prefer relay-backed event delivery. Direct Lark 长连接 is only a fallback when relay is not configured.
- Preserve the one-card-per-turn model: turn cards evolve through PATCH instead of emitting a new card for each phase.
- Keep turn execution event-driven. `run_turn()` should emit typed events and the main loop should react to them.

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
- `nemo/lark_channel.py`: Lark-backed channel implementation
- `nemo/claude_agent.py`: Claude-backed coding-agent implementation
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

## Typing Rule

- Do not introduce `Any` in repo Python code.
- For opaque runtime handles, use `object` or a narrow `Protocol`.
- For SDK/channel boundaries, define the minimal protocol the caller actually needs.
- For JSON-like payloads, use shared aliases from `nemo/types.py` instead of `dict[str, Any]`.

## Validation

- Minimum regression pass after core changes:
  ```bash
  pytest tests/test_main.py tests/test_interfaces.py tests/test_permissions.py tests/test_turn.py -q
  ```
- Run relay-injection e2e when touching channel/event/permission flow:
  ```bash
  python3 scripts/e2e_test.py --skip-sdk
  python3 scripts/e2e_test.py --perm
  ```
