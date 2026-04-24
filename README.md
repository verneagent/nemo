# Captain Nemo

Lark-connected coding agent daemon. Runs either the Claude Agent SDK (default) or the OpenAI Codex SDK as the underlying coding agent.

## Install

```bash
pipx install captain-nemo
```

Or with pip:

```bash
pip install captain-nemo
```

## Upgrade

```bash
pipx upgrade captain-nemo
```

## Setup

Create `~/.nemo/config.json`:

```json
{
  "app_id": "cli_xxx",
  "app_secret": "xxx",
  "email": "your@email.com"
}
```

You need a Lark/Feishu custom app with IM permissions and WebSocket enabled (event subscription mode: long connection).

## Usage

```bash
# Auto-discover chat from workspace tag
nemo

# Specify project directory
nemo --project-dir /path/to/project

# Specify chat directly
nemo --chat-id oc_xxx

# Sidecar mode (only respond to @mentions)
nemo --sidecar --chat-id oc_xxx

# Use a different model
nemo --model claude-sonnet-4-6

# Run with Codex instead of Claude
nemo --provider codex
nemo --provider codex --model gpt-5-codex

# Start with a reasoning effort preset
nemo --effort high
nemo --provider codex --effort medium

# Debug logging
nemo -v
```

### Providers

| Provider | Default model | Runtime | Extra requirements |
|---|---|---|---|
| `claude` (default) | `claude-opus-4-7` | In-process Claude Agent SDK | `ANTHROPIC_API_KEY` or logged-in Claude credentials |
| `codex` | `gpt-5-codex` | Node sidecar (`codex_sidecar/run_turn.mjs`) around `@openai/codex-sdk` | `node`, the `codex` CLI on `PATH`, and `OPENAI_API_KEY` (or `CODEX_API_KEY`). Install sidecar deps with `npm --prefix codex_sidecar install`. |

Switch at startup with `--provider`, or at runtime with `/model <name>` — nemo auto-rejects models that don't match the current provider.

### Reasoning effort

`--effort low|medium|high` (or `/effort` at runtime) controls how much the underlying agent thinks before responding. The shared `low/medium/high` levels map per provider:

| nemo | Claude (keyword in prompt) | Codex (`modelReasoningEffort`) |
|---|---|---|
| `low` | `think` | `low` |
| `medium` | `think hard` | `medium` |
| `high` | `ultrathink` | `high` |
| `off` / unset | no keyword | SDK default |

`/effort off` clears the setting. Effort changes apply to the next turn — no SDK reconnect, no session loss.

On first run with `--chat-id`, nemo writes a `workspace:{machine}-{folder}` tag to the group description. Future runs auto-discover the chat without `--chat-id`.

## Commands

Send these in the Lark group:

| Command | Description |
|---|---|
| `/model [name]` | Show or switch model |
| `/effort [low\|medium\|high\|off]` | Show or set reasoning effort |
| `/clear` | Reset conversation |
| `/cd <dir>` | Change working directory |
| `/esc` | Cancel current operation |
| `/ping` | Status check |
| `/cost` | Session API cost |
| `/norm add/remove/list` | Manage group norms |
| `/guest add/remove/list` | Manage guests |
| `/diag` | Run diagnostics |
| `/help` | Show help |
| `/exit` | Stop agent, keep group |
| `/dissolve` | Stop agent, dissolve group |
| `autoapprove on/off` | Toggle auto-approve |

## Group Config

Nemo stores persistent group config (guests, autoapprove, norms) in a pinned card titled `__nemo_config__`.

## License

MIT
