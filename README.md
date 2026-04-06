# Captain Nemo

Lark-connected coding agent daemon powered by Claude Agent SDK.

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

# Debug logging
nemo -v
```

On first run with `--chat-id`, nemo writes a `workspace:{machine}-{folder}` tag to the group description. Future runs auto-discover the chat without `--chat-id`.

## Commands

Send these in the Lark group:

| Command | Description |
|---|---|
| `/model [name]` | Show or switch model |
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
