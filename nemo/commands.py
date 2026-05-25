"""Built-in agent commands — dispatched before SDK processing."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .types import JsonObject
from .version import get_version_info as nemo_version_info


_EFFORT_LEVELS = ("low", "medium", "high", "max")


def _package_version(name: str) -> str:
  try:
    return version(name)
  except PackageNotFoundError:
    return "not installed"


def _cli_version(command: str) -> str:
  exe = shutil.which(command)
  if not exe:
    return "not found"
  try:
    result = subprocess.run(
      [exe, "--version"],
      capture_output=True,
      text=True,
      timeout=3,
    )
  except (OSError, subprocess.TimeoutExpired) as exc:
    return f"unavailable ({type(exc).__name__})"
  output = (result.stdout or result.stderr).strip()
  if result.returncode != 0:
    return f"unavailable (exit {result.returncode})"
  return output or "unknown"


def _sidecar_dependency_version(package_json: Path, dependency: str) -> str:
  import json

  try:
    data = json.loads(package_json.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return "unknown"
  deps = data.get("dependencies", {}) if isinstance(data, dict) else {}
  raw = deps.get(dependency, "") if isinstance(deps, dict) else ""
  return str(raw) if isinstance(raw, str) and raw else "unknown"


def format_version_report() -> str:
  """Return local Nemo/agent runtime versions without touching any LLM."""
  root = Path(__file__).resolve().parent
  codex_package = root / "codex_sidecar" / "package.json"
  opencode_package = root / "opencode_sidecar" / "package.json"
  nemo_info = nemo_version_info()
  metadata_note = ""
  if nemo_info.metadata_version and nemo_info.metadata_version != nemo_info.version:
    metadata_note = f", metadata `{nemo_info.metadata_version}`"
  path_note = f", path `{nemo_info.path}`" if nemo_info.path else ""
  return (
    "**Versions**\n\n"
    f"- Nemo: `{nemo_info.version}` ({nemo_info.source}{path_note}{metadata_note})\n"
    f"- Claude: CLI `{_cli_version('claude')}`, SDK "
    f"`{_package_version('claude-agent-sdk')}`\n"
    f"- Codex: CLI `{_cli_version('codex')}`, sidecar SDK "
    f"`{_sidecar_dependency_version(codex_package, '@openai/codex-sdk')}`\n"
    f"- OpenCode: CLI `{_cli_version('opencode')}`, sidecar SDK "
    f"`{_sidecar_dependency_version(opencode_package, '@opencode-ai/sdk')}`"
  )


def _format_model_catalog(catalog) -> str:
  """Render a ModelCatalog as a markdown block grouped by visibility + aliases."""
  lines: list[str] = []
  if catalog.visible:
    lines.append("Available: " + ", ".join(f"`{m}`" for m in catalog.visible))
  if getattr(catalog, "api_only", ()):
    lines.append(
      "API-only (ChatGPT account rejects these): "
      + ", ".join(f"`{m}`" for m in catalog.api_only)
    )
  if catalog.hidden:
    lines.append("Legacy: " + ", ".join(f"`{m}`" for m in catalog.hidden))
  if catalog.aliases:
    pairs = ", ".join(f"`{a}` → `{full}`" for a, full in catalog.aliases.items())
    lines.append("Aliases: " + pairs)
  if getattr(catalog, "note", ""):
    lines.append("Note: " + catalog.note)
  return "\n".join(lines) if lines else "(no models configured)"

# What each effort level actually does per agent kind.
_EFFORT_DETAIL: dict[str, dict[str, str]] = {
  "claude": {
    "": "high (SDK default)",
    "low": "low effort",
    "medium": "medium effort",
    "high": "high effort",
    "max": "max effort (no token cap)",
  },
  "codex": {
    "": "medium in Codex",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "high (Codex tops out at high)",
  },
  "opencode": {
    "": "normal reasoning",
    "low": "lighter prompt guidance",
    "medium": "default",
    "high": "stronger prompt guidance",
    "max": "stronger prompt guidance (max → high)",
  },
}


class AgentContext:
  """Minimal context for command handlers."""

  def __init__(self, model: str, project_dir: str, start_time: float):
    from .agent_factory import AgentKind
    self.model = model
    self.project_dir = project_dir
    self.start_time = start_time
    self.msg_count = 0
    self.total_cost = 0.0
    self.effort = ""
    self.agent: AgentKind = "claude"
    self.context_usage: JsonObject = {}
    self.context_updated_at = 0.0
    self.context_source = ""
    self.compact_pre_tokens = 0
    self.compact_post_tokens = 0
    self.compact_updated_at = 0.0

  def record_context_usage(
    self,
    usage: JsonObject,
    *,
    source: str = "last completed turn",
    updated_at: float | None = None,
  ) -> None:
    self.context_usage = dict(usage)
    self.context_source = source
    self.context_updated_at = updated_at if updated_at is not None else time.time()

  def record_compact(
    self,
    pre_tokens: int,
    post_tokens: int,
    *,
    updated_at: float | None = None,
  ) -> None:
    self.compact_pre_tokens = pre_tokens
    self.compact_post_tokens = post_tokens
    self.compact_updated_at = updated_at if updated_at is not None else time.time()

  def clear_context_usage(self) -> None:
    self.context_usage = {}
    self.context_updated_at = 0.0
    self.context_source = ""
    self.compact_pre_tokens = 0
    self.compact_post_tokens = 0
    self.compact_updated_at = 0.0


def _int_value(value: object) -> int:
  if isinstance(value, bool):
    return 0
  if isinstance(value, int):
    return value
  if isinstance(value, float):
    return int(value)
  if isinstance(value, str):
    try:
      return int(value)
    except ValueError:
      return 0
  return 0


def _format_token_count(value: int) -> str:
  return f"{value:,}" if value > 0 else "unknown"


def _format_token_snapshot(ctx: AgentContext) -> str:
  if not ctx.context_usage and not ctx.compact_pre_tokens:
    return (
      "Tokens: **unknown**\n\n"
      "No token snapshot has been reported yet. Run one turn first; "
      "Nemo will show the last usage snapshot after the SDK reports it."
    )

  input_tokens = _int_value(ctx.context_usage.get("input_tokens"))
  output_tokens = _int_value(ctx.context_usage.get("output_tokens"))
  cached_tokens = _int_value(ctx.context_usage.get("cached_input_tokens"))
  total_tokens = _int_value(ctx.context_usage.get("total_tokens"))
  current_tokens = total_tokens or input_tokens
  context_window = _int_value(ctx.context_usage.get("context_window"))

  lines = ["**Tokens**"]
  lines.append(f"- Current: **{_format_token_count(current_tokens)} tokens**")
  if context_window:
    pct = current_tokens / context_window * 100 if current_tokens else 0
    lines.append(f"- Window: {context_window:,} tokens ({pct:.1f}%)")
  else:
    lines.append("- Window: unknown")
  if input_tokens:
    lines.append(f"- Last input: {input_tokens:,}")
  if output_tokens:
    lines.append(f"- Last output: {output_tokens:,}")
  if cached_tokens:
    lines.append(f"- Cached input: {cached_tokens:,}")
  if ctx.context_source:
    lines.append(f"- Source: {ctx.context_source}")
  if ctx.context_updated_at:
    lines.append(f"- Updated: {time.strftime('%H:%M:%S', time.localtime(ctx.context_updated_at))}")
  if ctx.compact_pre_tokens:
    if ctx.compact_post_tokens:
      lines.append(
        f"- Last compact: {ctx.compact_pre_tokens:,} → "
        f"{ctx.compact_post_tokens:,}"
      )
    else:
      lines.append(f"- Last compact: {ctx.compact_pre_tokens:,}")
  return "\n".join(lines)


# Each handler returns (handled: bool, response_text: str | None).
# If handled=True the main loop skips SDK processing.
# Special responses starting with __ are action codes handled by agent.py.


def try_dispatch(text: str, ctx: AgentContext) -> tuple[bool, str | None]:
  """Check text against all commands. Returns (handled, response)."""
  t = text.strip().lower()

  # /clear
  if t in ("/clear", "clear", "清空", "重置"):
    return True, "__clear__"

  # /undo-clear — restore the SDK session id that was active just before
  # the most recent /clear. Only works while the daemon is alive (the
  # previous id is held in process memory; not persisted to DB).
  if t in ("/undo-clear", "/undoclear", "/undo", "撤销清空", "恢复"):
    return True, "__undo_clear__"

  # /model
  if t.startswith("/model"):
    from .agent_factory import is_model_compatible, model_catalog_for_agent
    parts = text.strip().split(None, 1)
    if len(parts) >= 2:
      new_model = parts[1].strip()
      if not is_model_compatible(ctx.agent, new_model, ctx.project_dir):
        catalog = model_catalog_for_agent(ctx.agent, ctx.project_dir)
        listing = _format_model_catalog(catalog)
        return True, (
          f"Unknown model `{new_model}` for agent **{ctx.agent}**.\n\n"
          f"{listing}"
        )
      return True, f"__model__:{new_model}"
    # No argument — emit an interactive picker card. The main loop
    # builds the option list (so the card stays in sync with whatever
    # model_catalog_for_agent returns at click time) and sends the
    # actual card; here we only signal the action.
    return True, "__model_picker__"

  # /agent
  if t.startswith("/agent"):
    from .agent_factory import default_model_for_agent
    valid = ("claude", "codex", "opencode")
    parts = text.strip().split(None, 1)
    if len(parts) >= 2:
      arg = parts[1].strip().lower()
      if arg not in valid:
        return True, (
          f"Unknown agent `{arg}`. "
          f"Use `/agent claude|codex|opencode`."
        )
      if arg == ctx.agent:
        return True, f"Already on agent **{ctx.agent}**."
      default_model = default_model_for_agent(arg)  # type: ignore[arg-type]
      return True, f"__agent__:{arg}:{default_model}"
    return True, (
      f"Current agent: **{ctx.agent}** (model **{ctx.model}**)\n\n"
      f"Available: `claude`, `codex`, `opencode`. "
      f"Switching resets the model to that agent's default and "
      f"keeps each agent's last session id separately, so flipping "
      f"back resumes the prior conversation.\n\n"
      f"Usage: `/agent <name>`"
    )

  # /effort
  if t.startswith("/effort"):
    parts = text.strip().split(None, 1)
    if len(parts) >= 2:
      arg = parts[1].strip().lower()
      if arg in ("off", "none", "clear", "default"):
        return True, "__effort__:"
      if arg in _EFFORT_LEVELS:
        return True, f"__effort__:{arg}"
      return True, (
        f"Unknown effort level: `{arg}`. "
        f"Use `/effort low|medium|high|max|default`."
      )
    current = ctx.effort or "default"
    detail_map = _EFFORT_DETAIL.get(ctx.agent, {})
    detail = detail_map.get(ctx.effort, "")
    hint = f" — {detail}" if detail else ""
    return True, (
      f"Current effort: **{current}**{hint}\n\n"
      f"Usage: `/effort low|medium|high|max|default`"
    )

  # /btw <question> — side question. Read-only, single response, runs in a
  # forked copy of the session so it NEVER enters conversation history, and
  # does NOT interrupt a running turn. Mirrors Claude Code's `/btw`.
  # Require the leading slash: bare "btw" is far too common in casual
  # follow-ups ("btw, also fix X") to safely hijack as a command.
  btw_m = re.match(
    r"^/btw(?:\s+(.+))?$",
    text.strip(),
    re.IGNORECASE | re.DOTALL,
  )
  if btw_m:
    question = (btw_m.group(1) or "").strip()
    if not question:
      return True, (
        "Usage: `/btw <question>` — ask a quick, read-only question about "
        "the current work. The answer is ephemeral (never enters "
        "conversation history) and does not interrupt a running turn. "
        "Claude agent only."
      )
    return True, f"__btw__:{question}"

  # /fork <prompt> — open a read-only forked sub-thread (Claude only). Unlike
  # /btw it is multi-turn and tool-enabled (incl. Bash), but runs sandboxed so
  # it physically cannot modify the project. Lives in its own Lark sub-thread;
  # keep replying in the thread to continue it. `/fork close` ends it.
  fork_m = re.match(
    r"^/fork(?:\s+(.+))?$",
    text.strip(),
    re.IGNORECASE | re.DOTALL,
  )
  if fork_m:
    arg = (fork_m.group(1) or "").strip()
    if arg.lower() == "close":
      return True, "__fork_close__"
    if not arg:
      return True, (
        "Usage: `/fork <prompt>` — open a read-only branch in a Lark "
        "sub-thread. It shares the current conversation context and has "
        "tools (including Bash), but is sandboxed so it CANNOT modify "
        "project files. Multi-turn — just keep replying in the thread. "
        "`/fork close` ends it. Claude agent only."
      )
    return True, f"__fork__:{arg}"

  # /esc — bare cancels the current turn; with trailing text, cancel then
  # send the text as a new message (case preserved from the original input).
  esc_m = re.match(
    r"^(?:/esc|esc|cancel|取消)(?:\s+(.+))?$",
    text.strip(),
    re.IGNORECASE | re.DOTALL,
  )
  if esc_m:
    follow_up = (esc_m.group(1) or "").strip()
    if follow_up:
      return True, f"__esc__:{follow_up}"
    return True, "__esc__"

  # /cd
  if t.startswith("/cd "):
    parts = text.strip().split(None, 1)
    if len(parts) >= 2:
      new_dir = os.path.expanduser(parts[1].strip())
      if os.path.isdir(new_dir):
        return True, f"__cd__:{os.path.abspath(new_dir)}"
      return True, f"Directory not found: `{new_dir}`"
    return True, None

  # /ping
  if t in ("/ping", "ping"):
    uptime = int(time.time() - ctx.start_time)
    h, m = divmod(uptime, 3600)
    mins, secs = divmod(m, 60)
    return True, (
      f"🏓 Pong!\n"
      f"- Model: **{ctx.model}**\n"
      f"- Uptime: {h}h {mins}m {secs}s\n"
      f"- Messages: {ctx.msg_count}\n"
      f"- Cost: ${ctx.total_cost:.4f}\n"
      f"- CWD: `{ctx.project_dir}`"
    )

  # /cost
  if t in ("/cost", "cost"):
    return True, (
      f"💰 Session Cost\n"
      f"- Total: **${ctx.total_cost:.4f}**\n"
      f"- Messages: {ctx.msg_count}\n"
      f"- Model: {ctx.model}"
    )

  # /usage
  if t in ("/usage", "usage"):
    if ctx.agent == "claude":
      return True, "Plan usage: [claude.ai/settings/usage](https://claude.ai/settings/usage)"
    if ctx.agent == "opencode":
      return True, "Usage is agent-specific under OpenCode. Run `opencode stats` locally for totals."
    return True, "Usage is agent-specific. Check the local CLI/account UI for totals."

  # /tokens
  if t in ("/tokens", "tokens"):
    return True, _format_token_snapshot(ctx)

  # /version
  if t in ("/version", "version"):
    return True, format_version_report()

  # /pid
  if t in ("/pid", "pid"):
    return True, f"Nemo PID: `{os.getpid()}`"

  # /restart — replace this daemon process with a fresh nemo invocation.
  if t in ("/restart", "restart"):
    return True, "__restart__"

  # /upgrade check — check PyPI without installing.
  if t in ("/upgrade check", "upgrade check"):
    return True, "__upgrade_check__"

  # /upgrade — pipx upgrade captain-nemo, then restart if successful.
  if t in ("/upgrade", "upgrade"):
    return True, "__upgrade__"

  # /help
  if t in ("/help", "help", "帮助"):
    return True, (
      "**Agent Commands**\n\n"
      "| Command | Description |\n"
      "|---|---|\n"
      "| `/model` | Show current model |\n"
      "| `/model <name>` | Switch model |\n"
      "| `/agent` | Show current agent |\n"
      "| `/agent <claude\\|codex\\|opencode>` | Switch coding agent (resets to its default model) |\n"
      "| `/effort` | Show current reasoning effort |\n"
      "| `/effort <low\\|medium\\|high\\|max\\|default>` | Set reasoning effort |\n"
      "| `/clear` | Reset conversation |\n"
      "| `/undo-clear` | Restore session from last `/clear` (in-memory only) |\n"
      "| `/cd <dir>` | Change working directory |\n"
      "| `/esc` | Cancel current operation |\n"
      "| `/esc <text>` | Cancel and send `<text>` as the next message |\n"
      "| `/btw <question>` | Read-only side question; ephemeral, never enters history, doesn't interrupt the turn (Claude only) |\n"
      "| `/fork <prompt>` | Open a read-only branch in a sub-thread: shares context, has tools (incl. Bash), but sandboxed so it can't modify project files. Multi-turn; `/fork close` to end (Claude only) |\n"
      "| `/ping` | Status check |\n"
      "| `/cost` | Session API cost |\n"
      "| `/usage` | Plan usage limits |\n"
      "| `/tokens` | Last known context token snapshot |\n"
      "| `/version` | Nemo and coding-agent runtime versions |\n"
      "| `/pid` | Current Nemo process ID |\n"
      "| `/restart` | Restart this Nemo daemon |\n"
      "| `/upgrade check` | Check whether a newer Nemo is available |\n"
      "| `/upgrade` | Run `pipx upgrade captain-nemo`, then restart |\n"
      "| `/mention` | Toggle @mention requirement |\n"
      "| `/name <name>` | Rename this group |\n"
      "| `/norm` | Manage group norms |\n"
      "| `/guest` | Manage guests |\n"
      "| `/diag` | Run diagnostics |\n"
      "| `/session list` | List past sessions in this project |\n"
      "| `/session info [uuid]` | Show session metadata plus first and last messages |\n"
      "| `/session recall <uuid>` | Recall a past session's contents into context |\n"
      "| `/session rm <uuid>` | Remove one past session |\n"
      "| `/session purge [uuid]` | Remove sessions older than uuid, or all except current |\n"
      "| `/exit` | Stop agent, keep group |\n"
      "| `/dissolve` | Stop agent, dissolve group |\n"
      "| `/help` | This help |\n"
      "| `/autoapprove` | Toggle auto-approve |\n"
      "| `/autoesc` | Toggle auto-cancel current turn on new message |"
    )

  # /norm
  if t.startswith("/norm"):
    parts = text.strip().split(None, 3)
    if len(parts) >= 2:
      sub = parts[1].lower()
      if sub == "list":
        return True, "__norm_list__"
      if sub == "add" and len(parts) >= 4:
        return True, f"__norm_add__:{parts[2]}:{parts[3]}"
      if sub == "remove" and len(parts) >= 3:
        return True, f"__norm_remove__:{parts[2]}"
    return True, (
      "**Norm Commands**\n\n"
      "| Command | Description |\n"
      "|---|---|\n"
      "| `/norm add <name> <text>` | Add or update a norm |\n"
      "| `/norm remove <name>` | Remove a norm |\n"
      "| `/norm list` | List all norms |"
    )

  # /mention
  if t in ("/mention", "mention"):
    return True, "__mention_toggle__"
  if t in ("/mention on", "mention on"):
    return True, "__mention__:on"
  if t in ("/mention off", "mention off"):
    return True, "__mention__:off"

  # /name <new name>
  if t.startswith("/name"):
    parts = text.strip().split(None, 1)
    if len(parts) < 2 or not parts[1].strip():
      return True, "Usage: `/name <new group name>`"
    new_name = parts[1].strip()
    return True, f"__name__:{new_name}"

  # /diag
  if t in ("/diag", "diag"):
    return True, "__diag__"

  # /autoapprove
  if t in ("/autoapprove", "autoapprove"):
    return True, "__autoapprove_toggle__"
  if re.match(r"/?(?:auto[\s\-]*approve|autoapprove)\s+(on|off)", t):
    enabled = "on" in t.split()[-1]
    return True, f"__autoapprove__:{'on' if enabled else 'off'}"

  # /autoesc — when on, a new user message during an in-flight turn
  # automatically cancels (esc) the running turn and starts a new one.
  if t in ("/autoesc", "autoesc"):
    return True, "__autoesc_toggle__"
  if re.match(r"/?(?:auto[\s\-]*esc|autoesc)\s+(on|off)", t):
    enabled = "on" in t.split()[-1]
    return True, f"__autoesc__:{'on' if enabled else 'off'}"

  # /guest
  if t.startswith("/guest"):
    parts = text.strip().split()
    if len(parts) < 2:
      return True, (
        "**Guest Commands**\n\n"
        "| Command | Description |\n"
        "|---|---|\n"
        "| `/guest list` | List all guests |\n"
        "| `/guest add <name> [coowner]` | Add a guest (optionally as coowner) |\n"
        "| `/guest add all [coowner]` | Add every member (except operator) |\n"
        "| `/guest remove <name>` | Remove a guest |"
      )
    sub = parts[1].strip().lower()
    if sub == "list":
      return True, "__guest_list__"
    if sub == "add" and len(parts) >= 3:
      name = parts[2].strip()
      role = "guest"
      if len(parts) >= 4 and parts[3].strip().lower() == "coowner":
        role = "coowner"
      if name.lower() == "all":
        return True, f"__guest_add_all__:{role}"
      return True, f"__guest_add__:{role}:{name}"
    if sub == "remove" and len(parts) >= 3:
      name = parts[2].strip()
      return True, f"__guest_remove__:{name}"
    return True, "Usage: `/guest list`, `/guest add <name>`, `/guest remove <name>`"

  # /session list  |  /session recall <id>  |  /session rm <id>
  # /session purge [id]
  # Browse and recall past coding-agent sessions stored locally
  # (claude JSONL / codex rollout). "recall" is read-only — it injects
  # the session's contents as memory context, never SDK-resumes (that
  # would cross the per-endpoint isolation boundary).
  if t.startswith("/session"):
    parts = text.strip().split(None, 2)
    if len(parts) >= 2:
      sub = parts[1].lower()
      if sub == "list":
        return True, "__session_list__"
      if sub == "info":
        target = parts[2].strip() if len(parts) >= 3 else ""
        return True, f"__session_info__:{target}"
      if sub == "recall" and len(parts) >= 3:
        return True, f"__session_recall__:{parts[2].strip()}"
      if sub == "rm" and len(parts) >= 3:
        return True, f"__session_rm__:{parts[2].strip()}"
      if sub == "purge":
        target = parts[2].strip() if len(parts) >= 3 else ""
        return True, f"__session_purge__:{target}"
    return True, (
      "**Session Commands**\n\n"
      "| Command | Description |\n"
      "|---|---|\n"
      "| `/session list` | List past sessions in this project (Claude + Codex) |\n"
      "| `/session info [uuid]` | Show session metadata plus first and last messages |\n"
      "| `/session recall <uuid>` | Read a past session and remember its contents |\n"
      "| `/session rm <uuid>` | Remove one past session |\n"
      "| `/session purge [uuid]` | Remove sessions older than uuid, or all except current |"
    )

  # /dissolve — stop agent, dissolve group (check before /exit to avoid prefix match)
  if t in ("/dissolve", "dissolve"):
    return True, "__dissolve__"

  # /exit — stop agent, keep group
  if t in ("/exit", "exit"):
    return True, "__exit__"

  return False, None


# Commands that require SDK restart — NOT safe during a turn.
_NEEDS_SDK = ("__clear__", "__undo_clear__", "__esc__",
              "__model__:", "__cd__:", "__agent__:",
              "__restart__", "__upgrade__")


def is_fork_close(text: str) -> bool:
  """True if `text` is the `/fork close` command (sent inside a fork thread)."""
  return bool(re.match(r"^/fork\s+close\s*$", text.strip(), re.IGNORECASE))


def is_inline_safe(response: str | None) -> bool:
  """Check if a command response can be handled during an active turn.

  Returns True for commands that don't interact with the SDK client:
  /ping, /cost, /help, /mention, /autoapprove, /norm, /guest, /diag, etc.
  Returns False for /clear, /esc, /model, /cd (need SDK restart).
  """
  if not response:
    return False
  return not any(response.startswith(p) for p in _NEEDS_SDK)
