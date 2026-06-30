"""Tests for nemo.commands — built-in agent commands."""

import os
from unittest.mock import patch
import unittest.mock

from nemo.commands import try_dispatch, is_inline_safe, AgentContext


def _ctx():
  return AgentContext(model="opus", project_dir="/tmp/test", start_time=0)


def test_clear_commands():
  for cmd in ("/clear", "clear", "清空", "重置"):
    handled, resp = try_dispatch(cmd, _ctx())
    assert handled
    assert resp == "__clear__"


def test_undo_clear_commands():
  for cmd in ("/undo-clear", "/undoclear", "/undo", "撤销清空", "恢复"):
    handled, resp = try_dispatch(cmd, _ctx())
    assert handled, f"{cmd!r} should be handled"
    assert resp == "__undo_clear__", f"{cmd!r} → {resp!r}"


def test_undo_clear_not_inline_safe():
  """/undo-clear must NOT be inline-safe — it triggers _restart_client(resume=…)
  which can't run mid-turn."""
  _, resp = try_dispatch("/undo-clear", _ctx())
  assert not is_inline_safe(resp)


def test_restart_command_not_inline_safe():
  handled, resp = try_dispatch("/restart", _ctx())
  assert handled
  assert resp == "__restart__"
  assert not is_inline_safe(resp)


def test_upgrade_command_not_inline_safe():
  handled, resp = try_dispatch("/upgrade", _ctx())
  assert handled
  assert resp == "__upgrade__"
  assert not is_inline_safe(resp)


def test_upgrade_check_command_is_inline_safe():
  handled, resp = try_dispatch("/upgrade check", _ctx())
  assert handled
  assert resp == "__upgrade_check__"
  assert is_inline_safe(resp)


def test_session_rm_command():
  handled, resp = try_dispatch("/session rm abc123", _ctx())
  assert handled
  assert resp == "__session_rm__:abc123"


def test_session_purge_command_with_and_without_target():
  handled, resp = try_dispatch("/session purge abc123", _ctx())
  assert handled
  assert resp == "__session_purge__:abc123"

  handled, resp = try_dispatch("/session purge", _ctx())
  assert handled
  assert resp == "__session_purge__:"


def test_session_recall_command_with_uuid():
  handled, resp = try_dispatch("/session recall abc123", _ctx())
  assert handled
  assert resp == "__session_recall__:abc123"
  # Recall doesn't restart the SDK (it injects an internal turn), so it
  # stays inline-safe.
  assert is_inline_safe(resp)


def test_session_recall_no_arg_emits_picker():
  for cmd in ("/session recall", "/session recall   "):
    handled, resp = try_dispatch(cmd, _ctx())
    assert handled, cmd
    assert resp == "__session_picker__", f"{cmd!r} → {resp!r}"
    assert is_inline_safe(resp)


def test_session_info_command_with_and_without_target():
  handled, resp = try_dispatch("/session info abc123", _ctx())
  assert handled
  assert resp == "__session_info__:abc123"

  handled, resp = try_dispatch("/session info", _ctx())
  assert handled
  assert resp == "__session_info__:"
  assert is_inline_safe(resp)


def test_btw_with_question():
  handled, resp = try_dispatch("/btw what was that config file", _ctx())
  assert handled
  assert resp == "__btw__:what was that config file"
  # Side questions don't restart the SDK → must be inline-safe so they
  # can run during a turn without being deferred.
  assert is_inline_safe(resp)


def test_btw_multiline_question():
  handled, resp = try_dispatch("/btw line one\nline two", _ctx())
  assert handled
  assert resp == "__btw__:line one\nline two"


def test_btw_no_arg_shows_usage():
  handled, resp = try_dispatch("/btw", _ctx())
  assert handled
  assert resp is not None and not resp.startswith("__btw__:")
  assert "Usage:" in resp


def test_bare_btw_is_not_a_command():
  """Bare 'btw' (no slash) is too common in casual follow-ups to hijack."""
  handled, _ = try_dispatch("btw, also fix the typo above", _ctx())
  assert not handled


def test_fork_with_prompt():
  handled, resp = try_dispatch("/fork investigate the auth flow", _ctx())
  assert handled
  assert resp == "__fork__:investigate the auth flow"
  # Opening a fork spawns a separate SDK client (not a MAIN-client restart),
  # so it must be inline-safe to start concurrently during a running turn.
  assert is_inline_safe(resp)


def test_fork_multiline_prompt():
  handled, resp = try_dispatch("/fork line one\nline two", _ctx())
  assert handled
  assert resp == "__fork__:line one\nline two"


def test_fork_close():
  from nemo.commands import is_fork_close
  handled, resp = try_dispatch("/fork close", _ctx())
  assert handled
  assert resp == "__fork_close__"
  assert is_inline_safe(resp)
  assert is_fork_close("/fork close")
  assert is_fork_close("  /Fork   close  ")
  assert not is_fork_close("/fork closely look at this")


def test_fork_no_arg_shows_usage():
  handled, resp = try_dispatch("/fork", _ctx())
  assert handled
  assert resp is not None and not resp.startswith("__fork__:")
  assert "Usage:" in resp


def test_model_show_emits_picker_action():
  """Bare `/model` returns the picker action code — the main loop turns
  this into an interactive dropdown card. Old behaviour (text listing)
  is preserved as a fallback inside the picker handler."""
  handled, resp = try_dispatch("/model", _ctx())
  assert handled
  assert resp == "__model_picker__"


def test_model_picker_is_inline_safe():
  """The picker card is purely UI — no SDK restart required, so it must
  flow through the inline-safe path (mid-turn invocation also works)."""
  assert is_inline_safe("__model_picker__") is True


def test_model_switch():
  handled, resp = try_dispatch("/model sonnet", _ctx())
  assert handled
  assert resp == "__model__:sonnet"


def test_model_typo_rejected():
  """Unknown model must not emit __model__:, so SDK isn't restarted."""
  handled, resp = try_dispatch("/model sonet", _ctx())
  assert handled
  assert resp is not None and not resp.startswith("__model__:")
  assert "Unknown model" in resp
  assert "sonnet" in resp  # available list included


def test_agent_show_emits_picker_action():
  """Bare `/agent` returns the picker action code — the main loop turns
  this into an interactive dropdown card (mirrors `/model` no-arg)."""
  ctx = _ctx()
  ctx.agent = "claude"
  ctx.model = "claude-opus-4-7"
  handled, resp = try_dispatch("/agent", ctx)
  assert handled
  assert resp == "__agent_picker__"


def test_agent_picker_is_inline_safe():
  """The picker card is purely UI — sending it during an active turn is
  fine; only the form-submit triggers a switch (which IS NEEDS_SDK)."""
  assert is_inline_safe("__agent_picker__") is True


def test_agent_switch_emits_action_with_default_model():
  ctx = _ctx()
  ctx.agent = "claude"
  handled, resp = try_dispatch("/agent codex", ctx)
  assert handled
  # Action code carries both the new agent and its default model so
  # the loop doesn't have to look it up again. gpt-5.5 is the codex
  # default per agent_factory._DEFAULT_MODEL_BY_PROVIDER.
  assert resp == "__agent__:codex:gpt-5.5"


def test_agent_same_as_current_is_noop_message():
  ctx = _ctx()
  ctx.agent = "codex"
  handled, resp = try_dispatch("/agent codex", ctx)
  assert handled
  # No action code — just a message; main loop won't tear down the
  # agent for a no-op switch.
  assert resp is not None
  assert not resp.startswith("__")
  assert "Already" in resp


def test_agent_unknown_name_rejected():
  ctx = _ctx()
  handled, resp = try_dispatch("/agent gpt-nonexistent", ctx)
  assert handled
  assert resp is not None and not resp.startswith("__")
  assert "Unknown agent" in resp
  # Help points at the valid set.
  assert "claude" in resp


def test_agent_switch_to_claude_cli_accepted():
  ctx = _ctx()
  ctx.agent = "claude"
  handled, resp = try_dispatch("/agent claude-cli", ctx)
  assert handled
  assert resp is not None and resp.startswith("__agent__:claude-cli:")


def test_agent_action_code_marked_needs_sdk():
  # Agent switch tears down the SDK adapter, so it must be in the
  # _NEEDS_SDK guard list — otherwise the loop would try to handle it
  # mid-turn.
  assert is_inline_safe("__agent__:codex:gpt-5.5") is False


def test_version_command_reports_all_agent_runtimes():
  from nemo.version import VersionInfo

  with __import__("unittest").mock.patch(
      "nemo.commands.nemo_version_info",
      return_value=VersionInfo(
        version="0.4.17",
        source="source checkout",
        path="/repo/nemo",
        metadata_version="0.4.0",
      ),
  ), __import__("unittest").mock.patch(
      "nemo.commands._package_version",
      side_effect=lambda name: {
        "claude-agent-sdk": "0.1.55",
      }[name],
  ), __import__("unittest").mock.patch(
      "nemo.commands._cli_version",
      side_effect=lambda name: f"{name} cli 1.2.3",
  ), __import__("unittest").mock.patch(
      "nemo.commands._sidecar_dependency_version",
      side_effect=lambda _path, dep: {
        "@openai/codex-sdk": "0.128.0",
        "@opencode-ai/sdk": "^1.4.7",
      }[dep],
  ):
    handled, resp = try_dispatch("/version", _ctx())

  assert handled
  assert resp is not None
  assert is_inline_safe(resp)
  assert "Nemo" in resp and "0.4.17" in resp
  assert "source checkout" in resp and "/repo/nemo" in resp
  assert "metadata `0.4.0`" in resp
  assert "Claude" in resp and "0.1.55" in resp
  assert "Codex" in resp and "0.128.0" in resp
  assert "OpenCode" in resp and "^1.4.7" in resp


def test_pid_command_reports_current_process_id():
  handled, resp = try_dispatch("/pid", _ctx())
  assert handled
  assert resp == f"Nemo PID: `{os.getpid()}`"
  assert is_inline_safe(resp)


def test_model_typo_for_codex():
  from nemo.agent_factory import ModelCatalog
  ctx = _ctx()
  ctx.agent = "codex"
  with patch(
    "nemo.agent_factory.query_codex_model_catalog",
    return_value=ModelCatalog(visible=("gpt-5.5",)),
  ):
    handled, resp = try_dispatch("/model claude-sonnet-4-6", ctx)
  assert handled
  assert resp is not None and not resp.startswith("__model__:")
  assert "gpt-5.5" in resp


def test_model_list_for_codex_uses_dynamic_catalog():
  """Codex model listing comes from `codex debug models`, not a static Nemo list."""
  from nemo.agent_factory import ModelCatalog, model_catalog_for_agent
  from nemo.commands import _format_model_catalog
  with patch(
    "nemo.agent_factory.query_codex_model_catalog",
    return_value=ModelCatalog(
      visible=("gpt-5.5", "gpt-5.3-codex-spark"),
      hidden=("gpt-5-hidden",),
      note="Dynamic models from `codex debug models`.",
    ),
  ):
    catalog = model_catalog_for_agent("codex")
  listing = _format_model_catalog(catalog)

  assert "Available: `gpt-5.5`, `gpt-5.3-codex-spark`" in listing
  assert "Legacy: `gpt-5-hidden`" in listing
  assert "codex debug models" in listing


def test_model_list_for_opencode_shows_dynamic_note():
  """Opencode's dynamic-models note must be visible in the picker note
  block — the dropdown can't surface that nuance on its own."""
  from nemo.agent_factory import model_catalog_for_agent
  from nemo.commands import _format_model_catalog
  with patch(
      "nemo.opencode_agent.query_opencode_model_catalog_data",
      return_value=(("anthropic/claude-sonnet-4-5",), "Config default: `anthropic/claude-sonnet-4-5`."),
  ):
    catalog = model_catalog_for_agent("opencode")
    listing = _format_model_catalog(catalog)
    assert "anthropic/claude-sonnet-4-5" in listing
    assert "Config default" in listing


def test_model_switch_for_opencode_accepts_provider_slug_model():
  ctx = _ctx()
  ctx.agent = "opencode"
  with patch(
      "nemo.opencode_agent.query_opencode_model_catalog_data",
      return_value=(("anthropic/claude-sonnet-4-5",), "note"),
  ):
    handled, resp = try_dispatch("/model anthropic/claude-sonnet-4-5", ctx)
    assert handled
    assert resp == "__model__:anthropic/claude-sonnet-4-5"


def test_esc():
  for cmd in ("/esc", "esc", "cancel", "取消"):
    handled, resp = try_dispatch(cmd, _ctx())
    assert handled
    assert resp == "__esc__"


def test_esc_with_follow_up_text():
  handled, resp = try_dispatch("/esc fix the bug", _ctx())
  assert handled
  assert resp == "__esc__:fix the bug"


def test_esc_with_follow_up_preserves_case():
  handled, resp = try_dispatch("/esc Use TypeScript", _ctx())
  assert handled
  assert resp == "__esc__:Use TypeScript"


def test_esc_with_follow_up_not_inline_safe():
  # /esc <text> must NOT run as an inline command — it needs SDK restart
  # (cancel current turn then re-queue the follow-up text).
  from nemo.commands import is_inline_safe
  _, resp = try_dispatch("/esc do something", _ctx())
  assert not is_inline_safe(resp)


def test_cd_valid(tmp_path):
  handled, resp = try_dispatch(f"/cd {tmp_path}", _ctx())
  assert handled
  assert "__cd__:" in resp


def test_cd_invalid():
  handled, resp = try_dispatch("/cd /nonexistent/dir", _ctx())
  assert handled
  assert "not found" in resp


def test_ping():
  ctx = _ctx()
  ctx.msg_count = 5
  ctx.total_cost = 1.23
  handled, resp = try_dispatch("/ping", ctx)
  assert handled
  assert "Pong" in resp
  assert "opus" in resp


def test_cost():
  ctx = _ctx()
  ctx.total_cost = 0.5
  handled, resp = try_dispatch("/cost", ctx)
  assert handled
  assert "$0.5" in resp


def test_usage():
  handled, resp = try_dispatch("/usage", _ctx())
  assert handled
  assert "usage" in resp.lower()


def test_usage_for_opencode():
  ctx = _ctx()
  ctx.agent = "opencode"
  handled, resp = try_dispatch("/usage", ctx)
  assert handled
  assert "opencode stats" in resp


def test_tokens_unknown_before_first_snapshot():
  handled, resp = try_dispatch("/tokens", _ctx())
  assert handled
  assert resp is not None
  assert "unknown" in resp.lower()
  assert "Run one turn first" in resp
  assert is_inline_safe(resp)


def test_tokens_reports_last_usage_snapshot():
  ctx = _ctx()
  ctx.record_context_usage({
    "input_tokens": 12345,
    "output_tokens": 678,
    "cached_input_tokens": 1000,
    "context_window": 200000,
  }, updated_at=0)

  handled, resp = try_dispatch("/tokens", ctx)

  assert handled
  assert resp is not None
  assert "12,345 tokens" in resp
  assert "200,000 tokens (6.2%)" in resp
  assert "Last output: 678" in resp
  assert "Cached input: 1,000" in resp
  assert "last completed turn" in resp
  assert is_inline_safe(resp)


def test_tokens_reads_canonical_cache_read_key():
  # The canonical schema (turn.canonical_usage) reports cache reads under
  # cache_read_input_tokens; /context must read that, not just the legacy key.
  ctx = _ctx()
  ctx.record_context_usage({
    "input_tokens": 800,
    "cache_read_input_tokens": 1200,
    "cache_creation_input_tokens": 0,
    "output_tokens": 90,
    "total_tokens": 2090,
  }, updated_at=0)

  handled, resp = try_dispatch("/tokens", ctx)

  assert handled
  assert resp is not None
  assert "2,090 tokens" in resp  # current = total_tokens (per-turn footprint)
  assert "Cached input: 1,200" in resp
  assert "Last output: 90" in resp
  assert is_inline_safe(resp)


def test_tokens_prefers_total_tokens_when_present():
  ctx = _ctx()
  ctx.record_context_usage({
    "input_tokens": 1000,
    "total_tokens": 2345,
    "context_window": 10000,
  }, updated_at=0)

  handled, resp = try_dispatch("tokens", ctx)

  assert handled
  assert resp is not None
  assert "2,345 tokens" in resp
  assert "10,000 tokens (23.4%)" in resp


def test_tokens_reports_last_compact_snapshot():
  ctx = _ctx()
  ctx.record_compact(180000, 42000, updated_at=0)

  handled, resp = try_dispatch("/tokens", ctx)

  assert handled
  assert resp is not None
  assert "Last compact: 180,000" in resp
  assert "42,000" in resp


def test_tokens_clear_resets_snapshots():
  ctx = _ctx()
  ctx.record_context_usage({"input_tokens": 100}, updated_at=0)
  ctx.record_compact(200, 50, updated_at=0)

  ctx.clear_context_usage()
  handled, resp = try_dispatch("/tokens", ctx)

  assert handled
  assert resp is not None
  assert "unknown" in resp.lower()
  assert "Last compact" not in resp


def test_context_slash_is_not_nemo_command():
  handled, resp = try_dispatch("/context", _ctx())
  assert not handled
  assert resp is None


def test_help():
  handled, resp = try_dispatch("/help", _ctx())
  assert handled
  assert "Commands" in resp
  assert "/tokens" in resp
  assert "/context" not in resp


def test_autoapprove_on():
  handled, resp = try_dispatch("autoapprove on", _ctx())
  assert handled
  assert resp == "__autoapprove__:on"


def test_autoapprove_slash_on():
  handled, resp = try_dispatch("/autoapprove on", _ctx())
  assert handled
  assert resp == "__autoapprove__:on"


def test_autoapprove_off():
  handled, resp = try_dispatch("autoapprove off", _ctx())
  assert handled
  assert resp == "__autoapprove__:off"


def test_autoapprove_toggle():
  handled, resp = try_dispatch("/autoapprove", _ctx())
  assert handled
  assert resp == "__autoapprove_toggle__"


def test_autoesc_on():
  handled, resp = try_dispatch("autoesc on", _ctx())
  assert handled
  assert resp == "__autoesc__:on"


def test_autoesc_slash_on():
  handled, resp = try_dispatch("/autoesc on", _ctx())
  assert handled
  assert resp == "__autoesc__:on"


def test_autoesc_off():
  handled, resp = try_dispatch("autoesc off", _ctx())
  assert handled
  assert resp == "__autoesc__:off"


def test_autoesc_toggle():
  handled, resp = try_dispatch("/autoesc", _ctx())
  assert handled
  assert resp == "__autoesc_toggle__"


def test_exit():
  handled, resp = try_dispatch("/exit", _ctx())
  assert handled
  assert resp == "__exit__"


def test_dissolve():
  handled, resp = try_dispatch("/dissolve", _ctx())
  assert handled
  assert resp == "__dissolve__"


def test_dissolve_plain():
  handled, resp = try_dispatch("dissolve", _ctx())
  assert handled
  assert resp == "__dissolve__"


def test_not_a_command():
  handled, resp = try_dispatch("hello world", _ctx())
  assert not handled
  assert resp is None


def test_greedy_model_no_match():
  """'model trains are cool' should NOT match the /model command."""
  handled, resp = try_dispatch("model trains are cool", _ctx())
  assert not handled
  assert resp is None


def test_greedy_cd_no_match():
  """'cd collections on sale' should NOT match the /cd command."""
  handled, resp = try_dispatch("cd collections on sale", _ctx())
  assert not handled
  assert resp is None


def test_slash_model_matches():
  """/model sonnet IS a valid command."""
  handled, resp = try_dispatch("/model sonnet", _ctx())
  assert handled
  assert resp == "__model__:sonnet"


def test_slash_cd_matches(tmp_path):
  """/cd /tmp IS a valid command."""
  handled, resp = try_dispatch(f"/cd {tmp_path}", _ctx())
  assert handled
  assert "__cd__:" in resp


# ---------------------------------------------------------------------------
# /norm commands
# ---------------------------------------------------------------------------

def test_norm_add():
  handled, resp = try_dispatch("/norm add brevity Keep it short", _ctx())
  assert handled
  assert resp == "__norm_add__:brevity:Keep it short"


def test_norm_remove():
  handled, resp = try_dispatch("/norm remove brevity", _ctx())
  assert handled
  assert resp == "__norm_remove__:brevity"


def test_norm_list():
  handled, resp = try_dispatch("/norm list", _ctx())
  assert handled
  assert resp == "__norm_list__"


def test_norm_help():
  handled, resp = try_dispatch("/norm", _ctx())
  assert handled
  assert "Norm Commands" in resp


def test_norm_help_unknown_sub():
  handled, resp = try_dispatch("/norm unknown", _ctx())
  assert handled
  assert "Norm Commands" in resp


# ---------------------------------------------------------------------------
# /diag command
# ---------------------------------------------------------------------------

def test_diag():
  handled, resp = try_dispatch("/diag", _ctx())
  assert handled
  assert resp == "__diag__"


def test_diag_bare():
  handled, resp = try_dispatch("diag", _ctx())
  assert handled
  assert resp == "__diag__"


# ---------------------------------------------------------------------------
# /name command
# ---------------------------------------------------------------------------

def test_name_rename():
  handled, resp = try_dispatch("/name My Project", _ctx())
  assert handled
  assert resp == "__name__:My Project"


def test_name_single_word():
  handled, resp = try_dispatch("/name foo", _ctx())
  assert handled
  assert resp == "__name__:foo"


def test_name_empty_shows_usage():
  handled, resp = try_dispatch("/name", _ctx())
  assert handled
  assert "Usage" in resp


def test_name_whitespace_only_shows_usage():
  handled, resp = try_dispatch("/name   ", _ctx())
  assert handled
  assert "Usage" in resp


# ---------------------------------------------------------------------------
# /mention command
# ---------------------------------------------------------------------------

def test_mention_toggle():
  handled, resp = try_dispatch("/mention", _ctx())
  assert handled
  assert resp == "__mention_toggle__"


def test_mention_on():
  handled, resp = try_dispatch("/mention on", _ctx())
  assert handled
  assert resp == "__mention__:on"


def test_mention_off():
  handled, resp = try_dispatch("/mention off", _ctx())
  assert handled
  assert resp == "__mention__:off"


def test_mention_in_help():
  handled, resp = try_dispatch("/help", _ctx())
  assert handled
  assert "mention" in resp.lower()


# ---------------------------------------------------------------------------
# is_inline_safe — classify commands for during-turn execution
# ---------------------------------------------------------------------------

def test_inline_safe_ping():
  _, resp = try_dispatch("/ping", _ctx())
  assert is_inline_safe(resp)


def test_inline_safe_cost():
  _, resp = try_dispatch("/cost", _ctx())
  assert is_inline_safe(resp)


def test_inline_safe_help():
  _, resp = try_dispatch("/help", _ctx())
  assert is_inline_safe(resp)


def test_inline_safe_mention():
  _, resp = try_dispatch("/mention on", _ctx())
  assert is_inline_safe(resp)


def test_inline_safe_autoapprove():
  _, resp = try_dispatch("/autoapprove on", _ctx())
  assert is_inline_safe(resp)


def test_inline_safe_autoesc():
  _, resp = try_dispatch("/autoesc on", _ctx())
  assert is_inline_safe(resp)


def test_inline_safe_norm():
  _, resp = try_dispatch("/norm list", _ctx())
  assert is_inline_safe(resp)


def test_inline_safe_diag():
  _, resp = try_dispatch("/diag", _ctx())
  assert is_inline_safe(resp)


def test_inline_safe_name():
  _, resp = try_dispatch("/name New Group", _ctx())
  assert is_inline_safe(resp)
  assert resp == "__name__:New Group"


def test_not_inline_safe_clear():
  _, resp = try_dispatch("/clear", _ctx())
  assert not is_inline_safe(resp)


def test_not_inline_safe_model():
  _, resp = try_dispatch("/model sonnet", _ctx())
  assert not is_inline_safe(resp)


def test_not_inline_safe_cd(tmp_path):
  _, resp = try_dispatch(f"/cd {tmp_path}", _ctx())
  assert not is_inline_safe(resp)


def test_not_inline_safe_esc():
  _, resp = try_dispatch("/esc", _ctx())
  assert not is_inline_safe(resp)


def test_not_inline_safe_none():
  assert not is_inline_safe(None)


# ---------------------------------------------------------------------------
# /effort command
# ---------------------------------------------------------------------------

def test_effort_show_default():
  handled, resp = try_dispatch("/effort", _ctx())
  assert handled
  assert "default" in resp
  assert "SDK default" in resp
  assert "Usage" in resp


def test_effort_show_current():
  ctx = _ctx()
  ctx.effort = "high"
  handled, resp = try_dispatch("/effort", ctx)
  assert handled
  assert "high" in resp


def test_effort_set_low():
  handled, resp = try_dispatch("/effort low", _ctx())
  assert handled
  assert resp == "__effort__:low"


def test_effort_set_medium():
  handled, resp = try_dispatch("/effort medium", _ctx())
  assert handled
  assert resp == "__effort__:medium"


def test_effort_set_high():
  handled, resp = try_dispatch("/effort high", _ctx())
  assert handled
  assert resp == "__effort__:high"


def test_effort_set_max():
  handled, resp = try_dispatch("/effort max", _ctx())
  assert handled
  assert resp == "__effort__:max"


def test_effort_off_clears():
  for arg in ("off", "none", "clear", "default"):
    handled, resp = try_dispatch(f"/effort {arg}", _ctx())
    assert handled
    assert resp == "__effort__:"


def test_effort_invalid():
  handled, resp = try_dispatch("/effort maximum", _ctx())
  assert handled
  assert "Unknown" in resp


def test_effort_inline_safe():
  _, resp = try_dispatch("/effort low", _ctx())
  assert is_inline_safe(resp)


def _ctx_agent(agent):
  ctx = AgentContext(model="opus", project_dir="/tmp/test", start_time=0)
  ctx.agent = agent
  return ctx


def test_compact_forwards_only_for_claude_cli():
  handled, resp = try_dispatch("/compact", _ctx_agent("claude-cli"))
  assert handled and resp == "__forward__:/compact"
  # other agents: handled, but a "not supported" message (not the sentinel)
  handled, resp = try_dispatch("/compact", _ctx_agent("claude"))
  assert handled and resp is not None and "__forward__" not in resp


def test_usage_forwards_for_claude_cli_else_static():
  handled, resp = try_dispatch("/usage", _ctx_agent("claude-cli"))
  assert handled and resp == "__forward__:/usage"
  handled, resp = try_dispatch("/usage", _ctx_agent("claude"))
  assert handled and resp is not None and "claude.ai/settings/usage" in resp
