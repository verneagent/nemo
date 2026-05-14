"""Tests for nemo.commands — built-in agent commands."""

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


def test_model_show():
  handled, resp = try_dispatch("/model", _ctx())
  assert handled
  assert "opus" in resp


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


def test_agent_show_lists_options():
  ctx = _ctx()
  ctx.agent = "claude"
  ctx.model = "claude-opus-4-7"
  handled, resp = try_dispatch("/agent", ctx)
  assert handled
  assert resp is not None
  # Lists all three options + names current.
  assert "claude" in resp
  assert "codex" in resp
  assert "opencode" in resp
  assert "claude-opus-4-7" in resp


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


def test_agent_action_code_marked_needs_sdk():
  # Agent switch tears down the SDK adapter, so it must be in the
  # _NEEDS_SDK guard list — otherwise the loop would try to handle it
  # mid-turn.
  assert is_inline_safe("__agent__:codex:gpt-5.5") is False


def test_model_typo_for_codex():
  ctx = _ctx()
  ctx.agent = "codex"
  handled, resp = try_dispatch("/model claude-sonnet-4-6", ctx)
  assert handled
  assert resp is not None and not resp.startswith("__model__:")
  assert "gpt-5.5" in resp


def test_model_list_separates_chatgpt_from_api_only_codex():
  """Codex /model listing must warn that -codex slugs need API auth."""
  ctx = _ctx()
  ctx.agent = "codex"
  handled, resp = try_dispatch("/model", ctx)
  assert handled
  assert resp is not None
  # ChatGPT-safe defaults surface under Available.
  available_line = next(l for l in resp.split("\n") if l.startswith("Available:"))
  assert "gpt-5.5" in available_line
  assert "gpt-5.4" in available_line
  # The codex-specialized variants must be in a separate API-only bucket,
  # not mixed into the plain Available list.
  assert "gpt-5.3-codex" not in available_line
  api_line = next(l for l in resp.split("\n") if l.startswith("API-only"))
  assert "gpt-5.3-codex" in api_line
  assert "ChatGPT" in api_line  # explains why they're segregated


def test_model_list_for_opencode_shows_dynamic_note():
  ctx = _ctx()
  ctx.agent = "opencode"
  with __import__("unittest").mock.patch(
      "nemo.opencode_agent.query_opencode_model_catalog_data",
      return_value=(("anthropic/claude-sonnet-4-5",), "Config default: `anthropic/claude-sonnet-4-5`."),
  ):
    handled, resp = try_dispatch("/model", ctx)
    assert handled
    assert resp is not None
    assert "anthropic/claude-sonnet-4-5" in resp
    assert "Config default" in resp


def test_model_switch_for_opencode_accepts_provider_slug_model():
  ctx = _ctx()
  ctx.agent = "opencode"
  with __import__("unittest").mock.patch(
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


def test_help():
  handled, resp = try_dispatch("/help", _ctx())
  assert handled
  assert "Commands" in resp


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
