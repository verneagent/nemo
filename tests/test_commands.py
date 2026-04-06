"""Tests for nemo.commands — built-in agent commands."""

from nemo.commands import try_dispatch, AgentContext


def _ctx():
  return AgentContext(model="opus", project_dir="/tmp/test", start_time=0)


def test_clear_commands():
  for cmd in ("/clear", "clear", "清空", "重置"):
    handled, resp = try_dispatch(cmd, _ctx())
    assert handled
    assert resp == "__clear__"


def test_model_show():
  handled, resp = try_dispatch("/model", _ctx())
  assert handled
  assert "opus" in resp


def test_model_switch():
  handled, resp = try_dispatch("/model sonnet", _ctx())
  assert handled
  assert resp == "__model__:sonnet"


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


def test_help():
  handled, resp = try_dispatch("/help", _ctx())
  assert handled
  assert "Commands" in resp


def test_autoapprove_on():
  handled, resp = try_dispatch("autoapprove on", _ctx())
  assert handled
  assert resp == "__autoapprove__:on"


def test_autoapprove_off():
  handled, resp = try_dispatch("autoapprove off", _ctx())
  assert handled
  assert resp == "__autoapprove__:off"


def test_exit():
  handled, resp = try_dispatch("/exit", _ctx())
  assert handled
  assert resp == "__exit__"


def test_exit_handback_compat():
  handled, resp = try_dispatch("handback", _ctx())
  assert handled
  assert resp == "__exit__"


def test_dissolve():
  handled, resp = try_dispatch("/dissolve", _ctx())
  assert handled
  assert resp == "__dissolve__"


def test_dissolve_handback_compat():
  handled, resp = try_dispatch("handback dissolve", _ctx())
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
