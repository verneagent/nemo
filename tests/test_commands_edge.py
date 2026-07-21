"""Edge case tests for nemo.commands — substring/boundary issues."""

from nemo.commands import try_dispatch, AgentContext


def _ctx():
  return AgentContext(model="opus", project_dir="/tmp/test", start_time=0)


def test_autoapprove_tone_not_matched():
  """'autoapprove tone' should NOT enable autoapprove — 'on' is a substring of 'tone'."""
  handled, resp = try_dispatch("autoapprove tone", _ctx())
  # The regex requires \\s+(on|off), so 'tone' should not match
  assert not handled


def test_autoapprove_on_with_spaces():
  handled, resp = try_dispatch("auto approve on", _ctx())
  assert not handled
  assert resp is None


def test_autoapprove_hyphen():
  handled, resp = try_dispatch("auto-approve off", _ctx())
  assert not handled
  assert resp is None


def test_exit_not_in_sentence():
  """'exit the building' should NOT trigger /exit."""
  handled, resp = try_dispatch("exit the building", _ctx())
  assert not handled


def test_dissolve_not_in_sentence():
  """'dissolve the salt' should NOT trigger /dissolve."""
  handled, resp = try_dispatch("dissolve the salt", _ctx())
  assert not handled


def test_cd_no_arg():
  """/cd with trailing space only — strip() turns it into '/cd' which doesn't match."""
  handled, _ = try_dispatch("/cd ", _ctx())
  # "/cd " strips to "/cd" which doesn't startswith "/cd " — not handled
  assert not handled


def test_norm_add_missing_text():
  """/norm add <name> without text should show help."""
  handled, resp = try_dispatch("/norm add brevity", _ctx())
  assert handled
  assert "Norm Commands" in resp


def test_guest_list():
  handled, resp = try_dispatch("/guest list", _ctx())
  assert handled
  assert resp == "__guest_list__"


def test_guest_add():
  handled, resp = try_dispatch("/guest add Alice", _ctx())
  assert handled
  assert resp == "__guest_add__:guest:Alice"


def test_guest_remove():
  handled, resp = try_dispatch("/guest remove Alice", _ctx())
  assert handled
  assert resp == "__guest_remove__:Alice"


def test_guest_help():
  handled, resp = try_dispatch("/guest", _ctx())
  assert handled
  assert "Guest Commands" in resp
