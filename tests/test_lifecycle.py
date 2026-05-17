"""Tests for Nemo lifecycle restart/upgrade helpers."""

from unittest import mock

from nemo import lifecycle


def test_build_restart_args_forces_resolved_runtime_values():
  spec = lifecycle.RestartSpec(
    chat_id="oc_real",
    project_dir="/repo",
    agent="codex",
    model="gpt-5.5",
    permission_mode="acceptEdits",
    effort="high",
  )

  args = lifecycle.build_restart_args(
    spec,
    [
      "--chat-name", "old",
      "--project-dir=/old",
      "--agent", "claude",
      "--model", "opus",
      "--profile", "dev",
      "--verbose",
    ],
  )

  assert "--chat-name" not in args
  assert "/old" not in args
  assert args[-12:] == [
    "--chat-id", "oc_real",
    "--project-dir", "/repo",
    "--agent", "codex",
    "--model", "gpt-5.5",
    "--permission-mode", "acceptEdits",
    "--effort", "high",
  ]
  assert "--profile" in args
  assert "dev" in args
  assert "--verbose" in args


def test_is_editable_install_reads_direct_url():
  dist = mock.Mock()
  dist.read_text.return_value = '{"dir_info":{"editable":true}}'
  with mock.patch("nemo.lifecycle.distribution", return_value=dist):
    assert lifecycle.is_editable_install()


def test_is_editable_install_false_without_metadata():
  dist = mock.Mock()
  dist.read_text.return_value = None
  with mock.patch("nemo.lifecycle.distribution", return_value=dist):
    assert not lifecycle.is_editable_install()


def test_run_pipx_upgrade_success():
  completed = mock.Mock(returncode=0, stdout="upgraded\n")
  with mock.patch("nemo.lifecycle.subprocess.run", return_value=completed) as run:
    result = lifecycle.run_pipx_upgrade()

  assert result.returncode == 0
  assert result.output == "upgraded\n"
  run.assert_called_once()
  assert run.call_args.args[0] == ["pipx", "upgrade", "captain-nemo"]


def test_run_pipx_upgrade_missing_pipx():
  with mock.patch("nemo.lifecycle.subprocess.run", side_effect=FileNotFoundError):
    result = lifecycle.run_pipx_upgrade()

  assert result.returncode == 127
  assert "pipx" in result.output


def test_run_pipx_upgrade_timeout():
  exc = lifecycle.subprocess.TimeoutExpired(
    ["pipx", "upgrade", "captain-nemo"], timeout=300, output="partial"
  )
  with mock.patch("nemo.lifecycle.subprocess.run", side_effect=exc):
    result = lifecycle.run_pipx_upgrade()

  assert result.returncode == 124
  assert "Timed out" in result.output


def test_spawn_lifecycle_helper_returns_log_path(tmp_path):
  spec = lifecycle.RestartSpec(
    chat_id="oc_real",
    project_dir="/repo",
    agent="claude",
    model="opus",
    permission_mode="bypassPermissions",
  )
  with mock.patch("nemo.lifecycle.helper_log_path", return_value=str(tmp_path / "life.log")), \
       mock.patch("nemo.lifecycle.subprocess.Popen") as popen:
    log_path = lifecycle.spawn_lifecycle_helper(spec, original_args=[])

  assert log_path == str(tmp_path / "life.log")
  popen.assert_called_once()
  cmd = popen.call_args.args[0]
  assert cmd[:3] == [lifecycle.sys.executable, "-m", "nemo.lifecycle"]
  assert "--chat-id" in cmd


def test_run_helper_does_not_restart_when_upgrade_fails(tmp_path):
  failed = mock.Mock(returncode=1, stdout="bad\n")
  with mock.patch("nemo.lifecycle.subprocess.run", return_value=failed), \
       mock.patch("nemo.lifecycle.subprocess.Popen") as popen:
    rc = lifecycle._run_helper(999999, str(tmp_path / "life.log"), [], True)

  assert rc == 1
  popen.assert_not_called()
