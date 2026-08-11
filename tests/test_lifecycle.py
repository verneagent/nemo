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


def test_check_pypi_upgrade_reports_available():
  resp = mock.Mock()
  resp.read.return_value = b'{"info":{"version":"0.4.21"}}'
  resp.__enter__ = mock.Mock(return_value=resp)
  resp.__exit__ = mock.Mock(return_value=False)
  with mock.patch("nemo.lifecycle.urllib.request.urlopen", return_value=resp), \
       mock.patch("nemo.version.get_version_info") as version_info:
    version_info.return_value = mock.Mock(
      version="0.4.20", source="installed package")
    result = lifecycle.check_pypi_upgrade()

  assert result.returncode == 0
  assert result.current_version == "0.4.20"
  assert result.latest_version == "0.4.21"
  assert result.update_available is True
  assert "/upgrade" in result.output


def test_check_pypi_upgrade_source_checkout_note():
  resp = mock.Mock()
  resp.read.return_value = b'{"info":{"version":"0.4.20"}}'
  resp.__enter__ = mock.Mock(return_value=resp)
  resp.__exit__ = mock.Mock(return_value=False)
  with mock.patch("nemo.lifecycle.urllib.request.urlopen", return_value=resp), \
       mock.patch("nemo.version.get_version_info") as version_info:
    version_info.return_value = mock.Mock(
      version="0.4.20", source="source checkout")
    result = lifecycle.check_pypi_upgrade()

  assert result.returncode == 0
  assert result.update_available is False
  assert "source checkout" in result.output


def test_check_pypi_upgrade_current_newer_than_pypi():
  resp = mock.Mock()
  resp.read.return_value = b'{"info":{"version":"0.4.19"}}'
  resp.__enter__ = mock.Mock(return_value=resp)
  resp.__exit__ = mock.Mock(return_value=False)
  with mock.patch("nemo.lifecycle.urllib.request.urlopen", return_value=resp), \
       mock.patch("nemo.version.get_version_info") as version_info:
    version_info.return_value = mock.Mock(
      version="0.4.20", source="source checkout")
    result = lifecycle.check_pypi_upgrade()

  assert result.returncode == 0
  assert result.update_available is False
  assert "newer than PyPI" in result.output


def test_spawn_lifecycle_helper_returns_log_path(tmp_path):
  spec = lifecycle.RestartSpec(
    chat_id="oc_real",
    project_dir="/repo",
    agent="claude",
    model="opus",
    permission_mode="bypassPermissions",
  )
  with mock.patch("nemo.lifecycle.helper_log_path", return_value=str(tmp_path / "life.log")), \
       mock.patch("nemo.lifecycle.shell_profile_env", return_value={}), \
       mock.patch("nemo.lifecycle.subprocess.Popen") as popen:
    log_path = lifecycle.spawn_lifecycle_helper(spec, original_args=[])

  assert log_path == str(tmp_path / "life.log")
  popen.assert_called_once()
  cmd = popen.call_args.args[0]
  assert cmd[:3] == [lifecycle.sys.executable, "-m", "nemo.lifecycle"]
  assert "--chat-id" in cmd


def test_spawn_lifecycle_helper_overlays_profile_env(tmp_path):
  spec = lifecycle.RestartSpec(
    chat_id="oc_real",
    project_dir="/repo",
    agent="opencode",
    model="oc-deepseek-v4-flash",
    permission_mode="bypassPermissions",
  )
  with mock.patch("nemo.lifecycle.helper_log_path", return_value=str(tmp_path / "life.log")), \
       mock.patch("nemo.lifecycle.shell_profile_env",
                  return_value={"OPENCODE_GO_API_KEY": "sk-new"}), \
       mock.patch("nemo.lifecycle.subprocess.Popen") as popen:
    lifecycle.spawn_lifecycle_helper(spec, original_args=[])

  _, kwargs = popen.call_args
  assert kwargs["env"]["OPENCODE_GO_API_KEY"] == "sk-new"
  assert kwargs["env"]["HOME"]  # base os.environ is preserved
  assert "--model" in popen.call_args.args[0]


def test_run_helper_does_not_restart_when_upgrade_fails(tmp_path):
  failed = mock.Mock(returncode=1, stdout="bad\n")
  with mock.patch("nemo.lifecycle.subprocess.run", return_value=failed), \
       mock.patch("nemo.lifecycle.subprocess.Popen") as popen:
    rc = lifecycle._run_helper(999999, str(tmp_path / "life.log"), [], True)

  assert rc == 1
  popen.assert_not_called()


def _env_output(stdout: str, rc: int = 0) -> mock.Mock:
  return mock.Mock(returncode=rc, stdout=stdout, stderr="")


def test_shell_profile_env_returns_profile_keys(monkeypatch):
  out = (
    "HOME=/Users/x\n"
    "PATH=/opt/homebrew/bin:/usr/bin:/bin\n"
    "PWD=/Users/x\n"
    "SHLVL=1\n"
    "OPENCODE_GO_API_KEY=sk-new\n"
    "DEEPSEEK_API_KEY=sk-rotated\n"
    "FOO=bar\n"
  )
  monkeypatch.setenv("HOME", "/Users/x")
  with mock.patch("nemo.lifecycle.subprocess.run",
                  return_value=_env_output(out)) as run:
    overlay = lifecycle.shell_profile_env()

  assert overlay == {
    "OPENCODE_GO_API_KEY": "sk-new",
    "DEEPSEEK_API_KEY": "sk-rotated",
    "FOO": "bar",
  }
  # Runtime/session vars are excluded so the daemon's own values win.
  assert "PATH" not in overlay
  assert "PWD" not in overlay
  assert "SHLVL" not in overlay
  assert run.call_args.args[0] == [
    "/bin/zsh", "-c",
    '[[ -f "$HOME/.zprofile" ]] && source "$HOME/.zprofile" 2>/dev/null; env',
  ]


def test_shell_profile_env_skips_malformed_lines(monkeypatch):
  out = "OPENCODE_GO_API_KEY=sk-new\nnot-a-key=???\nFOO\n=x\n"
  monkeypatch.setenv("HOME", "/Users/x")
  with mock.patch("nemo.lifecycle.subprocess.run",
                  return_value=_env_output(out)):
    assert lifecycle.shell_profile_env() == {"OPENCODE_GO_API_KEY": "sk-new"}


def test_shell_profile_env_empty_without_home(monkeypatch):
  monkeypatch.delenv("HOME", raising=False)
  assert lifecycle.shell_profile_env() == {}


def test_shell_profile_env_falls_back_when_zsh_missing(monkeypatch):
  monkeypatch.setenv("HOME", "/Users/x")
  with mock.patch("nemo.lifecycle.subprocess.run",
                  side_effect=FileNotFoundError):
    assert lifecycle.shell_profile_env() == {}


def test_shell_profile_env_falls_back_when_profile_errors(monkeypatch):
  monkeypatch.setenv("HOME", "/Users/x")
  with mock.patch("nemo.lifecycle.subprocess.run",
                  return_value=_env_output("", rc=127)):
    assert lifecycle.shell_profile_env() == {}


def test_is_supervised_false_when_unset_or_explicit_off(monkeypatch):
  monkeypatch.delenv("NEMO_SUPERVISED", raising=False)
  assert not lifecycle.is_supervised()
  for value in ("0", "false", "no", "off", "none"):
    monkeypatch.setenv("NEMO_SUPERVISED", value)
    assert not lifecycle.is_supervised(), value


def test_is_supervised_true_for_any_truthy_value(monkeypatch):
  for value in ("1", "true", "yes", "launchd", "systemd", "supervisord", " 1 "):
    monkeypatch.setenv("NEMO_SUPERVISED", value)
    assert lifecycle.is_supervised(), value
