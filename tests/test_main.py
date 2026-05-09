"""Tests for nemo.__main__ — CLI entry point."""

from unittest import mock

from nemo.__main__ import main


def _fake_asyncio_run(coro):
  coro.close()
  return 0


def test_no_chat_id_no_credentials():
  """Without --chat-id and no credentials, should return 1."""
  with mock.patch("sys.argv", ["nemo", "--project-dir", "/tmp"]):
    with mock.patch("nemo.__main__._ensure_provider_runtime"):
      with mock.patch("nemo.config.load_credentials", return_value=None):
        result = main()
        assert result == 1


def test_no_chat_id_no_matching_group(tmp_path):
  """Without --chat-id and no matching group, should return 1."""
  with mock.patch("sys.argv", ["nemo", "--project-dir", str(tmp_path)]):
    with mock.patch("nemo.__main__._ensure_provider_runtime"):
      with mock.patch("nemo.config.load_credentials",
                      return_value={"app_id": "a", "app_secret": "s", "email": ""}):
        with mock.patch("nemo.lark.auth.get_token", return_value="tok"):
          with mock.patch("nemo.workspace.discover_chat_id", return_value=None):
            result = main()
          assert result == 1


def test_invalid_project_dir():
  """Should return 1 for non-existent directory."""
  with mock.patch("sys.argv", ["nemo", "--chat-id", "oc_1",
                                "--project-dir", "/nonexistent/path"]):
    with mock.patch("nemo.__main__._ensure_provider_runtime"):
      result = main()
      assert result == 1


def test_invalid_chat_id_rejected_at_argparse():
  """A chat_id that doesn't start with 'oc_' must fail argparse early
  rather than spawn a daemon that mis-evicts every other nemo on the
  host (see _cmdline_targets_chat regression history)."""
  import pytest
  with mock.patch("sys.argv", ["nemo", "--chat-id", "0",
                                "--project-dir", "/tmp"]):
    with pytest.raises(SystemExit) as exc:
      main()
    assert exc.value.code == 2  # argparse error exit code


def test_empty_chat_id_still_allowed():
  """Empty --chat-id means auto-discover; must not be rejected."""
  with mock.patch("sys.argv", ["nemo", "--chat-id", "", "--project-dir", "/tmp"]):
    with mock.patch("nemo.__main__._ensure_provider_runtime"):
      with mock.patch("nemo.config.load_credentials", return_value=None):
        result = main()
        assert result == 1  # fails later for missing creds, NOT at argparse


def test_valid_args_calls_main_loop(tmp_path):
  """Should call main_loop with parsed args."""
  project = str(tmp_path)
  with mock.patch("sys.argv", ["nemo", "--chat-id", "oc_test",
                                "--project-dir", project,
                                "--model", "claude-sonnet-4-6"]):
    with mock.patch("nemo.__main__._ensure_provider_runtime"):
      with mock.patch("nemo.config.load_credentials",
                      return_value={"app_id": "a", "app_secret": "s", "email": ""}):
        with mock.patch("nemo.preflight.run_preflight", return_value=[]):
          with mock.patch("nemo.__main__.asyncio") as mock_asyncio:
            mock_asyncio.run.side_effect = _fake_asyncio_run
            result = main()
            assert result == 0
            mock_asyncio.run.assert_called_once()


def test_default_model(tmp_path):
  """Default model should be claude-opus-4-7."""
  project = str(tmp_path)
  with mock.patch("sys.argv", ["nemo", "--chat-id", "oc_1",
                                "--project-dir", project]):
    with mock.patch("nemo.__main__._ensure_provider_runtime"):
      with mock.patch("nemo.config.load_credentials",
                      return_value={"app_id": "a", "app_secret": "s", "email": ""}):
        with mock.patch("nemo.preflight.run_preflight", return_value=[]):
          with mock.patch("nemo.__main__.asyncio") as mock_asyncio:
            mock_asyncio.run.side_effect = _fake_asyncio_run
            main()
            mock_asyncio.run.assert_called_once()


def test_verbose_flag(tmp_path):
  """--verbose should set debug logging."""
  project = str(tmp_path)
  with mock.patch("sys.argv", ["nemo", "--chat-id", "oc_1",
                                "--project-dir", project, "-v"]):
    with mock.patch("nemo.__main__._ensure_provider_runtime"):
      with mock.patch("nemo.config.load_credentials",
                      return_value={"app_id": "a", "app_secret": "s", "email": ""}):
        with mock.patch("nemo.preflight.run_preflight", return_value=[]):
          with mock.patch("nemo.__main__.asyncio") as mock_asyncio:
            with mock.patch("logging.basicConfig") as mock_logging:
              mock_asyncio.run.side_effect = _fake_asyncio_run
              main()
              mock_logging.assert_called_once()
              assert mock_logging.call_args[1]["level"] == 10  # DEBUG


def test_system_prompt_file_is_read_and_passed(tmp_path):
  project = str(tmp_path)
  sp_file = tmp_path / "sp.txt"
  sp_file.write_text("Respect the code style.\n")
  captured = {}

  def _capture_asyncio_run(coro):
    captured["frame"] = coro.cr_frame
    coro.close()
    return 0

  with mock.patch("sys.argv", ["nemo", "--chat-id", "oc_1",
                                "--project-dir", project,
                                "--system-prompt-file", str(sp_file)]):
    with mock.patch("nemo.__main__._ensure_provider_runtime"):
      with mock.patch("nemo.config.load_credentials",
                      return_value={"app_id": "a", "app_secret": "s", "email": ""}):
        with mock.patch("nemo.preflight.run_preflight", return_value=[]):
          with mock.patch("nemo.__main__.asyncio") as mock_asyncio:
            mock_asyncio.run.side_effect = _capture_asyncio_run
            result = main()
            assert result == 0
            frame = captured["frame"]
            assert frame is not None
            assert frame.f_locals["system_prompt"] == "Respect the code style."


def test_system_prompt_file_missing_returns_error(tmp_path):
  project = str(tmp_path)
  missing = str(tmp_path / "nope.txt")
  with mock.patch("sys.argv", ["nemo", "--chat-id", "oc_1",
                                "--project-dir", project,
                                "--system-prompt-file", missing]):
    with mock.patch("nemo.__main__._ensure_provider_runtime"):
      with mock.patch("nemo.config.load_credentials",
                      return_value={"app_id": "a", "app_secret": "s", "email": ""}):
        result = main()
        assert result == 1


def test_codex_provider_uses_provider_default_model(tmp_path):
  project = str(tmp_path)
  captured = {}

  def _capture_asyncio_run(coro):
    captured["frame"] = coro.cr_frame
    coro.close()
    return 0

  with mock.patch("sys.argv", ["nemo", "--chat-id", "oc_1",
                                "--project-dir", project,
                                "--provider", "codex"]):
    with mock.patch("nemo.__main__._ensure_provider_runtime"):
      with mock.patch("nemo.config.load_credentials",
                      return_value={"app_id": "a", "app_secret": "s", "email": ""}):
        with mock.patch("nemo.preflight.run_preflight", return_value=[]):
          with mock.patch("nemo.__main__.asyncio") as mock_asyncio:
            mock_asyncio.run.side_effect = _capture_asyncio_run
            result = main()
            assert result == 0
            frame = captured["frame"]
            assert frame is not None
            assert frame.f_locals["model"] == "gpt-5.5"
            assert frame.f_locals["provider"] == "codex"


def test_opencode_provider_uses_provider_default_model(tmp_path):
  project = str(tmp_path)
  captured = {}

  def _capture_asyncio_run(coro):
    captured["frame"] = coro.cr_frame
    coro.close()
    return 0

  with mock.patch("sys.argv", ["nemo", "--chat-id", "oc_1",
                                "--project-dir", project,
                                "--provider", "opencode"]):
    with mock.patch("nemo.__main__._ensure_provider_runtime"):
      with mock.patch("nemo.config.load_credentials",
                      return_value={"app_id": "a", "app_secret": "s", "email": ""}):
        with mock.patch("nemo.preflight.run_preflight", return_value=[]):
          with mock.patch("nemo.__main__.asyncio") as mock_asyncio:
            mock_asyncio.run.side_effect = _capture_asyncio_run
            result = main()
            assert result == 0
            frame = captured["frame"]
            assert frame is not None
            assert frame.f_locals["model"] == "default"
            assert frame.f_locals["provider"] == "opencode"


def test_init_exception_logged_and_returns_1(tmp_path, caplog):
  """Regression: an uncaught exception during early init (e.g. get_token
  raising on a network glitch) must reach the per-PID log file via the
  logging system, not just stderr — `nemobg` redirects stderr to
  /dev/null, so an unlogged init failure leaves a 0-byte log file and no
  trace of why the daemon "didn't start".
  """
  import logging

  project = str(tmp_path)
  with mock.patch("sys.argv", ["nemo", "--project-dir", project]):
    with mock.patch("nemo.__main__._ensure_provider_runtime"):
      with mock.patch("nemo.config.load_credentials",
                      return_value={"app_id": "a", "app_secret": "s", "email": ""}):
        # Simulate the production failure: get_token raises during workspace
        # discovery (the same RuntimeError as a Lark API hiccup).
        with mock.patch("nemo.lark.auth.get_token",
                        side_effect=RuntimeError("Token error: HTTP 502")):
          with caplog.at_level(logging.ERROR, logger="nemo"):
            result = main()

  assert result == 1
  startup_errors = [
    r for r in caplog.records
    if r.levelno >= logging.ERROR and "Startup failed" in r.getMessage()
  ]
  assert startup_errors, (
    f"expected a 'Startup failed' ERROR record, got: "
    f"{[(r.levelname, r.getMessage()) for r in caplog.records]}"
  )
  rec = startup_errors[0]
  msg = rec.getMessage()
  assert "RuntimeError" in msg
  assert "Token error" in msg
  # Traceback must be attached so the log file gets the full diagnostic
  assert rec.exc_info is not None and rec.exc_info[0] is RuntimeError


def test_preflight_failure_logged_and_returns_1(tmp_path, caplog):
  """Regression: 'clean return 1' paths (preflight errors, missing
  credentials, etc.) print to stderr and return — they don't raise. With
  nemobg redirecting stderr to /dev/null, those failures left a 0-byte
  per-PID log too. _startup_fail must dual-write to log + stderr.
  """
  import logging

  project = str(tmp_path)
  with mock.patch("sys.argv",
                  ["nemo", "--chat-id", "oc_x", "--project-dir", project]):
    with mock.patch("nemo.__main__._ensure_provider_runtime"):
      with mock.patch("nemo.config.load_credentials",
                      return_value={"app_id": "a", "app_secret": "s", "email": ""}):
        # Preflight returns errors (Lark API hiccup) — main should log
        # each one to per-PID file before returning 1.
        with mock.patch("nemo.preflight.run_preflight",
                        return_value=[
                          "Bot info check failed: Remote end closed connection",
                          "Chat access check failed for oc_x: timeout",
                        ]):
          with caplog.at_level(logging.ERROR, logger="nemo"):
            result = main()

  assert result == 1
  preflight_errors = [
    r for r in caplog.records
    if r.levelno >= logging.ERROR and "Preflight error" in r.getMessage()
  ]
  assert len(preflight_errors) == 2, (
    f"expected both preflight errors logged, got: "
    f"{[r.getMessage() for r in preflight_errors]}"
  )
  assert any("Bot info check failed" in r.getMessage() for r in preflight_errors)
  assert any("Chat access check failed" in r.getMessage() for r in preflight_errors)


def test_invalid_project_dir_logged(tmp_path, caplog):
  """Same dual-write requirement for the project-dir-not-a-directory path."""
  import logging
  with mock.patch("sys.argv",
                  ["nemo", "--chat-id", "oc_x",
                   "--project-dir", "/definitely/does/not/exist/anywhere"]):
    with mock.patch("nemo.__main__._ensure_provider_runtime"):
      with caplog.at_level(logging.ERROR, logger="nemo"):
        result = main()
  assert result == 1
  assert any("is not a directory" in r.getMessage()
             for r in caplog.records if r.levelno >= logging.ERROR), (
    "project-dir failure must be logged via logger so per-PID file captures it"
  )
