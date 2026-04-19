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
  """Default model should be claude-opus-4-6."""
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
            assert frame.f_locals["model"] == "gpt-5.4"
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
