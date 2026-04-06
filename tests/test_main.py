"""Tests for nemo.__main__ — CLI entry point."""

import logging
from unittest import mock

from nemo.__main__ import main


def test_no_chat_id_no_credentials():
  """Without --chat-id and no credentials, should return 1."""
  with mock.patch("sys.argv", ["nemo", "--project-dir", "/tmp"]):
    with mock.patch("nemo.__main__._ensure_sdk"):
      with mock.patch("nemo.config.load_credentials", return_value=None):
        result = main()
        assert result == 1


def test_no_chat_id_no_matching_group(tmp_path):
  """Without --chat-id and no matching group, should return 1."""
  with mock.patch("sys.argv", ["nemo", "--project-dir", str(tmp_path)]):
    with mock.patch("nemo.__main__._ensure_sdk"):
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
    # Mock _ensure_sdk to skip SDK check
    with mock.patch("nemo.__main__._ensure_sdk"):
      result = main()
      assert result == 1


def test_valid_args_calls_main_loop(tmp_path):
  """Should call main_loop with parsed args."""
  project = str(tmp_path)
  with mock.patch("sys.argv", ["nemo", "--chat-id", "oc_test",
                                "--project-dir", project,
                                "--model", "claude-sonnet-4-6"]):
    with mock.patch("nemo.__main__._ensure_sdk"):
      with mock.patch("nemo.config.load_credentials",
                      return_value={"app_id": "a", "app_secret": "s", "email": ""}):
        with mock.patch("nemo.preflight.run_preflight", return_value=[]):
          with mock.patch("nemo.__main__.asyncio") as mock_asyncio:
            mock_asyncio.run.return_value = 0
            result = main()
            assert result == 0
            mock_asyncio.run.assert_called_once()


def test_default_model(tmp_path):
  """Default model should be claude-opus-4-6."""
  project = str(tmp_path)
  with mock.patch("sys.argv", ["nemo", "--chat-id", "oc_1",
                                "--project-dir", project]):
    with mock.patch("nemo.__main__._ensure_sdk"):
      with mock.patch("nemo.config.load_credentials",
                      return_value={"app_id": "a", "app_secret": "s", "email": ""}):
        with mock.patch("nemo.preflight.run_preflight", return_value=[]):
          with mock.patch("nemo.__main__.asyncio") as mock_asyncio:
            mock_asyncio.run.return_value = 0
            main()
            mock_asyncio.run.assert_called_once()


def test_verbose_flag(tmp_path):
  """--verbose should set debug logging."""
  project = str(tmp_path)
  with mock.patch("sys.argv", ["nemo", "--chat-id", "oc_1",
                                "--project-dir", project, "-v"]):
    with mock.patch("nemo.__main__._ensure_sdk"):
      with mock.patch("nemo.config.load_credentials",
                      return_value={"app_id": "a", "app_secret": "s", "email": ""}):
        with mock.patch("nemo.preflight.run_preflight", return_value=[]):
          with mock.patch("nemo.__main__.asyncio") as mock_asyncio:
            with mock.patch("logging.basicConfig") as mock_logging:
              mock_asyncio.run.return_value = 0
              main()
              mock_logging.assert_called_once()
              assert mock_logging.call_args[1]["level"] == 10  # DEBUG
