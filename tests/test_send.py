"""Tests for nemo.send — nemo-send CLI."""

from __future__ import annotations

import os
import sys
from unittest import mock

from nemo import send


_CREDS = {"app_id": "cli_app", "app_secret": "secret"}


def test_main_image_success(tmp_path, capsys):
  """image subcommand uploads + sends via lark_api with chat_id from env."""
  img_path = tmp_path / "shot.png"
  img_path.write_bytes(b"fake-png")

  with mock.patch.object(sys, "argv", ["nemo-send", "image", str(img_path)]):
    with mock.patch.dict(os.environ, {"NEMO_CHAT_ID": "oc_env"}, clear=False):
      with mock.patch("nemo.config.load_credentials", return_value=_CREDS):
        with mock.patch("nemo.lark.auth.get_token", return_value="tok"):
          with mock.patch(
            "nemo.lark.api.upload_image", return_value="img_key_1"
          ) as mock_up:
            with mock.patch(
              "nemo.lark.api.send_image", return_value="msg_1"
            ) as mock_send:
              rc = send.main()

  assert rc == 0
  mock_up.assert_called_once_with("tok", str(img_path))
  mock_send.assert_called_once_with("tok", "oc_env", "img_key_1")
  out = capsys.readouterr().out
  assert "msg_1" in out


def test_main_file_success(tmp_path, capsys):
  """file subcommand uploads + sends via lark_api."""
  doc_path = tmp_path / "doc.pdf"
  doc_path.write_bytes(b"%PDF-1.4")

  with mock.patch.object(sys, "argv", ["nemo-send", "file", str(doc_path)]):
    with mock.patch.dict(os.environ, {"NEMO_CHAT_ID": "oc_env"}, clear=False):
      with mock.patch("nemo.config.load_credentials", return_value=_CREDS):
        with mock.patch("nemo.lark.auth.get_token", return_value="tok"):
          with mock.patch(
            "nemo.lark.api.upload_file", return_value="file_key_1"
          ) as mock_up:
            with mock.patch(
              "nemo.lark.api.send_file", return_value="msg_2"
            ) as mock_send:
              rc = send.main()

  assert rc == 0
  mock_up.assert_called_once_with("tok", str(doc_path))
  mock_send.assert_called_once_with("tok", "oc_env", "file_key_1")
  out = capsys.readouterr().out
  assert "msg_2" in out


def test_main_missing_chat_id(tmp_path, capsys):
  """Returns 1 with stderr error when NEMO_CHAT_ID is not set."""
  img_path = tmp_path / "shot.png"
  img_path.write_bytes(b"fake")

  env = {k: v for k, v in os.environ.items() if k != "NEMO_CHAT_ID"}
  with mock.patch.object(sys, "argv", ["nemo-send", "image", str(img_path)]):
    with mock.patch.dict(os.environ, env, clear=True):
      rc = send.main()

  assert rc == 1
  err = capsys.readouterr().err
  assert "NEMO_CHAT_ID" in err


def test_main_missing_credentials(tmp_path, capsys):
  """Returns 1 with stderr error when no credentials configured."""
  img_path = tmp_path / "shot.png"
  img_path.write_bytes(b"fake")

  with mock.patch.object(sys, "argv", ["nemo-send", "image", str(img_path)]):
    with mock.patch.dict(os.environ, {"NEMO_CHAT_ID": "oc_env"}, clear=False):
      with mock.patch("nemo.config.load_credentials", return_value=None):
        rc = send.main()

  assert rc == 1
  err = capsys.readouterr().err
  assert "credentials" in err.lower()


def test_main_file_not_found(tmp_path, capsys):
  """Returns 1 when the given path does not exist."""
  missing = tmp_path / "does-not-exist.png"

  with mock.patch.object(sys, "argv", ["nemo-send", "image", str(missing)]):
    with mock.patch.dict(os.environ, {"NEMO_CHAT_ID": "oc_env"}, clear=False):
      with mock.patch("nemo.config.load_credentials", return_value=_CREDS):
        with mock.patch("nemo.lark.auth.get_token", return_value="tok"):
          rc = send.main()

  assert rc == 1
  err = capsys.readouterr().err
  assert "not found" in err.lower()


def test_main_no_subcommand(capsys):
  """No subcommand prints help and returns 1."""
  with mock.patch.object(sys, "argv", ["nemo-send"]):
    rc = send.main()

  assert rc == 1
  out = capsys.readouterr().out
  # argparse print_help writes usage/help to stdout
  assert "nemo-send" in out or "usage" in out.lower()
