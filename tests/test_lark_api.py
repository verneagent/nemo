"""Tests for nemo.lark.api."""

import json
from unittest import mock

from nemo.lark import api


def _mock_response(data: dict, status: int = 200):
  """Create a mock urlopen response."""
  resp = mock.MagicMock()
  resp.read.return_value = json.dumps(data).encode()
  resp.__enter__ = mock.MagicMock(return_value=resp)
  resp.__exit__ = mock.MagicMock(return_value=False)
  return resp


def test_request_get():
  data = {"code": 0, "data": {"bot": {"open_id": "ou_123"}}}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    result = api._request("https://example.com/api", "token123")
  assert result["code"] == 0


def test_request_post():
  data = {"code": 0, "data": {"message_id": "om_123"}}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    result = api._request("https://example.com/api", "token123", {"key": "val"})
  assert result["data"]["message_id"] == "om_123"


def test_send_card_success():
  data = {"code": 0, "data": {"message_id": "om_card1"}}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    msg_id = api.send_card("tok", "chat1", {"schema": "2.0"})
  assert msg_id == "om_card1"


def test_send_card_retries():
  fail = {"code": 99, "msg": "error"}
  ok = {"code": 0, "data": {"message_id": "om_ok"}}
  responses = [_mock_response(fail), _mock_response(fail), _mock_response(ok)]
  with mock.patch("urllib.request.urlopen", side_effect=responses):
    with mock.patch("time.sleep"):
      msg_id = api.send_card("tok", "chat1", {})
  assert msg_id == "om_ok"


def test_send_card_all_retries_fail():
  fail = {"code": 99, "msg": "error"}
  responses = [_mock_response(fail)] * 3
  with mock.patch("urllib.request.urlopen", side_effect=responses):
    with mock.patch("time.sleep"):
      try:
        api.send_card("tok", "chat1", {})
        assert False, "Should raise"
      except RuntimeError as e:
        assert "Failed to send card" in str(e)


def test_update_card_success():
  data = {"code": 0}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    api.update_card("tok", "om_1", {"schema": "2.0"})


def test_update_card_failure():
  data = {"code": 99, "msg": "fail"}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    try:
      api.update_card("tok", "om_1", {})
      assert False, "Should raise"
    except RuntimeError:
      pass


def test_send_text_success():
  data = {"code": 0, "data": {"message_id": "om_text1"}}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    msg_id = api.send_text("tok", "chat1", "hello")
  assert msg_id == "om_text1"


def test_get_bot_info():
  data = {"code": 0, "bot": {"open_id": "ou_bot"}}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    info = api.get_bot_info("tok")
  assert info["open_id"] == "ou_bot"


def test_lookup_open_id_by_email():
  data = {"code": 0, "data": {"user_list": [{"user_id": "ou_user1"}]}}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    oid = api.lookup_open_id_by_email("tok", "user@test.com")
  assert oid == "ou_user1"


def test_lookup_open_id_not_found():
  data = {"code": 0, "data": {"user_list": [{}]}}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    oid = api.lookup_open_id_by_email("tok", "nope@test.com")
  assert oid is None


def test_add_reaction_silent_failure():
  """add_reaction should not raise on error."""
  with mock.patch("urllib.request.urlopen", side_effect=Exception("network")):
    api.add_reaction("tok", "om_1", "THUMBSUP")


def test_download_file_path_traversal():
  """download_file should strip directory components from file_name."""
  import os
  # Verify os.path.basename is used — ../../etc/passwd → passwd
  assert os.path.basename("../../etc/passwd") == "passwd"

  # Verify download_file constructs the right path (mock the HTTP call)
  fake_resp = mock.MagicMock()
  fake_resp.read.return_value = b"data"
  fake_resp.__enter__ = mock.MagicMock(return_value=fake_resp)
  fake_resp.__exit__ = mock.MagicMock(return_value=False)

  with mock.patch("urllib.request.urlopen", return_value=fake_resp), \
       mock.patch("nemo.config.tmp_dir", return_value="/tmp/nemo-test"), \
       mock.patch("os.makedirs"), \
       mock.patch("builtins.open", mock.mock_open()):
    path = api.download_file("tok", "om_1", "fk_1", "../../etc/passwd")
  # The resulting path should use only the basename, not the traversal path
  assert path.endswith("/passwd")
  assert "../../" not in path
