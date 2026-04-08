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


# ---------------------------------------------------------------------------
# get_message
# ---------------------------------------------------------------------------

def test_get_message_success():
  data = {"code": 0, "data": {"items": [{"message_id": "om_1", "body": {}}]}}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    result = api.get_message("tok", "om_1")
  assert result["message_id"] == "om_1"


def test_get_message_not_found():
  data = {"code": 0, "data": {"items": []}}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    result = api.get_message("tok", "om_missing")
  assert result == {}


def test_get_message_error():
  data = {"code": 99, "msg": "not found"}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    result = api.get_message("tok", "om_1")
  assert result == {}


# ---------------------------------------------------------------------------
# delete_message
# ---------------------------------------------------------------------------

def test_delete_message():
  data = {"code": 0}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    api.delete_message("tok", "om_1")  # Should not raise


# ---------------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------------

def test_create_pin_success():
  data = {"code": 0}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    api.create_pin("tok", "om_1")


def test_create_pin_failure():
  data = {"code": 99, "msg": "fail"}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    try:
      api.create_pin("tok", "om_1")
      assert False, "Should raise"
    except RuntimeError:
      pass


def test_delete_pin():
  data = {"code": 0}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    api.delete_pin("tok", "om_1")


def test_list_pins_single_page():
  data = {"code": 0, "data": {
    "items": [{"pin_id": "p1"}, {"pin_id": "p2"}],
    "has_more": False,
  }}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    pins = api.list_pins("tok", "oc_1")
  assert len(pins) == 2


def test_list_pins_pagination():
  page1 = {"code": 0, "data": {
    "items": [{"pin_id": "p1"}],
    "has_more": True,
    "page_token": "tok2",
  }}
  page2 = {"code": 0, "data": {
    "items": [{"pin_id": "p2"}],
    "has_more": False,
  }}
  with mock.patch("urllib.request.urlopen",
                  side_effect=[_mock_response(page1), _mock_response(page2)]):
    pins = api.list_pins("tok", "oc_1")
  assert len(pins) == 2


def test_list_pins_error():
  data = {"code": 99}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    pins = api.list_pins("tok", "oc_1")
  assert pins == []


# ---------------------------------------------------------------------------
# Reply message / reply card
# ---------------------------------------------------------------------------

def test_reply_message_success():
  data = {"code": 0, "data": {"message_id": "om_reply1"}}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    msg_id = api.reply_message("tok", "om_parent", "reply text")
  assert msg_id == "om_reply1"


def test_reply_message_failure():
  data = {"code": 99, "msg": "fail"}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    try:
      api.reply_message("tok", "om_parent", "text")
      assert False, "Should raise"
    except RuntimeError:
      pass


def test_reply_card_success():
  data = {"code": 0, "data": {"message_id": "om_rc1"}}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    msg_id = api.reply_card("tok", "om_parent", {"schema": "2.0"})
  assert msg_id == "om_rc1"


def test_reply_card_failure():
  data = {"code": 99, "msg": "fail"}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    try:
      api.reply_card("tok", "om_parent", {})
      assert False, "Should raise"
    except RuntimeError:
      pass


# ---------------------------------------------------------------------------
# Chat tabs
# ---------------------------------------------------------------------------

def test_create_chat_tab_success():
  data = {"code": 0, "data": {"chat_tabs": [{"tab_id": "tab_1"}]}}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    tab_id = api.create_chat_tab("tok", "oc_1", "Docs", "https://docs.example.com")
  assert tab_id == "tab_1"


def test_create_chat_tab_failure():
  data = {"code": 99, "msg": "fail"}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    try:
      api.create_chat_tab("tok", "oc_1", "Docs", "https://docs.example.com")
      assert False, "Should raise"
    except RuntimeError:
      pass


def test_delete_chat_tab():
  data = {"code": 0}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    api.delete_chat_tab("tok", "oc_1", ["tab_1", "tab_2"])


def test_delete_chat_tab_failure():
  data = {"code": 99, "msg": "fail"}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    try:
      api.delete_chat_tab("tok", "oc_1", ["tab_1"])
      assert False, "Should raise"
    except RuntimeError:
      pass


def test_update_chat_tab():
  data = {"code": 0}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    api.update_chat_tab("tok", "oc_1", "tab_1", "New Name", "https://new.url")


def test_update_chat_tab_failure():
  data = {"code": 99}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    try:
      api.update_chat_tab("tok", "oc_1", "tab_1", "N", "https://x")
      assert False, "Should raise"
    except RuntimeError:
      pass


def test_list_chat_tabs_success():
  data = {"code": 0, "data": {"chat_tabs": [
    {"tab_id": "t1", "tab_name": "Docs"},
    {"tab_id": "t2", "tab_name": "CI"},
  ]}}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    tabs = api.list_chat_tabs("tok", "oc_1")
  assert len(tabs) == 2


def test_list_chat_tabs_failure():
  data = {"code": 99}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    try:
      api.list_chat_tabs("tok", "oc_1")
      assert False, "Should raise"
    except RuntimeError:
      pass


def test_sort_chat_tabs():
  data = {"code": 0}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    api.sort_chat_tabs("tok", "oc_1", ["t2", "t1"])


def test_sort_chat_tabs_failure():
  data = {"code": 99}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    try:
      api.sort_chat_tabs("tok", "oc_1", ["t1"])
      assert False, "Should raise"
    except RuntimeError:
      pass


# ---------------------------------------------------------------------------
# Stickers
# ---------------------------------------------------------------------------

def test_send_sticker_success():
  data = {"code": 0, "data": {"message_id": "om_stk1"}}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    msg_id = api.send_sticker("tok", "oc_1", "stk_abc")
  assert msg_id == "om_stk1"


def test_send_sticker_failure():
  data = {"code": 99}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    try:
      api.send_sticker("tok", "oc_1", "stk_abc")
      assert False, "Should raise"
    except RuntimeError:
      pass


def test_reply_sticker_success():
  data = {"code": 0, "data": {"message_id": "om_rstk1"}}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    msg_id = api.reply_sticker("tok", "om_parent", "stk_abc")
  assert msg_id == "om_rstk1"


def test_reply_sticker_failure():
  data = {"code": 99}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    try:
      api.reply_sticker("tok", "om_parent", "stk_abc")
      assert False, "Should raise"
    except RuntimeError:
      pass


# ---------------------------------------------------------------------------
# Reactions
# ---------------------------------------------------------------------------

def test_add_reaction_success():
  data = {"code": 0, "data": {"reaction_id": "r_1"}}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    rid = api.add_reaction("tok", "om_1", "THUMBSUP")
  assert rid == "r_1"


def test_remove_reaction_success():
  data = {"code": 0}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    api.remove_reaction("tok", "om_1", "r_1")  # Should not raise


def test_remove_reaction_error_swallowed():
  """remove_reaction should not raise on failure."""
  with mock.patch("urllib.request.urlopen", side_effect=Exception("network")):
    api.remove_reaction("tok", "om_1", "r_1")


# ---------------------------------------------------------------------------
# Send image / Send file
# ---------------------------------------------------------------------------

def test_send_image_success():
  data = {"code": 0, "data": {"message_id": "om_img1"}}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    msg_id = api.send_image("tok", "oc_1", "img_key_abc")
  assert msg_id == "om_img1"


def test_send_image_failure():
  data = {"code": 99}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    try:
      api.send_image("tok", "oc_1", "img_key_abc")
      assert False, "Should raise"
    except RuntimeError:
      pass


def test_send_file_success():
  data = {"code": 0, "data": {"message_id": "om_file1"}}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    msg_id = api.send_file("tok", "oc_1", "fk_abc")
  assert msg_id == "om_file1"


def test_send_file_failure():
  data = {"code": 99}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    try:
      api.send_file("tok", "oc_1", "fk_abc")
      assert False, "Should raise"
    except RuntimeError:
      pass


# ---------------------------------------------------------------------------
# Upload image / Upload file (multipart)
# ---------------------------------------------------------------------------

def test_upload_image_success(tmp_path):
  img = tmp_path / "test.png"
  img.write_bytes(b"\x89PNG\r\n\x1a\nfakedata")
  data = {"code": 0, "data": {"image_key": "img_uploaded"}}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    key = api.upload_image("tok", str(img))
  assert key == "img_uploaded"


def test_upload_image_failure(tmp_path):
  img = tmp_path / "test.png"
  img.write_bytes(b"data")
  data = {"code": 99, "msg": "fail"}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    try:
      api.upload_image("tok", str(img))
      assert False, "Should raise"
    except RuntimeError:
      pass


def test_upload_file_success(tmp_path):
  f = tmp_path / "doc.pdf"
  f.write_bytes(b"%PDF-1.4 fake")
  data = {"code": 0, "data": {"file_key": "fk_uploaded"}}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    key = api.upload_file("tok", str(f))
  assert key == "fk_uploaded"


def test_upload_file_failure(tmp_path):
  f = tmp_path / "doc.pdf"
  f.write_bytes(b"data")
  data = {"code": 99, "msg": "fail"}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    try:
      api.upload_file("tok", str(f))
      assert False, "Should raise"
    except RuntimeError:
      pass


# ---------------------------------------------------------------------------
# Chat operations (create, dissolve, members)
# ---------------------------------------------------------------------------

def test_create_chat_success():
  data = {"code": 0, "data": {"chat_id": "oc_new"}}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    cid = api.create_chat("tok", "Test Group", description="desc")
  assert cid == "oc_new"


def test_create_chat_failure():
  data = {"code": 99}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    try:
      api.create_chat("tok", "Test")
      assert False, "Should raise"
    except RuntimeError:
      pass


def test_dissolve_chat():
  data = {"code": 0}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    api.dissolve_chat("tok", "oc_1")


def test_dissolve_chat_failure():
  data = {"code": 99}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    try:
      api.dissolve_chat("tok", "oc_1")
      assert False, "Should raise"
    except RuntimeError:
      pass


def test_add_chat_members():
  data = {"code": 0}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    api.add_chat_members("tok", "oc_1", ["ou_1", "ou_2"])


def test_add_chat_members_failure():
  data = {"code": 99}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    try:
      api.add_chat_members("tok", "oc_1", ["ou_1"])
      assert False, "Should raise"
    except RuntimeError:
      pass


def test_remove_chat_members():
  data = {"code": 0}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    api.remove_chat_members("tok", "oc_1", ["ou_1"])


def test_remove_chat_members_failure():
  data = {"code": 99}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    try:
      api.remove_chat_members("tok", "oc_1", ["ou_1"])
      assert False, "Should raise"
    except RuntimeError:
      pass


# ---------------------------------------------------------------------------
# Chat info operations
# ---------------------------------------------------------------------------

def test_get_chat_info_success():
  data = {"code": 0, "data": {"name": "Test", "description": "desc"}}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    info = api.get_chat_info("tok", "oc_1")
  assert info["name"] == "Test"


def test_get_chat_info_error():
  data = {"code": 99}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    info = api.get_chat_info("tok", "oc_1")
  assert info == {}


def test_update_chat_info():
  data = {"code": 0}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    api.update_chat_info("tok", "oc_1", {"description": "new"})


def test_update_chat_info_failure():
  data = {"code": 99}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)):
    try:
      api.update_chat_info("tok", "oc_1", {"description": "new"})
      assert False, "Should raise"
    except RuntimeError:
      pass


# ---------------------------------------------------------------------------
# Paginated list operations
# ---------------------------------------------------------------------------

def test_list_bot_chats_pagination():
  page1 = {"code": 0, "data": {
    "items": [{"chat_id": "oc_1"}],
    "has_more": True,
    "page_token": "p2",
  }}
  page2 = {"code": 0, "data": {
    "items": [{"chat_id": "oc_2"}],
    "has_more": False,
  }}
  with mock.patch("urllib.request.urlopen",
                  side_effect=[_mock_response(page1), _mock_response(page2)]):
    chats = api.list_bot_chats("tok")
  assert len(chats) == 2


def test_get_chat_members_pagination():
  page1 = {"code": 0, "data": {
    "items": [{"member_id": "ou_1"}],
    "has_more": True,
    "page_token": "p2",
  }}
  page2 = {"code": 0, "data": {
    "items": [{"member_id": "ou_2"}],
    "has_more": False,
  }}
  with mock.patch("urllib.request.urlopen",
                  side_effect=[_mock_response(page1), _mock_response(page2)]):
    members = api.get_chat_members("tok", "oc_1")
  assert len(members) == 2


# ---------------------------------------------------------------------------
# _request edge cases
# ---------------------------------------------------------------------------

def test_request_http_error_with_json_body():
  """HTTPError with JSON body should return the parsed JSON."""
  import urllib.error
  err = urllib.error.HTTPError(
    "https://example.com", 400, "Bad Request", {}, None
  )
  err.read = mock.MagicMock(return_value=b'{"code": 400, "msg": "bad"}')
  with mock.patch("urllib.request.urlopen", side_effect=err):
    result = api._request("https://example.com", "tok")
  assert result["code"] == 400


def test_request_http_error_no_json_body():
  """HTTPError with non-JSON body should re-raise."""
  import urllib.error
  err = urllib.error.HTTPError(
    "https://example.com", 500, "Server Error", {}, None
  )
  err.read = mock.MagicMock(return_value=b"not json")
  with mock.patch("urllib.request.urlopen", side_effect=err):
    try:
      api._request("https://example.com", "tok")
      assert False, "Should raise"
    except urllib.error.HTTPError:
      pass


def test_request_auto_post_with_payload():
  """GET with payload should auto-switch to POST."""
  data = {"code": 0}
  with mock.patch("urllib.request.urlopen", return_value=_mock_response(data)) as m:
    api._request("https://example.com", "tok", {"key": "val"})
  req = m.call_args[0][0]
  assert req.get_method() == "POST"


# ---------------------------------------------------------------------------
# download_image
# ---------------------------------------------------------------------------

def test_download_image(tmp_path):
  fake_resp = mock.MagicMock()
  fake_resp.read.return_value = b"\x89PNGdata"
  fake_resp.__enter__ = mock.MagicMock(return_value=fake_resp)
  fake_resp.__exit__ = mock.MagicMock(return_value=False)

  with mock.patch("urllib.request.urlopen", return_value=fake_resp), \
       mock.patch("nemo.config.tmp_dir", return_value=str(tmp_path)):
    path = api.download_image("tok", "om_1", "img_key_123")
  assert path.endswith("img_key_123.png")
  import os
  assert os.path.isfile(path)
