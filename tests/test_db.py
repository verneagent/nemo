"""Tests for nemo.db."""

from unittest import mock

from nemo.db import Database


def test_activate_and_get_session(tmp_path):
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    db.activate("sess1", "chat1", "opus", operator_open_id="op1", bot_open_id="bot1")
    s = db.get_session("sess1")
    assert s is not None
    assert s["chat_id"] == "chat1"
    assert s["session_model"] == "opus"
    assert s["operator_open_id"] == "op1"
    assert s["bot_open_id"] == "bot1"
    assert s["need_mention"] is False
    assert s["autoapprove"] is False
    db.close()


def test_get_session_nonexistent(tmp_path):
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    assert db.get_session("nope") is None
    db.close()


def test_deactivate(tmp_path):
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    db.activate("sess1", "chat1", "opus")
    chat_id = db.deactivate("sess1")
    assert chat_id == "chat1"
    assert db.get_session("sess1") is None
    db.close()


def test_deactivate_nonexistent(tmp_path):
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    assert db.deactivate("nope") is None
    db.close()


def test_get_chat_owner(tmp_path):
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    db.activate("sess1", "chat1", "opus")
    assert db.get_chat_owner("chat1") == "sess1"
    assert db.get_chat_owner("chat999") is None
    db.close()


def test_set_autoapprove(tmp_path):
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    db.activate("sess1", "chat1", "opus")
    db.set_autoapprove("chat1", True)
    s = db.get_session("sess1")
    assert s["autoapprove"] is True
    db.set_autoapprove("chat1", False)
    s = db.get_session("sess1")
    assert s["autoapprove"] is False
    db.close()


def test_record_and_lookup_messages(tmp_path):
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    db.record_received("chat1", text="hello", source_message_id="om_1")
    db.record_sent("om_2", text="hi back", chat_id="chat1")
    parent = db.lookup_parent_message("om_2")
    assert parent is not None
    assert parent["text"] == "hi back"
    # Lookup also finds received messages by source_message_id —
    # needed so quoted user messages can be recovered from our own DB.
    received = db.lookup_parent_message("om_1")
    assert received is not None
    assert received["text"] == "hello"
    assert db.lookup_parent_message("om_999") is None
    db.close()


def test_working_state(tmp_path):
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    db.activate("sess1", "chat1", "opus")
    assert db.get_working("sess1") is None
    db.set_working("sess1", "msg_123")
    assert db.get_working("sess1") == "msg_123"
    db.clear_working("sess1")
    assert db.get_working("sess1") is None
    db.close()


def test_deactivate_clears_working(tmp_path):
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    db.activate("sess1", "chat1", "opus")
    db.set_working("sess1", "msg_123")
    db.deactivate("sess1")
    assert db.get_working("sess1") is None
    db.close()


def test_need_mention(tmp_path):
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    db.activate("sess1", "chat1", "opus", need_mention=True)
    s = db.get_session("sess1")
    assert s["need_mention"] is True
    db.close()


def test_get_current_session(tmp_path):
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    assert db.get_current_session() is None
    db.activate("sess1", "chat1", "opus")
    s = db.get_current_session()
    assert s is not None
    assert s["session_id"] == "sess1"
    db.close()
