"""Tests for nemo.db — cross-thread safety with check_same_thread=False."""

import threading
from unittest import mock

from nemo.db import Database


def test_concurrent_reads(tmp_path):
  """Multiple threads can read simultaneously without check_same_thread error."""
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    db.activate("sess1", "chat1", "opus")

    results = []
    errors = []

    def reader():
      try:
        s = db.get_session("sess1")
        results.append(s)
      except Exception as e:
        errors.append(e)

    # Sequential reads from different threads — verifies check_same_thread=False
    for _ in range(5):
      t = threading.Thread(target=reader)
      t.start()
      t.join()

    assert not errors
    assert len(results) == 5
    assert all(r["chat_id"] == "chat1" for r in results)
    db.close()


def test_write_from_different_thread(tmp_path):
  """Writing from a non-main thread works with check_same_thread=False."""
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    db.activate("sess1", "chat1", "opus")

    errors = []

    def writer():
      try:
        db.record_sent("msg_1", text="hello", chat_id="chat1")
      except Exception as e:
        errors.append(e)

    t = threading.Thread(target=writer)
    t.start()
    t.join()

    assert not errors
    # Verify the write was persisted
    parent = db.lookup_parent_message("msg_1")
    assert parent is not None
    assert parent["text"] == "hello"
    db.close()


def test_cross_thread_working_state(tmp_path):
  """Working state can be set from one thread and read from another."""
  with mock.patch("nemo.db.DB_BASE", str(tmp_path)):
    db = Database(str(tmp_path / "project"))
    db.activate("sess1", "chat1", "opus")

    result = []

    def writer():
      db.set_working("sess1", "msg_abc")

    def reader():
      # Read until we see the value (writer should have finished)
      r = db.get_working("sess1")
      result.append(r)

    t1 = threading.Thread(target=writer)
    t1.start()
    t1.join()

    t2 = threading.Thread(target=reader)
    t2.start()
    t2.join()

    assert result[0] == "msg_abc"
    db.close()
