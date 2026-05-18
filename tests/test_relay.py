"""Tests for nemo.relay — relay client (heartbeat, register_message)."""

from unittest.mock import patch, MagicMock
import json

from nemo.relay import (
    send_heartbeat, is_alive, release_heartbeat, register_message,
    heartbeat_status, _relay_request,
)


# ---------------------------------------------------------------------------
# _relay_request — basic behavior
# ---------------------------------------------------------------------------

def test_relay_request_no_config():
    """Should raise if relay_url is not configured."""
    with patch("nemo.relay.load_relay_config", return_value=("", "")):
        try:
            _relay_request("GET", "/test")
            assert False, "Expected RuntimeError"
        except RuntimeError as e:
            assert "not configured" in str(e)


def test_relay_request_builds_url():
    """Should build correct URL and headers."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"ok": true}'
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("nemo.relay.load_relay_config",
               return_value=("http://relay.test", "mykey")):
        with patch("nemo.relay.urllib.request.urlopen",
                   return_value=mock_resp) as mock_open:
            result = _relay_request("GET", "/heartbeat/chat:test")
            assert result == {"ok": True}
            req = mock_open.call_args[0][0]
            assert req.full_url == "http://relay.test/heartbeat/chat:test"
            assert req.get_header("Authorization") == "Bearer mykey"


# ---------------------------------------------------------------------------
# send_heartbeat
# ---------------------------------------------------------------------------

def test_send_heartbeat_calls_post():
    with patch("nemo.relay._relay_request") as mock:
        send_heartbeat("oc_123", pid=456, model="opus", machine="mac1")
        mock.assert_called_once_with("POST", "/heartbeat/chat:oc_123", {
            "pid": 456, "model": "opus", "machine": "mac1",
        })


def test_send_heartbeat_swallows_error():
    """Should not raise on failure."""
    with patch("nemo.relay._relay_request", side_effect=Exception("fail")):
        send_heartbeat("oc_123")  # Should not raise


# ---------------------------------------------------------------------------
# is_alive
# ---------------------------------------------------------------------------

def test_is_alive_true():
    with patch("nemo.relay._relay_request",
               return_value={"alive": True, "pid": 123}):
        assert is_alive("oc_123") is True


def test_is_alive_false():
    with patch("nemo.relay._relay_request",
               return_value={"alive": False}):
        assert is_alive("oc_123") is False


def test_is_alive_error_returns_false():
    """Network error should return False (treat as idle)."""
    with patch("nemo.relay._relay_request", side_effect=Exception("timeout")):
        assert is_alive("oc_123") is False


def test_heartbeat_status_returns_error_instead_of_idle_on_failure():
    with patch("nemo.relay._relay_request", side_effect=Exception("timeout")):
        result = heartbeat_status("oc_123")

    assert result["alive"] is False
    assert "timeout" in str(result["error"])


def test_heartbeat_status_returns_raw_response():
    with patch("nemo.relay._relay_request",
               return_value={"alive": True, "machine": "Mac"}):
        result = heartbeat_status("oc_123")

    assert result == {"alive": True, "machine": "Mac"}


# ---------------------------------------------------------------------------
# release_heartbeat
# ---------------------------------------------------------------------------

def test_release_heartbeat_calls_delete():
    with patch("nemo.relay._relay_request") as mock:
        release_heartbeat("oc_123")
        mock.assert_called_once_with("DELETE", "/heartbeat/chat:oc_123")


def test_release_heartbeat_swallows_error():
    with patch("nemo.relay._relay_request", side_effect=Exception("fail")):
        release_heartbeat("oc_123")  # Should not raise


# ---------------------------------------------------------------------------
# register_message
# ---------------------------------------------------------------------------

def test_register_message_calls_post():
    with patch("nemo.relay._relay_request") as mock:
        register_message("om_msg1", "oc_chat1")
        mock.assert_called_once_with("POST", "/register-message", {
            "message_id": "om_msg1", "chat_id": "oc_chat1",
        })


def test_register_message_swallows_error():
    with patch("nemo.relay._relay_request", side_effect=Exception("fail")):
        register_message("om_msg1", "oc_chat1")  # Should not raise


# ---------------------------------------------------------------------------
# send_stop
# ---------------------------------------------------------------------------

from nemo.relay import send_stop


def test_send_stop_calls_post():
    with patch("nemo.relay._relay_request") as mock_req:
        send_stop("oc_123")
        mock_req.assert_called_once_with("POST", "/stop/chat:oc_123")


def test_send_stop_swallows_error():
    """Should not raise on failure."""
    with patch("nemo.relay._relay_request", side_effect=Exception("fail")):
        send_stop("oc_123")  # Should not raise


# ---------------------------------------------------------------------------
# _relay_request — edge cases
# ---------------------------------------------------------------------------

def test_relay_request_no_api_key():
    """Should still work without API key (no Authorization header)."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"ok": true}'
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("nemo.relay.load_relay_config",
               return_value=("http://relay.test", "")):
        with patch("nemo.relay.urllib.request.urlopen",
                   return_value=mock_resp) as mock_open:
            result = _relay_request("GET", "/test")
            req = mock_open.call_args[0][0]
            assert req.get_header("Authorization") is None
            assert result == {"ok": True}


def test_relay_request_with_data():
    """Should encode data as JSON body."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"ok": true}'
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("nemo.relay.load_relay_config",
               return_value=("http://relay.test", "key")):
        with patch("nemo.relay.urllib.request.urlopen",
                   return_value=mock_resp) as mock_open:
            _relay_request("POST", "/test", {"foo": "bar"})
            req = mock_open.call_args[0][0]
            assert req.data == b'{"foo": "bar"}'


def test_relay_request_strips_trailing_slash():
    """Trailing slash in relay_url should be stripped."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"ok": true}'
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("nemo.relay.load_relay_config",
               return_value=("http://relay.test/", "key")):
        with patch("nemo.relay.urllib.request.urlopen",
                   return_value=mock_resp) as mock_open:
            _relay_request("GET", "/path")
            req = mock_open.call_args[0][0]
            assert req.full_url == "http://relay.test/path"
