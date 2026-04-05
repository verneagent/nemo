"""Tests for nemo.lark.auth."""

import json
import time
from unittest import mock

from nemo.lark.auth import LarkAuth, get_token


def _mock_urlopen(token="t-test123", expire=7200, code=0):
  """Create a mock urlopen context manager."""
  resp_data = json.dumps({
    "code": code,
    "tenant_access_token": token,
    "expire": expire,
  }).encode()
  mock_resp = mock.MagicMock()
  mock_resp.read.return_value = resp_data
  mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
  mock_resp.__exit__ = mock.MagicMock(return_value=False)
  return mock_resp


def test_get_token_fetches_and_caches():
  auth = LarkAuth()
  mock_resp = _mock_urlopen("t-abc")
  with mock.patch("urllib.request.urlopen", return_value=mock_resp) as m:
    tok1 = auth.get_token("app1", "secret1")
    tok2 = auth.get_token("app1", "secret1")
  assert tok1 == "t-abc"
  assert tok2 == "t-abc"
  # Only one HTTP call (cached)
  assert m.call_count == 1


def test_get_token_refreshes_when_expired():
  auth = LarkAuth()
  auth._token = "old"
  auth._expires_at = time.time() - 100  # expired
  mock_resp = _mock_urlopen("t-new")
  with mock.patch("urllib.request.urlopen", return_value=mock_resp):
    tok = auth.get_token("app1", "secret1")
  assert tok == "t-new"


def test_get_token_error_raises():
  auth = LarkAuth()
  mock_resp = _mock_urlopen(code=99)
  with mock.patch("urllib.request.urlopen", return_value=mock_resp):
    try:
      auth.get_token("app1", "secret1")
      assert False, "Should have raised"
    except RuntimeError as e:
      assert "Token error" in str(e)


def test_module_singleton():
  mock_resp = _mock_urlopen("t-single")
  with mock.patch("urllib.request.urlopen", return_value=mock_resp):
    with mock.patch("nemo.lark.auth._auth", LarkAuth()):
      tok = get_token("app1", "secret1")
  assert tok == "t-single"


def test_thread_safe_lock():
  """LarkAuth should have a threading.Lock for thread-safe token refresh."""
  import threading
  auth = LarkAuth()
  assert hasattr(auth, "_lock")
  assert isinstance(auth._lock, type(threading.Lock()))
