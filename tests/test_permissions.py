"""Tests for nemo.permissions — auto-approve and tool formatting."""

from nemo.permissions import is_auto_approve, format_tool


def test_auto_approve_handoff_ops():
  assert is_auto_approve("Bash", {"command": "python3 handoff_ops.py download"})
  assert is_auto_approve("Bash", {"command": "python3 send_to_group.py hello"})


def test_auto_approve_non_bash():
  assert not is_auto_approve("Read", {"file_path": "/tmp/x"})
  assert not is_auto_approve("Edit", {"file_path": "/tmp/x"})


def test_auto_approve_random_bash():
  assert not is_auto_approve("Bash", {"command": "rm -rf /"})
  assert not is_auto_approve("Bash", {"command": "ls -la"})


def test_format_tool_bash():
  result = format_tool("Bash", {"command": "ls -la", "description": "List files"})
  assert "Bash" in result
  assert "List files" in result


def test_format_tool_bash_long():
  long_cmd = "x" * 300
  result = format_tool("Bash", {"command": long_cmd})
  assert len(result) < 250  # truncated


def test_format_tool_edit():
  result = format_tool("Edit", {"file_path": "/a/b/main.py"})
  assert "Edit" in result
  assert "main.py" in result


def test_format_tool_read():
  result = format_tool("Read", {"file_path": "/x/y/config.json"})
  assert "Read" in result
  assert "config.json" in result


def test_format_tool_unknown():
  result = format_tool("CustomTool", {})
  assert "CustomTool" in result


def test_auto_approve_non_bash_tools():
  """is_auto_approve returns False for non-Bash tools regardless of input."""
  assert not is_auto_approve("Write", {"file_path": "/tmp/x"})
  assert not is_auto_approve("Agent", {"description": "handoff_ops.py"})
  assert not is_auto_approve("Glob", {"pattern": "*.py"})
  assert not is_auto_approve("Grep", {"pattern": "handoff_ops.py"})


def test_format_tool_bash_long_description():
  """Bash with description >200 chars should be truncated."""
  long_desc = "A" * 250
  result = format_tool("Bash", {"command": "ls", "description": long_desc})
  # The label uses description when present, and gets truncated to 200 chars
  assert len(result) < 250
  assert "..." in result
