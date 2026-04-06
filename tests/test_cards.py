"""Tests for nemo.cards — card builders and tool summary."""

from nemo.cards import (
  ToolRecord, build_turn_card, build_card, build_markdown_card,
  build_form_select, build_form_input,
  tool_use_summary, _elapsed_title, _elapsed_text, _usage_text,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def test_elapsed_title_escalation():
  assert _elapsed_title(0) == "Working..."
  assert _elapsed_title(19) == "Working..."
  assert _elapsed_title(20) == "Working hard..."
  assert _elapsed_title(90) == "Working incredibly hard..."
  assert _elapsed_title(120) == "Working unreasonably hard..."


def test_elapsed_text():
  assert _elapsed_text(30) == "30s"
  assert _elapsed_text(90) == "1m 30s"


def test_usage_text():
  assert _usage_text({}) == ""
  assert _usage_text({"input_tokens": 1000}) == "in: 1,000"
  assert _usage_text({"input_tokens": 1000, "output_tokens": 200}) == "in: 1,000 | out: 200"


# ---------------------------------------------------------------------------
# tool_use_summary
# ---------------------------------------------------------------------------

def test_tool_summary_bash():
  assert tool_use_summary("Bash", {"command": "ls -la"}) == "$ ls -la"
  assert tool_use_summary("Bash", {"description": "List files"}) == "$ List files"


def test_tool_summary_edit():
  assert tool_use_summary("Edit", {"file_path": "/a/b/main.py"}) == "Edit: main.py"
  assert tool_use_summary("Write", {"file_path": "/a/b/out.txt"}) == "Write: out.txt"


def test_tool_summary_read():
  assert tool_use_summary("Read", {"file_path": "/x/config.py"}) == "Read: config.py"


def test_tool_summary_grep():
  assert tool_use_summary("Grep", {"pattern": "TODO"}) == "Grep: TODO"
  assert tool_use_summary("Glob", {"pattern": "**/*.py"}) == "Glob: **/*.py"


def test_tool_summary_agent():
  assert tool_use_summary("Agent", {"description": "search code"}) == "Agent: search code"
  assert tool_use_summary("Agent", {}) == "Agent"


def test_tool_summary_unknown():
  assert tool_use_summary("CustomTool", {}) == "CustomTool"


# ---------------------------------------------------------------------------
# build_turn_card — working phase
# ---------------------------------------------------------------------------

def test_working_card_basic():
  card = build_turn_card("working", current_tool="Read: file.py", elapsed=5)
  assert card["schema"] == "2.0"
  assert card["header"]["template"] == "grey"
  assert "Working..." in card["header"]["title"]["content"]
  elements = card["body"]["elements"]
  # Should have: markdown (current tool) + stop button
  assert any(e["tag"] == "markdown" for e in elements)
  assert any(e["tag"] == "column_set" for e in elements)  # stop button


def test_working_card_with_tools():
  tools = [ToolRecord("Read", "Read: a.py"), ToolRecord("Edit", "Edit: b.py")]
  card = build_turn_card("working", current_tool="Edit: b.py", tools=tools)
  elements = card["body"]["elements"]
  # Should have collapsible panel for previous tools
  assert any(e.get("tag") == "collapsible_panel" for e in elements)


def test_working_card_escalating_title():
  card = build_turn_card("working", elapsed=100)
  assert "incredibly" in card["header"]["title"]["content"].lower()


def test_working_card_stop_button_has_chat_id():
  card = build_turn_card("working", chat_id="oc_123")
  elements = card["body"]["elements"]
  stop_btn = [e for e in elements if e.get("tag") == "column_set"][0]
  btn = stop_btn["columns"][0]["elements"][0]
  assert btn["value"]["action"] == "stop"
  assert btn["value"]["chat_id"] == "oc_123"


# ---------------------------------------------------------------------------
# build_turn_card — working phase with body (intermediate text)
# ---------------------------------------------------------------------------

def test_working_card_with_body():
  card = build_turn_card("working", body="Thinking about this...", elapsed=5)
  elements = card["body"]["elements"]
  # First element should be the body markdown
  assert elements[0]["tag"] == "markdown"
  assert elements[0]["content"] == "Thinking about this..."


def test_working_card_with_body_and_tool():
  card = build_turn_card(
    "working", body="Let me check.", current_tool="Read: foo.py", elapsed=5,
  )
  elements = card["body"]["elements"]
  # body markdown, then tool markdown, then stop button
  assert elements[0]["content"] == "Let me check."
  assert "`Read: foo.py`" in elements[1]["content"]


# ---------------------------------------------------------------------------
# build_turn_card — done phase
# ---------------------------------------------------------------------------

def test_done_card_basic():
  card = build_turn_card("done", body="All done.", elapsed=15)
  assert card["header"]["template"] == "green"
  assert card["header"]["title"]["content"] == "Done ✓"
  elements = card["body"]["elements"]
  assert any(e["tag"] == "markdown" for e in elements)
  # Footer note is now a markdown element with text_size="notation"
  footer = [e for e in elements if e.get("text_size") == "notation"][0]
  assert "15s" in footer["content"]


def test_done_card_with_usage():
  card = build_turn_card(
    "done", body="Result.", elapsed=90,
    usage={"input_tokens": 5000, "output_tokens": 300},
  )
  footer = [e for e in card["body"]["elements"] if e.get("text_size") == "notation"][0]
  text = footer["content"]
  assert "1m 30s" in text
  assert "5,000" in text
  assert "300" in text


def test_done_card_with_tools():
  tools = [ToolRecord("Read", "Read: x.py")]
  card = build_turn_card("done", body="Done.", tools=tools, elapsed=5)
  elements = card["body"]["elements"]
  assert any(e.get("tag") == "collapsible_panel" for e in elements)


# ---------------------------------------------------------------------------
# build_turn_card — invalid phase
# ---------------------------------------------------------------------------

def test_invalid_phase_raises():
  try:
    build_turn_card("invalid")
    assert False, "Should raise ValueError"
  except ValueError:
    pass


def test_response_phase_removed():
  """Response phase no longer exists — should raise."""
  try:
    build_turn_card("response")
    assert False, "Should raise ValueError"
  except ValueError:
    pass


# ---------------------------------------------------------------------------
# build_card (V2 simple card)
# ---------------------------------------------------------------------------

def test_build_card_basic():
  card = build_card("Hello", body="World", color="blue")
  assert card["schema"] == "2.0"
  assert card["header"]["template"] == "blue"
  assert card["header"]["title"]["content"] == "Hello"
  elements = card["body"]["elements"]
  assert elements[0]["tag"] == "markdown"
  assert elements[0]["content"] == "World"


def test_build_card_with_buttons():
  card = build_card(
    "Approve?",
    body="Run this tool?",
    buttons=[("Yes", "approve", "primary"), ("No", "deny", "danger")],
    chat_id="oc_1",
  )
  elements = card["body"]["elements"]
  btn_row = [e for e in elements if e.get("tag") == "column_set"][0]
  assert len(btn_row["columns"]) == 2
  assert btn_row["columns"][0]["elements"][0]["value"]["action"] == "approve"
  assert btn_row["columns"][1]["elements"][0]["value"]["action"] == "deny"


def test_build_card_with_note():
  card = build_card("Status", note="Session: abc123")
  elements = card["body"]["elements"]
  footer = [e for e in elements if e.get("text_size") == "notation"][0]
  assert "abc123" in footer["content"]


def test_build_card_empty_body():
  card = build_card("Title")
  elements = card["body"]["elements"]
  assert not any(e.get("tag") == "markdown" for e in elements)


# ---------------------------------------------------------------------------
# build_markdown_card
# ---------------------------------------------------------------------------

def test_markdown_card_basic():
  card = build_markdown_card("# Hello\nWorld")
  assert card["schema"] == "2.0"
  assert "header" not in card
  assert card["body"]["elements"][0]["content"] == "# Hello\nWorld"


def test_markdown_card_with_title():
  card = build_markdown_card("Content", title="Info", color="purple")
  assert card["header"]["title"]["content"] == "Info"
  assert card["header"]["template"] == "purple"


# ---------------------------------------------------------------------------
# build_form_select
# ---------------------------------------------------------------------------

def test_form_select_basic():
  options = [
    {"text": "Option A", "value": "a"},
    {"text": "Option B", "value": "b"},
  ]
  card = build_form_select("Pick one", options)
  assert card["schema"] == "2.0"
  assert card["header"]["title"]["content"] == "Pick one"
  elements = card["body"]["elements"]
  assert elements[0]["tag"] == "select_static"
  assert len(elements[0]["options"]) == 2
  assert elements[0]["options"][0]["value"] == "a"
  assert elements[0]["options"][1]["text"]["content"] == "Option B"


def test_form_select_with_chat_id():
  card = build_form_select("Title", [{"text": "X", "value": "x"}], chat_id="oc_99")
  select = card["body"]["elements"][0]
  assert select["value"]["chat_id"] == "oc_99"


def test_form_select_empty_options():
  card = build_form_select("Empty", [])
  assert card["body"]["elements"][0]["options"] == []


# ---------------------------------------------------------------------------
# build_form_input
# ---------------------------------------------------------------------------

def test_form_input_basic():
  card = build_form_input("Enter name")
  assert card["schema"] == "2.0"
  assert card["header"]["title"]["content"] == "Enter name"
  elements = card["body"]["elements"]
  assert elements[0]["tag"] == "input"
  assert elements[0]["name"] == "user_input"


def test_form_input_placeholder():
  card = build_form_input("Question", placeholder="Type your answer")
  inp = card["body"]["elements"][0]
  assert inp["placeholder"]["content"] == "Type your answer"


def test_form_input_default_placeholder():
  card = build_form_input("Question")
  inp = card["body"]["elements"][0]
  assert inp["placeholder"]["content"] == "Type here..."


def test_form_input_with_chat_id():
  card = build_form_input("Q", chat_id="oc_42")
  inp = card["body"]["elements"][0]
  assert inp["value"]["chat_id"] == "oc_42"
