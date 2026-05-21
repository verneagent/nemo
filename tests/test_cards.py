"""Tests for nemo.cards — card builders and tool summary."""

from nemo.cards import (
  ToolRecord, ThinkingStep, build_turn_card, build_card, build_markdown_card,
  build_form_select, build_form_input, build_ask_user_question_card,
  build_model_picker_card, build_shell_card, tool_use_summary, _elapsed_title,
  _elapsed_text, _usage_text, _collapsible_thinking,
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


def test_shell_card_running_has_abort_button():
  card = build_shell_card(
    "running",
    job_id="job123",
    command="echo hi",
    cwd="/tmp/project",
    elapsed=1,
    inject_context=True,
    chat_id="oc_test",
  )
  elements = card["body"]["elements"]
  button_row = elements[-1]
  button = button_row["columns"][0]["elements"][0]
  assert button["text"]["content"] == "Abort"
  assert button["value"]["action"] == "shell_abort"
  assert button["value"]["job_id"] == "job123"


def test_shell_card_done_removes_abort_button():
  card = build_shell_card(
    "done",
    job_id="job123",
    command="echo hi",
    cwd="/tmp/project",
    elapsed=1,
    inject_context=True,
    chat_id="oc_test",
    exit_code=0,
    stdout="hi\n",
  )
  assert "Shell done" == card["header"]["title"]["content"]
  assert all(el.get("tag") != "column_set" for el in card["body"]["elements"])


# ---------------------------------------------------------------------------
# tool_use_summary
# ---------------------------------------------------------------------------

def test_tool_summary_bash():
  assert tool_use_summary("Bash", {"command": "ls -la"}) == "Bash: ls -la"
  assert tool_use_summary("Bash", {"description": "List files"}) == "Bash: List files"


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
# ThinkingStep & _collapsible_thinking
# ---------------------------------------------------------------------------


def _panel_content(panel):
  """Join all markdown element contents; use '---' for hr elements."""
  parts = []
  for el in panel["elements"]:
    if el["tag"] == "hr":
      parts.append("---")
    else:
      parts.append(el.get("content", ""))
  return "\n".join(parts)


def test_collapsible_thinking_mixed():
  steps = [
    ThinkingStep("answer", "Let me check..."),
    ThinkingStep("tool", "Read: main.py"),
    ThinkingStep("answer", "Found the issue"),
    ThinkingStep("tool", "Edit: main.py"),
  ]
  panel = _collapsible_thinking(steps)
  assert panel["tag"] == "collapsible_panel"
  # Header shows group count (2 texts = 2 groups), not total step count
  assert panel["header"]["title"]["content"] == "Thinking (2)"
  content = _panel_content(panel)
  assert "Let me check..." in content
  assert "Read:" in content
  assert "main.py" in content
  assert "Found the issue" in content
  assert "Edit:" in content


def test_collapsible_thinking_groups_consecutive_tools():
  """When tools exceed limit, only last 5 are kept (still coalesced)."""
  # 7 grep tools, limit is 5, so 2 dropped + 5 kept (all same type → 1 line)
  steps = [ThinkingStep("tool", f"Grep: pattern{i}") for i in range(7)]
  panel = _collapsible_thinking(steps)
  content = _panel_content(panel)
  assert content.count("Grep:") == 1
  # First 2 patterns dropped (pattern0, pattern1)
  assert "pattern0" not in content
  assert "pattern1" not in content
  assert "pattern6" in content
  # Indicator for dropped tools
  assert "+2 earlier" in content


def test_collapsible_thinking_separates_different_tool_types():
  """Different tool types should not be grouped together."""
  steps = [
    ThinkingStep("tool", "Read: a.py"),
    ThinkingStep("tool", "Read: b.py"),
    ThinkingStep("tool", "Grep: foo"),
    ThinkingStep("tool", "Read: c.py"),
  ]
  panel = _collapsible_thinking(steps)
  content = _panel_content(panel)
  assert content.count("Read:") == 2  # two separate Read groups
  assert content.count("Grep:") == 1


def test_collapsible_thinking_limits_tools_per_group_to_5():
  """Each group shows at most 5 tool calls; older ones get 'N earlier'."""
  steps = [ThinkingStep("answer", "Doing stuff")]
  # 8 different tool types so coalescing doesn't merge them
  steps.extend([
    ThinkingStep("tool", "Read: a.py"),
    ThinkingStep("tool", "Grep: foo"),
    ThinkingStep("tool", "Bash: ls"),
    ThinkingStep("tool", "Edit: b.py"),
    ThinkingStep("tool", "Read: c.py"),
    ThinkingStep("tool", "Grep: bar"),
    ThinkingStep("tool", "Bash: pwd"),
    ThinkingStep("tool", "Edit: d.py"),
  ])
  panel = _collapsible_thinking(steps)
  content = _panel_content(panel)
  # Indicator for 3 dropped (8 - 5 = 3)
  assert "+3 earlier" in content
  # First 3 tools dropped
  assert "a.py" not in content
  assert "foo" not in content
  assert " ls" not in content  # leading space to avoid "tool calls"
  # Last 5 kept
  assert "b.py" in content
  assert "c.py" in content
  assert "bar" in content
  assert "pwd" in content
  assert "d.py" in content


def test_collapsible_thinking_group_count_in_header():
  """Header shows number of groups (text-initiated)."""
  steps = [
    ThinkingStep("answer", "Step 1"),
    ThinkingStep("tool", "Read: a.py"),
    ThinkingStep("answer", "Step 2"),
    ThinkingStep("tool", "Read: b.py"),
    ThinkingStep("answer", "Step 3"),
  ]
  panel = _collapsible_thinking(steps)
  assert panel["header"]["title"]["content"] == "Thinking (3)"


def test_collapsible_thinking_leading_tool_group():
  """If no leading text, the first tool-only group still counts as 1."""
  steps = [
    ThinkingStep("tool", "Read: a.py"),
    ThinkingStep("answer", "After tool"),
    ThinkingStep("tool", "Edit: a.py"),
  ]
  panel = _collapsible_thinking(steps)
  assert panel["header"]["title"]["content"] == "Thinking (2)"


def test_collapsible_thinking_thinking_not_counted_toward_tool_limit():
  """Thinking blocks don't count against the 5-tool limit."""
  steps = [ThinkingStep("answer", "Working")]
  steps.extend([ThinkingStep("thinking", f"thought {i}") for i in range(3)])
  steps.extend([ThinkingStep("tool", f"Bash: cmd{i}") for i in range(6)])
  panel = _collapsible_thinking(steps)
  content = _panel_content(panel)
  # 6 tools - limit 5 = 1 dropped
  assert "+1 earlier" in content
  # All 3 thinking blocks present
  assert "thought 0" in content
  assert "thought 2" in content


def test_collapsible_thinking_text_separates_groups_with_divider():
  """Each narrative text starts a new group with a --- divider."""
  steps = [
    ThinkingStep("answer", "Let me check"),
    ThinkingStep("tool", "Read: a.py"),
    ThinkingStep("answer", "Now fixing"),
    ThinkingStep("tool", "Edit: a.py"),
  ]
  panel = _collapsible_thinking(steps)
  content = _panel_content(panel)
  assert "---" in content
  # Divider appears between groups, not at the start
  assert not content.startswith("---")
  # Exactly one divider (between the two text blocks)
  assert content.count("---") == 1


def test_collapsible_thinking_no_divider_for_single_group():
  """A single group (no second text) should have no divider."""
  steps = [
    ThinkingStep("answer", "Let me check"),
    ThinkingStep("tool", "Read: a.py"),
  ]
  panel = _collapsible_thinking(steps)
  content = _panel_content(panel)
  assert "---" not in content


def test_collapsible_thinking_escapes_angle_brackets():
  """Grep patterns with <<<< should not render as &lt;&lt;&lt;."""
  steps = [ThinkingStep("tool", "Grep: <<<<<<<")]
  panel = _collapsible_thinking(steps)
  content = _panel_content(panel)
  assert "<<" not in content  # raw pattern stripped
  assert "&lt;" not in content
  assert "‹‹‹‹‹‹‹" in content


def test_collapsible_thinking_text_truncated():
  long_text = "x" * 500
  steps = [ThinkingStep("answer", long_text)]
  panel = _collapsible_thinking(steps)
  content = _panel_content(panel)
  assert len(content) <= 310  # 300 + "..."
  assert content.endswith("...")


def test_collapsible_thinking_empty():
  """Empty steps list — should not normally be rendered but handle gracefully."""
  panel = _collapsible_thinking([])
  assert "Thinking (0)" in panel["header"]["title"]["content"]


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


def test_working_card_with_steps():
  steps = [
    ThinkingStep("answer", "Checking..."),
    ThinkingStep("tool", "Read: a.py"),
    ThinkingStep("tool", "Edit: b.py"),
  ]
  card = build_turn_card("working", current_tool="Edit: b.py", steps=steps)
  elements = card["body"]["elements"]
  # Should have: current_tool markdown, thinking collapsible, stop button
  assert elements[0]["tag"] == "markdown"
  assert "`Edit: b.py`" in elements[0]["content"]
  assert elements[1]["tag"] == "collapsible_panel"
  # 1 text = 1 group
  assert "Thinking (1)" in elements[1]["header"]["title"]["content"]


def test_working_card_escalating_title():
  card = build_turn_card("working", elapsed=100)
  assert "incredibly" in card["header"]["title"]["content"].lower()


def test_working_card_stop_button_has_chat_id():
  card = build_turn_card("working", chat_id="oc_123")
  elements = card["body"]["elements"]
  stop_btn = [e for e in elements if e.get("tag") == "column_set"][0]
  btn = stop_btn["columns"][0]["elements"][0]
  assert btn["value"]["action"] == "__stop__"
  assert btn["value"]["chat_id"] == "oc_123"


def test_working_card_with_body_only():
  """Body param still accepted for backwards compat but not used in working."""
  card = build_turn_card("working", body="ignored", elapsed=5)
  elements = card["body"]["elements"]
  # Only stop button when no steps and no current_tool
  assert elements[0]["tag"] == "column_set"


def test_working_card_renders_rate_limit_notice_above_tool():
  """Rate-limit banner sits above the current_tool block on the working card."""
  card = build_turn_card(
    "working",
    current_tool="Read: a.py",
    rate_limit_notice="⛔ Rate limit hit (five_hour) resets in 12m",
    elapsed=5,
    chat_id="oc_x",
  )
  elements = card["body"]["elements"]
  # First element is the orange rate-limit banner, second is the tool box.
  assert elements[0]["tag"] == "markdown"
  assert "Rate limit hit" in elements[0]["content"]
  assert "<font color='orange'>" in elements[0]["content"]
  assert elements[1]["tag"] == "markdown"
  assert "`Read: a.py`" in elements[1]["content"]


def test_working_card_no_rate_limit_notice_by_default():
  """Without an explicit notice the working card stays unchanged."""
  card = build_turn_card("working", current_tool="Read: a.py", chat_id="oc_x")
  elements = card["body"]["elements"]
  assert elements[0]["tag"] == "markdown"
  # No orange banner — the first element is the tool, not a notice.
  assert "Rate limit" not in elements[0]["content"]


def test_working_card_renders_compact_notice_above_thinking():
  """Compact-notice banner sits at the top of the working card body —
  ABOVE the collapsible thinking panel — so a 10–60s silent compaction
  is explained as it happens instead of buried in the thinking timeline
  the user has to expand to see.

  Regression: pre-refactor compaction surfaced as a ThinkingStep
  ("compact", "🗜 上下文压缩中…") that got grouped into the
  ``collapsible_thinking`` panel. The pause was invisible until the
  user expanded thinking — exactly the silence we were trying to fix.
  """
  steps = [ThinkingStep("tool", "Read: a.py")]
  card = build_turn_card(
    "working",
    steps=steps,
    compact_notice="🗜 上下文压缩中…",
    elapsed=12,
    chat_id="oc_x",
  )
  elements = card["body"]["elements"]
  # First element is the grey compact-notice banner (outside the
  # collapsible thinking panel).
  assert elements[0]["tag"] == "markdown"
  assert "压缩中" in elements[0]["content"]
  assert "<font color='grey'>" in elements[0]["content"]
  # No element after the banner should be a collapsible_panel that
  # contains a "压缩" step — the notice must NOT be duplicated inside
  # the thinking timeline.
  for el in elements:
    if el.get("tag") == "collapsible_panel":
      assert "压缩" not in repr(el), (
        "compact notice leaked into collapsible thinking — banner only"
      )


def test_working_card_renders_both_notices_distinct_colors():
  """Rate-limit (orange) and compact (grey) can coexist as two banners
  at the top of the working card. They share the same slot but stack
  in declaration order so the user sees both."""
  card = build_turn_card(
    "working",
    current_tool="Read: a.py",
    rate_limit_notice="⛔ Rate limit hit",
    compact_notice="🗜 上下文压缩中…",
    elapsed=5,
    chat_id="oc_x",
  )
  elements = card["body"]["elements"]
  assert elements[0]["tag"] == "markdown"
  assert "Rate limit" in elements[0]["content"]
  assert "<font color='orange'>" in elements[0]["content"]
  assert elements[1]["tag"] == "markdown"
  assert "压缩中" in elements[1]["content"]
  assert "<font color='grey'>" in elements[1]["content"]


def test_working_card_no_compact_notice_by_default():
  """Without an explicit notice nothing changes."""
  card = build_turn_card("working", current_tool="Read: a.py", chat_id="oc_x")
  elements = card["body"]["elements"]
  for el in elements:
    if el.get("tag") == "markdown":
      assert "压缩" not in el["content"]


def test_stopping_card_preserves_working_content():
  steps = [ThinkingStep("tool", "Read: a.py"), ThinkingStep("answer", "Checking")]
  card = build_turn_card("stopping", current_tool="Read: a.py", steps=steps)
  assert card["header"]["title"]["content"] == "Stopping..."
  assert card["header"]["template"] == "orange"
  elements = card["body"]["elements"]
  assert elements[0]["tag"] == "markdown"
  assert "`Read: a.py`" in elements[0]["content"]
  assert elements[1]["tag"] == "collapsible_panel"
  assert not any(e["tag"] == "column_set" for e in elements)


def test_stopped_card_preserves_working_content():
  steps = [ThinkingStep("tool", "Edit: b.py")]
  card = build_turn_card("stopped", current_tool="Edit: b.py", steps=steps)
  assert card["header"]["title"]["content"] == "Stopped"
  assert card["header"]["template"] == "grey"
  elements = card["body"]["elements"]
  assert elements[0]["tag"] == "markdown"
  assert "`Edit: b.py`" in elements[0]["content"]
  assert not any(e["tag"] == "column_set" for e in elements)


# ---------------------------------------------------------------------------
# build_turn_card — done phase
# ---------------------------------------------------------------------------

def test_done_card_basic():
  card = build_turn_card("done", body="All done.", elapsed=15)
  assert card["header"]["template"] == "green"
  assert card["header"]["title"]["content"] == "Done ✓"
  elements = card["body"]["elements"]
  assert any(e["tag"] == "markdown" and e["content"] == "All done." for e in elements)
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


def test_done_card_with_steps():
  steps = [ThinkingStep("tool", "Read: x.py"), ThinkingStep("answer", "Found it")]
  card = build_turn_card("done", body="Done.", steps=steps, elapsed=5)
  elements = card["body"]["elements"]
  # Body inline, then thinking collapsible, then note
  assert elements[0]["content"] == "Done."
  assert elements[1]["tag"] == "collapsible_panel"
  assert "Thinking (2)" in elements[1]["header"]["title"]["content"]


def test_done_card_preserves_compact_notice():
  card = build_turn_card(
    "done",
    body="Done.",
    compact_notice="🗜 Context compacted: 12k → 4k tokens",
  )
  elements = card["body"]["elements"]
  assert elements[0]["tag"] == "markdown"
  assert "Context compacted" in elements[0]["content"]
  assert elements[1]["content"] == "Done."


# ---------------------------------------------------------------------------
# build_turn_card — error phase
# ---------------------------------------------------------------------------

def test_error_card_basic():
  card = build_turn_card("error", body="**Timeout**", elapsed=30)
  assert card["header"]["template"] == "red"
  assert card["header"]["title"]["content"] == "Error"
  elements = card["body"]["elements"]
  assert elements[0]["content"] == "**Timeout**"


def test_error_card_with_steps():
  steps = [ThinkingStep("tool", "$ long-running-cmd")]
  card = build_turn_card("error", body="**Timed out**", steps=steps, elapsed=60)
  elements = card["body"]["elements"]
  assert elements[1]["tag"] == "collapsible_panel"


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
# build_model_picker_card
# ---------------------------------------------------------------------------

def _find_element(elements, tag):
  for el in elements:
    if isinstance(el, dict) and el.get("tag") == tag:
      return el
  raise AssertionError(f"No element with tag {tag!r} in {elements!r}")


def test_model_picker_card_structure():
  """Picker wraps the dropdown in a Lark V2 form so selection only
  fires when Submit is clicked. Each option's value carries the
  ``model_switch:<name>`` discriminator the daemon routes on."""
  options = [
    ("claude-opus-4-7", "claude-opus-4-7"),
    ("claude-sonnet-4-6", "claude-sonnet-4-6"),
  ]
  card = build_model_picker_card(
    options,
    current_model="claude-opus-4-7",
    current_agent="claude",
    chat_id="oc_abc",
  )
  assert card["schema"] == "2.0"
  assert card["header"]["title"]["content"] == "Switch Model"

  elements = card["body"]["elements"]
  summary = _find_element(elements, "markdown")
  assert "claude-opus-4-7" in summary["content"]
  assert "claude" in summary["content"]

  form = _find_element(elements, "form")
  assert form["name"] == "model_picker_form"
  form_elements = form["elements"]

  select = _find_element(form_elements, "select_static")
  assert select["name"] == "model"
  values = [opt["value"] for opt in select["options"]]
  assert values == [
    "model_switch:claude-opus-4-7",
    "model_switch:claude-sonnet-4-6",
  ]
  labels = [opt["text"]["content"] for opt in select["options"]]
  assert labels == ["claude-opus-4-7", "claude-sonnet-4-6"]

  # Submit button — must declare form_action_type so Lark treats the
  # click as a form submission (delivering form_value) rather than a
  # bare button_action.
  col_set = _find_element(form_elements, "column_set")
  button = col_set["columns"][0]["elements"][0]
  assert button["tag"] == "button"
  assert button["form_action_type"] == "submit"
  assert button["type"] == "primary"
  assert button["value"]["chat_id"] == "oc_abc"
  # The submit button must NOT carry a ``name`` field — Lark includes
  # every named form child in form_value at submit time, and a named
  # submit would turn the single-field form_value into a multi-field
  # one which the relay JSON-encodes, breaking the daemon's
  # ``startswith("model_switch:")`` routing.
  assert "name" not in button, button


def test_model_picker_card_with_note():
  card = build_model_picker_card(
    [("opus", "claude-opus-4-7")],
    current_model="claude-opus-4-7",
    current_agent="claude",
    chat_id="",
    note="Available: opus, sonnet, haiku",
  )
  # The catalog/help note goes below the form so the dropdown is still
  # the focal element — but the note must still be present.
  last = card["body"]["elements"][-1]
  assert "Available" in str(last)


def test_model_picker_card_empty_options():
  card = build_model_picker_card(
    [],
    current_model="claude-opus-4-7",
    current_agent="claude",
    chat_id="",
  )
  form = _find_element(card["body"]["elements"], "form")
  select = _find_element(form["elements"], "select_static")
  assert select["options"] == []


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


# ---------------------------------------------------------------------------
# build_ask_user_question_card
# ---------------------------------------------------------------------------

def _question(text, header, options, multi_select=False):
  return {
    "question": text,
    "header": header,
    "options": [{"label": label} for label in options],
    "multiSelect": multi_select,
  }


def test_askq_card_single_question_basic():
  card = build_ask_user_question_card(
    questions=[_question("Where?", "Screen", ["Login", "Match"])],
    chat_id="oc_42",
    nonce="abc123",
  )
  assert card["schema"] == "2.0"
  assert card["header"]["title"]["content"] == "Question from Claude"
  # header markdown + question markdown + button row(s) + Other row + footer note
  elements = card["body"]["elements"]
  # Last element is the footer note
  assert any("notation" in str(e) or "10-min" in str(e) for e in elements)
  # Verify at least one column_set with our action prefix
  found = False
  for el in elements:
    if el.get("tag") == "column_set":
      for col in el.get("columns", []):
        for child in col.get("elements", []):
          action = child.get("value", {}).get("action", "")
          if action.startswith("askq:abc123:0:"):
            found = True
  assert found, "expected at least one askq:abc123:0:* button"


def test_askq_card_button_action_strings():
  """Each option button gets action `askq:{nonce}:{qidx}:{oidx}`."""
  card = build_ask_user_question_card(
    questions=[_question("Where?", "Screen", ["Login", "Match", "Profile"])],
    chat_id="oc_x",
    nonce="N1",
  )
  actions: list[str] = []
  for el in card["body"]["elements"]:
    if el.get("tag") == "column_set":
      for col in el.get("columns", []):
        for child in col.get("elements", []):
          a = child.get("value", {}).get("action", "")
          if a:
            actions.append(a)
  # 3 option buttons + 1 "Other" button
  assert "askq:N1:0:0" in actions
  assert "askq:N1:0:1" in actions
  assert "askq:N1:0:2" in actions
  assert "askq:N1:0:other" in actions


def test_askq_card_multi_question():
  """Multiple questions are stacked with hr separators."""
  card = build_ask_user_question_card(
    questions=[
      _question("Where?", "Screen", ["A", "B"]),
      _question("Severity?", "How bad?", ["P0", "P3"]),
    ],
    chat_id="oc_x",
    nonce="abc",
  )
  elements = card["body"]["elements"]
  hr_count = sum(1 for e in elements if e.get("tag") == "hr")
  assert hr_count == 1, "expected one hr divider between two questions"
  # Verify both q0 and q1 buttons are present
  actions: list[str] = []
  for el in elements:
    if el.get("tag") == "column_set":
      for col in el.get("columns", []):
        for child in col.get("elements", []):
          a = child.get("value", {}).get("action", "")
          if a:
            actions.append(a)
  assert any(a.startswith("askq:abc:0:") for a in actions)
  assert any(a.startswith("askq:abc:1:") for a in actions)


def test_askq_card_multi_select_has_done_button():
  card = build_ask_user_question_card(
    questions=[_question("Pick any", "Tags", ["a", "b", "c"], multi_select=True)],
    chat_id="oc_x",
    nonce="N",
  )
  actions: list[str] = []
  for el in card["body"]["elements"]:
    if el.get("tag") == "column_set":
      for col in el.get("columns", []):
        for child in col.get("elements", []):
          a = child.get("value", {}).get("action", "")
          if a:
            actions.append(a)
  assert "askq:N:0:done" in actions


def test_askq_card_renders_selected_with_check():
  """Already-selected options show with a leading check mark and primary color."""
  card = build_ask_user_question_card(
    questions=[_question("Where?", "Screen", ["Login", "Match"])],
    chat_id="oc_x",
    nonce="N",
    answers={0: "Login"},
  )
  found_check = False
  for el in card["body"]["elements"]:
    if el.get("tag") == "column_set":
      for col in el.get("columns", []):
        for child in col.get("elements", []):
          label = child.get("text", {}).get("content", "")
          if label.startswith("✓ Login"):
            assert child.get("type") == "primary"
            found_check = True
  assert found_check, "expected selected option to render with ✓ and primary type"


def test_askq_card_chat_id_in_buttons():
  """Every button carries chat_id so card actions route correctly."""
  card = build_ask_user_question_card(
    questions=[_question("Q?", "Q", ["a"])],
    chat_id="oc_42",
    nonce="N",
  )
  for el in card["body"]["elements"]:
    if el.get("tag") == "column_set":
      for col in el.get("columns", []):
        for child in col.get("elements", []):
          assert child.get("value", {}).get("chat_id") == "oc_42"


def test_askq_card_no_questions_still_renders():
  """Empty questions list produces a card with just the footer (defensive)."""
  card = build_ask_user_question_card(
    questions=[],
    chat_id="oc_x",
    nonce="N",
  )
  assert card["schema"] == "2.0"
  assert isinstance(card["body"]["elements"], list)


# ---------------------------------------------------------------------------
# build_turn_card with embedded AskUserQuestion state
# ---------------------------------------------------------------------------


def test_turn_card_working_renders_pending_question_inline():
  """A PendingQuestion on the working card injects question header,
  options buttons (with chat_id) and an 'Other' fallback button."""
  from nemo.channel import PendingQuestion

  pending = PendingQuestion(
    questions=[_question("Where?", "Screen", ["Login", "Match"])],
    nonce="N1",
  )
  card = build_turn_card(
    "working",
    steps=[],
    elapsed=5,
    chat_id="oc_x",
    pending_question=pending,
  )
  elements = card["body"]["elements"]
  # Header markdown for the question must appear inline (not via a
  # separate card body wrapper).
  assert any(
    e.get("tag") == "markdown" and "Screen" in e.get("content", "")
    for e in elements
  )
  # Option + Other action strings show up with the right nonce/qidx/oidx.
  actions: list[str] = []
  for el in elements:
    if el.get("tag") == "column_set":
      for col in el.get("columns", []):
        for child in col.get("elements", []):
          a = child.get("value", {}).get("action", "")
          if a:
            actions.append(a)
  assert "askq:N1:0:0" in actions
  assert "askq:N1:0:1" in actions
  assert "askq:N1:0:other" in actions


def test_turn_card_working_renders_answered_summary_above_thinking():
  """Answered questions render as a compact `❓ … → ✅ …` line near the
  top of the card, before the current_tool / thinking sections."""
  from nemo.channel import AnsweredQuestion

  answered = [
    AnsweredQuestion(header="Severity", question="How bad?", answer="P0"),
    AnsweredQuestion(header="Where", question="Which screen?", answer="Login"),
  ]
  card = build_turn_card(
    "working",
    steps=[ThinkingStep("thinking", "looking around")],
    elapsed=12,
    chat_id="oc_x",
    answered_questions=answered,
    current_tool="Bash: grep",
  )
  elements = card["body"]["elements"]
  # First element is the answered-questions block.
  assert elements[0].get("tag") == "markdown"
  content = elements[0]["content"]
  assert "❓ **Severity**" in content
  assert "✅ P0" in content
  assert "❓ **Where**" in content
  assert "✅ Login" in content
  # Answered block appears before current_tool and thinking panel.
  positions = {
    "answered": 0,
    "current_tool": next(
      i for i, e in enumerate(elements)
      if e.get("tag") == "markdown" and "`Bash: grep`" in e.get("content", "")
    ),
    "thinking": next(
      i for i, e in enumerate(elements)
      if e.get("tag") == "collapsible_panel"
    ),
  }
  assert positions["answered"] < positions["current_tool"] < positions["thinking"]


def test_turn_card_done_carries_answered_questions():
  """The done card keeps the answered-question summary so the terminal
  card shows what the user picked, not just the model's final reply."""
  from nemo.channel import AnsweredQuestion

  answered = [AnsweredQuestion(
    header="Severity", question="How bad?", answer="P0")]
  card = build_turn_card(
    "done",
    body="ok",
    steps=[],
    elapsed=20,
    answered_questions=answered,
  )
  found = False
  for el in card["body"]["elements"]:
    if el.get("tag") == "markdown" and "❓ **Severity**" in el.get("content", ""):
      assert "✅ P0" in el["content"]
      found = True
      break
  assert found, "expected answered summary in the done card body"


def test_turn_card_stopped_drops_pending_question():
  """Once the turn is stopping/stopped/done the in-flight question
  buttons must go away — there's no live handler waiting for clicks."""
  from nemo.channel import PendingQuestion

  pending = PendingQuestion(
    questions=[_question("Q?", "Q", ["a"])],
    nonce="N",
  )
  for phase in ("stopping", "stopped", "done", "error"):
    card = build_turn_card(
      phase,
      body="x",
      steps=[],
      elapsed=1,
      pending_question=pending,
    )
    for el in card["body"]["elements"]:
      if el.get("tag") == "column_set":
        for col in el.get("columns", []):
          for child in col.get("elements", []):
            a = child.get("value", {}).get("action", "")
            assert not a.startswith("askq:"), \
              f"{phase} card must not show askq buttons (saw {a})"


def test_turn_card_pending_other_answer_renders_confirmation_line():
  """When a question has a typed 'Other' answer that isn't in the
  option list, the card must render a visible '✓ Your answer: <text>'
  confirmation line — otherwise clicking Other + typing a reply has
  no visual feedback on the card and looks like the bot ignored it."""
  from nemo.channel import PendingQuestion

  pending = PendingQuestion(
    questions=[_question("颜色", "颜色", ["red", "blue", "green"])],
    nonce="N",
    answers={0: "绿色"},  # free-text, not in options
  )
  card = build_turn_card("working", steps=[], elapsed=1,
                         chat_id="oc_x", pending_question=pending)
  found = False
  for el in card["body"]["elements"]:
    if el.get("tag") == "markdown":
      content = el.get("content", "")
      if "Your answer" in content and "绿色" in content:
        found = True
        break
  assert found, "expected '✓ Your answer: 绿色' confirmation line in card"


def test_turn_card_pending_option_answer_does_not_add_confirmation_line():
  """When the answer matches a predefined option (click, not Other),
  the option button already shows ✓ + primary color — no extra
  confirmation line is needed."""
  from nemo.channel import PendingQuestion

  pending = PendingQuestion(
    questions=[_question("Where?", "Screen", ["Login", "Match"])],
    nonce="N",
    answers={0: "Login"},  # matches an option label
  )
  card = build_turn_card("working", steps=[], elapsed=1,
                         chat_id="oc_x", pending_question=pending)
  for el in card["body"]["elements"]:
    if el.get("tag") == "markdown":
      assert "Your answer" not in el.get("content", "")


def test_turn_card_multi_select_answer_renders_as_list():
  """Multi-select answers render comma-separated in the summary line."""
  from nemo.channel import AnsweredQuestion

  answered = [AnsweredQuestion(
    header="Tags", question="Pick any", answer=["a", "c"])]
  card = build_turn_card("done", body="", steps=[], elapsed=1,
                         answered_questions=answered)
  for el in card["body"]["elements"]:
    if el.get("tag") == "markdown" and "Tags" in el.get("content", ""):
      assert "a, c" in el["content"]
      return
  raise AssertionError("expected multi-select answer rendered as 'a, c'")
