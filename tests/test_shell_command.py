import asyncio
import shlex
import sys

from nemo.shell_command import (
  ShellJobManager,
  ShellResult,
  format_context,
  parse_shell_shortcut,
)


def test_parse_shell_shortcut_injecting():
  req = parse_shell_shortcut("!echo hi")
  assert req is not None
  assert req.command == "echo hi"
  assert req.inject_context is True


def test_parse_shell_shortcut_non_injecting():
  req = parse_shell_shortcut("!!echo hi")
  assert req is not None
  assert req.command == "echo hi"
  assert req.inject_context is False


def test_parse_shell_shortcut_ignores_normal_text():
  assert parse_shell_shortcut("hello !echo hi") is None


def test_format_context_contains_command_and_output():
  result = ShellResult(
    job_id="job1",
    command="echo hi",
    cwd="/tmp/project",
    status="done",
    exit_code=0,
    duration=0.1,
    stdout="hi\n",
    stderr="",
  )
  text = format_context(result)
  assert "[Nemo shell context]" in text
  assert "echo hi" in text
  assert "hi" in text


class _ShellChannel:
  def __init__(self):
    self.sent_cards = []
    self.updated_cards = []

  async def send_card(self, chat_id, card):
    self.sent_cards.append((chat_id, card))
    return "om_shell"

  async def update_card(self, message_id, card):
    self.updated_cards.append((message_id, card))
    return message_id


def test_shell_job_manager_runs_and_queues_context(tmp_path):
  contexts: list[str] = []
  channel = _ShellChannel()
  manager = ShellJobManager(
    channel,
    chat_id="oc_test",
    project_dir=str(tmp_path),
    on_context=contexts.append,
    timeout=5,
  )

  async def _run():
    req = parse_shell_shortcut("!printf hello")
    assert req is not None
    await manager.start(req)
    while not contexts:
      await asyncio.sleep(0.05)
    await manager.close()

  asyncio.run(_run())

  assert channel.sent_cards
  assert channel.updated_cards
  assert "printf hello" in contexts[0]
  assert "hello" in contexts[0]


def test_shell_job_manager_abort_marks_context_aborted(tmp_path):
  contexts: list[str] = []
  channel = _ShellChannel()
  manager = ShellJobManager(
    channel,
    chat_id="oc_test",
    project_dir=str(tmp_path),
    on_context=contexts.append,
    timeout=30,
  )

  async def _run():
    req = parse_shell_shortcut("!sleep 30")
    assert req is not None
    await manager.start(req)
    job_id = next(iter(manager._jobs))
    assert await manager.abort(job_id) is True
    while not contexts:
      await asyncio.sleep(0.05)
    await manager.close()

  asyncio.run(_run())

  assert "aborted" in contexts[0]
  assert "sleep 30" in contexts[0]
  final_card = channel.updated_cards[-1][1]
  assert final_card["header"]["title"]["content"] == "Shell aborted"


def test_shell_job_manager_abort_without_id_aborts_latest_running(tmp_path):
  contexts: list[str] = []
  channel = _ShellChannel()
  manager = ShellJobManager(
    channel,
    chat_id="oc_test",
    project_dir=str(tmp_path),
    on_context=contexts.append,
    timeout=30,
  )

  async def _run():
    req = parse_shell_shortcut("!sleep 30")
    assert req is not None
    await manager.start(req)
    assert await manager.abort("") is True
    while not contexts:
      await asyncio.sleep(0.05)
    await manager.close()

  asyncio.run(_run())

  assert "aborted" in contexts[0]


def test_shell_job_manager_double_bang_does_not_queue_context(tmp_path):
  contexts: list[str] = []
  channel = _ShellChannel()
  manager = ShellJobManager(
    channel,
    chat_id="oc_test",
    project_dir=str(tmp_path),
    on_context=contexts.append,
    timeout=5,
  )

  async def _run():
    req = parse_shell_shortcut("!!printf hidden")
    assert req is not None
    await manager.start(req)
    while not channel.updated_cards:
      await asyncio.sleep(0.05)
    await manager.close()

  asyncio.run(_run())

  assert contexts == []
  final_card = channel.updated_cards[-1][1]
  assert final_card["header"]["title"]["content"] == "Shell done"


def test_shell_job_manager_failure_captures_stderr_and_exit_code(tmp_path):
  contexts: list[str] = []
  channel = _ShellChannel()
  manager = ShellJobManager(
    channel,
    chat_id="oc_test",
    project_dir=str(tmp_path),
    on_context=contexts.append,
    timeout=5,
  )

  async def _run():
    req = parse_shell_shortcut("!sh -c 'echo boom >&2; exit 7'")
    assert req is not None
    await manager.start(req)
    while not contexts:
      await asyncio.sleep(0.05)
    await manager.close()

  asyncio.run(_run())

  assert "failed" in contexts[0]
  assert "exit 7" in contexts[0]
  assert "boom" in contexts[0]
  final_card = channel.updated_cards[-1][1]
  assert final_card["header"]["title"]["content"] == "Shell failed"


def test_shell_job_manager_timeout_terminates_process(tmp_path):
  contexts: list[str] = []
  channel = _ShellChannel()
  manager = ShellJobManager(
    channel,
    chat_id="oc_test",
    project_dir=str(tmp_path),
    on_context=contexts.append,
    timeout=0.2,
  )

  async def _run():
    req = parse_shell_shortcut("!sleep 5")
    assert req is not None
    await manager.start(req)
    while not contexts:
      await asyncio.sleep(0.05)
    await manager.close()

  asyncio.run(_run())

  assert "timeout" in contexts[0]
  final_card = channel.updated_cards[-1][1]
  assert final_card["header"]["title"]["content"] == "Shell timed out"


def test_shell_job_manager_double_bang_ignores_default_timeout(tmp_path):
  contexts: list[str] = []
  channel = _ShellChannel()
  manager = ShellJobManager(
    channel,
    chat_id="oc_test",
    project_dir=str(tmp_path),
    on_context=contexts.append,
    timeout=0.1,
  )
  code = "import time; time.sleep(0.25); print('done')"
  command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"

  async def _run():
    req = parse_shell_shortcut(f"!!{command}")
    assert req is not None
    await manager.start(req)
    while not channel.updated_cards:
      await asyncio.sleep(0.05)
    await manager.close()

  asyncio.run(_run())

  assert contexts == []
  final_card = channel.updated_cards[-1][1]
  assert final_card["header"]["title"]["content"] == "Shell done"


def test_shell_job_manager_streaming_updates_card_before_done(tmp_path):
  contexts: list[str] = []
  channel = _ShellChannel()
  update_now = asyncio.Event()
  manager = ShellJobManager(
    channel,
    chat_id="oc_test",
    project_dir=str(tmp_path),
    on_context=contexts.append,
    timeout=5,
  )
  code = (
    "import time\n"
    "print('line-1', flush=True)\n"
    "time.sleep(0.9)\n"
    "print('line-2', flush=True)\n"
    "time.sleep(0.9)\n"
    "print('line-3', flush=True)\n"
  )
  command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"

  async def _periodic_update(job):
    await update_now.wait()
    await manager._safe_update_card(job)

  async def _run():
    req = parse_shell_shortcut(f"!{command}")
    assert req is not None
    original_periodic_update = manager._periodic_update
    manager._periodic_update = _periodic_update
    try:
      await manager.start(req)
      await asyncio.sleep(0.1)
      update_now.set()
    finally:
      manager._periodic_update = original_periodic_update
    while not contexts:
      await asyncio.sleep(0.05)
    await manager.close()

  asyncio.run(_run())

  titles = [
    card["header"]["title"]["content"]
    for _, card in channel.updated_cards
  ]
  assert "Shell running" in titles
  assert titles[-1] == "Shell done"
  assert "line-3" in contexts[0]


def test_shell_job_manager_truncates_large_card_and_context_output(tmp_path):
  contexts: list[str] = []
  channel = _ShellChannel()
  manager = ShellJobManager(
    channel,
    chat_id="oc_test",
    project_dir=str(tmp_path),
    on_context=contexts.append,
    timeout=5,
  )
  code = "print('A' * 45000)"
  command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"

  async def _run():
    req = parse_shell_shortcut(f"!{command}")
    assert req is not None
    await manager.start(req)
    while not contexts:
      await asyncio.sleep(0.05)
    await manager.close()

  asyncio.run(_run())

  assert "chars omitted" in contexts[0]
  final_card = channel.updated_cards[-1][1]
  rendered = str(final_card)
  assert "chars omitted" in rendered
  assert len(rendered) < 20_000
