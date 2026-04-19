from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_event_mapper_ignores_user_message_parts() -> None:
  sidecar_dir = Path(__file__).resolve().parents[1] / "nemo" / "opencode_sidecar"
  program = """
import { createEventMapper } from "./events.mjs";

const mapEvent = createEventMapper("sess-1");
const ignored = mapEvent({
  type: "message.part.updated",
  properties: {
    part: { sessionID: "sess-1", messageID: "msg-user", type: "text", text: "echo" },
    delta: "echo",
  },
});
mapEvent({
  type: "message.updated",
  properties: {
    info: { id: "msg-assistant", sessionID: "sess-1", role: "assistant" },
  },
});
const accepted = mapEvent({
  type: "message.part.updated",
  properties: {
    part: { sessionID: "sess-1", messageID: "msg-assistant", type: "text", text: "pong" },
    delta: "pong",
  },
});
process.stdout.write(JSON.stringify({ ignored, accepted }));
"""
  result = subprocess.run(
    ["node", "--input-type=module", "-e", program],
    cwd=sidecar_dir,
    capture_output=True,
    text=True,
    check=True,
  )
  parsed = json.loads(result.stdout)
  assert parsed["ignored"] is None
  assert parsed["accepted"] == {
    "type": "item.completed",
    "item": {"type": "agent_message", "text": "pong"},
  }
