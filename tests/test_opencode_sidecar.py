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


def test_model_resolution_and_provider_injection() -> None:
  sidecar_dir = Path(__file__).resolve().parents[1] / "nemo" / "opencode_sidecar"
  program = """
import { modelBody, resolvableModel, injectedProvider } from "./model.mjs";

// JSON.stringify drops `undefined` values (the keys vanish), so keep keys by
// converting undefined → null for the expected-undefined fields.
const j = (v) => v === undefined ? null : v;

const out = {
  defaultModel: j(modelBody("default")),
  emptyModel: j(modelBody("")),
  slashModel: j(modelBody("deepseek/deepseek-v4-flash")),
  singleName: j(modelBody("oc-deepseek-v4-flash")),
  resolvableDefault: j(resolvableModel("default")),
  resolvableEmpty: j(resolvableModel("")),
  resolvableSlash: j(resolvableModel("deepseek/deepseek-v4-flash")),
  unresolvable: j(resolvableModel("oc-deepseek-v4-flash")),
  noInjection: j(injectedProvider({}, "deepseek-v4-flash")),
  injection: j(injectedProvider({
    NEMO_OPENCODE_PROVIDER_URL: "https://opencode.ai/zen/go/v1",
    NEMO_OPENCODE_PROVIDER_API_KEY: "sk-test",
    NEMO_OPENCODE_PROVIDER_NPM: "@ai-sdk/openai-compatible",
  }, "deepseek-v4-flash")),
  injectionNoKey: j(injectedProvider({
    NEMO_OPENCODE_PROVIDER_URL: "https://x",
  }, "m")),
};
process.stdout.write(JSON.stringify(out));
"""
  result = subprocess.run(
    ["node", "--input-type=module", "-e", program],
    cwd=sidecar_dir,
    capture_output=True,
    text=True,
    check=True,
  )
  parsed = json.loads(result.stdout)
  # `default` / empty resolve to OpenCode's default model.
  assert parsed["defaultModel"] is None
  assert parsed["emptyModel"] is None
  # A `provider/model` slug splits; a bare name does not.
  assert parsed["slashModel"] == {
    "providerID": "deepseek", "modelID": "deepseek-v4-flash"}
  assert parsed["singleName"] is None
  # Bare names must NOT silently fall back to the default model.
  assert parsed["resolvableDefault"] is True
  assert parsed["resolvableEmpty"] is True
  assert parsed["resolvableSlash"] is True
  assert parsed["unresolvable"] is False
  # No env → no injection.
  assert parsed["noInjection"] is None
  # Provider injected from env, declaring the requested model.
  assert parsed["injection"] == {
    "nemo": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Nemo endpoint",
      "options": {
        "baseURL": "https://opencode.ai/zen/go/v1",
        "apiKey": "sk-test",
      },
      "models": {"deepseek-v4-flash": {}},
    },
  }
  assert parsed["injectionNoKey"] == {
    "nemo": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Nemo endpoint",
      "options": {"baseURL": "https://x"},
      "models": {"m": {}},
    },
  }
