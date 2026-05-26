#!/usr/bin/env node

import { stdin, stdout, stderr, exit, argv, env } from "node:process";
import { Codex } from "@openai/codex-sdk";
import { streamToStdout } from "./resume.mjs";

const VALID_EFFORTS = new Set(["minimal", "low", "medium", "high", "xhigh"]);
const VALID_SANDBOX = new Set(["read-only", "workspace-write", "danger-full-access"]);

function parseArgs(rawArgs) {
  const options = {
    cwd: "",
    model: "",
    resume: "",
    effort: "",
    sandbox: "",
  };
  for (let i = 0; i < rawArgs.length; i += 1) {
    const arg = rawArgs[i];
    if (arg === "--cwd") {
      options.cwd = rawArgs[++i] ?? "";
    } else if (arg === "--model") {
      options.model = rawArgs[++i] ?? "";
    } else if (arg === "--resume") {
      options.resume = rawArgs[++i] ?? "";
    } else if (arg === "--effort") {
      const value = rawArgs[++i] ?? "";
      if (value && !VALID_EFFORTS.has(value)) {
        throw new Error(`invalid --effort value: ${value}`);
      }
      options.effort = value;
    } else if (arg === "--sandbox") {
      const value = rawArgs[++i] ?? "";
      if (value && !VALID_SANDBOX.has(value)) {
        throw new Error(`invalid --sandbox value: ${value}`);
      }
      options.sandbox = value;
    } else {
      throw new Error(`unknown arg: ${arg}`);
    }
  }
  return options;
}

async function readPrompt() {
  const chunks = [];
  for await (const chunk of stdin) {
    chunks.push(typeof chunk === "string" ? Buffer.from(chunk) : chunk);
  }
  return Buffer.concat(chunks).toString("utf8");
}

async function main() {
  const options = parseArgs(argv.slice(2));
  const prompt = await readPrompt();
  // When OPENAI_BASE_URL is set (preset registry pointed nemo at a
  // third-party endpoint) we forward baseUrl + apiKey to the codex
  // SDK constructor. apiKey is critical: the codex CLI otherwise
  // reads the user's ~/.codex/auth.json OAuth token first and
  // rejects non-whitelisted models on ChatGPT-account auth ("The
  // 'deepseek-v4-pro' model is not supported when using Codex with
  // a ChatGPT account"). Passing apiKey forces API-key mode.
  //
  // We deliberately do NOT touch wire_api here. Codex 0.128+
  // dropped support for `wire_api = "chat"` entirely
  // (https://github.com/openai/codex/discussions/7782) — the CLI
  // hard-rejects the value at config-load time. The default
  // "responses" protocol is the only option, which means codex
  // against a third-party endpoint requires that endpoint to
  // implement /v1/responses. DeepSeek's OpenAI-compatible host
  // currently only exposes /v1/chat/completions, so codex +
  // deepseek-v4-pro will 404 until DeepSeek ships a responses
  // adapter or codex restores wire_api=chat. Use Claude provider
  // (Anthropic-protocol DeepSeek endpoint) for now.
  const codexOptions = {};
  if (env.OPENAI_BASE_URL) codexOptions.baseUrl = env.OPENAI_BASE_URL;
  if (env.OPENAI_API_KEY) codexOptions.apiKey = env.OPENAI_API_KEY;
  const codex = new Codex(codexOptions);
  const threadOptions = {
    workingDirectory: options.cwd || undefined,
    model: options.model || undefined,
    modelReasoningEffort: options.effort || undefined,
    skipGitRepoCheck: true,
    // Default full-access for normal turns; /fork passes --sandbox read-only
    // so a forked branch physically cannot modify the project.
    sandboxMode: options.sandbox || "danger-full-access",
    approvalPolicy: "never",
  };
  await streamToStdout(
    codex, threadOptions, prompt, options.resume, stdout, stderr);
}

main().catch((error) => {
  const message = error instanceof Error ? error.stack || error.message : String(error);
  stderr.write(`${message}\n`);
  exit(1);
});
