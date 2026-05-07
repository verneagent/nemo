#!/usr/bin/env node

import { stdin, stdout, stderr, exit, argv, env } from "node:process";
import { Codex } from "@openai/codex-sdk";
import { streamToStdout } from "./resume.mjs";

const VALID_EFFORTS = new Set(["minimal", "low", "medium", "high", "xhigh"]);

function parseArgs(rawArgs) {
  const options = {
    cwd: "",
    model: "",
    resume: "",
    effort: "",
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
  // third-party endpoint like DeepSeek's OpenAI-compatible host) we
  // need to override two codex CLI defaults that bite us otherwise:
  //
  //   1. ChatGPT OAuth precedence. codex picks the user's
  //      ~/.codex/auth.json token even when OPENAI_API_KEY is set in
  //      the env, then refuses non-whitelisted models on that auth
  //      mode ("The 'deepseek-v4-pro' model is not supported when
  //      using Codex with a ChatGPT account"). Passing apiKey via
  //      the SDK forces API-key auth and skips the whitelist.
  //
  //   2. Wire protocol. codex's built-in "openai" provider talks to
  //      /v1/responses (OpenAI's Responses API). DeepSeek and most
  //      third-party "OpenAI-compatible" services only implement
  //      /v1/chat/completions, so the responses-format request 404s.
  //      We patch the built-in provider's wire_api → "chat" via the
  //      SDK's --config passthrough.
  const codexOptions = {};
  if (env.OPENAI_BASE_URL) {
    codexOptions.baseUrl = env.OPENAI_BASE_URL;
    codexOptions.config = {
      model_providers: {
        openai: { wire_api: "chat" },
      },
    };
  }
  if (env.OPENAI_API_KEY) codexOptions.apiKey = env.OPENAI_API_KEY;
  const codex = new Codex(codexOptions);
  const threadOptions = {
    workingDirectory: options.cwd || undefined,
    model: options.model || undefined,
    modelReasoningEffort: options.effort || undefined,
    skipGitRepoCheck: true,
    sandboxMode: "danger-full-access",
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
