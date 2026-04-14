#!/usr/bin/env node

import { stdin, stdout, stderr, exit, argv } from "node:process";
import { Codex } from "@openai/codex-sdk";

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
  const codex = new Codex();
  const threadOptions = {
    workingDirectory: options.cwd || undefined,
    model: options.model || undefined,
    modelReasoningEffort: options.effort || undefined,
    skipGitRepoCheck: true,
    sandboxMode: "danger-full-access",
    approvalPolicy: "never",
  };
  const thread = options.resume
    ? codex.resumeThread(options.resume, threadOptions)
    : codex.startThread(threadOptions);
  const { events } = await thread.runStreamed(prompt);
  for await (const event of events) {
    stdout.write(`${JSON.stringify(event)}\n`);
  }
}

main().catch((error) => {
  const message = error instanceof Error ? error.stack || error.message : String(error);
  stderr.write(`${message}\n`);
  exit(1);
});
