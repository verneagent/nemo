#!/usr/bin/env node

import { stdout, stderr, exit, argv } from "node:process";
import { createOpencodeClient } from "@opencode-ai/sdk/client";
import { createOpencodeServer } from "@opencode-ai/sdk/server";

function parseArgs(rawArgs) {
  const options = { cwd: "" };
  for (let i = 0; i < rawArgs.length; i += 1) {
    const arg = rawArgs[i];
    if (arg === "--cwd") {
      options.cwd = rawArgs[++i] ?? "";
    } else {
      throw new Error(`unknown arg: ${arg}`);
    }
  }
  return options;
}

function unwrap(result) {
  if (result && typeof result === "object" && "data" in result) {
    return result.data;
  }
  return result;
}

async function main() {
  const options = parseArgs(argv.slice(2));
  const directory = options.cwd || undefined;
  const server = await createOpencodeServer({ port: 0 });
  const client = createOpencodeClient({
    baseUrl: server.url,
    responseStyle: "data",
    throwOnError: true,
  });

  try {
    const configured = unwrap(await client.config.providers({
      query: { directory },
    }));
    const providers = Array.isArray(configured?.providers) ? configured.providers : [];
    const defaults = configured && typeof configured.default === "object" ? configured.default : {};
    const models = [];

    for (const provider of providers) {
      if (!provider || typeof provider !== "object") {
        continue;
      }
      const providerID = typeof provider.id === "string" ? provider.id : "";
      const providerModels = provider.models && typeof provider.models === "object"
        ? provider.models
        : {};
      for (const modelID of Object.keys(providerModels)) {
        if (providerID && modelID) {
          models.push(`${providerID}/${modelID}`);
        }
      }
    }

    stdout.write(JSON.stringify({
      models: Array.from(new Set(models)).sort(),
      default_model: typeof defaults.model === "string" ? defaults.model : "",
    }));
  } finally {
    server.close();
  }
}

main().catch((error) => {
  const message = error instanceof Error ? error.stack || error.message : String(error);
  stderr.write(`${message}\n`);
  exit(1);
});
