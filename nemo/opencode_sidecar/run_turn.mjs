#!/usr/bin/env node

import { stdin, stdout, stderr, exit, argv, env } from "node:process";
import { createOpencodeClient } from "@opencode-ai/sdk/client";
import { createOpencodeServer } from "@opencode-ai/sdk/server";
import { createEventMapper } from "./events.mjs";
import { modelBody, resolvableModel, injectedProvider } from "./model.mjs";

function parseArgs(rawArgs) {
  const options = {
    cwd: "",
    model: "",
    resume: "",
  };
  for (let i = 0; i < rawArgs.length; i += 1) {
    const arg = rawArgs[i];
    if (arg === "--cwd") {
      options.cwd = rawArgs[++i] ?? "";
    } else if (arg === "--model") {
      options.model = rawArgs[++i] ?? "";
    } else if (arg === "--resume") {
      options.resume = rawArgs[++i] ?? "";
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

function emit(event) {
  return new Promise((resolve, reject) => {
    stdout.write(`${JSON.stringify(event)}\n`, (error) => {
      if (error) {
        reject(error);
        return;
      }
      resolve();
    });
  });
}

function unwrap(result) {
  if (result && typeof result === "object" && "data" in result) {
    return result.data;
  }
  return result;
}

async function main() {
  const options = parseArgs(argv.slice(2));
  const prompt = await readPrompt();
  const systemPrompt = env.NEMO_OPENCODE_SYSTEM_PROMPT || undefined;
  const model = modelBody(options.model);
  if (!resolvableModel(options.model)) {
    await emit({
      type: "turn.failed",
      error: {
        message: `Unresolvable model: ${options.model} (expected provider/model, e.g. deepseek/deepseek-v4-flash)`,
      },
    });
    exit(1);
  }
  const provider = injectedProvider(env, model ? model.modelID : "");
  const server = await createOpencodeServer({
    port: 0,
    config: {
      permission: {
        edit: "allow",
        bash: "allow",
        webfetch: "allow",
        doom_loop: "allow",
        external_directory: "allow",
      },
      ...(provider ? { provider } : {}),
    },
  });
  const client = createOpencodeClient({
    baseUrl: server.url,
    responseStyle: "data",
    throwOnError: true,
  });
  const directory = options.cwd || undefined;
  let sessionID = options.resume || "";
  let events;
  // A turn is only complete once the server emits turn.completed / session.idle.
  // If the event stream ends without either (e.g. an unusable cwd silently
  // kills the session), exit 0 with zero events would look like a successful
  // empty turn to the daemon — fail loudly instead.
  let completed = false;
  const eventsAbort = new AbortController();
  try {
    events = await client.event.subscribe({
      query: { directory },
      signal: eventsAbort.signal,
    });
    if (!sessionID) {
      const session = unwrap(await client.session.create({
        query: { directory },
        body: { title: "Nemo" },
      }));
      sessionID = session.id;
    }

    await emit({ type: "session.started", session_id: sessionID });
    const mapEvent = createEventMapper(sessionID);

    await client.session.promptAsync({
      path: { id: sessionID },
      query: { directory },
      body: {
        model,
        system: systemPrompt,
        parts: [{ type: "text", text: prompt }],
      },
    });

    for await (const event of events.stream) {
      if (!event || typeof event !== "object") {
        continue;
      }
      if (event.type === "session.error") {
        if (event.properties?.sessionID && event.properties.sessionID !== sessionID) {
          continue;
        }
        const error = event.properties?.error;
        const message = error?.data?.message || error?.name || "OpenCode turn failed";
        await emit({ type: "turn.failed", error: { message } });
        return;
      }
      if (event.type === "permission.updated") {
        const permission = event.properties;
        if (!permission || permission.sessionID !== sessionID) {
          continue;
        }
        await client.postSessionIdPermissionsPermissionId({
          path: { id: sessionID, permissionID: permission.id },
          query: { directory },
          body: { response: "once" },
        });
        continue;
      }
      const mapped = mapEvent(event);
      if (mapped) {
        await emit(mapped);
        if (mapped.type === "turn.completed") {
          completed = true;
          eventsAbort.abort();
          server.close();
          exit(0);
        }
        continue;
      }
      if (event.type === "session.idle") {
        if (event.properties?.sessionID !== sessionID) {
          continue;
        }
        completed = true;
        eventsAbort.abort();
        server.close();
        exit(0);
      }
    }
    if (!completed) {
      await emit({
        type: "turn.failed",
        error: { message: "OpenCode turn ended without a completion event" },
      });
      exit(1);
    }
  } finally {
    eventsAbort.abort();
    server.close();
  }
}

main().catch((error) => {
  const message = error instanceof Error ? error.stack || error.message : String(error);
  stderr.write(`${message}\n`);
  emit({ type: "turn.failed", error: { message } });
  exit(1);
});
