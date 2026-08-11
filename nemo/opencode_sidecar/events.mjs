export function normalizeUsage(tokens) {
  if (!tokens || typeof tokens !== "object") {
    return {};
  }
  const usage = {};
  if (typeof tokens.input === "number") {
    usage.input_tokens = tokens.input;
  }
  if (typeof tokens.output === "number") {
    usage.output_tokens = tokens.output;
  }
  if (tokens.cache && typeof tokens.cache === "object" && typeof tokens.cache.read === "number") {
    usage.cached_input_tokens = tokens.cache.read;
  }
  return usage;
}

function handlePart(part, delta) {
  if (!part || typeof part !== "object") {
    return null;
  }
  if (part.type === "text") {
    const text = typeof delta === "string" && delta ? delta : typeof part.text === "string" ? part.text : "";
    if (!text) {
      return null;
    }
    return { type: "item.completed", item: { type: "agent_message", text } };
  }
  if (part.type === "reasoning") {
    const text = typeof delta === "string" && delta ? delta : typeof part.text === "string" ? part.text : "";
    if (!text) {
      return null;
    }
    return { type: "item.completed", item: { type: "reasoning", text } };
  }
  if (part.type === "tool" && part.state && typeof part.state === "object") {
    const state = part.state;
    if (state.status === "completed" || state.status === "running") {
      return {
        type: "item.completed",
        item: {
          type: "tool_call",
          tool: typeof part.tool === "string" ? part.tool : "",
          title: typeof state.title === "string" ? state.title : "",
          input: state.input && typeof state.input === "object" ? state.input : {},
          // Lets the daemon disarm its idle-stall timeout while a tool is in
          // flight (a long bash run produces no events until it returns).
          status: state.status,
        },
      };
    }
  }
  if (part.type === "step-finish") {
    return {
      type: "turn.completed",
      usage: normalizeUsage(part.tokens),
      cost: typeof part.cost === "number" ? part.cost : 0,
    };
  }
  return null;
}

export function createEventMapper(sessionID) {
  const assistantMessageIDs = new Set();
  return function mapEvent(event) {
    if (!event || typeof event !== "object") {
      return null;
    }
    if (event.type === "message.updated") {
      const info = event.properties?.info;
      if (info?.sessionID === sessionID && info.role === "assistant" && typeof info.id === "string") {
        assistantMessageIDs.add(info.id);
      }
      return null;
    }
    if (event.type !== "message.part.updated") {
      return null;
    }
    const part = event.properties?.part;
    if (!part || part.sessionID !== sessionID || !assistantMessageIDs.has(part.messageID)) {
      return null;
    }
    return handlePart(part, event.properties?.delta);
  };
}
