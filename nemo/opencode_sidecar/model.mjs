// Model-resolution helpers for the OpenCode sidecar. Kept in their own
// module so the test suite can exercise them without spawning a server.

// Provider name the daemon injects for single-name nemo presets (oc-*,
// deepseek, kimi, ...). The daemon translates such a preset into
// `nemo/<remote>` and hands the target endpoint over via env.
export const INJECTED_PROVIDER_ID = "nemo";

// OpenCode resolves models as `provider/model` slugs against providers it
// knows about. `default` (or empty) means "use OpenCode's configured default".
export function modelBody(model) {
  if (!model || model === "default") {
    return undefined;
  }
  const slash = model.indexOf("/");
  if (slash <= 0 || slash >= model.length - 1) {
    return undefined;
  }
  return {
    providerID: model.slice(0, slash),
    modelID: model.slice(slash + 1),
  };
}

// The requested model must resolve to a `provider/model` slug unless it is
// the default — a bare name used to silently fall back to OpenCode's DEFAULT
// model, which is a silent wrong-model bug. Now it is rejected loudly instead.
export function resolvableModel(model) {
  return !model || model === "default" || Boolean(modelBody(model));
}

// Build the `provider` config fragment that reaches the requested endpoint.
// OpenCode validates a model id against the provider's declared `models`, so
// a single-name preset (no `provider/` prefix) cannot be used as-is — the
// daemon injects it as a provider OpenCode can resolve.
// Returns `null` when no injection is requested.
export function injectedProvider(env, modelID) {
  if (!env.NEMO_OPENCODE_PROVIDER_URL || !modelID) {
    return null;
  }
  const options = { baseURL: env.NEMO_OPENCODE_PROVIDER_URL };
  if (env.NEMO_OPENCODE_PROVIDER_API_KEY) {
    options.apiKey = env.NEMO_OPENCODE_PROVIDER_API_KEY;
  }
  return {
    [INJECTED_PROVIDER_ID]: {
      npm: env.NEMO_OPENCODE_PROVIDER_NPM || "@ai-sdk/openai-compatible",
      name: "Nemo endpoint",
      options,
      models: { [modelID]: {} },
    },
  };
}
