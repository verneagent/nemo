// Resume-fallback helpers for the codex sidecar.
//
// The codex SDK's `resumeThread()` requires a local rollout file at
// ~/.codex/sessions/.../<thread-id>.jsonl. That file can be missing for
// several reasons that should NOT crash the daemon:
//   - Cross-provider session-id reuse: nemo persists `sdk_session_id` per
//     chat regardless of provider, so a Claude session UUID can leak into
//     a Codex run after the user switches providers.
//   - The rollout was never written (a previous run started a thread but
//     died before any successful turn persisted to disk).
//   - Manual cleanup of ~/.codex/sessions, or moving between machines.
//
// In all three cases, the only correct behavior is "fall back to a fresh
// thread, log loudly, and let nemo's DB pick up the new thread.started
// id from the next event so future turns resume the right thread."

export function isResumeUnrecoverable(error) {
  const message = String(error?.message || error || "");
  return (
    /no rollout found/i.test(message)
    || /thread\/resume\s*(failed|:)/i.test(message)
    || /thread\s+not\s+found/i.test(message)
  );
}

export async function startStream(codex, threadOptions, prompt, resumeId, stderr) {
  if (!resumeId) {
    const thread = codex.startThread(threadOptions);
    return { thread, events: (await thread.runStreamed(prompt)).events };
  }
  try {
    const thread = codex.resumeThread(resumeId, threadOptions);
    return { thread, events: (await thread.runStreamed(prompt)).events };
  } catch (error) {
    if (!isResumeUnrecoverable(error)) {
      throw error;
    }
    const summary = String(error?.message || error || "").split("\n")[0];
    stderr.write(
      `[codex sidecar] resume ${resumeId} unusable (${summary}) — starting fresh thread\n`
    );
    const thread = codex.startThread(threadOptions);
    return { thread, events: (await thread.runStreamed(prompt)).events };
  }
}
