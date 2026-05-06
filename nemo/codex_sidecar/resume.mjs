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
//
// IMPORTANT: the codex SDK throws "no rollout found" *lazily* from inside
// the events async iterator — `runStreamed()` itself returns
// successfully, but the first `next()` on `events` rejects. So the
// fallback has to wrap iteration, not just the call site. We only retry
// when nothing has been written to stdout yet — falling back mid-stream
// would splice events from two different codex threads into one nemo
// turn.

export function isResumeUnrecoverable(error) {
  const message = String(error?.message || error || "");
  return (
    /no rollout found/i.test(message)
    || /thread\/resume\s*(failed|:)/i.test(message)
    || /thread\s+not\s+found/i.test(message)
  );
}

export async function streamToStdout(codex, threadOptions, prompt, resumeId, stdout, stderr) {
  if (!resumeId) {
    const { events } = await codex.startThread(threadOptions).runStreamed(prompt);
    for await (const event of events) {
      stdout.write(`${JSON.stringify(event)}\n`);
    }
    return;
  }

  let emittedFromResume = false;
  try {
    const { events } = await codex.resumeThread(resumeId, threadOptions).runStreamed(prompt);
    for await (const event of events) {
      emittedFromResume = true;
      stdout.write(`${JSON.stringify(event)}\n`);
    }
    return;
  } catch (error) {
    if (emittedFromResume || !isResumeUnrecoverable(error)) {
      throw error;
    }
    const summary = String(error?.message || error || "").split("\n")[0];
    stderr.write(
      `[codex sidecar] resume ${resumeId} unusable (${summary}) — starting fresh thread\n`
    );
  }

  // Resume failed before yielding any event — safe to fall back.
  const { events } = await codex.startThread(threadOptions).runStreamed(prompt);
  for await (const event of events) {
    stdout.write(`${JSON.stringify(event)}\n`);
  }
}
