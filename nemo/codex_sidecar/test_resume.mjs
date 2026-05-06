// Tests for resume.mjs. Run with `node test_resume.mjs` — exits 0 on
// pass, prints the failing assertion and exits 1 otherwise. The Python
// pytest harness (tests/test_codex_sidecar_resume.py) shells out to this
// file so the regression travels with the JS source.

import { isResumeUnrecoverable, startStream } from "./resume.mjs";

let passed = 0;
const failures = [];

function check(name, cond, detail = "") {
  if (cond) {
    passed += 1;
  } else {
    failures.push(`${name}${detail ? ` — ${detail}` : ""}`);
  }
}

// ---------------------------------------------------------------------------
// isResumeUnrecoverable: the three error shapes we see from the codex SDK.
// ---------------------------------------------------------------------------

check(
  "isResumeUnrecoverable: no rollout found",
  isResumeUnrecoverable(new Error(
    "Codex Exec exited with code 1: Error: thread/resume: thread/resume failed: "
    + "no rollout found for thread id f1fab3f9-bad5-4281-ac43-c606a9c9c700"
  )),
);
check(
  "isResumeUnrecoverable: thread/resume failed (no rollout phrase)",
  isResumeUnrecoverable(new Error("thread/resume: failed for unknown reason")),
);
check(
  "isResumeUnrecoverable: thread not found",
  isResumeUnrecoverable(new Error("thread not found in store")),
);
check(
  "isResumeUnrecoverable: ignores unrelated errors",
  !isResumeUnrecoverable(new Error("ECONNREFUSED")),
);
check(
  "isResumeUnrecoverable: ignores empty / null",
  !isResumeUnrecoverable(undefined),
);

// ---------------------------------------------------------------------------
// startStream: with no resume id → goes straight to startThread.
// ---------------------------------------------------------------------------

function fakeStreamResult(label) {
  return { events: [`events-from-${label}`] };
}

function makeCodex({ onResume, onStart }) {
  return {
    resumeThread(id, _opts) {
      const thread = {
        runStreamed(_prompt) {
          if (onResume) return onResume(id);
          return fakeStreamResult(`resume-${id}`);
        },
      };
      return thread;
    },
    startThread(_opts) {
      return {
        runStreamed(_prompt) {
          if (onStart) return onStart();
          return fakeStreamResult("start");
        },
      };
    },
  };
}

class FakeStderr {
  constructor() { this.lines = []; }
  write(line) { this.lines.push(line); return true; }
}

(async () => {
  // No resume id → startThread, no fallback log.
  {
    const stderr = new FakeStderr();
    const codex = makeCodex({});
    const { events } = await startStream(codex, {}, "hi", "", stderr);
    check(
      "startStream: no resume → startThread",
      events[0] === "events-from-start",
      `events=${JSON.stringify(events)}`,
    );
    check(
      "startStream: no resume → no fallback log",
      stderr.lines.length === 0,
      `stderr=${JSON.stringify(stderr.lines)}`,
    );
  }

  // Resume id, resumeThread succeeds → return events from resume.
  {
    const stderr = new FakeStderr();
    const codex = makeCodex({});
    const { events } = await startStream(codex, {}, "hi", "abc-123", stderr);
    check(
      "startStream: successful resume → resume events",
      events[0] === "events-from-resume-abc-123",
      `events=${JSON.stringify(events)}`,
    );
    check(
      "startStream: successful resume → no fallback log",
      stderr.lines.length === 0,
      `stderr=${JSON.stringify(stderr.lines)}`,
    );
  }

  // Resume id, resumeThread throws "no rollout" → fall back to startThread,
  // emit a stderr line for operator visibility.
  {
    const stderr = new FakeStderr();
    const codex = makeCodex({
      onResume: () => {
        throw new Error(
          "thread/resume: thread/resume failed: no rollout found for thread id abc-123"
        );
      },
    });
    const { events } = await startStream(codex, {}, "hi", "abc-123", stderr);
    check(
      "startStream: unrecoverable resume → fall back to startThread",
      events[0] === "events-from-start",
      `events=${JSON.stringify(events)}`,
    );
    check(
      "startStream: fallback emits stderr breadcrumb",
      stderr.lines.length === 1
      && stderr.lines[0].includes("[codex sidecar]")
      && stderr.lines[0].includes("abc-123")
      && stderr.lines[0].includes("starting fresh thread"),
      `stderr=${JSON.stringify(stderr.lines)}`,
    );
  }

  // Resume id, resumeThread throws something OTHER than the resume-class —
  // do NOT swallow; let it bubble so genuine bugs surface.
  {
    const stderr = new FakeStderr();
    const codex = makeCodex({
      onResume: () => { throw new Error("ECONNREFUSED 127.0.0.1:1234"); },
    });
    let raised = null;
    try {
      await startStream(codex, {}, "hi", "abc-123", stderr);
    } catch (err) { raised = err; }
    check(
      "startStream: non-resume error bubbles",
      raised && /ECONNREFUSED/.test(raised.message),
      `raised=${raised && raised.message}`,
    );
    check(
      "startStream: non-resume error does NOT log fallback",
      stderr.lines.length === 0,
      `stderr=${JSON.stringify(stderr.lines)}`,
    );
  }

  if (failures.length === 0) {
    process.stdout.write(`OK: ${passed} assertions passed\n`);
    process.exit(0);
  } else {
    process.stderr.write(
      `FAIL: ${failures.length} of ${passed + failures.length} assertions failed:\n`
    );
    for (const f of failures) process.stderr.write(`  - ${f}\n`);
    process.exit(1);
  }
})();
