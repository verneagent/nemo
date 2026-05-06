// Tests for resume.mjs. Run with `node test_resume.mjs` — exits 0 on
// pass, prints failing assertions and exits 1 otherwise. The Python
// pytest harness (tests/test_codex_sidecar_resume.py) shells out to
// this file so the regression travels with the JS source.

import { isResumeUnrecoverable, streamToStdout } from "./resume.mjs";

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
// streamToStdout test scaffolding.
// ---------------------------------------------------------------------------

// Build an async iterable from a list. Items can be:
//   - a value: yielded normally
//   - {throw: errorMessage}: the iterator rejects on next() instead of
//     yielding (this models the codex SDK's lazy "no rollout" throw).
function fakeEvents(items) {
  const queue = items.slice();
  return {
    [Symbol.asyncIterator]() {
      return {
        async next() {
          if (queue.length === 0) return { done: true, value: undefined };
          const next = queue.shift();
          if (next && typeof next === "object" && "throw" in next) {
            throw new Error(next.throw);
          }
          return { done: false, value: next };
        },
      };
    },
  };
}

function makeCodex({ resumeEvents, startEvents, resumeSyncThrow }) {
  return {
    resumeThread(_id, _opts) {
      return {
        async runStreamed(_prompt) {
          if (resumeSyncThrow) throw new Error(resumeSyncThrow);
          return { events: fakeEvents(resumeEvents ?? []) };
        },
      };
    },
    startThread(_opts) {
      return {
        async runStreamed(_prompt) {
          return { events: fakeEvents(startEvents ?? []) };
        },
      };
    },
  };
}

class FakeStream {
  constructor() { this.lines = []; }
  write(line) { this.lines.push(line); return true; }
}

(async () => {
  // No resume id → only startThread is consulted.
  {
    const stdout = new FakeStream();
    const stderr = new FakeStream();
    const codex = makeCodex({ startEvents: [{ type: "ok", n: 1 }] });
    await streamToStdout(codex, {}, "hi", "", stdout, stderr);
    check(
      "no resume → events from startThread",
      stdout.lines.length === 1 && stdout.lines[0].includes('"type":"ok"'),
      `stdout=${JSON.stringify(stdout.lines)}`,
    );
    check(
      "no resume → no fallback log",
      stderr.lines.length === 0,
    );
  }

  // Successful resume → events flow, no fallback.
  {
    const stdout = new FakeStream();
    const stderr = new FakeStream();
    const codex = makeCodex({
      resumeEvents: [{ type: "started" }, { type: "done" }],
      startEvents: [{ type: "wrong-source" }],
    });
    await streamToStdout(codex, {}, "hi", "abc-1", stdout, stderr);
    check(
      "successful resume → only resume events",
      stdout.lines.length === 2
      && stdout.lines.every(l => !l.includes("wrong-source")),
      `stdout=${JSON.stringify(stdout.lines)}`,
    );
    check(
      "successful resume → no fallback log",
      stderr.lines.length === 0,
    );
  }

  // Lazy throw on first iteration → fall back, log, emit fresh-thread events.
  {
    const stdout = new FakeStream();
    const stderr = new FakeStream();
    const codex = makeCodex({
      resumeEvents: [
        { throw: "thread/resume: thread/resume failed: no rollout found for thread id abc-1" },
      ],
      startEvents: [{ type: "fresh-1" }, { type: "fresh-2" }],
    });
    await streamToStdout(codex, {}, "hi", "abc-1", stdout, stderr);
    check(
      "lazy iterator throw before any event → fallback emits fresh events",
      stdout.lines.length === 2
      && stdout.lines[0].includes("fresh-1")
      && stdout.lines[1].includes("fresh-2"),
      `stdout=${JSON.stringify(stdout.lines)}`,
    );
    check(
      "lazy iterator throw → stderr breadcrumb",
      stderr.lines.length === 1
      && stderr.lines[0].includes("[codex sidecar]")
      && stderr.lines[0].includes("abc-1")
      && stderr.lines[0].includes("starting fresh thread"),
      `stderr=${JSON.stringify(stderr.lines)}`,
    );
  }

  // Sync throw from runStreamed itself (older SDK behavior we still want to
  // recover from) → fall back.
  {
    const stdout = new FakeStream();
    const stderr = new FakeStream();
    const codex = makeCodex({
      resumeSyncThrow: "no rollout found for thread id abc-1",
      startEvents: [{ type: "fresh" }],
    });
    await streamToStdout(codex, {}, "hi", "abc-1", stdout, stderr);
    check(
      "sync throw from resumeThread.runStreamed → fallback",
      stdout.lines.length === 1 && stdout.lines[0].includes("fresh"),
      `stdout=${JSON.stringify(stdout.lines)}`,
    );
    check(
      "sync throw → stderr breadcrumb",
      stderr.lines.length === 1 && stderr.lines[0].includes("starting fresh thread"),
    );
  }

  // Unrelated error during iteration → MUST bubble (don't mask real bugs).
  {
    const stdout = new FakeStream();
    const stderr = new FakeStream();
    const codex = makeCodex({
      resumeEvents: [{ throw: "ECONNREFUSED 127.0.0.1:6152" }],
      startEvents: [{ type: "should-not-emit" }],
    });
    let raised = null;
    try {
      await streamToStdout(codex, {}, "hi", "abc-1", stdout, stderr);
    } catch (err) { raised = err; }
    check(
      "non-resume iterator error bubbles",
      raised && /ECONNREFUSED/.test(raised.message),
      `raised=${raised && raised.message}`,
    );
    check(
      "non-resume error does NOT fall back to fresh thread",
      stdout.lines.length === 0,
      `stdout=${JSON.stringify(stdout.lines)}`,
    );
  }

  // Mid-stream resume failure (events yielded, THEN iterator rejects) →
  // do NOT splice fresh thread events on top; bubble so the host sees one
  // turn ended badly rather than a frankenstein turn.
  {
    const stdout = new FakeStream();
    const stderr = new FakeStream();
    const codex = makeCodex({
      resumeEvents: [
        { type: "partial-1" },
        { throw: "no rollout found for thread id abc-1" },
      ],
      startEvents: [{ type: "should-not-emit" }],
    });
    let raised = null;
    try {
      await streamToStdout(codex, {}, "hi", "abc-1", stdout, stderr);
    } catch (err) { raised = err; }
    check(
      "mid-stream resume error bubbles (no splicing)",
      raised && /no rollout/.test(raised.message),
      `raised=${raised && raised.message}`,
    );
    check(
      "mid-stream error keeps the partial event but doesn't add fresh",
      stdout.lines.length === 1 && stdout.lines[0].includes("partial-1"),
      `stdout=${JSON.stringify(stdout.lines)}`,
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
