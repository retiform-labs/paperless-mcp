import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { after, before, describe, test } from "node:test";

/**
 * The runner must be able to FAIL.
 *
 * ⚠️ THIS FILE EXISTS BECAUSE THE PREVIOUS WRAPPER COULD NOT. It ended with
 *
 *     npx ts-node scripts/build-index.ts > "$log" 2>&1
 *     echo "EXIT=$?"
 *
 * and `echo` exits 0. So a build that died at document 1,727 of 2,466 was
 * reported by systemd as a clean exit, and the only evidence was a log nobody
 * had a reason to open, because the run had reported success. The cost was not
 * the failed build -- it was that the next session had to re-derive the failure
 * from scratch.
 *
 * A unit test on a shell script is unusual, and it is the only instrument that
 * sits between "the script looks right" and "the script propagates a status".
 * Every existing test in this suite would stay green if the wrapper reverted to
 * a bare echo, because none of them run it.
 */

const repoRoot = resolve(__dirname, "..", "..");
const runnerPath = join(repoRoot, "scripts", "run-index.sh");

let dir: string;

before(() => {
  dir = mkdtempSync(join(tmpdir(), "paperless-runner-"));
});

after(() => {
  rmSync(dir, { recursive: true, force: true });
});

/**
 * An env file whose build is guaranteed to fail quickly.
 *
 * The Paperless URL points at a closed port, so the failure happens on the
 * first document-list request rather than after minutes of embedding. The
 * embedder URL is left at the real one only because the config parser requires
 * it; the build never reaches an embedding call.
 */
function failingEnv(): string {
  const path = join(dir, "failing.env");
  writeFileSync(
    path,
    [
      "PAPERLESS_URL=http://127.0.0.1:59999",
      "PAPERLESS_API_KEY=not-a-real-token",
      "PAPERLESS_MCP_EMBEDDING_URL=http://127.0.0.1:11434",
      "PAPERLESS_MCP_EMBEDDING_MODEL=nomic-embed-text",
      "PAPERLESS_MCP_EMBEDDING_PREFIX_MODE=nomic",
      `PAPERLESS_MCP_INDEX_DIR=${join(dir, "index")}`,
      "PAPERLESS_MCP_MAX_CHUNKS=16",
    ].join("\n") + "\n",
    { mode: 0o600 }
  );
  return path;
}

interface RunResult {
  status: number | null;
  stderr: string;
  log: string;
}

function runRunner(envFile: string, logName: string): RunResult {
  const logPath = join(dir, logName);
  const result = spawnSync("bash", [runnerPath], {
    cwd: repoRoot,
    env: {
      ...process.env,
      PAPERLESS_INDEX_ENV: envFile,
      PAPERLESS_INDEX_LOG: logPath,
      NODE_BIN_DIR: process.env.NODE_BIN_DIR ?? "",
    },
    encoding: "utf8",
    timeout: 180_000,
  });
  return {
    status: result.status,
    stderr: result.stderr ?? "",
    log: existsSync(logPath) ? readFileSync(logPath, "utf8") : "",
  };
}

describe("the index runner reports the build's real exit code", () => {
  test("a failing build exits non-zero", () => {
    // ⚠️ THE ASSERTION IS THE WHOLE POINT. Against the old wrapper shape this
    // is 0 and the test fails; that is the exact defect, reproduced in one call
    // instead of in a twenty-call investigation of an "unavailable" server.
    const result = runRunner(failingEnv(), "failing.log");
    assert.notEqual(
      result.status,
      0,
      `a failed build must not exit 0 (stderr: ${result.stderr.slice(0, 400)})`
    );
  });

  test("the failure is visible without reading the log, and recorded in it", () => {
    // Two independent surfaces, because they serve different readers: the exit
    // code is what systemd sees, the log line is what a human greps later.
    const result = runRunner(failingEnv(), "failing2.log");
    assert.notEqual(result.status, 0);
    assert.match(result.log, /EXIT=0|\bEXIT=\d/);
    assert.doesNotMatch(
      result.log,
      /^EXIT=0$/m,
      "the logged code must not claim success for a failed build"
    );
    // The build writes its own verdict line before exiting; as long as the
    // wrapper appends the code, a reader can correlate the two.
    assert.match(result.stderr, /run-index: finished with \d+/);
  });

  test("a missing env file is a configuration error, not a crashed build", () => {
    // Distinct from a build failure on purpose: an operator who reads exit 1 as
    // "the embedding server is down" would go and restart a server that was
    // never the problem.
    const result = runRunner(join(dir, "does-not-exist.env"), "noenv.log");
    assert.equal(result.status, 2);
    assert.match(result.stderr, /no env file/);
  });

  test("the runner does not invoke npx through a shell string", () => {
    // A shell string would re-expand the paths and reintroduce a quoting layer
    // that the corpus text never reaches but the exit code does.
    const source = readFileSync(runnerPath, "utf8");
    assert.match(source, /^#!.*bash/m);
    // The last statement must be the exit, or the status is lost again.
    const tail = source.trimEnd().split("\n").slice(-3).join("\n");
    assert.match(
      tail,
      /exit "\$rc"/,
      "the final statement must re-raise the captured status"
    );
    assert.doesNotMatch(
      source.trimEnd().split("\n").slice(-1)[0],
      /^\s*echo\b/,
      "a trailing echo would overwrite the status, which is the original defect"
    );
  });
});
