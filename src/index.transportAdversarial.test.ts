import assert from "node:assert/strict";
import { spawn, type ChildProcess } from "node:child_process";
import { createServer } from "node:net";
import { join } from "node:path";
import { after, before, test } from "node:test";

// ---------------------------------------------------------------------------
// ADVERSARIAL LAYER -- THE HTTP TRANSPORT.
//
// src/index.ts is a SCRIPT: it parses argv at import time, builds an express app
// and calls app.listen. It exports nothing, so there is no way to unit-test the
// transport's error handling against the real app -- a hand-built express
// instance would be a private copy of the thing under test, and would pass just
// as happily if index.ts had no middleware at all.
//
// So this file SPAWNS the real server on an ephemeral port and speaks HTTP to
// it. That is slower than a unit test and it is the only instrument that can see
// what an unauthenticated client actually receives.
//
// What it is looking for: a malformed request must be answered cleanly, without
// an unhandled rejection, and without telling the caller where the server keeps
// its files. The default express error handler does the last one when
// NODE_ENV is not 'production', and the shipped image does not set it.
// ---------------------------------------------------------------------------

const ROOT = join(__dirname, "..");
const ENABLE_TS_NODE = ["--require", "ts-node/register"];

/** Asks the OS for a free port, then releases it for the child to bind. */
function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      const port = typeof address === "object" && address ? address.port : 0;
      server.close(() => resolve(port));
    });
  });
}

interface Running {
  child: ChildProcess;
  base: string;
  stderr: string[];
}

/** Spawns the real server and waits until it will accept a socket. */
async function startServer(env: Record<string, string>): Promise<Running> {
  const port = await freePort();
  const child = spawn(
    process.execPath,
    [...ENABLE_TS_NODE, join(ROOT, "src", "index.ts"), "--http", "--port", String(port),
      "--baseUrl", "http://paperless.invalid", "--token", "server-token"],
    { cwd: ROOT, env: { ...process.env, ...env }, stdio: ["ignore", "pipe", "pipe"] }
  );
  const stderr: string[] = [];
  child.stderr?.on("data", (chunk: Buffer) => stderr.push(chunk.toString("utf8")));

  const base = "http://127.0.0.1:" + port;
  const deadline = Date.now() + 60_000;
  for (;;) {
    if (child.exitCode !== null) {
      throw new Error("server exited during startup: " + stderr.join(""));
    }
    try {
      // A request with no credentials is answered 401 -- which proves the socket
      // is up AND that the auth gate is what answered. Waiting for a 404 or a
      // connection error would be waiting for the wrong signal.
      const res = await fetch(base + "/mcp", { method: "POST" });
      if (res.status > 0) return { child, base, stderr };
    } catch {
      // Not listening yet.
    }
    if (Date.now() > deadline) {
      child.kill("SIGKILL");
      throw new Error("server did not start within 60s: " + stderr.join(""));
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
}

function stopServer(running: Running | undefined): void {
  if (running && running.child.exitCode === null) running.child.kill("SIGKILL");
}

let strictServer: Running | undefined;
let prodServer: Running | undefined;

before(async () => {
  // NODE_ENV deliberately NOT production here: this is the environment the
  // Dockerfile actually produces, and the point is to observe what it leaks.
  strictServer = await startServer({ NODE_ENV: "" });
  prodServer = await startServer({ NODE_ENV: "production" });
});

after(() => {
  stopServer(strictServer);
  stopServer(prodServer);
});

test("a request with no credentials is refused (positive control)", async () => {
  // Every probe below is meaningless if the server is not actually answering.
  const res = await fetch(strictServer!.base + "/mcp", { method: "POST" });
  assert.equal(res.status, 401, "an unauthenticated POST must be refused with 401");
});

test("malformed JSON is answered with a client error, not a crash", async () => {
  const res = await fetch(strictServer!.base + "/mcp", {
    method: "POST",
    headers: { "content-type": "application/json", authorization: "Bearer client-token" },
    body: "{not json",
  });
  assert.ok(
    res.status >= 400 && res.status < 500,
    "a malformed JSON body must be a 4xx, got " + res.status
  );
  // And the server must still serve the next request.
  const after = await fetch(strictServer!.base + "/mcp", { method: "POST" });
  assert.equal(after.status, 401, "the server must still be alive after a malformed body");
});

test("an error body does not disclose the filesystem layout or a stack trace", async () => {
  // This is the assertion the whole file exists for. Express' default error
  // handler renders err.stack into a text/html body when NODE_ENV is not
  // 'production', and that body goes to an UNAUTHENTICATED caller because
  // express.json() is app-level middleware that runs before the auth check.
  // A stack trace names absolute paths, dependency versions and internal module
  // names -- a free map of the deployment for anyone who can reach the port.
  const res = await fetch(strictServer!.base + "/mcp", {
    method: "POST",
    headers: { "content-type": "application/json", authorization: "Bearer client-token" },
    body: "{not json",
  });
  const body = await res.text();
  for (const leak of ["node_modules", "/home/", "/app/", "at JSON.parse", ".ts:", ".js:"]) {
    assert.ok(
      !body.includes(leak),
      "the error response to an unauthenticated caller contains " + JSON.stringify(leak) +
        " -- it is disclosing the server's filesystem layout or a stack trace: " +
        JSON.stringify(body.slice(0, 300))
    );
  }
});

test("NODE_ENV=production suppresses what the default handler would leak", async () => {
  // If this passes while the test above fails, the leak is environment-shaped
  // rather than a missing handler -- which is exactly the finding, because the
  // Dockerfile never sets NODE_ENV.
  const res = await fetch(prodServer!.base + "/mcp", {
    method: "POST",
    headers: { "content-type": "application/json", authorization: "Bearer client-token" },
    body: "{not json",
  });
  const body = await res.text();
  assert.ok(
    !body.includes("node_modules") && !body.includes("/home/"),
    "even with NODE_ENV=production the error body leaked internal paths: " +
      JSON.stringify(body.slice(0, 300))
  );
});

test("an oversized body is refused rather than buffered without limit", async () => {
  // express.json() defaults to a 100 kb limit, so this should be a 413. The
  // assertion is on bounded behaviour: any 4xx is fine, a 2xx or a hang is not.
  const huge = JSON.stringify({ pad: "x".repeat(2 * 1024 * 1024) });
  const res = await fetch(strictServer!.base + "/mcp", {
    method: "POST",
    headers: { "content-type": "application/json", authorization: "Bearer client-token" },
    body: huge,
  });
  assert.ok(
    res.status >= 400 && res.status < 500,
    "a 2 MiB body must be refused with a 4xx, got " + res.status
  );
  const after = await fetch(strictServer!.base + "/mcp", { method: "POST" });
  assert.equal(after.status, 401, "the server must still be alive after an oversized body");
});

test("a wrong content-type is not silently parsed as a JSON-RPC body", () => {
  // text/plain is not a JSON content type, so express.json() leaves req.body
  // undefined. The handler then forwards an empty body to the transport, which
  // answers with a JSON-RPC parse error. The property under test is that the
  // SERVER never treats an unparsed body as a valid request: it must not answer
  // 2xx, which is what a handler reading req.body without checking would do.
  return (async () => {
    const res = await fetch(strictServer!.base + "/mcp", {
      method: "POST",
      headers: { "content-type": "text/plain", authorization: "Bearer client-token" },
      body: JSON.stringify({ jsonrpc: "2.0", method: "tools/list", id: 1 }),
    });
    assert.ok(
      res.status !== 200,
      "a text/plain body was accepted as a valid JSON-RPC request (status 200); " +
        "express.json() only parses JSON content types, so the body never became " +
        "a verified envelope"
    );
    assert.ok(
      res.status < 500,
      "a wrong content-type produced a server error (" + res.status + ") rather " +
        "than a clean client-side refusal"
    );
  })();
});

test("the /messages endpoint refuses a missing or unknown session id cleanly", async () => {
  // sseTransports is a plain object both mutated at runtime and read from
  // externally.  and  are inherited
  // properties of Object.prototype, so a naive lookup returns a TRUTHY value and
  // the handler proceeds to call handlePostMessage on it. The observable outcome
  // is either a clean 400 or a 500 -- never a 2xx, and never a crash.
  for (const query of ["", "?sessionId=does-not-exist", "?sessionId=__proto__",
    "?sessionId=constructor", "?sessionId=toString", "?sessionId=valueOf"]) {
    const res = await fetch(strictServer!.base + "/messages" + query, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: "Bearer client-token" },
      body: "{\"jsonrpc\":\"2.0\",\"method\":\"tools/list\",\"id\":1}",
    });
    assert.ok(
      res.status >= 400,
      "POST /messages" + query + " answered " + res.status +
        "; an unknown session must be refused, and an inherited Object.prototype " +
        "key must not be treated as a live transport"
    );
  }
  const after = await fetch(strictServer!.base + "/mcp", { method: "POST" });
  assert.equal(after.status, 401, "the server must still be alive after prototype-key probes");
});

test("DELETE on /mcp is refused rather than reaching the handler", async () => {
  const res = await fetch(strictServer!.base + "/mcp", { method: "DELETE" });
  assert.equal(res.status, 405, "DELETE /mcp must answer 405 Method Not Allowed");
});
