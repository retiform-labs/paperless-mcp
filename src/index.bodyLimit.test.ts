import assert from "node:assert/strict";
import { test } from "node:test";
import { spawn } from "node:child_process";
import { once } from "node:events";

/**
 * The HTTP transport must accept a body larger than express.json own 100 kb
 * default, because post_document legitimately carries a base64 file of up to
 * 100 MB. Before this was bounded explicitly, a legal upload was rejected by the
 * transport before the tool ever ran.
 *
 * These tests start the real server rather than a hand-built express app:
 * index.ts exports nothing, so assembling a lookalike would test a private copy
 * of the subject instead of the subject.
 */
const PORT = 3917;
const BASE = "http://127.0.0.1:" + PORT + "/mcp";

type Server = { child: ReturnType<typeof spawn>; stop: () => Promise<void> };

async function startServer(env: Record<string, string> = {}): Promise<Server> {
  const child = spawn(
    process.execPath,
    ["--require", "ts-node/register", "src/index.ts", "--http", "--port", String(PORT), "--baseUrl", "http://paperless.test", "--token", "body-limit-test"],
    { env: { ...process.env, ...env }, stdio: ["ignore", "pipe", "pipe"] }
  );

  const ready = new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("server did not start within 20s")), 20_000);
    child.stdout?.on("data", (chunk: Buffer) => {
      if (chunk.toString().includes("listening on port")) {
        clearTimeout(timer);
        resolve();
      }
    });
    child.on("exit", (code) => {
      clearTimeout(timer);
      reject(new Error("server exited early with code " + code));
    });
  });

  await ready;
  return {
    child,
    stop: async () => {
      child.kill("SIGTERM");
      await once(child, "exit").catch(() => undefined);
    },
  };
}

function post(body: string) {
  return fetch(BASE, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: "Bearer body-limit-test",
    },
    body,
  });
}

/** A JSON-RPC frame whose padding pushes the total body well past a given size. */
function paddedRequest(bytes: number): string {
  const pad = "x".repeat(Math.max(0, bytes));
  return JSON.stringify({
    jsonrpc: "2.0",
    id: 1,
    method: "initialize",
    params: { protocolVersion: "2024-11-05", capabilities: {}, clientInfo: { name: pad, version: "1" } },
  });
}

test("a body past the old 100 kb default is accepted, not silently refused", async () => {
  const server = await startServer();
  try {
    const body = paddedRequest(2 * 1024 * 1024);
    assert.ok(body.length > 1024 * 1024, "fixture is wrong: body must exceed 1 MB");

    const res = await post(body);
    // The transport must NOT answer 413/400 for a body it explicitly allows.
    // A JSON-RPC error about the request content is fine; a size rejection is not.
    assert.notEqual(res.status, 413, "a 2 MB body was refused as too large, but this server permits uploads up to 100 MB");
    assert.ok(res.status < 500, "unexpected server fault on a legal body: " + res.status);
  } finally {
    await server.stop();
  }
});

test("an oversized body is refused with a status, not a crash", async () => {
  const server = await startServer({ PAPERLESS_MCP_MAX_BODY_BYTES: String(64 * 1024) });
  try {
    const body = paddedRequest(512 * 1024);
    const res = await post(body);
    assert.equal(res.status, 413, "a body above the configured limit must be refused with 413");
  } finally {
    await server.stop();
  }
});

test("the refusal names no filesystem path or stack trace", async () => {
  const server = await startServer({ PAPERLESS_MCP_MAX_BODY_BYTES: String(64 * 1024) });
  try {
    const res = await post(paddedRequest(512 * 1024));
    const text = await res.text();
    assert.ok(
      !text.includes("node_modules") && !text.includes(".js:"),
      "the rejection body is disclosing the server filesystem layout: " + text.slice(0, 200)
    );
  } finally {
    await server.stop();
  }
});

test("malformed JSON is a clean 400, and the server survives it", async () => {
  const server = await startServer();
  try {
    const bad = await post("{not json");
    assert.equal(bad.status, 400, "malformed JSON must be a 400");

    // Positive control: the same server still answers a well-formed request, so
    // "survived" is measured rather than assumed.
    const good = await post(paddedRequest(0));
    assert.ok(good.status < 500, "the server did not survive a malformed body: " + good.status);
  } finally {
    await server.stop();
  }
});
