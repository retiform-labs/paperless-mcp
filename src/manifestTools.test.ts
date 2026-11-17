import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { createMcpServer } from "./server";

// The manifest carried a HAND-MAINTAINED tools[] and it drifted exactly the way
// the version field did: 13 entries against the 48 the server actually registers,
// with every semantic tool absent. A DXT/marketplace host renders the manifest, so
// a capability missing there is a capability the marketplace does not know exists.
//
// scripts/sync-manifest-tools.mjs now derives the list from a real stdio handshake.
// That fixes drift at pack time, but a script is not a guarantee: nothing stops a
// hand-edit of manifest.json, or a new server.tool(...) whose author forgets the
// sync. This test makes the advertised surface PROVABLY complete.

const root = join(__dirname, "..");

function manifestToolNames(): string[] {
  const manifest = JSON.parse(readFileSync(join(root, "manifest.json"), "utf8")) as {
    tools?: { name?: unknown }[];
  };
  assert.ok(Array.isArray(manifest.tools), "manifest.json has no tools[] array");
  return manifest.tools.map((t) => {
    assert.equal(
      typeof t.name,
      "string",
      "manifest tool entry without a name: " + JSON.stringify(t)
    );
    return t.name as string;
  });
}

/**
 * The live tool list, from a real handshake.
 *
 * THE SEARCH TOOLS ARE CONDITIONALLY REGISTERED. registerSearchTools returns
 * early unless context.isConfigured, and the config is read from process.env at
 * CALL time. A bare environment therefore yields the DEFAULT surface and
 * silently omits the four semantic tools -- which is the exact defect this test
 * exists to catch, reintroduced by the instrument. Setting one search variable
 * makes the gate true. The index is never opened: tools/list is answered from
 * the registered handlers and does not touch the store.
 */
async function liveToolNames(): Promise<string[]> {
  const savedUrl = process.env.PAPERLESS_MCP_EMBEDDING_URL;
  process.env.PAPERLESS_MCP_EMBEDDING_URL = "http://127.0.0.1:1";
  const server = createMcpServer({
    baseUrl: "http://paperless.test",
    token: "test-token",
    version: "0.0.0-test",
    publicUrl: "http://paperless.test",
  });
  const pair = InMemoryTransport.createLinkedPair();
  const client = new Client({ name: "manifest-tools-test", version: "1.0.0" });
  await Promise.all([server.connect(pair[1]), client.connect(pair[0])]);
  try {
    const { tools } = await client.listTools();
    assert.ok(
      tools.length > 0,
      "tools/list returned nothing -- the instrument measured an empty surface"
    );
    return tools.map((t) => t.name);
  } finally {
    await client.close();
    await server.close();
    if (savedUrl === undefined) delete process.env.PAPERLESS_MCP_EMBEDDING_URL;
    else process.env.PAPERLESS_MCP_EMBEDDING_URL = savedUrl;
  }
}

test("manifest.json advertises every tool the server actually registers", async () => {
  const advertised = manifestToolNames();
  const live = await liveToolNames();

  const missing = live.filter((name) => !advertised.includes(name)).sort();
  assert.deepEqual(
    missing,
    [],
    "manifest.json omits " +
      missing.length +
      " registered tool(s): " +
      missing.join(", ") +
      "\nThe manifest is what a DXT/marketplace host renders, so these capabilities\n" +
      "are invisible to it even though the server serves them.\n" +
      "Run: npm run sync-manifest-tools"
  );

  // The count is asserted explicitly too, so a change that leaves both sides
  // empty or equally wrong cannot pass as agreement.
  assert.equal(
    advertised.length,
    live.length,
    "manifest lists " + advertised.length + " tools, the server registers " + live.length
  );
});

test("manifest.json advertises no tool the server does not register", async () => {
  const advertised = manifestToolNames();
  const live = await liveToolNames();

  const phantom = advertised.filter((name) => !live.includes(name)).sort();
  assert.deepEqual(
    phantom,
    [],
    "manifest.json advertises " +
      phantom.length +
      " tool(s) no client can call: " +
      phantom.join(", ") +
      "\nA stale entry is worse than a missing one: a host offers it and every call fails."
  );

  const dupes = advertised.filter((name, i) => advertised.indexOf(name) !== i);
  assert.deepEqual(dupes, [], "manifest.json repeats tool name(s): " + dupes.join(", "));
});

test("manifest.json tools[] is name-sorted, so a regenerated file is byte-stable", () => {
  // sync-manifest-tools.mjs writes a localeCompare-sorted list and rewrites only
  // when the content differs. If the committed file is NOT in that order, the
  // very next pack rewrites it -- and a publish that dirties a tracked file is
  // the failure the CI pack gate exists for.
  const names = manifestToolNames();
  const sorted = [...names].sort((a, b) => a.localeCompare(b));
  assert.deepEqual(
    names,
    sorted,
    "manifest.json tools[] is not in the order the sync script writes"
  );
});
