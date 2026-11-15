import { test } from "node:test";
import assert from "node:assert/strict";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { createMcpServer } from "./server";

// The MCP spec on tools/list ordering:
//
//   "Servers SHOULD return tools in a deterministic order (i.e., the same
//    ordering across requests when the underlying set of tools has not
//    changed). Deterministic ordering enables clients to reliably cache the
//    tool list and improves LLM prompt cache hit rates when tools are included
//    in model context."
//
// The harness is the one schemaCompat.test.ts already uses -- a real server and
// a real client over InMemoryTransport -- rather than a second mechanism.
// Asserting against `server._registeredTools` would test an implementation
// detail; asserting against `client.listTools()` tests the wire.

async function listToolNames(): Promise<string[]> {
  const server = createMcpServer({
    baseUrl: "http://paperless.test",
    token: "test-token",
    version: "0.0.0-test",
    publicUrl: "http://paperless.test",
  });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  const client = new Client({ name: "tool-order-test", version: "1.0.0" });
  await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
  try {
    const { tools } = await client.listTools();
    return tools.map((tool) => tool.name);
  } finally {
    await client.close();
    await server.close();
  }
}

test("two independent server instances advertise tools in the same order", async () => {
  const first = await listToolNames();
  const second = await listToolNames();

  assert.ok(first.length > 0, "the server must advertise at least one tool");
  assert.deepEqual(
    second,
    first,
    "tools/list order differed between two instances built in the same process"
  );
});

test("repeated tools/list calls on one instance agree", async () => {
  const server = createMcpServer({
    baseUrl: "http://paperless.test",
    token: "test-token",
    version: "0.0.0-test",
    publicUrl: "http://paperless.test",
  });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  const client = new Client({ name: "tool-order-test", version: "1.0.0" });
  await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
  try {
    const first = (await client.listTools()).tools.map((tool) => tool.name);
    const second = (await client.listTools()).tools.map((tool) => tool.name);
    assert.deepEqual(second, first);
  } finally {
    await client.close();
    await server.close();
  }
});

test("the advertised order is ascending by name, code-unit comparison", async () => {
  // Pinning the ORDER ITSELF, not just its stability. Stability alone would
  // pass on any frozen accidental order; this fails if a registration is
  // reordered, if a tool is renamed, or if someone swaps in `localeCompare`
  // (which sorts by collation and diverges on punctuation and case).
  const names = await listToolNames();
  const expected = [...names].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  assert.deepEqual(names, expected, `tools/list is not sorted: ${names.join(", ")}`);
});

test("sorting does not drop, duplicate or rename any tool", async () => {
  // A sort that silently loses an entry would still be "deterministic", so
  // stability and ordering are not sufficient on their own.
  const names = await listToolNames();
  const unique = new Set(names);
  assert.equal(unique.size, names.length, "duplicate tool names in tools/list");

  // The document/tag/correspondent/type/custom-field/mail families are always
  // registered; the semantic-search trio is conditional on config and is
  // therefore correctly absent here.
  for (const required of [
    "list_documents",
    "get_document",
    "search_passages",
    "list_tags",
    "list_correspondents",
    "list_document_types",
    "list_custom_fields",
    "list_mail_accounts",
  ]) {
    assert.ok(unique.has(required), `${required} went missing from tools/list`);
  }
});
