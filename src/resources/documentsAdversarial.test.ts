import assert from "node:assert/strict";
import { test } from "node:test";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { Transport } from "@modelcontextprotocol/sdk/shared/transport";
import type { JSONRPCMessage } from "@modelcontextprotocol/sdk/types";
import type { AxiosResponse } from "axios";
import { PaperlessAPI } from "../api/PaperlessAPI";
import { registerDocumentResources } from "./documents";

// ---------------------------------------------------------------------------
// ADVERSARIAL LAYER -- RESOURCE IDENTIFIERS.
//
// A resource URI is caller-supplied text that ends up as a path segment on a
// request to Paperless.  is the only guard, and the interesting
// question is not 'does it reject nonsense' -- it is 'does it ever ACCEPT
// something that is not the integer the caller appeared to send'.
//
// Number() is a productive coercer: it accepts hex, binary, exponent notation,
// leading whitespace, a leading plus, and a trailing '.0'. Every one of those is
// a silent reinterpretation, which is the class of defect the brief names. The
// traversal question is answered by the same assertions: an accepted id is a
// positive integer, so it cannot become '../' or escape the documents namespace.
// ---------------------------------------------------------------------------

class TestTransport implements Transport {
  peer?: TestTransport;
  onclose?: () => void;
  onerror?: (error: Error) => void;
  onmessage?: (message: JSONRPCMessage) => void;

  async start(): Promise<void> {}

  async send(message: JSONRPCMessage): Promise<void> {
    queueMicrotask(() => this.peer?.onmessage?.(message));
  }

  async close(): Promise<void> {
    this.onclose?.();
  }
}

function createTransportPair() {
  const clientTransport = new TestTransport();
  const serverTransport = new TestTransport();
  clientTransport.peer = serverTransport;
  serverTransport.peer = clientTransport;
  return { clientTransport, serverTransport };
}

/**
 * Records what the API was asked for, so a test can assert on the RESULT of
 * parsing rather than on whether it threw. A rejected id and an id silently
 * rewritten to a different document look identical from the outside unless the
 * resolved argument is captured.
 */
function recordingApi(): { api: PaperlessAPI; calls: number[] } {
  const calls: number[] = [];
  const api = {
    downloadDocument: async (id: number) => {
      calls.push(id);
      return {
        data: Buffer.from("pdf"),
        headers: { "content-type": "application/pdf" },
      } as unknown as AxiosResponse;
    },
    getThumbnail: async (id: number) => {
      calls.push(id);
      return {
        data: Buffer.from("png"),
        headers: { "content-type": "image/png" },
      } as unknown as AxiosResponse;
    },
  } as unknown as PaperlessAPI;
  return { api, calls };
}

async function readVia(idSegment: string): Promise<{ error?: string; calls: number[] }> {
  const { api, calls } = recordingApi();
  const server = new McpServer({ name: "adv", version: "0.0.0" });
  registerDocumentResources(server, api);
  const { clientTransport, serverTransport } = createTransportPair();
  const client = new Client({ name: "adv-client", version: "0.0.0" });
  await server.connect(serverTransport);
  await client.connect(clientTransport);
  try {
    await client.readResource({
      uri: "paperless://documents/" + idSegment + "/download",
    });
    return { calls };
  } catch (error) {
    return { error: (error as Error).message, calls };
  } finally {
    await client.close();
  }
}

test("a plain document id reads that exact document (positive control)", async () => {
  // Without this, every rejection below could be a broken fixture rather than a
  // working guard.
  const { calls, error } = await readVia("42");
  assert.equal(error, undefined, "a well-formed id must not error: " + error);
  assert.deepEqual(calls, [42], "the API must be asked for document 42 and only 42");
});

test("identifiers that Number() would silently reinterpret are refused", async () => {
  // Each of these is accepted by Number() and would resolve to a real, DIFFERENT
  // document. Reading document 16 because a caller wrote '0x10' is a silent
  // reinterpretation with real consequences: the caller believes it addressed
  // one document and received another.
  const coercible = [
    "0x10",      // 16
    "0b11",      // 3
    "0o17",      // 15
    "1e3",       // 1000
    "5.0",       // 5
    "+5",        // 5
    " 5 ",       // 5
  ];
  // NOTE: a raw tab or newline is NOT in this list on purpose. The WHATWG URL
  // parser strips control characters from a URI before our code sees it, so
  // "5\t" arrives as "5" -- a platform normalisation, not a
  // silent reinterpretation we introduced or could defend against here.
  for (const segment of coercible) {
    const { calls, error } = await readVia(segment);
    assert.ok(
      error !== undefined,
      "id " + JSON.stringify(segment) + " was accepted and resolved to " +
        JSON.stringify(calls) + ". Number() coerces this into a different document " +
        "than the caller wrote; the identifier must be refused rather than reinterpreted."
    );
    assert.deepEqual(calls, [], "a rejected id must never reach the API");
  }
});

test("identifiers that resolve to a non-positive integer are refused", async () => {
  const invalid = ["-5", "0", "-0", "1_0", "Infinity", "-Infinity", "NaN", "", "5abc", "abc"];
  for (const segment of invalid) {
    const { calls, error } = await readVia(segment);
    assert.ok(
      error !== undefined,
      "id " + JSON.stringify(segment) + " was accepted (resolved to " +
        JSON.stringify(calls) + ") but it is not a positive integer"
    );
    assert.deepEqual(calls, [], "a rejected id must never reach the API");
  }
});

test("an id beyond Number.MAX_SAFE_INTEGER is not silently rounded", async () => {
  // 2^53 + 1 is unrepresentable: Number("9007199254740993") === 9007199254740992.
  // The rounded value is still a positive integer, so a guard that only checks
  // positivity accepts it and the request addresses a DIFFERENT document than
  // the caller wrote. There is no way to detect that from the response.
  //
  // Asserted WITHOUT a branch: an "if it was accepted, then..." shape would pass
  // silently in the fixed state and prove nothing, which is the vacuous green
  // this whole file is supposed to avoid. The id must be REFUSED, full stop.
  const past = "9007199254740993";
  const { calls, error } = await readVia(past);
  assert.ok(
    error !== undefined,
    "the unsafe integer " + past + " was accepted (resolved to " +
      JSON.stringify(calls) + "); Number() cannot represent it, so the request " +
      "addressed a different document than the caller wrote"
  );
  assert.deepEqual(calls, [], "a refused id must never reach the API");

  // Control: the largest representable safe integer is still a valid id, so the
  // refusal above is about safety, not about rejecting large numbers.
  const { calls: safeCalls, error: safeError } = await readVia("9007199254740991");
  assert.equal(safeError, undefined, "MAX_SAFE_INTEGER must still be accepted");
  assert.deepEqual(safeCalls, [9007199254740991], "control: the safe bound is honoured");
});

test("no id segment can escape the documents namespace", async () => {
  // Traversal is not the realistic vector here (the value is interpolated into a
  // typed API call, not a filesystem path), but the guard is the same one: an
  // accepted id is a positive integer, so nothing path-shaped survives it.
  const traversal = ["../secrets", "..%2Fsecrets", "1/../../x", "%2e%2e%2f", "1?x=1", "1#frag"];
  for (const segment of traversal) {
    const { calls, error } = await readVia(segment);
    assert.ok(
      error !== undefined,
      "id segment " + JSON.stringify(segment) + " was accepted; an accepted id must " +
        "be a positive integer and nothing path-shaped may survive"
    );
    assert.deepEqual(calls, [], "no traversal attempt may reach the API");
  }
});
