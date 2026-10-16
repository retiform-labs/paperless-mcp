import assert from "node:assert/strict";
import { after, before, describe, test } from "node:test";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types";
import { PaperlessAPI } from "../api/PaperlessAPI";
import { Document } from "../api/types";
import {
  buildSearchPassage,
  DEFAULT_CONTENT_MAX_CHARS,
  DEFAULT_MAX_INLINE_CHARS,
  DEFAULT_MAX_PASSAGES,
  extractSearchTerms,
  getMaxInlineChars,
  registerDocumentTools,
  resolveBoundedRead,
  sliceDocumentContent,
} from "./documents";
import { createDocument } from "../test/mocks/paperlessApi";

function parseToolText(result: CallToolResult) {
  const item = result.content?.[0];
  if (!item || item.type !== "text") {
    throw new Error("Expected text tool response");
  }
  return JSON.parse(item.text);
}

class TestTransport {
  peer?: TestTransport;
  onmessage?: (message: unknown) => void;
  onclose?: () => void;
  async start(): Promise<void> {}
  async send(message: unknown): Promise<void> {
    this.peer?.onmessage?.(message);
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

interface RecordingApi {
  api: PaperlessAPI;
  getDocumentCalls: number[];
  getDocumentsQueries: string[];
  document: Document;
  documentsResponse: {
    count: number;
    next: null;
    previous: null;
    all: number[];
    results: Document[];
  };
}

function createSearchApi(
  document: Document,
  extraResults: Document[] = []
): RecordingApi {
  const results = [document, ...extraResults];
  const recording = {
    getDocumentCalls: [] as number[],
    getDocumentsQueries: [] as string[],
    document,
    documentsResponse: {
      count: results.length,
      next: null,
      previous: null,
      all: results.map((d) => d.id),
      results,
    },
  };
  const api = {
    getDocument: async (id: number) => {
      recording.getDocumentCalls.push(id);
      return document;
    },
    getDocuments: async (query: string) => {
      recording.getDocumentsQueries.push(query);
      return recording.documentsResponse;
    },
    getCorrespondents: async () => ({ results: [] }),
    getDocumentTypes: async () => ({ results: [] }),
    getTags: async () => ({ results: [] }),
    getCustomFields: async () => ({ results: [] }),
  } as unknown as PaperlessAPI;
  return { ...recording, api };
}

async function withDocumentClient(
  api: PaperlessAPI,
  run: (client: Client) => Promise<void>
) {
  const server = new McpServer({ name: "paperless-bounded-test", version: "1.0.0" });
  registerDocumentTools(server, api);
  const client = new Client({ name: "paperless-bounded-test-client", version: "1.0.0" });
  const { clientTransport, serverTransport } = createTransportPair();
  await server.connect(serverTransport);
  await client.connect(clientTransport);
  try {
    await run(client);
  } finally {
    await client.close();
    await server.close();
  }
}

describe("resolveBoundedRead", () => {
  test("defaults to the bounded head window", () => {
    const read = resolveBoundedRead(undefined, undefined, 100000);
    assert.deepEqual(read, { offset: 0, maxChars: DEFAULT_CONTENT_MAX_CHARS });
  });

  test("keeps caller-requested windows inside the budget", () => {
    const read = resolveBoundedRead(500, 1200, 100000);
    assert.deepEqual(read, { offset: 500, maxChars: 1200 });
  });

  test("rejects offsets beyond the end of the document", () => {
    assert.throws(
      () => resolveBoundedRead(50, 10, 40),
      /beyond the end of the document/
    );
  });

  test("allows an offset of zero on an empty document", () => {
    const read = resolveBoundedRead(0, undefined, 0);
    assert.deepEqual(read, { offset: 0, maxChars: DEFAULT_CONTENT_MAX_CHARS });
  });

  test("fails loudly when max_chars exceeds the inline budget", () => {
    assert.throws(
      () => resolveBoundedRead(0, DEFAULT_MAX_INLINE_CHARS + 1, 100000, DEFAULT_MAX_INLINE_CHARS),
      /exceeds the inline budget/
    );
  });

  test("names the env var and the download resource in the budget error", () => {
    try {
      resolveBoundedRead(0, DEFAULT_MAX_INLINE_CHARS + 1, 100000, DEFAULT_MAX_INLINE_CHARS);
      assert.fail("expected the budget error");
    } catch (error) {
      const message = (error as Error).message;
      assert.match(message, /PAPERLESS_MCP_MAX_INLINE_CHARS/);
      assert.match(message, /download_document/);
      assert.match(message, /offset/);
    }
  });
});

describe("getMaxInlineChars", () => {
  const KEY = "PAPERLESS_MCP_MAX_INLINE_CHARS";

  before(() => {
    delete process.env[KEY];
  });
  after(() => {
    delete process.env[KEY];
  });

  test("defaults to 50000 without configuration", () => {
    assert.equal(getMaxInlineChars(), DEFAULT_MAX_INLINE_CHARS);
  });

  test("reads the environment at call time", () => {
    process.env[KEY] = "12345";
    assert.equal(getMaxInlineChars(), 12345);
    delete process.env[KEY];
    assert.equal(getMaxInlineChars(), DEFAULT_MAX_INLINE_CHARS);
  });

  test("ignores invalid values and falls back to the default", () => {
    process.env[KEY] = "not-a-number";
    assert.equal(getMaxInlineChars(), DEFAULT_MAX_INLINE_CHARS);
    process.env[KEY] = "0";
    assert.equal(getMaxInlineChars(), DEFAULT_MAX_INLINE_CHARS);
    process.env[KEY] = "-5";
    assert.equal(getMaxInlineChars(), DEFAULT_MAX_INLINE_CHARS);
    delete process.env[KEY];
  });
});

describe("sliceDocumentContent", () => {
  test("returns the requested window with honest pagination metadata", () => {
    const doc = createDocument({ content: "x".repeat(300) });
    const sliced = sliceDocumentContent(doc, 100, 50);
    assert.equal(sliced.content.length, 50);
    assert.equal(sliced.offset, 100);
    assert.equal(sliced.content_length, 300);
    assert.equal(sliced.truncated, true);
    assert.equal(sliced.has_more, true);
    assert.equal(sliced.returned_chars, 50);
  });

  test("marks the final window as complete", () => {
    const doc = createDocument({ content: "y".repeat(100) });
    const sliced = sliceDocumentContent(doc, 50, DEFAULT_CONTENT_MAX_CHARS);
    assert.equal(sliced.content.length, 50);
    assert.equal(sliced.truncated, false);
    assert.equal(sliced.has_more, false);
  });

  test("handles documents with null content", () => {
    const doc = createDocument({ content: null });
    const sliced = sliceDocumentContent(doc, 0, DEFAULT_CONTENT_MAX_CHARS);
    assert.equal(sliced.content, "");
    assert.equal(sliced.content_length, 0);
    assert.equal(sliced.truncated, false);
  });
});

describe("extractSearchTerms", () => {
  test("splits a plain query into lowercase terms", () => {
    assert.deepEqual(extractSearchTerms("Kuji IN"), ["kuji", "in"]);
  });

  test("strips query operators and punctuation", () => {
    assert.deepEqual(extractSearchTerms('"barber pole" AND worms'), [
      "barber",
      "pole",
      "and",
      "worms",
    ]);
  });

  test("drops empty fragments", () => {
    assert.deepEqual(extractSearchTerms("  -- ?? "), []);
  });
});

describe("buildSearchPassage", () => {
  test("centers the window on the first term occurrence", () => {
    const filler = "a".repeat(1000);
    const doc = createDocument({ content: `${filler} NEEDLE ${filler}` });
    const passage = buildSearchPassage(doc, ["needle"], 200);
    assert.equal(passage.id, doc.id);
    assert.ok(passage.passage.includes("NEEDLE"));
    // The window should be centered near the term, not starting at it.
    assert.ok(passage.offset < 1000);
    assert.ok(passage.offset > 900);
    assert.equal(passage.content_length, doc.content!.length);
  });

  test("is case-insensitive when locating the term", () => {
    const filler = "b".repeat(500);
    const doc = createDocument({ content: `${filler} mixedCaseToken ${filler}` });
    const passage = buildSearchPassage(doc, ["mixedcasetoken"], 100);
    assert.ok(passage.passage.includes("mixedCaseToken"));
  });

  test("falls back to the document head when the term is absent verbatim", () => {
    const doc = createDocument({ content: "z".repeat(2000) });
    const passage = buildSearchPassage(doc, ["missing"], 300);
    assert.equal(passage.offset, 0);
    assert.equal(passage.passage.length, 300);
  });

  test("carries the search hit score and highlights when provided", () => {
    const doc = createDocument({ content: "scored content here" });
    const passage = buildSearchPassage(doc, ["scored"], 100, {
      score: 17.03,
      highlights: "<b>scored</b> content",
    });
    assert.equal(passage.score, 17.03);
    assert.equal(passage.highlights, "<b>scored</b> content");
  });

  test("omits score and highlights when no search hit is present", () => {
    const doc = createDocument({ content: "plain content" });
    const passage = buildSearchPassage(doc, ["plain"], 100);
    assert.ok(!("score" in passage));
    assert.ok(!("highlights" in passage));
  });
});

describe("get_document_content bounded reads", () => {
  test("returns a bounded window instead of the whole document", async () => {
    const longContent = "c".repeat(100000);
    const { api, getDocumentCalls } = createSearchApi(
      createDocument({ id: 9, content: longContent })
    );
    await withDocumentClient(api, async (client) => {
      const result = (await client.callTool({
        name: "get_document_content",
        arguments: { id: 9 },
      })) as CallToolResult;
      assert.ok(!result.isError, parseToolText(result)?.error);
      const parsed = parseToolText(result);
      assert.equal(parsed.content.length, DEFAULT_CONTENT_MAX_CHARS);
      assert.equal(parsed.content_length, 100000);
      assert.equal(parsed.truncated, true);
      assert.equal(parsed.has_more, true);
      assert.equal(parsed.offset, 0);
    });
    assert.deepEqual(getDocumentCalls, [9]);
  });

  test("honors offset and max_chars", async () => {
    const content = "0123456789".repeat(1000);
    const { api } = createSearchApi(createDocument({ id: 3, content }));
    await withDocumentClient(api, async (client) => {
      const result = (await client.callTool({
        name: "get_document_content",
        arguments: { id: 3, offset: 100, max_chars: 10 },
      })) as CallToolResult;
      const parsed = parseToolText(result);
      assert.equal(parsed.content, content.slice(100, 110));
      assert.equal(parsed.offset, 100);
      assert.equal(parsed.returned_chars, 10);
    });
  });

  test("rejects a max_chars above the budget with a size-aware error", async () => {
    const { api } = createSearchApi(
      createDocument({ id: 4, content: "d".repeat(100000) })
    );
    await withDocumentClient(api, async (client) => {
      const result = (await client.callTool({
        name: "get_document_content",
        arguments: { id: 4, max_chars: DEFAULT_MAX_INLINE_CHARS + 1 },
      })) as CallToolResult;
      assert.ok(result.isError, "expected the budget error");
      const text = result.content?.[0];
      assert.ok(text && text.type === "text");
      const message = (text as { text: string }).text;
      assert.match(message, /exceeds the inline budget/);
      assert.match(message, /PAPERLESS_MCP_MAX_INLINE_CHARS/);
    });
  });
});

describe("search_passages", () => {
  test("returns bounded passages per hit, never full content", async () => {
    const hit = createDocument({
      id: 1,
      title: "Hayes",
      content: `${"q".repeat(2000)} kuji training ${"q".repeat(2000)}`,
      __search_hit__: { score: 9.5, highlights: "<b>kuji</b>", note_highlights: "", rank: 1 },
    });
    const other = createDocument({
      id: 2,
      title: "Other",
      content: `${"w".repeat(2000)} kuji reference ${"w".repeat(2000)}`,
      __search_hit__: { score: 4.5, highlights: "<b>kuji</b> ref", note_highlights: "", rank: 2 },
    });
    const { api, getDocumentsQueries } = createSearchApi(hit, [other]);
    await withDocumentClient(api, async (client) => {
      const result = (await client.callTool({
        name: "search_passages",
        arguments: { query: "kuji" },
      })) as CallToolResult;
      assert.ok(!result.isError, parseToolText(result)?.error);
      const parsed = parseToolText(result);
      assert.equal(parsed.query, "kuji");
      assert.equal(parsed.total_hits, 2);
      assert.equal(parsed.passages.length, 2);
      for (const passage of parsed.passages) {
        assert.ok(passage.passage.length <= 800);
        assert.ok(typeof passage.content_length === "number");
        assert.ok(typeof passage.offset === "number");
      }
      const first = parsed.passages[0];
      assert.ok(first.passage.includes("kuji"));
      assert.equal(first.score, 9.5);
      assert.equal(first.highlights, "<b>kuji</b>");
    });
    assert.equal(getDocumentsQueries.length, 1);
    assert.match(getDocumentsQueries[0], /query=kuji/);
  });

  test("caps passages at max_passages and the total hit count", async () => {
    const hit = createDocument({
      content: `${"e".repeat(500)} target text ${"e".repeat(500)}`,
      __search_hit__: { score: 1, highlights: "", note_highlights: "", rank: 1 },
    });
    const extras = [2, 3, 4, 5, 6, 7, 8].map((id) =>
      createDocument({
        id,
        content: `filler ${id}`,
        __search_hit__: { score: id, highlights: "", note_highlights: "", rank: id },
      })
    );
    const { api } = createSearchApi(hit, extras);
    await withDocumentClient(api, async (client) => {
      const result = (await client.callTool({
        name: "search_passages",
        arguments: { query: "target", max_passages: 3 },
      })) as CallToolResult;
      const parsed = parseToolText(result);
      assert.equal(parsed.total_hits, 8);
      assert.equal(parsed.passages.length, 3);
    });
  });

  test("rejects a window outside the supported range", async () => {
    const { api } = createSearchApi(createDocument({ content: "x" }));
    await withDocumentClient(api, async (client) => {
      const result = (await client.callTool({
        name: "search_passages",
        arguments: { query: "x", window: 50 },
      })) as CallToolResult;
      assert.ok(result.isError, "expected the window validation error");
    });
  });
});

describe("enhanced documents expose content_length", () => {
  test("list/query results carry content_length without content", async () => {
    const doc = createDocument({ id: 11, content: "m".repeat(1234) });
    const { api } = createSearchApi(doc);
    await withDocumentClient(api, async (client) => {
      const result = (await client.callTool({
        name: "query_documents",
        arguments: { query: "anything" },
      })) as CallToolResult;
      assert.ok(!result.isError, parseToolText(result)?.error);
      const parsed = parseToolText(result);
      assert.ok(!("content" in parsed.results[0]));
      assert.equal(parsed.results[0].content_length, 1234);
    });
  });
});
