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
  runDocumentSearch,
  sanitizeFullTextQuery,
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

  test("keeps accented and non-Latin terms intact", () => {
    assert.deepEqual(extractSearchTerms("café"), ["café"]);
    assert.deepEqual(extractSearchTerms("kuji-in 九字"), ["kuji", "in", "九字"]);
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

describe("buildSearchPassage highlight anchoring (live failure shapes)", () => {
  test("anchors on the highlighted phrase when the first query term occurs earlier in the doc", () => {
    // Live shape 1: common word ("best") appears at doc head; the real hit
    // ("minimize ejaculation") is far later, where the server highlighted it.
    const early = `the best documents are best read in the morning. `;
    const far = "z".repeat(2000);
    const hitText = `people concerned about it often think the best thing to do is minimize ejaculation of energy.`;
    const doc = createDocument({ content: early + far + hitText + far });
    const passage = buildSearchPassage(
      doc,
      ["best", "thing", "for", "them", "to", "do", "is", "minimize", "ejaculation", "of", "energy"],
      800,
      { score: 18.5, highlights: "the <b>best</b> thing for them to do is <b>minimize</b> <b>ejaculation</b>" }
    );
    // The window must contain the highlighted phrase, not merely the first
    // verbatim occurrence of the first term.
    assert.ok(
      passage.passage.includes("minimize ejaculation"),
      `window anchored at ${passage.offset} missed the highlight`
    );
    assert.ok(passage.offset >= early.length, "window should anchor at the highlight, not the first 'best'");
  });

  test("anchors on the highlighted phrase when stemming produced a hit absent from the raw text", () => {
    // Live shape 2: whoosh matched "minimal"; the doc contains "minimize" only.
    const doc = createDocument({ content: `x`.repeat(1000) + `the best thing to do is minimize ejaculation today.` });
    const passage = buildSearchPassage(
      doc,
      ["minimal", "ejaculation"],
      400,
      { score: 12.0, highlights: "...the best thing to do is <b>minimize</b> <b>ejaculation</b> today..." }
    );
    assert.ok(passage.passage.includes("minimize ejaculation"));
    // Centered window may start before the phrase; the assertion is that the
    // window sits at the hit (deep in the doc), not at the head.
    assert.ok(passage.offset >= 800, `offset ${passage.offset}`);
  });

  test("decodes entities and tolerates markup variants in highlights", () => {
    const doc = createDocument({ content: `y`.repeat(900) + `fish and chips &amp; gravy nightly` });
    const passage = buildSearchPassage(doc, ["fish"], 300, {
      score: 3,
      highlights: "<mark>fish</mark> and chips &amp; gravy",
    });
    assert.ok(passage.passage.includes("fish and chips"), `offset ${passage.offset}`);
  });

  test("skips very short highlight matches that cannot anchor", () => {
    const doc = createDocument({ content: `head a b here ${"w".repeat(800)} a rare anchoring phrase tail` });
    const passage = buildSearchPassage(doc, ["a", "rare"], 300, {
      score: 5,
      highlights: "a <b>rare</b> <b>a</b> phrase",
    });
    assert.ok(passage.passage.includes("rare anchoring phrase"), `offset ${passage.offset}`);
  });

  test("falls back to verbatim term anchoring when the highlight text is absent", () => {
    const doc = createDocument({ content: `${"k".repeat(600)} findme soon ${"k".repeat(300)}` });
    const passage = buildSearchPassage(doc, ["findme"], 200, {
      score: 2,
      highlights: "<b>completely different text</b>",
    });
    assert.ok(passage.passage.includes("findme"));
  });

  test("falls back to the document head when there is no hit and nothing matches", () => {
    const doc = createDocument({ content: "z".repeat(2000) });
    const passage = buildSearchPassage(doc, ["missing"], 300, {
      score: 1,
      highlights: "<b>unfindable</b>",
    });
    assert.equal(passage.offset, 0);
  });
});

describe("sanitizeFullTextQuery", () => {
  test("strips apostrophes (django-watson 400 source)", () => {
    assert.equal(sanitizeFullTextQuery("God's favorite prayer"), "Gods favorite prayer");
  });

  test("drops all quotes when the count is unbalanced", () => {
    assert.equal(sanitizeFullTextQuery('say "hello world'), "say hello world");
    assert.equal(sanitizeFullTextQuery('odd "quote and "two" here'), "odd quote and two here");
  });

  test("leaves balanced quotes and clean queries untouched", () => {
    assert.equal(sanitizeFullTextQuery('"barber pole" worms'), '"barber pole" worms');
    assert.equal(sanitizeFullTextQuery("plain terms"), "plain terms");
  });

  test("keeps a searchable stem after apostrophe removal", () => {
    assert.equal(sanitizeFullTextQuery("it's"), "its");
  });

  test("returns null when nothing searchable remains", () => {
    assert.equal(sanitizeFullTextQuery('   '), null);
    assert.equal(sanitizeFullTextQuery('"'), null);
  });

  test("collapses whitespace left by removals", () => {
    assert.equal(sanitizeFullTextQuery("what's   new"), "whats new");
  });
});

describe("runDocumentSearch (400 resilience)", () => {
  function createFailingSearchApi(
    failOn: string,
    succeedWith: { count: number; all: number[]; results: Document[] },
    log: { queries: string[] }
  ): PaperlessAPI {
    return {
      getDocuments: async (query: string) => {
        log.queries.push(query);
        if (query.includes(failOn)) {
          const err = new Error(
            "Request failed with status code 400"
          ) as Error & { isAxiosError?: boolean; response?: { status: number; data: unknown } };
          err.isAxiosError = true;
          err.response = { status: 400, data: { detail: "Error listing search results, check logs for more detail." } };
          throw err;
        }
        return succeedWith as never;
      },
      getCorrespondents: async () => ({ results: [] }),
      getDocumentTypes: async () => ({ results: [] }),
      getTags: async () => ({ results: [] }),
      getCustomFields: async () => ({ results: [] }),
    } as unknown as PaperlessAPI;
  }

  test("retries a sanitized query when the raw query 400s", async () => {
    const log = { queries: [] as string[] };
    const hit = createDocument({ content: `answer here ${"f".repeat(400)}` });
    const api = createFailingSearchApi("%27", { count: 1, all: [hit.id], results: [hit] }, log);
    const out = await runDocumentSearch(api, "God's favorite prayer");
    assert.equal(out.querySanitized, true);
    assert.equal(out.response.results.length, 1);
    assert.ok(log.queries[0].includes("God%27s"), "first attempt must be the raw query");
    assert.ok(!log.queries[1].includes("'"), "retry must be sanitized");
  });

  test("returns query_sanitized false when the raw query succeeds", async () => {
    const log = { queries: [] as string[] };
    const hit = createDocument({ content: "plain content" });
    const api = createFailingSearchApi("NEVER-MATCHES", { count: 1, all: [hit.id], results: [hit] }, log);
    const out = await runDocumentSearch(api, "plain terms");
    assert.equal(out.querySanitized, false);
    assert.equal(log.queries.length, 1);
  });

  test("surfaces a structured error when the sanitized retry also fails", async () => {
    const log = { queries: [] as string[] };
    const api = createFailingSearchApi("q", { count: 0, all: [], results: [] }, log);
    await assert.rejects(
      () => runDocumentSearch(api, "totally broken q'tery"),
      /400/
    );
    assert.equal(log.queries.length, 2, "expected raw attempt + one sanitized retry");
  });

  test("does not retry when sanitization has nothing to fix", async () => {
    const log = { queries: [] as string[] };
    const api = createFailingSearchApi("q", { count: 0, all: [], results: [] }, log);
    await assert.rejects(() => runDocumentSearch(api, "clean query"), /400/);
    assert.equal(log.queries.length, 1, "no retry when the query was already clean");
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

  test("caps the fetch at max_passages unless page_size is explicit", async () => {
    const hit = createDocument({
      content: `${"f".repeat(300)} findme ${"f".repeat(300)}`,
      __search_hit__: { score: 1, highlights: "", note_highlights: "", rank: 1 },
    });
    const { api, getDocumentsQueries } = createSearchApi(hit);
    await withDocumentClient(api, async (client) => {
      const result = (await client.callTool({
        name: "search_passages",
        arguments: { query: "findme", max_passages: 2 },
      })) as CallToolResult;
      assert.ok(!result.isError, parseToolText(result)?.error);
      assert.match(getDocumentsQueries[0], /page_size=2/);
    });
  });

  test("an explicit page_size overrides the passage cap", async () => {
    const hit = createDocument({
      content: `${"g".repeat(300)} findme ${"g".repeat(300)}`,
      __search_hit__: { score: 1, highlights: "", note_highlights: "", rank: 1 },
    });
    const { api, getDocumentsQueries } = createSearchApi(hit);
    await withDocumentClient(api, async (client) => {
      const result = (await client.callTool({
        name: "search_passages",
        arguments: { query: "findme", page_size: 10 },
      })) as CallToolResult;
      assert.ok(!result.isError, parseToolText(result)?.error);
      assert.match(getDocumentsQueries[0], /page_size=10/);
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
