import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, describe, test } from "node:test";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import type { PaperlessAPI } from "../api/PaperlessAPI";
import type { SearchConfig } from "../search/config";
import { EMBEDDING_DIMENSIONS, SearchIndex, toFtsMatch } from "../search/store";
import { EmbeddingUnavailableError, type Embedder } from "../search/embedder";
import { createMcpServer } from "../server";
import {
  MAX_LIMIT,
  STRONG_MATCH,
  SearchContext,
  buildDocumentFilter,
  buildPassage,
  embeddingStatus,
  loadVisibleDocuments,
  queryTerms,
  registerSearchTools,
  runSemanticSearch,
  type VisibleDocument,
} from "./search";

let dir: string;

before(() => {
  dir = mkdtempSync(join(tmpdir(), "paperless-tools-"));
});

after(() => {
  rmSync(dir, { recursive: true, force: true });
});

let counter = 0;
function tempIndexPath(): string {
  counter += 1;
  return join(dir, `index-${counter}.db`);
}

function axisVector(axis: number): number[] {
  return Array.from({ length: EMBEDDING_DIMENSIONS }, (_, index) => (index === axis ? 1 : 0));
}

/** A stub embedder that either answers deterministically or fails like a stopped Ollama. */
class StubEmbedder implements Embedder {
  readonly model = "stub-embed";
  readonly prefixMode = "none" as const;
  readonly dimensions = EMBEDDING_DIMENSIONS;
  failWith: Error | undefined;
  queryCalls = 0;
  constructor(private readonly axis = 1) {}

  async embedDocuments(texts: string[]): Promise<number[][]> {
    if (this.failWith) throw this.failWith;
    return texts.map(() => axisVector(this.axis));
  }

  async embedQuery(): Promise<number[]> {
    this.queryCalls += 1;
    if (this.failWith) throw this.failWith;
    return axisVector(this.axis);
  }
}

const DOCUMENTS: VisibleDocument[] = [
  {
    id: 1,
    title: "Integrative medicine — prana and the winds",
    content:
      "The three humours. " + "padding ".repeat(60) + " prana disorders arise from the wind element. ".repeat(3),
    correspondent: 1,
    document_type: 1,
    tags: [1],
    created: "2026-10-01",
  },
  {
    id: 2,
    title: "Practice notes",
    content: "Submission requires authorization. " + "padding ".repeat(200),
    correspondent: 2,
    document_type: 1,
    tags: [2],
    created: "2026-11-01",
  },
];

/**
 * Build a context over a real temp SQLite index.
 *
 * The index is real rather than stubbed because the degradation contract is
 * about what the STORE can still serve when the embedder is gone. A fake store
 * would let a broken contract pass.
 */
function makeContext(opts: {
  embedder?: Embedder | null;
  documents?: VisibleDocument[];
} = {}): {
  context: SearchContext;
  index: SearchIndex;
  documents: Map<number, VisibleDocument>;
} {
  // A private directory per context, so `indexPath()` lands on the file this
  // fixture seeds. `SearchContext` always resolves `<indexDir>/index.db`, so a
  // random file name here would seed one file and query another -- which reads
  // as "the store is empty" rather than as a fixture bug.
  const indexDir = tempIndexPath();
  mkdirSync(indexDir, { recursive: true });
  const path = join(indexDir, "index.db");
  const config: SearchConfig = {
    embeddingUrl: "http://127.0.0.1:11434",
    model: "stub-embed",
    prefixMode: "none",
    indexDir,
    maxChunks: 4000,
    syncBatchSize: 16,
    syncIntervalMs: 0,
  };
  // The factory states "no embedder" rather than leaving it unset: unset would
  // be resolved into a REAL Ollama client, and Ollama is listening on this box.
  // A test that silently reaches the network is not a test.
  const context = new SearchContext(config, () => opts.embedder ?? undefined);
  // Seed the index directly: the sync path is covered by sync.test.ts, and this
  // suite is about what the tools do with an index, not how it got filled.
  const index = new SearchIndex(path);
  index.open();
  index.writeMeta({
    schemaVersion: 1,
    dimensions: EMBEDDING_DIMENSIONS,
    model: "stub-embed",
    prefixMode: "none",
  });
  const docs = opts.documents ?? DOCUMENTS;
  for (const doc of docs) {
    index.replaceDocument(
      {
        id: doc.id,
        modified: "2026-11-15T00:00:00Z",
        title: doc.title,
        chunkCount: 2,
        truncated: false,
        metadataAt: "2026-11-15T00:00:00Z",
      },
      [
        { kind: "metadata", text: `Title: ${doc.title}.`, embedding: axisVector(1) },
        { kind: "content", text: doc.content ?? "", embedding: axisVector(doc.id) },
      ]
    );
  }
  index.close();

  const documents = new Map(docs.map((doc) => [doc.id, doc]));
  // `null` explicitly: "there is deliberately no embedder", never "unset".
  if (opts.embedder === null) context.setEmbedder(null);
  return { context, index: context.index(), documents };
}

/**
 * Resolve ids to documents, dropping the ones this fixture does not hold.
 *
 * Typed explicitly: `new Map(pairs)` over an inferred array widens to a
 * `(number | doc)[][]` and TS rejects it, so the tuple type is stated once here
 * instead of being cast at seven call sites.
 */
function resolveDocuments(
  ids: number[],
  documents: Map<number, VisibleDocument>
): Map<number, VisibleDocument> {
  const resolved = new Map<number, VisibleDocument>();
  for (const id of ids) {
    const doc = documents.get(id);
    if (doc) resolved.set(id, doc);
  }
  return resolved;
}

/**
 * The store's own tokenisation, as a term list.
 *
 * `toFtsMatch` returns a quoted OR expression rather than terms, so this wraps
 * it: the point of the assertion is that the two tokenisers AGREE, and comparing
 * a term list against an FTS expression would not test that.
 */
function toFtsMatchTerms(query: string): string[] {
  const match = toFtsMatch(query);
  if (!match) return [];
  return match.split(" OR ").map((term) => term.replace(/^"|"$/g, ""));
}

/**
 * Every array-valued `type` in a JSON Schema, with its path.
 *
 * Same walk as schemaCompat.test.ts, repeated here because that file's server
 * is built without search config and therefore cannot see these tools.
 */
function arrayFormTypes(node: unknown, path = ""): string[] {
  if (Array.isArray(node)) {
    return node.flatMap((item, i) => arrayFormTypes(item, `${path}[${i}]`));
  }
  if (!node || typeof node !== "object") {
    return [];
  }
  return Object.entries(node as Record<string, unknown>).flatMap(([key, value]) =>
    key === "type" && Array.isArray(value)
      ? [`${path}.type = ${JSON.stringify(value)}`]
      : arrayFormTypes(value, `${path}.${key}`)
  );
}

describe("buildPassage", () => {
  test("centres the window on the first query term", () => {
    const content = "a".repeat(500) + " NEEDLE " + "b".repeat(500);
    const result = buildPassage({ id: 1, title: "t", content }, ["needle"], 200);
    assert.equal(result.content_length, content.length);
    assert.ok(result.passage.toLowerCase().includes("needle"));
    assert.ok(result.offset > 400 && result.offset < 520);
  });

  test("falls back to the head when no term occurs", () => {
    const result = buildPassage({ id: 1, title: "t", content: "abcdef" }, ["zzz"], 4);
    assert.equal(result.offset, 0);
    assert.equal(result.passage, "abcd");
  });

  test("never returns a negative offset near the start", () => {
    const result = buildPassage({ id: 1, title: "t", content: "needle here" }, ["needle"], 400);
    assert.equal(result.offset, 0);
  });

  test("treats a null content as empty rather than throwing", () => {
    const result = buildPassage({ id: 1, title: "t", content: null }, ["x"], 100);
    assert.equal(result.passage, "");
    assert.equal(result.content_length, 0);
  });
});

describe("queryTerms", () => {
  test("splits on punctuation and lowercases", () => {
    assert.deepEqual(queryTerms("prana, Wind-Element!"), ["prana", "wind", "element"]);
  });

  test("keeps unicode letters, so a non-Latin query is not emptied", () => {
    assert.deepEqual(queryTerms("prana རླུང་"), ["prana", "རླུང"]);
  });

  test("keeps combining marks, so a Tibetan syllable stays one term", () => {
    // U+0FB3 and U+0F74 are category Mn (Mark), not L. A tokeniser on
    // [^\p{L}\p{N}_] therefore splits རླུང into ར and ང -- two fragments,
    // neither of which is the word, and neither of which occurs in the corpus.
    // Measured on this build; the corpus is heavily Tibetan, so without \p{M}
    // those terms are unfindable lexically while looking like a plain no-match.
    const terms = queryTerms("རླུང");
    assert.deepEqual(terms, ["རླུང"]);
    assert.equal(terms.length, 1);
    // And the same tokeniser must agree with the store's, or a passage is
    // anchored on one fragment and matched on another.
    assert.deepEqual(terms, toFtsMatchTerms("རླུང"));
  });
});

describe("the degradation contract", () => {
  test("an embedder outage STILL returns lexical results, marked unavailable", async () => {
    const embedder = new StubEmbedder();
    embedder.failWith = new EmbeddingUnavailableError("connect ECONNREFUSED 127.0.0.1:11434");
    const { context, index, documents } = makeContext({ embedder });
    try {
      const result = await runSemanticSearch(context, "prana disorders", {
        limit: 5,
        window: 400,
        loadDocuments: async (ids) => resolveDocuments(ids, documents),
      });

      // 1. It did not throw.
      // 2. It said so.
      assert.equal(result.semantic, "unavailable");
      assert.match(String(result.degraded_reason), /ECONNREFUSED/);
      // 3. It still returned real lexical hits -- NOT a silent empty list.
      assert.ok(result.passages.length > 0, "a degraded call must not return an empty list");
      assert.equal(result.passages[0].id, 1);
      assert.equal(result.passages[0].matched, "words");
      // 4. Every passage carries the search_passages shape.
      const passage = result.passages[0];
      for (const key of ["id", "title", "score", "matched", "passage", "offset", "content_length"]) {
        assert.ok(key in passage, `passage is missing ${key}`);
      }
      assert.ok(passage.passage.toLowerCase().includes("prana"));
      // 5. strong_match was computed from the lexical side alone, and a lexical
      //    hit counts: the caller should not re-query for words it just got.
      assert.equal(result.strong_match, true);
    } finally {
      index.close();
    }
  });

  test("an unconfigured embedder degrades the same way, with its own reason", async () => {
    const { context, index, documents } = makeContext({ embedder: null });
    try {
      const result = await runSemanticSearch(context, "practice term", {
        limit: 5,
        window: 200,
        loadDocuments: async (ids) => resolveDocuments(ids, documents),
      });
      assert.equal(result.semantic, "unavailable");
      assert.match(String(result.degraded_reason), /not configured/);
      assert.ok(result.passages.length > 0);
    } finally {
      index.close();
    }
  });

  test("a healthy embedder reports ok and can fuse both sides", async () => {
    const embedder = new StubEmbedder(2);
    const { context, index, documents } = makeContext({ embedder });
    try {
      const result = await runSemanticSearch(context, "prana disorders", {
        limit: 5,
        window: 300,
        loadDocuments: async (ids) => resolveDocuments(ids, documents),
      });
      assert.equal(result.semantic, "ok");
      assert.equal(result.degraded_reason, undefined);
      assert.ok(result.passages.length > 0);
      // Doc 2 is the vector neighbour (axis 2) and doc 1 is the lexical match,
      // so the fused head must contain both and say how each was found.
      const byId = new Map(result.passages.map((passage) => [passage.id, passage.matched]));
      assert.equal(byId.get(1), "both");
      assert.equal(byId.get(2), "meaning");
    } finally {
      index.close();
    }
  });

  test("a query that matches nothing returns empty WITH semantic ok -- absence is a real answer", async () => {
    const embedder = new StubEmbedder(1);
    const { context, index, documents } = makeContext({ embedder, documents: [] });
    try {
      const result = await runSemanticSearch(context, "zzzznothing", {
        limit: 5,
        window: 200,
        loadDocuments: async (ids) => resolveDocuments(ids, documents),
      });
      // The contract forbids a SILENT empty list. An empty list that says
      // semantic:"ok" is not silent: it is the honest answer "nothing matched",
      // and it is distinguishable from "the embedder is down".
      assert.equal(result.passages.length, 0);
      assert.equal(result.semantic, "ok");
      assert.equal(result.strong_match, false);
    } finally {
      index.close();
    }
  });

  test("strong_match respects the similarity floor", async () => {
    const embedder = new StubEmbedder(1);
    const { context, index, documents } = makeContext({ embedder });
    try {
      const result = await runSemanticSearch(context, "practice term", {
        limit: 5,
        window: 200,
        loadDocuments: async (ids) => resolveDocuments(ids, documents),
      });
      // A lexical hit is a strong match by construction.
      assert.equal(result.strong_match, true);
      assert.ok(STRONG_MATCH > 0 && STRONG_MATCH < 1, "the floor is a similarity, not a distance");
    } finally {
      index.close();
    }
  });

  test("limit is clamped to MAX_LIMIT rather than trusted", async () => {
    const embedder = new StubEmbedder(1);
    const { context, index, documents } = makeContext({ embedder });
    try {
      const result = await runSemanticSearch(context, "prana", {
        limit: MAX_LIMIT + 500,
        window: 200,
        loadDocuments: async (ids) => resolveDocuments(ids, documents),
      });
      assert.ok(result.passages.length <= MAX_LIMIT);
    } finally {
      index.close();
    }
  });

  test("an unfetchable document does not fail the whole search", async () => {
    const embedder = new StubEmbedder(1);
    const { context, index } = makeContext({ embedder });
    try {
      const result = await runSemanticSearch(context, "prana", {
        limit: 5,
        window: 200,
        // The lexical store knows both documents; the API only serves one.
        loadDocuments: async () => new Map([[1, DOCUMENTS[0]]]),
      });
      assert.ok(result.passages.length > 0);
      assert.ok(result.passages.every((passage) => passage.id === 1));
    } finally {
      index.close();
    }
  });
});

describe("buildDocumentFilter", () => {
  test("no arguments means no predicate at all", () => {
    assert.equal(buildDocumentFilter({}), undefined);
  });

  test("filters by correspondent, type and tag", () => {
    const filter = buildDocumentFilter({ correspondent: 1 })!;
    assert.equal(filter(DOCUMENTS[0]), true);
    assert.equal(filter(DOCUMENTS[1]), false);

    const byTag = buildDocumentFilter({ tag: 2 })!;
    assert.equal(byTag(DOCUMENTS[0]), false);
    assert.equal(byTag(DOCUMENTS[1]), true);
  });

  test("filters by created date bounds inclusively", () => {
    const filter = buildDocumentFilter({
      created__date__gte: "2026-10-15",
      created__date__lte: "2026-11-15",
    })!;
    assert.equal(filter(DOCUMENTS[0]), false);
    assert.equal(filter(DOCUMENTS[1]), true);
  });
});

describe("loadVisibleDocuments", () => {
  test("dedupes ids and tolerates a failed fetch", async () => {
    const api = {
      async request(path: string) {
        if (path === "/documents/1/") return DOCUMENTS[0];
        throw new Error("gone");
      },
    } as unknown as PaperlessAPI;
    const documents = await loadVisibleDocuments(api, [1, 1, 2]);
    assert.equal(documents.size, 1);
    assert.ok(documents.has(1));
  });
});

describe("embeddingStatus", () => {
  test("reports the store's own counts and the meta it was built with", async () => {
    const { context, index } = makeContext({ embedder: new StubEmbedder() });
    const api = { async request() { return { count: 3 }; } } as unknown as PaperlessAPI;
    try {
      const status = await embeddingStatus(context, api);
      assert.equal(status.documents, 2);
      assert.ok(status.chunks >= 4);
      assert.equal(status.dimensions, EMBEDDING_DIMENSIONS);
      assert.equal(status.model, "stub-embed");
      assert.equal(status.prefix_mode, "none");
      assert.equal(status.last_sync, "2026-11-15T00:00:00Z");
      assert.equal(status.pending, 1);
      assert.equal(status.truncated_documents, 0);
      assert.ok(status.index_path.endsWith("index.db"));
    } finally {
      index.close();
    }
  });

  test("pending is null, NOT zero, when the count cannot be read", async () => {
    const { context, index } = makeContext({ embedder: new StubEmbedder() });
    const api = {
      async request() {
        throw new Error("paperless unreachable");
      },
    } as unknown as PaperlessAPI;
    try {
      const status = await embeddingStatus(context, api);
      // A zero here would claim "the index is current" on the strength of a
      // failed request.
      assert.equal(status.pending, null);
    } finally {
      index.close();
    }
  });
});

describe("tool registration — walked from the real tools/list", () => {
  function configuredServer() {
    const config: SearchConfig = {
      embeddingUrl: "http://127.0.0.1:11434",
      model: "stub-embed",
      prefixMode: "none",
      indexDir: dir,
      maxChunks: 4000,
      syncBatchSize: 16,
      syncIntervalMs: 0,
    };
    const server = createMcpServer({
      baseUrl: "http://paperless.test",
      token: "test-token",
      version: "0.0.0-test",
      publicUrl: "http://paperless.test",
    });
    // Register a second time with an explicit context: createMcpServer resolves
    // config from process.env, and this suite must not depend on the ambient
    // environment. Duplicate names would throw, so the env path must be absent.
    const api = {} as PaperlessAPI;
    registerSearchTools(server, api, new SearchContext(config));
    return server;
  }

  test("all three tools are advertised when configured", async () => {
    const server = configuredServer();
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    const client = new Client({ name: "search-tools-test", version: "1.0.0" });
    await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
    try {
      const { tools } = await client.listTools();
      const names = tools.map((tool) => tool.name);
      for (const name of ["semantic_search", "embedding_status", "sync_embeddings"]) {
        assert.ok(names.includes(name), `${name} is not advertised`);
      }
      // sync_embeddings must take an explicit force, not a positional boolean
      // nobody can see at a call site.
      const sync = tools.find((tool) => tool.name === "sync_embeddings")!;
      const properties = (sync.inputSchema as { properties?: Record<string, unknown> }).properties;
      assert.ok(properties && "force" in properties);
    } finally {
      await client.close();
    }
  });

  test("no search tool advertises an array-form `type` (#138)", async () => {
    // schemaCompat.test.ts already guards the pre-existing tools, but it builds
    // its server WITHOUT search config, so these three never register there and
    // would have been uncovered. An array-valued `type` makes strict MCP clients
    // drop the whole tool silently, and it arrives by collapsing a union of
    // unchecked primitives -- which a future filter argument could introduce.
    // The zod pin (~4.4.3) is what prevents it; this is the assertion that the
    // pin is doing its job for THIS surface.
    const server = configuredServer();
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    const client = new Client({ name: "search-tools-test", version: "1.0.0" });
    await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
    try {
      const { tools } = await client.listTools();
      const searchTools = tools.filter((tool) =>
        ["semantic_search", "embedding_status", "sync_embeddings"].includes(tool.name)
      );
      assert.equal(searchTools.length, 3, "all three search tools must be present to be checked");
      const offenders = searchTools
        .flatMap((tool) =>
          arrayFormTypes(tool.inputSchema).map((hit) => `${tool.name} ${hit}`)
        )
        .sort();
      assert.deepEqual(offenders, []);
    } finally {
      await client.close();
    }
  });

  test("no tool is advertised when search is not configured", async () => {
    // The gateway passes one environment to every child, so an unconfigured
    // install must not advertise tools that would fail on first call.
    const server = createMcpServer({
      baseUrl: "http://paperless.test",
      token: "test-token",
      version: "0.0.0-test",
      publicUrl: "http://paperless.test",
    });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    const client = new Client({ name: "search-tools-test", version: "1.0.0" });
    await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
    try {
      const { tools } = await client.listTools();
      const names = tools.map((tool) => tool.name);
      for (const name of ["semantic_search", "embedding_status", "sync_embeddings"]) {
        assert.ok(!names.includes(name), `${name} must not be advertised when unconfigured`);
      }
      // And the pre-existing tools are all still there.
      assert.ok(names.includes("search_passages"));
      assert.ok(names.includes("query_documents"));
    } finally {
      await client.close();
    }
  });
});
