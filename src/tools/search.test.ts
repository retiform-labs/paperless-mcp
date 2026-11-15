import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, describe, test } from "node:test";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import type { PaperlessAPI } from "../api/PaperlessAPI";
import type { SearchConfig } from "../search/config";
import {
  EMBEDDING_DIMENSIONS,
  SCHEMA_VERSION,
  SearchIndex,
  toFtsMatch,
} from "../search/store";
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
    storage_path: 1,
    tags: [1],
    created: "2026-10-01",
  },
  {
    id: 2,
    title: "Practice notes",
    content: "Submission requires authorization. " + "padding ".repeat(200),
    correspondent: 2,
    document_type: 1,
    storage_path: 2,
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

/**
 * THE THREE DEGRADATION STATES, one test each.
 *
 * These had NO tests at all, and the predicate that governed them asserted a
 * mechanism that had not happened: it fired on `both lists empty` and announced
 * "no lexical fallback exists", which is false whenever FTS5 is present and
 * simply had no match.
 */
describe("degradation — the three states are distinguishable", () => {
  test("(a) both retrievers ran and nothing matched — allowed, and says so", async () => {
    const embedder = new StubEmbedder(1);
    // EMPTY corpus: the stub embedder answers a fixed axis regardless of the
    // query, so a populated index would match "zzzznothing" by vector and this
    // test would be asserting about a non-empty result.
    const { context, index, documents } = makeContext({ embedder, documents: [] });
    try {
      const result = await runSemanticSearch(context, "zzzznothing", {
        limit: 5,
        window: 200,
        loadDocuments: async (ids) => resolveDocuments(ids, documents),
      });
      // The absence of a result is a real answer, and the payload must carry
      // enough to prove which retrievers produced it.
      assert.equal(result.passages.length, 0);
      assert.equal(result.semantic, "ok");
      assert.deepEqual(result.retrievers.sort(), ["lexical", "vector"]);
      assert.equal(result.degraded_reason, undefined);
      assert.equal(result.filtered_out, 0);
    } finally {
      index.close();
    }
  });

  test("(b) lexical ran and the embedder was down — results, unavailable, and a reason", async () => {
    const embedder = new StubEmbedder(1);
    embedder.failWith = new EmbeddingUnavailableError("ollama is down");
    const { context, index, documents } = makeContext({ embedder });
    try {
      const result = await runSemanticSearch(context, "prana", {
        limit: 5,
        window: 200,
        loadDocuments: async (ids) => resolveDocuments(ids, documents),
      });
      assert.equal(result.semantic, "unavailable");
      assert.deepEqual(result.retrievers, ["lexical"]);
      assert.ok(result.passages.length > 0, "the lexical half must still answer");
      assert.match(String(result.degraded_reason), /ollama is down/);
    } finally {
      index.close();
    }
  });

  test("(b') lexical available, embedder down, NOTHING lexically matched — still not an error", async () => {
    // This is the case the old guard got wrong. FTS5 is present, the embedder is
    // down, and the lexical search legitimately finds nothing. The old predicate
    // (both empty + unavailable) threw "no lexical fallback exists" — a false
    // statement about the build, delivered to a caller that had simply searched
    // for a word the corpus does not contain.
    const embedder = new StubEmbedder(1);
    embedder.failWith = new EmbeddingUnavailableError("ollama is down");
    // Empty corpus for the same reason as (a): the lexical miss has to be real,
    // not a vector hit masquerading as one.
    const { context, index, documents } = makeContext({ embedder, documents: [] });
    try {
      const result = await runSemanticSearch(context, "zzzznothing", {
        limit: 5,
        window: 200,
        loadDocuments: async (ids) => resolveDocuments(ids, documents),
      });
      assert.equal(result.passages.length, 0);
      assert.equal(result.semantic, "unavailable");
      // The retriever list is the proof lexical RAN, which is exactly what the
      // old message denied.
      assert.deepEqual(result.retrievers, ["lexical"]);
    } finally {
      index.close();
    }
  });

  test("(c) neither retriever is capable — refuses with an error, never a silent empty", async () => {
    // A store built with `fts5: false` and no embedder: genuinely nothing to
    // search with. The capability, not the emptiness, is what must decide.
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
    const context = new SearchContext(config, () => undefined);
    const seed = new SearchIndex(path, { fts5: false });
    seed.open();
    seed.writeMeta({
      schemaVersion: SCHEMA_VERSION,
      dimensions: EMBEDDING_DIMENSIONS,
      model: "stub-embed",
      prefixMode: "none",
    });
    seed.close();
    // The context must open the same no-FTS5 store, or the fixture tests a
    // different build from the one the capability flag describes.
    const store = new SearchIndex(path, { fts5: false });
    store.open();
    (context as unknown as { store: SearchIndex }).store = store;
    try {
      assert.equal(store.lexicalAvailable, false);
      await assert.rejects(
        () =>
          runSemanticSearch(context, "anything", {
            limit: 5,
            window: 200,
            loadDocuments: async () => new Map(),
          }),
        /neither the vector side nor a lexical fallback can run/
      );
    } finally {
      store.close();
    }
  });
});

describe("filtered_out — post-filter accounting for total_hits", () => {
  test("an all-filtered-out result explains itself instead of returning 12 and 0", async () => {
    // The exact shape the brief named: total_hits 12, passages [], nothing said.
    const embedder = new StubEmbedder(1);
    const { context, index, documents } = makeContext({ embedder });
    try {
      const result = await runSemanticSearch(context, "prana", {
        limit: 5,
        window: 200,
        loadDocuments: async (ids) => resolveDocuments(ids, documents),
        // Rejects every document.
        filters: () => false,
      });
      assert.ok(result.total_hits > 0, "the retrievers did find things");
      assert.equal(result.passages.length, 0);
      assert.equal(result.filtered_out, result.total_hits);
      assert.deepEqual(result.filtered_out_reasons, {
        document_unavailable: 0,
        metadata_filter: result.total_hits,
      });
    } finally {
      index.close();
    }
  });

  test("an unfetchable document is counted separately from a filter rejection", async () => {
    const embedder = new StubEmbedder(1);
    const { context, index } = makeContext({ embedder });
    try {
      const result = await runSemanticSearch(context, "prana", {
        limit: 5,
        window: 200,
        // Serves only document 1; document 2 is unfetchable.
        loadDocuments: async () => new Map([[1, DOCUMENTS[0]]]),
      });
      assert.equal(result.filtered_out, 1);
      assert.deepEqual(result.filtered_out_reasons, {
        document_unavailable: 1,
        metadata_filter: 0,
      });
    } finally {
      index.close();
    }
  });

  test("nothing filtered out means no reasons block at all", async () => {
    const embedder = new StubEmbedder(1);
    const { context, index, documents } = makeContext({ embedder });
    try {
      const result = await runSemanticSearch(context, "prana", {
        limit: 5,
        window: 200,
        loadDocuments: async (ids) => resolveDocuments(ids, documents),
      });
      assert.equal(result.filtered_out, 0);
      assert.equal(result.filtered_out_reasons, undefined);
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

  test("filters by storage_path, which used to be advertised and ignored", () => {
    // THE DEFECT: `storage_path` was in FILTER_SHAPE (so the model was told it
    // worked) and absent from this predicate (so it did nothing). A caller got
    // unfiltered results with no error, no warning and no way to notice, which
    // is worse than not offering the filter at all -- the schema is a promise.
    const filter = buildDocumentFilter({ storage_path: 2 })!;
    assert.ok(filter, "a storage_path argument must produce a predicate at all");
    assert.equal(filter(DOCUMENTS[0]), false, "document 1 is filed under path 1");
    assert.equal(filter(DOCUMENTS[1]), true, "document 2 is filed under path 2");
  });

  test("an unfiled document can never match a storage_path filter", () => {
    // storage_path === null is the common case on the live corpus, and null must
    // not coerce into matching 0 or into being skipped.
    const unfiled: VisibleDocument = { ...DOCUMENTS[0], id: 9, storage_path: null };
    const filter = buildDocumentFilter({ storage_path: 1 })!;
    assert.equal(filter(unfiled), false);
  });

  test("every advertised filter key is honoured", () => {
    // The schema-vs-implementation comparison, as an assertion rather than a
    // comment: each key in FILTER_SHAPE, set to a value that must EXCLUDE
    // document 1, has to actually exclude it. A key that returns the document
    // unchanged is the `storage_path` bug.
    const probes: Array<[string, Parameters<typeof buildDocumentFilter>[0]]> = [
      ["correspondent", { correspondent: 99 }],
      ["document_type", { document_type: 99 }],
      ["tag", { tag: 99 }],
      ["storage_path", { storage_path: 99 }],
      ["created__date__gte", { created__date__gte: "2999-01-01" }],
      ["created__date__lte", { created__date__lte: "1999-01-01" }],
    ];
    for (const [name, args] of probes) {
      const filter = buildDocumentFilter(args);
      assert.ok(filter, `${name} produced no predicate — it is advertised but unimplemented`);
      assert.equal(filter(DOCUMENTS[0]), false, `${name} did not exclude a non-matching document`);
    }
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

/**
 * THE ORDERING GUARANTEE FOR `force: true`, driven through the real tool.
 *
 * The old order was delete -> run -> throw on outage: a transient embedder blip
 * destroyed a working index and then reported the outage, leaving nothing on a
 * rebuild that takes ~58 minutes. The fix is probe-before-delete, and the only
 * honest way to test ORDERING is to observe which side effect happened before
 * the failure -- hence the two-way assertion below rather than a single
 * "it threw" check.
 */
describe("sync_embeddings force:true — probe before delete", () => {
  /** A server whose context holds a stub embedder and a seeded index. */
  async function forceServer(opts: { embedderFails: boolean }) {
    const indexDir = tempIndexPath();
    mkdirSync(indexDir, { recursive: true });
    const config: SearchConfig = {
      embeddingUrl: "http://127.0.0.1:11434",
      model: "stub-embed",
      prefixMode: "none",
      indexDir,
      maxChunks: 4000,
      syncBatchSize: 16,
      syncIntervalMs: 0,
    };
    const embedder = new StubEmbedder(1);
    if (opts.embedderFails) {
      embedder.failWith = new EmbeddingUnavailableError("ollama is down");
    }
    const context = new SearchContext(config, () => embedder);

    // Seed two documents so a delete is observable as a count change.
    const store = context.index();
    store.writeMeta({
      schemaVersion: SCHEMA_VERSION,
      dimensions: EMBEDDING_DIMENSIONS,
      model: "stub-embed",
      prefixMode: "none",
    });
    for (const doc of DOCUMENTS) {
      store.replaceDocument(
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

    // An API that fails LOUDLY on the first sync request, so reaching the sync
    // is distinguishable from refusing before it.
    const api = {
      async request(): Promise<never> {
        throw new Error("API_REACHED");
      },
    } as unknown as PaperlessAPI;

    const server = createMcpServer({
      baseUrl: "http://paperless.test",
      token: "test-token",
      version: "0.0.0-test",
      publicUrl: "http://paperless.test",
    });
    registerSearchTools(server, api, context);
    return { server, store, embedder };
  }

  async function callSync(server: ReturnType<typeof createMcpServer>) {
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    const client = new Client({ name: "force-test", version: "1.0.0" });
    await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
    try {
      return await client.callTool({ name: "sync_embeddings", arguments: { force: true } });
    } finally {
      await client.close();
    }
  }

  test("embedder down + force:true leaves the index UNCHANGED", async () => {
    const { server, store } = await forceServer({ embedderFails: true });
    try {
      const before = store.counts().documents;
      assert.equal(before, 2, "the fixture must start with a populated index");

      const result = (await callSync(server)) as { isError?: boolean; content: unknown[] };
      assert.equal(result.isError, true, "a dead embedder must refuse, not proceed");
      assert.match(JSON.stringify(result.content), /ollama is down/);

      // THE ASSERTION THAT MAKES THE ORDERING REAL: the index is untouched.
      assert.equal(
        store.counts().documents,
        before,
        "force:true destroyed the index before discovering the embedder was down"
      );
    } finally {
      store.close();
    }
  });

  test("embedder up + force:true DOES delete before the sync runs", async () => {
    // The other half of the ordering claim. With a working embedder the probe
    // passes, the delete executes, and the failure then comes from the API --
    // so the counts are zero and the error names the API, not the embedder.
    const { server, store } = await forceServer({ embedderFails: false });
    try {
      assert.equal(store.counts().documents, 2);
      const result = (await callSync(server)) as { isError?: boolean; content: unknown[] };
      assert.equal(result.isError, true);
      // The failure is the API's, proving the probe was passed and the delete ran.
      assert.match(JSON.stringify(result.content), /API_REACHED/);
      assert.equal(store.counts().documents, 0, "force:true must have deleted the index");
    } finally {
      store.close();
    }
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
