import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, describe, test } from "node:test";
import type { PaperlessAPI } from "../api/PaperlessAPI";
import { EMBEDDING_DIMENSIONS, SearchIndex } from "./store";
import { EmbeddingUnavailableError, type Embedder } from "./embedder";
import {
  DOCUMENT_LIST_PATH,
  IndexSync,
  METADATA_REFRESH_MS,
  pathAfterApi,
  type SyncReport,
} from "./sync";

let dir: string;

before(() => {
  dir = mkdtempSync(join(tmpdir(), "paperless-sync-"));
});

after(() => {
  rmSync(dir, { recursive: true, force: true });
});

let counter = 0;
function tempIndexPath(): string {
  counter += 1;
  return join(dir, `index-${counter}.db`);
}

/** A deterministic unit vector, so two documents are only "similar" if chosen so. */
function axisVector(axis: number): number[] {
  return Array.from({ length: EMBEDDING_DIMENSIONS }, (_, index) => (index === axis ? 1 : 0));
}

/**
 * An embedder with no network, and a switch to make it fail the way a stopped
 * Ollama fails. The suite must never touch the real one -- a test that passes
 * because a local server happened to be running is not a test.
 */
class StubEmbedder implements Embedder {
  readonly model = "stub-embed";
  readonly prefixMode = "none" as const;
  readonly dimensions = EMBEDDING_DIMENSIONS;
  calls = 0;
  failWith: Error | undefined;

  async embedDocuments(texts: string[]): Promise<number[][]> {
    this.calls += 1;
    if (this.failWith) throw this.failWith;
    // Width matters: the store rejects anything but EMBEDDING_DIMENSIONS, so a
    // stub that returned short vectors would fail for the wrong reason.
    return texts.map((_, i) => axisVector((i + 1) % 64));
  }

  async embedQuery(query: string): Promise<number[]> {
    if (this.failWith) throw this.failWith;
    return axisVector(query.length % 64);
  }
}

interface FakeDoc {
  id: number;
  title: string;
  content: string;
  created: string;
  modified: string;
  correspondent: number | null;
  document_type: number | null;
  tags: number[];
}

/**
 * A fake PaperlessAPI covering exactly the routes the sync uses, with recorded
 * calls so a test can assert what was NOT requested.
 *
 * `listTruncation` is the knob the mid-run-mutation test needs: it makes the
 * list endpoint report a larger `count` than the rows it returns, which is
 * precisely what a delete during paging looks like from the client's side.
 */
class FakeApi {
  documents = new Map<number, FakeDoc>();
  listTruncation = 0;
  failDocumentIds = new Set<number>();
  requests: string[] = [];
  private listPageSize = 1000;

  add(doc: Partial<FakeDoc> & { id: number }): FakeDoc {
    const full: FakeDoc = {
      id: doc.id,
      title: doc.title ?? `Document ${doc.id}`,
      content: doc.content ?? `Body of document ${doc.id}.`,
      created: doc.created ?? "2026-11-01",
      modified: doc.modified ?? "2026-11-15T00:00:00Z",
      correspondent: doc.correspondent ?? null,
      document_type: doc.document_type ?? null,
      tags: doc.tags ?? [],
    };
    this.documents.set(full.id, full);
    return full;
  }

  remove(id: number): void {
    this.documents.delete(id);
  }

  async request<T>(path: string): Promise<T> {
    this.requests.push(path);

    if (path.startsWith(DOCUMENT_LIST_PATH)) {
      const all = [...this.documents.values()]
        .sort((a, b) => a.id - b.id)
        .map(({ id, modified }) => ({ id, modified }));
      return {
        count: all.length + this.listTruncation,
        next: null,
        results: all,
      } as unknown as T;
    }

    if (path.startsWith("/correspondents/")) {
      return { count: 1, next: null, results: [{ id: 1, name: "Ward's" }] } as unknown as T;
    }
    if (path.startsWith("/document_types/")) {
      return { count: 1, next: null, results: [{ id: 1, name: "Note" }] } as unknown as T;
    }
    if (path.startsWith("/tags/")) {
      return {
        count: 2,
        next: null,
        results: [
          { id: 1, name: "ai-processed" },
          { id: 2, name: "via-ocr" },
        ],
      } as unknown as T;
    }

    const detail = /^\/documents\/(\d+)\/$/.exec(path);
    if (detail) {
      const id = Number(detail[1]);
      if (this.failDocumentIds.has(id)) {
        throw new Error(`document ${id} exploded`);
      }
      const doc = this.documents.get(id);
      if (!doc) throw new Error(`no document ${id}`);
      return doc as unknown as T;
    }

    throw new Error(`FakeApi got an unexpected request: ${path}`);
  }
}

function makeSync(opts: {
  api: FakeApi;
  index: SearchIndex;
  embedder: Embedder;
  now?: () => number;
  maxChunks?: number;
}) {
  return new IndexSync({
    api: opts.api as unknown as PaperlessAPI,
    store: opts.index,
    embedder: opts.embedder,
    ...(opts.now ? { now: opts.now } : {}),
    ...(opts.maxChunks ? { maxChunks: opts.maxChunks } : {}),
  });
}

function openIndex(): SearchIndex {
  const index = new SearchIndex(tempIndexPath());
  index.open();
  index.writeMeta({
    schemaVersion: 1,
    dimensions: EMBEDDING_DIMENSIONS,
    model: "stub-embed",
    prefixMode: "none",
  });
  return index;
}

describe("pathAfterApi", () => {
  test("strips the base and keeps the query string", () => {
    assert.equal(
      pathAfterApi("http://paperless.test/api/documents/?page=2&page_size=1000"),
      "/documents/?page=2&page_size=1000"
    );
  });

  test("keeps a port in the base out of the result", () => {
    assert.equal(
      pathAfterApi("http://192.0.2.10:8000/api/tags/?page=3"),
      "/tags/?page=3"
    );
  });

  test("returns an unmangled path when the /api/ marker is absent", () => {
    // The reference slices at indexOf("/api/") + 4, which amputates four
    // characters when there is no marker. A wrong path that looks like a right
    // one is worse than a loud failure at the request.
    assert.equal(
      pathAfterApi("http://paperless.test/documents/?page=2"),
      "/documents/?page=2"
    );
  });
});

describe("IndexSync — the incremental diff", () => {
  test("adds new documents and reports them once", async () => {
    const api = new FakeApi();
    api.add({ id: 1 });
    api.add({ id: 2 });
    const index = openIndex();
    try {
      const report = await makeSync({ api, index, embedder: new StubEmbedder() }).run();
      assert.equal(report.added, 2);
      assert.equal(report.updated, 0);
      assert.equal(report.deleted, 0);
      assert.equal(report.failed, 0);
      assert.equal(index.counts().documents, 2);
      // A metadata chunk plus at least one content chunk per document.
      assert.ok(index.counts().chunks >= 4);
    } finally {
      index.close();
    }
  });

  test("re-syncs a document whose modified changed, and only that one", async () => {
    const api = new FakeApi();
    api.add({ id: 1 });
    api.add({ id: 2 });
    const index = openIndex();
    try {
      await makeSync({ api, index, embedder: new StubEmbedder() }).run();
      api.requests = [];
      api.add({ id: 1, modified: "2026-11-15T01:00:00Z" });

      const report = await makeSync({ api, index, embedder: new StubEmbedder() }).run();
      assert.equal(report.added, 0);
      assert.equal(report.updated, 1);
      const detailFetches = api.requests.filter((path) => /^\/documents\/\d+\/$/.test(path));
      assert.deepEqual(detailFetches, ["/documents/1/"]);
    } finally {
      index.close();
    }
  });

  test("does nothing when nothing changed", async () => {
    const api = new FakeApi();
    api.add({ id: 1 });
    const index = openIndex();
    try {
      await makeSync({ api, index, embedder: new StubEmbedder() }).run();
      api.requests = [];
      const report = await makeSync({ api, index, embedder: new StubEmbedder() }).run();
      assert.equal(report.added + report.updated + report.deleted, 0);
      const detailFetches = api.requests.filter((path) => /^\/documents\/\d+\/$/.test(path));
      assert.deepEqual(detailFetches, []);
    } finally {
      index.close();
    }
  });

  test("compares modified as an instant, not as a string", async () => {
    // The same moment written with a different offset must not reindex the
    // corpus: a string compare would see them as different on every run.
    const api = new FakeApi();
    api.add({ id: 1, modified: "2026-11-15T00:00:00Z" });
    const index = openIndex();
    try {
      await makeSync({ api, index, embedder: new StubEmbedder() }).run();
      api.add({ id: 1, modified: "2026-11-14T19:00:00-05:00" });
      const report = await makeSync({ api, index, embedder: new StubEmbedder() }).run();
      assert.equal(report.updated, 0);
    } finally {
      index.close();
    }
  });
});

describe("IndexSync — deletes", () => {
  test("deletes a document that is gone from Paperless", async () => {
    const api = new FakeApi();
    api.add({ id: 1 });
    api.add({ id: 2 });
    const index = openIndex();
    try {
      await makeSync({ api, index, embedder: new StubEmbedder() }).run();
      assert.equal(index.counts().documents, 2);
      api.remove(2);
      const report = await makeSync({ api, index, embedder: new StubEmbedder() }).run();
      assert.equal(report.deleted, 1);
      assert.equal(report.deletedSkipped, false);
      assert.equal(index.counts().documents, 1);
      assert.deepEqual([...index.indexed().keys()], [1]);
    } finally {
      index.close();
    }
  });

  test("SKIPS the delete step when the list rows disagree with its own count", async () => {
    // A delete during paging is indistinguishable from a document that was
    // never listed. Trusting the short list would delete a LIVE document from
    // the index, so the step must be skipped rather than inferred.
    const api = new FakeApi();
    api.add({ id: 1 });
    api.add({ id: 2 });
    const index = openIndex();
    try {
      await makeSync({ api, index, embedder: new StubEmbedder() }).run();
      assert.equal(index.counts().documents, 2);

      // The list now reports one more document than it returns: the classic
      // mid-run mutation signature.
      api.listTruncation = 1;
      const report = await makeSync({ api, index, embedder: new StubEmbedder() }).run();

      assert.equal(report.deletedSkipped, true);
      assert.equal(report.deleted, 0);
      // Both documents are still indexed. This is the assertion that matters.
      assert.equal(index.counts().documents, 2);
      assert.deepEqual([...index.indexed().keys()].sort(), [1, 2]);
    } finally {
      index.close();
    }
  });

  test("reports deletedSkipped false on a complete list", async () => {
    const api = new FakeApi();
    api.add({ id: 1 });
    const index = openIndex();
    try {
      const report = await makeSync({ api, index, embedder: new StubEmbedder() }).run();
      assert.equal(report.deletedSkipped, false);
    } finally {
      index.close();
    }
  });
});

describe("IndexSync — the embedding abort", () => {
  test("aborts the whole run on EmbeddingUnavailableError and writes nothing", async () => {
    const api = new FakeApi();
    api.add({ id: 1 });
    api.add({ id: 2 });
    const index = openIndex();
    const embedder = new StubEmbedder();
    embedder.failWith = new EmbeddingUnavailableError("ollama is down");
    try {
      await assert.rejects(
        () => makeSync({ api, index, embedder }).run(),
        (err: Error) => err instanceof EmbeddingUnavailableError
      );
      // Not a partial index: nothing was written at all, so nothing claims to be
      // complete. A skipped-document index would look finished and be wrong.
      assert.equal(index.counts().documents, 0);
      assert.equal(index.counts().chunks, 0);
    } finally {
      index.close();
    }
  });

  test("a non-embedding document failure is counted and the run continues", async () => {
    // The distinction the abort rule rests on: an embedding outage is global and
    // expected; one document failing is local and the rest are still real.
    const api = new FakeApi();
    api.add({ id: 1 });
    api.add({ id: 2 });
    api.failDocumentIds.add(1);
    const index = openIndex();
    try {
      const report = await makeSync({ api, index, embedder: new StubEmbedder() }).run();
      assert.equal(report.failed, 1);
      assert.equal(report.added, 1);
      assert.equal(index.counts().documents, 1);
      assert.deepEqual([...index.indexed().keys()], [2]);
    } finally {
      index.close();
    }
  });
});

describe("IndexSync — metadata refresh", () => {
  test("refreshes metadata of an unchanged document once it is a day old", async () => {
    const api = new FakeApi();
    api.add({ id: 1, correspondent: 1, tags: [1] });
    const index = openIndex();
    const start = Date.parse("2026-11-15T00:00:00Z");
    try {
      await makeSync({ api, index, embedder: new StubEmbedder(), now: () => start }).run();
      api.requests = [];

      // One hour later: nothing is stale, so no detail fetch happens.
      const fresh = await makeSync({
        api,
        index,
        embedder: new StubEmbedder(),
        now: () => start + 60 * 60 * 1000,
      }).run();
      assert.equal(fresh.metadataRefreshed, 0);

      // Past the refresh window: the metadata chunk is rebuilt even though
      // `modified` never moved. Without this a corrected sender stays in the
      // metadata chunk -- which is indexed both ways -- forever.
      const stale = await makeSync({
        api,
        index,
        embedder: new StubEmbedder(),
        now: () => start + METADATA_REFRESH_MS + 1000,
      }).run();
      assert.equal(stale.metadataRefreshed, 1);
      assert.equal(stale.updated, 0);
    } finally {
      index.close();
    }
  });

  test("the metadata chunk excludes ai- and via- prefixed tags", async () => {
    const api = new FakeApi();
    api.add({ id: 1, tags: [1, 2] });
    const index = openIndex();
    try {
      await makeSync({ api, index, embedder: new StubEmbedder() }).run();
      const chunks = index.chunksFor(1);
      const metadata = chunks.find((chunk) => chunk.kind === "metadata");
      assert.ok(metadata);
      // Both tags are scaffolding prefixes, so the tag list is empty -- the
      // labels emitted must be English, and the skipped prefixes must be gone.
      assert.ok(metadata.text.includes("Tags: ."));
      assert.ok(metadata.text.startsWith("Title: "));
    } finally {
      index.close();
    }
  });
});

describe("IndexSync — re-entrancy", () => {
  test("two concurrent run() calls share one pass", async () => {
    const api = new FakeApi();
    api.add({ id: 1 });
    const index = openIndex();
    try {
      const sync = makeSync({ api, index, embedder: new StubEmbedder() });
      const [first, second] = await Promise.all([sync.run(), sync.run()]);
      // Same object identity is the proof they were one pass, not two that
      // happened to agree.
      assert.equal(first, second);
      assert.equal(first.added, 1);
      assert.equal(index.counts().documents, 1);
    } finally {
      index.close();
    }
  });

  test("isRunning is false again after the pass settles", async () => {
    const api = new FakeApi();
    api.add({ id: 1 });
    const index = openIndex();
    try {
      const sync = makeSync({ api, index, embedder: new StubEmbedder() });
      assert.equal(sync.isRunning, false);
      const inFlight = sync.run();
      assert.equal(sync.isRunning, true);
      await inFlight;
      assert.equal(sync.isRunning, false);
    } finally {
      index.close();
    }
  });
});

describe("IndexSync — chunk truncation", () => {
  test("marks a document truncated when it exceeds the injected chunk cap", async () => {
    const api = new FakeApi();
    // 5000 chars at 2000/1800 stride is 3 chunks uncapped; cap at 2 to truncate.
    api.add({ id: 1, content: "x".repeat(5000) });
    const index = openIndex();
    try {
      await makeSync({ api, index, embedder: new StubEmbedder(), maxChunks: 2 }).run();
      assert.equal(index.truncatedCount(), 1);
    } finally {
      index.close();
    }
  });

  test("does not mark a short document truncated", async () => {
    const api = new FakeApi();
    api.add({ id: 1, content: "short body" });
    const index = openIndex();
    try {
      await makeSync({ api, index, embedder: new StubEmbedder() }).run();
      assert.equal(index.truncatedCount(), 0);
    } finally {
      index.close();
    }
  });
});
