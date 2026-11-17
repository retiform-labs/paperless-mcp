import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, describe, test } from "node:test";
import type { PaperlessAPI } from "../api/PaperlessAPI";
import { EMBEDDING_DIMENSIONS, SCHEMA_VERSION, SearchIndex } from "./store";
import { EmbeddingUnavailableError, type Embedder } from "./embedder";
import {
  DOCUMENT_LIST_PATH,
  IndexSync,
  METADATA_REFRESH_MS,
  WATERMARK_PATH,
  corpusChanged,
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
  /**
   * Set to make the stub report split events, the way a real embedder does when
   * it had to shorten an input. The sync must surface this in its report --
   * a run that split 400 chunks and one that split none otherwise produce
   * identical counts, and the number is the only evidence the fallback ran.
   */
  splitCount = 0;
  get lastSplits() {
    return Array.from({ length: this.splitCount }, () => ({
      reason: "text" as const,
      texts: 1,
      chars: 2000,
      depth: 0,
    }));
  }

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
  /** Make the watermark probe itself fail, the way a Paperless outage would. */
  failWatermark = false;
  /**
   * Override the `modified` the watermark probe reports, so a test can express
   * "the corpus moved" or "the same instant in a different offset" without
   * having to mutate every document it holds.
   */
  watermarkModifiedOverride: string | undefined;
  /** Override the count the probe reports, independently of the list's count. */
  watermarkCountOverride: number | undefined;
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

    // The watermark probe. Matched EXACTLY rather than by prefix: the document
    // list path shares its first 27 characters with this one, and a prefix test
    // would route the probe into the list branch and quietly make every
    // watermark assertion vacuous.
    if (path === WATERMARK_PATH) {
      if (this.failWatermark) throw new Error("paperless is down");
      const all = [...this.documents.values()];
      let newest: string | null = null;
      let newestAt = -Infinity;
      for (const doc of all) {
        const at = new Date(doc.modified).getTime();
        if (Number.isFinite(at) && at > newestAt) {
          newestAt = at;
          newest = doc.modified;
        }
      }
      return {
        // `listTruncation` is applied here too, and that is deliberate: it
        // models "the server's count is higher than the rows a page returns",
        // which is a fact about `count`. A watermark that disagreed with the
        // list about the corpus size would make the delete-skip test pass for
        // the wrong reason.
        count: this.watermarkCountOverride ?? all.length + this.listTruncation,
        next: null,
        results: this.watermarkModifiedOverride !== undefined
          ? [{ id: 0, modified: this.watermarkModifiedOverride }]
          : newest === null
            ? []
            : [{ id: 0, modified: newest }],
      } as unknown as T;
    }

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
      pathAfterApi("http://10.1.2.3:8000/api/tags/?page=3"),
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

describe("IndexSync — the index's identity is written BY THE RUN", () => {
  /**
   * These two tests exist because `writeMeta` and `assertSchemaCompatible` were
   * BOTH dead code: zero production call sites for either.
   *
   * The suite stayed green anyway, because every fixture called `writeMeta`
   * itself in `openIndex()`. That is the exact defect shape the brief named --
   * the tool asserted the FUNCTION while nothing asserted the WIRING, so
   * `embedding_status` could report `dimensions: 0, model: "", prefix_mode: ""`
   * on a real index and no test could see it.
   *
   * ⚠️ These tests therefore build the index WITHOUT calling writeMeta, and read
   * the result back. A fixture that writes the metadata itself cannot fail.
   */
  test("sync.run() writes the embedder's real model, prefix and dimensions", async () => {
    const api = new FakeApi();
    api.add({ id: 1 });
    // A deliberately non-default model and prefix, so a hard-coded value in the
    // implementation would show up rather than coincidentally matching.
    const embedder = new StubEmbedder();
    (embedder as unknown as { model: string }).model = "nomic-embed-text-v1.5";
    (embedder as unknown as { prefixMode: string }).prefixMode = "nomic";

    const index = new SearchIndex(tempIndexPath());
    index.open();
    // NOTHING writes metadata here. This is the whole point.
    assert.equal(index.meta(), undefined, "the index must start with no metadata");

    try {
      await makeSync({ api, index, embedder }).run();
      const meta = index.meta();
      assert.ok(meta, "sync.run() must have written the index metadata");
      assert.equal(meta.model, "nomic-embed-text-v1.5");
      assert.equal(meta.prefixMode, "nomic");
      assert.equal(meta.dimensions, EMBEDDING_DIMENSIONS);
      assert.equal(meta.schemaVersion, SCHEMA_VERSION);
    } finally {
      index.close();
    }
  });

  test("a run that fails to embed still records no false identity", async () => {
    // The metadata must not claim a model whose vectors were never written. A
    // write-first ordering means the identity is present even on a failed run,
    // and that is deliberate: the index exists, and what it says about itself
    // must be the truth about what this build would write.
    const api = new FakeApi();
    api.add({ id: 1 });
    const embedder = new StubEmbedder();
    embedder.failWith = new EmbeddingUnavailableError("ollama is down");

    const index = new SearchIndex(tempIndexPath());
    index.open();
    try {
      await assert.rejects(() => makeSync({ api, index, embedder }).run());
      const meta = index.meta();
      assert.ok(meta, "the identity is written before the work, so it survives a failure");
      assert.equal(meta.model, "stub-embed");
      // And no vectors were written under a claim the run could not honour.
      assert.equal(index.counts().documents, 0);
    } finally {
      index.close();
    }
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

describe("IndexSync — a failure names its document, and says whether the run stops", () => {
  /** Capture what the sync logs, without letting it reach the test output. */
  function captureLog(): { lines: string[]; restore: () => void } {
    const lines: string[] = [];
    const original = console.error;
    console.error = (...args: unknown[]) => {
      lines.push(args.map((arg) => String(arg)).join(" "));
    };
    return {
      lines,
      restore: () => {
        console.error = original;
      },
    };
  }

  test("an aborted run names the document by id AND title", async () => {
    // ⚠️ THE WHOLE POINT OF THIS TEST. The run this fixes failed with a bare
    // "the embedding server is unavailable, stopping the run" and no document
    // identity at all -- identifying the culprit took a separate investigation
    // and a hand-built probe. An id here would have been one call.
    const api = new FakeApi();
    api.add({ id: 1944, title: "HkMp5ArmorersManual" });
    api.add({ id: 1945 });
    const embedder = new StubEmbedder();
    embedder.failWith = new EmbeddingUnavailableError(
      "status 400: the input length exceeds the context length"
    );
    const index = openIndex();
    const { lines, restore } = captureLog();
    try {
      await assert.rejects(() => makeSync({ api, index, embedder }).run());
    } finally {
      restore();
      index.close();
    }
    const failureLine = lines.find((line) => line.includes("ABORTING"));
    assert.ok(failureLine, `no ABORTING line was logged (${lines.join(" | ")})`);
    assert.match(failureLine!, /1944/);
    assert.match(failureLine!, /HkMp5ArmorersManual/);
    // The decision is stated, not implied: a reader must not have to infer from
    // silence whether the next document was attempted.
    assert.doesNotMatch(failureLine!, /CONTINUING/);
  });

  test("a per-document failure names the document and says the run continues", async () => {
    // The inverse case. These two lines are the whole diagnosis surface: with
    // both in place, "why did document N disappear" is answerable from the log.
    //
    // ⚠️ THE FAILURE HERE IS IN THE DETAIL FETCH, WHICH IS THE HARD CASE: the
    // title has not been read yet, so the honest line carries the id and says
    // the title is unavailable. Substituting a placeholder name would look
    // tidier and would be a fabrication in a diagnostic line.
    const api = new FakeApi();
    api.add({ id: 7, title: "Exploding Document" });
    api.add({ id: 8 });
    api.failDocumentIds.add(7);
    const index = openIndex();
    const { lines, restore } = captureLog();
    let report: SyncReport;
    try {
      report = await makeSync({ api, index, embedder: new StubEmbedder() }).run();
    } finally {
      restore();
      index.close();
    }
    assert.equal(report.failed, 1);
    const failureLine = lines.find((line) => line.includes("CONTINUING"));
    assert.ok(failureLine, `no CONTINUING line was logged (${lines.join(" | ")})`);
    assert.match(failureLine!, /7 /);
    assert.match(failureLine!, /title unavailable/);
  });

  test("a per-document EMBEDDING failure names the title, because it was read", async () => {
    // The other half of the same surface. Here the detail fetch succeeded, so
    // the title exists and must appear -- that is the whole saving: no second
    // round trip to find out which of 2,466 documents this was.
    const api = new FakeApi();
    api.add({ id: 11, title: "Tibetan Grammar Scan" });
    api.add({ id: 12 });
    const embedder = new StubEmbedder();
    let call = 0;
    const realEmbed = embedder.embedDocuments.bind(embedder);
    embedder.embedDocuments = async (texts: string[]) => {
      call += 1;
      if (call === 1) throw new Error("vector store rejected the batch");
      return realEmbed(texts);
    };
    const index = openIndex();
    const { lines, restore } = captureLog();
    let report: SyncReport;
    try {
      report = await makeSync({ api, index, embedder }).run();
    } finally {
      restore();
      index.close();
    }
    assert.equal(report.failed, 1);
    assert.equal(report.added, 1, "the run continued to the second document");
    const failureLine = lines.find((line) => line.includes("CONTINUING"));
    assert.ok(failureLine, `no CONTINUING line was logged (${lines.join(" | ")})`);
    assert.match(failureLine!, /11 /);
    assert.match(failureLine!, /Tibetan Grammar Scan/);
  });

  test("the report counts splits, so the fallback is visible and not merely present", async () => {
    // Silence is the failure mode being designed against: without this counter
    // a corpus that needed 400 splits looks exactly like one that needed none,
    // and there is no way to confirm the new path is load-bearing.
    const api = new FakeApi();
    api.add({ id: 1 });
    api.add({ id: 2 });
    const embedder = new StubEmbedder();
    embedder.splitCount = 3;
    const index = openIndex();
    try {
      const report = await makeSync({ api, index, embedder }).run();
      assert.equal(report.splits, 6, "3 per document, two documents");
      assert.equal(report.added, 2);
    } finally {
      index.close();
    }
  });

  test("an embedder with no split counter is not a failure", async () => {
    // `lastSplits` is optional on the interface so that every existing test
    // double stays valid. A required field would have made this a breaking
    // change for fakes that have nothing to report.
    const api = new FakeApi();
    api.add({ id: 1 });
    const plain: Embedder = {
      model: "plain",
      prefixMode: "none",
      dimensions: EMBEDDING_DIMENSIONS,
      async embedDocuments(texts: string[]) {
        return texts.map((_, i) => axisVector((i + 1) % 64));
      },
      async embedQuery() {
        return axisVector(1);
      },
    };
    const index = openIndex();
    try {
      const report = await makeSync({ api, index, embedder: plain }).run();
      assert.equal(report.splits, 0);
      assert.equal(report.added, 1);
    } finally {
      index.close();
    }
  });
});

describe("IndexSync — the watermark", () => {
  /**
   * ⚠️ EVERY TEST HERE COUNTS REQUESTS OFF `api.requests`, NOT OFF THE REPORT.
   *
   * Asserting `skipped: true` would pass against an implementation that walked
   * the entire document list and then decided there was nothing to do. The whole
   * point of the watermark is that a no-op run costs ONE request instead of a
   * paged sweep, and only a request count can tell those apart. `requests` is
   * appended to at the top of `FakeApi.request`, so it is a census of every call
   * the sync made, in order.
   */
  function requestsDuring(api: FakeApi, fn: () => Promise<unknown>): Promise<void> {
    api.requests = [];
    return fn().then(() => undefined);
  }

  test("an unchanged corpus makes EXACTLY ONE request and reports skipped", async () => {
    const api = new FakeApi();
    api.add({ id: 1, modified: "2026-11-15T00:00:00Z" });
    api.add({ id: 2, modified: "2026-11-14T00:00:00Z" });
    const index = openIndex();
    const embedder = new StubEmbedder();
    try {
      const sync = makeSync({ api, index, embedder });
      await sync.run(); // first run: builds and records the watermark

      // Counted as a DELTA: the first run embedded both documents, so an
      // absolute count here would fail for a reason that has nothing to do with
      // the skip.
      const embedCallsBefore = embedder.calls;

      let report: SyncReport | undefined;
      await requestsDuring(api, async () => {
        report = await sync.run();
      });

      assert.equal(report?.skipped, true, "the second run must report skipped");
      assert.match(String(report?.skipReason), /matches the last run/);
      // THE ASSERTION THAT MATTERS. One request, and it is the probe.
      assert.deepEqual(api.requests, [WATERMARK_PATH]);
      assert.equal(
        embedder.calls - embedCallsBefore,
        0,
        "no document may be re-embedded on a skip"
      );
    } finally {
      index.close();
    }
  });

  test("a moved corpus runs the diff and does not report skipped", async () => {
    const api = new FakeApi();
    api.add({ id: 1, modified: "2026-11-15T00:00:00Z" });
    const index = openIndex();
    const embedder = new StubEmbedder();
    try {
      const sync = makeSync({ api, index, embedder });
      await sync.run();

      // A new document arrives. This changes BOTH count and max(modified).
      api.add({ id: 2, modified: "2026-11-15T06:00:00Z" });

      let report: SyncReport | undefined;
      await requestsDuring(api, async () => {
        report = await sync.run();
      });

      assert.equal(report?.skipped, false, "a moved corpus must not skip");
      assert.equal(report?.skipReason, undefined);
      assert.equal(report?.added, 1);
      // It walked the list, which is at least one more request than the probe.
      assert.ok(
        api.requests.filter((path) => path.startsWith(DOCUMENT_LIST_PATH)).length > 0,
        "the diff must actually page the document list"
      );
    } finally {
      index.close();
    }
  });

  test("a DROPPED count runs the delete path even though max(modified) cannot see it", async () => {
    // The trap this test exists for: a soft delete in Paperless goes through a
    // queryset `.update()`, so `auto_now` never fires and `modified` does not
    // move. `max(modified)` is structurally blind to it. Only `count` drops.
    //
    // The deleted document is deliberately the OLDEST, so removing it leaves
    // max(modified) identical -- isolating count as the sole signal. Without
    // that, this test would pass on a `max(modified)` check alone and prove
    // nothing about the count.
    const api = new FakeApi();
    api.add({ id: 1, modified: "2026-11-15T00:00:00Z" });
    api.add({ id: 2, modified: "2026-11-14T00:00:00Z" });
    const index = openIndex();
    const embedder = new StubEmbedder();
    try {
      const sync = makeSync({ api, index, embedder });
      await sync.run();
      assert.equal(index.counts().documents, 2);

      api.remove(2); // the oldest, so max(modified) is unchanged

      let report: SyncReport | undefined;
      await requestsDuring(api, async () => {
        report = await sync.run();
      });

      assert.equal(report?.skipped, false, "a dropped count must not skip");
      assert.equal(report?.deleted, 1, "the delete path must run");
      assert.equal(report?.deletedSkipped, false);
      assert.equal(index.counts().documents, 1, "the row must be gone from the index");
      assert.equal(index.indexed().has(2), false);
    } finally {
      index.close();
    }
  });

  test("force:true does the full run even when the watermark says nothing changed", async () => {
    // `force` means "the stored vectors are not trustworthy", which is a
    // statement about the INDEX. If the watermark short-circuited it, `force`
    // would become a no-op on the second call and silently lie about its effect.
    const api = new FakeApi();
    api.add({ id: 1, modified: "2026-11-15T00:00:00Z" });
    const index = openIndex();
    const embedder = new StubEmbedder();
    try {
      const sync = makeSync({ api, index, embedder });
      await sync.run();

      let report: SyncReport | undefined;
      await requestsDuring(api, async () => {
        report = await sync.run({ force: true });
      });

      assert.equal(report?.skipped, false, "force must never skip");
      assert.equal(report?.updated, 1, "force must re-embed the document");
      assert.ok(embedder.calls > 1, "the embedder must have been called again");
    } finally {
      index.close();
    }
  });

  test("the same instant in a different offset is NOT a change", async () => {
    // Paperless may re-serialise one moment with a different UTC offset. A
    // string compare would report a change on every run, defeating the watermark
    // while looking exactly like a working check.
    const api = new FakeApi();
    api.add({ id: 1, modified: "2026-11-15T00:00:00Z" });
    const index = openIndex();
    try {
      const sync = makeSync({ api, index, embedder: new StubEmbedder() });
      await sync.run();

      // The first run recorded max(modified) as "2026-11-15T00:00:00Z".
      // The probe now reports the SAME INSTANT written as
      // "2026-11-15T05:00:00+05:00".
      //
      // ⚠️ As STRINGS the probe value sorts GREATER ("05:00" > "00:00" at
      // offset 11), so a string comparison reports "changed" and the run
      // proceeds -- this test then fails. As INSTANTS they are the same moment,
      // so it skips. That disagreement is what makes this a real falsifier.
      //
      // (The earlier arrangement used -05:00 against a Z value, where the
      // string ALSO sorted lower and a string compare therefore said "not
      // changed" too. It passed under both implementations. Caught by the
      // mutation run, not by reading it.)
      api.watermarkModifiedOverride = "2026-11-15T05:00:00+05:00";
      // The list still reports the original, which is what the diff compares --
      // but the PROBE now says something that differs as a string and not as an
      // instant. If the comparison were a string compare, the run would proceed.
      let report: SyncReport | undefined;
      await requestsDuring(api, async () => {
        report = await sync.run();
      });

      assert.equal(
        report?.skipped,
        true,
        "the same instant in another offset must compare equal"
      );
      assert.deepEqual(api.requests, [WATERMARK_PATH]);
    } finally {
      index.close();
    }
  });

  test("an index with no recorded watermark runs the full diff", async () => {
    // The upgrade path. An index built before the watermark existed must not be
    // treated as "nothing changed" -- that would skip the first sync after an
    // upgrade and then keep skipping forever.
    const api = new FakeApi();
    api.add({ id: 1, modified: "2026-11-15T00:00:00Z" });
    const index = openIndex();
    try {
      assert.equal(index.watermark(), undefined, "the fixture must start with none");
      const report = await makeSync({ api, index, embedder: new StubEmbedder() }).run();
      assert.equal(report.skipped, false);
      assert.equal(report.added, 1);
      // And the run records one for next time.
      assert.equal(index.watermark()?.count, 1);
    } finally {
      index.close();
    }
  });

  test("a run whose list disagreed with its count records NO watermark", async () => {
    // ⚠️ THE GUARD THIS PINS IS `if (listComplete)` AROUND `writeWatermark`.
    //
    // A partial page set means documents were moving while the walk ran, so the
    // maximum it saw is a maximum over a SUBSET. Recording that as the
    // watermark would state "the corpus is unchanged through here" on the
    // strength of a list that was known to be short — and the NEXT run would
    // then compare equal and skip, permanently, on an index that is missing
    // whatever arrived mid-flight.
    //
    // Found by a surviving mutation: removing the guard broke nothing, which
    // means nothing was asserting it.
    const api = new FakeApi();
    api.add({ id: 1, modified: "2026-11-15T00:00:00Z" });
    const index = openIndex();
    try {
      // The server counts two documents but returns one: the mid-run-mutation
      // signature.
      api.listTruncation = 1;
      const report = await makeSync({
        api,
        index,
        embedder: new StubEmbedder(),
      }).run();

      assert.equal(report.deletedSkipped, true, "the list was short, so deletes skip");
      assert.equal(
        index.watermark(),
        undefined,
        "a known-incomplete walk must not record a watermark"
      );

      // And the consequence, stated directly: the next run does the real work
      // rather than skipping on a watermark it should never have trusted.
      api.listTruncation = 0;
      const after = await makeSync({ api, index, embedder: new StubEmbedder() }).run();
      assert.equal(after.skipped, false, "the next run must not skip");
      assert.equal(index.watermark()?.count, 1, "a complete run records one");
    } finally {
      index.close();
    }
  });

  test("a failed watermark probe falls back to the full diff rather than failing", async () => {
    // A cosmetic optimisation must not become a hard dependency. If the probe
    // query is refused, the list walk it already knows how to do still works.
    const api = new FakeApi();
    api.add({ id: 1, modified: "2026-11-15T00:00:00Z" });
    const index = openIndex();
    try {
      const sync = makeSync({ api, index, embedder: new StubEmbedder() });
      await sync.run();

      api.failWatermark = true;
      const report = await sync.run();
      assert.equal(report.skipped, false, "an unreadable probe must not skip");
    } finally {
      index.close();
    }
  });
});

describe("corpusChanged", () => {
  const watermark = { count: 3, modified: "2026-11-15T00:00:00Z", at: "2026-11-15T00:01:00Z" };

  test("equal count and instant is unchanged", () => {
    assert.equal(
      corpusChanged({ count: 3, modified: "2026-11-15T00:00:00Z" }, watermark),
      false
    );
  });

  test("a different count is a change, even with the same instant", () => {
    // The deletion case. This is the assertion that keeps a future "tidy-up"
    // from collapsing the two scalars into one max(modified) check.
    assert.equal(
      corpusChanged({ count: 2, modified: "2026-11-15T00:00:00Z" }, watermark),
      true
    );
  });

  test("a newer instant is a change", () => {
    assert.equal(
      corpusChanged({ count: 3, modified: "2026-11-15T00:00:01Z" }, watermark),
      true
    );
  });

  test("the same instant in another offset is not a change", () => {
    // ⚠️ THE OFFSETS ARE CHOSEN SO THE TWO IMPLEMENTATIONS DISAGREE.
    //
    // Live is "2026-11-15T00:00:00Z"; the watermark holds the SAME instant as
    // "2026-11-14T19:00:00-05:00". Compared as strings, the live value sorts
    // GREATER (11-15 beats 11-14), so a string compare reports "changed" and
    // this assertion fails. Compared as instants they are equal, so it passes.
    //
    // The first version of this test used the opposite arrangement, in which a
    // string compare ALSO said "not changed" — so it passed under both
    // implementations and was worthless as a falsifier. The mutation run is
    // what caught it.
    const sameInstant = "2026-11-15T00:00:00Z";
    const serialisedElsewhere = "2026-11-14T19:00:00-05:00";
    assert.notEqual(sameInstant, serialisedElsewhere, "the strings must differ");
    assert.equal(
      new Date(sameInstant).getTime(),
      new Date(serialisedElsewhere).getTime(),
      "the instants must be identical"
    );
    assert.ok(
      sameInstant > serialisedElsewhere,
      "and the string ordering must DISAGREE with the instant ordering"
    );

    assert.equal(
      corpusChanged({ count: 3, modified: sameInstant }, {
        ...watermark,
        modified: serialisedElsewhere,
      }),
      false
    );
  });

  test("a live corpus that reports no modification time differs from one that has one", () => {
    // Same count, but the live side says there is no newest document while the
    // watermark names one. That is a real disagreement and must read as changed:
    // silently treating a null as "nothing moved" would freeze the index on any
    // Paperless that briefly answered with an empty page.
    assert.equal(corpusChanged({ count: 3, modified: null }, watermark), true);
    // And the mirror: a watermark that recorded an empty corpus cannot be
    // matched by a live side that now has documents.
    assert.equal(
      corpusChanged({ count: 0, modified: null }, { ...watermark, count: 0 }),
      true
    );
  });
});

describe("the watermark's persistence", () => {
  test("round-trips through a close and reopen", async () => {
    const api = new FakeApi();
    api.add({ id: 1, modified: "2026-11-15T00:00:00Z" });
    api.add({ id: 2, modified: "2026-11-15T05:00:00Z" });
    const path = tempIndexPath();
    const index = new SearchIndex(path);
    index.open();
    try {
      await makeSync({
        api,
        index,
        embedder: new StubEmbedder(),
        now: () => Date.parse("2026-11-15T12:00:00Z"),
      }).run();
      const written = index.watermark();
      assert.deepEqual(written, {
        count: 2,
        modified: "2026-11-15T05:00:00Z",
        at: "2026-11-15T12:00:00.000Z",
      });
    } finally {
      index.close();
    }

    const reopened = new SearchIndex(path);
    reopened.open();
    try {
      assert.deepEqual(reopened.watermark()?.count, 2);
      assert.equal(reopened.watermark()?.modified, "2026-11-15T05:00:00Z");
    } finally {
      reopened.close();
    }
  });

  test("an index built WITHOUT the watermark keys opens and queries unchanged", async () => {
    // The upgrade path, stated as a property of the STORE rather than the run.
    // `meta()` reads four named keys and never enumerates, so extra rows are
    // structurally invisible to `assertSchemaCompatible` -- but that is a claim
    // about the code, and this is the test that holds it.
    const path = tempIndexPath();
    const index = new SearchIndex(path);
    index.open();
    index.writeMeta({
      schemaVersion: SCHEMA_VERSION,
      dimensions: EMBEDDING_DIMENSIONS,
      model: "stub-embed",
      prefixMode: "none",
    });
    // NO writeWatermark. This is exactly the shape an older build produced.
    assert.equal(index.watermark(), undefined);
    index.close();

    const reused = new SearchIndex(path, {
      expected: {
        dimensions: EMBEDDING_DIMENSIONS,
        model: "stub-embed",
        prefixMode: "none",
      },
    });
    // Must not throw: the schema guard has no opinion about a watermark.
    reused.open();
    try {
      const meta = reused.assertSchemaCompatible();
      assert.equal(meta.model, "stub-embed");
      assert.equal(reused.watermark(), undefined);
      assert.deepEqual(reused.counts().documents, 0);
    } finally {
      reused.close();
    }
  });

  test("writing a watermark leaves the schema version and identity untouched", async () => {
    const path = tempIndexPath();
    const index = new SearchIndex(path);
    index.open();
    try {
      index.writeMeta({
        schemaVersion: SCHEMA_VERSION,
        dimensions: EMBEDDING_DIMENSIONS,
        model: "stub-embed",
        prefixMode: "none",
      });
      const before = index.meta();
      index.writeWatermark({
        count: 7,
        modified: "2026-11-15T00:00:00Z",
        at: "2026-11-15T00:01:00Z",
      });
      // The whole reason the watermark is not a schema bump: adding it must not
      // change what the index says about how to read its vectors.
      assert.deepEqual(index.meta(), before);
    } finally {
      index.close();
    }
  });

  test("a half-written watermark reads as absent, not as a measurement", async () => {
    const path = tempIndexPath();
    const index = new SearchIndex(path);
    index.open();
    try {
      index.writeMeta({
        schemaVersion: SCHEMA_VERSION,
        dimensions: EMBEDDING_DIMENSIONS,
        model: "stub-embed",
        prefixMode: "none",
      });
      index.writeWatermark({
        count: 7,
        modified: "2026-11-15T00:00:00Z",
        at: "2026-11-15T00:01:00Z",
      });
      // Simulate rows written by different builds: drop one key.
      index.close();
      const raw = new SearchIndex(path);
      raw.open();
      const db = raw.open();
      db.prepare("DELETE FROM index_meta WHERE key = 'watermark_modified'").run();
      assert.equal(raw.watermark(), undefined);
      raw.close();
    } finally {
      index.close();
    }
  });
});
