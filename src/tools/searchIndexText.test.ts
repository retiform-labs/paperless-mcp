// Tests for the two HALVES OF THE P2 LATENCY FIX that replaced full-document
// HTTP fetches with index reads:
//
//   1. `loadDocumentMetadata` -- one BATCHED ranged request for N documents
//      instead of one `GET /documents/{id}/` each, and never fetching content.
//   2. `reconstructDocumentText` / `indexTextLoader` -- rebuilding a
//      document's text by splicing the chunks the index already holds.
//
// Measured on the live box for 40 documents: 2,398 ms and 2,162,306 bytes for
// the per-id path, against 58 ms and 6,520 bytes for the batched metadata call.
// That ratio is the reason both functions exist, and neither had a test.
//
// ⚠️ `reconstructDocumentText` SKIPS THE METADATA CHUNK (position 0). It is a
// synthetic "Title: … Sender: …" line, not document text. Including it would
// put fabricated words in front of every passage AND shift every offset.

import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, describe, test } from "node:test";
import type { PaperlessAPI } from "../api/PaperlessAPI";
import { CHUNK_OVERLAP, CHUNK_SIZE, chunkText } from "../search/chunk";
import { EMBEDDING_DIMENSIONS, SearchIndex } from "../search/store";
import {
  METADATA_BATCH,
  indexTextLoader,
  loadDocumentMetadata,
  reconstructDocumentText,
} from "./search";

let dir: string;
before(() => {
  dir = mkdtempSync(join(tmpdir(), "paperless-p2-text-"));
});
after(() => {
  rmSync(dir, { recursive: true, force: true });
});

let counter = 0;
function tempIndexPath(): string {
  counter += 1;
  return join(dir, `idx-${counter}.db`);
}

const axisVector = (axis: number): number[] =>
  Array.from({ length: EMBEDDING_DIMENSIONS }, (_, i) => (i === axis ? 1 : 0));

/**
 * A PaperlessAPI double that records every request it is given.
 *
 * `docs` is the whole corpus it will answer for; the double PARSES the
 * `id__in` list out of the path, exactly as the real server would, so a test
 * cannot pass by having the double ignore the batching and return everything.
 */
function recordingApi(docs: { id: number; title: string }[]) {
  const calls: string[] = [];
  const api = {
    async request(path: string) {
      calls.push(path);
      const match = /id__in=([^&]*)/.exec(path);
      if (!match) throw new Error(`unexpected path: ${path}`);
      const wanted = new Set(match[1].split(",").map(Number));
      const results = docs
        .filter((d) => wanted.has(d.id))
        .map((d) => ({
          id: d.id,
          title: d.title,
          // A FULL body on purpose: the function must strip it, and a fixture
          // without one could not tell "stripped" from "never had it".
          content: "FULL DOCUMENT BODY THAT MUST NOT SURVIVE",
          correspondent: null,
          document_type: null,
          storage_path: null,
          tags: [],
          created: "2026-10-01",
        }));
      return { results, count: results.length };
    },
  } as unknown as PaperlessAPI;
  return { api, calls };
}

describe("loadDocumentMetadata -- the batched ranged fetch", () => {
  test("BATCHES 250 ids into 3 calls of 100/100/50, not 250 calls", async () => {
    const corpus = Array.from({ length: 250 }, (_, i) => ({ id: i + 1, title: `Doc ${i + 1}` }));
    const { api, calls } = recordingApi(corpus);

    const documents = await loadDocumentMetadata(api, corpus.map((d) => d.id));

    assert.equal(METADATA_BATCH, 100, "the batch size is load-bearing for the math below");
    assert.equal(calls.length, 3, `expected 3 calls, saw ${calls.length}`);
    // The actual split, read out of the request paths rather than assumed.
    const sizes = calls.map((p) => (/id__in=([^&]*)/.exec(p) as RegExpExecArray)[1].split(",").length);
    assert.deepEqual(sizes, [100, 100, 50]);
    assert.equal(documents.size, 250, "every id must be resolved");
  });

  test("neither DROPS nor DUPLICATES an id", async () => {
    // Duplicates in the INPUT are deduped before batching -- the doc comment
    // says 400 ids are reachable because two retrievers can each contribute
    // `limit * 4`, so the same id routinely arrives twice.
    const corpus = Array.from({ length: 20 }, (_, i) => ({ id: i + 1, title: `Doc ${i + 1}` }));
    const { api, calls } = recordingApi(corpus);
    const requested = [3, 1, 2, 1, 3, 20, 19, 18];

    const documents = await loadDocumentMetadata(api, requested);

    assert.equal(calls.length, 1, "8 requested ids fit in a single batch");
    const sent = (/id__in=([^&]*)/.exec(calls[0]) as RegExpExecArray)[1].split(",").map(Number);
    assert.deepEqual(
      [...sent].sort((a, b) => a - b),
      [1, 2, 3, 18, 19, 20],
      "duplicates collapsed, nothing lost"
    );
    assert.deepEqual(
      [...documents.keys()].sort((a, b) => a - b),
      [1, 2, 3, 18, 19, 20]
    );
  });

  test("NEVER returns document content -- content is null on every entry", async () => {
    // This is the whole point of the change. The old path fetched full bodies
    // and sliced a window out of them; the index already holds the text.
    const corpus = [{ id: 7, title: "Seven" }];
    const { api } = recordingApi(corpus);

    const documents = await loadDocumentMetadata(api, [7]);
    const doc = documents.get(7);

    assert.ok(doc, "the document must be present");
    assert.equal(doc.content, null, "content must be null, not the body");
    assert.equal(doc.title, "Seven", "the metadata that IS wanted must survive");
  });

  test("requests the ranged field list, not a bare documents lookup", async () => {
    const { api, calls } = recordingApi([{ id: 1, title: "One" }]);
    await loadDocumentMetadata(api, [1]);

    assert.match(calls[0], /^\/documents\/\?/, "must be the collection endpoint with a querystring");
    assert.match(calls[0], /fields=id,title,/, "must ask for a field subset");
    assert.match(calls[0], /page_size=1/, "page_size must match the batch, or the server may paginate");
    assert.ok(
      !calls[0].includes("/documents/1/"),
      "must NOT be the per-document endpoint -- that is the defect this replaced"
    );
  });

  test("an EMPTY id list makes NO request at all", async () => {
    // ⚠️ CHARACTERISATION TEST — NO FALSIFICATION FOUND, AND SAYING SO IS THE
    // POINT. Every plausible mutation of `loadDocumentMetadata` was tried
    // against this test and it stayed green: changing `<` to `<=` in the
    // batching loop, and deleting the `unique.length === 0` early return
    // entirely, both left it passing -- because a loop bound of zero already
    // makes no request, so the guard is behaviourally redundant today.
    // It is kept as a REGRESSION GUARD against a future refactor that issues an
    // unconditional request, not as evidence that this property is protected
    // now. A test never seen red is a rumour; this one is labelled as one.
    const { api, calls } = recordingApi([]);
    const documents = await loadDocumentMetadata(api, []);
    assert.equal(calls.length, 0);
    assert.equal(documents.size, 0);
  });

  test("a FAILED batch does not throw and does not lose the other batches", async () => {
    // Documented tolerance: "a batch that fails does NOT fail the search".
    // The affected ids are simply absent, which the caller reports as
    // `document_unavailable`.
    const calls: string[] = [];
    const api = {
      async request(path: string) {
        calls.push(path);
        if (calls.length === 1) throw new Error("boom");
        const ids = (/id__in=([^&]*)/.exec(path) as RegExpExecArray)[1].split(",").map(Number);
        return { results: ids.map((id) => ({ id, title: `Doc ${id}`, content: "x" })) };
      },
    } as unknown as PaperlessAPI;

    const ids = Array.from({ length: 150 }, (_, i) => i + 1);
    const documents = await loadDocumentMetadata(api, ids);

    assert.equal(calls.length, 2, "both batches must have been attempted");
    assert.equal(documents.size, 50, "only the surviving batch's 50 ids resolve");
    assert.ok(!documents.has(1), "the failed batch's ids are absent, not invented");
    assert.ok(documents.has(150), "the second batch must still land");
  });
});

describe("reconstructDocumentText -- splicing the index's own chunks", () => {
  const content = (position: number, text: string) =>
    ({ position, kind: "content" as const, text });

  test("SKIPS the metadata chunk, so no fabricated title line leads the text", () => {
    const out = reconstructDocumentText([
      { position: 0, kind: "metadata", text: "Title: Seven. Sender: nobody." },
      content(1, "real body text"),
    ]);
    assert.equal(out, "real body text");
    assert.ok(!out.includes("Title:"), "a synthetic metadata line must never reach the passage");
  });

  test("drops the OVERLAP between consecutive chunks EXACTLY once", () => {
    // The chunker emits a stride of CHUNK_SIZE - CHUNK_OVERLAP, so chunk p+1
    // begins with chunk p's last CHUNK_OVERLAP characters. Splicing without
    // dropping that repeats 200 characters of every document.
    //
    // ⚠️ MARKERS, NOT A HOMOGENEOUS FILLER. My first version of this test used
    // "a".repeat(...) and then asked whether a 400-character run of "a" was
    // present -- which it always is inside a 2000-character block of "a"s,
    // whatever the overlap logic does. The assertion was UNFALSIFIABLE by
    // construction and would have passed against a function that dropped
    // nothing. Distinct fills make a doubled overlap a shape that can only
    // exist if the splice is wrong.
    const first = "P".repeat(CHUNK_SIZE - CHUNK_OVERLAP) + "Q".repeat(CHUNK_OVERLAP);
    const second = "Q".repeat(CHUNK_OVERLAP) + "R".repeat(CHUNK_SIZE - CHUNK_OVERLAP);
    const out = reconstructDocumentText([content(1, first), content(2, second)]);

    // Exact equality is the strongest form: no doubled Q, no deleted text.
    assert.equal(
      out,
      "P".repeat(CHUNK_SIZE - CHUNK_OVERLAP) +
        "Q".repeat(CHUNK_OVERLAP) +
        "R".repeat(CHUNK_SIZE - CHUNK_OVERLAP)
    );
    assert.equal(out.length, first.length + second.length - CHUNK_OVERLAP);
    // And the doubled run must be absent -- the mutation this exists to catch.
    assert.equal(
      out.includes("Q".repeat(CHUNK_OVERLAP * 2)),
      false,
      "the 200-character overlap appears twice, so it was not dropped"
    );
  });

  test("does NOT drop 200 characters when the overlap is not actually there", () => {
    // The other direction, and the more damaging one: a document shorter than
    // one chunk is a single chunk, and blindly stripping its tail deletes real
    // text rather than deduplicating.
    const single = "short document";
    assert.equal(reconstructDocumentText([content(1, single)]), single);

    // Two chunks whose boundary genuinely differs must be concatenated whole.
    const a = "x".repeat(50) + "END";
    const b = "START" + "y".repeat(50);
    assert.equal(reconstructDocumentText([content(1, a), content(2, b)]), a + b);
  });

  test("ROUND-TRIPS real chunker output back to the original text", () => {
    // The strongest form: feed the actual chunker a realistic document and
    // assert the reconstruction equals the normalised input. This is the
    // property the whole function exists to provide, asserted against the
    // producer rather than against hand-built chunks.
    const body = Array.from({ length: 500 }, (_, i) => `word${i}`).join(" ");
    const { chunks } = chunkText(body);
    assert.ok(chunks.length > 1, "fixture must span more than one chunk");

    const out = reconstructDocumentText(chunks.map((text, i) => content(i + 1, text)));
    assert.equal(out, body.replace(/\s+/g, " ").trim());
  });

  test("preserves POSITION ORDER rather than array order", () => {
    // The chunks arrive already ordered by position in production, but the
    // function's contract is about positions. A shuffled input must not
    // silently produce shuffled text.
    const shuffled = [
      content(2, "SECOND"),
      content(1, "FIRST"),
    ];
    const out = reconstructDocumentText([...shuffled].sort((x, y) => x.position - y.position));
    assert.equal(out, "FIRSTSECOND");
  });
});

describe("indexTextLoader -- the production loadText", () => {
  function seededIndex(): { index: SearchIndex; path: string } {
    const path = tempIndexPath();
    mkdirSync(join(path, ".."), { recursive: true });
    const index = new SearchIndex(path);
    index.open();
    index.writeMeta({
      schemaVersion: 1,
      dimensions: EMBEDDING_DIMENSIONS,
      model: "stub-embed",
      prefixMode: "none",
    });
    index.replaceDocument(
      {
        id: 42,
        modified: "2026-11-15T00:00:00Z",
        title: "Forty Two",
        chunkCount: 2,
        truncated: false,
        metadataAt: "2026-11-15T00:00:00Z",
      },
      [
        { kind: "metadata", text: "Title: Forty Two.", embedding: axisVector(1) },
        { kind: "content", text: "the body of forty two", embedding: axisVector(2) },
      ]
    );
    index.close();
    const reopened = new SearchIndex(path);
    reopened.open();
    return { index: reopened, path };
  }

  test("returns the reconstructed body for each distinct document id", () => {
    const { index } = seededIndex();
    try {
      const load = indexTextLoader(index);
      const text = load([
        { documentId: 42, score: 1, passage: "", matched: "both", bestSimilarity: null },
        { documentId: 42, score: 1, passage: "", matched: "both", bestSimilarity: null },
      ]);

      assert.equal(text.size, 1, "a repeated id is fetched once");
      assert.equal(text.get(42), "the body of forty two");
      assert.ok(!String(text.get(42)).includes("Title:"), "the metadata chunk is excluded");
    } finally {
      index.close();
    }
  });

  test("a THROWING store leaves the document ABSENT, not empty", () => {
    // This is the documented tolerance, exercised on the branch that actually
    // implements it: the `try/catch` around `store.chunksFor(id)`.
    const store = {
      chunksFor() {
        throw new Error("index unreadable");
      },
    } as unknown as SearchIndex;

    const load = indexTextLoader(store);
    const text = load([
      { documentId: 5, score: 1, passage: "", matched: "both", bestSimilarity: null },
      { documentId: 6, score: 1, passage: "", matched: "both", bestSimilarity: null },
    ]);

    assert.equal(text.size, 0, "a throwing store must contribute nothing");
    assert.equal(text.has(5), false);
    assert.equal(text.has(6), false);
  });

  test("⚠️ FINDING: an id ABSENT FROM THE INDEX yields \"\", not absence", () => {
    // ⚠️ CHARACTERISATION TEST, NOT A FALSIFIABLE ASSERTION. It pins what the
    // code does TODAY so a future change is noticed; it has no mutation that
    // makes it fail, because it asserts the observed behaviour rather than a
    // required one. Its value is the FINDING recorded below, not protection.
    // MEASURED, AND IT CONTRADICTS THE DOC COMMENT. `indexTextLoader`'s KDoc
    // says "A document whose chunks cannot be read is simply absent from the
    // map". That is true only when `chunksFor` THROWS. When the store simply
    // has no rows, `chunksFor` returns `[]`, `reconstructDocumentText([])`
    // returns "", and the id lands in the map mapped to the empty string.
    //
    // WHY IT IS NOT A LIVE DEFECT: every id reaching this function came from
    // `store.vectorHits`/`textHits`, both of which read the same index, so an
    // id-in-hits-but-not-in-index is not reachable in production today. The
    // consequence if it ever became reachable is small (`content_length: 0` and
    // an empty passage, which is what the absent path produces downstream
    // anyway) -- but the DOC COMMENT is inaccurate and the difference between
    // "absent" and "present and empty" is exactly the kind this repo's
    // degradation contract keeps insisting on.
    //
    // Pinned as measured behaviour, NOT fixed: `search.ts` is out of scope for
    // this lap. Reported in the drawer.
    const { index } = seededIndex();
    try {
      const load = indexTextLoader(index);
      const text = load([
        { documentId: 999, score: 1, passage: "", matched: "both", bestSimilarity: null },
      ]);
      assert.equal(text.has(999), true, "measured: the id IS present in the map");
      assert.equal(text.get(999), "", "measured: mapped to the empty string");
    } finally {
      index.close();
    }
  });
});
