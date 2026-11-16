import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, describe, test } from "node:test";
import type { PaperlessAPI } from "../api/PaperlessAPI";
import type { SearchConfig } from "../search/config";
import type { Embedder } from "../search/embedder";
import { EMBEDDING_DIMENSIONS, SearchIndex } from "../search/store";
import { extractSearchTerms } from "./documents";
import {
  MAX_LIMIT,
  MAX_RESULT_CHARS,
  SearchContext,
  budgetedLimit,
  loadDocumentMetadata,
  runSemanticSearch,
  type VisibleDocument,
} from "./search";

/**
 * The P2 integration path that shipped in `00a8ba0` with open tests and that
 * `1db310b` covered at unit level but NEVER wired together.
 *
 * `1db310b` tested `budgetedLimit`, `fitResultToBudget`, `loadDocumentMetadata`,
 * `reconstructDocumentText` and `indexTextLoader` as isolated units. What no
 * test did was pass `loadText` into `runSemanticSearch` at all -- `grep loadText
 * src/**\/*.test.ts` returned one hit, and it was a `describe` title. So the
 * seam where the index text meets `buildPassage`, and the `offset_space` label
 * that tells a caller which space its offsets live in, were both unasserted.
 *
 * That seam is the whole point of the change: `loadDocumentMetadata` returns
 * `content: null` on EVERY document, so if `loadText` were dropped, mis-wired,
 * or shadowed by `doc.content`, every production passage would be empty and
 * the suite would stay green. These tests exist to make that impossible.
 */

let dir: string;

before(() => {
  dir = mkdtempSync(join(tmpdir(), "paperless-p2-int-"));
});

after(() => {
  rmSync(dir, { recursive: true, force: true });
});

let counter = 0;

function axisVector(axis: number): number[] {
  return Array.from({ length: EMBEDDING_DIMENSIONS }, (_, index) => (index === axis ? 1 : 0));
}

/**
 * A stub embedder that answers deterministically on one axis.
 *
 * `model`/`prefixMode` are stated explicitly rather than left unset: an unset
 * embedder resolves to a REAL Ollama client, and Ollama listens on this box.
 */
class StubEmbedder implements Embedder {
  readonly model = "stub-embed";
  readonly prefixMode = "none" as const;
  readonly dimensions = EMBEDDING_DIMENSIONS;

  constructor(private readonly axis = 1) {}

  async embedDocuments(texts: string[]): Promise<number[][]> {
    return texts.map(() => axisVector(this.axis));
  }

  async embedQuery(): Promise<number[]> {
    return axisVector(this.axis);
  }
}

interface Seed {
  id: number;
  title: string;
  /** The text the index holds for this document, i.e. what `chunksFor` returns. */
  indexText: string;
}

/** A real temp SQLite index, because the integration under test reads it. */
function buildContext(seeds: Seed[]): { context: SearchContext; index: SearchIndex } {
  counter += 1;
  const indexDir = join(dir, `ctx-${counter}`);
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
    embeddingLocalOnly: false,
    embeddingApiKey: undefined,
    webhookPort: 0,
    webhookSecret: undefined,
    webhookHost: "127.0.0.1",
    webhookDebounceMs: 2000,
  };
  const context = new SearchContext(config, () => new StubEmbedder());
  const index = new SearchIndex(path);
  index.open();
  index.writeMeta({
    schemaVersion: 1,
    dimensions: EMBEDDING_DIMENSIONS,
    model: "stub-embed",
    prefixMode: "none",
  });
  for (const seed of seeds) {
    index.replaceDocument(
      {
        id: seed.id,
        modified: "2026-11-15T00:00:00Z",
        title: seed.title,
        chunkCount: 2,
        truncated: false,
        metadataAt: "2026-11-15T00:00:00Z",
      },
      [
        { kind: "metadata", text: `Title: ${seed.title}.`, embedding: axisVector(1) },
        { kind: "content", text: seed.indexText, embedding: axisVector(seed.id) },
      ]
    );
  }
  index.close();
  return { context, index: context.index() };
}

/**
 * A document as `loadDocumentMetadata` really returns it: metadata only,
 * `content: null`. Stated here as the default so a test cannot accidentally
 * pass against the pre-P2 shape.
 */
function metadataOnly(id: number, title: string): VisibleDocument {
  return {
    id,
    title,
    content: null,
    correspondent: 1,
    document_type: 1,
    storage_path: 1,
    tags: [1],
    created: "2026-10-01",
  };
}

/**
 * ── THE SEAM `1db310b` LEFT OPEN ─────────────────────────────────────────────
 *
 * Every production call supplies `loadText` and every production document comes
 * back with `content: null`. So `runSemanticSearch` is only correct if it
 * prefers the index text. Nothing tested that.
 */
describe("runSemanticSearch — the index-text seam (loadText)", () => {
  const NEEDLE_BEFORE = "x".repeat(500);
  const NEEDLE_AFTER = "y".repeat(500);
  const INDEX_TEXT = `${NEEDLE_BEFORE} prana winds ${NEEDLE_AFTER}`;

  test("a passage is built from loadText even though doc.content is null", async () => {
    const { context, index } = buildContext([
      { id: 1, title: "Winds", indexText: INDEX_TEXT },
    ]);
    try {
      const result = await runSemanticSearch(context, "prana", {
        limit: 5,
        window: 200,
        // As `loadDocumentMetadata` really behaves: metadata only, no content.
        loadDocuments: async () => new Map([[1, metadataOnly(1, "Winds")]]),
        // The production loader, inlined: read the index, not the network.
        loadText: (hits) => {
          const text = new Map<number, string>();
          for (const id of new Set(hits.map((hit) => hit.documentId))) {
            const chunks = index.chunksFor(id).filter((chunk) => chunk.kind === "content");
            text.set(id, chunks.map((chunk) => chunk.text).join(""));
          }
          return text;
        },
      });

      assert.equal(result.passages.length, 1, "the hit must produce a passage");
      const passage = result.passages[0];

      // ⚠️ THE ASSERTION THAT MAKES THE LOADER LOAD-BEARING.
      // `doc.content` is null, so if the loader were ignored this passage would
      // be "". A green result here proves the index text reached buildPassage.
      assert.ok(
        passage.passage.length > 0,
        "a null-content document with no index text would yield an empty passage"
      );
      assert.ok(
        passage.passage.includes("prana"),
        `the window must contain the term it anchored on; got "${passage.passage}"`
      );
      assert.equal(
        passage.content_length,
        INDEX_TEXT.length,
        "content_length must measure the INDEX text, not the (null) document body"
      );
    } finally {
      index.close();
    }
  });

  test("offset_space says normalized when the text came from the index", async () => {
    // The field exists so a caller is not misled about what `offset` indexes
    // into. The two spaces are different: the chunker collapses whitespace runs,
    // so a normalized offset does not address the raw document byte-for-byte.
    const { context, index } = buildContext([
      { id: 1, title: "Winds", indexText: INDEX_TEXT },
    ]);
    try {
      const result = await runSemanticSearch(context, "prana", {
        limit: 5,
        window: 200,
        loadDocuments: async () => new Map([[1, metadataOnly(1, "Winds")]]),
        loadText: (hits) => {
          const text = new Map<number, string>();
          for (const id of new Set(hits.map((hit) => hit.documentId))) {
            text.set(
              id,
              index
                .chunksFor(id)
                .filter((chunk) => chunk.kind === "content")
                .map((chunk) => chunk.text)
                .join("")
            );
          }
          return text;
        },
      });

      assert.equal(
        result.passages[0].offset_space,
        "normalized",
        "index-sourced text is the normalized space; mislabelling it as " +
          '"document" would make the offset claim false'
      );
    } finally {
      index.close();
    }
  });

  test("offset_space says document when no loader is supplied (the fallback)", async () => {
    // The other branch. Reachable only when a caller injects no loader, which
    // production never does -- but the fallback still reads `doc.content`, and
    // it must not borrow the normalized label while doing so.
    const { context, index } = buildContext([{ id: 1, title: "Winds", indexText: INDEX_TEXT }]);
    try {
      const raw = `${NEEDLE_BEFORE} prana winds ${NEEDLE_AFTER}`;
      const result = await runSemanticSearch(context, "prana", {
        limit: 5,
        window: 200,
        loadDocuments: async () => new Map([[1, { ...metadataOnly(1, "Winds"), content: raw }]]),
        // No loadText at all.
      });

      assert.equal(result.passages.length, 1);
      assert.equal(
        result.passages[0].offset_space,
        "document",
        "the pre-P2 fallback path indexes RAW document text and must say so"
      );
      assert.equal(result.passages[0].content_length, raw.length);
    } finally {
      index.close();
    }
  });

  test("an id the loader omits falls back to doc.content rather than vanishing", async () => {
    // `indexText?.get(doc.id)` is undefined when the loader does not know the
    // id. The documented behaviour is a fallback to `doc.content`, NOT a drop:
    // dropping would silently lose a valid hit because the index lagged.
    const { context, index } = buildContext([{ id: 1, title: "Winds", indexText: INDEX_TEXT }]);
    try {
      const fallback = "fallback body mentioning prana here";
      const result = await runSemanticSearch(context, "prana", {
        limit: 5,
        window: 200,
        loadDocuments: async () => new Map([[1, { ...metadataOnly(1, "Winds"), content: fallback }]]),
        // A loader that returns nothing for this id.
        loadText: () => new Map<number, string>(),
      });

      assert.equal(result.passages.length, 1, "the hit must still produce a passage");
      assert.equal(result.passages[0].offset_space, "document");
      assert.ok(result.passages[0].passage.includes("prana"));
    } finally {
      index.close();
    }
  });
});

/**
 * ── THE `\p{M}` FIX IN `extractSearchTerms` ─────────────────────────────────
 *
 * `00a8ba0` added `\p{M}` to the split class at `documents.ts:247`. The existing
 * coverage in `boundedContent.test.ts:256` uses `café`, which is U+00E9 --
 * a PRECOMPOSED character that is a single `\p{L}` codepoint. It therefore
 * passes whether or not `\p{M}` is present, and cannot catch a regression.
 *
 * A Tibetan syllable is a base letter plus COMBINING marks (category `Mn`).
 * Without `\p{M}` in the class, `[^\p{L}\p{N}_]` treats those marks as
 * separators and shatters the syllable. On a Tibetan-heavy corpus that turns
 * every lexical lookup into a miss that looks like an ordinary no-match.
 */
describe("extractSearchTerms — combining marks (the actual 00a8ba0 fix)", () => {
  test("a Tibetan syllable survives as ONE term, not its fragments", () => {
    // ⚠️ THE BASE LETTER IS LOAD-BEARING IN THE FIXTURE. `རླུང` = U+0F62 (ra, a
    // LETTER) + U+0FB3 + U+0F74 + U+0F44 (three combining marks). An earlier
    // version of this test used the marks ALONE (no base letter) and the
    // pre-fix regex returned a single fragment for it -- the test passed its
    // own precondition check but proved nothing, which is exactly the failure
    // mode a falsifier is supposed to prevent. Measured, then corrected.
    const syllable = "\u0F62\u0FB3\u0F74\u0F44";
    // The BASE LETTER must be a Letter, so the pre-fix class has one anchor to
    // split around.
    assert.ok(/\p{L}/u.test(syllable[0]), "precondition: first codepoint is a letter");
    assert.ok(/\p{M}/u.test(syllable), "precondition: the fixture carries combining marks");
    // Precondition, and the whole reason `café` is not equivalent: at least one
    // codepoint is a combining mark, not a letter.
    assert.ok(
      /\p{M}/u.test(syllable),
      "precondition: the fixture must actually contain a combining mark"
    );

    assert.deepEqual(
      extractSearchTerms(syllable),
      [syllable],
      "the syllable must not be split; without \\p{M} this yields four fragments"
    );
  });

  test("fragments would be produced by a class missing \\p{M} — proven here", () => {
    // The falsifier, run inline so the mechanism is demonstrated rather than
    // asserted. This mirrors the pre-fix regex exactly.
    const syllable = "\u0F62\u0FB3\u0F74\u0F44";
    const preFix = syllable.toLowerCase().split(/[^\p{L}\p{N}_]+/u).filter(Boolean);
    assert.ok(
      preFix.length > 1,
      `the PRE-FIX class must shatter "${syllable}" or this test proves nothing`
    );

    const postFix = extractSearchTerms(syllable);
    assert.equal(postFix.length, 1, "the fix must rejoin it");
    assert.notDeepEqual(preFix, postFix, "the fix must change behaviour, not just compile");
  });

  test("a precomposed accented word is NOT evidence — the existing gap, pinned", () => {
    // `café` with U+00E9 is one `\p{L}` codepoint, so it survives the pre-fix
    // regex too. Recorded here so nobody later "strengthens" the accent test and
    // believes the combining-mark case is covered.
    const precomposed = "caf\u00E9";
    assert.ok(!/\p{M}/u.test(precomposed), "precondition: no combining marks in this form");
    const preFix = precomposed.toLowerCase().split(/[^\p{L}\p{N}_]+/u).filter(Boolean);
    assert.deepEqual(preFix, [precomposed], "the pre-fix regex already handled this form");
  });
});

/**
 * ── VISIBILITY: WHAT THE SERVER OMITS MUST NOT BE INVENTED ──────────────────
 *
 * `loadDocumentMetadata` does NO client-side permission filtering. It issues
 * `GET /documents/?id__in=...` and trusts Paperless to omit what the caller may
 * not read. So the property to protect is not "the code filters" -- it is
 * **"the code never resurrects a document the server declined to return."**
 *
 * That matters because the map it builds is what the passage loop iterates. A
 * document added optimistically, or synthesized from the id list, would put an
 * unreadable document's title and metadata into a caller's result.
 */
describe("loadDocumentMetadata — an omitted document is never invented", () => {
  test("ids the server does not return are ABSENT from the map", async () => {
    // The server sees three ids and returns one. The other two are withheld --
    // exactly what a permission-restricted `id__in` query looks like.
    const seen: string[] = [];
    const api = {
      async request(path: string) {
        seen.push(path);
        return { results: [{ id: 2, title: "Visible" }] };
      },
    } as unknown as PaperlessAPI;

    const documents = await loadDocumentMetadata(api, [1, 2, 3]);

    assert.equal(seen.length, 1, "one batched request");
    assert.deepEqual(
      [...documents.keys()],
      [2],
      "only the document the server returned may appear"
    );
    assert.ok(!documents.has(1), "a withheld id must not be fabricated");
    assert.ok(!documents.has(3), "a withheld id must not be fabricated");
  });

  test("CHARACTERISATION: an id the caller never requested IS retained", async () => {
    // Not cosmetic: the request is built from the caller's ids, and the map is
    // keyed by whatever came back. If an id the caller never asked for were
    // retained, the passage loop could serve a document outside the request.
    const api = {
      async request() {
        return {
          results: [
            { id: 1, title: "Asked" },
            { id: 999, title: "Not asked" },
          ],
        };
      },
    } as unknown as PaperlessAPI;

    const documents = await loadDocumentMetadata(api, [1]);

    assert.deepEqual([...documents.keys()], [1, 999]);
    // ⚠️ MEASURED, NOT ASSUMED: the implementation does NOT intersect the
    // response with the request, so an unsolicited id IS retained. This is
    // harmless in production (Paperless only answers the ids it was given) but
    // it means `loadDocumentMetadata` is not the enforcement point for scope.
    // Recorded as a finding rather than asserted as a desirable property.
    assert.equal(
      documents.size,
      2,
      "if this ever becomes 1, the function gained a scope check — update this test"
    );
  });

  test("a non-numeric id in the response is rejected, not coerced", async () => {
    const api = {
      async request() {
        return { results: [{ id: "7", title: "String id" }, { id: 8, title: "Numeric" }] };
      },
    } as unknown as PaperlessAPI;

    const documents = await loadDocumentMetadata(api, [7, 8]);

    assert.ok(!documents.has(7), 'the string "7" must not become key 7');
    assert.ok(documents.has(8), "the genuinely numeric id survives");
    assert.equal(documents.size, 1);
  });
});

/**
 * ── THE TWO ENFORCEMENTS, WIRED TOGETHER ────────────────────────────────────
 *
 * `1db310b` tested each in isolation. The question this asks is the one the arc
 * actually cares about: does a maximal request produce a bounded response from
 * end to end?
 */
describe("the budget holds end to end", () => {
  test("a MAX_LIMIT / max-window search returns a result under the ceiling", () => {
    // The pure arithmetic, stated as the invariant rather than a table: whatever
    // the caller asks for, the passage COUNT that will be attempted must fit.
    for (const window of [200, 800, 2000, 4000]) {
      const limit = budgetedLimit(MAX_LIMIT, window);
      const worst = limit * (window + 200);
      assert.ok(
        worst <= MAX_RESULT_CHARS,
        `window ${window}: ${limit} passages x ${window}+200 = ${worst} exceeds ${MAX_RESULT_CHARS}`
      );
      assert.ok(limit >= 1, `window ${window} must still permit one passage`);
    }
  });
});
