import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, describe, test } from "node:test";
import Database from "better-sqlite3";
import { load as loadVecExtension } from "sqlite-vec";
import {
  CHUNK_ROWID_STRIDE,
  DeleteCountError,
  EMBEDDING_DIMENSIONS,
  HAS_FTS5,
  LexicalUnavailableError,
  SCHEMA_VERSION,
  SchemaIncompatibleError,
  SearchIndex,
  StoredChunk,
  chunkRowid,
  decodeChunkRowid,
  toFtsMatch,
} from "./store";
// The mismatch class lives with the prefix logic, not with the store, so a
// `open()` test asserting on it has to reach across. Importing it rather than
// matching the message keeps the two in step if the wording ever changes.
import { PrefixModeMismatchError } from "./embedder";

let dir: string;

before(() => {
  dir = mkdtempSync(join(tmpdir(), "paperless-search-"));
});

after(() => {
  rmSync(dir, { recursive: true, force: true });
});

let counter = 0;
function tempIndexPath(): string {
  counter += 1;
  return join(dir, `index-${counter}.db`);
}

/** A unit vector along one axis: identical axes are identical, others are 1 apart. */
function axisVector(axis: number): number[] {
  return Array.from({ length: EMBEDDING_DIMENSIONS }, (_, index) =>
    index === axis ? 1 : 0
  );
}

function chunk(kind: "metadata" | "content", text: string, axis: number): StoredChunk {
  return { kind, text, embedding: axisVector(axis) };
}

function makeDoc(
  id: number,
  chunks: StoredChunk[],
  overrides: Partial<{ modified: string; title: string; truncated: boolean }> = {}
) {
  return {
    id,
    modified: overrides.modified ?? "2026-11-15T00:00:00Z",
    title: overrides.title ?? `Document ${id}`,
    chunkCount: chunks.length,
    truncated: overrides.truncated ?? false,
    metadataAt: "2026-11-15T00:00:00Z",
  };
}

function openIndex(path = tempIndexPath()): SearchIndex {
  const index = new SearchIndex(path);
  index.open();
  return index;
}

/**
 * Write a document and its chunks in one call.
 *
 * replaceDocument takes the chunks twice - once inside the document record
 * (which carries chunkCount) and once as the chunk list - because the store
 * asserts the two agree. This helper derives both from one source so a test
 * cannot accidentally declare a count it does not supply.
 */
function put(
  index: SearchIndex,
  id: number,
  chunks: StoredChunk[],
  overrides: Partial<{ modified: string; title: string; truncated: boolean }> = {}
): void {
  index.replaceDocument(makeDoc(id, chunks, overrides), chunks);
}

// ── Capability ─────────────────────────────────────────────────────────────

describe("FTS5 availability", () => {
  test("the build reports FTS5 present (measured, not assumed)", () => {
    // Measured on the build host under node v24.21.0 / better-sqlite3 13.2.0:
    // compile_options contains ENABLE_FTS5, sqlite 3.54.1, and a
    // `create virtual table ... using fts5(x)` succeeds.
    assert.equal(HAS_FTS5, true);
  });

  test("without FTS5 the lexicon REFUSES rather than returning silence", () => {
    // The whole point of the check. Returning [] would be indistinguishable
    // from "this query matched nothing", so the ranker would believe it had
    // searched lexically and found nothing. The build on this box HAS FTS5, so
    // the flag is injected; otherwise this branch would be untested here and
    // its only evidence would be that it compiles.
    const index = new SearchIndex(tempIndexPath(), { fts5: false });
    index.open();
    try {
      put(index, 1, [chunk("content", "meditation on emptiness", 0)]);
      assert.equal(index.counts().ftsRows, null, "no fts table was created");
      assert.throws(() => index.textHits("emptiness", 5), LexicalUnavailableError);
      // The vector half must still work — the feature degrades, it does not die.
      assert.equal(index.counts().vectors, 1);
    } finally {
      index.close();
    }
  });

  test("a lexical search works when FTS5 is present", () => {
    const index = openIndex();
    try {
      put(index, 1, [chunk("content", "meditation on emptiness", 0)]);
      const hits = index.textHits("emptiness", 5);
      assert.equal(hits.length, 1);
      assert.equal(hits[0].documentId, 1);
      assert.ok(hits[0].score > 0, "score is negated to higher-is-better");
    } finally {
      index.close();
    }
  });

  test("diacritics fold in both directions", () => {
    const index = openIndex();
    try {
      put(index, 1, [chunk("content", "Título del documento", 0)]);
      assert.equal(index.textHits("titulo", 5).length, 1);
      assert.equal(index.textHits("Título", 5).length, 1);
    } finally {
      index.close();
    }
  });

  test("raw FTS5 syntax in a user query is neutralised, not thrown", () => {
    const index = openIndex();
    try {
      put(index, 1, [chunk("content", "beans with ghee and cumin", 0)]);
      // Raw `c++ (test)` and `a-b` both throw on a MATCH; these must not.
      for (const query of ["c++ (test)", "a-b", "beans AND ghee", "quo\"te"]) {
        assert.doesNotThrow(() => index.textHits(query, 5), query);
      }
      assert.equal(toFtsMatch("   "), null);
      assert.equal(toFtsMatch(""), null);
    } finally {
      index.close();
    }
  });
});

// ── The five measured traps ────────────────────────────────────────────────

describe("trap 1 — vec0 rowid parameters", () => {
  test("a plain bound rowid is rejected while CAST is accepted (the trap itself)", () => {
    // This test exists to prove the trap is real on this build. If it ever
    // stops failing, the CAST can be reconsidered — until then, the cast is
    // load-bearing and the evidence lives here rather than in a comment.
    const db = new Database(":memory:");
    try {
      loadVecExtension(db);
      db.exec("CREATE VIRTUAL TABLE v USING vec0(embedding float[4] distance_metric=cosine)");
      const blob = Buffer.from(Float32Array.from([1, 0, 0, 0]).buffer);
      assert.throws(
        () => db.prepare("INSERT INTO v(rowid, embedding) VALUES (?, ?)").run(500, blob),
        /Only integers are allows for primary key values/
      );
      assert.doesNotThrow(() =>
        db.prepare("INSERT INTO v(rowid, embedding) VALUES (CAST(? AS INTEGER), ?)").run(500, blob)
      );
      assert.doesNotThrow(() =>
        db.prepare("INSERT INTO v(rowid, embedding) VALUES (501, ?)").run(blob)
      );
    } finally {
      db.close();
    }
  });

  test("the store writes and reads through the cast path", () => {
    const index = openIndex();
    try {
      put(index, 7, [chunk("content", "one", 0), chunk("content", "two", 1)]);
      const hits = index.vectorHits(axisVector(0), 5);
      assert.equal(hits.length, 1);
      assert.equal(hits[0].documentId, 7);
      assert.equal(hits[0].position, 0);
    } finally {
      index.close();
    }
  });
});

describe("trap 2 — asserting on `changes`, not on the absence of an error", () => {
  test("a bound delete of a MISSING rowid reports changes: 0 and does not throw", () => {
    // The recon note blamed vec0 for this. It is ordinary SQL: the row was not
    // there. Recorded as a test so the next reader sees the real mechanism.
    const db = new Database(":memory:");
    try {
      loadVecExtension(db);
      db.exec("CREATE VIRTUAL TABLE v USING vec0(embedding float[4] distance_metric=cosine)");
      const blob = Buffer.from(Float32Array.from([1, 0, 0, 0]).buffer);
      db.prepare("INSERT INTO v(rowid, embedding) VALUES (CAST(? AS INTEGER), ?)").run(1, blob);
      db.prepare("INSERT INTO v(rowid, embedding) VALUES (CAST(? AS INTEGER), ?)").run(2, blob);
      const missing = db.prepare("DELETE FROM v WHERE rowid = ?").run(5);
      assert.equal(missing.changes, 0);
      const present = db.prepare("DELETE FROM v WHERE rowid = ?").run(2);
      assert.equal(present.changes, 1);
    } finally {
      db.close();
    }
  });

  test("a range delete reports its true count for every bound numeric type", () => {
    for (const value of [5, 5.0, "5"]) {
      const db = new Database(":memory:");
      try {
        loadVecExtension(db);
        db.exec("CREATE VIRTUAL TABLE v USING vec0(embedding float[4] distance_metric=cosine)");
        const blob = Buffer.from(Float32Array.from([1, 0, 0, 0]).buffer);
        db.prepare("INSERT INTO v(rowid, embedding) VALUES (CAST(? AS INTEGER), ?)").run(5, blob);
        const removed = db.prepare("DELETE FROM v WHERE rowid = ?").run(value as never);
        assert.equal(removed.changes, 1, `bound ${String(value)}`);
      } finally {
        db.close();
      }
    }
  });

  test("the store throws DeleteCountError when the vector table is out of step", () => {
    const index = openIndex();
    try {
      put(index, 11, [chunk("content", "alpha", 0), chunk("content", "beta", 1)]);
      // Reach past the API to corrupt the pairing: drop one vector row while
      // leaving both chunk rows. deleteDocumentChunks must notice.
      const db = new Database(join(dir, "corrupt.db"));
      db.close();
      const handle = (index as unknown as { db: Database.Database }).db;
      handle.prepare("DELETE FROM vec_chunks WHERE rowid = CAST(? AS INTEGER)").run(chunkRowid(11, 1));
      assert.throws(() => index.deleteDocuments([11]), DeleteCountError);
    } finally {
      index.close();
    }
  });
});

describe("trap 3 — there is no INSERT OR REPLACE on vec0", () => {
  test("a duplicate rowid insert throws rather than silently replacing", () => {
    const db = new Database(":memory:");
    try {
      loadVecExtension(db);
      db.exec("CREATE VIRTUAL TABLE v USING vec0(embedding float[4] distance_metric=cosine)");
      const blob = Buffer.from(Float32Array.from([1, 0, 0, 0]).buffer);
      const insert = db.prepare("INSERT INTO v(rowid, embedding) VALUES (CAST(? AS INTEGER), ?)");
      insert.run(42, blob);
      assert.throws(() => insert.run(42, blob), /UNIQUE constraint failed/);
      assert.throws(
        () => db.prepare("INSERT OR REPLACE INTO v(rowid, embedding) VALUES (CAST(? AS INTEGER), ?)").run(42, blob),
        /UNIQUE constraint failed/
      );
    } finally {
      db.close();
    }
  });

  test("replaceDocument replaces cleanly, twice over", () => {
    const index = openIndex();
    try {
      put(index, 3, [chunk("content", "first body", 0)]);
      put(index, 3, [chunk("content", "second body", 1)]);
      const counts = index.counts();
      assert.equal(counts.documents, 1);
      assert.equal(counts.chunks, 1);
      assert.equal(counts.vectors, 1);
      assert.equal(index.textHits("first", 5).length, 0);
      assert.equal(index.textHits("second", 5).length, 1);
    } finally {
      index.close();
    }
  });

  test("a vec0 insert inside a rolled-back transaction disappears", () => {
    const db = new Database(":memory:");
    try {
      loadVecExtension(db);
      db.exec("CREATE VIRTUAL TABLE v USING vec0(embedding float[4] distance_metric=cosine)");
      const blob = Buffer.from(Float32Array.from([1, 0, 0, 0]).buffer);
      db.exec("BEGIN");
      db.prepare("INSERT INTO v(rowid, embedding) VALUES (CAST(? AS INTEGER), ?)").run(777, blob);
      db.exec("ROLLBACK");
      const row = db.prepare("SELECT rowid FROM v WHERE rowid = 777").all();
      assert.deepEqual(row, []);
    } finally {
      db.close();
    }
  });
});

describe("trap 4 — rowid IN () composes with MATCH", () => {
  test("a rowid filter narrows a kNN result", () => {
    const db = new Database(":memory:");
    try {
      loadVecExtension(db);
      db.exec("CREATE VIRTUAL TABLE v USING vec0(embedding float[4] distance_metric=cosine)");
      const blob = Buffer.from(Float32Array.from([1, 0, 0, 0]).buffer);
      const other = Buffer.from(Float32Array.from([0, 1, 0, 0]).buffer);
      const insert = db.prepare("INSERT INTO v(rowid, embedding) VALUES (CAST(? AS INTEGER), ?)");
      [100, 200, 300].forEach((rowid) => insert.run(rowid, blob));
      insert.run(400, other);
      const query = Buffer.from(Float32Array.from([1, 0, 0, 0]).buffer);
      const filtered = db
        .prepare("SELECT rowid FROM v WHERE embedding MATCH ? AND k = 5 AND rowid IN (200)")
        .all(query);
      assert.deepEqual(filtered, [{ rowid: 200 }]);
    } finally {
      db.close();
    }
  });
});

describe("trap 5 — cosine must be declared, and similarity is 1 - distance", () => {
  test("an orthogonal vector scores ~0 on a cosine table, not ~16", () => {
    const index = openIndex();
    try {
      put(index, 1, [chunk("content", "x axis", 0)]);
      put(index, 2, [chunk("content", "y axis", 1)]);
      const hits = index.vectorHits(axisVector(0), 5);
      const first = hits.find((hit) => hit.documentId === 1);
      const second = hits.find((hit) => hit.documentId === 2);
      assert.ok(first && second);
      assert.ok(Math.abs(first.score - 1) < 1e-6, "self similarity is 1");
      assert.ok(Math.abs(second.score) < 1e-6, "orthogonal similarity is 0, so distance was cosine");
    } finally {
      index.close();
    }
  });

  test("results are ordered best-first by similarity", () => {
    const index = openIndex();
    try {
      put(index, 1, [chunk("content", "x", 0)]);
      put(index, 2, [chunk("content", "y", 1)]);
      const hits = index.vectorHits(axisVector(0), 5);
      assert.deepEqual(
        hits.map((hit) => hit.documentId),
        [1, 2]
      );
    } finally {
      index.close();
    }
  });
});

// ── The rowid arithmetic, which the brief had wrong ────────────────────────

describe("chunk rowid stride", () => {
  test("the stride is above the configured chunk ceiling", () => {
    // The brief proposed docId * 1001 + position, which collides as soon as a
    // document has more than 1000 chunks — and MAX_CHUNKS defaults to 4000.
    assert.ok(CHUNK_ROWID_STRIDE > 4000);
  });

  test("the brief's 1001 stride collides; the real stride does not", () => {
    const collide = new Set<number>();
    let collisions = 0;
    for (let docId = 1; docId <= 4; docId++) {
      for (let position = 0; position <= 4000; position++) {
        const rowid = docId * 1001 + position;
        if (collide.has(rowid)) collisions += 1;
        else collide.add(rowid);
      }
    }
    assert.ok(collisions > 0, "the brief's stride really does collide");

    const safe = new Set<number>();
    for (let docId = 1; docId <= 4; docId++) {
      for (let position = 0; position <= 4000; position++) {
        const rowid = docId * CHUNK_ROWID_STRIDE + position;
        assert.ok(!safe.has(rowid), `collision at doc ${docId} pos ${position}`);
        safe.add(rowid);
      }
    }
  });

  test("encode and decode round-trip", () => {
    for (const [docId, position] of [[1, 0], [7, 3], [2466, 3999], [99999, 8191]]) {
      const rowid = chunkRowid(docId, position);
      assert.deepEqual(decodeChunkRowid(rowid), { docId, position });
    }
  });

  test("a position beyond the stride is refused rather than silently colliding", () => {
    assert.throws(() => chunkRowid(1, CHUNK_ROWID_STRIDE), /does not fit the rowid stride/);
    assert.throws(() => chunkRowid(0, 0), /docId must be a positive integer/);
    assert.throws(() => chunkRowid(1, -1), /position must be a non-negative integer/);
  });
});

// ── Metadata and schema compatibility ──────────────────────────────────────

describe("index metadata", () => {
  test("a fresh index has no meta", () => {
    const index = openIndex();
    try {
      assert.equal(index.meta(), undefined);
    } finally {
      index.close();
    }
  });

  test("meta round-trips", () => {
    const index = openIndex();
    try {
      index.writeMeta({
        schemaVersion: SCHEMA_VERSION,
        dimensions: EMBEDDING_DIMENSIONS,
        model: "nomic-embed-text",
        prefixMode: "nomic",
      });
      assert.deepEqual(index.meta(), {
        schemaVersion: SCHEMA_VERSION,
        dimensions: EMBEDDING_DIMENSIONS,
        model: "nomic-embed-text",
        prefixMode: "nomic",
      });
    } finally {
      index.close();
    }
  });

  /**
   * THE ASSERT HAS TO FIRE ON `open()`, NOT ONLY WHEN SOMEONE CALLS IT.
   *
   * `assertSchemaCompatible` was dead code -- no production call site -- so a
   * prefix-mode mismatch could never be refused, no matter what its own KDoc
   * promised. These four tests drive it through the CONSTRUCTOR path, which is
   * what a real query does, rather than calling the assertion directly (which
   * is what makes the function green while the guard is absent).
   */
  test("open() throws on a prefix-mode mismatch when an expected meta is supplied", () => {
    const path = tempIndexPath();
    const first = new SearchIndex(path);
    first.open();
    first.writeMeta({
      schemaVersion: SCHEMA_VERSION,
      dimensions: EMBEDDING_DIMENSIONS,
      model: "nomic-embed-text",
      prefixMode: "nomic",
    });
    first.close();

    assert.throws(
      () =>
        new SearchIndex(path, {
          expected: { dimensions: EMBEDDING_DIMENSIONS, model: "nomic-embed-text", prefixMode: "e5" },
        }).open(),
      PrefixModeMismatchError
    );
  });

  test("open() throws on a same-width model swap", () => {
    // Same 768 dimensions, different model: the vector space differs even though
    // every arithmetic check passes. This is the case a dimension check cannot
    // catch, and the one that returns plausible nonsense instead of an error.
    const path = tempIndexPath();
    const first = new SearchIndex(path);
    first.open();
    first.writeMeta({
      schemaVersion: SCHEMA_VERSION,
      dimensions: EMBEDDING_DIMENSIONS,
      model: "nomic-embed-text",
      prefixMode: "nomic",
    });
    first.close();

    assert.throws(
      () =>
        new SearchIndex(path, {
          expected: { dimensions: EMBEDDING_DIMENSIONS, model: "some-other-768-model", prefixMode: "nomic" },
        }).open(),
      /built with model "nomic-embed-text"/
    );
  });

  test("open() throws on a bumped schema_version", () => {
    const path = tempIndexPath();
    const first = new SearchIndex(path);
    first.open();
    first.writeMeta({
      schemaVersion: SCHEMA_VERSION,
      dimensions: EMBEDDING_DIMENSIONS,
      model: "nomic-embed-text",
      prefixMode: "nomic",
    });
    first.close();

    // Force a future version into the row through a plain SQLite handle, the way
    // a newer build would have written it: `writeMeta` refuses to downgrade, so
    // it is the wrong instrument for manufacturing the state under test.
    const raw = new Database(path);
    raw.prepare("UPDATE index_meta SET value = '99' WHERE key = 'schema_version'").run();
    raw.close();

    assert.throws(
      () =>
        new SearchIndex(path, {
          expected: { dimensions: EMBEDDING_DIMENSIONS, model: "nomic-embed-text", prefixMode: "nomic" },
        }).open(),
      /schema_version 99/
    );
  });

  test("open() succeeds on a matching meta, and on a brand-new empty index", () => {
    // The negative cases above are useless if the guard refuses everything. Two
    // positives: a matching index opens, and a FRESH one opens too -- because
    // refusing a fresh index would make the first `sync_embeddings` unable to
    // create the thing it exists to create.
    const path = tempIndexPath();
    const first = new SearchIndex(path);
    first.open();
    first.writeMeta({
      schemaVersion: SCHEMA_VERSION,
      dimensions: EMBEDDING_DIMENSIONS,
      model: "nomic-embed-text",
      prefixMode: "nomic",
    });
    first.close();

    const reopened = new SearchIndex(path, {
      expected: { dimensions: EMBEDDING_DIMENSIONS, model: "nomic-embed-text", prefixMode: "nomic" },
    });
    reopened.open();
    assert.equal(reopened.meta()?.prefixMode, "nomic");
    reopened.close();

    const fresh = new SearchIndex(tempIndexPath(), {
      expected: { dimensions: EMBEDDING_DIMENSIONS, model: "nomic-embed-text", prefixMode: "nomic" },
    });
    fresh.open();
    assert.equal(fresh.meta(), undefined);
    fresh.close();
  });

  test("assertSchemaCompatible refuses an unknown schema_version", () => {
    const index = openIndex();
    try {
      index.writeMeta({
        schemaVersion: 99,
        dimensions: EMBEDDING_DIMENSIONS,
        model: "nomic-embed-text",
        prefixMode: "nomic",
      });
      assert.throws(
        () => index.assertSchemaCompatible(),
        (err: unknown) => {
          assert.ok(err instanceof SchemaIncompatibleError);
          assert.match(String(err), /schema_version 99/);
          return true;
        }
      );
    } finally {
      index.close();
    }
  });

  test("assertSchemaCompatible refuses a mismatched prefix mode", () => {
    const index = openIndex();
    try {
      index.writeMeta({
        schemaVersion: SCHEMA_VERSION,
        dimensions: EMBEDDING_DIMENSIONS,
        model: "nomic-embed-text",
        prefixMode: "nomic",
      });
      assert.throws(
        () => index.assertSchemaCompatible({ prefixMode: "e5" }),
        /built with EMBEDDING_PREFIX_MODE="nomic"/
      );
      assert.doesNotThrow(() => index.assertSchemaCompatible({ prefixMode: "nomic" }));
    } finally {
      index.close();
    }
  });

  test("assertSchemaCompatible refuses a mismatched model", () => {
    const index = openIndex();
    try {
      index.writeMeta({
        schemaVersion: SCHEMA_VERSION,
        dimensions: EMBEDDING_DIMENSIONS,
        model: "nomic-embed-text",
        prefixMode: "nomic",
      });
      assert.throws(
        () => index.assertSchemaCompatible({ model: "mxbai-embed-large" }),
        /different spaces/
      );
    } finally {
      index.close();
    }
  });

  test("assertSchemaCompatible refuses an index with no metadata at all", () => {
    const index = openIndex();
    try {
      assert.throws(() => index.assertSchemaCompatible(), /no metadata/);
    } finally {
      index.close();
    }
  });

  test("writeMeta refuses to downgrade an existing schema version", () => {
    const index = openIndex();
    try {
      index.writeMeta({
        schemaVersion: 2,
        dimensions: EMBEDDING_DIMENSIONS,
        model: "nomic-embed-text",
        prefixMode: "nomic",
      });
      assert.throws(
        () =>
          index.writeMeta({
            schemaVersion: SCHEMA_VERSION,
            dimensions: EMBEDDING_DIMENSIONS,
            model: "nomic-embed-text",
            prefixMode: "nomic",
          }),
        SchemaIncompatibleError
      );
    } finally {
      index.close();
    }
  });
});

// ── Bookkeeping ────────────────────────────────────────────────────────────

describe("document bookkeeping", () => {
  test("indexed() returns id to modified", () => {
    const index = openIndex();
    try {
      put(index, 1, [chunk("content", "a", 0)], { modified: "2026-01-01T00:00:00Z" });
      put(index, 2, [chunk("content", "b", 1)], { modified: "2026-02-02T00:00:00Z" });
      const indexed = index.indexed();
      assert.equal(indexed.get(1), "2026-01-01T00:00:00Z");
      assert.equal(indexed.get(2), "2026-02-02T00:00:00Z");
      assert.equal(indexed.size, 2);
    } finally {
      index.close();
    }
  });

  test("counts() reflects what was written", () => {
    const index = openIndex();
    try {
      put(index, 1, [chunk("metadata", "meta", 0), chunk("content", "one", 1)]);
      put(index, 2, [chunk("content", "two", 2)]);
      const counts = index.counts();
      assert.equal(counts.documents, 2);
      assert.equal(counts.chunks, 3);
      assert.equal(counts.vectors, 3);
      assert.equal(counts.ftsRows, HAS_FTS5 ? 3 : null);
    } finally {
      index.close();
    }
  });

  test("deleteDocuments removes the document, its chunks and its vectors", () => {
    const index = openIndex();
    try {
      put(index, 1, [chunk("content", "keep", 0)]);
      put(index, 2, [chunk("content", "drop", 1)]);
      index.deleteDocuments([2]);
      const counts = index.counts();
      assert.equal(counts.documents, 1);
      assert.equal(counts.chunks, 1);
      assert.equal(counts.vectors, 1);
      assert.equal(index.textHits("drop", 5).length, 0);
    } finally {
      index.close();
    }
  });

  test("deleteDocuments on an empty list is a no-op, not an error", () => {
    const index = openIndex();
    try {
      index.deleteDocuments([]);
      assert.equal(index.counts().documents, 0);
    } finally {
      index.close();
    }
  });

  test("replaceMetadata updates only position 0 and the metadata stamp", () => {
    const index = openIndex();
    try {
      put(index, 1, [chunk("metadata", "old metadata", 0), chunk("content", "body", 1)]);
      index.replaceMetadata(1, chunk("metadata", "new metadata", 2), "2026-12-25T00:00:00Z");
      const counts = index.counts();
      assert.equal(counts.documents, 1);
      assert.equal(counts.chunks, 2);
      assert.equal(counts.vectors, 2);
      assert.equal(index.textHits("old", 5).length, 0);
      assert.equal(index.textHits("new", 5).length, 1);
      assert.equal(index.metadataStamps().get(1), "2026-12-25T00:00:00Z");
    } finally {
      index.close();
    }
  });

  test("replaceMetadata clears the OLD text from the lexical index too", () => {
    // Regression: the first implementation deleted the vector row and inserted
    // both new rows, but never deleted the FTS row. The old metadata text kept
    // matching forever, so one document answered to two contradictory strings.
    // Found by the test above, not by reading the code.
    const index = openIndex();
    try {
      put(index, 1, [chunk("metadata", "sesame oil", 0), chunk("content", "body", 1)]);
      index.replaceMetadata(1, chunk("metadata", "coconut oil", 2), "2026-12-25T00:00:00Z");
      assert.equal(index.textHits("sesame", 5).length, 0, "stale metadata still indexed");
      assert.equal(index.textHits("coconut", 5).length, 1);
      // A single document must never occupy two result slots for one query.
      const hits = index.textHits("oil", 5);
      assert.equal(hits.length, 1);
      assert.equal(hits[0].documentId, 1);
    } finally {
      index.close();
    }
  });

  test("replaceMetadata refuses a content chunk", () => {
    const index = openIndex();
    try {
      put(index, 1, [chunk("content", "body", 0)]);
      assert.throws(
        () => index.replaceMetadata(1, chunk("content", "nope", 1), "2026-01-01T00:00:00Z"),
        /requires a chunk of kind 'metadata'/
      );
    } finally {
      index.close();
    }
  });

  test("replaceDocument refuses a chunk count that does not match", () => {
    const index = openIndex();
    try {
      const doc = makeDoc(1, [chunk("content", "a", 0)]);
      assert.throws(
        () => index.replaceDocument(doc, []),
        /declares 1 chunks but received 0/
      );
    } finally {
      index.close();
    }
  });

  test("a short k result is normal, not an error", () => {
    const index = openIndex();
    try {
      put(index, 1, [chunk("content", "only", 0)]);
      const hits = index.vectorHits(axisVector(0), 25);
      assert.equal(hits.length, 1);
    } finally {
      index.close();
    }
  });

  test("vectorHits partitions by document so one document cannot fill the list", () => {
    const index = openIndex();
    try {
      // Document 1 has three chunks all at distance 0; document 2 has one.
      put(index, 1, [
          chunk("content", "a", 0),
          chunk("content", "b", 0),
          chunk("content", "c", 0),
        ]);
      put(index, 2, [chunk("content", "d", 0)]);
      const hits = index.vectorHits(axisVector(0), 10);
      assert.deepEqual(
        hits.map((hit) => hit.documentId).sort((a, b) => a - b),
        [1, 2]
      );
    } finally {
      index.close();
    }
  });

  test("one giant document plus many small ones still fills the limit", () => {
    // THE FIXTURE THAT IS THE WHOLE POINT OF THE WIDENING FIX.
    //
    // The store returns one hit per document, so a fixed over-fetch of `k`
    // candidates can only ever yield as many documents as those k rows happen
    // to contain. Measured live: 10 requested, **1** returned, with 59 other
    // documents present -- every one of the 200 (and then 800) nearest rows
    // belonged to a single 4,000-chunk document.
    //
    // Built here with document 1 owning a full 4,000 chunks, which is the
    // MAX_CHUNKS default and therefore the real live shape rather than a
    // contrived one. Documents 2..20 are single-chunk.
    const index = openIndex();
    try {
      const giant: StoredChunk[] = [];
      for (let position = 0; position < 4000; position += 1) {
        // All at distance 0 from the query axis, so they occupy the entire top
        // of the ranking and the budget must widen past every one of them.
        giant.push(chunk("content", `g${position}`, 0));
      }
      giant.unshift(chunk("metadata", "Giant document.", 0));
      put(index, 1, giant);
      for (let id = 2; id <= 20; id += 1) {
        put(index, id, [chunk("content", `small ${id}`, 0)]);
      }

      const hits = index.vectorHits(axisVector(0), 10);
      const distinct = new Set(hits.map((hit) => hit.documentId));
      assert.equal(
        distinct.size,
        10,
        `expected 10 distinct documents, got ${distinct.size} -- the giant ` +
          `document consumed the whole candidate budget`
      );
    } finally {
      index.close();
    }
  });

  test("the widening is bounded, so a corpus smaller than the limit terminates", () => {
    // The loop widens until `limit` distinct documents are found OR the pool is
    // exhausted OR the ceiling is hit. A corpus of 3 documents asked for 50 must
    // return 3 and stop -- not loop, not throw.
    const index = openIndex();
    try {
      for (let id = 1; id <= 3; id += 1) put(index, id, [chunk("content", `c${id}`, 0)]);
      const hits = index.vectorHits(axisVector(0), 50);
      assert.equal(new Set(hits.map((hit) => hit.documentId)).size, 3);
    } finally {
      index.close();
    }
  });

  test("an empty embedding is refused before it reaches SQLite", () => {
    const index = openIndex();
    try {
      assert.throws(
        () => index.vectorHits([1, 2, 3], 5),
        /has 3 dimensions; the index stores 768/
      );
    } finally {
      index.close();
    }
  });

  test("an unopened index refuses to operate rather than opening implicitly", () => {
    const index = new SearchIndex(tempIndexPath());
    assert.equal(index.isOpen, false);
    assert.throws(() => index.counts(), /not open/);
  });
});

// ── Persistence and the WAL path ───────────────────────────────────────────

describe("persistence", () => {
  test("data survives a close and reopen", () => {
    const path = tempIndexPath();
    const first = openIndex(path);
    put(first, 1, [chunk("content", "persisted", 0)]);
    first.close();

    const second = openIndex(path);
    try {
      assert.equal(second.textHits("persisted", 5).length, 1);
      assert.equal(second.counts().documents, 1);
    } finally {
      second.close();
    }
  });

  test("the file database runs in WAL mode", () => {
    const path = tempIndexPath();
    const index = openIndex(path);
    try {
      const handle = (index as unknown as { db: Database.Database }).db;
      const mode = handle.pragma("journal_mode", { simple: true });
      assert.equal(String(mode).toLowerCase(), "wal");
      const busy = handle.pragma("busy_timeout", { simple: true });
      assert.ok(Number(busy) >= 1000, "a busy timeout is set so a second writer waits");
    } finally {
      index.close();
    }
  });

  test("reopening twice is idempotent, so a restart does not re-create tables", () => {
    const path = tempIndexPath();
    const first = openIndex(path);
    put(first, 1, [chunk("content", "x", 0)]);
    first.close();
    const second = openIndex(path);
    second.close();
    const third = openIndex(path);
    try {
      assert.equal(third.counts().documents, 1);
    } finally {
      third.close();
    }
  });
});
