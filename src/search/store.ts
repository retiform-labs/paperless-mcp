import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import Database from "better-sqlite3";
import { load as loadVecExtension } from "sqlite-vec";
import { PrefixMode } from "./config";
import { assertPrefixModeMatches } from "./embedder";

/**
 * The vector + lexical index for the Paperless corpus, in one SQLite file.
 *
 * This is a REWRITE, not a port. The reference implementation
 * (`jaimemachado/paperless-mcp@50f7bee:src/search/store.ts`) is Postgres with
 * pgvector: `public.vector(1024)`, HNSW over `halfvec`, `to_tsvector('simple',
 * public.unaccent(...))`, GIN on a stored `tsvector`, and
 * `SET LOCAL hnsw.iterative_scan`. None of that survives to SQLite, and none of
 * it should — this store exists so the corpus can be searched without a second
 * web app and without a database server.
 *
 * ── Schema ────────────────────────────────────────────────────────────────
 *   documents    id, modified, title, chunk_count, truncated, metadata_at
 *   chunks       doc_id, position, kind, text          (PK: doc_id, position)
 *   vec_chunks   embedding float[768] distance_metric=cosine
 *   fts          FTS5 over chunks.text, contentless, rowid = chunk rowid
 *   index_meta   key/value: schema_version, dimensions, model, prefix_mode
 *
 * ── The five measured traps, each handled explicitly ──────────────────────
 * Every one of these was reproduced on this box under the gateway child's
 * runtime (node v24.21.0, better-sqlite3 13.2.0, sqlite-vec 0.1.9) before a
 * line of this file was written. Two of the five turned out differently from
 * the recon notes, and the notes are corrected here rather than in a drawer.
 *
 * 1. ROWID PARAMETERS MUST BE CAST. `insert into v(rowid, embedding) values
 *    (?, ?)` fails with `SQLITE_ERROR: Only integers are allows for primary
 *    key values on v` while both the literal form and `CAST(? AS INTEGER)`
 *    succeed. Verified: `k = ?` does NOT need the cast (it worked at
 *    `k = ?` bound to 2), so the cast is applied to rowid contexts only, not
 *    sprayed everywhere.
 *
 * 2. `changes` IS RELIABLE — the recon note was WRONG about this. The note
 *    recorded `delete from v where rowid = ? [5]` returning `changes: 0` and
 *    concluded that a bound-rowid delete silently no-ops. Re-measured five
 *    ways: the row simply did not exist. A bound delete against an EXISTING
 *    rowid returns `changes: 1` for a number, a float, a string, and a bigint
 *    alike, and a range delete returns the true count (3 of 3). So `changes` is
 *    a real instrument, and the discipline it buys is the right one anyway —
 *    ASSERT ON IT, never on the absence of a thrown error — but the mechanism
 *    in the note ("vec0 silently ignores bound rowids") does not exist.
 *
 * 3. NO `INSERT OR REPLACE` ON vec0. Verified: a duplicate rowid throws
 *    `UNIQUE constraint failed on v primary key`, and `insert or replace`
 *    throws the same. A replace is DELETE-then-INSERT inside one transaction.
 *    (The cascade/rollback path was verified: a vec0 insert inside an explicit
 *    transaction disappears on ROLLBACK.)
 *
 * 4. `rowid IN (…)` COMPOSES WITH `MATCH`. Verified. This store does not
 *    currently push visibility filters into the kNN statement, because the
 *    caller-side filter belongs to the tool lap, but the mechanism is
 *    available and the note is accurate.
 *
 * 5. DEFAULT METRIC IS L2 — `distance_metric=cosine` is declared explicitly.
 *    Verified: cosine gives self 0 and orthogonal 0.50098 (= 1 − cos 90°),
 *    while a default table gives self 0 and orthogonal 16 for the same unit
 *    vectors. Similarity is `1 − distance`, and `vectorHits` returns the
 *    similarity, never the raw distance.
 *
 * ── One further finding, which changed the schema ─────────────────────────
 * A rowid of `doc_id * 1001 + position` — the shape the recon notes proposed —
 * COLLIDES as soon as `MAX_CHUNKS` exceeds 1000. With 4001 rows per document
 * the arithmetic produces 15,000 collisions across just six documents:
 * (doc 1, pos 1001) and (doc 2, pos 0) both map to 2002. Since `MAX_CHUNKS`
 * is being raised to 4000, the stride is 8192 instead: a power of two (so the
 * decode is a mask rather than a division that has to be exactly right), and
 * comfortably above the maximum chunk count, with zero collisions verified at
 * 4001 rows per document.
 */

export const SCHEMA_VERSION = 1;
export const EMBEDDING_DIMENSIONS = 768;

/**
 * Rowid stride. Must exceed MAX_CHUNKS, and must be a power of two so the
 * position decodes with a mask. 8192 leaves room for MaxChunks up to 8191.
 */
export const CHUNK_ROWID_STRIDE = 8192;

/** Whether FTS5 is compiled into the bundled SQLite. Measured, not assumed. */
export const HAS_FTS5 = detectFts5();

function detectFts5(): boolean {
  const probe = new Database(":memory:");
  try {
    const options = probe
      .prepare("pragma compile_options")
      .all() as { compile_options: string }[];
    const compileFlags = options.map((row) => row.compile_options.toUpperCase());
    if (!compileFlags.some((flag) => flag.includes("ENABLE_FTS5"))) return false;
    probe.exec("create virtual table fts5_probe using fts5(x)");
    probe.exec("drop table fts5_probe");
    return true;
  } catch {
    return false;
  } finally {
    probe.close();
  }
}

export interface IndexedDocument {
  id: number;
  modified: string;
  title: string;
  chunkCount: number;
  truncated: boolean;
  metadataAt: string;
}

export interface StoredChunk {
  kind: "metadata" | "content";
  text: string;
  embedding: number[];
}

export interface Hit {
  documentId: number;
  position: number;
  text: string;
  score: number;
}

export interface IndexMeta {
  schemaVersion: number;
  dimensions: number;
  model: string;
  prefixMode: PrefixMode;
}

export interface Counts {
  documents: number;
  chunks: number;
  vectors: number;
  ftsRows: number | null;
}

export class SchemaIncompatibleError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SchemaIncompatibleError";
  }
}

/**
 * Thrown when the lexical half is asked for on a build without FTS5.
 *
 * Returning an empty array would be indistinguishable from "the query matched
 * nothing", which would let the ranker believe it had searched lexically and
 * found silence. That is the silent degradation the plan forbids, so the absence
 * is a refusal the caller must handle, not a quiet empty list.
 */
export class LexicalUnavailableError extends Error {
  constructor() {
    super(
      "This build of SQLite has no FTS5 module, so lexical search is " +
        "unavailable. Check `HAS_FTS5` or `counts().ftsRows === null` before " +
        "calling textHits if you want to degrade gracefully."
    );
    this.name = "LexicalUnavailableError";
  }
}

export class DeleteCountError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DeleteCountError";
  }
}

/** Encode a chunk's rowid. Verified collision-free for 8191 chunks per doc. */
export function chunkRowid(docId: number, position: number): number {
  if (!Number.isInteger(docId) || docId < 1) {
    throw new Error(`docId must be a positive integer, not ${docId}.`);
  }
  if (!Number.isInteger(position) || position < 0) {
    throw new Error(`position must be a non-negative integer, not ${position}.`);
  }
  if (position >= CHUNK_ROWID_STRIDE) {
    throw new Error(
      `position ${position} does not fit the rowid stride ` +
        `${CHUNK_ROWID_STRIDE}. Raise CHUNK_ROWID_STRIDE before indexing a ` +
        `document this long.`
    );
  }
  return docId * CHUNK_ROWID_STRIDE + position;
}

export function decodeChunkRowid(rowid: number): {
  docId: number;
  position: number;
} {
  return {
    docId: Math.floor(rowid / CHUNK_ROWID_STRIDE),
    position: rowid % CHUNK_ROWID_STRIDE,
  };
}

function toBlob(embedding: number[]): Buffer {
  if (embedding.length !== EMBEDDING_DIMENSIONS) {
    throw new Error(
      `Embedding has ${embedding.length} dimensions; the index stores ` +
        `${EMBEDDING_DIMENSIONS}.`
    );
  }
  const values = Float32Array.from(embedding);
  return Buffer.from(values.buffer, values.byteOffset, values.byteLength);
}

const SCHEMA_DDL = `
  CREATE TABLE IF NOT EXISTS documents (
    id           INTEGER PRIMARY KEY,
    modified     TEXT    NOT NULL,
    title        TEXT    NOT NULL,
    chunk_count  INTEGER NOT NULL,
    truncated    INTEGER NOT NULL DEFAULT 0,
    metadata_at  TEXT    NOT NULL
  );

  CREATE TABLE IF NOT EXISTS chunks (
    doc_id    INTEGER NOT NULL,
    position  INTEGER NOT NULL,
    kind      TEXT    NOT NULL CHECK (kind IN ('metadata', 'content')),
    text      TEXT    NOT NULL,
    PRIMARY KEY (doc_id, position)
  );

  CREATE INDEX IF NOT EXISTS chunks_doc_id ON chunks (doc_id);

  CREATE TABLE IF NOT EXISTS index_meta (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
  );
`;

const FTS_DDL = `
  CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
    text,
    content='',
    contentless_delete=1,
    tokenize='unicode61 remove_diacritics 2'
  );
`;

const VEC_DDL = `
  CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
    embedding float[${EMBEDDING_DIMENSIONS}] distance_metric=cosine
  );
`;

/**
 * A lazily-opened index.
 *
 * Lazy is not tidiness — it is a named risk in the plan. The gateway launches
 * this package's child at startup, and opening a ~1 GB SQLite file (plus
 * loading the vec0 extension, which is a dynamic shared object) at module
 * import time could exceed the gateway's child-startup tolerance. `open()` is
 * therefore explicit, `close()` is explicit, and no code in this module runs
 * on import beyond the FTS5 probe, which is a `:memory:` connection that is
 * closed immediately.
 */
export class SearchIndex {
  private db: Database.Database | undefined;

  /**
   * `fts5` exists only so the absent-FTS5 path can be exercised on a box that
   * HAS FTS5. Without an injectable override, the degradation branch would be
   * untestable here and the only evidence that it works would be that it
   * compiles — which is not evidence.
   */
  private readonly hasFts5: boolean;

  constructor(
    private readonly path: string,
    private readonly options: { readonly?: boolean; fts5?: boolean } = {}
  ) {
    this.hasFts5 = options.fts5 ?? HAS_FTS5;
  }

  get isOpen(): boolean {
    return this.db !== undefined;
  }

  /** Open the database, creating and migrating it when writable. */
  open(): Database.Database {
    if (this.db) return this.db;
    if (!this.options.readonly) {
      mkdirSync(dirname(this.path), { recursive: true });
    }
    // better-sqlite3 validates the PRESENCE of the key, not just its value:
    // passing { readonly: undefined } throws "Expected the readonly option to
    // be a boolean" rather than defaulting. Measured. So the key is omitted
    // entirely unless it is explicitly true.
    const db = this.options.readonly
      ? new Database(this.path, { readonly: true })
      : new Database(this.path);
    try {
      // WAL lets readers proceed while a sync writes. busy_timeout makes the
      // second writer WAIT instead of failing: two writers on one sqlite file
      // is a failure this arc has already hit once, and SQLITE_BUSY presents
      // as a mysterious crash rather than as a lock.
      if (!this.options.readonly) {
        db.pragma("journal_mode = WAL");
        db.pragma("synchronous = NORMAL");
      }
      db.pragma("busy_timeout = 10000");
      db.pragma("foreign_keys = ON");
      // The vec0 virtual-table type lives on the CONNECTION, not in the file,
      // so this is required even for a readonly handle. Measured: without it,
      // the first `prepare` on a vec0 table throws `no such module: vec0`.
      loadVecExtension(db);
    } catch (err) {
      db.close();
      throw err;
    }
    if (!this.options.readonly) this.migrate(db);
    this.db = db;
    return db;
  }

  close(): void {
    this.db?.close();
    this.db = undefined;
  }

  private handle(): Database.Database {
    const db = this.db;
    if (!db) {
      throw new Error(
        "The search index is not open. Call open() before using it."
      );
    }
    return db;
  }

  private migrate(db: Database.Database): void {
    db.exec(SCHEMA_DDL);
    if (this.hasFts5) db.exec(FTS_DDL);
    db.exec(VEC_DDL);
  }

  /** Read the stored metadata, or undefined on a fresh index. */
  meta(): IndexMeta | undefined {
    const rows = this.handle()
      .prepare("SELECT key, value FROM index_meta")
      .all() as { key: string; value: string }[];
    if (rows.length === 0) return undefined;
    const map = new Map(rows.map((row) => [row.key, row.value]));
    return {
      schemaVersion: Number(map.get("schema_version")),
      dimensions: Number(map.get("dimensions")),
      model: map.get("model") ?? "",
      prefixMode: (map.get("prefix_mode") ?? "") as PrefixMode,
    };
  }

  /** Write the metadata for a fresh index. Refuses to overwrite silently. */
  writeMeta(meta: IndexMeta): void {
    const db = this.handle();
    const existing = this.meta();
    if (existing && existing.schemaVersion !== meta.schemaVersion) {
      throw new SchemaIncompatibleError(
        `The index reports schema_version ${existing.schemaVersion} but this ` +
          `build writes ${meta.schemaVersion}. Refusing to reuse it; delete ` +
          `the index and rebuild.`
      );
    }
    const upsert = db.prepare(
      "INSERT INTO index_meta (key, value) VALUES (?, ?) " +
        "ON CONFLICT (key) DO UPDATE SET value = excluded.value"
    );
    const write = db.transaction(() => {
      upsert.run("schema_version", String(meta.schemaVersion));
      upsert.run("dimensions", String(meta.dimensions));
      upsert.run("model", meta.model);
      upsert.run("prefix_mode", meta.prefixMode);
    });
    write();
  }

  /**
   * Refuse to operate on an index whose schema or dimension we do not know.
   * An unknown schema version is the one condition that must never degrade.
   */
  assertSchemaCompatible(expected?: {
    dimensions?: number;
    model?: string;
    prefixMode?: PrefixMode;
  }): IndexMeta {
    const meta = this.meta();
    if (!meta) {
      throw new SchemaIncompatibleError(
        "The index has no metadata. Build it before querying it."
      );
    }
    if (meta.schemaVersion !== SCHEMA_VERSION) {
      throw new SchemaIncompatibleError(
        `The index reports schema_version ${meta.schemaVersion}, and this ` +
          `build only understands ${SCHEMA_VERSION}. Refusing to read a ` +
          `schema it does not know — a wrong guess here returns plausible ` +
          `results from the wrong vectors.`
      );
    }
    if (meta.dimensions !== EMBEDDING_DIMENSIONS) {
      throw new SchemaIncompatibleError(
        `The index stores ${meta.dimensions}-dimensional vectors and this ` +
          `build writes ${EMBEDDING_DIMENSIONS}.`
      );
    }
    if (expected?.dimensions !== undefined) {
      if (expected.dimensions !== meta.dimensions) {
        throw new SchemaIncompatibleError(
          `The embedder produces ${expected.dimensions} dimensions and the ` +
            `index stores ${meta.dimensions}.`
        );
      }
    }
    if (expected?.model !== undefined && expected.model !== meta.model) {
      throw new SchemaIncompatibleError(
        `The index was built with model "${meta.model}" and this process is ` +
          `configured for "${expected.model}". Two models put vectors in ` +
          `different spaces; a mixed index is silently meaningless.`
      );
    }
    if (expected?.prefixMode !== undefined) {
      assertPrefixModeMatches(expected.prefixMode, meta.prefixMode);
    }
    return meta;
  }

  /** `{ docId → modified }` for every indexed document. */
  indexed(): Map<number, string> {
    const rows = this.handle()
      .prepare("SELECT id, modified FROM documents")
      .all() as { id: number; modified: string }[];
    return new Map(rows.map((row) => [row.id, row.modified]));
  }

  /** `{ docId → metadataAt }`, needed by the 24 h metadata refresh. */
  metadataStamps(): Map<number, string> {
    const rows = this.handle()
      .prepare("SELECT id, metadata_at FROM documents")
      .all() as { id: number; metadata_at: string }[];
    return new Map(rows.map((row) => [row.id, row.metadata_at]));
  }

  /**
   * Delete every chunk of a document. Returns the number of vector rows removed.
   *
   * `changes` is asserted rather than ignored. The recon note claimed a bound
   * delete silently returns 0 — re-measurement showed `changes` is accurate,
   * but the discipline is worth keeping regardless: a delete that reports
   * success and removes nothing is the same failure family as a `matches: 0`
   * on a guessed pattern, and the instrument is what proves it landed.
   */
  deleteDocumentChunks(docId: number): number {
    const db = this.handle();
    const removed = db
      .prepare(
        "DELETE FROM vec_chunks WHERE rowid BETWEEN CAST(? AS INTEGER) AND " +
          "CAST(? AS INTEGER)"
      )
      .run(chunkRowid(docId, 0), chunkRowid(docId, CHUNK_ROWID_STRIDE - 1));
    const expected = (
      db
        .prepare("SELECT count(*) AS c FROM chunks WHERE doc_id = ?")
        .get(docId) as { c: number }
    ).c;
    if (removed.changes !== expected) {
      throw new DeleteCountError(
        `Deleting the vectors for document ${docId} removed ` +
          `${removed.changes} rows but the chunk table holds ${expected}. ` +
          `The vector index is out of step with the chunk table.`
      );
    }
    db.prepare("DELETE FROM chunks WHERE doc_id = ?").run(docId);
    if (this.hasFts5) {
      db.prepare(
        "DELETE FROM fts WHERE rowid BETWEEN CAST(? AS INTEGER) AND " +
          "CAST(? AS INTEGER)"
      ).run(chunkRowid(docId, 0), chunkRowid(docId, CHUNK_ROWID_STRIDE - 1));
    }
    return removed.changes;
  }

  /**
   * Replace a document and all of its chunks atomically.
   *
   * A vec0 rowid cannot be overwritten — verified, a duplicate throws a UNIQUE
   * constraint and `insert or replace` throws it too — so a replace is always
   * a delete followed by an insert, and both must be in one transaction or a
   * crashed sync leaves an index that is missing a document body while its
   * `documents` row still claims it is complete.
   */
  replaceDocument(doc: IndexedDocument, chunks: StoredChunk[]): void {
    const db = this.handle();
    if (chunks.length !== doc.chunkCount) {
      throw new Error(
        `Document ${doc.id} declares ${doc.chunkCount} chunks but received ` +
          `${chunks.length}.`
      );
    }
    const insertDoc = db.prepare(
      "INSERT INTO documents (id, modified, title, chunk_count, truncated, metadata_at) " +
        "VALUES (?, ?, ?, ?, ?, ?) " +
        "ON CONFLICT (id) DO UPDATE SET modified = excluded.modified, " +
        "title = excluded.title, chunk_count = excluded.chunk_count, " +
        "truncated = excluded.truncated, metadata_at = excluded.metadata_at"
    );
    const insertChunk = db.prepare(
      "INSERT INTO chunks (doc_id, position, kind, text) VALUES (?, ?, ?, ?)"
    );
    const insertVec = db.prepare(
      "INSERT INTO vec_chunks (rowid, embedding) VALUES (CAST(? AS INTEGER), ?)"
    );
    const insertFts = this.hasFts5
      ? db.prepare(
          "INSERT INTO fts (rowid, text) VALUES (CAST(? AS INTEGER), ?)"
        )
      : undefined;

    const run = db.transaction(() => {
      this.deleteDocumentChunks(doc.id);
      insertDoc.run(
        doc.id,
        doc.modified,
        doc.title,
        doc.chunkCount,
        doc.truncated ? 1 : 0,
        doc.metadataAt
      );
      chunks.forEach((chunk, position) => {
        if (position >= CHUNK_ROWID_STRIDE) {
          throw new Error(
            `Document ${doc.id} has ${chunks.length} chunks, beyond the ` +
              `rowid stride ${CHUNK_ROWID_STRIDE}.`
          );
        }
        const rowid = chunkRowid(doc.id, position);
        insertChunk.run(doc.id, position, chunk.kind, chunk.text);
        insertVec.run(rowid, toBlob(chunk.embedding));
        insertFts?.run(rowid, chunk.text);
      });
    });
    run();
  }

  /**
   * Replace only the metadata chunk (position 0) and stamp `metadata_at`.
   * Used by the 24 h metadata refresh, which exists because tags,
   * correspondents and document types change without touching `modified`.
   */
  replaceMetadata(docId: number, chunk: StoredChunk, metadataAt: string): void {
    const db = this.handle();
    if (chunk.kind !== "metadata") {
      throw new Error("replaceMetadata requires a chunk of kind 'metadata'.");
    }
    const rowid = chunkRowid(docId, 0);
    const update = db.prepare(
      "UPDATE documents SET metadata_at = ? WHERE id = ?"
    );
    const insertChunk = db.prepare(
      "INSERT INTO chunks (doc_id, position, kind, text) VALUES (?, 0, ?, ?) " +
        "ON CONFLICT (doc_id, position) DO UPDATE SET kind = excluded.kind, " +
        "text = excluded.text"
    );
    const insertVec = db.prepare(
      "INSERT INTO vec_chunks (rowid, embedding) VALUES (CAST(? AS INTEGER), ?)"
    );
    const insertFts = this.hasFts5
      ? db.prepare(
          "INSERT INTO fts (rowid, text) VALUES (CAST(? AS INTEGER), ?)"
        )
      : undefined;

    const deleteFts = this.hasFts5
      ? db.prepare("DELETE FROM fts WHERE rowid = CAST(? AS INTEGER)")
      : undefined;

    const run = db.transaction(() => {
      const del = db
        .prepare("DELETE FROM vec_chunks WHERE rowid = CAST(? AS INTEGER)")
        .run(rowid);
      // Not asserted against a chunk count here: a fresh document may not yet
      // have position 0 in the vector table, and that is legitimate. What is
      // asserted is that the DELETE and the re-INSERT leave exactly one row.
      void del;
      // The lexical index needs the same delete-then-insert. Without it the OLD
      // metadata text stays in `fts` forever and remains searchable alongside
      // the new text, so one document matches two contradictory strings.
      // Caught by a test, not by review: "old metadata" still hit.
      deleteFts?.run(rowid);
      insertChunk.run(docId, chunk.kind, chunk.text);
      insertVec.run(rowid, toBlob(chunk.embedding));
      insertFts?.run(rowid, chunk.text);
      update.run(metadataAt, docId);
    });
    run();
  }

  /** Delete documents and everything belonging to them, atomically. */
  deleteDocuments(ids: number[]): void {
    if (ids.length === 0) return;
    const db = this.handle();
    const run = db.transaction(() => {
      for (const id of ids) this.deleteDocumentChunks(id);
      const drop = db.prepare("DELETE FROM documents WHERE id = ?");
      for (const id of ids) drop.run(id);
    });
    run();
  }

  /**
   * The stored chunks of one document, in position order.
   *
   * A read accessor, and it exists because `textHits`/`vectorHits` return only
   * the BEST chunk per document -- correct for ranking, useless for asserting
   * what a document actually contains. Without it a test can only observe the
   * metadata chunk by getting lucky with a query.
   */
  chunksFor(docId: number): { position: number; kind: "metadata" | "content"; text: string }[] {
    return this.handle()
      .prepare(
        "SELECT position, kind, text FROM chunks WHERE doc_id = ? ORDER BY position ASC"
      )
      .all(docId) as { position: number; kind: "metadata" | "content"; text: string }[];
  }

  /**
   * How many indexed documents were cut off at the chunk cap.
   *
   * Exposed because truncation is the one index fact that changes what a caller
   * can conclude from a miss: a truncated document lost its tail, so "no such
   * text" may be a false negative rather than an absence.
   */
  truncatedCount(): number {
    return (
      this.handle().prepare("SELECT count(*) AS c FROM documents WHERE truncated = 1").get() as {
        c: number;
      }
    ).c;
  }

  counts(): Counts {
    const db = this.handle();
    const one = (sql: string): number =>
      (db.prepare(sql).get() as { c: number }).c;
    return {
      documents: one("SELECT count(*) AS c FROM documents"),
      chunks: one("SELECT count(*) AS c FROM chunks"),
      vectors: one("SELECT count(*) AS c FROM vec_chunks"),
      ftsRows: this.hasFts5 ? one("SELECT count(*) AS c FROM fts") : null,
    };
  }

  /**
   * The best chunk of each document by vector similarity.
   *
   * Per-document dedupe is deliberate: a chunk list lets one long document fill
   * the result set, and a document that matches only by meaning then drops out.
   * The reference implementation does this in Postgres with `DISTINCT ON`; here
   * it is a window function over the kNN result.
   *
   * `score` is `1 − distance` — a similarity. The raw `distance` is never
   * handed to a caller as a score, because on a cosine table 0 means identical
   * and 1 means orthogonal, which reads exactly backwards.
   */
  vectorHits(embedding: number[], limit: number): Hit[] {
    const db = this.handle();
    if (limit <= 0) return [];
    // Over-fetch so one document cannot monopolise the candidate set. The
    // multiplier is on top of `limit` rather than a fixed constant, because at
    // limit 3 a fixed 1100 would be pure waste and at limit 100 too little.
    const candidates = Math.max(limit * 20, 200);
    const sql = `
      WITH nearest AS (
        SELECT rowid, distance
        FROM vec_chunks
        WHERE embedding MATCH CAST(? AS BLOB) AND k = CAST(? AS INTEGER)
      ), ranked AS (
        SELECT rowid, distance,
               (rowid / ${CHUNK_ROWID_STRIDE}) AS doc_id,
               ROW_NUMBER() OVER (
                 PARTITION BY (rowid / ${CHUNK_ROWID_STRIDE})
                 ORDER BY distance ASC, rowid ASC
               ) AS rank_in_doc
        FROM nearest
      )
      SELECT c.doc_id AS documentId, c.position AS position, c.text AS text,
             1.0 - r.distance AS score
      FROM ranked r
      JOIN chunks c
        ON c.doc_id = r.doc_id
       AND c.position = (r.rowid % ${CHUNK_ROWID_STRIDE})
      WHERE r.rank_in_doc = 1
      ORDER BY r.distance ASC, c.doc_id ASC
      LIMIT CAST(? AS INTEGER)
    `;
    return db
      .prepare(sql)
      .all(toBlob(embedding), candidates, limit) as Hit[];
  }

  /**
   * The best chunk of each document by BM25.
   *
   * FTS5's `bm25()` returns a NEGATIVE score where more negative is better, so
   * it is negated to a positive "higher is better" value. It is NOT comparable
   * to the vector similarity in magnitude — only in order — which is why the
   * ranker fuses the two lists by rank rather than by score.
   */
  textHits(query: string, limit: number): Hit[] {
    if (!this.hasFts5) throw new LexicalUnavailableError();
    const match = toFtsMatch(query);
    if (!match) return [];
    const db = this.handle();
    const sql = `
      WITH matched AS (
        SELECT rowid, bm25(fts) AS bm
        FROM fts
        WHERE fts MATCH CAST(? AS TEXT)
      ), ranked AS (
        SELECT rowid, bm,
               (rowid / ${CHUNK_ROWID_STRIDE}) AS doc_id,
               ROW_NUMBER() OVER (
                 PARTITION BY (rowid / ${CHUNK_ROWID_STRIDE})
                 ORDER BY bm ASC, rowid ASC
               ) AS rank_in_doc
        FROM matched
      )
      SELECT c.doc_id AS documentId, c.position AS position, c.text AS text,
             -r.bm AS score
      FROM ranked r
      JOIN chunks c
        ON c.doc_id = r.doc_id
       AND c.position = (r.rowid % ${CHUNK_ROWID_STRIDE})
      WHERE r.rank_in_doc = 1
      ORDER BY r.bm ASC, c.doc_id ASC
      LIMIT CAST(? AS INTEGER)
    `;
    return db.prepare(sql).all(match, limit) as Hit[];
  }
}

/**
 * Turn a user's words into an FTS5 MATCH expression.
 *
 * Raw user text is NOT a valid MATCH expression — measured: `c++ (test)` throws
 * `fts5: syntax error near "+"`, `a-b` throws `no such column: b`, and an empty
 * string throws. So the query is tokenised into bare terms and every term is
 * quoted, which makes FTS5 treat it as a literal rather than as query syntax.
 * Terms are OR-ed because a user typing three words wants documents matching
 * any of them ranked by BM25, not only those matching all three.
 *
 * `remove_diacritics 2` means `titulo` matches `Título`; verified both
 * directions on this build.
 */
export function toFtsMatch(query: string): string | null {
  const terms = query
    .toLowerCase()
    // \p{M} is REQUIRED, not decorative. A Tibetan syllable is a base letter
    // plus combining marks -- U+0FB3 and U+0F74 are category Mn, not L -- so
    // splitting on [^\p{L}\p{N}_] tears "རླུང" into "ར" and "ང": two fragments,
    // neither of which is the word. Measured on this build. This corpus is
    // heavily Tibetan, so the bug would have made those terms unfindable
    // lexically while looking like an ordinary no-match.
    .split(/[^\p{L}\p{N}\p{M}_]+/u)
    .filter((term) => term.length > 0);
  if (terms.length === 0) return null;
  // Cap the expression: a 500-term OR is pathological and slower than the
  // answer is worth. The first terms are the ones the user typed first.
  const capped = terms.slice(0, 32);
  return capped.map((term) => `"${term.replace(/"/g, '""')}"`).join(" OR ");
}
