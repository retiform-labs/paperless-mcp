// Incremental sync between the Paperless corpus and the local vector index.
//
// Algorithm ported from jaimemachado/paperless-mcp (src/search/sync.ts at
// 50f7beec0d36804cab8fbcaa7b47f85e562e8061). The algorithm is theirs; the
// bindings are not. Every adaptation is listed here because several of them are
// corrections rather than translations, and a silent one would be a defect:
//
//   1. `../metrics` and `../tracing` are NOT ported. Neither module exists in
//      this fork (the store lap made the same call), and adding OpenTelemetry
//      and prom-client to emit counters nobody reads is not a trade worth
//      making. Console diagnostics replace the two console.error calls the
//      reference already used.
//   2. `EmbeddingUnavailable` -> `EmbeddingUnavailableError`, and the abort is
//      preserved: an embedding outage stops the run rather than writing an
//      index that is missing documents it will claim to have.
//   3. The reference's `replaceDocument` call omits `chunkCount`, because its
//      Postgres store derived the count. Ours REQUIRES it and rejects a
//      mismatch against the chunk array, so it is passed explicitly.
//   4. The metadata-refresh loop in the reference reads `info.metadataAt` off
//      `indexed()`. Our `indexed()` returns `id -> modified` only, so the
//      stamp comes from `metadataStamps()`. Using the reference's shape here
//      would have compared `undefined` on every document and refreshed the
//      entire corpus every single run -- a correct-looking port that redoes
//      58 minutes of work on each call.
//   5. `pathAfterApi` is defensive. The reference slices at
//      `indexOf("/api/") + 4`, which silently eats four characters when the
//      marker is absent. Ours returns the path unchanged in that case instead
//      of repairing a URL into a wrong one.
//   6. `MAX_CHUNKS` is injected rather than assumed, so the sync and the
//      chunker cannot disagree about how much of a document was kept.
//
// ── The five load-bearing choices, each verified against our install ────────
//
//   `modified` IS the incremental key. Verified on this install: the documents
//   table has a `modified` column and `max(modified)` is today. A document
//   edited in Paperless bumps it; a document whose tags changed does NOT.
//
//   DELETES ONLY RUN WHEN `rows.length === count`. A list whose rows disagree
//   with its own count changed while it was being paged, and the missing rows
//   are indistinguishable from deleted documents. Deleting on a short list
//   would remove live documents from the index, so the delete step is SKIPPED
//   and reported instead. Tested.
//
//   A 24 HOUR METADATA REFRESH. Tags, correspondents and document types change
//   without touching `modified`, so a document whose sender was corrected
//   would keep the old name in its metadata chunk forever. That chunk is
//   vector- and lexical-indexed, so the stale name stays searchable.
//
//   AN EMBEDDING OUTAGE ABORTS THE WHOLE RUN. The alternative -- skip the
//   document, carry on, report `failed` -- leaves an index that is missing
//   documents while `documents` rows claim they are indexed, which is worse
//   than no index because it looks complete.
//
//   A RE-ENTRANCY GUARD. Two concurrent syncs against one SQLite file is a
//   failure this arc has already hit once. `run()` returns the in-flight
//   promise rather than starting a second pass.
//
// ── Triggering ─────────────────────────────────────────────────────────────
// On demand is the primary path: the `sync_embeddings` tool. `start()` exists
// but nothing calls it unless `PAPERLESS_MCP_SYNC_INTERVAL_MS` is set, because
// a stdio child's process lifetime belongs to the gateway -- a hidden timer
// refreshes on a schedule nobody can see or schedule around. An observable tool
// call can be scheduled with the scheduler we already use, where the schedule
// is legible.

import { PaperlessAPI } from "../api/PaperlessAPI";
import { chunkText, metadataText, MAX_CHUNKS } from "./chunk";
import { EmbeddingUnavailableError, type Embedder } from "./embedder";
import { SCHEMA_VERSION, SearchIndex, type StoredChunk } from "./store";

/** How often a document's metadata chunk is rebuilt regardless of `modified`. */
export const METADATA_REFRESH_MS = 24 * 60 * 60 * 1000;

/**
 * Paged in id order, with the fields the diff needs and nothing else.
 *
 * Ordering by id matters: an insert does not move a row, so a document added
 * while the list is being paged lands at the end rather than shuffling the
 * pages already read. A delete DOES move rows, which is why the row count is
 * compared to the reported count below.
 */
export const DOCUMENT_LIST_PATH =
  "/documents/?fields=id,modified&page_size=1000&ordering=id";

/** A repeated or endless `next` would otherwise keep a run pending forever. */
export const MAX_LIST_PAGES = 10_000;

/** Without a limit, a Paperless that never answers keeps a run pending forever. */
export const REQUEST_TIMEOUT_MS = 60_000;

export interface SyncReport {
  added: number;
  updated: number;
  deleted: number;
  metadataRefreshed: number;
  failed: number;
  /** True when the delete step was skipped because the list changed mid-run. */
  deletedSkipped: boolean;
}

interface NameMaps {
  correspondents: Map<number, string>;
  documentTypes: Map<number, string>;
  tags: Map<number, string>;
}

interface DocumentDetail {
  title: string;
  content: string | null;
  created: string;
  modified: string;
  correspondent: number | null;
  document_type: number | null;
  tags: number[];
}

interface PaperlessPage<T> {
  count: number;
  next: string | null;
  results: T[];
}

/**
 * Turn an absolute `next` URL into the path this API client accepts.
 *
 * The list answer carries `next` as a full URL (verified: `GET
 * /api/documents/?fields=id,modified&page_size=3&ordering=id` returns an
 * absolute URL), while `PaperlessAPI.request` prepends `/api` itself. So the
 * base is stripped and the query string kept.
 *
 * Deliberately NOT the reference's arithmetic: it slices at
 * `indexOf("/api/") + 4`, which returns a four-character-amputated path when
 * the marker is missing -- a wrong URL that looks like a right one. An absent
 * marker returns the input unchanged, which fails loudly at the request instead.
 */
export function pathAfterApi(url: string): string {
  const { pathname, search } = new URL(url);
  const marker = pathname.indexOf("/api/");
  if (marker < 0) return pathname + search;
  return pathname.slice(marker + "/api".length) + search;
}

function toNameMap(rows: { id: number; name: string }[]): Map<number, string> {
  return new Map(rows.map((row) => [row.id, row.name]));
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function log(line: string): void {
  console.error(`[paperless-mcp] index: ${line}`);
}

export class IndexSync {
  private readonly api: PaperlessAPI;
  private readonly store: SearchIndex;
  private readonly embedder: Embedder;
  private readonly maxChunks: number;
  private readonly now: () => number;
  private readonly requestTimeoutMs: number;
  private readonly maxListPages: number;
  private timer?: ReturnType<typeof setInterval>;
  private running?: Promise<SyncReport>;

  constructor(opts: {
    api: PaperlessAPI;
    store: SearchIndex;
    embedder: Embedder;
    maxChunks?: number;
    now?: () => number;
    requestTimeoutMs?: number;
    maxListPages?: number;
  }) {
    this.api = opts.api;
    this.store = opts.store;
    this.embedder = opts.embedder;
    this.maxChunks = opts.maxChunks ?? MAX_CHUNKS;
    this.now = opts.now ?? Date.now;
    this.requestTimeoutMs = opts.requestTimeoutMs ?? REQUEST_TIMEOUT_MS;
    this.maxListPages = opts.maxListPages ?? MAX_LIST_PAGES;
  }

  /**
   * Run one sync. Re-entrant calls join the run in flight.
   *
   * Two concurrent writers on one SQLite file is a failure this arc has already
   * hit once, and WAL plus `busy_timeout` only makes the second writer wait --
   * it does not make two simultaneous passes correct, because both would diff
   * against the same stale `indexed()` snapshot and the loser's deletes would
   * remove the winner's work.
   */
  run(): Promise<SyncReport> {
    if (!this.running) {
      this.running = this.doRun().finally(() => {
        this.running = undefined;
      });
    }
    return this.running;
  }

  /**
   * Run once and then on an interval. NOT called by this package.
   *
   * A stdio child's lifetime is the gateway's, so a timer here refreshes on a
   * schedule nobody can see. The observable path is the `sync_embeddings` tool,
   * which can be scheduled where the schedule is legible.
   */
  start(intervalMs: number): void {
    if (this.timer) return;
    const scheduled = () =>
      this.run().catch((err) => log(`scheduled run failed: ${errorMessage(err)}`));
    void scheduled();
    this.timer = setInterval(scheduled, intervalMs);
    // Do not hold the process open for a refresh nobody asked for.
    this.timer.unref?.();
  }

  stop(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = undefined;
  }

  get isRunning(): boolean {
    return this.running !== undefined;
  }

  private async doRun(): Promise<SyncReport> {
    const started = this.now();
    try {
      const report = await this.syncAll();
      const seconds = ((this.now() - started) / 1000).toFixed(1);
      log(
        `run finished in ${seconds}s: +${report.added} added, ` +
          `~${report.updated} updated, -${report.deleted} deleted, ` +
          `${report.metadataRefreshed} metadata refreshed, ${report.failed} failed` +
          (report.deletedSkipped ? " (deletes SKIPPED: list changed mid-run)" : "")
      );
      return report;
    } catch (err) {
      log(`run failed: ${errorMessage(err)}`);
      throw err;
    }
  }

  private async syncAll(): Promise<SyncReport> {
    const report: SyncReport = {
      added: 0,
      updated: 0,
      deleted: 0,
      metadataRefreshed: 0,
      failed: 0,
      deletedSkipped: false,
    };

    // ⚠️ THE INDEX'S IDENTITY IS WRITTEN BY THE RUN, NOT BY THE FIXTURE.
    //
    // `writeMeta` and `assertSchemaCompatible` both existed and were both dead:
    // no production call site for either, so `embedding_status` reported
    // `dimensions: 0, model: "", prefix_mode: ""` on a real index, and a
    // prefix-mode mismatch could never be refused. The suite stayed green
    // because the FIXTURES called `writeMeta` themselves -- the tool asserted
    // the function while nothing asserted the wiring, which is the
    // no-op-stub-in-a-test-fake family: a contract leg with no observable
    // output and therefore no test.
    //
    // Written FIRST, before any document work, so the index never holds vectors
    // whose provenance it cannot state. It is also cheap: four upserts.
    this.store.writeMeta({
      schemaVersion: SCHEMA_VERSION,
      dimensions: this.embedder.dimensions,
      model: this.embedder.model,
      prefixMode: this.embedder.prefixMode,
    });

    const remote = new Map<number, string>();
    const { rows, count } = await this.listAll<{ id: number; modified: string }>(
      DOCUMENT_LIST_PATH
    );
    for (const row of rows) remote.set(row.id, row.modified);

    const indexed = this.store.indexed();

    // A list whose rows disagree with its own count changed while it was being
    // paged. A missing row is then indistinguishable from a deleted document,
    // so the delete step is skipped rather than trusted.
    const listComplete = rows.length === count;
    report.deletedSkipped = !listComplete;
    if (!listComplete) {
      log(
        `the list returned ${rows.length} documents but Paperless counts ` +
          `${count}; the delete step is skipped`
      );
    }

    if (listComplete) {
      const toDelete = [...indexed.keys()].filter((id) => !remote.has(id));
      if (toDelete.length > 0) {
        this.store.deleteDocuments(toDelete);
        report.deleted = toDelete.length;
        log(`deleted ${toDelete.length} documents: ${toDelete.join(",")}`);
      }
    }

    const toSync = [...remote.entries()].filter(([id, modified]) => {
      const stored = indexed.get(id);
      if (stored === undefined) return true;
      // Compared as instants, not strings: Paperless may re-serialise the same
      // moment with a different offset, and a string compare would then reindex
      // the whole corpus on every run.
      return new Date(modified).getTime() !== new Date(stored).getTime();
    });

    let names: NameMaps | undefined;
    const reindexed = new Set<number>();

    for (const [id] of toSync) {
      try {
        names ??= await this.loadNames();
        await this.syncDocument(id, names);
        reindexed.add(id);
        if (indexed.has(id)) report.updated++;
        else report.added++;
      } catch (err) {
        if (err instanceof EmbeddingUnavailableError) {
          log("the embedding server is unavailable, stopping the run");
          throw err;
        }
        report.failed++;
        log(`document ${id} failed: ${errorMessage(err)}`);
      }
    }

    // The 24 h metadata refresh. Tags, correspondents and document types change
    // without touching `modified`, so without this a corrected sender stays in
    // the metadata chunk -- which is indexed both ways -- forever.
    const stamps = this.store.metadataStamps();
    const refreshBefore = this.now() - METADATA_REFRESH_MS;
    for (const id of indexed.keys()) {
      if (reindexed.has(id) || !remote.has(id)) continue;
      const metadataAt = stamps.get(id);
      if (metadataAt && new Date(metadataAt).getTime() >= refreshBefore) continue;
      try {
        names ??= await this.loadNames();
        await this.refreshMetadata(id, names);
        report.metadataRefreshed++;
      } catch (err) {
        if (err instanceof EmbeddingUnavailableError) {
          log("the embedding server is unavailable, stopping the run");
          throw err;
        }
        report.failed++;
        log(`document ${id} metadata failed: ${errorMessage(err)}`);
      }
    }

    return report;
  }

  /**
   * Page through a list endpoint, following `next`.
   *
   * A repeated `next` would loop forever and an endless one would never finish,
   * so pages already fetched are tracked and the run fails past a page cap
   * rather than hanging with no diagnostic.
   */
  private async listAll<T>(path: string): Promise<{ rows: T[]; count: number }> {
    const rows: T[] = [];
    const seen = new Set<string>();
    let count = 0;
    let next: string | null = path;
    while (next) {
      if (seen.has(next)) {
        throw new Error(`Paperless returned the same next page twice: ${next}`);
      }
      if (seen.size >= this.maxListPages) {
        throw new Error(
          `Paperless returned more than ${this.maxListPages} pages for ${path}`
        );
      }
      seen.add(next);
      const page = await this.get<PaperlessPage<T>>(next);
      if (seen.size === 1) count = page.count;
      rows.push(...(page.results ?? []));
      next = page.next ? pathAfterApi(page.next) : null;
    }
    return { rows, count };
  }

  /**
   * One request with a timeout.
   *
   * `PaperlessAPI.request` passes `options.signal` through to axios, which is
   * what makes the budget real: before that was wired, this signal was accepted
   * and silently ignored, so a Paperless that accepted the connection and then
   * stalled would hold the sync open forever while the code read as though it
   * had a 60 second limit.
   */
  private get<T>(path: string): Promise<T> {
    return this.api.request<T>(path, {
      signal: AbortSignal.timeout(this.requestTimeoutMs),
    });
  }

  private async loadNames(): Promise<NameMaps> {
    const [correspondents, documentTypes, tags] = await Promise.all([
      this.listAll<{ id: number; name: string }>("/correspondents/?page_size=1000"),
      this.listAll<{ id: number; name: string }>("/document_types/?page_size=1000"),
      this.listAll<{ id: number; name: string }>("/tags/?page_size=1000"),
    ]);
    return {
      correspondents: toNameMap(correspondents.rows),
      documentTypes: toNameMap(documentTypes.rows),
      tags: toNameMap(tags.rows),
    };
  }

  private metadataTextFor(doc: DocumentDetail, names: NameMaps): string {
    const tags = doc.tags
      .map((id) => names.tags.get(id))
      .filter((name): name is string => !!name);
    return metadataText(
      { title: doc.title, created: doc.created },
      {
        correspondent: doc.correspondent
          ? names.correspondents.get(doc.correspondent)
          : undefined,
        documentType: doc.document_type
          ? names.documentTypes.get(doc.document_type)
          : undefined,
        tags,
      }
    );
  }

  private async syncDocument(id: number, names: NameMaps): Promise<void> {
    const doc = await this.get<DocumentDetail>(`/documents/${id}/`);
    const metaText = this.metadataTextFor(doc, names);
    const { chunks, truncated } = chunkText(doc.content ?? "", this.maxChunks);
    const texts = [metaText, ...chunks];
    const embeddings = await this.embedder.embedDocuments(texts);

    const storedChunks: StoredChunk[] = embeddings.map((embedding, i) => ({
      kind: i === 0 ? "metadata" : "content",
      text: texts[i],
      embedding,
    }));

    this.store.replaceDocument(
      {
        id,
        modified: doc.modified,
        title: doc.title,
        // Our store rejects a chunk count that does not match the array it was
        // handed, which is the assertion that catches a mis-assembled document
        // before it reaches the vector table. The reference store derived this
        // and so never passed it.
        chunkCount: storedChunks.length,
        truncated,
        metadataAt: new Date(this.now()).toISOString(),
      },
      storedChunks
    );
  }

  private async refreshMetadata(id: number, names: NameMaps): Promise<void> {
    const doc = await this.get<DocumentDetail>(`/documents/${id}/`);
    const metaText = this.metadataTextFor(doc, names);
    const [embedding] = await this.embedder.embedDocuments([metaText]);
    this.store.replaceMetadata(
      id,
      { kind: "metadata", text: metaText, embedding },
      new Date(this.now()).toISOString()
    );
  }
}
