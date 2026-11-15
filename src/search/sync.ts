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
//   ⚠️ THAT REFRESH SURVIVES THE WATERMARK, AND IT IS NOT REDUNDANT WITH IT.
//   Measured against Paperless' own bulk-edit code: `set_correspondent`,
//   `set_document_type`, `set_storage_path` and `add_tag` all go through
//   `qs.update(...)` or a `bulk_create` on the through-table. Neither fires
//   `auto_now`, so none of them bumps `modified`. A corpus-level watermark is
//   therefore structurally blind to an M2M edit — it cannot see one even in
//   principle, let alone in this implementation. Deleting the refresh on the
//   theory that the watermark subsumes it would re-open the stale-metadata hole
//   it was written to close. It is a candidate for WIDENING (a cheaper M2M
//   probe), never for removal.
//
//   THE WATERMARK. `count` and `max(modified)` are recorded at the end of every
//   completed run, and consulted at the start of the next one as a single
//   request. A run against an unchanged corpus returns in ~218 ms without
//   paging a single document. `force` bypasses it, because `force` is a
//   statement about the stored vectors rather than about the corpus.
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
import {
  SCHEMA_VERSION,
  SearchIndex,
  type StoredChunk,
  type Watermark,
} from "./store";

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

/**
 * One request that answers "how big is the corpus, and when did it last move".
 *
 * Deliberately NOT a walk of the document list. The list is paged at 1000 rows
 * with a full `id,modified` payload per row; on a corpus of a few thousand that
 * is several requests and a few hundred kilobytes to learn two scalars. This is
 * ONE request and about 200 bytes.
 *
 * `ordering=-modified&page_size=1` makes `results[0].modified` the maximum over
 * the whole corpus, which is the value the incremental diff is keyed on.
 */
export const WATERMARK_PATH =
  "/documents/?fields=id,modified&ordering=-modified&page_size=1";

/** The two scalars, and nothing else. */
export interface CorpusState {
  count: number;
  /** `max(modified)` over the corpus, or null when the corpus is empty. */
  modified: string | null;
}

/**
 * Whether the live corpus differs from the indexed one.
 *
 * ⚠️ COMPARED AS INSTANTS, NOT STRINGS. `modified` arrives as an ISO 8601 string
 * with an offset, and the same moment can be serialised with a different offset
 * (`…+00:00` one day, `…-05:00` the next). A string compare would report a
 * change on every run and defeat the entire point of the watermark, while
 * looking exactly like a working check. The existing document diff already got
 * this right; this is the same treatment applied to the corpus-level value.
 *
 * `count` is checked FIRST and separately. It is the only signal a soft delete
 * produces: Paperless clears `deleted_at` through a queryset `.update()`, which
 * does not fire `auto_now`, so a deleted document never bumps `max(modified)`.
 * The count still drops, and the count is free in the same response.
 */
export function corpusChanged(live: CorpusState, watermark: Watermark): boolean {
  if (live.count !== watermark.count) return true;
  if (live.modified === null) return watermark.modified !== null;
  if (watermark.modified === null) return true;
  return new Date(live.modified).getTime() > new Date(watermark.modified).getTime();
}

/**
 * The corpus's size and newest `modified`, from ONE request.
 *
 * Returns undefined when the request fails. A watermark probe that THROWS would
 * turn a cosmetic optimisation into a hard dependency: the sync would stop
 * working on any Paperless that declines this particular query, when the list
 * walk it already knows how to do would have succeeded. Falling back to the full
 * diff is always correct, merely slower.
 *
 * Module-level rather than a private method because two callers need it — the
 * sync's skip check and the search tool's opt-in staleness report — and two
 * copies of "which fields, which ordering, which page size" is how they drift
 * apart. One request shape, one place.
 */
export async function readCorpusState(
  api: PaperlessAPI,
  requestTimeoutMs?: number
): Promise<CorpusState | undefined> {
  try {
    const page = await api.request<PaperlessPage<{ id: number; modified: string }>>(
      WATERMARK_PATH,
      requestTimeoutMs !== undefined
        ? { signal: AbortSignal.timeout(requestTimeoutMs) }
        : {}
    );
    const first = (page.results ?? [])[0];
    return { count: page.count, modified: first?.modified ?? null };
  } catch (err) {
    log(`the corpus watermark could not be read: ${errorMessage(err)}`);
    return undefined;
  }
}

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
  /**
   * Text splits performed because the embedding server refused an input as
   * over-long, summed over the run.
   *
   * Reported because the alternative is silence: a run that split 400 chunks
   * and one that split none produce identical counts, and the number is the
   * only evidence the fallback is load-bearing rather than merely present.
   */
  splits: number;
  /**
   * True when the run returned without touching the index because the live
   * corpus matched the recorded watermark.
   *
   * Reported rather than left implicit: a skipped run and a run that found
   * nothing to do produce identical counts (all zero), and the difference is
   * the whole reason the watermark exists. A caller that cannot tell them apart
   * cannot tell a 218 ms no-op from a wasted full sweep.
   */
  skipped: boolean;
  /** Why the run skipped, when it did. Absent otherwise. */
  skipReason?: string;
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

/**
 * A failure that knows which document produced it.
 *
 * Attached to the error rather than wrapping it, because the class identity is
 * what the caller above checks: `EmbeddingUnavailableError` must still be
 * recognised after the context is added, or the abort-on-outage path would stop
 * aborting. A wrapper would have to copy the prototype to keep that true, and
 * a context object on the error does it without touching the type.
 */
interface DocumentContext {
  documentId: number;
  documentTitle: string;
}

function withDocumentContext<T>(err: T, context: DocumentContext): T {
  if (err && typeof err === "object") {
    Object.assign(err as object, context);
  }
  return err;
}

/**
 * Name the document in one line, so a failure is diagnosable from the log
 * without a second round trip.
 *
 * Id first, because the id is what a person pastes into the Paperless URL; the
 * title second, because it is what makes the line recognisable at a glance. A
 * title is a document HEADING, never body text, so it is safe to log.
 */
function describeDocument(err: unknown, fallbackId: number): string {
  const id =
    err && typeof err === "object" && "documentId" in err
      ? (err as DocumentContext).documentId
      : fallbackId;
  const title =
    err && typeof err === "object" && "documentTitle" in err
      ? (err as DocumentContext).documentTitle
      : undefined;
  if (!title) return `${id} (title unavailable: the document could not be read)`;
  return `${id} \u201c${title}\u201d`;
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
  run(opts: { force?: boolean } = {}): Promise<SyncReport> {
    if (!this.running) {
      this.running = this.doRun(opts).finally(() => {
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

  /**
   * True while a background timer is live.
   *
   * Distinct from `isRunning`, which reports a pass in flight: between two
   * scheduled ticks the timer is set and nothing is running. A wiring test that
   * asked `isRunning` would report "no schedule" on a working schedule whenever
   * it happened to look between ticks.
   */
  get isScheduled(): boolean {
    return this.timer !== undefined;
  }

  private async doRun(opts: { force?: boolean }): Promise<SyncReport> {
    const started = this.now();
    try {
      const report = await this.syncAll(opts);
      const seconds = ((this.now() - started) / 1000).toFixed(1);
      if (report.skipped) {
        log(`run finished in ${seconds}s: SKIPPED (${report.skipReason})`);
        return report;
      }
      log(
        `run finished in ${seconds}s: +${report.added} added, ` +
          `~${report.updated} updated, -${report.deleted} deleted, ` +
          `${report.metadataRefreshed} metadata refreshed, ${report.failed} failed, ` +
          `${report.splits} split` +
          (report.deletedSkipped ? " (deletes SKIPPED: list changed mid-run)" : "")
      );
      return report;
    } catch (err) {
      log(`run failed: ${errorMessage(err)}`);
      throw err;
    }
  }

  private async syncAll(opts: { force?: boolean }): Promise<SyncReport> {
    const report: SyncReport = {
      added: 0,
      updated: 0,
      deleted: 0,
      metadataRefreshed: 0,
      failed: 0,
      deletedSkipped: false,
      splits: 0,
      skipped: false,
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

    // ── THE WATERMARK, CONSULTED BEFORE ANY PAGING ──────────────────────────
    //
    // One request, two scalars. If the corpus has not moved since the last run
    // there is no diff to compute and the run returns having cost ~218 ms
    // instead of a paged walk of every document id.
    //
    // ⚠️ `force` BYPASSES THIS ENTIRELY. `force` means "the stored vectors are
    // not trustworthy" — it is a statement about the INDEX, not about the
    // corpus. Letting the watermark short-circuit it would make `force` a no-op
    // on the second call, which is a silent lie about what the flag does.
    const live = opts.force
      ? undefined
      : await readCorpusState(this.api, this.requestTimeoutMs);
    if (live) {
      const watermark = this.store.watermark();
      // No stored watermark means this index predates the watermark, or has
      // never completed a run. Either way there is nothing to compare against
      // and the diff must run: treating an absent watermark as "nothing
      // changed" would skip the first sync after an upgrade and then keep
      // skipping, permanently and silently.
      if (watermark && !corpusChanged(live, watermark)) {
        report.skipped = true;
        report.skipReason =
          `the corpus matches the last run (${live.count} documents, ` +
          `max(modified) ${live.modified ?? "none"}); the document diff was ` +
          `skipped`;
        // ⚠️ A SKIP SKIPS THE DIFF, NOT THE METADATA REFRESH.
        //
        // This is the whole reason the watermark cannot subsume the refresh: a
        // tag, correspondent or document-type edit goes through a queryset
        // `.update()` on a through-table, `auto_now` never fires, and neither
        // `count` nor `max(modified)` moves. A corpus can be byte-identical in
        // both scalars while a document's metadata chunk is stale.
        //
        // So the refresh runs here too. On a freshly-synced index every stamp
        // is inside the window and this loop makes ZERO requests, which is what
        // keeps the "one request" promise true in the common case. When it does
        // have work to do, that work is real and skipping it would leave a
        // corrected sender's old name searchable forever.
        //
        // ⚠️ The `remote` membership filter the main path uses is NOT available
        // here — there is no list walk — so this iterates the indexed ids
        // directly. A document deleted since the last run could therefore be
        // refreshed once before the next count-changing run removes it. One
        // wasted embedding, self-correcting, and strictly better than a stale
        // metadata chunk living forever.
        await this.refreshStaleMetadata(report, this.store.indexed(), undefined);
        return report;
      }
    }

    const remote = new Map<number, string>();
    const { rows, count } = await this.listAll<{ id: number; modified: string }>(
      DOCUMENT_LIST_PATH
    );
    for (const row of rows) remote.set(row.id, row.modified);

    const indexed = this.store.indexed();

    // ── DELETES, AND WHY THE COUNT IS THE ONLY SIGNAL THAT SEES THEM ────────
    //
    // A soft delete in Paperless sets `deleted_at` through a queryset
    // `.update()`. That bypasses the model layer, so `auto_now` never fires and
    // `modified` does not move — which means `max(modified)` is structurally
    // BLIND to a deletion and can never be the signal that triggers one.
    //
    // `count` is not blind. It drops. And it costs nothing here: it arrived in
    // the same list response this walk already reads.
    //
    // So the watermark comparison above decomposes into three cases, and this
    // is why it checks `count` FIRST rather than treating the two scalars as
    // interchangeable:
    //   count down, modified unchanged  -> a deletion. The delete step below
    //                                      runs; nothing else has changed.
    //   count same, modified newer      -> additions or edits. The diff runs.
    //   count same, modified same       -> nothing at all. The run skips.
    //
    // ⚠️ DO NOT "SIMPLIFY" THIS INTO A SINGLE max(modified) CHECK. It reads like
    // a tidier condition and it silently stops deleting anything, forever.
    //
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
      // `force` re-embeds every document, not merely the changed ones. Bypassing
      // the watermark alone would make `force` a no-op on a corpus whose
      // `modified` values are all unchanged, which is the common case — the flag
      // exists precisely for "the stored vectors are not trustworthy", and a
      // trustworthy-looking `modified` says nothing about that.
      if (opts.force) return true;
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
        const context = await this.syncDocument(id, names);
        reindexed.add(id);
        report.splits += context.splits;
        if (indexed.has(id)) report.updated++;
        else report.added++;
      } catch (err) {
        if (err instanceof EmbeddingUnavailableError) {
          // ⚠️ ABORT vs CONTINUE, AND THE LOG SAYS WHICH.
          //
          // An embedding outage (or a text that cannot be made small enough to
          // embed) is not a per-document problem: every remaining document
          // would fail the same way, so the run stops and the index keeps what
          // it has. The line names the document so the next investigation
          // starts from an id instead of from a scan of 2,466 candidates.
          log(
            `document ${describeDocument(err, id)} failed and the run is ` +
              `ABORTING (this failure would repeat for every remaining ` +
              `document): ${errorMessage(err)}`
          );
          throw err;
        }
        report.failed++;
        log(
          `document ${describeDocument(err, id)} failed; CONTINUING with the ` +
            `next document: ${errorMessage(err)}`
        );
      }
    }

    // The 24 h metadata refresh. Tags, correspondents and document types change
    // without touching `modified`, so without this a corrected sender stays in
    // the metadata chunk -- which is indexed both ways -- forever.
    //
    // Documents just reindexed above are excluded: their metadata chunk was
    // rebuilt as part of the reindex seconds ago.
    const refreshTargets = new Map(
      [...indexed.keys()].filter((id) => !reindexed.has(id)).map((id) => [id, ""] )
    );
    await this.refreshStaleMetadata(report, refreshTargets, remote);

    // ── THE WATERMARK IS WRITTEN AT THE END, FROM THE LIST WE ACTUALLY READ ──
    //
    // Not from a second probe. The list walk already reported `count`, and
    // `remote` holds every `modified` it returned, so the maximum is arithmetic
    // over data already in hand rather than another request.
    //
    // ⚠️ DELIBERATELY NOT `Math.max(...remote.values())`. Spreading a few
    // thousand values into an argument list is a stack-depth question for no
    // benefit, and `new Date(x).getTime()` is the comparison the rest of this
    // file already uses. Instants, not strings.
    let newest: string | null = null;
    let newestAt = -Infinity;
    for (const modified of remote.values()) {
      const at = new Date(modified).getTime();
      if (Number.isFinite(at) && at > newestAt) {
        newestAt = at;
        newest = modified;
      }
    }
    // ⚠️ ONLY WHEN THE LIST WAS COMPLETE. A partial page set means documents
    // were moving while the walk ran, so the maximum it saw is a maximum over a
    // subset — recording it would state "the corpus is up to date through here"
    // on the strength of a list that was known to be short.
    if (listComplete) {
      this.store.writeWatermark({
        count,
        modified: newest,
        at: new Date(this.now()).toISOString(),
      });
    }

    return report;
  }

  /**
   * Rebuild the metadata chunk of every indexed document whose stamp is older
   * than the refresh window.
   *
   * Extracted because TWO paths need it: the full run, and the watermark skip.
   * A skip that returned before this would silently retire the 24 h refresh on
   * any corpus that had stopped changing — and a stopped corpus is exactly the
   * one whose tags and correspondents still get edited by hand, because nobody
   * is uploading to it.
   *
   * `remote` is optional: the full run passes the live id set so a document
   * deleted since the last sync is not refreshed, while the skip path has no
   * list to consult and passes undefined.
   */
  private async refreshStaleMetadata(
    report: SyncReport,
    indexed: Map<number, string>,
    remote: Map<number, string> | undefined
  ): Promise<void> {
    const stamps = this.store.metadataStamps();
    const refreshBefore = this.now() - METADATA_REFRESH_MS;
    let names: NameMaps | undefined;
    for (const id of indexed.keys()) {
      if (remote && !remote.has(id)) continue;
      const metadataAt = stamps.get(id);
      if (metadataAt && new Date(metadataAt).getTime() >= refreshBefore) continue;
      try {
        names ??= await this.loadNames();
        report.splits += await this.refreshMetadata(id, names);
        report.metadataRefreshed++;
      } catch (err) {
        if (err instanceof EmbeddingUnavailableError) {
          log(
            `document ${describeDocument(err, id)} metadata failed and the ` +
              `run is ABORTING: ${errorMessage(err)}`
          );
          throw err;
        }
        report.failed++;
        log(
          `document ${describeDocument(err, id)} metadata failed; CONTINUING ` +
            `with the next document: ${errorMessage(err)}`
        );
      }
    }
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

  private async syncDocument(
    id: number,
    names: NameMaps
  ): Promise<{ splits: number }> {
    // A detail fetch can fail on its own, before any embedding happens. The id
    // is still the useful part of the identity, so it is attached here and the
    // title is left absent rather than invented.
    let doc: DocumentDetail;
    try {
      doc = await this.get<DocumentDetail>(`/documents/${id}/`);
    } catch (err) {
      throw withDocumentContext(err, { documentId: id, documentTitle: "" });
    }
    const metaText = this.metadataTextFor(doc, names);
    const { chunks, truncated } = chunkText(doc.content ?? "", this.maxChunks);
    const texts = [metaText, ...chunks];

    // ⚠️ THE TITLE IS ATTACHED HERE BECAUSE THIS IS THE ONLY FRAME THAT HAS IT.
    //
    // A bare "document 1944 failed" costs a second round trip to identify, and
    // on a 2,466-document corpus walking candidates by hand is how twenty tool
    // calls get spent on one diagnosis. The title is a document heading rather
    // than body text, so naming it in a log line places no content at risk.
    let embeddings: number[][];
    try {
      embeddings = await this.embedder.embedDocuments(texts);
    } catch (err) {
      throw withDocumentContext(err, {
        documentId: id,
        documentTitle: doc.title,
      });
    }
    const splits = this.embedder.lastSplits?.length ?? 0;

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
    return { splits };
  }

  private async refreshMetadata(id: number, names: NameMaps): Promise<number> {
    let doc: DocumentDetail;
    try {
      doc = await this.get<DocumentDetail>(`/documents/${id}/`);
    } catch (err) {
      throw withDocumentContext(err, { documentId: id, documentTitle: "" });
    }
    const metaText = this.metadataTextFor(doc, names);
    let embedding: number[];
    try {
      [embedding] = await this.embedder.embedDocuments([metaText]);
    } catch (err) {
      throw withDocumentContext(err, {
        documentId: id,
        documentTitle: doc.title,
      });
    }
    this.store.replaceMetadata(
      id,
      { kind: "metadata", text: metaText, embedding },
      new Date(this.now()).toISOString()
    );
    return this.embedder.lastSplits?.length ?? 0;
  }
}
