// Semantic search tools over the local vector index.
//
// Three tools, and one of them is the whole reason this file exists:
//
//   semantic_search    meaning + words, fused, with a degradation contract
//   embedding_status   what the index actually holds
//   sync_embeddings    bring the index up to date, explicitly
//
// ── The degradation contract ───────────────────────────────────────────────
// The embedding server being down is EXPECTED, not exceptional. It is a local
// process, restarted for model changes and absent on a fresh machine. So:
//
//   * No tool here throws because Ollama is down. A thrown error from an MCP
//     tool reads as "the tool is broken", which is a lie about a tool that is
//     working correctly and reporting a real condition.
//   * Lexical results are STILL RETURNED, with `semantic: "unavailable"`. The
//     lexical half lives in the same SQLite file and does not need the
//     embedder, so an outage degrades the ranking, not the capability.
//   * A degraded call NEVER returns a silent empty list. An empty result and
//     "the embedder is down" are different facts and the caller cannot tell
//     them apart from `passages: []` — which is the exact lie this contract
//     exists to prevent. `semantic` and `degraded_reason` carry it instead.
//
// ── Return shape ───────────────────────────────────────────────────────────
// `passages[]` mirrors `search_passages` exactly — { id, title, score?,
// highlights?, passage, offset, content_length } — so a client that has learned
// one has learned the other. The one addition is `matched`, which earns its
// place because it tells the caller whether to trust the hit or re-query
// lexically: "words" means the hit is a lexical match the vector side did not
// find, so a caller looking for a paraphrase should not conclude absence.
//
// ── Lazy store open ────────────────────────────────────────────────────────
// The gateway launches this package's child at startup. Opening a ~1 GB SQLite
// file and loading the vec0 extension at import time could exceed its child
// startup tolerance, so nothing here opens the store until the first tool call,
// and the open is memoised. That is a named risk in the plan, not tidiness.

import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { PaperlessAPI } from "../api/PaperlessAPI";
import {
  searchConfig,
  indexPath,
  webhookRefusal,
  type SearchConfig,
} from "../search/config";
import { OllamaEmbedder, EmbeddingUnavailableError, type Embedder } from "../search/embedder";
import {
  EMBEDDING_DIMENSIONS,
  SearchIndex,
  LexicalUnavailableError,
  type Hit,
} from "../search/store";
import { CHUNK_OVERLAP } from "../search/chunk";
import { fuse, type Fused } from "../search/rank";
import {
  IndexSync,
  corpusChanged,
  readCorpusState,
  type CorpusState,
  type SyncReport,
} from "../search/sync";
import { WebhookReceiver } from "../search/webhook";
import { withErrorHandling } from "./utils/middlewares";

/** A similarity at or above this is a strong match. Similarity, never distance.
 *
 * ⚠️ NOT USED AS A GATE ANY MORE, and the measurement is why. On this corpus
 * the floor cannot separate a good query from a bad one. Measured on the live
 * index (2,476 docs), best `1 - distance` per query:
 *
 *   relevant   "kuji nine invisibility…"            0.700  "naga offering…"      0.746
 *              "barber pole worm in goats…"         0.761  "channels and winds"  0.766
 *              "three humours and prana disorders"  0.731
 *   irrelevant "xyzzy plugh frobnicate quux"        0.599  "stock market options" 0.595
 *              "quantum chromodynamics…"            0.656  "best pizza dough…"    0.766
 *              "how to change a flat tire"          0.730
 *
 * The ranges OVERLAP: two of five irrelevant queries score above the worst
 * relevant one, and no threshold in 0.500–0.800 separates the sets. A floor at
 * 0.45 sits 0.145 BELOW the worst irrelevant query, so it could never fire —
 * `strong_match` was constant `true` whenever the vector arm ran.
 *
 * Kept exported only so existing importers compile; treat any use of it as a
 * claim that needs this card's measurement re-run on the corpus in question.
 */
export const STRONG_MATCH = 0.45;

/** Chars per returned passage, matching search_passages' default. */
export const DEFAULT_WINDOW = 800;
export const DEFAULT_LIMIT = 10;
export const MAX_LIMIT = 50;

/**
 * The largest JSON body a caller can be handed, in characters.
 *
 * This is not our budget to choose freely: it is the ceiling the CONSUMER
 * applies. Mneme truncates any non-memory tool result at 20,000 characters by
 * taking head and tail, and the agent on the other side cannot detect that it
 * happened -- a passage that was silently cut reads exactly like a short one.
 *
 * `MAX_LIMIT * max window` is 208,485 characters, ten times this number, so
 * until this constant existed the tool's own advertised argument range could
 * produce a result the client would mangle. Bounding it in the schema
 * description would be advice; the enforcement is `budgetedLimit` (so a
 * maximal call does not waste work) plus `fitResultToBudget` (so the returned
 * bytes are measured, not predicted).
 */
export const MAX_RESULT_CHARS = 20_000;

/**
 * JSON scaffolding a single passage costs, roughly: id, title, score, matched,
 * offset, content_length, offset_space, quotes and commas.
 *
 * Used only to pick the passage COUNT before any work is done. `PASSAGE_OVERHEAD_CHARS` is deliberately
 * generous: over-estimating costs a passage, under-estimating costs a wasted
 * build that `fitResultToBudget` then discards anyway.
 */
export const PASSAGE_OVERHEAD_CHARS = 200;

/**
 * Fraction of MAX_RESULT_CHARS that `budgetedLimit` will spend on passages,
 * leaving the rest for the envelope (query, counts, retriever list, reasons).
 */
const RESULT_BUDGET_FRACTION = 0.9;

/**
 * How many passages of `window` characters can fit inside the consumer's
 * ceiling. Never zero: a call that cannot afford even one passage should still
 * return one and let `fitResultToBudget` trim if the bytes really do not fit.
 */
export function budgetedLimit(limit: number, window: number): number {
  const ceiling = Math.floor(
    (MAX_RESULT_CHARS * RESULT_BUDGET_FRACTION) / (window + PASSAGE_OVERHEAD_CHARS)
  );
  return Math.max(1, Math.min(limit, MAX_LIMIT, ceiling));
}

/**
 * The filter arguments, mirroring `search_passages`.
 *
 * Kept in one object so the two tools cannot drift: if search_passages gains a
 * filter, this is the one place that has to learn it.
 *
 * ── WHERE THEY ARE APPLIED, AND WHAT THAT COSTS ────────────────────────────
 * They are evaluated in JS against each hit's fetched document, NOT pushed into
 * the kNN statement. That is a deliberate trade with a known cost, stated here
 * because the previous comment argued against this approach while describing
 * it as the approach:
 *
 *   PRO  The store's kNN returns one hit per document, and the documents are
 *        fetched anyway to build passages, so the filter is free at the point
 *        it is needed and needs no new SQL.
 *   CON  `rowid IN (…)` is NOT composed into the MATCH statement, so the
 *        filtered set can come back SHORT of `limit` with no signal that
 *        ranking was trimmed. This is why `filtered_out` and its reason
 *        breakdown exist: the shortfall is reported rather than hidden.
 *
 * Pushing the filter into the kNN would remove the CON, at the cost of a
 * per-query SQL shape and of needing each filter to be expressible as a set of
 * document ids before the query runs. That is the right eventual design; it is
 * not this change.
 *
 * ⚠️ EVERY KEY HERE MUST BE HONOURED BY `buildDocumentFilter`. A key that is
 * advertised to the model and silently ignored is worse than an absent one: the
 * caller gets unfiltered results with no error, no warning, and no way to
 * notice. `storage_path` was exactly that until it was implemented.
 */
const FILTER_SHAPE = {
  correspondent: z.number().optional(),
  document_type: z.number().optional(),
  tag: z.number().optional(),
  storage_path: z.number().optional(),
  created__date__gte: z.string().optional(),
  created__date__lte: z.string().optional(),
};

export interface VisibleDocument {
  id: number;
  title: string;
  content: string | null;
  correspondent: number | null;
  document_type: number | null;
  /**
   * Paperless' storage path id, or null when the document is unfiled.
   *
   * Nullable because most documents are: on the live corpus it is the exception
   * rather than the rule. It is typed `number | null` rather than optional
   * because `undefined` (field absent) and `null` (explicitly unfiled) filter
   * differently, and `/api/documents/{id}/` always returns the key.
   */
  storage_path: number | null;
  tags: number[];
  created: string;
}

export interface SemanticPassage {
  id: number;
  title: string;
  /**
   * The fusion score, NOT a similarity. See the `score` note on
   * `SemanticSearchResult` — the range is (0, 2/61] and 0.033 is the ceiling,
   * not a weak match. Read `matched` and `highlights` for quality, not this.
   */
  score?: number;
  /** The query term the passage was anchored on, when one was found. */
  highlights?: string;
  matched: "meaning" | "words" | "both";
  passage: string;
  /**
   * Where `passage` starts, IN THE SPACE NAMED BY `offset_space`.
   *
   * This used to be an offset into the raw document, because the raw document
   * was fetched to build the window. It is not fetched any more, so the space
   * changed and the change is stated rather than silent: an offset the caller
   * cannot interpret is worse than no offset.
   */
  offset: number;
  /** Length of the text `offset` indexes into. Same space as `offset`. */
  content_length: number;
  /**
   * Which text `offset` and `content_length` are measured against.
   *
   * `"normalized"` — the document as the chunker stored it: whitespace runs
   * collapsed to single spaces, ends trimmed. This is the normal case now, and
   * offsets are only directly comparable with `search_passages` offsets for
   * documents whose text had no such runs.
   * `"document"` — raw document text, the pre-P2 fallback path. Only reachable
   * on a call that supplied no index text loader.
   */
  offset_space: "normalized" | "document";
}

export interface SemanticSearchResult {
  query: string;
  semantic: "ok" | "unavailable";
  degraded_reason?: string;
  /**
   * Whether ANY passage was returned.
   *
   * ⚠️ THIS IS NOT A QUALITY SIGNAL, and it used to pretend to be one. It was
   * computed against `STRONG_MATCH` (a 0.45 similarity floor), which on this
   * corpus cannot fire — measured best similarities for five irrelevant
   * queries were 0.595, 0.599, 0.656, 0.730 and 0.766, all far above it — so
   * the field was a constant `true` on every call where the vector arm ran.
   * A field that cannot be false is not information.
   *
   * It is retained, and REDEFINED, rather than deleted: deleting it changes the
   * response shape for every consumer, and the honest minimum is a field that
   * means something small and says so. It is now exactly equivalent to
   * `passages.length > 0` and is kept only for callers that branch on it.
   *
   * ⚠️ REMOVING IT IS THE RIGHT EVENTUAL FIX. That needs
   * `src/tools/search.test.ts` edited, which was outside this change's file
   * scope, so it is flagged in the report rather than done.
   *
   * Judge result quality from `matched` (per passage) and `retrievers` (which
   * halves ran), never from this.
   */
  strong_match: boolean;
  /** Pre-filter: how many DISTINCT documents the retrievers returned. */
  total_hits: number;
  /**
   * Post-filter: how many of `total_hits` were dropped before becoming
   * passages.
   *
   * Exists because `total_hits: 12, passages: []` was previously a legal and
   * completely unexplained response. A caller had no way to tell "everything
   * was filtered out by your own filter arguments" from "the ranking found 12
   * documents and every one has since vanished".
   */
  filtered_out: number;
  /**
   * Why they were dropped, as counts. Absent when nothing was dropped.
   *
   * Split by cause because the two are actionable in different ways: a
   * missing document means Paperless would not serve an id the index holds
   * (deleted mid-flight, permissions), whereas a metadata-filter rejection
   * means the arguments the caller passed did their job.
   */
  filtered_out_reasons?: {
    document_unavailable: number;
    metadata_filter: number;
  };
  /**
   * Which retrievers actually RAN, versus which merely returned nothing.
   *
   * This is the field that makes the three degradation states legible. An
   * empty `passages` with `retrievers: ["vector", "lexical"]` is the honest
   * answer "both searched and neither matched"; with `["lexical"]` it means
   * only one half could run. Absence of this field would leave the caller
   * inferring capability from a list length, which is how the old guard came
   * to assert a mechanism that had not happened.
   */
  retrievers: Array<"vector" | "lexical">;
  /**
   * Whether the index still reflects the live corpus. Present ONLY when the
   * caller asked for it with `check_staleness`, because the check costs one
   * extra request and most calls do not need it.
   *
   * ⚠️ `null` MEANS "I COULD NOT CHECK", NEVER "NOT STALE". That distinction is
   * the entire point of making this nullable: a `false` produced because
   * Paperless did not answer would be a fabricated clean bill of health, and
   * fabricating one is the failure mode this whole arc has been written against.
   * A caller that sees `null` knows to retry; a caller that sees `false` knows
   * the index is current.
   */
  stale?: boolean | null;
  /** Age of the index in seconds, from the recorded sync time. Null when unknown. */
  index_age_s?: number | null;
  /** Live document count, or null when the probe failed. */
  live_count?: number | null;
  /** How many documents the index's own watermark claims. Null when unknown. */
  indexed_count?: number | null;
  /** Why staleness could not be determined, when it could not. */
  staleness_reason?: string;
  /**
   * Set when the response had to be shortened to fit the consumer's ceiling.
   *
   * Present rather than silent for the same reason `filtered_out` is: a caller
   * that asked for 50 passages and got 4 needs to know whether the corpus held
   * 4 or whether 46 were dropped, and the two are indistinguishable from
   * `passages.length`.
   */
  trimmed?: {
    reason: "result_size";
    dropped: number;
    limit_chars: number;
    /** True when the echoed `query` itself was cut to fit, not just passages. */
    query_truncated?: boolean;
  };
  /**
   * Ranked passages, bounded.
   *
   * ⚠️ ABOUT `score`: it is the reciprocal-rank-fusion SUM, `Σ 1/(60 + rank)`,
   * not a similarity. With two retrievers the maximum reachable value is
   * `2 / 61 = 0.0328`, so a well-matched passage reports about 0.033 and reads
   * as terrible if you assume a 0..1 similarity scale.
   *
   * It is left un-normalised ON PURPOSE. Normalising against the best score in
   * the same result set would report 1.0 for the head of a set where every hit
   * was bad, which is a worse lie than a small number. A meaningful 0..1
   * score needs a floor calibrated against labelled good and bad results, and
   * that calibration has not been done on this corpus.
   */
  passages: SemanticPassage[];
}

/**
 * Cut a bounded passage around the most DISTINCTIVE place the query terms appear.
 *
 * Mirrors buildSearchPassage's anchoring priority minus the server highlight,
 * which does not exist on this path: the vector side has no highlighted span,
 * and the lexical side's BM25 match gives a chunk rather than an offset. So the
 * anchor is (1) the LONGEST matching query term, (2) the head of the document.
 *
 * Length rather than query order, deliberately: English queries open with
 * articles, so first-match anchored "the practice of a wrathful deity with a
 * lion head" on "the" at character 3 and put eleven of eighteen windows at the
 * document head. Containment still passed -- "the" really was in the window --
 * which is why the defect survived a green suite and a code review.
 *
 * A window that does not contain the hit is worse than no window, which is why
 * the term search runs against the whole document rather than the chunk.
 */
export function buildPassage(
  doc: { id: number; title: string; content: string | null },
  terms: string[],
  window: number
): { passage: string; offset: number; content_length: number; term?: string } {
  const full = doc.content ?? "";
  const lower = full.toLowerCase();
  let center = -1;
  let found: string | undefined;
  // ANCHOR ON THE MOST SPECIFIC TERM, not the first one in query order.
  //
  // Measured defect this fixes: the query "the practice of a wrathful deity with
  // a lion head" anchored on "the" at character 3, because `indexOf` walks the
  // query left-to-right and English queries open with articles. 11 of 18 windows
  // landed at the document head. Containment still passed -- "the" really is in
  // the window -- which is exactly why no test caught it: the property that
  // holds is the weak one, and the one that matters (a window around the
  // DISTINCTIVE term) was never asserted.
  //
  // Length is the cheap proxy for specificity here, and it is the right one:
  // a longer term is what distinguishes this query from most others. Ties keep
  // query order via the stable sort. If nothing matches, offset 0 and say so --
  // a head window is honest when the query genuinely appears nowhere.
  const candidates = [...terms].sort((a, b) => b.length - a.length);
  for (const term of candidates) {
    const at = lower.indexOf(term.toLowerCase());
    if (at >= 0) {
      center = at;
      found = term;
      break;
    }
  }
  const resolved = center < 0 ? 0 : center;
  const start = Math.max(0, resolved - Math.floor(window / 2));
  // The end is widened only when doing so is required to CONTAIN the term it
  // is anchoring on. A window that stops one character short of the match it
  // was centred for is the exact defect the containment rule forbids, and with
  // a term longer than `window` the plain `start + window` would do it.
  const end = Math.max(start + window, resolved + (found ? found.length : 0));
  return {
    passage: full.slice(start, end),
    offset: start,
    content_length: full.length,
    ...(found ? { term: found } : {}),
  };
}

/** Query terms for passage anchoring, from the same tokenisation the FTS side uses. */
export function queryTerms(query: string): string[] {
  return query
    .toLowerCase()
    // \p{M} is required for the same reason it is in store.toFtsMatch: the
    // anchoring terms and the lexical terms must tokenise identically, or a
    // passage is anchored on one fragment of a syllable and matched on another.
    .split(/[^\p{L}\p{N}\p{M}_]+/u)
    .filter((term) => term.length > 0);
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/**
 * Everything the three tools share, built once and lazily.
 *
 * `store` is `undefined` until the first call. `openFailed` remembers a failure
 * so a broken index does not retry a 1 GB open on every single tool call -- and
 * remembers the MESSAGE, so embedding_status can report why.
 */
export class SearchContext {
  private store?: SearchIndex;
  private openFailed?: string;
  private sync?: IndexSync;
  private syncStarted = false;
  private webhook?: WebhookReceiver;
  private webhookStarted = false;
  private embedder?: Embedder | null;
  private readonly configOverride?: SearchConfig;
  private readonly embedderFactory?: (config: SearchConfig) => Embedder | undefined;

  /**
   * `embedderFactory` exists so "no embedder" can be stated rather than
   * inferred. A context whose embedder is merely unset would reach
   * `embedderOrUndefined` and CONSTRUCT a real client — which then makes a
   * network call. That is correct in production and unacceptable in a test, and
   * the difference has to be expressible, so the factory can return undefined
   * and mean it.
   */
  constructor(
    config?: SearchConfig,
    embedderFactory?: (config: SearchConfig) => Embedder | undefined
  ) {
    this.configOverride = config;
    this.embedderFactory = embedderFactory;
  }

  /** State the embedder explicitly. `null` means "there is deliberately none". */
  setEmbedder(embedder: Embedder | null): void {
    this.embedder = embedder;
    this.sync = undefined;
  }

  /**
   * Resolve config from the environment at CALL time.
   *
   * Not at import: the gateway sets the child's environment before launch, but
   * tests and the CLI set it after import, and reading once at module load
   * makes the config untestable.
   */
  config(): SearchConfig | undefined {
    return this.configOverride ?? searchConfig(process.env);
  }

  /** True when the search tools should register at all. */
  get isConfigured(): boolean {
    return this.config() !== undefined;
  }

  /** Open the index, memoised. Throws the underlying error on first failure. */
  index(): SearchIndex {
    if (this.store) return this.store;
    if (this.openFailed) {
      throw new Error(`The search index could not be opened: ${this.openFailed}`);
    }
    const config = this.config();
    if (!config) {
      throw new Error(
        "Semantic search is not configured. Set PAPERLESS_MCP_EMBEDDING_URL " +
          "and the other PAPERLESS_MCP_* search variables to enable it."
      );
    }
    // `expected` is what makes the deployment policy-adjacent schema guard REAL at query
    // time. Without it, an index built with `e5` and queried with `nomic`
    // returned `semantic: "ok"` and quietly worse results, while
    // `assertPrefixModeMatches`'s own KDoc said the mismatch "must be refused
    // loudly". Same for a same-width model swap and a bumped schema_version.
    const store = new SearchIndex(indexPath(config), {
      expected: {
        dimensions: EMBEDDING_DIMENSIONS,
        model: config.model,
        prefixMode: config.prefixMode,
      },
    });
    try {
      store.open();
    } catch (err) {
      this.openFailed = errorMessage(err);
      throw err;
    }
    this.store = store;
    return store;
  }

  embedderOrUndefined(): Embedder | undefined {
    // `null` is memoised and means "deliberately none" — checked explicitly so
    // it never falls through to construction.
    if (this.embedder === null) return undefined;
    if (this.embedder) return this.embedder;
    const config = this.config();
    if (!config) return undefined;
    try {
      const built = this.embedderFactory
        ? this.embedderFactory(config)
        : new OllamaEmbedder({
            url: config.embeddingUrl,
            model: config.model,
            prefixMode: config.prefixMode,
            // Threaded rather than defaulted here: the value is read and
            // range-checked in config.ts, and a knob that is validated but
            // never passed on is an advertised setting with no effect.
            batchSize: config.syncBatchSize,
            localOnly: config.embeddingLocalOnly,
            ...(config.embeddingApiKey ? { apiKey: config.embeddingApiKey } : {}),
          });
      this.embedder = built ?? null;
    } catch {
      // A misconfigured endpoint is not an unavailable embedder, but on this
      // path both mean "no vectors this turn", and the lexical half still works.
      this.embedder = null;
      return undefined;
    }
    return this.embedder ?? undefined;
  }

  syncRunner(api: PaperlessAPI): IndexSync {
    if (this.sync) return this.sync;
    const embedder = this.embedderOrUndefined();
    if (!embedder) {
      throw new Error(
        "Semantic search cannot sync without an embedder. Check " +
          "PAPERLESS_MCP_EMBEDDING_URL."
      );
    }
    const config = this.config();
    this.sync = new IndexSync({
      api,
      store: this.index(),
      embedder,
      ...(config ? { maxChunks: config.maxChunks } : {}),
    });
    return this.sync;
  }

  /**
   * Start the background sync timer when the deployment asked for one.
   *
   * ⚠️ THIS METHOD IS THE CALLER `IndexSync.start()` NEVER HAD. The interval was
   * read and range-checked in config.ts from the day it was added, and nothing
   * ever acted on it: `start()` existed, was documented, and was invoked from
   * nowhere. A validated setting with no consumer reads as a working feature
   * and is not one.
   *
   * Called from `registerSearchTools`, so the wiring is exercised by every
   * server that configures search at all. A test that constructs the timer
   * directly proves `start()` works and says nothing about whether anything
   * calls it; the caller is what has to be asserted.
   *
   * A background sync that cannot start must not take down tool registration:
   * the observable `sync_embeddings` tool still works, so failure here is
   * logged and swallowed rather than thrown.
   */
  startBackgroundSync(api: PaperlessAPI): void {
    if (this.syncStarted) return;
    const config = this.config();
    if (!config || config.syncIntervalMs <= 0) return;
    try {
      this.syncRunner(api).start(config.syncIntervalMs);
      this.syncStarted = true;
    } catch (err) {
      console.error(
        `[paperless-mcp] background sync not started: ${errorMessage(err)}`
      );
    }
  }

  /** True when a background sync timer is live. Used by the wiring test. */
  get isBackgroundSyncActive(): boolean {
    return this.sync?.isScheduled === true;
  }

  /**
   * Start the change-notification receiver when the deployment asked for one.
   *
   * ⚠️ THE SAME CALLER PROBLEM AS `startBackgroundSync`, ONE LAYER OUT. A
   * receiver class nobody constructs is a feature nobody has, and the interval
   * in this file is the standing proof: it was read, validated and never acted
   * on. So construction happens here, on the path `registerSearchTools` always
   * walks, rather than behind a bootstrap only a test would call.
   *
   * Started asynchronously and never awaited. Registration is synchronous and
   * happens before the transport connects; a bind that fails must not delay or
   * break the tool surface, because the observable `sync_embeddings` tool still
   * works and the watermark still converges on the next start without it.
   */
  startWebhook(api: PaperlessAPI): void {
    if (this.webhookStarted) return;
    const config = this.config();
    if (!config || config.webhookPort === 0) return;

    const refusal = webhookRefusal(config);
    if (refusal) {
      console.error(`[paperless-mcp] webhook not started: ${refusal}`);
      return;
    }

    // The runner is resolved lazily, per event: building it needs an embedder
    // and an open store, both of which are expensive and neither of which
    // should be paid at startup by an install that never receives an event.
    const receiver = new WebhookReceiver({
      secret: config.webhookSecret ?? "",
      port: config.webhookPort,
      host: config.webhookHost,
      debounceMs: config.webhookDebounceMs,
      runner: () => this.syncRunner(api),
    });
    this.webhook = receiver;
    this.webhookStarted = true;

    void receiver.start().catch((err: unknown) => {
      // A port already in use, or a refused bind. Logged and dropped: the
      // receiver is optional infrastructure and must not kill the server.
      console.error(
        `[paperless-mcp] webhook not started: ${errorMessage(err)}`
      );
      this.webhook = undefined;
      this.webhookStarted = false;
    });
  }

  /** True when a receiver is bound. Used by the wiring test. */
  get isWebhookListening(): boolean {
    return this.webhook?.isListening === true;
  }

  /** The port the receiver actually bound, or undefined when it is off. */
  get webhookPort(): number | undefined {
    return this.webhook?.boundPort;
  }

  close(): void {
    // Stop the timer and the receiver BEFORE dropping the store, or a stdio
    // child shutting down leaves a scheduled run firing against a closed index.
    //
    // ⚠️ The receiver's close is ASYNC and `close()` is not, so it is started
    // here and not awaited. That is safe rather than sloppy: the receiver's own
    // `close()` clears its debounce timer synchronously first, so no new run can
    // be scheduled from this point on -- only an already-firing one can still be
    // in flight, and that one is bounded by the store's own `isOpen` state.
    this.sync?.stop();
    if (this.webhook) {
      void this.webhook.close().catch(() => {
        // Nothing useful to do while shutting down.
      });
      this.webhook = undefined;
    }
    this.webhookStarted = false;
    this.store?.close();
    this.store = undefined;
    this.sync = undefined;
    this.syncStarted = false;
  }
}

/**
 * Fuse the two lists, tolerating an absent lexical half or an absent embedder.
 *
 * This is the function the contract is built on, and it is deliberately one
 * function: the degradation lives in one place so no caller can accidentally
 * reintroduce a silent empty list.
 */
export async function runSemanticSearch(
  context: SearchContext,
  query: string,
  opts: {
    limit: number;
    window: number;
    loadDocuments: (ids: number[]) => Promise<Map<number, VisibleDocument>>;
    /**
     * Document text, read from the index instead of over HTTP.
     *
     * ⚠️ OPTIONAL ONLY FOR TEST INJECTION; production always supplies it. When
     * present, the passage is built from THIS text and the metadata documents
     * never need a `content` field — which is what removes one full-document
     * GET per hit. When absent, the old `doc.content` path is used, and
     * `offset_space` says `"document"` so the caller is not misled about what
     * its offsets index into.
     */
    loadText?: (hits: Fused[]) => Map<number, string>;
    filters?: (doc: VisibleDocument) => boolean;
    /** Opt in to the staleness report. Costs one extra request when true. */
    checkStaleness?: boolean;
    /**
     * How to read the live corpus. Injected rather than imported so the search
     * path has no opinion about which API client is in use, and so a test can
     * supply a corpus without a network.
     */
    corpusState?: () => Promise<CorpusState | undefined>;
    /** Injectable clock, so `index_age_s` is testable. */
    now?: () => number;
  }
): Promise<SemanticSearchResult> {
  const store = context.index();
  // ⚠️ BUDGETED, not merely clamped. `MAX_LIMIT` alone permits
  // `50 * 4000 = 208,485` characters, which is 10x the ceiling the consuming
  // client applies and silently mangles. This is the first of the two
  // enforcements: it stops the work from being done. `fitResultToBudget` is
  // the second and it measures real bytes.
  const limit = budgetedLimit(Math.min(Math.max(opts.limit, 1), MAX_LIMIT), opts.window);
  const terms = queryTerms(query);

  // ── THE STALENESS REPORT, BUILT BEFORE THE SEARCH SO ONE SPREAD SENDS IT ──
  //
  // Computed once and spread into BOTH return paths (the empty-fused early
  // return and the main one). Computing it twice would be two probes for one
  // answer, and computing it only on the main path would silently omit it from
  // exactly the response where a caller is most likely to wonder whether the
  // index is current — an empty result.
  //
  // ⚠️ A FAILED PROBE YIELDS `stale: null`, NEVER `false`. "I could not check"
  // and "I checked and it is current" are different facts, and a search that
  // fabricates the second because Paperless was down would be lying about the
  // one thing this field exists to report. The search itself is unaffected: the
  // degradation contract already covers a dead embedder, and a dead Paperless
  // costs only this annotation.
  const stalenessFields = opts.checkStaleness
    ? await readStaleness(opts.corpusState, store, opts.now)
    : {};

  let semantic: SemanticSearchResult["semantic"] = "ok";
  const reasons: string[] = [];
  let vector: Hit[] = [];
  let text: Hit[] = [];

  // ⚠️ THE THREE STATES, kept distinct, because collapsing them produced a FALSE
  // STATEMENT OF MECHANISM. The old guard fired when BOTH lists came back empty
  // and the embedder was down, and announced "no lexical fallback exists" --
  // which was untrue whenever FTS5 was present and simply had no match. A
  // caller then could not tell "nothing in the corpus says zzzznothing" from
  // "search is broken", which is the exact distinction the degradation contract
  // exists to preserve.
  //
  //   (a) BOTH RAN, NOTHING MATCHED  -> semantic ok|unavailable, empty list,
  //                                     and a field that says so. ALLOWED.
  //   (b) LEXICAL RAN, EMBEDDER DOWN -> results + unavailable + the reason.
  //                                    Allowed; already covered above.
  //   (c) NEITHER IS CAPABLE         -> refuse with an error. There is
  //                                    genuinely nothing to search with, and
  //                                    an empty list would read as "no such
  //                                    document".
  //
  // The predicate for (c) is CAPABILITY, not emptiness: `store.lexicalAvailable`
  // is a property of the build, and the embedder's absence is a property of the
  // configuration. Neither can be inferred from a list length.

  // The lexical half needs no network and no model. It runs whenever the build
  // supports it, and it is the reason a degraded call still returns something.
  let lexicalCapable = store.lexicalAvailable;
  if (lexicalCapable) {
    try {
      // Over-fetch because filters are applied below: fetching exactly `limit`
      // would let one filtered-out document empty the result.
      text = store.textHits(query, limit * 4);
    } catch (err) {
      if (!(err instanceof LexicalUnavailableError)) throw err;
      // The build CLAIMED FTS5 and the query path disagreed. Treat the
      // capability as false rather than swallowing it: this is the one place
      // the property could be wrong, and state (c) must not be missed because
      // of it.
      lexicalCapable = false;
      reasons.push(`Lexical search failed despite FTS5 being present: ${errorMessage(err)}`);
    }
  } else {
    reasons.push("Lexical search is unavailable in this build: SQLite has no FTS5 module.");
  }

  const embedder = context.embedderOrUndefined();
  let vectorCapable = false;
  if (!embedder) {
    semantic = "unavailable";
    reasons.push("The embedding client is not configured.");
  } else {
    try {
      const embedding = await embedder.embedQuery(query);
      vector = store.vectorHits(embedding, limit * 4);
      vectorCapable = true;
    } catch (err) {
      if (!(err instanceof EmbeddingUnavailableError)) throw err;
      // EXPECTED. The lexical list above already holds whatever could be found.
      semantic = "unavailable";
      reasons.push(errorMessage(err));
    }
  }

  // State (c): neither retriever is capable. Refuse loudly.
  if (!lexicalCapable && !vectorCapable) {
    throw new Error(
      `Semantic search is fully unavailable: neither the vector side nor a ` +
        `lexical fallback can run. ${reasons.join(" ")}`
    );
  }

  // One reason string, but FIRST-REASON-WINS is now explicit rather than the
  // `??=` accident it was: `??=` let a lexical-unavailable reason mask a later
  // embedder reason, so a call with no FTS5 AND a dead embedder reported only
  // the lexical half and left the reader guessing about the vectors.
  const degradedReason = reasons.length > 0 ? reasons.join(" ") : undefined;

  const fused: Fused[] = fuse(vector, text);

  // Which retrievers actually RAN. Reported rather than inferred: an empty
  // result with both here is "both searched and neither matched", with one is
  // "only that half could run", and the difference is the whole point.
  const retrievers: Array<"vector" | "lexical"> = [];
  if (vectorCapable) retrievers.push("vector");
  if (lexicalCapable) retrievers.push("lexical");

  if (fused.length === 0) {
    return {
      query,
      semantic,
      ...(degradedReason ? { degraded_reason: degradedReason } : {}),
      strong_match: false,
      total_hits: 0,
      filtered_out: 0,
      retrievers,
      ...stalenessFields,
      // State (a) on its own terms: nothing matched and the payload says so —
      // `semantic`, `retrievers`, and an empty list together are unambiguous.
      passages: [],
    };
  }

  // Anchoring and filters both need METADATA. They do NOT need content, and
  // fetching it per hit was the defect: one `GET /documents/{id}/` per hit, up
  // to 80 for a 10-passage call, each returning a full body that was sliced to
  // a window and thrown away.
  const documents = await opts.loadDocuments(fused.map((hit) => hit.documentId));

  // The text the window is cut from, from the index. `undefined` rather than an
  // empty map when no loader was injected, so the fallback is explicit.
  const indexText = opts.loadText ? opts.loadText(fused) : undefined;

  const passages: SemanticPassage[] = [];
  // Post-fetch accounting. Without it `total_hits: 12, passages: []` was a
  // legal, unexplained response: the caller could not tell that its own filter
  // arguments had removed everything.
  let documentUnavailable = 0;
  let metadataFiltered = 0;
  for (const hit of fused) {
    const doc = documents.get(hit.documentId);
    if (!doc) {
      documentUnavailable += 1;
      continue;
    }
    if (opts.filters && !opts.filters(doc)) {
      metadataFiltered += 1;
      continue;
    }
    const indexed = indexText?.get(doc.id);
    const fromIndex = indexed !== undefined;
    const anchored = buildPassage(
      { id: doc.id, title: doc.title, content: fromIndex ? indexed : doc.content },
      terms,
      opts.window
    );
    passages.push({
      id: doc.id,
      title: doc.title,
      score: hit.score,
      matched: hit.matched,
      passage: anchored.passage,
      offset: anchored.offset,
      content_length: anchored.content_length,
      // `buildPassage` reports the term it anchored on; it is promoted to
      // `highlights` because that is the field consumers already read. The
      // containment rule holds by construction: the term was located IN the
      // exact text the window was cut from, and the window's end is widened
      // when needed to keep it inside.
      ...(anchored.term ? { highlights: anchored.term } : {}),
      offset_space: fromIndex ? "normalized" : "document",
    });
    if (passages.length >= limit) break;
  }

  // ⚠️ `strong_match` is now literally "a passage was returned". The 0.45
  // similarity gate it used to consult could never fire on this corpus — see
  // the STRONG_MATCH note — so the old computation reported `true` on every
  // call that produced a hit, regardless of quality. A field that cannot be
  // false is not evidence.
  const strong = passages.length > 0;

  // `filtered_out` counts what the loop dropped, NOT `total_hits - passages`
  // -- those differ when the `limit` break fires with hits still unexamined.
  // The honest decomposition is: fused = shown + dropped + never-reached.
  const filteredOut = documentUnavailable + metadataFiltered;

  return fitResultToBudget({
    query,
    semantic,
    ...(degradedReason ? { degraded_reason: degradedReason } : {}),
    strong_match: strong,
    total_hits: fused.length,
    filtered_out: filteredOut,
    ...(filteredOut > 0
      ? { filtered_out_reasons: { document_unavailable: documentUnavailable, metadata_filter: metadataFiltered } }
      : {}),
    retrievers,
    ...stalenessFields,
    passages,
  });
}

/**
 * The staleness annotation, as a spreadable fragment.
 *
 * Returns the fields with `stale: null` when the corpus cannot be read or the
 * index carries no watermark. Neither invents a value.
 *
 * ⚠️ AN INDEX WITH NO WATERMARK IS `stale: null`, NOT `stale: true`. It is
 * genuinely unknown — the index predates the watermark, or has never completed
 * a run — and asserting either answer would be a guess dressed as a measurement.
 * The actionable thing for such a caller is `sync_embeddings`, and
 * `staleness_reason` says exactly that.
 */
async function readStaleness(
  corpusState: (() => Promise<CorpusState | undefined>) | undefined,
  store: SearchIndex,
  now: (() => number) | undefined
): Promise<Partial<SemanticSearchResult>> {
  if (!corpusState) {
    return {
      stale: null,
      index_age_s: null,
      live_count: null,
      indexed_count: null,
      staleness_reason: "No corpus probe was available to this call.",
    };
  }
  const live = await corpusState();
  const watermark = store.watermark();
  const indexedCount = watermark?.count ?? null;

  if (!watermark) {
    return {
      stale: null,
      index_age_s: null,
      live_count: live?.count ?? null,
      indexed_count: null,
      staleness_reason:
        "The index has no recorded watermark (it predates one, or has never " +
        "completed a sync), so staleness cannot be determined. Run " +
        "sync_embeddings.",
    };
  }

  // Age is computable from the stored timestamp alone, so it is reported even
  // when the live probe failed — it is a fact about the index, not about the
  // corpus.
  const syncedAt = new Date(watermark.at).getTime();
  const indexAge = Number.isFinite(syncedAt)
    ? Math.max(0, Math.round(((now ?? Date.now)() - syncedAt) / 1000))
    : null;

  if (!live) {
    return {
      stale: null,
      index_age_s: indexAge,
      live_count: null,
      indexed_count: indexedCount,
      staleness_reason:
        "The live corpus could not be read, so staleness cannot be " +
        "determined. The search results above are unaffected.",
    };
  }

  return {
    stale: corpusChanged(live, watermark),
    index_age_s: indexAge,
    live_count: live.count,
    indexed_count: indexedCount,
  };
}

/** Register the three semantic tools. No-op when search is not configured. */
export function registerSearchTools(
  server: McpServer,
  api: PaperlessAPI,
  context = new SearchContext()
): SearchContext {
  if (!context.isConfigured) return context;

  // Opt-in background sync. No-op unless PAPERLESS_MCP_SYNC_INTERVAL_MS > 0.
  context.startBackgroundSync(api);
  // Opt-in change-notification receiver. No-op unless
  // PAPERLESS_MCP_WEBHOOK_PORT > 0, and refuses without a secret.
  context.startWebhook(api);

  const filterArgs = FILTER_SHAPE;

  server.tool(
    "semantic_search",
    "Search the document corpus by MEANING as well as by words, using a local vector index fused with the corpus' own BM25 lexical search. Returns bounded passages in the same shape as search_passages (id, title, score, passage, offset, content_length) plus `matched`, which says whether the hit came from meaning, words, or both -- a 'words' hit is a lexical match the vector side did not find, so do not read it as absence of a paraphrase. HONEST DEGRADATION: `semantic` is 'unavailable' and lexical results are still returned when the local embedding server is down; a degraded call never returns a silent empty list. strong_match is true exactly when at least one passage was returned; it is a COARSE 'did-anything-come-back' flag, NOT a quality judgement -- the 0.45 similarity floor it used to consult cannot fire on this corpus (best raw similarities run 0.73-0.79), so judge quality from `matched`, `retrievers` and the passages themselves. Requires the local index to have been built with sync_embeddings.",
    {
      query: z.string().min(1).describe("What to search for, in natural language."),
      limit: z
        .number()
        .int()
        .min(1)
        .max(MAX_LIMIT)
        .optional()
        .describe(`Maximum passages to return (default ${DEFAULT_LIMIT}, max ${MAX_LIMIT}).`),
      window: z
        .number()
        .int()
        .min(200)
        .max(4000)
        .optional()
        .describe(`Characters per passage (default ${DEFAULT_WINDOW}).`),
      check_staleness: z
        .boolean()
        .optional()
        .describe(
          "Also report whether the index still matches the live corpus: `stale`, `index_age_s`, `live_count`, `indexed_count`. Costs one extra request (~44% on a sub-second call), so it is off by default. `stale: null` means the check could not be made — NOT that the index is current."
        ),
      ...filterArgs,
    },
    withErrorHandling(async (args) => {
      // `check_staleness` is destructured OUT of `filters`, not left in it.
      // `buildDocumentFilter` ignores unknown keys today, so leaving it there
      // would work and would also mean every future filter key has to remember
      // to exclude it — a silent coupling rather than a stated one.
      const { limit, window, check_staleness, ...filters } = args;
      const result = await runSemanticSearch(context, args.query, {
        limit: limit ?? DEFAULT_LIMIT,
        window: window ?? DEFAULT_WINDOW,
        // ONE batched metadata request for N documents, content never fetched.
        loadDocuments: (ids) => loadDocumentMetadata(api, ids),
        // The text the window is cut from comes from the index, which already
        // holds it. Reading it here instead of over HTTP is the 41x.
        loadText: indexTextLoader(context.index()),
        filters: buildDocumentFilter(filters),
        ...(check_staleness ? { checkStaleness: true } : {}),
        // Bound late: `api` is this server's client, and the probe must use the
        // same one the rest of the tool does.
        corpusState: () => readCorpusState(api),
      });
      return {
        content: [{ type: "text", text: JSON.stringify(result) }],
      };
    })
  );

  server.tool(
    "embedding_status",
    "Report what the local semantic-search index actually holds: document and chunk counts, embedding dimensions, the model and prefix mode it was built with, when it last synced, how many documents are pending, how much disk the index uses, and whether any documents were truncated at the chunk cap. Use this before trusting semantic_search, and to tell 'the index is empty' apart from 'nothing matched'.",
    {},
    withErrorHandling(async () => {
      const status = await embeddingStatus(context, api);
      return {
        content: [{ type: "text", text: JSON.stringify(status) }],
      };
    })
  );

  server.tool(
    "sync_embeddings",
    "Bring the local semantic-search index up to date with the Paperless corpus. Fetches every document whose `modified` changed since the last run, re-chunks and re-embeds it, removes documents that no longer exist, and refreshes metadata for documents whose tags or correspondent changed without the document being edited. Long-running on a first build (tens of minutes for a large corpus) and it stops with an error if the local embedding server is unavailable, rather than leaving a half-built index that claims to be complete. Run embedding_status afterwards to confirm. force:true re-embeds every document and should only be used when the chunker or the model changed.",
    {
      force: z
        .boolean()
        .optional()
        .describe(
          "Re-embed every document even if `modified` is unchanged. Only needed after a chunker or model change; a normal sync is incremental."
        ),
    },
    withErrorHandling(async (args) => {
      const runner = context.syncRunner(api);
      if (args.force) {
        // ⚠️ PROBE BEFORE DELETE. The old order was delete -> run -> throw on
        // outage, which turned a transient embedder blip into "you no longer
        // have an index" on a rebuild that takes ~58 minutes. The failure the
        // caller can afford is a refusal; the failure it cannot afford is
        // having destroyed the working index to discover the server is down.
        //
        // The probe is a real embedding of a constant string (see
        // `OllamaEmbedder.probe`), not a GET: a server that is up without its
        // model pulled answers GET and then fails on the first real request.
        const embedder = context.embedderOrUndefined();
        if (embedder) {
          if (embedder.probe) {
            await embedder.probe();
          } else {
            // A fake without probe(): still exercise the same path the sync
            // will use, so the ordering guarantee holds for every embedder.
            await embedder.embedQuery("ping");
          }
        }
        // force means "the stored vectors are not trustworthy", which on this
        // schema means deleting the document rows so the diff sees everything
        // as new. Done through deleteDocuments rather than a file unlink so the
        // FTS and vector tables are emptied in the same transaction.
        const store = context.index();
        store.deleteDocuments([...store.indexed().keys()]);
      }
      // ⚠️ `force` IS PASSED THROUGH, not merely acted on locally. The pre-sync
      // delete below empties the document rows so the diff sees everything as
      // new; the flag ALSO has to reach `run()` because the watermark check
      // there would otherwise skip the run it just prepared for. Both halves are
      // required: without the delete the vectors are stale, and without the flag
      // the run does nothing at all.
      const report: SyncReport = await runner.run({ force: args.force === true });
      return {
        content: [{ type: "text", text: JSON.stringify(report) }],
      };
    })
  );

  return context;
}

/** A predicate over visible documents, from the search_passages filter args. */
export function buildDocumentFilter(filters: {
  correspondent?: number;
  document_type?: number;
  tag?: number;
  storage_path?: number;
  created__date__gte?: string;
  created__date__lte?: string;
}): ((doc: VisibleDocument) => boolean) | undefined {
  // Every key in FILTER_SHAPE is honoured here. `storage_path` used to be the
  // exception: advertised to the model, absent from this predicate, and
  // therefore silently ignored -- a caller got unfiltered results with no
  // error, no warning, and no way to notice. That is worse than not offering
  // the filter at all, because the schema is a promise.
  const active =
    filters.correspondent !== undefined ||
    filters.document_type !== undefined ||
    filters.tag !== undefined ||
    filters.storage_path !== undefined ||
    filters.created__date__gte !== undefined ||
    filters.created__date__lte !== undefined;
  if (!active) return undefined;
  return (doc) => {
    if (filters.correspondent !== undefined && doc.correspondent !== filters.correspondent) {
      return false;
    }
    if (filters.document_type !== undefined && doc.document_type !== filters.document_type) {
      return false;
    }
    if (filters.tag !== undefined && !doc.tags.includes(filters.tag)) return false;
    // An unfiled document (storage_path === null) can never match a numeric
    // storage_path, which is correct: the caller asked for a filing location.
    if (filters.storage_path !== undefined && doc.storage_path !== filters.storage_path) {
      return false;
    }
    if (filters.created__date__gte !== undefined && doc.created < filters.created__date__gte) {
      return false;
    }
    if (filters.created__date__lte !== undefined && doc.created > filters.created__date__lte) {
      return false;
    }
    return true;
  };
}

/**
 * Fetch the documents behind a set of ids, tolerating individual failures.
 *
 * One unfetchable document must not fail the whole search: the remaining hits
 * are still real, and a search that returns four good passages and one gap is
 * strictly better than an error. The gap is visible because the passage for
 * that id is absent.
 */
export async function loadVisibleDocuments(
  api: PaperlessAPI,
  ids: number[]
): Promise<Map<number, VisibleDocument>> {
  const documents = new Map<number, VisibleDocument>();
  const unique = [...new Set(ids)];
  const results = await Promise.all(
    unique.map(async (id) => {
      try {
        const doc = (await api.request(`/documents/${id}/`)) as VisibleDocument;
        return doc;
      } catch {
        return undefined;
      }
    })
  );
  for (const doc of results) {
    if (doc && typeof doc.id === "number") documents.set(doc.id, doc);
  }
  return documents;
}

/**
 * Ids per batched metadata request.
 *
 * A single call can legitimately want metadata for `limit * 4` distinct
 * documents from EACH retriever, so 400 is reachable. One request per document
 * is what this replaces; one request for 400 ids would be a 2 KB URL, so the
 * list is cut into batches rather than trusted to be short.
 */
export const METADATA_BATCH = 100;

/**
 * Fetch metadata for many documents in as few requests as possible.
 *
 * ⚠️ THIS NEVER FETCHES DOCUMENT CONTENT, and that is the point of the whole
 * change. The previous path issued one `GET /documents/{id}/` per hit -- up to
 * 80 of them for a 10-passage call -- and read every full body only to slice a
 * window out of it. Measured on the live box for 40 documents: 2,398 ms and
 * 2,162,306 bytes, against 58 ms and 6,520 bytes for the same metadata in one
 * ranged call. The text the window needs was already in the index.
 *
 * `content` is therefore `null` on every returned document, which the
 * `VisibleDocument` type already allows. Nothing on this path reads it.
 *
 * A batch that fails does NOT fail the search: the affected ids are simply
 * absent from the map, which the caller already reports as
 * `document_unavailable`. That is the same tolerance the per-id version had,
 * kept deliberately so one probe failure cannot turn a search into an error.
 */
export async function loadDocumentMetadata(
  api: PaperlessAPI,
  ids: number[]
): Promise<Map<number, VisibleDocument>> {
  const documents = new Map<number, VisibleDocument>();
  const unique = [...new Set(ids)];
  if (unique.length === 0) return documents;
  for (let offset = 0; offset < unique.length; offset += METADATA_BATCH) {
    const batch = unique.slice(offset, offset + METADATA_BATCH);
    const path =
      `/documents/?id__in=${batch.join(",")}` +
      `&fields=id,title,correspondent,document_type,storage_path,tags,created,modified` +
      `&page_size=${batch.length}`;
    try {
      const page = (await api.request(path)) as { results?: VisibleDocument[] };
      for (const doc of page.results ?? []) {
        if (typeof doc.id === "number") {
          documents.set(doc.id, { ...doc, content: null });
        }
      }
    } catch {
      // Left absent, exactly as the per-id path left a failed fetch absent.
    }
  }
  return documents;
}

/**
 * Rebuild a document's text from the chunks the index already holds.
 *
 * The chunker emits `CHUNK_SIZE` characters at a stride of
 * `CHUNK_SIZE - CHUNK_OVERLAP`, so consecutive chunks share exactly the
 * trailing `CHUNK_OVERLAP` characters of the previous one. Splicing therefore
 * has to drop that overlap -- but ONLY when it is really there, because a
 * document shorter than one chunk is a single chunk and dropping 200 characters
 * from it would delete text rather than deduplicate it.
 *
 * The metadata chunk at position 0 is skipped: it is a synthetic
 * "Title: … Sender: …" line, not document text, and including it would put
 * fabricated words in front of a passage and shift every offset by its length.
 *
 * ⚠️ The result is NORMALISED text -- `chunkText` collapses whitespace runs
 * before slicing. Offsets derived from it are labelled `offset_space:
 * "normalized"` for that reason.
 */
export function reconstructDocumentText(
  chunks: { position: number; kind: "metadata" | "content"; text: string }[]
): string {
  // ⚠️ PIECES + ONE JOIN, NEVER `built += …`. The obvious implementation
  // (`built += …; built.endsWith(head)`) is O(n²) on a long document and the
  // cost is not theoretical: measured on the live index it took **14.7 seconds**
  // to reconstruct a 3,606-chunk document, because `endsWith` flattens V8's
  // rope on every iteration while the rope keeps growing. The array-plus-join
  // form does the same splice in ~2 ms.
  const pieces: string[] = [];
  let previous: string | undefined;
  for (const chunk of chunks) {
    if (chunk.kind !== "content") continue;
    // The overlap is compared against the PREVIOUS CHUNK, not against the
    // accumulated text. It is the same 200 characters — the chunker emits
    // consecutive slices at a stride of CHUNK_SIZE - CHUNK_OVERLAP — but
    // comparing a short, already-flat string is O(200) rather than O(document).
    let head = 0;
    if (previous !== undefined) {
      const probe = Math.min(CHUNK_OVERLAP, previous.length, chunk.text.length);
      if (probe > 0 && previous.slice(-probe) === chunk.text.slice(0, probe)) {
        head = probe;
      }
    }
    pieces.push(chunk.text.slice(head));
    previous = chunk.text;
  }
  return pieces.join("");
}

/**
 * The production `loadText`: document text straight out of the open index.
 *
 * One prepared SELECT per document, against a local SQLite file, replacing one
 * HTTPS GET per document that returned the whole body. Measured on the live
 * index (read + splice): 2 ms for a typical document, 34–42 ms for the largest
 * (3,606 chunks / 7.2 M characters).
 *
 * A document whose chunks cannot be read is simply absent from the map, which
 * makes the caller fall back to `doc.content` — `null` on the batched-metadata
 * path, hence an empty passage rather than an error. That is the same
 * "one unusable document must not fail the search" tolerance the fetch path
 * had, and `content_length: 0` makes it visible instead of silent.
 */
export function indexTextLoader(
  store: SearchIndex
): (hits: Fused[]) => Map<number, string> {
  return (hits) => {
    const text = new Map<number, string>();
    for (const id of new Set(hits.map((hit) => hit.documentId))) {
      try {
        text.set(id, reconstructDocumentText(store.chunksFor(id)));
      } catch {
        // Absent on purpose: reported downstream as an empty passage.
      }
    }
    return text;
  };
}

/**
 * Trim a result until its own serialised body fits the consumer's ceiling.
 *
 * Measured on the REAL bytes, not predicted from `window * passages`. That
 * distinction is the whole mechanism: titles, ids and counts vary, and an
 * arithmetic budget that is right on average is exactly the kind of bound that
 * fails on the one call that matters. `budgetedLimit` keeps the work small;
 * this is what makes the guarantee actual.
 */
export function fitResultToBudget(result: SemanticSearchResult): SemanticSearchResult {
  if (JSON.stringify(result).length <= MAX_RESULT_CHARS) return result;
  // ⚠️ A PASSAGE-ONLY TRIM IS NOT A BOUND. Measured: `fitResultToBudget` dropped
  // passages and never touched the ECHOED QUERY, which has no length limit -- a
  // 30,000-char query returned 30,128 chars with zero passages, over the ceiling
  // the function exists to enforce. Dropping every passage cannot fix an
  // oversize field that is not a passage. Echo less.
  const echo = typeof result.query === "string" ? result.query : "";
  const passages = [...result.passages];
  let dropped = 0;
  let candidate: SemanticSearchResult = result;
  while (passages.length > 0) {
    passages.pop();
    dropped += 1;
    candidate = {
      ...result,
      passages,
      trimmed: { reason: "result_size", dropped, limit_chars: MAX_RESULT_CHARS },
    };
    if (JSON.stringify(candidate).length <= MAX_RESULT_CHARS) return candidate;
  }
  // Every passage is gone and it is STILL over budget: the query echo alone is
  // too large. Truncate the echo rather than return an oversize result, and say
  // so -- a silently truncated query is indistinguishable from a short one.
  //
  // ⚠️ MEASURE AFTER ADDING THE FLAG, NOT BEFORE. The first cut of this computed
  // the overflow against the payload WITHOUT `query_truncated`, then added that
  // field -- which is 24 chars -- and returned 20,074 against a 20,000 ceiling.
  // A bound that its own error message breaks is not a bound, and the test
  // caught it only because it asserted the serialised length rather than the
  // passage count. Loop until it actually fits; the flag costs bytes, so assume
  // it is present from the first attempt.
  const declared = {
    ...candidate,
    trimmed: {
      reason: "result_size" as const,
      dropped,
      limit_chars: MAX_RESULT_CHARS,
      query_truncated: true,
    },
  };
  let keep = echo.length;
  for (;;) {
    const over = JSON.stringify({ ...declared, query: echo.slice(0, keep) }).length - MAX_RESULT_CHARS;
    if (over <= 0 || keep === 0) break;
    keep = Math.max(0, keep - over - 1);
  }
  return { ...declared, query: echo.slice(0, keep) };
}

export interface EmbeddingStatus {
  documents: number;
  chunks: number;
  dimensions: number;
  model: string;
  prefix_mode: string;
  last_sync: string | null;
  pending: number | null;
  store_bytes: number | null;
  truncated_documents: number;
  index_path: string;
  open_error?: string;
}

/**
 * Report the store's own view of itself.
 *
 * `pending` is computed against Paperless' document count, so it is `null`
 * rather than `0` when that count cannot be read: a zero here would say "the
 * index is current" on the strength of a failed request.
 */
export async function embeddingStatus(
  context: SearchContext,
  api: PaperlessAPI
): Promise<EmbeddingStatus> {
  const config = context.config();
  const path = config ? indexPath(config) : "";
  const base: EmbeddingStatus = {
    documents: 0,
    chunks: 0,
    dimensions: 0,
    model: "",
    prefix_mode: "",
    last_sync: null,
    pending: null,
    store_bytes: null,
    truncated_documents: 0,
    index_path: path,
  };

  let store: SearchIndex;
  try {
    store = context.index();
  } catch (err) {
    return { ...base, open_error: errorMessage(err) };
  }

  const counts = store.counts();
  const meta = store.meta();
  const documents = store.indexed();
  const stamps = store.metadataStamps();

  let lastSync: string | null = null;
  for (const stamp of stamps.values()) {
    if (lastSync === null || stamp > lastSync) lastSync = stamp;
  }

  // Truncation is the one index fact that changes what a caller can conclude
  // from a miss: a truncated document lost its tail, so "not found" may be a
  // false negative. Counted with one query, not one query per document.
  const truncated = store.truncatedCount();

  let storeBytes: number | null = null;
  try {
    const { statSync } = await import("node:fs");
    storeBytes = statSync(path).size;
  } catch {
    // Not an error: a fresh index file may not exist yet.
  }

  let pending: number | null = null;
  try {
    const page = (await api.request(
      "/documents/?fields=id&page_size=1"
    )) as { count: number };
    if (typeof page.count === "number") {
      pending = Math.max(0, page.count - documents.size);
    }
  } catch {
    // Left null: a failed count is not a count of zero.
  }

  return {
    ...base,
    documents: counts.documents,
    chunks: counts.chunks,
    dimensions: meta?.dimensions ?? 0,
    model: meta?.model ?? "",
    prefix_mode: meta?.prefixMode ?? "",
    last_sync: lastSync,
    pending,
    store_bytes: storeBytes,
    truncated_documents: truncated,
  };
}
