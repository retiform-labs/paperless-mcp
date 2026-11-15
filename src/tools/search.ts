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
import { searchConfig, indexPath, type SearchConfig } from "../search/config";
import { OllamaEmbedder, EmbeddingUnavailableError, type Embedder } from "../search/embedder";
import {
  EMBEDDING_DIMENSIONS,
  SearchIndex,
  LexicalUnavailableError,
  type Hit,
} from "../search/store";
import { fuse, type Fused } from "../search/rank";
import { IndexSync, type SyncReport } from "../search/sync";
import { withErrorHandling } from "./utils/middlewares";

/** A similarity at or above this is a strong match. Similarity, never distance. */
export const STRONG_MATCH = 0.45;

/** Chars per returned passage, matching search_passages' default. */
export const DEFAULT_WINDOW = 800;
export const DEFAULT_LIMIT = 10;
export const MAX_LIMIT = 50;

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
  score?: number;
  highlights?: string;
  matched: "meaning" | "words" | "both";
  passage: string;
  offset: number;
  content_length: number;
}

export interface SemanticSearchResult {
  query: string;
  semantic: "ok" | "unavailable";
  degraded_reason?: string;
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
  passages: SemanticPassage[];
}

/**
 * Cut a bounded passage around the first place the query terms appear.
 *
 * Mirrors buildSearchPassage's anchoring priority minus the server highlight,
 * which does not exist on this path: the vector side has no highlighted span,
 * and the lexical side's BM25 match gives a chunk rather than an offset. So the
 * anchor is (1) the first verbatim query term, (2) the head of the document.
 *
 * A window that does not contain the hit is worse than no window, which is why
 * the term search runs against the whole document rather than the chunk.
 */
export function buildPassage(
  doc: { id: number; title: string; content: string | null },
  terms: string[],
  window: number
): { passage: string; offset: number; content_length: number } {
  const full = doc.content ?? "";
  const lower = full.toLowerCase();
  let center = -1;
  for (const term of terms) {
    const at = lower.indexOf(term.toLowerCase());
    if (at >= 0) {
      center = at;
      break;
    }
  }
  const resolved = center < 0 ? 0 : center;
  const start = Math.max(0, resolved - Math.floor(window / 2));
  return {
    passage: full.slice(start, start + window),
    offset: start,
    content_length: full.length,
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

  close(): void {
    // Stop the timer BEFORE dropping the reference, or a stdio child shutting
    // down leaves an interval firing against a closed store.
    this.sync?.stop();
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
    filters?: (doc: VisibleDocument) => boolean;
  }
): Promise<SemanticSearchResult> {
  const store = context.index();
  const limit = Math.min(Math.max(opts.limit, 1), MAX_LIMIT);
  const terms = queryTerms(query);

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
      // State (a) on its own terms: nothing matched and the payload says so —
      // `semantic`, `retrievers`, and an empty list together are unambiguous.
      passages: [],
    };
  }

  // Anchoring and filters both need the documents, so fetch each visible id
  // once. The documents themselves are what the caller is shown; the store's
  // chunk text is only a candidate.
  const documents = await opts.loadDocuments(fused.map((hit) => hit.documentId));

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
    const anchored = buildPassage(doc, terms, opts.window);
    passages.push({
      id: doc.id,
      title: doc.title,
      score: hit.score,
      matched: hit.matched,
      ...anchored,
    });
    if (passages.length >= limit) break;
  }

  const strong = passages.some((passage) => {
    if (passage.matched === "words" || passage.matched === "both") return true;
    const source = fused.find((hit) => hit.documentId === passage.id);
    return source?.bestSimilarity !== null && source?.bestSimilarity !== undefined
      ? source.bestSimilarity >= STRONG_MATCH
      : false;
  });

  // `filtered_out` counts what the loop dropped, NOT `total_hits - passages`
  // -- those differ when the `limit` break fires with hits still unexamined.
  // The honest decomposition is: fused = shown + dropped + never-reached.
  const filteredOut = documentUnavailable + metadataFiltered;

  return {
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
    passages,
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

  const filterArgs = FILTER_SHAPE;

  server.tool(
    "semantic_search",
    "Search the document corpus by MEANING as well as by words, using a local vector index fused with the corpus' own BM25 lexical search. Returns bounded passages in the same shape as search_passages (id, title, score, passage, offset, content_length) plus `matched`, which says whether the hit came from meaning, words, or both -- a 'words' hit is a lexical match the vector side did not find, so do not read it as absence of a paraphrase. HONEST DEGRADATION: `semantic` is 'unavailable' and lexical results are still returned when the local embedding server is down; a degraded call never returns a silent empty list. strong_match is false when nothing scored above the similarity floor. Requires the local index to have been built with sync_embeddings.",
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
      ...filterArgs,
    },
    withErrorHandling(async (args) => {
      const { limit, window, ...filters } = args;
      const result = await runSemanticSearch(context, args.query, {
        limit: limit ?? DEFAULT_LIMIT,
        window: window ?? DEFAULT_WINDOW,
        loadDocuments: (ids) => loadVisibleDocuments(api, ids),
        filters: buildDocumentFilter(filters),
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
      const report: SyncReport = await runner.run();
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
