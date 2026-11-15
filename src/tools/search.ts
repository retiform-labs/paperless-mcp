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
import { SearchIndex, LexicalUnavailableError, type Hit } from "../search/store";
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
 * filter, this is the one place that has to learn it. They are applied in JS
 * against the chunk's document id rather than pushed into the kNN statement,
 * because the store returns one hit per document and a filter that removed a
 * document after the fact would silently shrink the result set below `limit`
 * with no way for the caller to know the ranking had been trimmed.
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
  total_hits: number;
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
    const store = new SearchIndex(indexPath(config));
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

  close(): void {
    this.store?.close();
    this.store = undefined;
    this.sync = undefined;
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
  let degradedReason: string | undefined;
  let vector: Hit[] = [];
  let text: Hit[] = [];

  // The lexical half needs no network and no model. It runs unconditionally and
  // is the reason a degraded call still returns something.
  try {
    // Over-fetch for the vector side's sake: the store returns one hit per
    // document and filters are applied below, so fetching exactly `limit` would
    // let one filtered-out document empty the result.
    text = store.textHits(query, limit * 4);
  } catch (err) {
    if (!(err instanceof LexicalUnavailableError)) throw err;
    // No FTS5 in this build. Not an embedding problem, and NOT an empty
    // result -- `semantic` stays "ok" for the vector side and the reason is
    // carried so the caller knows the ranking has only one input.
    degradedReason = `Lexical search is unavailable in this build: ${errorMessage(err)}`;
  }

  const embedder = context.embedderOrUndefined();
  if (!embedder) {
    semantic = "unavailable";
    degradedReason ??= "The embedding client is not configured.";
  } else {
    try {
      const embedding = await embedder.embedQuery(query);
      vector = store.vectorHits(embedding, limit * 4);
    } catch (err) {
      if (!(err instanceof EmbeddingUnavailableError)) throw err;
      // EXPECTED. The lexical list above already holds whatever could be found.
      semantic = "unavailable";
      degradedReason = errorMessage(err);
    }
  }

  // No FTS5 AND no embedder means there is genuinely nothing to search with.
  // Refuse rather than return an empty list that reads as "no such document".
  if (text.length === 0 && vector.length === 0 && degradedReason && semantic === "unavailable") {
    throw new Error(
      `Semantic search is fully unavailable and no lexical fallback exists: ${degradedReason}`
    );
  }

  let fused: Fused[] = fuse(vector, text);
  if (fused.length === 0) {
    return {
      query,
      semantic,
      ...(degradedReason ? { degraded_reason: degradedReason } : {}),
      strong_match: false,
      total_hits: 0,
      passages: [],
    };
  }

  // Anchoring and filters both need the documents, so fetch each visible id
  // once. The documents themselves are what the caller is shown; the store's
  // chunk text is only a candidate.
  const documents = await opts.loadDocuments(fused.map((hit) => hit.documentId));

  const passages: SemanticPassage[] = [];
  for (const hit of fused) {
    const doc = documents.get(hit.documentId);
    if (!doc) continue;
    if (opts.filters && !opts.filters(doc)) continue;
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

  return {
    query,
    semantic,
    ...(degradedReason ? { degraded_reason: degradedReason } : {}),
    strong_match: strong,
    total_hits: fused.length,
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
  // storage_path is not on the lightweight document shape this path fetches, so
  // a caller passing it would otherwise be silently ignored -- the filter is
  // named here so the omission is a decision rather than an oversight.
  const active =
    filters.correspondent !== undefined ||
    filters.document_type !== undefined ||
    filters.tag !== undefined ||
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
