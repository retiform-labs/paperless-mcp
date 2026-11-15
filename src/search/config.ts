import { homedir } from "node:os";
import { join } from "node:path";

/**
 * Semantic-search configuration for the local SQLite index.
 *
 * Deliberate differences from the reference implementation this was ported
 * from (`jaimemachado/paperless-mcp@50f7bee`):
 *
 *  - A remote embedding provider IS reachable by configuration, because this
 *    package is published for general use and a third-party install must be
 *    able to point it at whichever provider it has. What protects a deployment
 *    that must not send text off the host is a flag it sets --
 *    `PAPERLESS_MCP_EMBEDDING_LOCAL_ONLY=true` -- not a restriction compiled
 *    into the library on its behalf.
 *  - There is NO `useHttp` gate. That gate made semantic search dead on arrival
 *    in our deployment, because the gateway child is launched with no CLI args
 *    (`args: ["/path/to/build/index.js"]`) and so `--http`
 *    is never true. Nothing about a local sqlite index requires HTTP transport.
 *  - There is NO `DATABASE_URL`. The store is a file path.
 *  - Every value here has a working default, so "all four or none" would be
 *    meaningless. The rule instead is: any search variable present enables the
 *    feature; an invalid value refuses loudly rather than silently defaulting.
 */

export const PREFIX_MODES = ["nomic", "none", "e5"] as const;
export type PrefixMode = (typeof PREFIX_MODES)[number];

export interface SearchConfig {
  readonly embeddingUrl: string;
  readonly model: string;
  readonly prefixMode: PrefixMode;
  readonly indexDir: string;
  readonly maxChunks: number;
  readonly syncBatchSize: number;
  readonly syncIntervalMs: number;
  /** Restrict the embedding endpoint to a literal loopback/RFC1918 address. */
  readonly embeddingLocalOnly: boolean;
  /** Bearer token for the embedding endpoint, when one is required. */
  readonly embeddingApiKey: string | undefined;
}

/**
 * Rowid stride for the vector index: a chunk's rowid is
 * `docId * CHUNK_ROWID_STRIDE + position`, so a document can hold at most
 * `CHUNK_ROWID_STRIDE - 1` chunks before two documents collide.
 *
 * It lives in config.ts rather than store.ts so the CONFIG derives its own
 * upper bound from the one number the store actually encodes with. A second
 * literal would be a second source of truth, and the cost of the two
 * disagreeing is paid mid-build: `PAPERLESS_MCP_MAX_CHUNKS=100000` was accepted
 * by config and could only ever be rejected by `chunkRowid`, on document 2,000
 * of a 58-minute run.
 *
 * A power of two, so a position decodes with a mask rather than a division that
 * has to be exactly right.
 */
export const CHUNK_ROWID_STRIDE = 8192;

/** The largest chunk count the stride can encode: positions 0 .. stride-1. */
export const MAX_CHUNKS_LIMIT = CHUNK_ROWID_STRIDE - 1;

export const DEFAULT_EMBEDDING_URL = "http://127.0.0.1:11434";
export const DEFAULT_EMBEDDING_MODEL = "nomic-embed-text";
export const DEFAULT_PREFIX_MODE: PrefixMode = "nomic";
export const DEFAULT_MAX_CHUNKS = 4000;
export const DEFAULT_SYNC_BATCH_SIZE = 16;
/**
 * Off by default. An install that must be able to prove indexed text does not
 * leave the host turns it on; one that is pointing at a hosted provider must
 * not have to fight a policy it never asked for.
 */
export const DEFAULT_EMBEDDING_LOCAL_ONLY = false;
/** 0 means "no background sync". Opting in is explicit, never a default. */
export const DEFAULT_SYNC_INTERVAL_MS = 0;

export function defaultIndexDir(): string {
  return join(homedir(), ".local", "share", "paperless-mcp", "index");
}

/**
 * Each variable has a namespaced name and a generic fallback. The namespaced
 * form wins. The generic form exists only because the reference implementation
 * used it; the gateway passes one environment to all of its children, so a
 * generic `EMBEDDING_URL` is a collision surface, not a convenience.
 */
interface EnvSpec {
  preferred: string;
  fallback?: string;
}

const SPECS = {
  embeddingUrl: { preferred: "PAPERLESS_MCP_EMBEDDING_URL", fallback: "EMBEDDING_URL" },
  model: { preferred: "PAPERLESS_MCP_EMBEDDING_MODEL", fallback: "EMBEDDING_MODEL" },
  prefixMode: { preferred: "PAPERLESS_MCP_EMBEDDING_PREFIX_MODE", fallback: "EMBEDDING_PREFIX_MODE" },
  indexDir: { preferred: "PAPERLESS_MCP_INDEX_DIR" },
  maxChunks: { preferred: "PAPERLESS_MCP_MAX_CHUNKS" },
  syncBatchSize: { preferred: "PAPERLESS_MCP_SYNC_BATCH_SIZE" },
  syncIntervalMs: { preferred: "PAPERLESS_MCP_SYNC_INTERVAL_MS", fallback: "SYNC_INTERVAL_MS" },
  embeddingLocalOnly: { preferred: "PAPERLESS_MCP_EMBEDDING_LOCAL_ONLY" },
  embeddingApiKey: { preferred: "PAPERLESS_MCP_EMBEDDING_API_KEY", fallback: "EMBEDDING_API_KEY" },
} as const satisfies Record<string, EnvSpec>;

function readVar(
  env: Record<string, string | undefined>,
  spec: EnvSpec
): { name: string; value: string } | undefined {
  const names = spec.fallback ? [spec.preferred, spec.fallback] : [spec.preferred];
  for (const name of names) {
    const value = env[name]?.trim();
    if (value) return { name, value };
  }
  return undefined;
}

interface NumberBounds {
  min: number;
  max: number;
  integer: boolean;
  /**
   * Clamp a value above `max` down to `max` instead of refusing it.
   *
   * Set for bounds that express a CAPABILITY rather than a preference. A caller
   * asking for more chunks than the index can encode is asking for the most it
   * can have, not making a mistake worth failing a build over; whereas a value
   * BELOW `min` (0, negative) is a mistake that clamping would hide.
   */
  clampMax?: boolean;
}

function toNumber(where: string, raw: string, bounds: NumberBounds): number {
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${where} must be a number, not "${raw}".`);
  }
  if (bounds.integer && !Number.isInteger(parsed)) {
    throw new Error(`${where} must be a whole number, not "${raw}".`);
  }
  if (parsed < bounds.min || (!bounds.clampMax && parsed > bounds.max)) {
    throw new Error(
      `${where} must be between ${bounds.min} and ${bounds.max}, not "${raw}".`
    );
  }
  // Above the ceiling with clamping permitted: the caller asked for more than
  // the store can encode, which is a ceiling rather than an error.
  if (parsed > bounds.max) return bounds.max;
  return parsed;
}

/**
 * Parse a boolean flag strictly.
 *
 * A flag that silently treats `"yes"`, `"1"` or a typo as false is worse than
 * one that refuses: an operator who set a safety flag and got the default
 * instead has been told the opposite of the truth by a green startup. So the
 * accepted forms are enumerated and everything else throws.
 */
function toBoolean(where: string, raw: string): boolean {
  const lowered = raw.toLowerCase();
  if (lowered === "true" || lowered === "1") return true;
  if (lowered === "false" || lowered === "0") return false;
  throw new Error(`${where} must be true or false, not "${raw}".`);
}

function toPrefixMode(where: string, raw: string): PrefixMode {
  const modes: readonly string[] = PREFIX_MODES;
  if (!modes.includes(raw)) {
    throw new Error(
      `${where} must be one of ${PREFIX_MODES.join(", ")}, not "${raw}".`
    );
  }
  return raw as PrefixMode;
}

/** Resolve a leading `~` without pulling in a shell. */
export function expandHome(input: string): string {
  if (input === "~") return homedir();
  if (input.startsWith("~/")) return join(homedir(), input.slice(2));
  return input;
}

/**
 * Returns the full configuration, or `undefined` when the feature is off.
 *
 * Off means: none of the search variables are set. Setting any one of them
 * turns the feature on, and every set value is then validated strictly —
 * a bad `EMBEDDING_PREFIX_MODE` throws rather than falling back to `nomic`,
 * because a silent fallback here would silently change every stored vector.
 */
export function searchConfig(
  env: Record<string, string | undefined>
): SearchConfig | undefined {
  const embeddingUrl = readVar(env, SPECS.embeddingUrl);
  const model = readVar(env, SPECS.model);
  const prefixMode = readVar(env, SPECS.prefixMode);
  const indexDir = readVar(env, SPECS.indexDir);
  const maxChunks = readVar(env, SPECS.maxChunks);
  const syncBatchSize = readVar(env, SPECS.syncBatchSize);
  const syncIntervalMs = readVar(env, SPECS.syncIntervalMs);
  const embeddingLocalOnly = readVar(env, SPECS.embeddingLocalOnly);
  const embeddingApiKey = readVar(env, SPECS.embeddingApiKey);

  // ⚠️ `embeddingApiKey` deliberately does NOT appear in this list, and the
  // asymmetry is the point: a credential is a modifier, not a request to turn
  // the feature on. Counting it here would mean an environment that happens to
  // export a provider key silently switches on semantic search for an install
  // that never configured an index.
  const anySet = [
    embeddingUrl,
    model,
    prefixMode,
    indexDir,
    maxChunks,
    syncBatchSize,
    syncIntervalMs,
    embeddingLocalOnly,
  ].some(Boolean);
  if (!anySet) return undefined;

  return {
    embeddingUrl: embeddingUrl?.value ?? DEFAULT_EMBEDDING_URL,
    model: model?.value ?? DEFAULT_EMBEDDING_MODEL,
    prefixMode: prefixMode
      ? toPrefixMode(prefixMode.name, prefixMode.value)
      : DEFAULT_PREFIX_MODE,
    indexDir: expandHome(indexDir?.value ?? defaultIndexDir()),
    maxChunks: maxChunks
      ? toNumber(maxChunks.name, maxChunks.value, {
          min: 1,
          max: MAX_CHUNKS_LIMIT,
          integer: true,
          clampMax: true,
        })
      : DEFAULT_MAX_CHUNKS,
    syncBatchSize: syncBatchSize
      ? toNumber(syncBatchSize.name, syncBatchSize.value, { min: 1, max: 256, integer: true })
      : DEFAULT_SYNC_BATCH_SIZE,
    syncIntervalMs: syncIntervalMs
      ? toNumber(syncIntervalMs.name, syncIntervalMs.value, { min: 0, max: 86400000, integer: true })
      : DEFAULT_SYNC_INTERVAL_MS,
    embeddingLocalOnly: embeddingLocalOnly
      ? toBoolean(embeddingLocalOnly.name, embeddingLocalOnly.value)
      : DEFAULT_EMBEDDING_LOCAL_ONLY,
    embeddingApiKey: embeddingApiKey?.value,
  };
}

/** The index file inside the configured directory. */
export function indexPath(config: SearchConfig): string {
  return join(config.indexDir, "index.db");
}
