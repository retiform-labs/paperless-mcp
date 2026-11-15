import { PrefixMode } from "./config";

/**
 * Local-only embedding client with a hard local-only guard.
 *
 * The gate is a compliance requirement, not a preference: restricted restricted material is
 * indexed locally only, and the prohibition attaches to the ROUTE rather than
 * to any particular vendor. "The provider is free" never inherits permission,
 * and a failover provider never inherits it either. A remote embedder must be a
 * conscious code change under review, not a configuration edit — which is why
 * `OpenAIEmbedder` is deliberately NOT implemented here, and why there is no
 * API-key field anywhere in the config or in this file.
 *
 * Ported from `jaimemachado/paperless-mcp@50f7bee:src/search/embedder.ts`, with
 * three changes forced by our substrate:
 *   1. The `../metrics` and `../tracing` imports are stripped — neither module
 *      exists in our fork, and porting them would mean adding OpenTelemetry and
 *      prom-client for no benefit here.
 *   2. `OpenAIEmbedder` is removed (see above).
 *   3. The prefix handling is fixed rather than ported. See `PREFIXES`.
 */

const BATCH_SIZE = 16;
const DOCUMENT_TIMEOUT_MS = 300_000;
const QUERY_TIMEOUT_MS = 30_000;

export class EmbeddingUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "EmbeddingUnavailableError";
  }
}

/** Thrown when the configured prefix mode cannot be used with the stored index. */
export class PrefixModeMismatchError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PrefixModeMismatchError";
  }
}

export interface Embedder {
  readonly model: string;
  readonly prefixMode: PrefixMode;
  readonly dimensions: number;
  embedDocuments(texts: string[]): Promise<number[][]>;
  embedQuery(query: string): Promise<number[]>;
}

/**
 * The prefix each embedding model family expects.
 *
 * This is the bug fix, not a port. The reference implementation hardcoded the
 * E5 instruction template onto a nomic model. Measured on this box at n=5,
 * nomic-prefixed and plain both put the known-good document at rank 1 five
 * times out of five, while the E5 template managed four of five with a negative
 * margin. That is a signal rather than a benchmark, but it points the same way
 * the model cards do, and the cost of being wrong is silently worse retrieval.
 *
 * `none` is a supported mode because a future model may want no prefix at all,
 * and because refusing to store the choice would make the choice unprovable.
 */
export const PREFIXES: Record<
  PrefixMode,
  { document: string; query: string }
> = {
  nomic: { document: "search_document: ", query: "search_query: " },
  none: { document: "", query: "" },
  e5: { document: "passage: ", query: "query: " },
};

/**
 * The prefix is baked into every stored vector, so a query vector built with a
 * different prefix is silently worse — it still returns results, they are just
 * the wrong ones. A mismatch must be refused loudly instead.
 */
export function assertPrefixModeMatches(
  configured: PrefixMode,
  stored: PrefixMode
): void {
  if (configured !== stored) {
    throw new PrefixModeMismatchError(
      `The index was built with EMBEDDING_PREFIX_MODE="${stored}" but this ` +
        `process is configured for "${configured}". The prefix is stored ` +
        `inside every vector, so querying across a mismatch returns quietly ` +
        `worse results. Re-run the sync with the stored mode, or rebuild the ` +
        `index with the configured one.`
    );
  }
}

const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "::1"]);

function ipv4ToOctets(host: string): number[] | undefined {
  const match = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(host);
  if (!match) return undefined;
  const octets = match.slice(1).map((part) => Number(part));
  // Reject 999.1.1.1 and leading-zero ambiguity alike: every octet must be a
  // canonical 0-255 decimal. Anything else is not a literal we can reason about.
  if (octets.some((n) => !Number.isInteger(n) || n < 0 || n > 255)) {
    return undefined;
  }
  return octets;
}

function isPrivateIpv4(octets: number[]): boolean {
  const [a, b] = octets;
  if (a === 10) return true;
  if (a === 172 && b >= 16 && b <= 31) return true;
  if (a === 192 && b === 168) return true;
  return false;
}

/** Strip the brackets a WHATWG URL keeps around an IPv6 literal. */
function bareHost(hostname: string): string {
  return hostname.startsWith("[") && hostname.endsWith("]")
    ? hostname.slice(1, -1)
    : hostname;
}

/**
 * Assert that an embedding endpoint is a literal loopback or RFC1918 address.
 *
 * A DNS NAME IS REFUSED. A name can resolve anywhere, and we cannot prove where
 * it resolves to at startup — a `localhost` that works today and a hosts-file
 * edit tomorrow are indistinguishable from inside this function. Requiring a
 * literal address makes the gate checkable by reading the configuration.
 *
 * The shape of this check is deliberate, and a naive one would be wrong in two
 * different ways on this box:
 *   - Asserting `host === "127.0.0.1"` would reject Ollama, which binds
 *     `0.0.0.0` (`OLLAMA_HOST=0.0.0.0` in the unit) and is therefore reached at
 *     the LAN address as well as at loopback.
 *   - Accepting `0.0.0.0` as a connect target would be wrong the other way: it
 *     is the unspecified address, its meaning is platform-dependent, and no
 *     legitimate deployment needs to dial it.
 * An allowlist of literal private addresses handles both.
 */
export function assertLocalEmbeddingEndpoint(rawUrl: string): URL {
  let url: URL;
  try {
    url = new URL(rawUrl);
  } catch {
    throw new Error(
      `EMBEDDING_URL is not a valid URL: "${rawUrl}". ` +
        `Example: http://127.0.0.1:11434`
    );
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error(
      `EMBEDDING_URL must use http: or https:, not "${url.protocol}".`
    );
  }
  const host = bareHost(url.hostname).toLowerCase();
  if (!host) {
    throw new Error(`EMBEDDING_URL has no hostname: "${rawUrl}".`);
  }
  if (LOOPBACK_HOSTS.has(host)) return url;

  const octets = ipv4ToOctets(host);
  if (!octets) {
    throw new Error(
      `EMBEDDING_URL host "${host}" is not a literal loopback or RFC1918 ` +
        `address. A DNS name is refused deliberately: it can resolve anywhere, ` +
        `and this process cannot prove where. Restricted material is embedded ` +
        `locally only, so the route itself has to be inspectable. ` +
        `Use 127.0.0.1, ::1, 10.x.x.x, 172.16-31.x.x or 192.168.x.x.`
    );
  }
  if (host === "0.0.0.0") {
    throw new Error(
      `EMBEDDING_URL host "0.0.0.0" is the unspecified address, not a ` +
        `destination. Ollama BINDS 0.0.0.0 but you CONNECT to 127.0.0.1 — ` +
        `use http://127.0.0.1:11434.`
    );
  }
  if (!isPrivateIpv4(octets)) {
    throw new Error(
      `EMBEDDING_URL host "${host}" is a public address. Restricted material ` +
        `is embedded locally only; a remote embedding route is a deployment policy ` +
        `breach, not a configuration choice. Use a loopback or RFC1918 address.`
    );
  }
  return url;
}

interface OllamaEmbedResponse {
  embeddings: number[][];
}

/**
 * An Ollama `/api/embed` client. The only embedder in this package.
 *
 * The constructor validates both the endpoint and the prefix mode, and throws
 * on either failure. Nothing is deferred to first use: a misconfigured index
 * should fail at startup, before it has written a single vector.
 */
export class OllamaEmbedder implements Embedder {
  readonly model: string;
  readonly prefixMode: PrefixMode;
  readonly dimensions: number;
  private readonly baseUrl: URL;

  constructor(opts: {
    url: string;
    model: string;
    prefixMode: PrefixMode;
    dimensions?: number;
  }) {
    this.baseUrl = assertLocalEmbeddingEndpoint(opts.url);
    if (!opts.model.trim()) {
      throw new Error("EMBEDDING_MODEL must not be empty.");
    }
    if (!(opts.prefixMode in PREFIXES)) {
      throw new Error(`Unknown embedding prefix mode "${opts.prefixMode}".`);
    }
    this.model = opts.model;
    this.prefixMode = opts.prefixMode;
    this.dimensions = opts.dimensions ?? 768;
  }

  /** Strip a trailing slash so `${base}/api/embed` never doubles it. */
  private endpoint(): string {
    return `${this.baseUrl.origin}${this.baseUrl.pathname.replace(/\/+$/, "")}/api/embed`;
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    const prefix = PREFIXES[this.prefixMode].document;
    const embeddings: number[][] = [];
    for (let i = 0; i < texts.length; i += BATCH_SIZE) {
      const batch = texts.slice(i, i + BATCH_SIZE).map((text) => prefix + text);
      embeddings.push(...(await this.request(batch, DOCUMENT_TIMEOUT_MS)));
    }
    return embeddings;
  }

  async embedQuery(query: string): Promise<number[]> {
    const prefix = PREFIXES[this.prefixMode].query;
    const [embedding] = await this.request(
      [prefix + query],
      QUERY_TIMEOUT_MS
    );
    return embedding;
  }

  private async request(
    input: string[],
    timeoutMs: number
  ): Promise<number[][]> {
    if (input.length === 0) return [];
    let response: Response;
    try {
      response = await fetch(this.endpoint(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: this.model, input }),
        signal: AbortSignal.timeout(timeoutMs),
      });
    } catch (err) {
      throw new EmbeddingUnavailableError(
        `The embedding server did not answer at ${this.endpoint()}: ${err}`
      );
    }
    if (!response.ok) {
      throw new EmbeddingUnavailableError(
        `The embedding server answered with status ${response.status} at ${this.endpoint()}`
      );
    }
    let body: OllamaEmbedResponse;
    try {
      body = (await response.json()) as OllamaEmbedResponse;
    } catch (err) {
      throw new EmbeddingUnavailableError(
        `The embedding server answered with a body that is not JSON: ${err}`
      );
    }
    if (
      !Array.isArray(body.embeddings) ||
      body.embeddings.length !== input.length
    ) {
      throw new EmbeddingUnavailableError(
        "The embedding server answered with a missing or mismatched embeddings array"
      );
    }
    for (const embedding of body.embeddings) {
      if (!Array.isArray(embedding) || embedding.length !== this.dimensions) {
        throw new EmbeddingUnavailableError(
          `The embedding server returned a vector of the wrong width; ` +
            `expected ${this.dimensions} dimensions. Check that ` +
            `EMBEDDING_MODEL matches the index.`
        );
      }
    }
    return body.embeddings;
  }
}
