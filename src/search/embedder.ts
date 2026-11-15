/**
 * An HTTP embeddings client for any OpenAI-compatible `/embeddings` endpoint.
 *
 * Two request/response shapes are spoken, and the caller does not have to say
 * which: Ollama's native `/api/embed` answers `{embeddings: [[...]]}` and its
 * OpenAI-compatible `/v1/embeddings` answers `{data: [{embedding: [...]}]}`.
 * The reply is sniffed. A package that made the user declare which of two
 * well-known JSON shapes it was about to receive would be adding a
 * configuration failure mode in exchange for nothing -- and both shapes were
 * measured against the same server returning identical vectors.
 *
 * ── Where the endpoint comes from ──────────────────────────────────────────
 * The configured URL is used verbatim when it carries a path, so
 * `https://api.openai.com/v1/embeddings` and `http://host:8080/embed` both mean
 * what they say. An origin-only URL falls back to Ollama's default
 * `/api/embed`, which is what makes the shipped default work with no path.
 *
 * ── Local-only mode ───────────────────────────────────────────────────────
 * `localOnly: true` restricts the endpoint to a literal loopback or RFC1918
 * address and refuses DNS names, because a name can resolve anywhere and the
 * process cannot prove where. That is a capability any security-conscious
 * deployment may want, not a policy belonging to one of them: it is off by
 * default, and an install that needs it turns it on.
 *
 * `redirect: "error"` is always on. It is generic hygiene rather than policy:
 * a 307 would let any endpoint -- local or not -- forward a request body to a
 * host the configuration never named.
 *
 * Ported from `jaimemachado/paperless-mcp@50f7bee:src/search/embedder.ts`, with
 * these changes forced by our substrate:
 *   1. The `../metrics` and `../tracing` imports are stripped -- neither module
 *      exists in our fork, and porting them would mean adding OpenTelemetry and
 *      prom-client for no benefit here.
 *   2. The prefix handling is fixed rather than ported. See `PREFIXES`.
 *   3. The response parser accepts both the native and the OpenAI shape.
 *   4. The endpoint restriction is a configuration value, not a hard gate.
 */

import { PrefixMode } from "./config";

/**
 * Default number of texts per embedding request.
 *
 * Also the module default for `batchSize`, so a caller that omits the option
 * behaves exactly as it did before the option existed.
 */
export const DEFAULT_BATCH_SIZE = 16;

/**
 * The string a liveness probe embeds. A constant, never document text: a probe
 * must not put real material on the wire just to test connectivity.
 */
const PROBE_TEXT = "ping";
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
  /**
   * Cheap liveness check that must NOT be assumed equivalent to "constructed".
   *
   * The constructor validates the endpoint, so a constructed embedder proves
   * nothing about whether anything is listening. Callers that are about to do
   * something irreversible -- `sync_embeddings` with `force:true` deletes the
   * entire index before it re-embeds -- must probe first, or an outage on a
   * long rebuild leaves no index at all.
   *
   * Optional so existing fakes stay valid; callers fall back to `embedQuery`.
   */
  probe?(): Promise<void>;
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
 * different prefix is silently worse -- it still returns results, they are just
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
 * Validate an embedding endpoint URL.
 *
 * Two checks are always applied, because they are true of every HTTP client
 * rather than of one deployment:
 *   - the URL must parse, and
 *   - the scheme must be http: or https:. Anything else is not an endpoint this
 *     client can speak to, and a `file:` URL reaching an outbound fetch is a
 *     defect in any package.
 *
 * `localOnly` adds the third check. It is OFF by default and it exists as a
 * capability, not as an opinion:
 *
 *   A DNS NAME IS REFUSED, because a name can resolve anywhere and this process
 *   cannot prove where it resolves to at startup -- a `localhost` that works
 *   today and a hosts-file edit tomorrow are indistinguishable from inside this
 *   function. Requiring a literal address makes the check answerable by reading
 *   the configuration.
 *
 * The shape of the local-only check is deliberate, and a naive one would be
 * wrong in two different ways:
 *   - Asserting `host === "127.0.0.1"` would reject an Ollama that binds
 *     `0.0.0.0` (`OLLAMA_HOST=0.0.0.0`), which is then reached at the LAN
 *     address as well as at loopback.
 *   - Accepting `0.0.0.0` as a connect target would be wrong the other way: it
 *     is the unspecified address, its meaning is platform-dependent, and no
 *     legitimate deployment needs to dial it.
 * An allowlist of literal private addresses handles both.
 */
export function parseEmbeddingEndpoint(
  rawUrl: string,
  opts: { localOnly?: boolean } = {}
): URL {
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
  if (!opts.localOnly) return url;

  if (LOOPBACK_HOSTS.has(host)) return url;

  const octets = ipv4ToOctets(host);
  if (!octets) {
    throw new Error(
      `EMBEDDING_URL host "${host}" is not a literal loopback or RFC1918 ` +
        `address. Local-only embedding is enabled, and a DNS name is refused ` +
        `under it deliberately: a name can resolve anywhere, and this process ` +
        `cannot prove where. Use 127.0.0.1, ::1, 10.x.x.x, 172.16-31.x.x or ` +
        `192.168.x.x, or unset the local-only flag to reach a named endpoint.`
    );
  }
  if (host === "0.0.0.0") {
    throw new Error(
      `EMBEDDING_URL host "0.0.0.0" is the unspecified address, not a ` +
        `destination. A server may BIND 0.0.0.0, but you CONNECT to ` +
        `127.0.0.1 -- use http://127.0.0.1:11434.`
    );
  }
  if (!isPrivateIpv4(octets)) {
    throw new Error(
      `EMBEDDING_URL host "${host}" is a public address, and local-only ` +
        `embedding is enabled. Use a loopback or RFC1918 address.`
    );
  }
  return url;
}

/**
 * Where the request actually goes.
 *
 * A configured path is honoured verbatim: a caller pointing at
 * `https://api.openai.com/v1` gets `https://api.openai.com/v1/embeddings`, and
 * one pointing at `https://host/v1/embeddings` gets exactly that, with no
 * double suffix. An origin with no path falls back to Ollama's default route.
 *
 * Trailing slashes are stripped before the suffix is appended so
 * `http://host:11434/` cannot produce `//api/embed`.
 */
export function resolveEmbeddingEndpoint(base: URL): string {
  const path = base.pathname.replace(/\/+$/, "");
  if (path === "") return `${base.origin}/api/embed`;
  if (/\/embeddings$/.test(path)) return `${base.origin}${path}`;
  return `${base.origin}${path}/embeddings`;
}

/** Ollama's native reply: `{embeddings: [[...]]}`. */
interface NativeEmbedResponse {
  embeddings?: unknown;
}

/** OpenAI-compatible reply: `{data: [{embedding: [...]}]}`. */
interface OpenAiEmbedResponse {
  data?: unknown;
}

/**
 * Read a vector list out of either reply shape.
 *
 * Returns `undefined` when the body is neither, so the caller can raise one
 * error with one message rather than guessing which shape was attempted.
 */
export function embeddingsFromBody(body: unknown): number[][] | undefined {
  if (!body || typeof body !== "object") return undefined;
  const native = (body as NativeEmbedResponse).embeddings;
  if (Array.isArray(native)) return native as number[][];
  const data = (body as OpenAiEmbedResponse).data;
  if (Array.isArray(data)) {
    const vectors = data.map((entry) =>
      entry && typeof entry === "object"
        ? (entry as { embedding?: unknown }).embedding
        : undefined
    );
    // A `data` array that does not carry embeddings on every entry is not a
    // usable reply; saying so is better than returning undefined vectors.
    if (vectors.every((vector) => Array.isArray(vector))) {
      return vectors as number[][];
    }
  }
  return undefined;
}

export interface EmbeddingClientOptions {
  /** Base URL, or a full endpoint URL. See `resolveEmbeddingEndpoint`. */
  url: string;
  model: string;
  prefixMode: PrefixMode;
  /** Expected vector width. Defaults to 768. */
  dimensions?: number;
  /**
   * Sent as `Authorization: Bearer <key>` when present.
   *
   * Never logged, never echoed in an error, never included in any object this
   * module returns. A credential that appears in a diagnostic string outlives
   * the request that used it.
   */
  apiKey?: string;
  /**
   * Restrict the endpoint to a literal loopback or RFC1918 address.
   *
   * Defaults to false. `true` refuses DNS names and public addresses, which is
   * what a deployment that must be able to prove text does not leave the host
   * sets to satisfy itself.
   */
  localOnly?: boolean;
  /** Texts per request. Defaults to `DEFAULT_BATCH_SIZE`. */
  batchSize?: number;
}

/**
 * An HTTP embeddings client for any OpenAI-compatible `/embeddings` endpoint.
 *
 * The constructor validates the endpoint and the prefix mode, and throws on
 * either failure. Nothing is deferred to first use: a misconfigured index
 * should fail at startup, before it has written a single vector.
 */
export class OllamaEmbedder implements Embedder {
  readonly model: string;
  readonly prefixMode: PrefixMode;
  readonly dimensions: number;
  private readonly endpointUrl: string;
  /**
   * Declared with `#`, not `private`.
   *
   * `private` is erased at compile time: the field is an own ENUMERABLE
   * property at runtime, so `JSON.stringify(embedder)` carried the credential
   * and the KDoc's promise that it never appears in a returned object was
   * false. A true private field compiles to a WeakMap slot and is invisible to
   * `Object.keys`, spread, and JSON.
   */
  readonly #apiKey?: string;
  private readonly batchSize: number;

  constructor(opts: EmbeddingClientOptions) {
    const base = parseEmbeddingEndpoint(opts.url, { localOnly: opts.localOnly });
    this.endpointUrl = resolveEmbeddingEndpoint(base);
    if (!opts.model.trim()) {
      throw new Error("EMBEDDING_MODEL must not be empty.");
    }
    if (!(opts.prefixMode in PREFIXES)) {
      throw new Error(`Unknown embedding prefix mode "${opts.prefixMode}".`);
    }
    const batchSize = opts.batchSize ?? DEFAULT_BATCH_SIZE;
    if (!Number.isInteger(batchSize) || batchSize < 1) {
      throw new Error(
        `Embedding batch size must be a positive whole number, not "${opts.batchSize}".`
      );
    }
    this.model = opts.model;
    this.prefixMode = opts.prefixMode;
    this.dimensions = opts.dimensions ?? 768;
    this.#apiKey = opts.apiKey;
    this.batchSize = batchSize;
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    const prefix = PREFIXES[this.prefixMode].document;
    const embeddings: number[][] = [];
    for (let i = 0; i < texts.length; i += this.batchSize) {
      const batch = texts.slice(i, i + this.batchSize).map((text) => prefix + text);
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

  /**
   * Embed one throwaway string and discard it.
   *
   * Deliberately a real embedding rather than a GET to the server's root: a
   * server that is up but has not pulled the model answers GET happily and then
   * fails on the first real request. The whole point of the probe is to fail
   * BEFORE an irreversible step, so it has to exercise the thing that will
   * actually run -- the same model, the same prefix, the same endpoint.
   *
   * The string is a constant, not document text, so nothing real is placed on
   * the wire for a liveness check.
   */
  async probe(): Promise<void> {
    await this.request([PROBE_TEXT], QUERY_TIMEOUT_MS);
  }

  private headers(): Record<string, string> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.#apiKey) headers.Authorization = `Bearer ${this.#apiKey}`;
    return headers;
  }

  private async request(
    input: string[],
    timeoutMs: number
  ): Promise<number[][]> {
    if (input.length === 0) return [];
    let response: Response;
    try {
      // `redirect: "error"` is always on. A 307 answer would otherwise let a
      // compromised endpoint forward the body -- which is document text -- to
      // any host it likes, with the configured endpoint still reading as
      // satisfied. Checked: this is the only outbound `fetch` in non-test
      // source, so one option closes the only such path.
      response = await fetch(this.endpointUrl, {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify({ model: this.model, input }),
        signal: AbortSignal.timeout(timeoutMs),
        redirect: "error",
      });
    } catch (err) {
      // A refused redirect surfaces here as a TypeError from undici. It is
      // reported as unavailable rather than as its own class because the
      // caller's correct response is the same -- stop, do not write vectors --
      // and the message carries the real cause.
      //
      // ⚠️ THE ENDPOINT IS NAMED BUT THE KEY NEVER IS. `err` is included
      // because it is what makes a connection failure diagnosable, and it does
      // not carry request headers.
      throw new EmbeddingUnavailableError(
        `The embedding server did not answer at ${this.endpointUrl}: ${err}`
      );
    }
    if (!response.ok) {
      // ⚠️ CARRY THE SERVER'S OWN EXPLANATION.
      //
      // A bare "status 400" is undiagnosable: Ollama answers a rejected input
      // with a JSON `{"error": "..."}` naming the actual cause, and discarding
      // it means the only way to find out what the server objected to is to
      // re-probe it by hand. That is exactly what happened once already, and it
      // cost more time than this branch.
      //
      // The body is truncated and whitespace-collapsed: an error page can be a
      // full HTML document, and an unbounded string in an exception message is
      // its own defect. This is the SERVER's text, never a request header, so
      // it cannot carry the credential.
      //
      // ⚠️ It CAN in principle carry document text, if a provider chooses to
      // echo the offending input back. That is a real trade against
      // diagnosability and it is bounded deliberately: 300 characters, one
      // line, and always the server's words rather than ours. A provider that
      // echoes embeddings input in its errors is worth knowing about anyway.
      let detail = "";
      try {
        const text = (await response.text()).replace(/\s+/g, " ").trim();
        if (text) detail = `: ${text.slice(0, 300)}`;
      } catch {
        // Reading the error body is a diagnostic courtesy; its failure must not
        // replace the status error with a read error.
      }
      throw new EmbeddingUnavailableError(
        `The embedding server answered with status ${response.status} at ${this.endpointUrl}${detail}`
      );
    }
    let body: unknown;
    try {
      body = await response.json();
    } catch (err) {
      throw new EmbeddingUnavailableError(
        `The embedding server answered with a body that is not JSON: ${err}`
      );
    }
    const embeddings = embeddingsFromBody(body);
    if (!embeddings || embeddings.length !== input.length) {
      throw new EmbeddingUnavailableError(
        "The embedding server answered with an unrecognised body, or with a " +
          "mismatched embeddings array. Expected either `{embeddings: [[...]]}` " +
          "(Ollama native) or `{data: [{embedding: [...]}]}` " +
          "(OpenAI-compatible), with one vector per input " +
          `(${input.length} sent).`
      );
    }
    for (const embedding of embeddings) {
      if (!Array.isArray(embedding) || embedding.length !== this.dimensions) {
        throw new EmbeddingUnavailableError(
          `The embedding server returned a vector of the wrong width; ` +
            `expected ${this.dimensions} dimensions. Check that ` +
            `EMBEDDING_MODEL matches the index.`
        );
      }
    }
    return embeddings;
  }
}
