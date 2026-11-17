import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { timingSafeEqual } from "node:crypto";
import type { SyncReport } from "./sync";

/**
 * A webhook endpoint that keeps the vector index in step with document changes.
 *
 * WHY THIS EXISTS AT ALL, AND WHY IT IS NOT A POLLER. Paperless' own webhook
 * action fires on document add/update, so the index can be told about a change
 * instead of asking. That matters for one case the watermark structurally
 * cannot see: a tag, correspondent or document-type edit goes through a
 * queryset update on a through-table, `auto_now` never runs, and neither
 * `count` nor `max(modified)` moves. A poller checks, sees nothing, and the
 * stale metadata chunk stays searchable indefinitely. The event is the only
 * signal that exists for that change.
 *
 * ⚠️ THIS IS NOT A REPLACEMENT FOR THE WATERMARK. See `IndexSync` for what the
 * two scalars still cover: events that happened while this process was not
 * running. A webhook only ever reports events it was alive to receive.
 *
 * ⚠️ AND IT CANNOT REPORT DELETES, because there is nothing to report. The
 * trigger set in the server this was written against enumerates four types --
 * consumption started, document added, document updated, scheduled -- and no
 * delete type. The document `post_delete` receiver in the same source only
 * feeds a different subsystem's index. So a deleted document produces no
 * delivery, and the `count` half of the watermark remains the only thing that
 * detects one. Nothing here pretends otherwise.
 */

/**
 * The header an operator puts the shared secret in, on the webhook action's
 * `headers` field.
 *
 * A header rather than a query string because the secret belongs in one, and
 * rather than a signature because a signature scheme would need a canonical
 * request encoding that this receiver does not have; a shared secret over a
 * header is the whole of what an operator can configure on that side.
 */
export const SECRET_HEADER = "x-paperless-mcp-secret";

/** Accept the secret as a bearer token too, because that needs no custom header. */
const AUTH_PREFIX = "Bearer ";

/**
 * A body larger than this is refused rather than buffered.
 *
 * The payload is a handful of scalar fields -- the largest of them a title --
 * so a megabyte is already two orders of magnitude of headroom. The cap is not
 * about legitimate payloads; it is so that an unauthenticated flood, or a
 * misconfigured operator pointing a different system at this port, cannot make
 * the process hold an arbitrary amount of memory before the size is known.
 */
export const MAX_BODY_BYTES = 1024 * 1024;

/**
 * How long the receiver waits for quiet before acting.
 *
 * A bulk edit -- "add this tag to these 200 documents" -- delivers one request
 * per document within a few seconds. Acting on each one would run 200 syncs
 * over a corpus that changed once. Waiting for the events to stop converts that
 * into a single run naming every id, which is both cheaper and less likely to
 * collide with itself.
 */
export const DEFAULT_DEBOUNCE_MS = 2000;

/**
 * What the receiver will run. Narrower than `IndexSync` so a test can count
 * runs without a real index, and so the receiver cannot reach for a method the
 * sync does not offer.
 */
export interface SyncRunner {
  run(opts?: { force?: boolean; documentIds?: number[] }): Promise<SyncReport>;
  readonly isRunning: boolean;
}

export interface ParsedEvent {
  /** Every document id the delivery named. May be empty. */
  ids: number[];
  /**
   * True when the delivery named no id at all.
   *
   * `use_params` with an empty `params` sends an empty body, and an operator
   * who configured neither `params` nor `body` gets exactly that. Those events
   * still mean "something changed", so they are acted on as a plain sync rather
   * than dropped -- but they cannot be targeted, and saying so here is what
   * keeps the caller from confusing "no ids" with "nothing happened".
   */
  idless: boolean;
}

/**
 * Read document ids out of a delivery.
 *
 * ⚠️ THE BODY SHAPE IS NOT GUARANTEED. The sending side renders a template into
 * a dictionary and then either serialises it as JSON or posts it form-encoded
 * depending on one boolean. Both are legitimate configurations of the same
 * feature, so both are parsed here. A body that is neither is not an error: it
 * is an event that named no id, which is a state the caller already handles.
 */
export function parseEvent(body: string, contentType: string | undefined): ParsedEvent {
  const type = (contentType ?? "").toLowerCase();
  const fields: Record<string, unknown> = {};

  if (type.includes("json")) {
    try {
      const parsed: unknown = JSON.parse(body);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        Object.assign(fields, parsed as Record<string, unknown>);
      }
    } catch {
      // A JSON-labelled body that is not JSON. Fall through: no ids, and the
      // caller runs a plain sync rather than refusing an event it received.
    }
  } else if (type.includes("urlencoded") || (!type && body.includes("="))) {
    for (const [key, value] of new URLSearchParams(body)) fields[key] = value;
  } else if (body.trim().startsWith("{")) {
    // No content type, or an unexpected one, but the body is plainly JSON.
    try {
      const parsed: unknown = JSON.parse(body);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        Object.assign(fields, parsed as Record<string, unknown>);
      }
    } catch {
      // Same reasoning as above.
    }
  }

  const ids: number[] = [];
  // Two names because the sending side offers the document's primary key under
  // a short alias and a longer one, and an operator may place either in the
  // template. Both are the same value.
  for (const key of ["id", "doc_id", "document_id"]) {
    const raw = fields[key];
    // Canonical decimal only. parseInt(.., 10) stops at the first non-digit,
    // so every one of these malformed payloads silently names a DIFFERENT real
    // document: "5.9" becomes 5, "12abc" becomes 12, "1e400" becomes 1, and
    // [1,2] becomes "1,2" then 1. The sync then re-indexes the wrong document
    // and reports success, so nothing tells the operator the payload was bad.
    // A non-string, non-number value (an array, an object, null) names nothing.
    const canonicallyNumeric =
      typeof raw === "number" ||
      (typeof raw === "string" && /^[1-9][0-9]*$/.test(raw));
    const value = canonicallyNumeric ? Number(raw) : Number.NaN;
    if (
      Number.isSafeInteger(value) &&
      value > 0 &&
      !ids.includes(value)
    ) {
      ids.push(value);
    }
  }

  return { ids, idless: ids.length === 0 };
}

/**
 * Compare two strings without leaking their length or content through timing.
 *
 * `timingSafeEqual` refuses buffers of different length, so the length is
 * compared first and the result is discarded -- a wrong-length secret and a
 * wrong-content secret then both fail, and only the second comparison is
 * constant-time. Returning early on a length mismatch would leak the secret's
 * length, which is a small leak but a free one to avoid.
 */
export function secretMatches(provided: string | undefined, expected: string): boolean {
  if (typeof provided !== "string") return false;
  const a = Buffer.from(provided, "utf8");
  const b = Buffer.from(expected, "utf8");
  if (a.length !== b.length) {
    // Still do work proportional to the expected value, so the fast path is not
    // distinguishable by timing from the slow one.
    timingSafeEqual(b, b);
    return false;
  }
  return timingSafeEqual(a, b);
}

/** Pull the presented secret out of the headers, in either accepted form. */
export function presentedSecret(headers: IncomingMessage["headers"]): string | undefined {
  const direct = headers[SECRET_HEADER];
  if (typeof direct === "string" && direct.length > 0) return direct;
  const authorization = headers.authorization;
  if (typeof authorization === "string" && authorization.startsWith(AUTH_PREFIX)) {
    return authorization.slice(AUTH_PREFIX.length);
  }
  return undefined;
}

export interface WebhookOptions {
  /** Shared secret. The receiver refuses to start without one. */
  secret: string;
  /** Port to listen on. 0 asks the OS for an ephemeral one, which tests use. */
  port: number;
  /**
   * Address to bind.
   *
   * Defaults to loopback. It is deliberately NOT defaulted to a wildcard: an
   * endpoint that writes to a local index should not be reachable from off the
   * host unless the operator asked for it, and a containerised deployment is
   * the case that genuinely needs the wider bind and therefore has to say so.
   */
  host?: string;
  /** Quiet period before the coalesced run. */
  debounceMs?: number;
  /** Resolves the runner at fire time, so a failure to build it is not fatal. */
  runner: () => SyncRunner;
  /** Where to report. Defaults to stderr, matching the rest of this package. */
  log?: (line: string) => void;
}

/**
 * The receiver.
 *
 * Lifecycle: `start()` binds and returns once listening; `close()` stops the
 * debounce timer first and then closes the server, so a run is never kicked off
 * against a store that is being closed underneath it.
 */
export class WebhookReceiver {
  private readonly options: WebhookOptions;
  private server?: Server;
  private timer?: ReturnType<typeof setTimeout>;
  /** Ids heard since the last run. Empty means "something changed, location unknown". */
  private readonly pending = new Set<number>();
  private pendingWasIdless = false;
  private closed = false;

  constructor(options: WebhookOptions) {
    this.options = options;
  }

  private log(line: string): void {
    (this.options.log ?? ((l: string) => console.error(`[paperless-mcp] webhook: ${l}`)))(line);
  }

  /**
   * Bind and listen.
   *
   * ⚠️ AN UNSET SECRET IS A REFUSAL, NOT A WARNING. This endpoint causes work --
   * embedding calls against whatever provider the install points at. An
   * unauthenticated trigger for that is a resource-abuse vector, and a warning
   * in a log nobody reads is not a mitigation. Failing to start is.
   */
  async start(): Promise<{ port: number; host: string }> {
    if (this.options.secret.length === 0) {
      throw new Error(
        "The webhook receiver needs a shared secret. Set " +
          "PAPERLESS_MCP_WEBHOOK_SECRET to a value the Paperless workflow will " +
          "send in the " +
          SECRET_HEADER +
          " header, or leave PAPERLESS_MCP_WEBHOOK_PORT unset to keep the " +
          "receiver off."
      );
    }
    if (this.server) {
      throw new Error("The webhook receiver is already started.");
    }

    const host = this.options.host ?? "127.0.0.1";
    const server = createServer((req, res) => {
      void this.handle(req, res);
    });
    this.server = server;

    await new Promise<void>((resolve, reject) => {
      const onError = (err: Error) => {
        server.removeListener("listening", onListening);
        this.server = undefined;
        reject(err);
      };
      const onListening = () => {
        server.removeListener("error", onError);
        resolve();
      };
      server.once("error", onError);
      server.once("listening", onListening);
      server.listen(this.options.port, host);
    });

    const address = server.address();
    const bound = typeof address === "object" && address !== null ? address.port : this.options.port;
    this.log(`listening on ${host}:${bound}`);
    return { port: bound, host };
  }

  /**
   * The port actually bound, or undefined when not listening.
   *
   * Reported rather than assumed: a test that binds port 0 needs the real port,
   * and a caller that hardcoded the requested one would connect to nothing.
   */
  get boundPort(): number | undefined {
    const address = this.server?.address();
    if (typeof address === "object" && address !== null) return address.port;
    return undefined;
  }

  get isListening(): boolean {
    return this.server?.listening === true;
  }

  /**
   * Stop listening and cancel any pending run.
   *
   * The timer is cleared FIRST. A run scheduled microseconds ago would otherwise
   * fire against a closed store, which is the same failure the background
   * interval is stopped ahead of `close()` to avoid.
   */
  async close(): Promise<void> {
    this.closed = true;
    if (this.timer) clearTimeout(this.timer);
    this.timer = undefined;
    this.pending.clear();
    this.pendingWasIdless = false;

    const server = this.server;
    this.server = undefined;
    if (!server) return;

    await new Promise<void>((resolve) => {
      // Idle keep-alive sockets would hold the server open for as long as the
      // peer keeps them; dropping them makes close() mean closed.
      const closeAll = () => {
        server.closeAllConnections?.();
        resolve();
      };
      server.close(() => resolve());
      closeAll();
    });
  }

  private async handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    try {
      if (req.method !== "POST" || !this.isEventPath(req.url)) {
        // A health probe on the root is useful and harmless; anything else is
        // not this endpoint's business.
        res.writeHead(req.method === "GET" ? 200 : 404, { "content-type": "application/json" });
        res.end(JSON.stringify({ ok: req.method === "GET", path: req.url ?? "" }));
        return;
      }

      if (!secretMatches(presentedSecret(req.headers), this.options.secret)) {
        // No sync, no body buffering, no detail. 401 and close.
        res.writeHead(401, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: "unauthorized" }));
        return;
      }

      const body = await this.readBody(req, res);
      if (body === undefined) return; // readBody already answered (413 or aborted).

      const event = parseEvent(body, req.headers["content-type"]);

      // ⚠️ ACKNOWLEDGE BEFORE WORKING. The sending side has a five-second
      // timeout and retries on failure, so a receiver that embedded first and
      // answered after would turn one edit into up to four deliveries of the
      // same event. Acknowledging immediately is what makes those retries
      // harmless -- and the coalescing below is what makes the duplicates cheap.
      res.writeHead(202, { "content-type": "application/json" });
      res.end(JSON.stringify({ accepted: true }));

      this.enqueue(event);
    } catch (err) {
      this.log(`request failed: ${err instanceof Error ? err.message : String(err)}`);
      if (!res.headersSent) {
        res.writeHead(500, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: "internal" }));
      } else {
        res.end();
      }
    }
  }

  /** The event path, tolerating a trailing slash. */
  private isEventPath(url: string | undefined): boolean {
    if (!url) return false;
    const path = url.split("?")[0];
    return path === "/webhook/paperless" || path === "/webhook/paperless/";
  }

  /**
   * Buffer the body under a cap, answering 413 when it is exceeded.
   *
   * Returns undefined when it has already responded. The connection is destroyed
   * after the refusal because the unread remainder would otherwise be left in
   * the socket for the next request to trip over.
   */
  private async readBody(
    req: IncomingMessage,
    res: ServerResponse
  ): Promise<string | undefined> {
    const declared = Number.parseInt(String(req.headers["content-length"] ?? ""), 10);
    if (Number.isFinite(declared) && declared > MAX_BODY_BYTES) {
      this.refuseTooLarge(res);
      return undefined;
    }

    const chunks: Buffer[] = [];
    let total = 0;
    try {
      for await (const chunk of req) {
        const buf = typeof chunk === "string" ? Buffer.from(chunk) : (chunk as Buffer);
        total += buf.length;
        if (total > MAX_BODY_BYTES) {
          this.refuseTooLarge(res);
          return undefined;
        }
        chunks.push(buf);
      }
    } catch (err) {
      // A client that vanished mid-body is not this endpoint's failure.
      this.log(`body read aborted: ${err instanceof Error ? err.message : String(err)}`);
      return undefined;
    }
    return Buffer.concat(chunks).toString("utf8");
  }

  private refuseTooLarge(res: ServerResponse): void {
    // Declared oversize is refused without reading a byte; an oversize that
    // arrived unannounced has already had its prefix consumed, so the socket is
    // torn down rather than trying to resynchronise it.
    res.writeHead(413, { "content-type": "application/json", connection: "close" });
    res.end(JSON.stringify({ error: "payload too large", limit: MAX_BODY_BYTES }));
  }

  /**
   * Fold an event into the pending set and (re)arm the quiet timer.
   *
   * Re-arming on every event is what makes the wait a quiet period rather than a
   * fixed window: a burst that lasts longer than the debounce still produces one
   * run, at the end of the burst.
   */
  private enqueue(event: ParsedEvent): void {
    if (this.closed) return;
    if (event.idless) this.pendingWasIdless = true;
    for (const id of event.ids) this.pending.add(id);
    if (this.timer) clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.timer = undefined;
      void this.flush();
    }, this.options.debounceMs ?? DEFAULT_DEBOUNCE_MS);
    // A pending run must not keep the process alive on its own; the MCP client
    // owns this process's lifetime.
    this.timer.unref?.();
  }

  /** Run once for everything gathered, then clear. */
  private async flush(): Promise<void> {
    if (this.closed) return;
    // Snapshot and clear FIRST: events arriving during the run belong to the
    // next one, and a set mutated mid-iteration by a later delivery would make
    // the run's target list depend on network timing.
    const ids = [...this.pending];
    const idless = this.pendingWasIdless;
    this.pending.clear();
    this.pendingWasIdless = false;

    try {
      const runner = this.options.runner();
      // ⚠️ An idless event means "something changed, and the sender did not say
      // what". Passing no ids makes the run consult the full diff, which is the
      // expensive-but-correct answer; passing an empty target list would look
      // like "nothing to do" and do nothing.
      const opts = ids.length > 0 ? { documentIds: ids } : {};
      const report = await runner.run(opts);
      this.log(
        `run finished: ${ids.length > 0 ? `${ids.length} targeted` : idless ? "untargeted" : "no-op"}` +
          `, +${report.added} ~${report.updated} -${report.deleted}` +
          (report.skipped ? ` (skipped: ${report.skipReason ?? "no reason given"})` : "")
      );
    } catch (err) {
      // A failed run must not take down the receiver: the next event retries,
      // and the watermark still converges on the next start.
      this.log(`run failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
}
