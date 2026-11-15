import { ToolCallback } from "@modelcontextprotocol/sdk/server/mcp";
import { ZodRawShape } from "zod";

/**
 * Turns a thrown error into a sentence a model can act on, then rethrows it.
 *
 * The THROW is load-bearing and must not change: the MCP SDK catches it and
 * converts it into a tool *execution* error (`isError: true`), which the spec
 * requires for input-validation failures because "clients SHOULD provide tool
 * execution errors to language models to enable self-correction". A JSON-RPC
 * protocol error is documented as "less likely to result in successful
 * recovery".
 *
 * The defect this fixes was the payload, not the mechanism. The old form threw
 * `JSON.stringify({error, responseData, status})` -- a blob that names no field
 * and suggests no fix. The spec's own example of a good execution error is a
 * sentence: "Invalid departure date: must be in the future. Current date is
 * 08/08/2025."
 *
 * Two hazards are handled explicitly rather than assumed away, both measured
 * against a live axios error on this repo:
 *   1. `err.config.headers.Authorization` carries the API token, and it
 *      survives `JSON.stringify(err)` and `err.toJSON()`. We never serialise
 *      the config, and we additionally scrub any harvested credential value
 *      out of the final sentence in case the server echoes it back.
 *   2. `err.response` is circular. `JSON.stringify` on it throws, which would
 *      replace a useful error with a TypeError from inside the handler.
 */

/** Cap on any quoted server detail, so a 200-field validation body stays readable. */
const MAX_DETAIL_CHARS = 240;

/** Headers whose values are credentials; never emitted, always scrubbed. */
const SENSITIVE_HEADER = /^(authorization|proxy-authorization|cookie|set-cookie|x-api-key|api-key)$/i;

/** Error codes that mean "we never got a response". */
const UNREACHABLE_CODES = new Set([
  "ECONNREFUSED",
  "ECONNRESET",
  "ENOTFOUND",
  "EAI_AGAIN",
  "EHOSTUNREACH",
  "ENETUNREACH",
  "EPIPE",
  "ERR_NETWORK",
]);

/** Error codes that mean "we gave up waiting". */
const TIMEOUT_CODES = new Set(["ECONNABORTED", "ETIMEDOUT", "UND_ERR_CONNECT_TIMEOUT"]);

function truncate(value: string, max: number): string {
  const collapsed = value.replace(/\s+/g, " ").trim();
  return collapsed.length <= max ? collapsed : `${collapsed.slice(0, Math.max(0, max - 1))}…`;
}

/**
 * JSON.stringify that cannot throw and cannot recurse forever. Circular
 * references become "[circular]"; a value that still defeats serialisation
 * falls back to String(), and only a String() that also throws returns
 * undefined. Never call JSON.stringify directly in this file.
 */
function safeStringify(value: unknown, max: number): string | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value === "string") return truncate(value, max);
  const seen = new WeakSet<object>();
  let out: string | undefined;
  try {
    out = JSON.stringify(value, (_key, val) => {
      if (typeof val === "object" && val !== null) {
        if (seen.has(val)) return "[circular]";
        seen.add(val);
      }
      return val;
    });
  } catch {
    try {
      out = String(value);
    } catch {
      return undefined;
    }
  }
  return out === undefined ? undefined : truncate(out, max);
}

/**
 * Harvests credential values worth scrubbing out of anything we emit.
 *
 * Measured hazard, not a theoretical one: an axios error carries the request
 * config, and `config.headers.Authorization` still holds `Token <api-token>`
 * after the failure. Anything that serialises the whole error object publishes
 * the credential. We never emit the config, but a server can echo a header back
 * in an error body, so the values are scrubbed from the final sentence too.
 */
function credentialValues(err: unknown): string[] {
  const found = new Set<string>();
  const headers = (err as any)?.config?.headers;
  if (!headers || typeof headers !== "object") return [];
  const entries: Array<[string, unknown]> =
    typeof (headers as any).toJSON === "function"
      ? Object.entries((headers as any).toJSON())
      : Object.entries(headers as Record<string, unknown>);
  for (const [name, value] of entries) {
    if (typeof value !== "string" || value.length < 4) continue;
    if (!SENSITIVE_HEADER.test(name)) continue;
    // `Token abc123` -> scrub the bare credential as well as the whole header.
    found.add(value);
    const stripped = value.replace(/^(Bearer|Token|Basic)\s+/i, "").trim();
    if (stripped.length >= 8) found.add(stripped);
  }
  return [...found];
}

function scrub(text: string, secrets: string[]): string {
  let out = text;
  for (const secret of secrets) {
    if (secret && out.includes(secret)) out = out.split(secret).join("[redacted]");
  }
  return out;
}

/**
 * Names the host a request was aimed at, from the axios config. Returns
 * undefined rather than a guess when the config carries no usable URL.
 */
function requestHost(err: unknown): string | undefined {
  const config = (err as any)?.config;
  if (!config) return undefined;
  const candidates = [config.url, config.baseURL];
  for (const candidate of candidates) {
    if (typeof candidate !== "string") continue;
    try {
      const parsed = new URL(candidate);
      return parsed.host;
    } catch {
      /* relative url against a separate baseURL -- try the pair below */
    }
  }
  const { baseURL, url } = config;
  if (typeof baseURL === "string" && typeof url === "string") {
    try {
      return new URL(url, baseURL).host;
    } catch {
      return undefined;
    }
  }
  return undefined;
}

/**
 * Extracts `field -> detail` pairs from a DRF-style validation body.
 *
 * paperless-ngx is Django REST Framework, so a 400 normally arrives as
 * `{"field": ["message"]}` or `{"non_field_errors": ["message"]}`. Naming the
 * field is the whole point of this function: an agent that cannot see which
 * field was wrong cannot fix its call.
 */
function fieldDetails(body: unknown, depth = 0): Array<[string, string]> {
  if (depth > 2 || !body || typeof body !== "object" || Array.isArray(body)) return [];
  const pairs: Array<[string, string]> = [];
  for (const [field, value] of Object.entries(body as Record<string, unknown>)) {
    if (typeof value === "string") {
      pairs.push([field, truncate(value, MAX_DETAIL_CHARS)]);
      continue;
    }
    if (Array.isArray(value)) {
      const parts = value
        .map((item) =>
          typeof item === "string" ? item : safeStringify(item, MAX_DETAIL_CHARS) ?? ""
        )
        .filter((item) => item.length > 0);
      if (parts.length > 0) pairs.push([field, truncate(parts.join("; "), MAX_DETAIL_CHARS)]);
      continue;
    }
    // Nested shape, e.g. {"custom_fields": {"12": ["bad"]}} -- flatten one level.
    for (const [childField, childValue] of fieldDetails(value, depth + 1)) {
      pairs.push([`${field}.${childField}`, childValue]);
    }
  }
  return pairs;
}

/** Renders the field list as `field: detail; field: detail`, capped. */
function describeFields(body: unknown): string | undefined {
  const pairs = fieldDetails(body);
  if (pairs.length === 0) return undefined;
  const rendered = pairs.map(([field, detail]) => `${field}: ${detail}`).join("; ");
  return truncate(rendered, MAX_DETAIL_CHARS);
}

/**
 * Builds the caller-facing sentence. Every branch names what went wrong AND
 * what to do about it, because that is what makes the error actionable; the
 * original message is always appended so nothing is lost.
 */
function buildMessage(err: unknown): string {
  const status = (err as any)?.response?.status;
  const body = (err as any)?.response?.data;
  const code = (err as any)?.code;
  const original = err instanceof Error ? err.message : String(err ?? "Unknown error");
  const host = requestHost(err);

  // --- No response. Two very different things land here, and conflating them
  // is worse than the original defect: a LOCAL validation error thrown by a
  // tool body (e.g. documents.ts "Confirmation required for destructive
  // operation. Set confirm: true to proceed.") also carries no `response`.
  // Reporting that as a network problem tells the caller to check its base URL
  // when the fix is a different argument. So a transport failure is only
  // claimed when the error actually looks like one -- an axios error, or a
  // recognised network code -- and anything else is passed through, because a
  // locally-authored message is already a sentence aimed at the model.
  if (status === undefined) {
    const isTransportError =
      Boolean((err as any)?.isAxiosError) ||
      (typeof code === "string" &&
        (TIMEOUT_CODES.has(code) || UNREACHABLE_CODES.has(code))) ||
      Boolean((err as any)?.config);

    if (!isTransportError) {
      return original;
    }

    if (typeof code === "string" && TIMEOUT_CODES.has(code)) {
      const where = host ? ` at ${host}` : "";
      return `Request to the paperless-ngx server${where} timed out. The server may be overloaded or unreachable -- retry, and if it keeps timing out check that the server is up. (${original})`;
    }
    if (typeof code === "string" && UNREACHABLE_CODES.has(code)) {
      const where = host ? ` at ${host}` : "";
      return `Could not reach the paperless-ngx server${where} (${code}). The host may be down, misconfigured, or unreachable from here -- verify the base URL and that the server is running. (${original})`;
    }
    const where = host ? ` at ${host}` : "";
    return `The request to the paperless-ngx server${where} failed before any response was received. This is a connection problem, not a problem with your arguments. (${original})`;
  }

  // --- Server returned an error status.
  const fields = describeFields(body);
  const quote = fields ?? safeStringify(body, MAX_DETAIL_CHARS);

  if (status === 400 || status === 422) {
    if (fields) {
      return `The paperless-ngx server rejected the request as invalid -- ${fields}. Fix those field(s) and retry. (HTTP ${status})`;
    }
    if (quote) {
      return `The paperless-ngx server rejected the request as invalid: ${quote}. Check the document ids, field names and value formats, then retry. (HTTP ${status})`;
    }
    return `The paperless-ngx server rejected the request as invalid and returned no detail. Check the argument names and value formats against the tool's input schema, then retry. (HTTP ${status})`;
  }

  if (status === 401) {
    return `The paperless-ngx server rejected the credentials (HTTP 401): the API token was not accepted, or none was sent. This is a server configuration problem, not something a different tool argument will fix. (${original})`;
  }

  if (status === 403) {
    const detail = quote ? ` Server said: ${quote}.` : "";
    return `The paperless-ngx server refused this operation as forbidden (HTTP 403): the token is valid but lacks permission for it.${detail} Do not retry unchanged -- use a token with the required permission or ask the server owner to grant it. (${original})`;
  }

  if (status === 404) {
    const detail = quote ? ` Server said: ${quote}.` : "";
    return `The paperless-ngx server found nothing at the requested path (HTTP 404).${detail} A document, tag, correspondent or other id in the arguments does not exist -- most often it was deleted, or the id belongs to a different object type. Verify the id with a list tool before retrying. (${original})`;
  }

  if (status === 409) {
    const detail = quote ? ` Server said: ${quote}.` : "";
    return `The paperless-ngx server reported a conflict (HTTP 409).${detail} The object likely changed or already exists; re-read it, then retry. (${original})`;
  }

  if (status === 429) {
    const detail = quote ? ` Server said: ${quote}.` : "";
    return `The paperless-ngx server is rate limiting requests (HTTP 429).${detail} Wait briefly and retry the same call; reduce how many calls you issue in parallel. (${original})`;
  }

  if (status >= 500) {
    const detail = quote ? ` Server said: ${quote}.` : "";
    return `The paperless-ngx server failed internally (HTTP ${status}).${detail} This is a server-side fault, not a problem with your arguments -- retrying the same call unchanged is reasonable, but the fix is on the server. (${original})`;
  }

  // Any other status: still a sentence, still keeps the detail and the original.
  const detail = quote ? ` Server said: ${quote}.` : "";
  return `The paperless-ngx server returned an unexpected HTTP ${status}.${detail} (${original})`;
}

/**
 * Wraps a tool callback so a thrown error reaches the model as an actionable
 * sentence. The signature and generic parameters are unchanged so the ~48 call
 * sites keep typechecking untouched.
 */
export const withErrorHandling = <Args extends ZodRawShape>(
  cb: ToolCallback<Args>
): ToolCallback<Args> => {
  return (async (args, extra) => {
    try {
      return await cb(args, extra);
    } catch (err) {
      // `buildMessage` never throws: every serialisation goes through
      // `safeStringify`, and a failure there degrades to the original message.
      const secrets = credentialValues(err);
      const message = scrub(buildMessage(err), secrets);
      throw new Error(message);
    }
  }) as ToolCallback<Args>;
};
