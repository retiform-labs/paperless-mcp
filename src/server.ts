import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type express from "express";
import { PaperlessAPI } from "./api/PaperlessAPI";
import { registerDocumentResources } from "./resources/documents";
import { registerCorrespondentTools } from "./tools/correspondents";
import { registerCustomFieldTools } from "./tools/customFields";
import { registerDocumentTools } from "./tools/documents";
import { registerDocumentTypeTools } from "./tools/documentTypes";
import { registerMailTools } from "./tools/mail";
import { registerNoteTools } from "./tools/notes";
import { registerSearchTools, SearchContext } from "./tools/search";
import { registerTagTools } from "./tools/tags";

export interface CreateMcpServerOptions {
  baseUrl: string;
  token: string;
  version: string;
  publicUrl: string;
}

export function createMcpServer({
  baseUrl,
  token,
  version,
  publicUrl,
}: CreateMcpServerOptions): McpServer {
  const api = new PaperlessAPI(baseUrl, token);
  const server = new McpServer(
    { name: "paperless-ngx", version },
    { instructions: buildInstructions(publicUrl) }
  );
  registerDocumentTools(server, api);
  registerDocumentResources(server, api);
  registerNoteTools(server, api);
  registerTagTools(server, api);
  registerCorrespondentTools(server, api);
  registerDocumentTypeTools(server, api);
  registerCustomFieldTools(server, api);
  registerMailTools(server, api);
  // Semantic search registers ONLY when its configuration is present. The
  // gateway passes one environment to every child, so an unconfigured install
  // must not advertise three tools that would fail on first call -- and must
  // not open a ~1 GB SQLite file at startup to decide. SearchContext resolves
  // the config and opens nothing; the open happens on the first tool call.
  const searchContext = registerSearchTools(server, api);
  // `onclose` lives on the low-level Server (exposed as `server.server`), not on
  // McpServer, and the SDK does not set it -- verified in protocol.js: `_onclose()`
  // calls `this.onclose?.()`, and mcp.js never assigns it. Chaining rather than
  // overwriting so a future SDK assignment is not silently clobbered.
  const previousOnClose = server.server.onclose;
  server.server.onclose = () => {
    previousOnClose?.();
    searchContext.close();
  };
  // Enforced last, so it sees the complete set including the conditional
  // semantic-search tools.
  enforceDeterministicToolOrder(server);

  return server;
}

export interface ResolveTokenOptions {
  /**
   * Server-configured token used as a fallback for unauthenticated requests.
   * Only consulted when `allowAnonymous` is true.
   */
  fallbackToken?: string;
  /**
   * When true, requests without a `Bearer` header fall back to `fallbackToken`
   * (the legacy behaviour). When false (the default in HTTP mode), a request
   * without a `Bearer` header is rejected and never uses the server token.
   */
  allowAnonymous: boolean;
}

export function getBearerToken(
  req: express.Request,
  options: ResolveTokenOptions
): string | undefined {
  const authHeader = req.headers["authorization"];
  if (authHeader && authHeader.startsWith("Bearer ")) {
    return authHeader.slice(7);
  }
  if (options.allowAnonymous) {
    return options.fallbackToken || undefined;
  }
  return undefined;
}

export function sendUnauthorized(res: express.Response): void {
  // Log operator-facing guidance server-side; keep the wire response minimal so
  // we don't echo configuration hints back to unauthenticated callers.
  console.error(
    "[paperless-mcp] Rejected request with no 'Authorization: Bearer <paperless-ngx-api-token>' header. " +
      "As of v2.0.0, HTTP mode requires a per-request Bearer token and no longer falls back to the " +
      "server-configured token for unauthenticated requests. Have clients send their Paperless-NGX API " +
      "token as a Bearer token, or restart the server with --no-auth to use the server token for " +
      "unauthenticated requests (trusted/local networks only)."
  );
  res
    .status(401)
    .set("WWW-Authenticate", 'Bearer realm="paperless-mcp"')
    .json({ error: "unauthorized" });
}

/**
 * Makes the `tools/list` ordering a property of the tool SET rather than of the
 * order the registration functions happened to run in.
 *
 * The MCP spec: "Servers SHOULD return tools in a deterministic order (i.e.,
 * the same ordering across requests when the underlying set of tools has not
 * changed). Deterministic ordering enables clients to reliably cache the tool
 * list and improves LLM prompt cache hit rates when tools are included in
 * model context."
 *
 * Registration order is already stable today -- module evaluation order is
 * deterministic in Node and nothing here reads a directory or races. But that
 * makes determinism an ACCIDENT of the call order, not a guarantee. Two things
 * break it without anyone intending it:
 *   - reordering a `register*Tools(...)` call, or moving a `server.tool(...)`
 *     call within one, silently reshuffles the list;
 *   - `RegisteredTool.update({ name })` is a delete-then-add, so a rename
 *     moves that tool to the end of the list.
 * Sorting at the point the list is returned makes both of those no-ops.
 *
 * The comparator is `a.name < b.name`, NOT `localeCompare`: locale collation
 * varies with the host's ICU data, which would trade one non-determinism for a
 * subtler one. Code-unit comparison is identical on every Node build.
 *
 * If the SDK ever stops holding the handler in `_requestHandlers`, this returns
 * without wrapping and the list falls back to registration order -- still
 * deterministic, just no longer enforced. It never throws.
 */
export function enforceDeterministicToolOrder(server: McpServer): void {
  const handlers = (server.server as any)?._requestHandlers;
  if (!(handlers instanceof Map)) return;
  // Measured from the live handler map, not assumed: the key is the literal
  // method string, and the SDK has already registered it by the time every
  // register*Tools() call has returned.
  const method = "tools/list";
  const original = handlers.get(method);
  if (typeof original !== "function") return;
  handlers.set(method, async (request: unknown, extra: unknown) => {
    const result = await original(request, extra);
    if (!result || !Array.isArray(result.tools)) return result;
    const tools = [...result.tools].sort((a: any, b: any) =>
      a.name < b.name ? -1 : a.name > b.name ? 1 : 0
    );
    return { ...result, tools };
  });
}

function buildInstructions(publicUrl: string): string {
  return `
Paperless-NGX MCP Server Instructions

⚠️ CRITICAL: Always differentiate between operations on specific documents vs operations on the entire system:

- REMOVE operations (e.g., remove_tag in bulk_edit_documents): Affect only the specified documents, items remain in the system
- DELETE operations (e.g., delete_tag, delete_correspondent): Permanently delete items from the entire system, affecting ALL documents that use them

When a user asks to "remove" something, prefer operations that affect specific documents. Only use DELETE operations when explicitly asked to delete from the system.

To view documents in your Paperless-NGX web interface, construct URLs using this pattern:
${publicUrl}/documents/{document_id}/

Example: If your base URL is "http://localhost:8000", the web interface URL would be "http://localhost:8000/documents/123/" for document ID 123.

The document tools return JSON data with document IDs that you can use to construct these URLs.
        `;
}
