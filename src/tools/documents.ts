import { McpServer } from "@modelcontextprotocol/sdk/server/mcp";
import { z } from "zod";
import { readFile, access, stat, realpath } from "fs/promises";
import { constants } from "fs";
import { basename, isAbsolute } from "path";
import { convertDocsWithNames } from "../api/documentEnhancer";
import { Document, DocumentsResponse, SearchHit } from "../api/types";
import { PaperlessAPI } from "../api/PaperlessAPI";
import { arrayNotEmpty } from "./utils/empty";
import {
  BuildDocumentQueryArgs,
  buildDocumentQueryString,
  LIST_DOCUMENTS_ARGS_SHAPE,
  QUERY_DOCUMENTS_ARGS_SHAPE,
  SEARCH_DOCUMENTS_ARGS_SHAPE,
} from "./utils/documentQuery";
import { withErrorHandling } from "./utils/middlewares";
import { validateCustomFields } from "./utils/monetary";
import { resolveSelectCustomFieldValues } from "./utils/selectFields";
import { CUSTOM_FIELD_VALUE_DESCRIPTION } from "./utils/descriptions";
import {
  buildDocumentResourceUri,
  buildThumbnailResourceUri,
} from "./utils/resourceUri";

export type BulkCustomFieldValue = string | number | boolean | number[] | null;

export type BulkCustomFieldUpdate = {
  field: number;
  value: BulkCustomFieldValue;
};

export type BulkCustomFieldParameters = {
  add_custom_fields?: Record<string, BulkCustomFieldValue>;
  remove_custom_fields?: number[];
};

const MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024;

const ALLOWED_UPLOAD_PATHS = process.env.PAPERLESS_MCP_UPLOAD_PATHS
  ? process.env.PAPERLESS_MCP_UPLOAD_PATHS.split(":")
  : [];

/** Default number of characters returned by a bounded content read. */
export const DEFAULT_CONTENT_MAX_CHARS = 8000;

/** Hard cap on characters a single tool response may inline. */
export const DEFAULT_MAX_INLINE_CHARS = 50000;

/** Default passage size for search_passages. */
export const DEFAULT_PASSAGE_WINDOW = 800;

/** Default maximum number of passages returned by search_passages. */
export const DEFAULT_MAX_PASSAGES = 5;

/**
 * Reads the inline budget from the environment at call time (not module
 * load), so tests can exercise different budgets without re-importing.
 */
export function getMaxInlineChars(): number {
  const parsed = Number.parseInt(
    process.env.PAPERLESS_MCP_MAX_INLINE_CHARS ?? "",
    10
  );
  return Number.isFinite(parsed) && parsed > 0
    ? parsed
    : DEFAULT_MAX_INLINE_CHARS;
}

/**
 * Normalizes bounded-read arguments into a validated {offset, maxChars} pair.
 *
 * Invariants:
 * - offset must be >= 0 and inside the document;
 * - maxChars is clamped to [1, budget] where budget is the inline budget;
 * - a caller-requested window beyond the budget fails loudly instead of
 *   silently shrinking (never silently slice what the caller asked for).
 */
export function resolveBoundedRead(
  offset: number | undefined,
  maxChars: number | undefined,
  contentLength: number,
  budget = getMaxInlineChars()
): { offset: number; maxChars: number } {
  const resolvedOffset = Math.max(0, Math.floor(offset ?? 0));
  if (resolvedOffset > 0 && resolvedOffset >= contentLength) {
    throw new Error(
      `offset ${resolvedOffset} is beyond the end of the document (content_length ${contentLength})`
    );
  }
  const requested = maxChars ?? DEFAULT_CONTENT_MAX_CHARS;
  if (requested > budget) {
    throw new Error(
      `max_chars ${requested} exceeds the inline budget ${budget} ` +
        `(PAPERLESS_MCP_MAX_INLINE_CHARS). Read in windows of <= ${budget} chars ` +
        `using offset, or use the download_document resource for bulk access.`
    );
  }
  const clamped = Math.max(1, Math.floor(requested));
  return { offset: resolvedOffset, maxChars: clamped };
}

/**
 * Extracts a bounded window of document content and describes it.
 *
 * The window is always derived from the stored content server-side; the full
 * text never crosses this boundary un-sliced.
 */
export function sliceDocumentContent(
  doc: Pick<Document, "id" | "title" | "content">,
  offset: number,
  maxChars: number
): {
  id: number;
  title: string;
  content: string;
  offset: number;
  returned_chars: number;
  content_length: number;
  truncated: boolean;
  has_more: boolean;
} {
  const full = doc.content ?? "";
  const window = full.slice(offset, offset + maxChars);
  return {
    id: doc.id,
    title: doc.title,
    content: window,
    offset,
    returned_chars: window.length,
    content_length: full.length,
    truncated: offset + window.length < full.length,
    has_more: offset + window.length < full.length,
  };
}

/**
 * Builds one bounded passage of document content centered on a search-term
 * occurrence.
 *
 * The window is cut from the stored content server-side; the full text never
 * crosses this boundary un-sliced. When the query term cannot be located
 * verbatim (whoosh stems differently than the stored text), the passage falls
 * back to the head of the document rather than returning nothing.
 */
export function buildSearchPassage(
  doc: Pick<Document, "id" | "title" | "content">,
  terms: string[],
  window: number,
  searchHit?: Pick<SearchHit, "score" | "highlights">
): {
  id: number;
  title: string;
  score?: number;
  highlights?: string;
  passage: string;
  offset: number;
  content_length: number;
} {
  const full = doc.content ?? "";
  // Anchor priority: (1) the server's own highlighted phrase, (2) the first
  // verbatim term occurrence, (3) the document head. The highlight is the
  // only anchor that survives whoosh stemming — the search index may match
  // word variants the stored text does not contain verbatim, and the first
  // query term can occur anywhere in the document, so anchoring on either
  // risks cutting a window that does not contain the actual hit.
  const lowerFull = full.toLowerCase();
  let center = -1;
  if (searchHit?.highlights) {
    for (const phrase of extractHighlightPhrases(searchHit.highlights)) {
      const idx = lowerFull.indexOf(phrase.toLowerCase());
      if (idx >= 0) {
        center = idx;
        break;
      }
    }
  }
  if (center < 0) {
    for (const term of terms) {
      const idx = lowerFull.indexOf(term.toLowerCase());
      if (idx >= 0) {
        center = idx;
        break;
      }
    }
  }
  const resolved = center < 0 ? 0 : center;
  const start = Math.max(0, resolved - Math.floor(window / 2));
  const passage = full.slice(start, start + window);
  return {
    id: doc.id,
    title: doc.title,
    ...(searchHit ? { score: searchHit.score, highlights: searchHit.highlights } : {}),
    passage,
    offset: start,
    content_length: full.length,
  };
}

/**
 * Extracts candidate anchor phrases from the server's highlight HTML: markup
 * stripped, HTML entities decoded, whitespace collapsed. Phrases shorter than
 * four characters cannot anchor a window reliably and are skipped so stray
 * short matches never drag the window to a wrong position.
 */
export function extractHighlightPhrases(highlights: string): string[] {
  const text = highlights
    .replace(/<[^>]*>/g, " ")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/\s+/g, " ")
    .trim();
  if (!text) return [];
  // Prefer the longest surrounding run of substantial words: the hit phrase,
  // not a lone term that can match anywhere.
  const words = text.split(" ");
  const phrases: string[] = [];
  let current: string[] = [];
  for (const word of words) {
    if (word.length >= 4) {
      current.push(word);
    } else if (current.length) {
      phrases.push(current.join(" "));
      current = [];
    }
  }
  if (current.length) phrases.push(current.join(" "));
  return phrases.filter((phrase) => phrase.length >= 4).sort((a, b) => b.length - a.length);
}

/**
 * Splits a full-text query into lowercase search terms for passage location.
 * Extracts letter/number runs (Unicode-aware, so accented and non-Latin terms
 * survive intact) so operators and quoting from the query syntax do not
 * pollute the verbatim lookup.
 */
export function extractSearchTerms(query: string): string[] {
  const terms = query
    .toLowerCase()
    .split(/[^\p{L}\p{N}_]+/u)
    .filter((term) => term.length > 0);
  return terms;
}

/** Validates that a file path is safe to read for document upload. */
export async function validateFilePath(filePath: string): Promise<void> {
  if (!isAbsolute(filePath)) {
    throw new Error("file_path must be an absolute path");
  }

  // Resolve symlinks to get canonical path for allowlist checks
  let realPath: string;
  try {
    realPath = await realpath(filePath);
  } catch {
    throw new Error("File not found");
  }

  if (ALLOWED_UPLOAD_PATHS.length > 0) {
    const isAllowed = ALLOWED_UPLOAD_PATHS.some((allowedPath) => {
      return realPath.startsWith(allowedPath + "/") || realPath === allowedPath;
    });
    if (!isAllowed) {
      throw new Error(
        "file_path is outside allowed upload directories. " +
        "Configure PAPERLESS_MCP_UPLOAD_PATHS environment variable to specify allowed paths."
      );
    }
  }

  const stats = await stat(realPath);

  if (!stats.isFile()) {
    throw new Error("Path must point to a regular file");
  }

  if (stats.size > MAX_FILE_SIZE_BYTES) {
    throw new Error(
      `File size (${Math.round(stats.size / 1024 / 1024)}MB) exceeds maximum allowed size (${MAX_FILE_SIZE_BYTES / 1024 / 1024}MB)`
    );
  }

  if (stats.size === 0) {
    throw new Error("File is empty");
  }
}

/**
 * Builds Paperless-NGX bulk edit parameters from base parameters plus optional
 * custom field updates.
 *
 * Paperless-NGX expects custom field bulk updates as an `add_custom_fields`
 * record keyed by custom field id. `addCustomFields` is accepted as an array for
 * the MCP tool schema and transformed into that id-to-value record while
 * preserving supported value types, including `number[]` document links and
 * `null` resets. Passing an empty `addCustomFields` array intentionally produces
 * an empty `add_custom_fields` record.
 *
 * When `includeCustomFieldDefaults` is true, the function also initializes
 * `add_custom_fields` and `remove_custom_fields` with empty defaults using
 * nullish coalescing (`??=`). This keeps the `modify_custom_fields` method's
 * payload shape acceptable to Paperless even when no field values are supplied.
 *
 * @param parameters - Base bulk edit parameters to include in the result.
 * @param addCustomFields - Optional custom field updates to map by field id.
 * @param includeCustomFieldDefaults - Whether to include empty custom field
 * defaults required by `modify_custom_fields`.
 * @returns The merged API parameters with custom field updates transformed into
 * Paperless-NGX's `add_custom_fields` record shape.
 */
export function buildBulkEditParameters<T extends Record<string, unknown>>(
  parameters: T,
  addCustomFields?: BulkCustomFieldUpdate[],
  includeCustomFieldDefaults = false,
  includeTagDefaults = false
): T & BulkCustomFieldParameters {
  const apiParameters: T & BulkCustomFieldParameters = {
    ...parameters,
  };

  if (addCustomFields) {
    apiParameters.add_custom_fields = Object.fromEntries(
      addCustomFields.map((customField) => [
        String(customField.field),
        customField.value,
      ])
    );
  }

  if (includeCustomFieldDefaults) {
    apiParameters.add_custom_fields ??= {};
    apiParameters.remove_custom_fields ??= [];
  }

  if (includeTagDefaults) {
    (apiParameters as Record<string, unknown>).add_tags ??= [];
    (apiParameters as Record<string, unknown>).remove_tags ??= [];
  }

  return apiParameters;
}

/**
 * Defensive rewrite for django-watson full-text queries: strips apostrophes
 * (a known server-side 400 trigger) and rebalances double quotes. Returns
 * null when the rewrite leaves nothing searchable (caller should not retry).
 */
export function sanitizeFullTextQuery(query: string): string | null {
  let cleaned = query.replace(/[\u2018\u2019']/g, "");
  const quotes = (cleaned.match(/"/g) || []).length;
  if (quotes % 2 === 1) {
    cleaned = cleaned.replace(/"/g, " ");
  }
  cleaned = cleaned.replace(/\s+/g, " ").trim();
  // A rewrite that leaves only operators/punctuation is not searchable.
  if (!/[\p{L}\p{N}]/u.test(cleaned)) return null;
  return cleaned;
}

interface SanitizedSearch {
  response: DocumentsResponse;
  querySanitized: boolean;
}

/**
 * Executes a full-text search with one sanitized retry: django-watson rejects
 * some syntactically valid queries with an opaque HTTP 400, so on a 400 the
 * query is defensively rewritten and attempted once more. Django-ORM filter
 * values are never rewritten — only the full-text query string. A persistent
 * failure rethrows the original error with the attempted queries attached.
 */
export async function runDocumentSearch(
  api: PaperlessAPI,
  query: string,
  queryParams: BuildDocumentQueryArgs = {}
): Promise<SanitizedSearch> {
  const attempt = (q: string) =>
    api.getDocuments(buildDocumentQueryString({ ...queryParams, query: q }));
  try {
    return { response: await attempt(query), querySanitized: false };
  } catch (error) {
    const status = (error as { response?: { status?: number } })?.response?.status;
    if (status !== 400) throw error;
    const sanitized = sanitizeFullTextQuery(query);
    if (sanitized === null || sanitized === query) {
      throw enrichSearchError(error, [query]);
    }
    try {
      return { response: await attempt(sanitized), querySanitized: true };
    } catch (retryError) {
      throw enrichSearchError(retryError, [query, sanitized]);
    }
  }
}

function enrichSearchError(error: unknown, queries: string[]): Error {
  const base = error instanceof Error ? error.message : String(error);
  return new Error(
    `${base}. Paperless full-text search rejected these queries: ${queries
      .map((q) => JSON.stringify(q))
      .join(", ")}. Try removing quotes or apostrophes, or use query_documents with tag/correspondent filters instead.`
  );
}

async function executeDocumentQuery(
  api: PaperlessAPI,
  args: BuildDocumentQueryArgs
) {
  const docsResponse = await api.getDocuments(buildDocumentQueryString(args));
  return convertDocsWithNames(docsResponse, api);
}

export function registerDocumentTools(server: McpServer, api: PaperlessAPI) {
  server.tool(
    "bulk_edit_documents",
    "Perform bulk operations on multiple documents. Note: 'remove_tag' removes a tag from specific documents (tag remains in system), while 'delete_tag' permanently deletes a tag from the entire system. ⚠️ WARNING: 'delete' method permanently deletes documents and requires confirmation.",
    {
      documents: z.array(z.number()),
      method: z.enum([
        "set_correspondent",
        "set_document_type",
        "set_storage_path",
        "add_tag",
        "remove_tag",
        "modify_tags",
        "modify_custom_fields",
        "delete",
        "reprocess",
        "set_permissions",
        "merge",
        "split",
        "rotate",
        "delete_pages",
      ]),
      correspondent: z.number().optional(),
      document_type: z.number().optional(),
      storage_path: z.number().optional(),
      tag: z.number().optional(),
      add_tags: z.array(z.number()).optional().transform(arrayNotEmpty),
      remove_tags: z.array(z.number()).optional().transform(arrayNotEmpty),
      add_custom_fields: z
        .array(
          z.object({
            field: z.number(),
            value: z.union([
              z.string(),
              z.number(),
              z.boolean(),
              z.array(z.number()),
              z.null(),
            ]).describe(CUSTOM_FIELD_VALUE_DESCRIPTION),
          })
        )
        .optional()
        .transform(arrayNotEmpty),
      remove_custom_fields: z
        .array(z.number())
        .optional()
        .transform(arrayNotEmpty),
      set_permissions: z
        .object({
          view: z
            .object({
              users: z.array(z.number()).optional(),
              groups: z.array(z.number()).optional(),
            })
            .strict()
            .optional(),
          change: z
            .object({
              users: z.array(z.number()).optional(),
              groups: z.array(z.number()).optional(),
            })
            .strict()
            .optional(),
        })
        // strict: a misspelled action ("read") must not silently collapse
        // into {} and clear permissions/ownership.
        .strict()
        .optional()
        .describe(
          "For set_permissions: view/change permissions to apply. Omitted actions (view/change) and omitted users/groups lists are left untouched; an empty list [] removes all (unless merge is true). Omit entirely for owner-only changes."
        ),
      owner: z
        .number()
        .int()
        .nullable()
        .optional()
        .describe(
          "For set_permissions: new owner user ID, or null to remove the owner. Unless merge is true, omitting owner also clears the current owner."
        ),
      merge: z
        .boolean()
        .optional()
        .describe(
          "For set_permissions: true adds to existing permissions and keeps the current owner; false (default) replaces the listed users/groups"
        ),
      metadata_document_id: z.number().optional(),
      delete_originals: z.boolean().optional(),
      pages: z.string().optional(),
      degrees: z.number().optional(),
      confirm: z
        .boolean()
        .optional()
        .describe(
          "Must be true when method is 'delete' to confirm destructive operation"
        ),
    },
    withErrorHandling(async (args, extra) => {
      if (!api) throw new Error("Please configure API connection first");
      if (args.method === "delete" && !args.confirm) {
        throw new Error(
          "Confirmation required for destructive operation. Set confirm: true to proceed."
        );
      }
      if (
        args.method === "set_permissions" &&
        !args.set_permissions &&
        args.owner === undefined
      ) {
        throw new Error(
          "Method 'set_permissions' requires set_permissions and/or owner."
        );
      }
      const { documents, method, add_custom_fields, confirm, ...parameters } = args;
      if (method === "set_permissions") {
        // Paperless rejects (Paperless <= 3.0.5: crashes with a 500 on) a missing
        // set_permissions key even for owner-only changes.
        parameters.set_permissions ??= {};
      }

      validateCustomFields(add_custom_fields);
      const resolvedCustomFields = await resolveSelectCustomFieldValues(
        api,
        add_custom_fields,
        "stored"
      );

      const response = await api.bulkEditDocuments(
        documents,
        method,
        method === "delete"
          ? {}
          : buildBulkEditParameters(
              parameters,
              resolvedCustomFields,
              method === "modify_custom_fields",
              method === "modify_tags"
            )
      );
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify({ result: response.result || response }),
          },
        ],
      };
    })
  );

  const postDocumentBaseSchema = z.object({
    file: z.string().optional().describe("Base64-encoded file content. Either 'file' or 'file_path' must be provided."),
    file_path: z.string().optional().describe("Absolute path to a file on the server's filesystem. Either 'file' or 'file_path' must be provided. The filename is derived from the path unless 'filename' is also specified. For security, configure PAPERLESS_MCP_UPLOAD_PATHS to restrict allowed directories."),
    filename: z.string().optional().describe("Filename for the uploaded document. Required when using 'file', optional when using 'file_path' (defaults to the basename of the path)."),
    title: z.string().optional(),
    created: z.string().optional(),
    correspondent: z.number().optional(),
    document_type: z.number().optional(),
    storage_path: z.number().optional(),
    tags: z.array(z.number()).optional(),
    archive_serial_number: z.number().optional(),
    custom_fields: z.array(z.number()).optional(),
  });

  const postDocumentSchema = postDocumentBaseSchema.superRefine((data, ctx) => {
    const hasFile = data.file !== undefined;
    const hasFilePath = data.file_path !== undefined;

    if (!hasFile && !hasFilePath) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Either 'file' (base64) or 'file_path' must be provided.",
        path: ["file"],
      });
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Either 'file' (base64) or 'file_path' must be provided.",
        path: ["file_path"],
      });
    }

    if (hasFile && hasFilePath) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Only one of 'file' or 'file_path' should be provided, not both.",
        path: ["file"],
      });
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Only one of 'file' or 'file_path' should be provided, not both.",
        path: ["file_path"],
      });
    }

    if (hasFile && !data.filename) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "'filename' is required when using 'file' (base64 mode).",
        path: ["filename"],
      });
    }

    if (hasFilePath && data.file_path && !isAbsolute(data.file_path)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "file_path must be an absolute path",
        path: ["file_path"],
      });
    }
  });

  server.tool(
    "post_document",
    "Upload a new document to Paperless-NGX with optional metadata like title, correspondent, document type, tags, and custom fields. Provide either 'file' (base64-encoded content) or 'file_path' (absolute path to a file on the server's filesystem). Using file_path avoids base64 encoding overhead for large files. SECURITY: When using file_path, set PAPERLESS_MCP_UPLOAD_PATHS environment variable to restrict uploads to specific directories (colon-separated paths).",
    postDocumentBaseSchema.shape,
    withErrorHandling(async (args, extra) => {
      if (!api) throw new Error("Please configure API connection first");

      const validationResult = postDocumentSchema.safeParse(args);
      if (!validationResult.success) {
        throw new Error(validationResult.error.issues.map(e => e.message).join("; "));
      }

      let document: Buffer;
      let filename: string;

      if (args.file_path) {
        await validateFilePath(args.file_path);

        try {
          document = await readFile(args.file_path);
        } catch (err) {
          throw new Error("Failed to read file");
        }

        filename = args.filename || basename(args.file_path);
        if (!filename) {
          throw new Error("Could not derive filename from file_path");
        }
      } else if (args.file) {
        const base64Regex = /^[A-Za-z0-9+/]*={0,2}$/;
        if (!base64Regex.test(args.file)) {
          throw new Error(
            "Invalid base64-encoded file data. Please provide a valid base64 string."
          );
        }

        document = Buffer.from(args.file, "base64");

        if (document.length > MAX_FILE_SIZE_BYTES) {
          throw new Error(
            `File size (${Math.round(document.length / 1024 / 1024)}MB) exceeds maximum allowed size (${MAX_FILE_SIZE_BYTES / 1024 / 1024}MB)`
          );
        }
        if (document.length === 0) {
          throw new Error("File is empty");
        }

        filename = args.filename!;
      } else {
        // This should never happen due to schema validation, but TypeScript needs it
        throw new Error("Either 'file' (base64) or 'file_path' must be provided.");
      }

      const { file, file_path, filename: _fn, ...metadata } = args;

      const response = await api.postDocument(document, filename, metadata);
      let result;
      if (typeof response === "string" && /^\d+$/.test(response)) {
        result = { id: Number(response) };
      } else {
        result = { status: response };
      }
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(result),
          },
        ],
      };
    })
  );

  server.tool(
    "list_documents",
    "List and filter documents with pagination and common Paperless filters such as title search, correspondent, document type, tag, storage path, creation date, archive serial number, and simple custom field filters. Use 'query_documents' for full-text query, structured custom field conditions, or advanced documented /api/documents/ query parameters. IMPORTANT: For queries like 'the last 3 contributions' or when searching by tag, correspondent, document type, or storage path, first use the relevant lookup tool to find the correct ID. Note: Document content is excluded from results by default. Use 'get_document_content' when you need the document text.",
    LIST_DOCUMENTS_ARGS_SHAPE,
    withErrorHandling(async (args, extra) => {
      if (!api) throw new Error("Please configure API connection first");
      return executeDocumentQuery(api, args);
    })
  );

  server.tool(
    "search_passages",
    "Search documents full-text and return bounded text passages around each hit instead of whole documents. Locates the query terms in each matching document's stored content, cuts a window (default 800 chars) around the first occurrence, and returns it with the document id, title, score, offset, content_length, and the server's own highlighted snippet. Use this to locate and skim evidence across many documents without flooding context; page further with get_document_content offsets.",
    {
      query: z
        .string()
        .min(1)
        .describe("Full-text query (same syntax as query_documents)."),
      window: z
        .number()
        .int()
        .min(200)
        .max(4000)
        .optional()
        .describe("Characters per passage (default 800)."),
      max_passages: z
        .number()
        .int()
        .min(1)
        .max(20)
        .optional()
        .describe("Maximum passages to return (default 5)."),
      page_size: z.number().optional(),
      correspondent: z.number().optional(),
      document_type: z.number().optional(),
      tag: z.number().optional(),
      storage_path: z.number().optional(),
      created__date__gte: z.string().optional(),
      created__date__lte: z.string().optional(),
      ordering: z.string().optional(),
      custom_fields__icontains: z.string().optional(),
      paperless_filters: z
        .record(
          z.string(),
          z.union([
            z.string(),
            z.number(),
            z.boolean(),
            z.array(z.union([z.string(), z.number(), z.boolean()])),
          ])
        )
        .optional(),
    },
    withErrorHandling(async (args, extra) => {
      if (!api) throw new Error("Please configure API connection first");
      const terms = extractSearchTerms(args.query);
      // Reuse the same query-string builder as query_documents so filters
      // behave identically; page_size bounds how many documents are scanned.
      // window / max_passages are tool-only knobs; everything else in args
      // (including the query itself) is a valid query_documents parameter.
      const {
        window: _window,
        max_passages: _maxPassages,
        ...searchArgs
      } = args;
      const window = args.window ?? DEFAULT_PASSAGE_WINDOW;
      const maxPassages = args.max_passages ?? DEFAULT_MAX_PASSAGES;
      // Each hit arrives with its full stored content, so cap the fetch at
      // the number of passages we will actually return unless the caller
      // asked for a larger scan explicitly.
      // The full-text query rides the same 400-resilient path as the other
      // search tools; structured filters (in searchArgs) are never rewritten.
      const { query: _rawQuery, ...filterArgs } = searchArgs;
      const { response: docsResponse, querySanitized } = await runDocumentSearch(
        api,
        args.query,
        { ...filterArgs, page_size: args.page_size ?? maxPassages }
      );
      const passages = (docsResponse.results || [])
        .slice(0, maxPassages)
        .map((doc) => buildSearchPassage(doc, terms, window, doc.__search_hit__));
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify({
              query: args.query,
              total_hits: docsResponse.count,
              ...(querySanitized ? { query_sanitized: true } : {}),
              passages,
            }),
          },
        ],
      };
    })
  );

  server.tool(
    "query_documents",
    "Query documents using the full-text query engine plus structured Paperless filters. Use this for complex filtering, custom field conditions, or any documented /api/documents/ query parameters that are not exposed as first-class arguments. Prefer the dedicated top-level arguments where available. custom_field_query supports [field_name_or_id, operator, value] leaves or ['AND'|'OR', [clause1, clause2]] groups. Note: Document content is excluded from results by default. Use 'get_document_content' when you need the document text.",
    QUERY_DOCUMENTS_ARGS_SHAPE,
    withErrorHandling(async (args, extra) => {
      if (!api) throw new Error("Please configure API connection first");
      return executeDocumentQuery(api, args);
    })
  );

  server.tool(
    "get_document",
    "Get a specific document by ID with full details including correspondent, document type, tags, and custom fields. Note: Document content is excluded from results by default. Use 'get_document_content' to retrieve content when needed.",
    {
      id: z.number(),
    },
    withErrorHandling(async (args, extra) => {
      if (!api) throw new Error("Please configure API connection first");
      const doc = await api.getDocument(args.id);
      return convertDocsWithNames(doc, api);
    })
  );

  server.tool(
    "get_document_content",
    "Get the text content of a specific document by ID, in bounded windows. Returns at most max_chars characters (default 8000) starting at offset, plus content_length and pagination metadata so callers can page through long documents. Use offset + max_chars to read successive windows of a long document; use the download_document resource for bulk access to the raw file.",
    {
      id: z.number(),
      offset: z
        .number()
        .int()
        .min(0)
        .optional()
        .describe("Character offset to start reading from (default 0)."),
      max_chars: z
        .number()
        .int()
        .min(1)
        .optional()
        .describe(
          "Maximum characters to return (default 8000). Requests above the inline budget fail with the document size instead of returning everything."
        ),
    },
    withErrorHandling(async (args, extra) => {
      if (!api) throw new Error("Please configure API connection first");
      const doc = await api.getDocument(args.id);
      const { offset, maxChars } = resolveBoundedRead(
        args.offset,
        args.max_chars,
        doc.content?.length ?? 0
      );
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              sliceDocumentContent(doc, offset, maxChars)
            ),
          },
        ],
      };
    })
  );

  server.tool(
    "search_documents",
    "Deprecated compatibility wrapper for full-text document search. Use 'query_documents' with the 'query' argument for new integrations. Note: Document content is excluded from results by default. Use 'get_document_content' to retrieve content when needed.",
    SEARCH_DOCUMENTS_ARGS_SHAPE,
    withErrorHandling(async (args, extra) => {
      if (!api) throw new Error("Please configure API connection first");
      return executeDocumentQuery(api, args);
    })
  );

  server.tool(
    "download_document",
    "Download a document file by ID. Returns a paperless:// resource URI; read the resource to fetch the file content.",
    {
      id: z.number().int().positive(),
      original: z.boolean().optional(),
    },
    withErrorHandling(async (args, extra) => {
      if (!api) throw new Error("Please configure API connection first");
      const uri = buildDocumentResourceUri(args.id, {
        original: args.original,
      });
      return {
        content: [
          // Legacy clients surface only content[].text and drop resource blocks
          // entirely, so the URI is repeated here to stay reachable (issue #134).
          { type: "text", text: uri },
          {
            type: "resource",
            resource: {
              uri,
              // MCP SDK 1.11 embedded resources require text or blob. Keep the
              // existing resource-shaped tool result while making resources/read
              // the canonical place for the large binary payload.
              text: "",
              mimeType: "application/octet-stream",
            },
          },
        ],
      };
    })
  );

  server.tool(
    "get_document_thumbnail",
    "Get a document thumbnail (image preview) by ID. Returns a paperless:// resource URI; read the resource to fetch the image content.",
    {
      id: z.number().int().positive(),
    },
    withErrorHandling(async (args, extra) => {
      if (!api) throw new Error("Please configure API connection first");
      const uri = buildThumbnailResourceUri(args.id);
      return {
        content: [
          // See download_document above: the URI is repeated as text for legacy
          // clients that drop resource blocks.
          { type: "text", text: uri },
          {
            type: "resource",
            resource: {
              uri,
              // See download_document above: the binary thumbnail is fetched
              // lazily through resources/read instead of embedded here.
              text: "",
              mimeType: "image/webp",
            },
          },
        ],
      };
    })
  );

  server.tool(
    "update_document",
    "Update a specific document with new values (title, correspondent, document type, storage path, tags, custom fields, and more). Top-level fields you omit are left unchanged. IMPORTANT: custom_fields is the exception — see its parameter description; it replaces the document's entire custom-field set.",
    {
      id: z.number().describe("The ID of the document to update"),
      title: z
        .string()
        .max(128)
        .optional()
        .describe("The new title for the document (max 128 characters)"),
      correspondent: z
        .number()
        .int()
        .nullable()
        .optional()
        .describe("The ID of the correspondent to assign"),
      document_type: z
        .number()
        .int()
        .nullable()
        .optional()
        .describe("The ID of the document type to assign"),
      storage_path: z
        .number()
        .int()
        .nullable()
        .optional()
        .describe("The ID of the storage path to assign"),
      tags: z
        .array(z.number())
        .optional()
        .describe("Array of tag IDs to assign to the document"),
      content: z
        .string()
        .optional()
        .describe("The raw text content of the document (used for searching)"),
      created: z
        .string()
        .optional()
        .describe("The creation date in YYYY-MM-DD format"),
      archive_serial_number: z
        .number()
        .optional()
        .describe("The archive serial number (0-4294967295)"),
      owner: z
        .number()
        .int()
        .nullable()
        .optional()
        .describe("The ID of the user who owns the document"),
      custom_fields: z
        .array(
          z.object({
            field: z.number().describe("The custom field ID"),
            value: z
              .union([
                z.string(),
                z.number(),
                z.boolean(),
                z.array(z.number()),
                z.null(),
              ])
              .describe(CUSTOM_FIELD_VALUE_DESCRIPTION),
          })
        )
        .optional()
        .describe(
          "Custom field values for the document. ⚠️ REPLACES the document's entire custom-field set — any field not included here will be CLEARED. To update or add a single field without losing the others, first call get_document to read the existing custom_fields, then pass the full merged array. To add/set fields additively without fetching, use bulk_edit_documents with method 'modify_custom_fields' instead."
        ),
    },
    withErrorHandling(async (args, extra) => {
      if (!api) throw new Error("Please configure API connection first");
      const { id, ...updateData } = args;

      validateCustomFields(updateData.custom_fields);
      updateData.custom_fields = await resolveSelectCustomFieldValues(
        api,
        updateData.custom_fields,
        "stored"
      );

      const response = await api.updateDocument(id, updateData);

      return convertDocsWithNames(response, api);
    })
  );
}
