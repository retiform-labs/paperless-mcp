<!-- [![MseeP.ai Security Assessment Badge](https://mseep.net/pr/nloui-paperless-mcp-badge.png)](https://mseep.ai/app/nloui-paperless-mcp) -->

# Paperless-NGX MCP Server

![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/retiform-labs/paperless-mcp?utm_source=oss&utm_medium=github&utm_campaign=retiform-labs%2Fpaperless-mcp&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)

An MCP (Model Context Protocol) server for interacting with a Paperless-NGX API server. This server provides tools for managing documents, tags, correspondents, and document types in your Paperless-NGX instance.

## About this fork

This is a fork of [baruchiro/paperless-mcp](https://github.com/baruchiro/paperless-mcp)
(ISC license, (c) Baruch Odem), maintained by **Retiform Labs** and published as
**`@retiform/paperless-mcp`**. Fork point: tag `upstream-v2.2.1`.

Upstream returns whole documents. That is fine for a human clicking through a UI
and crippling for an agent: a single `get_document_content` call on a
several-hundred-page PDF can consume an entire context window, and there is no
way to ask for a slice. This fork adds bounded reads and local semantic
search:

- **Ranged `get_document_content`** -- `offset` + `max_chars` windows, with honest
  `content_length` / `truncated` / `has_more` metadata.
- **`search_passages`** -- full-text hits returned as bounded passages around the
  actual match, with score and offset, so an agent can locate and then page into
  the exact region it needs.
- **An inline budget** -- `PAPERLESS_MCP_MAX_INLINE_CHARS` makes an oversized
  request fail loudly instead of silently flooding the caller.
- **`content_length`** on list/query/get results, so a client can size a read
  before making it.
- **Query resilience** -- an apostrophe in a search query is retried sanitized
  instead of returning a hard 400.
- **Semantic search** -- `semantic_search` searches by meaning as well as by
  words, fusing a local vector index with the lexical search above and
  returning the same bounded-passage shape as `search_passages`. Optional,
  local-only, and built explicitly with `sync_embeddings`; `embedding_status`
  reports what the index holds. See [Semantic search setup](#semantic-search-setup).

We also fix the build so it is idempotent (`tsc` no longer consumes its own
`.d.ts` output on a rebuild). Upstream was unable to take this work, and agentic
clients need it now, so it is published here. The delta from `upstream-v2.2.1`
is small and self-contained if upstream ever wants it.

## Quick Start

[![Install MCP Server](https://cursor.com/deeplink/mcp-install-light.svg)](https://cursor.com/install-mcp?name=paperless&config=eyJjb21tYW5kIjoibnB4IC15IEByZXRpZm9ybS9wYXBlcmxlc3MtbWNwQGxhdGVzdCIsImVudiI6eyJQQVBFUkxFU1NfVVJMIjoiaHR0cDovL3lvdXItcGFwZXJsZXNzLWluc3RhbmNlOjgwMDAiLCJQQVBFUkxFU1NfQVBJX0tFWSI6InlvdXItYXBpLXRva2VuIn19)

### Installation

Add these to your MCP config file:

// STDIO mode (recommended for local or CLI use)
```json
"paperless": {
  "command": "npx",
  "args": [
    "-y",
    "@retiform/paperless-mcp@latest",
  ],
  "env": {
    "PAPERLESS_URL": "http://your-paperless-instance:8000",
    "PAPERLESS_API_KEY": "your-api-token",
    "PAPERLESS_PUBLIC_URL": "https://your-public-domain.com"
  }
}
```

// HTTP mode (recommended for Docker or remote use)
```json
"paperless": {
  "command": "docker",
  "args": [
    "run",
    "-i",
    "--rm",
    "ghcr.io/retiform-labs/paperless-mcp:latest",
  ],
  "env": {
    "PAPERLESS_URL": "http://your-paperless-instance:8000",
    "PAPERLESS_API_KEY": "your-api-token",
    "PAPERLESS_PUBLIC_URL": "https://your-public-domain.com"
  }
}
```

3. Get your API token:
   1. Log into your Paperless-NGX instance
   2. Click your username in the top right
   3. Select "My Profile"
   4. Click the circular arrow button to generate a new token

4. Replace the placeholders in your MCP config:
   - `http://your-paperless-instance:8000` with your Paperless-NGX URL
   - `your-api-token` with the token you just generated
   - `https://your-public-domain.com` with your public Paperless-NGX URL (optional, falls back to PAPERLESS_URL)

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `PAPERLESS_URL` | Yes | — | Base URL of your Paperless-NGX instance |
| `PAPERLESS_API_KEY` | Yes | — | API token from your Paperless-NGX profile |
| `PAPERLESS_PUBLIC_URL` | No | `PAPERLESS_URL` | Public-facing URL for document links |
| `PAPERLESS_API_VERSION` | No | `9` | Paperless-ngx REST API version. `9` works on Paperless-ngx v2.x (recent) and v3.x. Paperless-ngx v3.0.0 dropped support for versions below `9`, so older defaults now return HTTP 406. If you see HTTP 406 errors, set this to a version your server supports. |
| `PAPERLESS_MCP_UPLOAD_PATHS` | No | — | Colon-separated list of allowed directories for `file_path` uploads. **Recommended for security.** Example: `/var/uploads:/tmp/scans` |
| `PAPERLESS_MCP_MAX_INLINE_CHARS` | No | `50000` | Hard cap on characters a single tool response may inline. Requests above the budget fail with the document size and a hint toward bounded reads instead of silently truncating or flooding context. |
| `PAPERLESS_MCP_EMBEDDING_URL` *(Search)* | No | `http://127.0.0.1:11434` | Base URL of an OpenAI-compatible `/embeddings` endpoint. Either reply envelope is accepted — `{embeddings: [[...]]}` (Ollama native) or `{data: [{embedding: [...]}]}` (OpenAI-compatible) — so no setting tells the client which shape to expect. Setting this (or any other `PAPERLESS_MCP_*` search variable) enables the semantic-search tools. |
| `PAPERLESS_MCP_EMBEDDING_LOCAL_ONLY` *(Search)* | No | `false` | When `true`, the endpoint must be a literal loopback (`127.0.0.1`, `::1`) or RFC1918 address (`10.x`, `172.16-31.x`, `192.168.x`); a DNS name is refused, because a name can resolve anywhere. Set this if indexed text must not leave your network. Off by default so a hosted provider works without a policy you did not ask for. |
| `PAPERLESS_MCP_EMBEDDING_API_KEY` *(Search)* | No | — | Sent as `Authorization: Bearer <key>`. Never logged, never included in an error message, and held in a private field so it does not appear in a serialised client object. Leaving it unset sends no authorization header at all. |
| `PAPERLESS_MCP_EMBEDDING_MODEL` *(Search)* | No | `nomic-embed-text` | Embedding model name, requested from the server above. Must produce 768-dimension vectors. |
| `PAPERLESS_MCP_EMBEDDING_PREFIX_MODE` *(Search)* | No | `nomic` | One of `nomic`, `none`, `e5` — the prefix template applied to embedded text. The prefix is **baked into every stored vector at build time**, so a query built with a different mode returns quietly worse results rather than an error. A mismatch between the stored mode and the configured one is refused loudly; changing this requires a rebuild. |
| `PAPERLESS_MCP_INDEX_DIR` *(Search)* | No | `~/.local/share/paperless-mcp/index` | Directory holding `index.db`. Outside your Paperless-ngx container by design. |
| `PAPERLESS_MCP_MAX_CHUNKS` *(Search)* | No | `4000` | Maximum chunks stored per document. A longer document is truncated, and `embedding_status` reports how many were. Clamped to what the row-id encoding can address; a larger value is reduced rather than refused. |
| `PAPERLESS_MCP_SYNC_BATCH_SIZE` *(Search)* | No | `16` | Texts sent per embedding request during a sync (1–256). Lower it if your embedding server falls over on larger batches. |
| `PAPERLESS_MCP_SYNC_INTERVAL_MS` *(Search)* | No | `0` (off) | Interval between automatic incremental syncs (0–86400000). A positive value starts a background timer when the tools register; it runs one sync immediately, then once per interval. `0` means no timer, which is the default — drive `sync_embeddings` from your client instead. The timer is cleared on `close()`. |
| `PAPERLESS_MCP_WEBHOOK_PORT` *(Search)* | No | `0` (off) | Port for the change-notification receiver. A positive value starts a listener that re-indexes documents as they change, so the index stays current without anything polling. **Off by default**: an install that never asked for a listening socket must not acquire one by upgrading. Requires `PAPERLESS_MCP_WEBHOOK_SECRET`; the receiver refuses to start without one. |
| `PAPERLESS_MCP_WEBHOOK_SECRET` *(Search)* | Only with the port | — | Shared secret the sender must present, in the `x-paperless-mcp-secret` header or as a `Bearer` token. Compared in constant time. The receiver **refuses to start** when this is unset and the port is set, because an unauthenticated endpoint would let anyone trigger embedding work. |
| `PAPERLESS_MCP_WEBHOOK_HOST` *(Search)* | No | `127.0.0.1` | Address to bind. Loopback unless you widen it deliberately. **A containerised Paperless-ngx cannot reach `127.0.0.1` on the host** — see the receiver section below. |
| `PAPERLESS_MCP_WEBHOOK_DEBOUNCE_MS` *(Search)* | No | `2000` | Quiet period before a burst of deliveries becomes one sync. A bulk edit arrives as one request per document within a few seconds; coalescing them means 200 events cost one run, not 200. |

Rows marked *(Search)* belong to the optional semantic-search feature: **setting any one of them
enables the three search tools**, and each value you set is validated strictly rather than silently
defaulted. Four of them also accept an unprefixed fallback name — `EMBEDDING_URL`, `EMBEDDING_MODEL`,
`EMBEDDING_PREFIX_MODE`, `SYNC_INTERVAL_MS` — tried only when the prefixed form is unset. Prefer the
prefixed names: the unprefixed ones are generic, so any other MCP server sharing the same process
environment can collide with them.

That's it! Now you can ask Claude to help you manage your Paperless-NGX documents.

### Example Usage

Here are some things you can ask Claude to do:

- "Show me all documents tagged as 'Invoice'"
- "Search for documents containing 'tax return'"
- "Create a new tag called 'Receipts' with color #FF0000"
- "Download document #123"
- "List all correspondents"
- "Create a new document type called 'Bank Statement'"

## Available Tools

### Document Operations

#### list_documents
Get a paginated list of documents with simple filters. Use this for straightforward listing tasks. For full-text queries, structured custom field filtering, or advanced Paperless filters, use `query_documents`.

Parameters:
- page (optional): Page number
- page_size (optional): Number of documents per page
- search (optional): Simple Paperless search term
- correspondent (optional): Correspondent ID
- document_type (optional): Document type ID
- tag (optional): Tag ID
- storage_path (optional): Storage path ID
- created__date__gte (optional): Created date on or after YYYY-MM-DD
- created__date__lte (optional): Created date on or before YYYY-MM-DD
- ordering (optional): Paperless ordering field
- archive_serial_number (optional): Archive serial number
- archive_serial_number__isnull (optional): Whether the archive serial number is empty
- custom_field_query (optional): Raw JSON-encoded Paperless custom field query string
- custom_fields__icontains (optional): Case-insensitive substring match across custom field values

```typescript
list_documents({
  page: 1,
  page_size: 25
})
```

#### query_documents
Canonical document query tool. Supports full-text querying, simple Paperless search, custom field filters, and documented `/api/documents/` Paperless query parameters.

Parameters:
- page (optional): Page number
- page_size (optional): Number of documents per page
- ordering (optional): Paperless ordering field
- query (optional): Full-text query string
- search (optional): Simple Paperless search term
- more_like_id (optional): Find documents similar to this document ID
- correspondent (optional): Correspondent ID
- document_type (optional): Document type ID
- tag (optional): Tag ID
- storage_path (optional): Storage path ID
- created__date__gte (optional): Created date on or after YYYY-MM-DD
- created__date__lte (optional): Created date on or before YYYY-MM-DD
- custom_field_query (optional): Structured Paperless custom field query using `[field_name_or_id, operator, value]` leaves or `["AND" | "OR", [clause1, clause2]]` groups
- paperless_filters (optional): Additional documented `/api/documents/` Paperless query parameters, passed as key/value pairs

```typescript
// Full-text query
query_documents({
  query: "invoice 2024"
})

// Simple search term
query_documents({
  search: "acme"
})

// Custom field exact match
query_documents({
  custom_field_query: ["Invoice Number", "exact", "12345"]
})

// Custom field empty
query_documents({
  custom_field_query: ["OR", [
    ["Invoice Number", "isnull", true],
    ["Invoice Number", "exact", ""]
  ]]
})

// Custom field missing
query_documents({
  custom_field_query: ["Invoice Number", "exists", false]
})

// Combined filters
query_documents({
  query: "invoice",
  tag: 5,
  created__date__gte: "2024-01-01",
  custom_field_query: ["Invoice Number", "exists", true]
})

// One documented Paperless filter that is not a first-class argument
query_documents({
  paperless_filters: {
    id__in: [101, 202, 303]
  }
})
```

#### get_document
Get a specific document by ID.

Parameters:
- id: Document ID

```typescript
get_document({
  id: 123
})
```

#### search_documents
Deprecated compatibility wrapper for full-text search. Prefer `query_documents({ query: ... })` for new integrations.

Parameters:
- query: Search query string

```typescript
search_documents({
  query: "invoice 2024"
})
```

#### get_document_content
Get the text content of a document by ID **in bounded windows**. Returns at most `max_chars` characters (default 8000) starting at `offset`, plus pagination metadata so callers can page through long documents without flooding context.

Parameters:
- id: Document ID
- offset (optional): Character offset to start reading from (default 0)
- max_chars (optional): Maximum characters to return (default 8000)

```typescript
get_document_content({ id: 123 })
// → { content: "<first 8000 chars>", offset: 0, content_length: 726573, returned_chars: 8000, truncated: true, has_more: true }

get_document_content({ id: 123, offset: 8000, max_chars: 8000 })
// → next window
```

Requests above the inline budget (`PAPERLESS_MCP_MAX_INLINE_CHARS`, default 50000) fail with the document size and a hint toward offset reads or the `download_document` resource — never a silent slice, never an unbounded return. The response always carries `content_length`, `truncated`, and `has_more` so callers can page deterministically.

#### search_passages
Search documents full-text and return **bounded text passages around each hit** instead of whole documents. Each window is anchored on the server's own highlighted phrase (falling back to the **longest matching query term**, then the document head), so the returned passage contains the actual hit even when the search index stemmed a word variant that does not appear verbatim in the stored text. Returns the document id, title, score, offset, `content_length`, and the highlighted snippet.

Queries that the server rejects with an opaque HTTP 400 (django-watson rejects some syntactically valid queries, notably those containing apostrophes) are retried once with a defensively rewritten query — apostrophes and stray quotes removed; structured filter values are never rewritten. Responses from a rewritten retry carry `query_sanitized: true`. Persistent failures throw a structured error naming every attempted query.

Parameters:
- query: Search query string (same syntax as `query_documents`)
- window (optional): Characters per passage (default 800, range 200–4000)
- max_passages (optional): Maximum passages to return (default 5, cap 20)

```typescript
search_passages({ query: "invoice 2024" })
// → { query: "invoice 2024", total_hits: 3, passages: [{ id, title, score, highlights, passage, offset, content_length }, ...] }

search_passages({ query: "God's will" })
// → { ..., query_sanitized: true, ... }  // apostrophe stripped on retry after a 400
```

Use this to locate and skim evidence across many documents in a single call; page further into a hit with `get_document_content` offsets. Standard `query_documents` filters (`correspondent`, `tag`, `document_type`, `storage_path`, date ranges, `paperless_filters`) are accepted and behave identically. Note: `total_hits` reflects the server's OR-expanded match count and may exceed the number of top-ranked passages returned.

#### download_document
Download a document file by ID.

Parameters:
- id: Document ID
- original (optional): If true, downloads original file instead of archived version

```typescript
download_document({
  id: 123,
  original: false
})
```

#### get_document_thumbnail
Get a document thumbnail (image preview) by ID. Returns the thumbnail as a base64-encoded WebP image resource.

Parameters:
- id: Document ID

```typescript
get_document_thumbnail({
  id: 123
})
```

#### bulk_edit_documents
Perform bulk operations on multiple documents.

Parameters:
- documents: Array of document IDs
- method: One of:
  - set_correspondent: Set correspondent for documents
  - set_document_type: Set document type for documents
  - set_storage_path: Set storage path for documents
  - add_tag: Add a tag to documents
  - remove_tag: Remove a tag from documents
  - modify_tags: Add and/or remove multiple tags
  - delete: Delete documents
  - reprocess: Reprocess documents
  - set_permissions: Set document permissions
  - merge: Merge multiple documents
  - split: Split a document into multiple documents
  - rotate: Rotate document pages
  - delete_pages: Delete specific pages from a document
- Additional parameters based on method:
  - correspondent: ID for set_correspondent
  - document_type: ID for set_document_type
  - storage_path: ID for set_storage_path
  - tag: ID for add_tag/remove_tag
  - add_tags: Array of tag IDs for modify_tags
  - remove_tags: Array of tag IDs for modify_tags
  - set_permissions: Object for set_permissions with view/change users and groups (`{"view": {"users": [], "groups": []}, "change": {...}}`). Omitted actions/lists are left untouched
  - owner: User ID (or null to remove) for set_permissions. Unless merge is true, omitting owner clears the current owner
  - merge: Boolean for set_permissions — true adds to existing permissions and keeps the owner; false (default) replaces the listed users/groups
  - metadata_document_id: ID for merge to specify metadata source
  - delete_originals: Boolean for merge/split
  - pages: String for split "[1,2-3,4,5-7]" or delete_pages "[2,3,4]"
  - degrees: Number for rotate (90, 180, or 270)

Examples:
```typescript
// Add a tag to multiple documents
bulk_edit_documents({
  documents: [1, 2, 3],
  method: "add_tag",
  tag: 5
})

// Set correspondent and document type
bulk_edit_documents({
  documents: [4, 5],
  method: "set_correspondent",
  correspondent: 2
})

// Merge documents
bulk_edit_documents({
  documents: [6, 7, 8],
  method: "merge",
  metadata_document_id: 6,
  delete_originals: true
})

// Split document into parts
bulk_edit_documents({
  documents: [9],
  method: "split",
  pages: "[1-2,3-4,5]"
})

// Modify multiple tags at once
bulk_edit_documents({
  documents: [10, 11],
  method: "modify_tags",
  add_tags: [1, 2],
  remove_tags: [3, 4]
})

// Modify custom fields
bulk_edit_documents({
  documents: [12, 13],
  method: "modify_custom_fields",
  add_custom_fields: [
    { field: 2, value: "year" }
  ],
  remove_custom_fields: []
})

// Set an empty custom field value, e.g. a date field used as a pending marker
bulk_edit_documents({
  documents: [14],
  method: "modify_custom_fields",
  add_custom_fields: [
    { field: 9, value: "" }
  ],
  remove_custom_fields: []
})
```

#### post_document
Upload a new document to Paperless-NGX.

**Two upload modes:**

1. **Base64 mode** (traditional): Provide `file` (base64-encoded content) + `filename`
2. **Filesystem mode** (efficient): Provide `file_path` (absolute path on server)

**Security Note:** When using `file_path`, set the `PAPERLESS_MCP_UPLOAD_PATHS` environment variable (colon-separated list of allowed directories) to restrict uploads to specific locations. Without this, any file on the server's filesystem could be uploaded.

Parameters:
- file (optional): Base64 encoded file content. Either `file` or `file_path` required.
- file_path (optional): Absolute path to file on server's filesystem. Either `file` or `file_path` required.
- filename (optional): Name of the file. Required with `file`, optional with `file_path` (derives from path).
- title (optional): Title for the document
- created (optional): DateTime when the document was created (e.g. "2024-01-19" or "2024-01-19 06:15:00+02:00")
- correspondent (optional): ID of a correspondent
- document_type (optional): ID of a document type
- storage_path (optional): ID of a storage path
- tags (optional): Array of tag IDs
- archive_serial_number (optional): Archive serial number
- custom_fields (optional): Array of custom field IDs

**File size limit:** 100MB for both modes

```typescript
// Base64 mode (traditional)
post_document({
  file: "base64_encoded_content",
  filename: "invoice.pdf",
  title: "January Invoice",
  created: "2024-01-19",
  correspondent: 1,
  document_type: 2,
  tags: [1, 3],
  archive_serial_number: "2024-001",
  custom_fields: [1, 2]
})

// Filesystem mode (more efficient for large files)
post_document({
  file_path: "/var/uploads/invoice.pdf",
  title: "January Invoice",
  correspondent: 1,
  document_type: 2,
  tags: [1, 3]
})
```

### Document Notes

#### list_document_notes
List all notes attached to a document.

Parameters:
- id: Document ID

```typescript
list_document_notes({
  id: 123
})
```

#### create_document_note
Add a note to a document. Returns the document's full list of notes.

Parameters:
- id: Document ID
- note: The note text to add

```typescript
create_document_note({
  id: 123,
  note: "Invoice paid on 2026-06-30 from Commerzbank account."
})
```

#### delete_document_note
⚠️ Delete a single note from a document by its note ID. This operation is irreversible.

Parameters:
- id: Document ID
- note_id: The ID of the note to delete
- confirm: Must be `true` to confirm this destructive operation

```typescript
delete_document_note({
  id: 123,
  note_id: 5,
  confirm: true
})
```

### Tag Operations

#### list_tags
Get all tags.

```typescript
list_tags()
```

#### create_tag
Create a new tag.

Parameters:
- name: Tag name
- color (optional): Hex color code (e.g. "#ff0000")
- match (optional): Text pattern to match
- matching_algorithm (optional): Number between 0 and 6:
  0 - None
  1 - Any word
  2 - All words
  3 - Exact match
  4 - Regular expression
  5 - Fuzzy word
  6 - Automatic

```typescript
create_tag({
  name: "Invoice",
  color: "#ff0000",
  match: "invoice",
  matching_algorithm: 5
})
```

### Correspondent Operations

#### list_correspondents
Get all correspondents.

```typescript
list_correspondents()
```

#### create_correspondent
Create a new correspondent.

Parameters:
- name: Correspondent name
- match (optional): Text pattern to match
- matching_algorithm (optional): Number between 0 and 6:
  0 - None
  1 - Any word
  2 - All words
  3 - Exact match
  4 - Regular expression
  5 - Fuzzy word
  6 - Automatic

```typescript
create_correspondent({
  name: "ACME Corp",
  match: "ACME",
  matching_algorithm: 5
})
```

### Document Type Operations

#### list_document_types
Get all document types.

```typescript
list_document_types()
```

#### create_document_type
Create a new document type.

Parameters:
- name: Document type name
- match (optional): Text pattern to match
- matching_algorithm (optional): Number between 0 and 6:
  0 - None
  1 - Any word
  2 - All words
  3 - Exact match
  4 - Regular expression
  5 - Fuzzy word
  6 - Automatic

```typescript
create_document_type({
  name: "Invoice",
  match: "invoice total amount due",
  matching_algorithm: 1
})
```

### Custom Field Operations

#### list_custom_fields
Get all custom fields.

```typescript
list_custom_fields()
```

#### get_custom_field
Get a specific custom field by ID.

Parameters:
- id: Custom field ID

```typescript
get_custom_field({
  id: 1
})
```

#### create_custom_field
Create a new custom field.

Parameters:
- name: Custom field name
- data_type: One of "string", "url", "date", "boolean", "integer", "float", "monetary", "documentlink", "select"
- extra_data (optional): Extra data for the custom field, such as select options

```typescript
create_custom_field({
  name: "Invoice Number",
  data_type: "string"
})
```

#### update_custom_field
Update an existing custom field.

Parameters:
- id: Custom field ID
- name (optional): New custom field name
- data_type (optional): New data type
- extra_data (optional): Extra data for the custom field

```typescript
update_custom_field({
  id: 1,
  name: "Updated Invoice Number",
  data_type: "string"
})
```

#### delete_custom_field
Delete a custom field.

Parameters:
- id: Custom field ID

```typescript
delete_custom_field({
  id: 1
})
```

#### bulk_edit_custom_fields
Perform bulk operations on multiple custom fields.

Parameters:
- custom_fields: Array of custom field IDs
- operation: One of "delete"

```typescript
bulk_edit_custom_fields({
  custom_fields: [1, 2, 3],
  operation: "delete"
})
```

### Mail Operations

Tools for managing Paperless mail accounts and the mail rules that drive
automatic email ingestion. Account passwords/tokens are never exposed: they are
redacted from every tool response.

#### list_mail_accounts
List mail accounts so you can pick the account ID needed when creating a mail
rule. Passwords are redacted.

Parameters:
- page (optional): Page number
- page_size (optional): Number of results per page

```typescript
list_mail_accounts()
```

#### get_mail_account
Get a single mail account by ID. Password/token fields are redacted.

Parameters:
- id: Mail account ID

```typescript
get_mail_account({
  id: 1
})
```

#### process_mail_account
Manually trigger Paperless mail processing for one account. This can consume
matching mails according to the account's enabled mail rules.

Parameters:
- id: Mail account ID

```typescript
process_mail_account({
  id: 1
})
```

#### list_mail_rules
List mail rules with optional pagination.

Parameters:
- page (optional): Page number
- page_size (optional): Number of results per page

```typescript
list_mail_rules()
```

#### get_mail_rule
Get a single mail rule by ID.

Parameters:
- id: Mail rule ID

```typescript
get_mail_rule({
  id: 1
})
```

#### create_mail_rule
Create a mail rule. Use `list_mail_accounts` first to choose the account.

Required parameters:
- name: Rule name
- account: Mail account ID
- folder: IMAP folder to scan (e.g. "INBOX")

Common optional parameters:
- enabled (default true): Whether the rule is active
- filter_from / filter_to / filter_subject / filter_body: Match incoming mail
- maximum_age: Only process mail newer than this many days
- action: 1=Delete, 2=Move to folder, 3=Mark as read, 4=Flag, 5=Tag
- action_parameter: Target folder/tag for the chosen action
- assign_title_from: 1=Subject, 2=Attachment filename, 3=Do not assign
- assign_tags / assign_correspondent / assign_document_type: Metadata to apply
- assign_correspondent_from: 1=None, 2=Mail address, 3=Sender name, 4=Use assign_correspondent
- attachment_type: 1=Attachments only, 2=All files incl. inline
- consumption_scope: 1=Attachments only, 2=Full mail as .eml, 3=Both
- pdf_layout: 0=System default, 1=Text+HTML, 2=HTML+text, 3=HTML only, 4=Text only

```typescript
create_mail_rule({
  name: "Invoices",
  account: 1,
  folder: "INBOX",
  filter_subject: "invoice",
  action: 3,
  attachment_type: 1
})
```

#### update_mail_rule
Patch an existing mail rule. Only the fields you supply are changed.

Parameters:
- id: Mail rule ID
- ...any of the `create_mail_rule` fields to update

```typescript
update_mail_rule({
  id: 1,
  enabled: false
})
```

#### delete_mail_rule
Delete a mail rule. Requires an explicit confirmation flag. This changes future
mail ingestion behavior but does not delete any existing documents.

Parameters:
- id: Mail rule ID
- confirm: Must be `true` to confirm deletion

```typescript
delete_mail_rule({
  id: 1,
  confirm: true
})
```

### Semantic Search

Three optional tools that search the corpus by **meaning** as well as by words, using a vector index fused with the corpus' own lexical (BM25) search. They register only when at least one of the search variables below is set — the nine marked *Search* in the Environment Variables table, not `PAPERLESS_MCP_MAX_INLINE_CHARS` or `PAPERLESS_MCP_UPLOAD_PATHS`. With none of the nine set, the server behaves exactly as before.

Any OpenAI-compatible `/embeddings` endpoint works: a local server such as Ollama, or a hosted provider with an API key. Both reply envelopes are accepted without configuration. If indexed text must not leave your network, set `PAPERLESS_MCP_EMBEDDING_LOCAL_ONLY=true` — the endpoint is then required to be a literal loopback or RFC 1918 address and a DNS name is refused. It is off by default so an install pointing at a hosted provider is not fighting a policy it never chose.

#### semantic_search
Search by meaning and by words together, returning bounded passages in the same shape as `search_passages`.

Parameters:
- query: What to search for, in natural language
- limit (optional): Maximum passages to return (default 10, max 50)
- window (optional): Characters per passage (default 800, range 200–4000)
- correspondent, document_type, tag, storage_path, created__date__gte, created__date__lte (optional): Same filters as `search_passages`, and every one of them is honoured. A document with no filing location cannot match a numeric `storage_path` — asking for a storage path means asking for a document that has one.

```typescript
semantic_search({ query: "what did the inspector say about the roof" })
// → {
//     query, semantic: "ok", strong_match: true, total_hits: 4,
//     passages: [{ id, title, score, matched, highlights, passage, offset, content_length, offset_space }, ...]
//   }
```

**Return shape.** `passages[]` mirrors `search_passages` — `id`, `title`, `score`, `passage`, `offset`, `content_length` — so a client that has learned one has learned the other. `matched` is added, saying whether the hit came from meaning, words, or both.

**Where the window anchors.** On the **longest matching query term**, which is the cheap correct proxy for the most *distinctive* one. It used to anchor on the first term in query order, and English queries open with articles — so `"the practice of a wrathful deity with a lion head"` anchored on `"the"` at character 3 and the window landed at the document head. Longest-match fixes that. If no query term appears in the document at all, the window falls back to offset 0 and reports no anchoring term; that is a deliberate, honest head window rather than a fabricated match.

⚠️ **`offset` is in the RECONSTRUCTED text space, not the raw file.** The passage is rebuilt by joining the document's chunks, so chunk *p* begins at `(p−1)×1800`. `offset` and `content_length` address that reconstruction. **Each passage carries `offset_space`** — `"normalized"` on this path, `"document"` when it fell back to the API's own content — so a caller can tell which space it is in without assuming. Do not feed these offsets to a raw-file byte reader.

**`matched`** is `"meaning"`, `"words"`, or `"both"`. It tells you whether to trust the hit or re-query lexically: `"words"` means the hit came from the lexical side and the vector side did not find it, so **do not read a words-only result as absence of a paraphrase** that is worded differently.

**`total_hits`** is the number of distinct documents the ranking returned **before** your filter
arguments were applied, so it can exceed `passages.length`. The difference is not left unexplained:

- **`filtered_out`** — how many candidates your filter arguments dropped. When non-zero, `filtered_out_reasons` says which filter did it: `metadata_filter` (a filter you passed) or `document_unavailable` (the document existed in the ranking but could no longer be fetched).
- **`retrievers`** — which halves of the search actually ran, as `"vector"` and/or `"lexical"`. An empty `passages` with both names present is a real result: both engines ran and neither found anything. An empty `passages` with only one name is a partial answer, and `semantic` plus `degraded_reason` say why.

**`trimmed`** is present **only when the response was too large to return whole**, and it says exactly what was cut rather than leaving the caller to guess:

```typescript
trimmed?: {
  reason: "result_size";      // the only reason today
  dropped: number;            // how many passages were removed
  limit_chars: 20000;         // the ceiling that forced it
  query_truncated?: boolean;  // true only when the ECHOED query was cut too
}
```

⚠️ **The bound covers the whole serialised result, not just the passages.** A pathologically long `query` is echoed back, so it is also truncated when necessary — that is what `query_truncated` announces. **A trimmed response is a real answer, not a failure**: the passages you got are the highest-ranked ones, and `dropped` tells you how many you did not.

⚠️ **Note the gap this leaves.** Filtering happens after ranking, so a filtered query can return fewer
passages than `limit` even when more documents matched the filter. `filtered_out` is how you see that
happened rather than assuming the corpus is thin.

**`strong_match`** is `true` exactly when at least one passage was returned. It is a **coarse "did anything come back" flag, not a quality judgement.**

⚠️ **It used to consult a 0.45 similarity floor, and that floor could not fire on this corpus.** Measured: best raw similarities run **0.73–0.79**, so 0.45 sits *below the worst irrelevant query* — the old field reported `true` on every call that produced a hit, regardless of quality. Rather than keep a field that cannot be `false`, it now says what it actually means.

Judge result quality from **`matched`** (meaning / words / both), **`retrievers`** (which halves ran), and the passages themselves. Do not write `if (!strong_match) requery()` — that branch is unreachable.

**Degradation contract.** The embedding server being down is an expected condition, not an error, so `semantic_search` never fails because of it. It returns the lexical results it can still produce, with `semantic: "unavailable"` and a `degraded_reason` naming the cause:

```typescript
// → { semantic: "unavailable", degraded_reason: "connect ECONNREFUSED 127.0.0.1:11434",
//     strong_match: true, total_hits: 1, passages: [...] }
```

**A degraded call never returns a silent empty list.** `passages: []` alone cannot distinguish "nothing matched" from "half the search engine was missing", so the state is carried explicitly. Three outcomes, and `retrievers` is what tells them apart:

| `semantic` | `retrievers` | Meaning |
|---|---|---|
| `"ok"` | both | Both engines ran. An empty list here **is** a real answer — nothing matched. |
| `"unavailable"` | `["lexical"]` | The embedder was unreachable; `degraded_reason` names the cause. Results are word-matches only, so a paraphrase will be missed. |
| absent | `[]` | Neither engine could run. The tool returns an **error** rather than an empty list that would read as "no such document". |

⚠️ `semantic_search` requires the index to have been built first — see `sync_embeddings` and [Semantic search setup](#semantic-search-setup).

#### embedding_status
Report what the index actually holds. No parameters.

```typescript
embedding_status()
// → {
//     documents: 1240, chunks: 9678, dimensions: 768,
//     model: "nomic-embed-text", prefix_mode: "nomic",
//     last_sync: "2026-11-15T12:04:11.000Z",
//     pending: 3, store_bytes: 148373504, truncated_documents: 2,
//     index_path: "/home/you/.local/share/paperless-mcp/index/index.db"
//   }
```

- `documents` / `chunks` — what is indexed now
- `dimensions`, `model`, `prefix_mode` — what the index was **built** with; a mismatch against the running configuration is refused at query time
- `last_sync` — the most recent metadata stamp written, or `null`
- `pending` — remote document count minus indexed count. **`null` means the count could not be read**, which is not the same as `0`
- `store_bytes` — size of `index.db` on disk, or `null` if the file does not exist yet
- `truncated_documents` — documents that hit `PAPERLESS_MCP_MAX_CHUNKS` and lost their tail. **On a truncated document "not found" may be a false negative**, which is why this is reported
- `open_error` — present only when the index could not be opened, and then the counts above are zeros

Use this to tell "the index is empty" apart from "nothing matched".

#### sync_embeddings
Bring the index up to date with the corpus. Long-running on a first build.

Parameters:
- force (optional): Re-embed every document even if `modified` is unchanged. Only needed after a chunker or model change; a normal sync is incremental.

```typescript
sync_embeddings({})
// → { added: 12, updated: 3, deleted: 0, metadataRefreshed: 4, failed: 0, deletedSkipped: false }
```

A normal sync fetches every document whose `modified` changed since the last run, re-chunks and re-embeds it, removes documents that no longer exist, and rebuilds metadata (tags, correspondent, document type) for documents whose metadata age exceeds 24 hours without the document itself being edited.

- `failed` counts documents that errored individually and were skipped; `embedding_status` will still show them as pending
- `deletedSkipped: true` means the remote list changed mid-run (row count disagreed with the reported count), so the delete step was skipped rather than risk removing live documents

**An embedding outage aborts the run.** It stops rather than writing an index that is missing documents while claiming to be complete. Run `embedding_status` afterwards to confirm.

⚠️ **`force: true` discards the existing vectors before it rebuilds them, and it proves the embedder is alive first.** The probe is a real embedding of a short constant string, not a health GET — a server that answers GET without its model pulled fails on the first real request, so a GET would clear the index and then discover the problem. If the probe fails, the call refuses and **your existing index is untouched**. Only after a successful probe are the document rows deleted, in the same transaction as their lexical rows. An interrupted rebuild still leaves you with no fallback, so confirm with `embedding_status` afterwards.

## Semantic search setup

Semantic search is entirely optional and off until you configure it.

1. **Point it at an embedding server.** The default is `http://127.0.0.1:11434` with `nomic-embed-text`, which is Ollama's out-of-the-box shape. Set `PAPERLESS_MCP_EMBEDDING_URL` to any OpenAI-compatible `/embeddings` endpoint — a local server or a hosted provider — and `PAPERLESS_MCP_EMBEDDING_API_KEY` if that provider needs a bearer token. A base URL with no path is resolved to `/api/embed` (Ollama) and any other path is used as given; a URL already ending in `/embeddings` is used verbatim.
2. **Set at least one `PAPERLESS_MCP_*` search variable** (see the Environment Variables table). Any one of them enables the tools; every value you set is validated strictly, and an invalid value refuses loudly rather than silently defaulting.
3. **Build the index by calling `sync_embeddings`.** Nothing is built at startup — a first build only happens when you ask for it. Set `PAPERLESS_MCP_SYNC_INTERVAL_MS` to a positive value if you want incremental syncs on a timer afterwards; the first run is immediate, and the timer is cleared when the server closes. On a large corpus the first build takes tens of minutes; call `embedding_status` to watch `documents` and `chunks` climb.
4. **Search.** `semantic_search` fuses the vector index with the corpus' lexical search.

The index lives at `~/.local/share/paperless-mcp/index/index.db` by default, or wherever `PAPERLESS_MCP_INDEX_DIR` points. It is deliberately **outside** your Paperless-ngx container: it is derived data, it is regenerable by re-running `sync_embeddings`, and putting it inside the container would drag it into that container's backup scope.

**Keeping text on your own network.** If indexed content must not leave the host, set `PAPERLESS_MCP_EMBEDDING_LOCAL_ONLY=true`. The endpoint is then required to be a literal loopback or RFC 1918 address, and a DNS name is refused — a name can resolve to anywhere. By default this restriction is off, because the package is published for general use and a third-party install must be able to point it at whichever provider it has.

## Keeping the index current without polling

An incremental sync compares a recorded watermark against the live corpus, so it can tell you
"nothing changed" in a single request. That works for additions, edits and deletions — but it is
**structurally blind to one case**: a tag, correspondent or document-type edit is applied through a
bulk update that never touches a document's `modified` timestamp, and never changes the document
count. Both watermark scalars therefore read "nothing happened" while the affected documents'
metadata is genuinely stale in the index.

The only signal that exists for that change is the event. So the server can optionally listen for
Paperless-ngx's own webhooks and re-index the documents a delivery names.

### Turning it on

Set a port and a shared secret:

```bash
PAPERLESS_MCP_WEBHOOK_PORT=8899
PAPERLESS_MCP_WEBHOOK_SECRET=<a-long-random-string>
```

The receiver is **off by default** (`PAPERLESS_MCP_WEBHOOK_PORT=0`) and **refuses to start if the
secret is unset**, logging the reason instead. That refusal is deliberate: the endpoint causes
embedding work, and an unauthenticated trigger for it is a resource-abuse vector, so a warning line
nobody reads would not be a mitigation.

It is not a replacement for `sync_embeddings` or the background interval. A webhook only reports
events the process was alive to receive; a document added while the server was down produces no
delivery at all. The watermark still covers downtime, and the periodic metadata refresh still runs
as a backstop for a delivery that was attempted and never arrived (the sender retries a few times
and then gives up).

### Creating the workflow in Paperless-ngx

Paperless-ngx fires webhooks through a *workflow*, which you create once. Nothing here creates it
for you — it is your instance to configure.

```bash
curl -X POST "$PAPERLESS_URL/api/workflows/" \
  -H "Authorization: Token $PAPERLESS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Keep the semantic index current",
    "enabled": true,
    "triggers": [
      { "type": 2, "matching_algorithm": 0 },
      { "type": 3, "matching_algorithm": 0 }
    ],
    "actions": [
      {
        "type": 4,
        "webhook": {
          "url": "http://172.17.0.1:8899/webhook/paperless",
          "use_params": true,
          "as_json": true,
          "params": { "id": "{{ doc_id }}" },
          "headers": { "X-Paperless-MCP-Secret": "<a-long-random-string>" }
        }
      }
    ]
  }'
```

Notes on that payload, each of which is a way to get it subtly wrong:

- **`type: 2` is "Document Added" and `type: 3` is "Document Updated"** on the trigger; **`type: 4`
  is "Webhook"** on the action. The trigger types are a separate numbering from the action types.
- **`"matching_algorithm": 0`** means "no content filter", so the trigger matches every document. A
  non-zero value makes the trigger filter on document text, which is not what you want here.
- The template uses **Jinja syntax** (`{{ doc_id }}`), not a different placeholder style, and is
  rendered before the request is sent. `use_params` with a `params` dictionary posts a form; add
  `"as_json": true` to post JSON instead. Both are parsed.
- **The document id must be in the body.** A delivery that names no document still triggers a sync,
  but an untargeted one — correct, merely more expensive. The receiver cannot infer which document
  changed, because the payload carries no change description.
- **The secret goes in `headers`.** The receiver also accepts it as a `Bearer` token if you would
  rather not add a custom header.

### The container networking reality

**A Paperless-ngx running in a container cannot reach `127.0.0.1` on your host.** That address is
the *container's* own loopback. If you leave `PAPERLESS_MCP_WEBHOOK_HOST` at its default and point
the workflow at `127.0.0.1`, deliveries will fail with a connection error.

Two things have to be true instead:

1. **The receiver binds an address the container can route to.** In a default Docker bridge setup
   the host is reachable from containers at the bridge gateway, commonly `172.17.0.1`. Set
   `PAPERLESS_MCP_WEBHOOK_HOST` to that (or `0.0.0.0` to listen on all interfaces) and use the same
   address in the workflow URL. If your containers run on a user-defined network, the gateway is a
   different address in that subnet; `docker network inspect <network>` shows it.
2. **Your host firewall permits the traffic.** Most default-DROP firewall configurations will
   silently swallow a connection from the bridge subnet, which looks exactly like the receiver not
   running. Permit the port from the bridge subnet specifically, not from everywhere.

Binding `0.0.0.0` widens the endpoint to every interface the host has. That is sometimes the only
workable choice, and it is why the default is loopback: it should be a decision, not an inheritance.

### How a delivery is handled

- The endpoint answers **`202 Accepted` before doing any work.** The sender times out after a few
  seconds and retries on failure, so acknowledging first is what turns those retries into harmless
  duplicates instead of repeated embedding runs.
- Deliveries are **coalesced**: ids are collected and one sync runs after a quiet period. A bulk
  edit that produces hundreds of deliveries produces one run naming every affected document.
- A delivery for a document that no longer exists is **counted as a failure and skipped**; the rest
  of the batch proceeds. A delivery can be in flight when its document is deleted, and that is not a
  reason to abandon the others.
- **Deletions produce no delivery.** Paperless-ngx's trigger set has no delete type, so the
  document count in the watermark remains the only thing that notices one. This is stated plainly
  rather than worked around: there is no event to listen for.

### Verifying it

Send one delivery yourself — the receiver accepts anything that presents the secret:

```bash
curl -i -X POST "http://127.0.0.1:8899/webhook/paperless" \
  -H "X-Paperless-MCP-Secret: <a-long-random-string>" \
  -H "Content-Type: application/json" \
  -d '{"id": 1}'
```

A `202` means it was accepted; the sync then runs after the debounce period, and its log line
reports what it found. A `401` means the secret did not match, and a `413` means the body was over
the 1 MB cap.

## Error Handling

The server will show clear error messages if:
- The Paperless-NGX URL or API token is incorrect
- The Paperless-NGX server is unreachable
- The requested operation fails
- The provided parameters are invalid

## Testing

### Unit tests

Run the unit test suite (no external dependencies required):

```bash
npm test
```

### E2E tests

The E2E suite boots an empty Paperless-ngx instance, runs the compiled MCP server, and drives a deterministic serial scenario through `tools/call` requests — creating a tag, correspondent, and document type, uploading a PDF, then exercising list / get / search / download / thumbnail / bulk-edit on the same document. No LLM and no Paperless REST client outside MCP.

**Prerequisites:** Docker, Docker Compose, and `jq`.

```bash
# 1. Build the MCP server
npm run build

# 2. Start Paperless-ngx
docker compose -f docker-compose.e2e.yml up -d

# 3. Wait for Paperless to be ready, then get a token
TOKEN=$(curl -s -X POST http://localhost:8000/api/token/ \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin123"}' | jq -r '.token')

# 4. Start the MCP server
node build/index.js --http --port 3001 \
  --baseUrl http://localhost:8000 --token "$TOKEN" &
MCP_PID=$!

# 5. Run the E2E tests
MCP_URL=http://localhost:3001/mcp \
PAPERLESS_URL=http://localhost:8000 \
PAPERLESS_TOKEN="$TOKEN" \
npm run test:e2e

# 6. Cleanup
kill "$MCP_PID"
docker compose -f docker-compose.e2e.yml down -v
```

E2E tests also run automatically in CI on every pull request and push to `main`, covering both the `build/index.js` CLI and the published Docker image.

## Development

Want to contribute or modify the server? Here's what you need to know:

1. Clone the repository
2. Install dependencies:
```bash
npm install
```

3. Make your changes in `src/`
4. Build and test locally:
```bash
npm run build
node build/index.js http://localhost:8000 your-test-token
```

Run the test suite with `npm test`.

The server is built with:
- [@modelcontextprotocol/sdk](https://github.com/modelcontextprotocol/typescript-sdk): the official TypeScript SDK for MCP servers
- [zod](https://github.com/colinhacks/zod): TypeScript-first schema validation

## API Documentation

This MCP server implements endpoints from the Paperless-NGX REST API. For more details about the underlying API, see the [official documentation](https://docs.paperless-ngx.com/api/).

## Running the MCP Server

The MCP server can be run in two modes:

### 1. stdio (default)

This is the default mode. The server communicates over stdio, suitable for CLI and direct integrations.

```
npm run start -- <baseUrl> <token>
```

### 2. HTTP (Streamable HTTP Transport)

To run the server as an HTTP service, use the `--http` flag. You can also specify the port with `--port` (default: 3000). This mode requires [Express](https://expressjs.com/) to be installed (it is included as a dependency).

```
npm run start -- <baseUrl> <token> --http --port 3000
```

- The MCP API will be available at `POST /mcp` on the specified port.
- Each request is handled statelessly, following the [StreamableHTTPServerTransport](https://github.com/modelcontextprotocol/typescript-sdk) pattern.
- GET and DELETE requests to `/mcp` will return 405 Method Not Allowed.

#### Per-request API token (HTTP/Docker mode)

In HTTP mode, clients authenticate by supplying a Paperless-NGX API token via the standard `Authorization` header:

```
Authorization: Bearer <paperless-ngx-api-token>
```

The token is passed straight through to Paperless-NGX, so each client's own Paperless permissions are enforced end-to-end. This lets a single server instance serve multiple users, each with their own token. The same behaviour applies to both `/mcp` and `/sse` endpoints.

> **⚠️ Breaking change in v2.0.0 — HTTP mode is now authenticated by default.**
>
> Previously, a request with no `Authorization` header silently fell back to the server-configured `PAPERLESS_API_KEY`, which left the HTTP endpoint open to anyone who could reach the port. As of v2.0.0, requests without a `Bearer` token are rejected with `401 Unauthorized`. The server token is **never** used for unauthenticated requests unless you explicitly opt in with `--no-auth`.

| Scenario | `--no-auth` off (default) | `--no-auth` on |
|---|---|---|
| Client sends `Authorization: Bearer <tok>` | `<tok>` (client-supplied) | `<tok>` (client-supplied) |
| No header, `PAPERLESS_API_KEY` / `--token` set | `401 Unauthorized` | server token |
| No header, no server token | `401 Unauthorized` | `401 Unauthorized` |

**Migrating from v1.x:** if you relied on the old fallback (a single shared `PAPERLESS_API_KEY` with clients that don't send a token), you have two options:

1. **Recommended:** have each client send `Authorization: Bearer <paperless-token>`.
2. **Restore the old behaviour** (trusted/local networks only): start the server with the `--no-auth` flag, e.g. append it to the Docker `command`/args or your CLI invocation. This requires a server token (`PAPERLESS_API_KEY` or `--token`) to be configured.

<details>
<summary>Docker Deployment</summary>

The MCP server can be deployed using Docker and Docker Compose. The Docker image automatically runs in HTTP mode with SSE (Server-Sent Events) support on port 3000.

### Docker Compose Configuration

Create a `docker-compose.yml` file:

```yaml
services:
  paperless-mcp:
    container_name: paperless-mcp
    image: ghcr.io/retiform-labs/paperless-mcp:latest
    environment:
      - PAPERLESS_URL=http://your-paperless-ngx-server:8000
      - PAPERLESS_API_KEY=your-paperless-api-key
      - PAPERLESS_PUBLIC_URL=https://paperless-ngx.yourpublicurl.com
    ports:
      - "3000:3000"
    restart: unless-stopped
```

Then run:
```bash
docker-compose up -d
```

### Using with Continue VS Code Extension

If you're using the [Continue VS Code extension](https://continue.dev/), you can configure it to use the Dockerized MCP server via SSE.

Create or edit `.continue/mcpServers/paperless-mcp.yaml` at your workspace root:

```yaml
name: Paperless
version: 0.0.1
schema: v1
mcpServers:
  - name: Paperless
    type: sse
    url: http://localhost:3000/sse
```

**Notes:**
- Replace `localhost` with your Docker host's IP address or hostname if running on a remote server
- The Docker container handles authentication via environment variables, so no credentials are needed in the Continue config
- The SSE endpoint is available at `/sse` on the configured port (default: 3000)

</details>

# Credits

This project is a fork of [nloui/paperless-mcp](https://github.com/nloui/paperless-mcp). Many thanks to the original author for their work. Contributions and improvements may be returned upstream.

## Debugging

To debug the MCP server in VS Code, use the following launch configuration:

```json
{
    "type": "node",
    "request": "launch",
    "name": "Debug Paperless MCP (HTTP, ts-node ESM)",
    "program": "${workspaceFolder}/node_modules/ts-node/dist/bin.js",
    "args": [
        "--esm",
        "src/index.ts",
        "--http",
        "--baseUrl",
        "http://your-paperless-instance:8000",
        "--token",
        "your-api-token",
        "--port",
        "3002"
    ],
    "env": {
        "NODE_OPTIONS": "--loader ts-node/esm",
    },
    "console": "integratedTerminal",
    "skipFiles": [
        "<node_internals>/**"
    ]
}
```

**Important:** Before debugging, uncomment the following line in `src/index.ts` (around line 175):

```typescript
// await new Promise((resolve) => setTimeout(resolve, 1000000));
```

This prevents the server from exiting immediately and allows you to set breakpoints and debug the code.
