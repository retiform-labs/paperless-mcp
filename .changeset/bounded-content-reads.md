---
"@baruchiro/paperless-mcp": minor
---

Bound document content responses: `get_document_content` now returns bounded windows (default 8000 chars) with `offset`/`max_chars` pagination and honest metadata (`content_length`, `truncated`, `has_more`), requests above the new `PAPERLESS_MCP_MAX_INLINE_CHARS` budget (default 50000) fail loudly instead of returning the whole document, and a new `search_passages` tool returns bounded passages around full-text hits with score/highlight/offset metadata. List/query/get results now carry `content_length`.
