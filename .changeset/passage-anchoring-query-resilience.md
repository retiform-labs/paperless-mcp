---
"@baruchiro/paperless-mcp": patch
---

Anchor `search_passages` windows on the server's own highlight (falling back to verbatim terms, then the document head) so returned passages contain the actual hit even when stemming matched a word variant absent from the stored text. Full-text queries rejected by the server with an HTTP 400 are retried once with a defensively sanitized query (apostrophes and stray quotes removed; filter values untouched); such responses carry `query_sanitized: true` and persistent failures name every attempted query. The npm `test` script glob is quoted so nested `src/tools/utils` suites actually run.
