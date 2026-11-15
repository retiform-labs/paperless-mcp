---
"@retiform/paperless-mcp": minor
---

Add local semantic search: `semantic_search`, `embedding_status`, and `sync_embeddings`.

`semantic_search` searches the corpus by meaning as well as by words, fusing a
local vector index with the lexical BM25 search the fork already had. Its
`passages[]` mirrors `search_passages` — `id`, `title`, `score`, `passage`,
`offset`, `content_length` — so a client that has learned one has learned the
other. It adds `matched` (`"meaning"` / `"words"` / `"both"`), which says whether
a hit came from the vector side, the lexical side, or both; a `"words"` hit is a
lexical match the vector side did not find, so it should not be read as absence
of a paraphrase.

`embedding_status` reports what the index actually holds — document and chunk
counts, the dimensions/model/prefix mode it was built with, the last sync stamp,
how many documents are pending, index size on disk, and how many documents were
truncated at the chunk cap. A truncated document can produce a false negative on
"not found", which is why it is reported rather than inferred.

`sync_embeddings` builds and updates the index. A normal run is incremental
(driven by each document's `modified` stamp, plus a 24-hour metadata refresh for
tags/correspondent/document-type changes that do not touch `modified`); `force:
true` re-embeds everything.

**The degradation contract.** An unavailable embedding server is an expected
condition, not an error. Search returns the lexical results it can still produce
with `semantic: "unavailable"` and a `degraded_reason` naming the cause, and
never a silent empty list — an empty result and "half the engine was missing" are
different facts that `passages: []` cannot distinguish. An empty list with
`semantic: "ok"` is a genuine answer. Sync is the opposite: it aborts rather than
writing an index that is missing documents while claiming to be complete.

The whole feature is optional and off until at least one `PAPERLESS_MCP_*` search
variable is set; with none set the server registers nothing new and behaves
exactly as before. Embedding is local-only by design: the endpoint must be a
literal loopback or RFC 1918 address, a DNS name is refused, and there is no
credential field or remote-embedding code path. The index is a SQLite file
outside the Paperless container, regenerable by re-running `sync_embeddings`.

Known limitation in this release: `storage_path` is accepted in
`semantic_search`'s argument schema but is not applied — the lightweight document
shape this path fetches does not carry it. `search_passages` should be used when
that filter matters.

Both dependencies are new: `better-sqlite3` and `sqlite-vec`.
