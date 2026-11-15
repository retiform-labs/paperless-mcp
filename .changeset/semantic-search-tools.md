---
"@retiform/paperless-mcp": minor
---

Add semantic search: `semantic_search`, `embedding_status`, and `sync_embeddings`.

`semantic_search` searches the corpus by meaning as well as by words, fusing a
vector index with the lexical BM25 search the fork already had. Its `passages[]`
mirrors `search_passages` — `id`, `title`, `score`, `passage`, `offset`,
`content_length` — so a client that has learned one has learned the other. It
adds `matched` (`"meaning"` / `"words"` / `"both"`), which says whether a hit came
from the vector side, the lexical side, or both; a `"words"` hit is a lexical
match the vector side did not find, so it should not be read as absence of a
paraphrase.

`embedding_status` reports what the index actually holds — document and chunk
counts, the dimensions/model/prefix mode it was built with, the last sync stamp,
how many documents are pending, index size on disk, and how many documents were
truncated at the chunk cap. A truncated document can produce a false negative on
"not found", which is why it is reported rather than inferred.

`sync_embeddings` builds and updates the index. A normal run is incremental
(driven by each document's `modified` stamp, plus a 24-hour metadata refresh for
tags/correspondent/document-type changes that do not touch `modified`);
`force: true` re-embeds everything, after first proving the embedder is alive so
a failed rebuild cannot leave you with no index to fall back on.

## Embedding providers

Any OpenAI-compatible `/embeddings` endpoint works: a local server such as
Ollama, or a hosted provider with an API key. Both reply envelopes are accepted
without configuration — `{embeddings: [[...]]}` (Ollama native) and
`{data: [{embedding: [...]}]}` (OpenAI-compatible) — so there is no setting
telling the client which shape to expect.

- `PAPERLESS_MCP_EMBEDDING_LOCAL_ONLY=true` restricts the endpoint to a literal
  loopback or RFC 1918 address and refuses DNS names. Off by default, so an
  install pointing at a hosted provider does not have to fight a policy it never
  chose. Set it if indexed text must not leave your network.
- `PAPERLESS_MCP_EMBEDDING_API_KEY` is sent as `Authorization: Bearer <key>` and
  is never logged or included in a returned object.

## Background sync

`PAPERLESS_MCP_SYNC_INTERVAL_MS` starts an incremental sync timer when the tools
register: one run immediately, then once per interval, cleared on close. `0`
remains the default and means no timer. `PAPERLESS_MCP_SYNC_BATCH_SIZE` now
controls the number of texts per embedding request; both settings were previously
read and range-checked with nothing consuming them.

## Degradation contract

An unavailable embedding server is an expected condition, not an error. Search
returns the lexical results it can still produce with `semantic: "unavailable"`
and a `degraded_reason` naming the cause, and never a silent empty list — an empty
result and "half the engine was missing" are different facts that `passages: []`
cannot distinguish. `retrievers` lists which halves ran, and `filtered_out` says
how many candidates your filters dropped. An empty list with `semantic: "ok"` is a
genuine answer. Sync is the opposite: it aborts rather than writing an index that
is missing documents while claiming to be complete.

The whole feature is optional and off until at least one `PAPERLESS_MCP_*` search
variable is set; with none set the server registers nothing new and behaves
exactly as before. The index is a SQLite file outside the Paperless container,
regenerable by re-running `sync_embeddings`.

Both dependencies are new: `better-sqlite3` and `sqlite-vec`.
