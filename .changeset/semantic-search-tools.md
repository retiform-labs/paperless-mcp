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

`score` is a **reciprocal-rank-fusion sum, not a similarity** — `Σ 1/(k + bestRank)`
across the arms that returned the document. With two retrievers the maximum is
`2/(k+1)`, so the ceiling is a property of the fusion constant rather than of the
engine, and it is **not** a 0–1 similarity scale. Read rank order, not magnitude.
Both the package README and the tool's KDoc say so, because a bare `score: 0.18`
with no explanation is worse than no score at all.

### Bounded results

`semantic_search` never returns whole documents and never floods a caller. Each
passage is a window (`window`, default 800, range 200–4000) around the hit, and
the **total serialised result** is capped at 20,000 characters. A result that
would exceed that is cut and says exactly what was cut:

```typescript
trimmed?: { reason: "result_size"; dropped: number; limit_chars: 20000; query_truncated?: boolean }
```

The bound covers the echoed `query` as well as the passages (`query_truncated`
announces the former), because a caller echoing a very long query was otherwise
the way this overflowed. A trimmed response is a real answer — the passages it
kept are the highest-ranked ones.

### How the passage window is anchored

Each window is placed on the **rarest matching query term**, measured as that
term's chunk count across the index and computed once per request, with term
length only as a tiebreak. Two earlier heuristics were wrong: anchoring on the
first term in query order landed on "the" at character 3 for English queries, and
anchoring on the longest term then picked a common long word over a short
distinctive one (`doctor`, 1,903 chunks, over `ankle`, 808). Rarity matched a
human read in 3 of 3 probes where length matched 0 of 3.

⚠️ **`offset` addresses the RECONSTRUCTED text, not the raw file** — passages are
rebuilt by joining the document's chunks. Each passage carries `offset_space`
(`"normalized"` on this path, `"document"` when it fell back to the API's own
content) so a caller can tell which space it is in without assuming.

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
