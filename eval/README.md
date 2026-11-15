# eval/ — search acceptance harness

Measures recall of the three retrieval arms against the live index, **per arm and
per depth**, and refuses to score when a case's premise is false.

```bash
set -a && . /REDACTED/.paperless-index-build.env && set +a
npm run eval
# or: node eval/run.mjs
```

Exit status: `0` all cases scored · `1` a precondition failed, or the control was
not retrieved. **Both non-zero paths mean "do not read the numbers".**

## Why it refuses to score

An earlier harness scored 10/20 against a ≥16/20 gate, and at least one of those
misses was **bad ground truth, not a retrieval failure**. It queried:

> "a wrathful deity practice with a **horse head**" → target doc 2681

The target is *The Practice of **Sinhamukha***. `sinha` means **lion**. Measured
in the corpus: `sinhamukha=49` occurrences, `horse=0`, `lion=0`. The engine was
never asked the question the score assumed it failed.

So every case carries `anchors`: terms verified present in the target by reading
every chunk of it. `run.mjs` re-verifies them **in the same run** and exits 1 if
any is absent:

```
REFUSING TO SCORE -- fixture premises are false:
  PRECONDITION FAILED: doc 2681 lacks "horse"
```

A miss against a false premise measures the fixture, not the engine.

## What k means

**k is DISTINCT DOCUMENTS RETURNED, not kNN candidates.**

`store.vectorHits()` returns each document's single best chunk and widens its
candidate pool until it holds `limit` distinct documents or reaches
`VECTOR_CANDIDATE_CEILING` (4096). `store.textHits()` likewise keeps only each
document's best chunk. So both arms are ranked in the same unit, and `@10` means
"the target was one of the ten documents returned".

A hit is counted **only if the target's rank is ≤ k**. This matters: a fused arm
built from arms fetched at `k*4` returns up to `4k` entries, so counting
"appeared anywhere in the list" scored "anywhere in 80" as a hit@10 — measured,
that inflated fused hit@10 from 15 to 19 before it was fixed.

## The arms

| arm | how |
|---|---|
| `vector-only` | `POST :11434/api/embeddings` with the mandatory `search_query: ` prefix → `vec_chunks` kNN via `store.vectorHits` |
| `lexical-only` | FTS5 BM25 via `store.textHits` |
| `fused (shipped)` | RRF k=60 over arms fetched at `limit*4` — the shape `runSemanticSearch` actually uses |
| `fused (deep)` | RRF k=60 over both arms at 200 documents, to separate "fusion is weak" from "the pool was shallow" |

Measurement is against the **store**, not the tool, so HTTP and passage-assembly
overhead cannot contaminate the retrieval number. The full tool path is a
different question and is not measured here.

## The control case

Doc 2573's query shares verbatim vocabulary with its target, so a lexical win
there is **expected and proves nothing about embeddings**. It is marked
`control: true` and excluded from the non-control summary. If the control is not
retrieved, the harness itself is broken and exits 1 — because at that point a low
score means "my pipeline is wrong", not "retrieval is weak".

## Query hygiene

Queries are **paraphrases in words the document actually contains**. Do not
replace a query with the document title: that tests BM25 term overlap and tells
you nothing about the vector arm. If a query names a term the target does not
contain, **rewrite the query** — do not delete the case, and do not lower the
anchor to make it pass.

## Known corpus limitation

Several dharma documents are **OCR-degraded scans**, and their low recall must be
read against that before it is read as an engine result. Doc 2681 measured:
11 chunks / 19,324 chars, opening `CHOGYALNAMKHAINORBU The Practice of SINHAMUKHA
• ·¥~~ Shang Shung Edidoni`, and throughout `wr.uhful`, `h1 general`, `!obseve1`,
`posSibility`, `daktntSinhatmtkha`. Doc 2698 (`c� 1r c:::=:J`) and doc 2699
(`TI-IE PR_ACTICE FOR THE NAG.A`) are the same class.

## Environment

| var | default |
|---|---|
| `PAPERLESS_MCP_INDEX_DB` | `$PAPERLESS_MCP_INDEX_DIR/index.db` |
| `PAPERLESS_MCP_INDEX_DIR` | `~/.local/share/paperless-mcp/index` |
| `PAPERLESS_MCP_EMBEDDING_URL` | `http://127.0.0.1:11434` |
| `PAPERLESS_MCP_EMBEDDING_MODEL` | `nomic-embed-text` |

The index is opened **readonly** — a measurement pass must never write to a live
1.9 GB index.
