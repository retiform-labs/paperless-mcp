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

## This is a manual acceptance gate, NOT a CI step

**`npm run eval` cannot run in GitHub Actions, and it is deliberately not wired
into `.github/workflows/`.** It needs two things CI does not have:

1. **The index** — a `index.db` built offline against the live document fleet
   (measured 2,023,505,920 B ≈ 1.9 GB, 2,476 documents / 311,615 chunks). CI
   checks out a repo; it has no corpus. `PAPERLESS_URL` and `PAPERLESS_API_KEY`
   point at a real Paperless-ngx instance, and building it is a long embedding
   pass (an hour-scale job — time it yourself rather than trusting a figure
   quoted here). There is no fixture that substitutes for it: the questions this
   harness answers are about *our* corpus's recall, and a toy index would answer
   them about a toy corpus.
2. **Ollama** serving `nomic-embed-text` on `:11434`, because the vector arm
   embeds each query at measurement time.

A gate that cannot run where it is declared teaches people to ignore gates. So
this one declares itself manual, and the CI job gates the thing CI *can* see:
that the build compiles, that its output layout is correct, and that the tests
pass. **Recall is checked by a human running the command below on a machine with
the index, and by comparing against the committed baselines.**

```bash
set -a && . /REDACTED/.paperless-index-build.env && set +a
npm run build          # the harness scores the BUILD, so it must be current
npm run eval
```

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

## What a `score` is NOT (the trap that makes 0.033 look broken)

**The fused `score` is an unnormalised reciprocal-rank-fusion sum, not a
similarity.** For each list a document appears in, it contributes
`1 / (k + rank)` with `k = 60`, using its best rank in that list:

```
score = Σ_arms  1 / (60 + bestRank_in_that_arm)
```

The ceiling is therefore `1/61 + 1/61 = 2/61 ≈ 0.0328`, reached only by a
document ranked **first by both arms**. So:

| value | means |
|---|---|
| `0.0328` | ranked **first by both arms** (`2/61`) — the ceiling |
| `0.0164` | ranked first by exactly one arm (`1/61`) |
| `0.0161` | ranked second by one arm, absent from the other (`1/62`) |
| `0.0091` | ranked 50th by one arm, absent from the other (`1/110`) |
| `0.0080` | ranked 65th by one arm, absent from the other (`1/125`) |

*(Computed, not eyeballed: `1/(60+r)` for r = 1, 2, 50, 65.)*

**`0.033` is not "3% relevant". It is a perfect score on this scale.** Any
threshold a caller invents against this number will be wrong unless it is
expressed in RRF units, which is why the tool no longer presents it as a
relevance percentage.

**`strong_match` is not a quality signal either.** The 0.45 similarity floor it
used to consult sits *below* the worst irrelevant query on this corpus (measured
0.595–0.766 for five unrelated queries), so it could never fire — it was constant
`true` whenever the vector arm ran. It now means exactly
`passages.length > 0`: "a passage was returned". Branch on it if you must; do not
read it as confidence.

A **raw similarity** (`1 − cosine distance`, range 0–1) exists too, on the vector
arm before fusion, but it is not what a fused result's `score` carries.

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

## The staleness gate (verified, with its message)

The harness imports **`../build/search/store.js`** and **`../build/search/rank.js`**
— the compiled artifacts, not the TypeScript source, because the built output is
what a consumer receives. A ts-node import would prove nothing about the shipped
package.

The cost of that choice is that a stale `build/` silently measures code nobody
will run. So `checkBuildFreshness()` compares the mtime of each built file
against its source and refuses to score when the build is older:

```
$ touch src/search/store.ts && node eval/run.mjs
=== paperless-mcp search acceptance harness ===

REFUSING TO SCORE -- build/ is OLDER than src/:

  build/search/store.js is older than src/search/store.ts

Run `npm run build` first. Measuring a stale build reports numbers for code
that is not the code under review.
$ echo $?
1
```

This gate was verified by exactly that mutation on 2026-11-15, then the mtime was
restored. **When you see it, run `npm run build` and re-run — do not pass it a
lower number, and do not import `src/` instead.**

## Baselines — read these, do not invent numbers

Committed under `eval/results/`. These are measurements with timestamps, not
constants; re-run and compare rather than quoting them forward.

**`l8-2026-11-15-baseline.log`** — the 20-case gate, arms at the shipped shape:

```
arm                                    @10       @50      @200
vector-only                          14/20     16/20     19/20
lexical-only                         16/20     19/20     20/20
fused (shipped: arms at k*4)         15/20     18/20     20/20
fused (arms fixed at 200)            15/20     18/20     20/20
```

**`fusion-bakeoff-2026-11-15T22-39-47.log`** — 24 fusion strategies at a fixed
arm depth of 200. Measured @10 distribution, counted from the log:

```
18/20   11 strategies   max-of-two, rrf k=0/1/2/5/10, zscore sum (2 variants),
                        interleave (2 variants), rrf k=0 head=10 only
17/20    4 strategies   rrf k=20, minmax sum, minmax sum wV=2,
                        rrf k=60 head=10 only
16/20    5 strategies   lexical-only, wrrf wL=0.5, wrrf wV=2,
                        minmax sum wL=2, zscore miss=min wL=2
15/20    1 strategy     rrf k=60 (THE SHIPPED FORM)
14/20    3 strategies   vector-only, wrrf wV=0.5, wrrf wL=2
```

**The shipped `rrf k=60` is the WORST fusion strategy in the field at @10.** The
mechanism is arithmetic: with `k=60`, `2/(60+D) > 1/61` for every `D < 62`, so any
document found by *both* arms anywhere in the top 62 outranks *any* perfect
single-arm hit. RRF optimises for consensus, and consensus is the wrong prior
when the arms are materially unequal on a query class — ours are (Tibetan-heavy
dharma against mostly-English prose).

⚠️ **The strategy is deliberately still unchosen.** `max-of-two` ties the best
score with *no fusion at all*, which is a real argument for deleting the ranker
rather than tuning it — but at @50 and @200 several strategies tie, so the
fixture is not yet sharp enough to separate them. Sharpen the fixture, then choose.

⚠️ **Only the 22:39 run is admissible — and the criterion is a validation count,
not a timestamp.** That run reported `membershipChecks: 120, membershipMismatch: 0`
(every recorded rank re-verified against a call at that depth). The 22:31 run
reported **13 mismatches** of 120 (`doc 215 lexical@10: at-k=null deep-trunc=73`),
and the 22:42 run pinned `arm depth fixed at 40`, which corrupts its own @200
column by construction. Three runs, three different answers — the difference was
instrument integrity, not the corpus.

⚠️ **`@200 = 20/20` on both the fused and lexical arms means every target is
retrievable — coverage is complete. A low `@10` is a RANKING result, not a
missing-corpus result.** Read those two rows together before concluding anything.

## Two different env sets — do not conflate them

| to BUILD the index | to RUN the eval |
|---|---|
| `PAPERLESS_URL` | `PAPERLESS_MCP_INDEX_DIR` (or `_INDEX_DB`) |
| `PAPERLESS_API_KEY` | `PAPERLESS_MCP_EMBEDDING_URL` |
| `PAPERLESS_MCP_EMBEDDING_*` | `PAPERLESS_MCP_EMBEDDING_MODEL` |

`eval/run.mjs` reads **only** the right-hand column — it never touches
`PAPERLESS_URL` or `PAPERLESS_API_KEY`. Those are consumed by
`scripts/build-index.ts`. The shared env file supplies both, which is why
sourcing one file works for both jobs.

## Environment

| var | default |
|---|---|
| `PAPERLESS_MCP_INDEX_DB` | `$PAPERLESS_MCP_INDEX_DIR/index.db` |
| `PAPERLESS_MCP_INDEX_DIR` | `~/.local/share/paperless-mcp/index` |
| `PAPERLESS_MCP_EMBEDDING_URL` | `http://127.0.0.1:11434` |
| `PAPERLESS_MCP_EMBEDDING_MODEL` | `nomic-embed-text` |

The index is opened **readonly** — a measurement pass must never write to a live
1.9 GB index.
