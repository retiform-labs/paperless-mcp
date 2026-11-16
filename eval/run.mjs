#!/usr/bin/env node
// Acceptance harness for the paperless-mcp semantic search arms.
//
// WHY THIS EXISTS: an earlier harness scored 10/20 against a >=16/20 gate and at
// least one of those misses was bad ground truth, not a retrieval failure. This
// runner refuses to score a case whose target document does not contain the
// query's own anchor terms. A measurement you cannot trust is worse than no
// measurement.
//
// WHAT IT MEASURES: the three arms directly against the index, so HTTP and
// tool-layer overhead cannot contaminate the retrieval number:
//   vector  - Ollama embedding -> vec_chunks kNN          (store.vectorHits)
//   lexical - FTS5 BM25                                     (store.textHits)
//   fused   - reciprocal rank fusion at the SHIPPED k       (fuse, rank.js)
//
// WHAT k MEANS HERE: k is DISTINCT DOCUMENTS RETURNED, not kNN candidates.
// store.vectorHits() does per-document dedupe and widens its candidate pool
// until it has `limit` distinct documents or hits VECTOR_CANDIDATE_CEILING.
// store.textHits() likewise keeps only each document's best chunk. So the
// quantity varied below is documents, and every arm is compared on the same one.

import Database from "better-sqlite3";
import * as sqliteVec from "sqlite-vec";
import { readFileSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { SearchIndex } from "../build/search/store.js";
import { fuse, DEFAULT_FUSE_K } from "../build/search/rank.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const DEPTHS = [10, 50, 200];
// Read the SHIPPED constant rather than restating it. A harness that hard-codes
// its own copy of k measures a form that may no longer be the one in production
// -- and it would keep doing so silently, which is precisely the false-premise
// class this runner already refuses to score on.
const FUSE_K = DEFAULT_FUSE_K;

const DB_PATH =
  process.env.PAPERLESS_MCP_INDEX_DB ||
  join(
    process.env.PAPERLESS_MCP_INDEX_DIR ||
      join(process.env.HOME || ".", ".local/share/paperless-mcp/index"),
    "index.db"
  );

// Results logs are COMMITTED, so the absolute home directory must never reach
// one. Scrub at the point of writing; scrubbing afterwards leaked twice.
const HOME_DIR = process.env.HOME || "";
const displayPath = (p) =>
  HOME_DIR && p.startsWith(HOME_DIR) ? "~" + p.slice(HOME_DIR.length) : p;

const EMBED_URL = (process.env.PAPERLESS_MCP_EMBEDDING_URL || "http://127.0.0.1:11434").replace(/\/$/, "");
const EMBED_MODEL = process.env.PAPERLESS_MCP_EMBEDDING_MODEL || "nomic-embed-text";
const EMBED_PATH = "/api/embeddings";

const fixture = JSON.parse(readFileSync(join(HERE, "queries.json"), "utf8"));
const cases = fixture.cases;

function norm(s) {
  return String(s || "").toLowerCase();
}

// ---------------------------------------------------------------------------
// PRECONDITION GATE. Reads every chunk of the target document in THIS run and
// asserts each anchor term is present. Refusing to score is the whole point:
// a case whose premise is false produces a "miss" that is a fact about the
// fixture, not about the retriever.
// ---------------------------------------------------------------------------
function checkPreconditions(db) {
  const chunkStmt = db.prepare("SELECT text FROM chunks WHERE doc_id=?");
  const docStmt = db.prepare("SELECT id, title FROM documents WHERE id=?");
  const failures = [];
  const notes = [];

  for (const c of cases) {
    const doc = docStmt.get(c.id);
    if (!doc) {
      failures.push(`PRECONDITION FAILED: doc ${c.id} is not in the index`);
      continue;
    }
    const body = norm(chunkStmt.all(c.id).map((r) => r.text).join("\n"));
    if (!body) {
      failures.push(`PRECONDITION FAILED: doc ${c.id} has no chunk text`);
      continue;
    }
    for (const anchor of c.anchors) {
      const n = body.split(norm(anchor)).length - 1;
      if (n === 0) {
        failures.push(
          `PRECONDITION FAILED: doc ${c.id} lacks ${JSON.stringify(anchor)}`
        );
      }
    }
    notes.push({ id: c.id, chars: body.length, anchors: c.anchors.length });
  }
  return { failures, notes };
}

// ---------------------------------------------------------------------------
// Embedding. The query prefix is MANDATORY for nomic-embed-text: the index was
// built with prefix_mode=nomic, and a bare query embeds into a different region
// of the space. Applying it here keeps the arm honest.
// ---------------------------------------------------------------------------
async function embedQuery(query) {
  const res = await fetch(`${EMBED_URL}${EMBED_PATH}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: EMBED_MODEL, prompt: `search_query: ${query}` }),
  });
  if (!res.ok) {
    throw new Error(`embedding HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
  }
  const json = await res.json();
  if (!Array.isArray(json.embedding) || json.embedding.length === 0) {
    throw new Error("embedding response carried no vector");
  }
  return json.embedding;
}

function rankOf(list, id) {
  const i = list.findIndex((h) => h.documentId === id);
  return i === -1 ? null : i + 1;
}

/**
 * This runner exercises the COMPILED store, not the TypeScript source:
 * importing build/ is what a consumer actually receives (and a ts-node import
 * would prove nothing about the shipped artifact). The cost is that a stale
 * build silently measures code nobody is going to run.
 *
 * Measured failure this prevents: `build/` once landed at `build/src/...`
 * because one test import widened tsc's rootDir, so `build/index.js` was a Nov 5
 * artifact while the tree was current. 473 green tests did not catch it.
 */
function checkBuildFreshness() {
  const pairs = [
    ["../build/search/store.js", "../src/search/store.ts"],
    ["../build/search/rank.js", "../src/search/rank.ts"],
  ];
  const stale = [];
  for (const [built, source] of pairs) {
    try {
      const b = statSync(new URL(built, import.meta.url)).mtimeMs;
      const s = statSync(new URL(source, import.meta.url)).mtimeMs;
      if (b < s) stale.push({ built, source, b, s });
    } catch {
      // A missing file is a different failure and will throw on import anyway.
    }
  }
  return stale;
}

async function main() {
  console.log("=== paperless-mcp search acceptance harness ===\n");

  const stale = checkBuildFreshness();
  if (stale.length > 0) {
    console.error("REFUSING TO SCORE -- build/ is OLDER than src/:\n");
    for (const x of stale) {
      console.error(
        `  ${x.built.replace("../", "")} is older than ${x.source.replace("../", "")}`
      );
    }
    console.error(
      "\nRun `npm run build` first. Measuring a stale build reports numbers for code\n" +
        "that is not the code under review."
    );
    process.exit(1);
  }
  console.log(`index    ${displayPath(DB_PATH)}`);
  console.log(`embedder ${EMBED_URL} model=${EMBED_MODEL}`);
  console.log(`cases    ${cases.length}  depths ${DEPTHS.join(", ")}\n`);

  // ---- PRECONDITIONS FIRST, BEFORE ANY SCORING -----------------------------
  const pdb = new Database(DB_PATH, { readonly: true });
  sqliteVec.load(pdb);
  const { failures, notes } = checkPreconditions(pdb);
  pdb.close();

  if (failures.length > 0) {
    console.error("REFUSING TO SCORE -- fixture premises are false:\n");
    for (const f of failures) console.error("  " + f);
    console.error(
      `\n${failures.length} precondition failure(s). No recall numbers are reported,` +
        ` because a miss against a false premise measures the fixture, not the engine.`
    );
    process.exit(1);
  }
  console.log(`preconditions OK: ${cases.length}/${cases.length} cases, ` +
    `${notes.reduce((a, n) => a + n.anchors, 0)} anchor terms verified present in the corpus\n`);

  // ---- ARMS ----------------------------------------------------------------
  const index = new SearchIndex(DB_PATH, { readonly: true });
  index.open();
  if (!index.lexicalAvailable) {
    console.error("REFUSING TO SCORE: this SQLite build has no FTS5, so the lexical arm cannot run.");
    process.exit(1);
  }

  const maxDepth = Math.max(...DEPTHS);
  const perCase = [];
  let i = 0;

  for (const c of cases) {
    i++;
    const embedding = await embedQuery(c.query);

    // Each depth is measured by calling the shipped method AT that limit, so the
    // recorded number is what a caller asking for k would actually receive.
    const vectorByDepth = {};
    const lexicalByDepth = {};
    const fusedShippedByDepth = {};
    const fusedDeepByDepth = {};

    // One fixed-depth fetch of each arm, fused once. This exists to separate
    // "fusion is weak" from "the pool was shallow": the shipped path scales its
    // arm depth with k (k*4), so at k=10 it fuses two lists of 40 while at k=200
    // it fuses two of 800 -- a comparison across k that changes two variables at
    // once. This arm holds the arm depth CONSTANT at `maxDepth` so the only thing
    // varying is k. Note the consequence, stated plainly: at k=200 the shipped
    // arm is actually DEEPER than this one (800 vs 200), which is why the two
    // diverge in per-case ranks while tying on the summary.
    const vectorDeep = index.vectorHits(embedding, maxDepth);
    const lexicalDeep = index.textHits(c.query, maxDepth);
    const fusedDeep = fuse(vectorDeep, lexicalDeep, FUSE_K);

    for (const k of DEPTHS) {
      const v = index.vectorHits(embedding, k);
      const t = index.textHits(c.query, k);
      // Shipped shape: runSemanticSearch fetches each arm at limit*4 then fuses.
      const fusedShipped = fuse(
        index.vectorHits(embedding, k * 4),
        index.textHits(c.query, k * 4),
        FUSE_K
      );
      vectorByDepth[k] = rankOf(v, c.id);
      lexicalByDepth[k] = rankOf(t, c.id);
      fusedShippedByDepth[k] = rankOf(fusedShipped, c.id);
      fusedDeepByDepth[k] = rankOf(fusedDeep, c.id);
    }

    perCase.push({
      id: c.id,
      control: !!c.control,
      query: c.query,
      vector: vectorByDepth,
      lexical: lexicalByDepth,
      fusedShipped: fusedShippedByDepth,
      fusedDeep: fusedDeepByDepth,
    });

    const mark = (r) => (r === null ? "  -" : String(r).padStart(3));
    console.log(
      `[${String(i).padStart(2)}/${cases.length}] doc ${String(c.id).padStart(4)}` +
        `  vec ${DEPTHS.map((k) => mark(vectorByDepth[k])).join(" ")}` +
        `  lex ${DEPTHS.map((k) => mark(lexicalByDepth[k])).join(" ")}` +
        `  fus ${DEPTHS.map((k) => mark(fusedShippedByDepth[k])).join(" ")}` +
        (c.control ? "   <- CONTROL" : "")
    );
  }

  // ---- SUMMARY -------------------------------------------------------------
  const arms = [
    ["vector-only", "vector"],
    ["lexical-only", "lexical"],
    ["fused (shipped: arms at k*4)", "fusedShipped"],
    ["fused (arms fixed at 200)", "fusedDeep"],
  ];

  console.log("\n=== RECALL: hit@k, counted in DISTINCT DOCUMENTS ===");
  console.log(`(n=${cases.length} cases; a rank of "-" means not returned at that depth)\n`);
  const header = ["arm".padEnd(32), ...DEPTHS.map((k) => `@${k}`.padStart(9))].join(" ");
  console.log(header);
  console.log("-".repeat(header.length));

  const summary = {};
  for (const [label, key] of arms) {
    const cells = [];
    const row = {};
    for (const k of DEPTHS) {
      // A hit means the target is in the arm's TOP k. `rankOf` reports the
      // position in the whole returned list, which for a fused arm can be up to
      // 4k long because the arms are fetched at k*4. Counting "not null" would
      // therefore score "appears anywhere in 80" as a hit@10. Measured: that
      // inflated fused hit@10 from 15 to 19 on this fixture.
      let hits = 0;
      for (const c of perCase) {
        const r = c[key][k];
        if (r !== null && r <= k) hits++;
      }
      cells.push(`${hits}/${cases.length}`.padStart(9));
      row[k] = hits;
    }
    summary[key] = row;
    console.log([label.padEnd(32), ...cells].join(" "));
  }

  // The control case is reported separately: it shares verbatim vocabulary with
  // its target, so it calibrates the instrument and is not evidence for vectors.
  const control = perCase.find((c) => c.control);
  if (control) {
    console.log(
      `\nCONTROL doc ${control.id}: vector ${DEPTHS.map((k) => control.vector[k] ?? "-").join("/")}` +
        `  lexical ${DEPTHS.map((k) => control.lexical[k] ?? "-").join("/")}` +
        `  fused ${DEPTHS.map((k) => control.fusedShipped[k] ?? "-").join("/")}`
    );
    const controlFound =
      control.fusedShipped[maxDepth] !== null && control.fusedShipped[maxDepth] <= maxDepth;
    console.log(
      controlFound
        ? "  control retrieved at max depth -- the harness pipeline is working."
        : "  CONTROL NOT RETRIEVED -- the harness itself is broken; do not read the numbers above."
    );
    if (!controlFound) process.exitCode = 1;
  }

  const nonControl = perCase.filter((c) => !c.control);
  console.log(`\n=== EXCLUDING THE CONTROL CASE (n=${nonControl.length}) ===`);
  for (const [label, key] of arms) {
    const cells = [];
    for (const k of DEPTHS) {
      let hits = 0;
      for (const c of nonControl) {
        const r = c[key][k];
        if (r !== null && r <= k) hits++;
      }
      cells.push(`${hits}/${nonControl.length}  (${((hits / nonControl.length) * 100).toFixed(0)}%)`.padStart(16));
    }
    console.log([label.padEnd(32), ...cells].join(" "));
  }

  // --- machine-readable block, for a future regression gate -----------------
  console.log("\n=== JSON ===");
  console.log(
    JSON.stringify(
      {
        index: displayPath(DB_PATH),
        cases: cases.length,
        depths: DEPTHS,
        fuseK: FUSE_K,
        recall: summary,
        perCase: perCase.map((c) => ({
          id: c.id,
          control: c.control,
          vector: c.vector,
          lexical: c.lexical,
          fusedShipped: c.fusedShipped,
          fusedDeep: c.fusedDeep,
        })),
      },
      null,
      2
    )
  );
}

main().catch((err) => {
  console.error("HARNESS ERROR:", err?.stack || err);
  process.exit(1);
});
