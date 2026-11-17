#!/usr/bin/env node
// FUSION BAKE-OFF -- measurement only. Touches nothing under src/.
//
// WHY: production fusion is RRF, and the value of k is a real lever.
// (History: production ran k=60 until 2026-11-16, inherited from the upstream
// fork. This harness is what showed it was the weakest arm in the field.)
// At @10 on the 20-case fixture k=60 scores 15/20 while the best single arm
// scores 16/20, and every worst regression is a document only ONE arm
// retrieves. The arithmetic explains why: with k a doc both arms find at rank D
// scores 2/(k+D) while a doc one arm finds at rank 1 scores 1/(k+1) -- so any
// dual-arm doc with D < k+2 outranks any perfect single-arm hit, and at k=60
// that means the entire top 62. That is a property of RRF, not a bug.
//
// WHAT THIS DOES: holds the retrievers FIXED (the shipped store methods) and
// varies ONLY the fusion. Every strategy consumes the same two deep lists per
// case, so a difference between rows is a difference in fusion, full stop.
//
// WHAT IT DOES NOT DO: it does not change production behaviour, and it does not
// decide. 20 cases is a small sample; where two strategies differ by one case
// that is inside the fixture's noise and the report says so.
//
// VALIDATION BUILT IN: the "rrf k=60 (upstream default)" and "rrf k=10 (SHIPPED
// FORM)" rows must reproduce their previously measured values (15/@10 and 18/@10
// at arms pinned at 200). If either does not, the harness is wrong.
//
// ⚠️ Strategy names are module CONSTANTS (MAX_OF_TWO / RRF_UPSTREAM /
// RRF_SHIPPED) because the reporting section indexes this table BY NAME. Rename
// a label without updating those and the run dies mid-table with a TypeError.

import { readFileSync, statSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import Database from "better-sqlite3";
import * as sqliteVec from "sqlite-vec";
import { SearchIndex } from "../build/search/store.js";
import { fuse } from "../build/search/rank.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const DEPTHS = [10, 50, 200];
const DEEP = Number(process.env.BAKEOFF_ARM_DEPTH || 200); // fixed arm depth for every fusion strategy

const DB_PATH =
  process.env.PAPERLESS_MCP_INDEX_DB ||
  join(
    process.env.PAPERLESS_MCP_INDEX_DIR ||
      join(process.env.HOME || ".", ".local/share/paperless-mcp/index"),
    "index.db"
  );

// Results logs are COMMITTED (see .gitignore's `!eval/results/*.log`), so the
// absolute home directory must never reach one. Scrub at the point of writing,
// not afterwards -- the previous scrub-by-hand approach leaked twice.
const HOME_DIR = process.env.HOME || "";
const displayPath = (p) =>
  HOME_DIR && p.startsWith(HOME_DIR) ? "~" + p.slice(HOME_DIR.length) : p;
const EMBED_URL = (process.env.PAPERLESS_MCP_EMBEDDING_URL || "http://127.0.0.1:11434").replace(/\/$/, "");
const EMBED_MODEL = process.env.PAPERLESS_MCP_EMBEDDING_MODEL || "nomic-embed-text";

const fixture = JSON.parse(readFileSync(join(HERE, "queries.json"), "utf8"));
const cases = fixture.cases;

function norm(s) { return String(s || "").toLowerCase(); }

// --- build staleness gate (same reasoning as run.mjs: a stale build measures
// --- code nobody will run) -------------------------------------------------
function checkBuildFreshness() {
  const pairs = [["../build/search/store.js", "../src/search/store.ts"], ["../build/search/rank.js", "../src/search/rank.ts"]];
  const stale = [];
  for (const [built, source] of pairs) {
    try {
      const b = statSync(new URL(built, import.meta.url)).mtimeMs;
      const s = statSync(new URL(source, import.meta.url)).mtimeMs;
      if (b < s) stale.push({ built, source });
    } catch { /* missing file throws on import */ }
  }
  return stale;
}

// --- precondition gate: refuse to score a case whose target lacks the anchors
function checkPreconditions(db) {
  const chunkStmt = db.prepare("SELECT text FROM chunks WHERE doc_id=?");
  const docStmt = db.prepare("SELECT id, title FROM documents WHERE id=?");
  const failures = [];
  let anchorTotal = 0;
  for (const c of cases) {
    const doc = docStmt.get(c.id);
    if (!doc) { failures.push(`doc ${c.id} is not in the index`); continue; }
    const body = norm(chunkStmt.all(c.id).map((r) => r.text).join("\n"));
    if (!body) { failures.push(`doc ${c.id} has no chunk text`); continue; }
    for (const anchor of c.anchors) {
      anchorTotal++;
      if (body.split(norm(anchor)).length - 1 === 0) failures.push(`doc ${c.id} lacks ${JSON.stringify(anchor)}`);
    }
  }
  return { failures, anchorTotal };
}

async function embedQuery(q) {
  const res = await fetch(`${EMBED_URL}/api/embeddings`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: EMBED_MODEL, prompt: `search_query: ${q}` }),
  });
  if (!res.ok) throw new Error(`embedding HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
  const json = await res.json();
  if (!Array.isArray(json.embedding) || json.embedding.length === 0) throw new Error("embedding response carried no vector");
  return json.embedding;
}

// ---------------------------------------------------------------------------
// FUSION STRATEGIES
// Every strategy consumes the SAME two deep lists per case and returns an
// ordered array of documentIds. Nothing here touches the retriever.
//
// Both arms' `score` is already "higher is better":
//   vector  score = 1 - distance          (store.js:728)
//   lexical score = -bm25(fts)            (store.js:771, bm25 is negative-best-first)
// The store's own KDoc says the two are comparable only in ORDER, not in
// magnitude -- which is exactly the problem the normalising families address.
// ---------------------------------------------------------------------------

function bestRanks(list) {
  const b = new Map();
  list.forEach((h, i) => { if (!b.has(h.documentId)) b.set(h.documentId, i + 1); });
  return b;
}

// Deterministic ordering: score desc, then the document's best rank in any arm
// asc (a doc that was genuinely near the top wins a tie), then id asc.
function order(map, bv, bt) {
  return [...map.entries()]
    .sort((a, b) => {
      if (b[1] !== a[1]) return b[1] - a[1];
      const ra = Math.min(bv.get(a[0]) ?? Infinity, bt.get(a[0]) ?? Infinity);
      const rb = Math.min(bv.get(b[0]) ?? Infinity, bt.get(b[0]) ?? Infinity);
      if (ra !== rb) return ra - rb;
      return String(a[0]).localeCompare(String(b[0]));
    })
    .map(([d]) => d);
}

function rrf(vList, tList, k, wv = 1, wt = 1) {
  const bv = bestRanks(vList), bt = bestRanks(tList);
  const m = new Map();
  for (const [d, r] of bv) m.set(d, (m.get(d) ?? 0) + wv / (k + r));
  for (const [d, r] of bt) m.set(d, (m.get(d) ?? 0) + wt / (k + r));
  return order(m, bv, bt);
}

function minmaxNorm(list) {
  const vals = list.map((h) => h.score);
  const m = new Map();
  if (vals.length === 0) return m;
  const mn = Math.min(...vals), mx = Math.max(...vals);
  const span = mx - mn;
  list.forEach((h) => {
    if (!m.has(h.documentId)) m.set(h.documentId, span > 0 ? (h.score - mn) / span : 1);
  });
  return m;
}

function zNorm(list, missing) {
  const vals = list.map((h) => h.score);
  const m = new Map();
  if (vals.length === 0) return m;
  const n = vals.length;
  const mean = vals.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
  list.forEach((h) => {
    if (!m.has(h.documentId)) m.set(h.documentId, sd > 0 ? (h.score - mean) / sd : 0);
  });
  // A document one arm never found has no z-score. Writing 0 means "at the mean",
  // which REWARDS absence (the worst real hit is below the mean). Writing the
  // arm's own minimum means "worst in this arm", which is the honest reading and
  // is what `missing:"min"` selects.
  if (missing === "min") {
    const mn = Math.min(...m.values());
    m.__missing = mn;
  }
  return m;
}

function scoreSum(vList, tList, normFn, wv = 1, wt = 1) {
  const bv = bestRanks(vList), bt = bestRanks(tList);
  const a = normFn(vList), b = normFn(tList);
  const fillA = a.__missing ?? 0, fillB = b.__missing ?? 0;
  const m = new Map();
  const all = new Set([...a.keys(), ...b.keys()]);
  for (const d of all) {
    const sa = a.has(d) ? a.get(d) : fillA;
    const sb = b.has(d) ? b.get(d) : fillB;
    m.set(d, wv * sa + wt * sb);
  }
  return order(m, bv, bt);
}

// The trivial floor: whichever arm found it better. Not a ship candidate; it is
// the ceiling any fusion has to beat.
function singleArmMax(vList, tList) {
  const bv = bestRanks(vList), bt = bestRanks(tList);
  const m = new Map();
  const all = new Set([...bv.keys(), ...bt.keys()]);
  for (const d of all) {
    const rv = bv.get(d) ?? Infinity;
    const rt = bt.get(d) ?? Infinity;
    m.set(d, -Math.min(rv, rt)); // higher = better rank
  }
  return order(m, bv, bt);
}

function listOnly(list) { return list.map((h) => h.documentId); }

// ROUND-ROBIN INTERLEAVE. Every other strategy MINES the union and re-ranks it,
// which means a document sitting at rank 9 in one arm can be pushed out of the
// top 10 by documents that arm never returned. Interleaving instead PRESERVES
// each arm's head by construction: position 2n is vector's (n+1)th document and
// 2n+1 is lexical's, so the fused top 10 always contains each arm's top 5.
// It is a genuinely different family -- no scores, no ranks beyond position --
// and it is the only tested policy whose guarantee is analytic rather than
// emergent. `first` breaks ties when only one arm has a document left.
function interleave(vList, tList, first = "vector") {
  const v = vList.map((h) => h.documentId);
  const t = tList.map((h) => h.documentId);
  const seen = new Set();
  const out = [];
  let i = 0, j = 0;
  const a = first === "vector" ? v : t;
  const b = first === "vector" ? t : v;
  while (i < a.length || j < b.length) {
    if (i < a.length) { const d = a[i++]; if (!seen.has(d)) { seen.add(d); out.push(d); } }
    if (j < b.length) { const d = b[j++]; if (!seen.has(d)) { seen.add(d); out.push(d); } }
  }
  return out;
}

// RRF RESTRICTED TO THE UNION OF EACH ARM'S HEAD. The shipped candidate pool is
// the two full deep lists, so a dual-arm document at ranks 40/40 can outscore a
// single-arm document at rank 1 (the arithmetic in the header). Restricting the
// candidate set to documents that appear in either arm's first `head` positions
// removes that failure mode without touching the score formula.
function rrfHeaded(vList, tList, k, head) {
  const v = vList.slice(0, head), t = tList.slice(0, head);
  const allow = new Set([...v.map((h) => h.documentId), ...t.map((h) => h.documentId)]);
  const bv = bestRanks(vList), bt = bestRanks(tList);
  const m = new Map();
  for (const [d, r] of bv) if (allow.has(d)) m.set(d, (m.get(d) ?? 0) + 1 / (k + r));
  for (const [d, r] of bt) if (allow.has(d)) m.set(d, (m.get(d) ?? 0) + 1 / (k + r));
  return order(m, bv, bt);
}

function rankOf(list, id) {
  const i = list.findIndex((d) => d === id);
  return i === -1 ? null : i + 1;
}

// ---------------------------------------------------------------------------
// THE STRATEGY TABLE. Order matters only for reporting; each row is independent.
// ---------------------------------------------------------------------------
// Strategy names are CONSTANTS and the reporting below indexes by them. Rename
// a label without updating every lookup and the run dies mid-table with a
// TypeError, as it did on 2026-11-16 -- which is the good failure (it refuses
// rather than reporting from a partial map), but it is still avoidable.
const MAX_OF_TWO = "single: max-of-two (floor)";
const RRF_UPSTREAM = "rrf k=60 (upstream default)";
const RRF_SHIPPED = "rrf k=10 (SHIPPED FORM)";

const STRATEGIES = [
  ["single: vector-only",        (v, t) => listOnly(v)],
  ["single: lexical-only",       (v, t) => listOnly(t)],
  [MAX_OF_TWO,                   (v, t) => singleArmMax(v, t)],
  [RRF_UPSTREAM,                 (v, t) => rrf(v, t, 60)],
  ["rrf k=20",                   (v, t) => rrf(v, t, 20)],
  [RRF_SHIPPED,                  (v, t) => rrf(v, t, 10)],
  ["rrf k=5",                    (v, t) => rrf(v, t, 5)],
  ["rrf k=2",                    (v, t) => rrf(v, t, 2)],
  ["rrf k=1",                    (v, t) => rrf(v, t, 1)],
  ["rrf k=0",                    (v, t) => rrf(v, t, 0)],
  ["wrrf k=60 wV=0.5",           (v, t) => rrf(v, t, 60, 0.5, 1)],
  ["wrrf k=60 wL=0.5",           (v, t) => rrf(v, t, 60, 1, 0.5)],
  ["wrrf k=60 wV=2",             (v, t) => rrf(v, t, 60, 2, 1)],
  ["wrrf k=60 wL=2",             (v, t) => rrf(v, t, 60, 1, 2)],
  ["minmax sum",                 (v, t) => scoreSum(v, t, minmaxNorm, 1, 1)],
  ["minmax sum wV=2",            (v, t) => scoreSum(v, t, minmaxNorm, 2, 1)],
  ["minmax sum wL=2",            (v, t) => scoreSum(v, t, minmaxNorm, 1, 2)],
  ["zscore sum missing=min",     (v, t) => scoreSum(v, t, (l) => zNorm(l, "min"), 1, 1)],
  ["zscore sum missing=0",       (v, t) => scoreSum(v, t, (l) => zNorm(l, "zero"), 1, 1)],
  ["zscore sum miss=min wL=2",   (v, t) => scoreSum(v, t, (l) => zNorm(l, "min"), 1, 2)],
  ["interleave v-first",         (v, t) => interleave(v, t, "vector")],
  ["interleave l-first",         (v, t) => interleave(v, t, "text")],
  ["rrf k=60 head=10 only",      (v, t) => rrfHeaded(v, t, 60, 10)],
  ["rrf k=0  head=10 only",      (v, t) => rrfHeaded(v, t, 0, 10)],
];

async function main() {
  const log = [];
  const say = (s) => { console.log(s); log.push(s); };

  say("=== fusion bake-off -- paperless-mcp ===");
  const stale = checkBuildFreshness();
  if (stale.length > 0) {
    console.error("REFUSING: build/ older than src/:", JSON.stringify(stale));
    process.exit(1);
  }
  say(`index    ${displayPath(DB_PATH)}`);
  say(`embedder ${EMBED_URL} model=${EMBED_MODEL}`);
  say(`cases    ${cases.length}   depths ${DEPTHS.join(", ")}   arm depth fixed at ${DEEP}\n`);

  const pdb = new Database(DB_PATH, { readonly: true });
  sqliteVec.load(pdb);
  const { failures, anchorTotal } = checkPreconditions(pdb);
  pdb.close();
  if (failures.length > 0) {
    console.error("REFUSING TO SCORE -- fixture premises false:");
    for (const f of failures) console.error("  " + f);
    process.exit(1);
  }
  say(`preconditions OK: ${cases.length}/${cases.length} cases, ${anchorTotal} anchor terms verified present\n`);

  const index = new SearchIndex(DB_PATH, { readonly: true });
  index.open();
  if (!index.lexicalAvailable) { console.error("REFUSING: no FTS5 in this SQLite build."); process.exit(1); }

  // One embed per query, reused by every strategy. Stage 2a of the task brief.
  const embedCache = new Map();
  const rows = [];
  let i = 0;

  for (const c of cases) {
    i++;
    if (!embedCache.has(c.query)) embedCache.set(c.query, await embedQuery(c.query));
    const emb = embedCache.get(c.query);

    const vDeep = index.vectorHits(emb, DEEP);
    const tDeep = index.textHits(c.query, DEEP);

    // Per-depth arms via the shipped method, so the baseline row is measured the
    // way the baseline was measured, not reconstructed from the deep list.
    const armAtDepth = {};
    for (const k of DEPTHS) {
      armAtDepth[k] = {
        vector: rankOf(index.vectorHits(emb, k).map((h) => h.documentId), c.id),
        lexical: rankOf(index.textHits(c.query, k).map((h) => h.documentId), c.id),
      };
    }

    const out = {};
    for (const [name, fn] of STRATEGIES) {
      const ordered = fn(vDeep, tDeep);
      const byDepth = {};
      for (const k of DEPTHS) byDepth[k] = rankOf(ordered, c.id);
      out[name] = byDepth;
    }

    rows.push({ id: c.id, control: !!c.control, query: c.query, armAtDepth, strategies: out });

    const mk = (r) => (r === null ? "  -" : String(r).padStart(3));
    const f60 = out[RRF_UPSTREAM][10], f10 = out[RRF_SHIPPED][10];
    say(`[${String(i).padStart(2)}/${cases.length}] doc ${String(c.id).padStart(4)}` +
        `  vec ${DEPTHS.map((k) => mk(armAtDepth[k].vector)).join(" ")}` +
        `  lex ${DEPTHS.map((k) => mk(armAtDepth[k].lexical)).join(" ")}` +
        `  rrf60@10 ${mk(f60)}  rrf10@10 ${mk(f10)}` +
        (c.control ? "   <- CONTROL" : ""));
  }

  // ---- VALIDATION ---------------------------------------------------------
  // The honest question is not "is the position identical" but "does the DEEP
  // list contain the same TOP-k SET as the call-at-k list". Two different
  // candidate pools can rank the same document differently while agreeing on
  // membership, and only membership decides a hit@k. The earlier version of this
  // check compared positions and flagged every "null vs 21" as a mismatch, which
  // was a bug in the check, not a finding about the data.
  let memChecks = 0, memMismatch = 0;
  const memDetail = [];
  let setOverlap = 0, setTotal = 0;
  for (const c of rows) {
    for (const k of DEPTHS) {
      for (const side of ["vector", "lexical"]) {
        memChecks++;
        const deepKey = side === "vector" ? "single: vector-only" : "single: lexical-only";
        const atPos = c.armAtDepth[k][side];
        const deepPos = c.strategies[deepKey][k];
        const atHit = atPos !== null && atPos <= k;
        const deepHit = deepPos !== null && deepPos <= k;
        if (atHit !== deepHit) {
          memMismatch++;
          if (memDetail.length < 12) memDetail.push(`doc ${c.id} ${side}@${k}: at-k=${atPos} deep-trunc=${deepPos}`);
        }
        // How much of the top-k membership survives: 0 or 1 per (case,side,k)
        setTotal++;
        if (atHit === deepHit) setOverlap++;
      }
    }
  }
  say(`\nVALIDATION top-k MEMBERSHIP, deep(${DEEP})-list truncated vs call-at-k: ` +
      `${memChecks - memMismatch}/${memChecks} agree` + (memMismatch ? `  (${memMismatch} DISAGREE)` : "  (identical)"));
  for (const d of memDetail) say("   " + d);

  // ---- RECALL TABLE --------------------------------------------------------
  const nonControl = rows.filter((r) => !r.control);
  say("\n=== RECALL: hits@k over n=" + rows.length + " (rank counted against k) ===");
  const hdr = ["strategy".padEnd(30), ...DEPTHS.map((k) => `@${k}`.padStart(9)), "  helped/hurt@10 vs best single arm"].join(" ");
  say(hdr);
  say("-".repeat(hdr.length));

  const table = {};
  for (const [name] of STRATEGIES) {
    const cells = [];
    const rec = {};
    for (const k of DEPTHS) {
      let hits = 0;
      for (const r of rows) { const v = r.strategies[name][k]; if (v !== null && v <= k) hits++; }
      rec[k] = hits;
      cells.push(`${hits}/${rows.length}`.padStart(9));
    }
    // helped / hurt at @10 vs the better of the two arms AT @10
    let helped = 0, hurt = 0, sameHit = 0, sameMiss = 0;
    const helpedIds = [], hurtIds = [];
    for (const r of rows) {
      const k = 10;
      const av = r.armAtDepth[k].vector, al = r.armAtDepth[k].lexical;
      const armHit = (av !== null && av <= k) || (al !== null && al <= k);
      const sv = r.strategies[name][k];
      const fuseHit = sv !== null && sv <= k;
      if (fuseHit && !armHit) { helped++; helpedIds.push(r.id); }
      else if (!fuseHit && armHit) { hurt++; hurtIds.push(r.id); }
      else if (fuseHit) sameHit++; else sameMiss++;
    }
    table[name] = { recall: rec, helped, hurt, sameHit, sameMiss, helpedIds, hurtIds };
    say([name.padEnd(30), ...cells, `  +${helped} / -${hurt}`].join(" "));
  }

  // ---- n=19 (control excluded) --------------------------------------------
  say(`\n=== EXCLUDING THE CONTROL CASE (n=${nonControl.length}) ===`);
  for (const [name] of STRATEGIES) {
    const cells = [];
    for (const k of DEPTHS) {
      let hits = 0;
      for (const r of nonControl) { const v = r.strategies[name][k]; if (v !== null && v <= k) hits++; }
      cells.push(`${hits}/${nonControl.length}`.padStart(9));
    }
    say([name.padEnd(30), ...cells].join(" "));
  }

  const control = rows.find((r) => r.control);
  if (control) {
    say(`\nCONTROL doc ${control.id}: ` + STRATEGIES.slice(0, 3).map(([n]) => `${n.split(": ")[1]}=${control.strategies[n][10]}`).join(" ") +
        `  rrf60=${control.strategies[RRF_UPSTREAM][10]}  rrf10=${control.strategies[RRF_SHIPPED][10]}`);
  }

  // ---- per-case detail for the @10 discriminating cases --------------------
  say("\n=== PER-CASE @10: single-arm rank vs each strategy rank ===");
  say(["doc".padEnd(6), "vec".padStart(5), "lex".padStart(5), "max".padStart(5), "rrf60".padStart(6), "rrf0".padStart(5), "mm".padStart(5), "zmin".padStart(5), "ilv".padStart(5)].join(" "));
  const fmt = (x) => (x === null ? "-" : String(x));
  for (const r of rows) {
    say([
      String(r.id).padEnd(6),
      fmt(r.armAtDepth[10].vector).padStart(5),
      fmt(r.armAtDepth[10].lexical).padStart(5),
      fmt(r.strategies[MAX_OF_TWO][10]).padStart(5),
      fmt(r.strategies[RRF_UPSTREAM][10]).padStart(6),
      fmt(r.strategies["rrf k=0"][10]).padStart(5),
      fmt(r.strategies["minmax sum"][10]).padStart(5),
      fmt(r.strategies["zscore sum missing=min"][10]).padStart(5),
      fmt(r.strategies["interleave v-first"][10]).padStart(5),
    ].join(" ") + (r.control ? "  <- CONTROL" : ""));
  }

  // ---- JSON ---------------------------------------------------------------
  say("\n=== JSON ===");
  say(JSON.stringify({
    index: displayPath(DB_PATH), cases: rows.length, depths: DEPTHS, armDepth: DEEP,
    validation: { membershipChecks: memChecks, membershipMismatch: memMismatch, detail: memDetail },
    table,
    perCase: rows.map((r) => ({ id: r.id, control: r.control, armAtDepth: r.armAtDepth, strategies: r.strategies })),
  }, null, 1));

  mkdirSync(join(HERE, "results"), { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const outPath = join(HERE, "results", `fusion-bakeoff-${stamp}.log`);
  writeFileSync(outPath, log.join("\n") + "\n");
  console.error(`\nwrote ${outPath}`);
}

main().catch((e) => { console.error("BAKEOFF ERROR:", e?.stack || e); process.exit(1); });
