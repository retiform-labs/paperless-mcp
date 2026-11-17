#!/usr/bin/env node
// Post-hoc analysis over a completed bakeoff log's JSON block.
// Reads MEASURED per-case data; computes nothing that requires the index.
// Why a separate script: the arm UNION's membership is the real ceiling of a
// two-arm system, and it is not the same number as the best ordering of that
// union. An ordering that scores 18/20 when membership allows 19/20 is leaving
// a case on the floor, and only this comparison shows it.
import { readFileSync } from "node:fs";

const path = process.argv[2];
if (!path) { console.error("usage: analyze.mjs <logfile>"); process.exit(1); }
const raw = readFileSync(path, "utf8");
const idx = raw.indexOf("=== JSON ===");
if (idx === -1) { console.error("no JSON block found in " + path); process.exit(1); }
const body = raw.slice(idx + "=== JSON ===".length).trim();
const end = body.lastIndexOf("}");
const data = JSON.parse(body.slice(0, end + 1));

const DEPTHS = data.depths;
const rows = data.perCase;
const nonControl = rows.filter((r) => !r.control);

function hits(pred, subset = rows) {
  let n = 0;
  for (const r of subset) if (pred(r)) n++;
  return n;
}

console.log(`\n=== ARM UNION vs BEST ORDERING ===`);
console.log(`arm depth used for every strategy: ${data.armDepth}`);
console.log(["", ...DEPTHS.map((k) => `@${k}`.padStart(10))].join(" "));
for (const k of DEPTHS) { /* header only */ }
const unionRow = [], maxRow = [];
for (const k of DEPTHS) {
  const u = hits((r) => {
    const v = r.armAtDepth[k].vector, l = r.armAtDepth[k].lexical;
    return (v !== null && v <= k) || (l !== null && l <= k);
  });
  const m = hits((r) => {
    const v = r.strategies["single: max-of-two (floor)"][k];
    return v !== null && v <= k;
  });
  unionRow.push(u); maxRow.push(m);
}
console.log(["UNION MEMBERSHIP (ceiling)".padEnd(30), ...unionRow.map((n) => `${n}/${rows.length}`.padStart(10))].join(" "));
console.log(["max-of-two ORDERING".padEnd(30), ...maxRow.map((n) => `${n}/${rows.length}`.padStart(10))].join(" "));

// Which documents are recoverable but unranked?
for (const k of DEPTHS) {
  const lost = [];
  for (const r of rows) {
    const v = r.armAtDepth[k].vector, l = r.armAtDepth[k].lexical;
    const inUnion = (v !== null && v <= k) || (l !== null && l <= k);
    const mv = r.strategies["single: max-of-two (floor)"][k];
    const inMax = mv !== null && mv <= k;
    if (inUnion && !inMax) lost.push(`doc ${r.id} (vec=${v === null ? "-" : v} lex=${l === null ? "-" : l} -> max=${mv})`);
  }
  if (lost.length) console.log(`  @${k} recoverable-but-unranked: ${lost.join(", ")}`);
}

// Pure-arm-only reachability: documents NO ordering can surface at @10
for (const k of DEPTHS) {
  const lost = rows.filter((r) => {
    const v = r.armAtDepth[k].vector, l = r.armAtDepth[k].lexical;
    return !((v !== null && v <= k) || (l !== null && l <= k));
  });
  console.log(`  @${k} unreachable by BOTH arms: ${lost.length ? lost.map((r) => "doc " + r.id).join(", ") : "none"}`);
}

// n=19 versions
console.log(`\n--- excluding control (n=${nonControl.length}) ---`);
const u19 = [], m19 = [];
for (const k of DEPTHS) {
  u19.push(hits((r) => {
    const v = r.armAtDepth[k].vector, l = r.armAtDepth[k].lexical;
    return (v !== null && v <= k) || (l !== null && l <= k);
  }, nonControl));
  m19.push(hits((r) => {
    const v = r.strategies["single: max-of-two (floor)"][k];
    return v !== null && v <= k;
  }, nonControl));
}
console.log(["UNION MEMBERSHIP".padEnd(30), ...u19.map((n) => `${n}/${nonControl.length}`.padStart(10))].join(" "));
console.log(["max-of-two ORDERING".padEnd(30), ...m19.map((n) => `${n}/${nonControl.length}`.padStart(10))].join(" "));

// How many distinct orderings do all the strategies actually produce?
console.log(`\n=== DISTINCT ORDERINGS @10 (do strategies differ at all?) ===`);
const sig = new Map();
for (const [name, rec] of Object.entries(data.table)) {
  const keys = rows.map((r) => r.strategies[name][10] ?? "-").join(",");
  if (!sig.has(keys)) sig.set(keys, []);
  sig.get(keys).push(name);
}
let g = 0;
for (const [, names] of sig) {
  g++;
  console.log(`  group ${g} (${names.length}): ${names.join(" | ")}`);
}
console.log(`  => ${g} distinct @10 orderings across ${Object.keys(data.table).length} strategies`);

// ---------------------------------------------------------------------------
// HELPED/HURT vs THE BEST SINGLE ARM (the brief's exact question).
// The bakeoff's own +N/-N column measures against the UNION of the two arms,
// which is the ceiling and is always 0-helped by construction -- no fusion can
// retrieve a document neither retriever returned. The question asked was
// "versus the best single arm", and the best single arm at @10 is lexical
// (16/20). These are different baselines and only one of them is the spec.
// ---------------------------------------------------------------------------
console.log(`\n=== HELPED / HURT vs THE BEST SINGLE ARM ===`);
for (const k of [10, 50]) {
  const bestArmKey = (() => {
    let best = null, bestN = -1;
    for (const side of ["vector", "lexical"]) {
      const n = hits((r) => { const v = r.armAtDepth[k][side]; return v !== null && v <= k; });
      if (n > bestN) { bestN = n; best = side; }
    }
    return { side: best, n: bestN };
  })();
  console.log(`\n@${k}  best single arm = ${bestArmKey.side} (${bestArmKey.n}/${rows.length})`);
  const lines = [];
  for (const [name] of Object.entries(data.table)) {
    let helped = 0, hurt = 0;
    const hIds = [], uIds = [];
    for (const r of rows) {
      const av = r.armAtDepth[k][bestArmKey.side];
      const armHit = av !== null && av <= k;
      const sv = r.strategies[name][k];
      const fHit = sv !== null && sv <= k;
      if (fHit && !armHit) { helped++; hIds.push(r.id); }
      if (!fHit && armHit) { hurt++; uIds.push(r.id); }
    }
    const total = hits((r) => { const sv = r.strategies[name][k]; return sv !== null && sv <= k; });
    lines.push({ name, total, helped, hurt, hIds, uIds, net: total - bestArmKey.n });
  }
  lines.sort((a, b) => (b.total - a.total) || (a.hurt - b.hurt));
  console.log(["strategy".padEnd(30), "recall".padStart(8), "Δ".padStart(4), "help".padStart(5), "hurt".padStart(5), "  helped docs / hurt docs"].join(" "));
  for (const l of lines) {
    console.log([
      l.name.padEnd(30),
      `${l.total}/${rows.length}`.padStart(8),
      (l.net >= 0 ? "+" + l.net : String(l.net)).padStart(4),
      String(l.helped).padStart(5),
      String(l.hurt).padStart(5),
      "  " + (l.hIds.length ? "help:" + l.hIds.join(",") : "") + (l.uIds.length ? " hurt:" + l.uIds.join(",") : ""),
    ].join(" "));
  }
}

// Ties: which strategies produce the SAME recall vector across all depths?
console.log(`\n=== STRATEGIES INDISTINGUISHABLE BY RECALL (all depths) ===`);
const byRec = new Map();
for (const [name, rec] of Object.entries(data.table)) {
  const key = DEPTHS.map((k) => rec.recall[k]).join("/");
  if (!byRec.has(key)) byRec.set(key, []);
  byRec.get(key).push(name);
}
for (const [k, names] of [...byRec.entries()].sort((a, b) => b[1].length - a[1].length)) {
  console.log(`  ${k.padEnd(12)} (${names.length}): ${names.join(" | ")}`);
}
