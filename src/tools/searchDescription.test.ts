import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";

// __dirname, not import.meta: this repo compiles to CommonJS and tsconfig has
// no ESM module target, so import.meta is a type error here.
const root = join(__dirname, "..", "..");
const searchSrc = readFileSync(join(root, "src/tools/search.ts"), "utf8");
const readme = readFileSync(join(root, "README.md"), "utf8");
const docsSrc = readFileSync(join(root, "src/tools/documents.ts"), "utf8");

// The tool description IS the contract a client receives -- it ships in
// tools/list and an agent writes control flow against it. The defect this
// pins: `strong_match`'s description promised "false when nothing scored
// above the similarity floor" long after the code had been reduced to
// `passages.length > 0`, so the false branch was unreachable and a caller
// would have written `if (!strong_match) requery()` against a lie.
//
// A CODE test could not catch this: the code was correct. Only a text
// assertion catches a stale contract.
test("no surface still promises the 0.45 similarity floor for strong_match", () => {
  const dead = [
    /strong_match is false when nothing scored above/i,
    /cleared the similarity floor \(0\.45\)/i,
    /A `false` here means everything returned is weak/i,
  ];
  for (const [name, text] of [["search.ts", searchSrc], ["README.md", readme]] as const) {
    for (const re of dead) {
      assert.ok(!re.test(text), `${name} still carries the retired claim: ${re}`);
    }
  }
});

test("the honest description states what strong_match actually means", () => {
  assert.match(
    searchSrc,
    /strong_match is true exactly when at least one passage was returned/,
    "the live tool description must say the real condition"
  );
  assert.match(
    readme,
    /coarse "did anything come back" flag/,
    "the README must not describe a quality judgement it cannot make"
  );
});

// The same class, one field over: `highlights` was documented as "always
// absent on this path" after P2 started returning it.
test("the README does not claim highlights is always absent", () => {
  assert.ok(
    !/highlights` is always absent/i.test(readme),
    "P2 added highlights to this path; the docs must not say otherwise"
  );
});

// And the anchor: documented as "the first verbatim query term" both before
// and after the fix that changed it, which is how a stale doc survives a
// behaviour change.
//
// ⚠️ THIS TEST WAS ITSELF THE FAILURE, ONCE. The first version checked the
// README only, and the re-gate found `search.ts`'s OWN JSDoc still saying
// "first verbatim query term" -- a doc correct in the copy a CLIENT reads and
// wrong in the copy a MAINTAINER reads. A pin test that checks one surface is
// a pin test that confirms the surface you happened to open.
// ⚠️ THIS TEST OUTLIVED ITS MECHANISM, TWICE, AND THE SECOND TIME IT WAS PINNING
// A FALSEHOOD. Version 1 checked the README only and missed `search.ts`'s own
// JSDoc. Version 2 checked all three surfaces and demanded the word "longest" --
// correct while length was the heuristic, and WRONG the moment `5cb3492` moved
// the anchor to RARITY. It stayed green by certifying a claim the code no longer
// made, which is worse than being red: a stale doc the suite actively endorses.
//
// The durable form is to assert the PROPERTY the code actually implements, and
// to forbid the retired wordings -- so a future heuristic change fails this test
// instead of being silently blessed by it.
test("each path describes the anchor it actually implements, and neither describes query order", () => {
  // Query order is retired EVERYWHERE -- it anchored English queries on "the".
  // Length and rarity, by contrast, are BOTH live, on different tools: this is
  // not one heuristic documented twice, and collapsing them would "fix" one
  // correct surface into a false one. (It did: I rewrote README's search_passages
  // paragraph to say rarity, and search_passages still walks longest-first
  // because it has no index access and never will.)
  for (const [name, text] of [["search.ts", searchSrc], ["documents.ts", docsSrc], ["README.md", readme]] as const) {
    assert.ok(
      !/first verbatim (query )?term/i.test(text),
      `${name} still describes the retired query-order anchor`
    );
  }
  // semantic_search: rarity, because it holds the index.
  assert.match(searchSrc, /rarityOf|termRarity/);
  assert.match(readme, /rarest matching query term/i);
  // search_passages: length, because it does not -- and it must SAY so.
  assert.match(docsSrc, /sort\(\(a, b\) => b\.length - a\.length\)/);
  assert.match(readme, /longest matching query term/i);
  // And the divergence must be disclosed, not left for a reader to discover.
  assert.match(readme, /rank their fallback anchor differently, on purpose/i);
});


// N1 from the re-gate: `trimmed` was emitted but documented NOWHERE, so a
// caller could receive a shortened query with no way to know the contract.
// F11's class exactly, one field over -- a field the code returns and the docs
// do not describe.
test("the README documents the trimmed contract it can receive", () => {
  assert.match(readme, /\*\*`trimmed`\*\* is present/, "trimmed must be documented");
  for (const key of ["reason", "dropped", "limit_chars", "query_truncated"]) {
    assert.match(readme, new RegExp(key), `README must name trimmed.${key}`);
  }
});

// N4: the example return shape omitted `highlights` (emitted when anchored)
// and `offset_space` (emitted unconditionally).
test("the semantic_search example shows the fields the code actually returns", () => {
  const example = readme.slice(readme.indexOf("semantic_search({ query"), readme.indexOf("**Return shape.**"));
  for (const field of ["highlights", "offset_space"]) {
    assert.match(example, new RegExp(field), `the example shape must show ${field}`);
  }
});

// The live corpus caught THIS, and no unit test could have: I wrote the example
// shape with `offset_space` at the TOP level, while the code puts it on every
// PASSAGE. A doc-vs-code drift I introduced myself while fixing another one --
// which is the whole argument for testing the docs rather than rereading them.
test("the example puts offset_space on the passage, not on the result", () => {
  const example = readme.slice(readme.indexOf("semantic_search({ query"), readme.indexOf("**Return shape.**"));
  const passagesLine = example.slice(example.indexOf("passages: ["));
  assert.match(passagesLine, /offset_space/, "offset_space belongs inside passages[]");
  assert.ok(
    !/total_hits: \d+,\s*offset_space/.test(example),
    "offset_space is per-passage; the top-level result does not carry it"
  );
});

// And it must survive as a real per-passage field in the type, not merely in
// prose -- the README example is derived from the interface, so if the field
// ever moves the example is wrong again.
test("SemanticPassage declares offset_space and the result type does not", () => {
  const passageIface = searchSrc.slice(searchSrc.indexOf("export interface SemanticPassage"), searchSrc.indexOf("export interface SemanticSearchResult"));
  assert.match(passageIface, /offset_space/, "SemanticPassage must declare it");
  const resultIface = searchSrc.slice(searchSrc.indexOf("export interface SemanticSearchResult"), searchSrc.indexOf("export interface SemanticSearchResult") + 3000);
  assert.ok(!/^\s*offset_space/m.test(resultIface), "SemanticSearchResult must not declare it");
});
