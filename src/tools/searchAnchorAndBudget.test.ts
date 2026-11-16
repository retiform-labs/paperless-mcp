import { test } from "node:test";
import assert from "node:assert/strict";
import { buildPassage, fitResultToBudget, MAX_RESULT_CHARS } from "./search";

// ---------------------------------------------------------------------------
// THE ANCHOR PROPERTY. The defect these exist for was measured, not imagined:
// `buildPassage` used to anchor on the FIRST query term found anywhere, and
// English queries open with articles, so "the practice of a wrathful deity with
// a lion head" anchored on "the" at character 3. Eleven of eighteen windows
// landed at the document head. Containment still passed -- "the" IS in the
// window -- which is precisely why the old suite was green. The property that
// holds was the weak one; the one that matters was never asserted.
// ---------------------------------------------------------------------------

const GARBAGE = "The quick brown fox jumps over the lazy dog. ".repeat(400);

test("anchor skips a leading stopword and centres on the distinctive term", () => {
  const marker = "Sinhamukha appears here";
  const content = GARBAGE + marker + GARBAGE;

  // Right-to-left in specificity, left-to-right in the query: the query leads
  // with "the", which occurs at character 3.
  const { offset, term, passage } = buildPassage(
    { id: 1, title: "t", content },
    ["the", "practice", "wrathful", "deity", "lion", "sinhamukha"],
    800
  );

  assert.equal(term, "sinhamukha", "must anchor on the rarest/longest matching term");
  assert.ok(
    offset > 1000,
    `offset ${offset} is at the document head -- it anchored on a stopword`
  );
  assert.ok(passage.includes("Sinhamukha"), "the window must contain the match");
});

// ---------------------------------------------------------------------------
// RARITY BEATS LENGTH, AND THIS IS THE CASE THAT PROVED IT ON THE LIVE CORPUS.
//
// My first correction to the stopword defect ranked terms by LENGTH. Measured
// against the real index, it chose `doctor` (1,903 chunks) over `ankle` (808)
// because "doctor" is one character longer -- so the window landed on a
// discussion of doctors instead of the ankle the user asked about. Rarity
// agreed with a human read in 3 of 3 live probes; length in 0 of 3.
//
// The test below is the same shape in miniature: a LONGER but COMMON term and a
// shorter but RARE one, both present.
// ---------------------------------------------------------------------------
test("a rare term beats a longer common one when counts are supplied", () => {
  const content = GARBAGE + "the ankle" + GARBAGE + "doctor" + GARBAGE;
  const rarity = new Map([
    ["doctor", 1903],   // longer, but common
    ["ankle", 808],     // shorter, but rare -> must win
    ["the", 296835],
    ["about", 63960],
  ]);
  const { term } = buildPassage(
    { id: 9, title: "t", content },
    ["the", "about", "doctor", "ankle"],
    400,
    rarity
  );
  assert.equal(term, "ankle", "rarity must outrank length");
});

test("without counts it still improves on query order via length", () => {
  const content = GARBAGE + "naropa" + GARBAGE;
  const { term } = buildPassage(
    { id: 10, title: "t", content },
    ["the", "six", "yogas", "naropa"],
    400
  );
  assert.equal(term, "naropa", "the degraded path must still skip stopwords");
});

test("an unscored term does not outrank a scored one", () => {
  // A term the caller could not score sorts last, not first.
  const content = "zzzqqq" + GARBAGE + "wombat" + GARBAGE;
  const rarity = new Map([["wombat", 412]]);
  const { term } = buildPassage(
    { id: 11, title: "t", content },
    ["zzzqqq", "wombat"],
    400,
    rarity
  );
  assert.equal(term, "wombat");
});

test("a word that is absent never wins the anchor", () => {
  const { term } = buildPassage(
    { id: 2, title: "t", content: GARBAGE + "naropa" + GARBAGE },
    ["the", "six", "yogas", "naropa"],
    400
  );
  assert.equal(term, "naropa");
});

test("no match at all falls back to the head and says nothing matched", () => {
  const { offset, term } = buildPassage(
    { id: 3, title: "t", content: GARBAGE },
    ["zzzznotpresent"],
    200
  );
  assert.equal(offset, 0, "an honest head window when the query appears nowhere");
  assert.equal(term, undefined, "no term is claimed when none matched");
});

test("the window is WIDENED when needed to contain the anchoring term", () => {
  // A term longer than the window: the plain start+window would truncate it,
  // and a window that stops one character short of its own hit is the defect
  // the containment rule forbids.
  const long = "abcdefghijklmnopqrstuvwxyz";
  const { passage, term } = buildPassage(
    { id: 4, title: "t", content: "x".repeat(50) + long },
    [long],
    10
  );
  assert.equal(term, long);
  assert.ok(passage.includes(long), "the anchoring term must survive inside the window");
});

// ---------------------------------------------------------------------------
// THE BUDGET IS A BOUND ON THE WHOLE RESULT, NOT ON PASSAGES.
// Measured defect: `fitResultToBudget` dropped passages and never touched the
// ECHOED QUERY, which has no length limit. A 30,000-char query returned 30,128
// chars with zero passages -- over the very ceiling the function exists for.
// Dropping passages cannot fix an oversize field that is not a passage.
// ---------------------------------------------------------------------------

function result(query: string, passageText: string, n = 5) {
  return {
    query,
    semantic: "ok" as const,
    strong_match: true,
    total_hits: n,
    filtered_out: 0,
    retrievers: ["vector", "lexical"] as ("vector" | "lexical")[],
    passages: Array.from({ length: n }, (_, i) => ({
      id: i + 1,
      title: `doc ${i}`,
      score: 0.03,
      passage: passageText,
      offset: 0,
      content_length: passageText.length,
    })),
  } as any;
}

test("an undersized result is returned UNCHANGED, not rewritten", () => {
  const r = result("goats", "short passage", 2);
  const out = fitResultToBudget(r);
  assert.equal(out, r, "identity: a result that fits must not be reconstructed");
  assert.equal(out.trimmed, undefined);
});

test("an oversized result is trimmed and SAYS SO", () => {
  const r = result("goats", "y".repeat(3000), 20);
  const out = fitResultToBudget(r as any);
  assert.ok(JSON.stringify(out).length <= MAX_RESULT_CHARS, "must fit the ceiling");
  assert.ok(out.passages.length < 20, "something must have been dropped");
  assert.equal(out.trimmed?.reason, "result_size");
  assert.ok((out.trimmed?.dropped ?? 0) > 0);
});

test("a HUGE QUERY alone cannot blow the ceiling -- the echo is truncated", () => {
  // No passages to drop that could fix this: the oversize part is the query.
  const r = result("q".repeat(30_000), "tiny", 0);
  const out = fitResultToBudget(r as any);
  assert.ok(
    JSON.stringify(out).length <= MAX_RESULT_CHARS,
    `returned ${JSON.stringify(out).length} chars, ceiling is ${MAX_RESULT_CHARS}`
  );
  assert.equal(out.trimmed?.query_truncated, true, "truncation must be declared");
  assert.ok(
    (out.query ?? "").length < 30_000,
    "the echoed query must actually be shorter"
  );
});
