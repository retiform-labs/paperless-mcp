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

  assert.equal(term, "sinhamukha", "must anchor on the longest matching term");
  assert.ok(
    offset > 1000,
    `offset ${offset} is at the document head -- it anchored on a stopword`
  );
  assert.ok(passage.includes("Sinhamukha"), "the window must contain the match");
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
