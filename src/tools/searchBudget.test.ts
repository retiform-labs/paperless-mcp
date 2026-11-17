// Tests for the P2 OUTPUT BOUND: the two enforcements that keep a tool result
// inside the ceiling the CONSUMER applies.
//
// WHY THIS FILE EXISTS: commit 00a8ba0 added `budgetedLimit`, `fitResultToBudget`
// and the batched metadata path and landed with ZERO tests -- 494 before, 494
// after, 386 new lines. Its own commit message named the gap: "budget arithmetic
// with no test is arithmetic nobody has falsified."
//
// WHY THE NUMBER IS LOAD-BEARING: MCP clients commonly truncate a tool result at
// MAX_RESULT_CHARS by taking head and tail, and the agent cannot detect that it
// happened. A passage silently cut reads exactly like a short one, so the tool
// must bound itself BELOW the ceiling rather than trust the caller to do it.
//
// EVERY TEST HERE WAS WATCHED FAILING against a deliberately broken
// implementation before it was accepted. See the report for each red.

import assert from "node:assert/strict";
import { describe, test } from "node:test";
import {
  MAX_LIMIT,
  MAX_RESULT_CHARS,
  PASSAGE_OVERHEAD_CHARS,
  budgetedLimit,
  fitResultToBudget,
  type SemanticPassage,
  type SemanticSearchResult,
} from "./search";

/** A passage whose serialised form is roughly `window` characters of passage. */
function makePassage(
  id: number,
  opts: { window?: number; phrase?: string } = {}
): SemanticPassage {
  const phrase = opts.phrase ?? `needle${id}`;
  const window = opts.window ?? 800;
  const filler = "x".repeat(Math.max(0, window - phrase.length));
  return {
    id,
    title: `Document ${id}`,
    score: 1 / 61,
    highlights: phrase,
    matched: "both",
    passage: `${phrase}${filler}`,
    offset: 0,
    content_length: window,
    offset_space: "normalized",
  };
}

function makeResult(passages: SemanticPassage[]): SemanticSearchResult {
  return {
    query: "a query",
    semantic: "ok",
    strong_match: passages.length > 0,
    total_hits: passages.length,
    filtered_out: 0,
    retrievers: ["vector", "lexical"],
    passages,
  };
}

const serialised = (value: unknown): number => JSON.stringify(value).length;

describe("budgetedLimit -- the first enforcement, which stops the work", () => {
  test("the WORST CASE the schema actually permits fits inside the ceiling", () => {
    // The schema allows limit <= 50 and window <= 4000. Naive arithmetic gives
    // 50 * 4000 = 200,000 -- ten times the ceiling. This is the case the
    // constant exists for, so it is the case the test uses.
    const naive = MAX_LIMIT * 4000;
    assert.equal(naive, 200_000);
    assert.ok(
      naive > MAX_RESULT_CHARS * 9,
      "precondition: the un-budgeted form must really be ~10x over"
    );

    const limit = budgetedLimit(MAX_LIMIT, 4000);
    const spent = limit * (4000 + PASSAGE_OVERHEAD_CHARS);

    assert.ok(limit >= 1, "a maximal call must still return a passage");
    assert.ok(
      spent <= MAX_RESULT_CHARS,
      `worst case spent ${spent} of ${MAX_RESULT_CHARS} -- it must fit`
    );
    // 18000 / 4200 = 4.28 -> 4. Four passages of 4000 chars.
    assert.equal(limit, 4);
  });

  test("RESULT_BUDGET_FRACTION is actually applied, not just documented", () => {
    // The fraction is deliberately NOT exported, so this asserts its EFFECT:
    // at window 800 the raw ceiling would allow 20000/1000 = 20 passages, and
    // an implementation ignoring the 0.9 would return exactly 20 and spend
    // exactly MAX_RESULT_CHARS. Spending STRICTLY LESS proves the fraction ran.
    const limit = budgetedLimit(MAX_LIMIT, 800);
    const spent = limit * (800 + PASSAGE_OVERHEAD_CHARS);

    assert.equal(limit, 18, "0.9 * 20000 / 1000 floored is 18");
    assert.ok(
      spent < MAX_RESULT_CHARS,
      `spent ${spent} -- equal to the ceiling means the fraction was skipped`
    );
  });

  test("the computed safe counts, against the figures quoted in the plan", () => {
    // ⚠️ THE PLAN QUOTED 20 / 9 / 4 FOR WINDOWS 800 / 2000 / 4000. The
    // implementation gives 18 / 8 / 4, because those plan figures are
    // MAX_RESULT_CHARS / (window + overhead) WITHOUT the 0.9 envelope
    // reserve. The reserve exists for the envelope (query, counts, retriever
    // list, reasons), so the implementation is MORE conservative than the
    // quoted table, not less -- every value is at or below the plan figure.
    // Pinned here so the disagreement is recorded rather than rediscovered.
    const computed = [800, 2000, 4000].map((w) => budgetedLimit(MAX_LIMIT, w));
    assert.deepEqual(computed, [18, 8, 4]);

    const planFigures = [20, 9, 4];
    for (let i = 0; i < computed.length; i += 1) {
      assert.ok(
        computed[i] <= planFigures[i],
        `window ${[800, 2000, 4000][i]}: computed ${computed[i]} exceeds plan ${planFigures[i]}`
      );
    }
  });

  test("a caller asking for fewer passages gets exactly what it asked for", () => {
    assert.equal(budgetedLimit(3, 800), 3);
    assert.equal(budgetedLimit(1, 4000), 1);
  });

  test("never exceeds MAX_LIMIT, whatever the caller asks for", () => {
    // ⚠️ MEASURED, AND IT IS A FINDING ABOUT THE SCHEMA, NOT ABOUT THIS TEST:
    // the schema's own minimum window is 200, which yields a ceiling of
    // floor(18000 / 400) = 45. So no legal call can ever reach MAX_LIMIT (50)
    // and the `MAX_LIMIT` term in the `min(...)` is unreachable in practice.
    // Harmless redundancy rather than a defect -- but it means the honest
    // assertion is the BOUND, not an equality that no legal input produces.
    for (const window of [200, 400, 800, 2000, 4000]) {
      assert.ok(
        budgetedLimit(500, window) <= MAX_LIMIT,
        `window ${window} exceeded MAX_LIMIT`
      );
    }
    // And the tightest legal window really does land below MAX_LIMIT, which is
    // the measurement behind the note above.
    assert.equal(budgetedLimit(500, 200), 45);
    assert.ok(budgetedLimit(500, 200) < MAX_LIMIT);
  });

  test("is never zero, even when the arithmetic leaves no room at all", () => {
    // Documented behaviour: "a call that cannot afford even one passage should
    // still return one and let fitResultToBudget trim". A window large enough
    // to zero the ceiling is the only way to reach it.
    assert.equal(budgetedLimit(10, 100_000), 1);
  });
});

describe("fitResultToBudget -- the second enforcement, which measures real bytes", () => {
  test("TRIMS a result whose serialised body exceeds the ceiling", () => {
    // 30 passages of 800 chars is ~24,000 of passage text alone, comfortably
    // over the 20,000 ceiling. Asserted on the SERIALISED length because that
    // is the byte string that crosses the wire -- a sum of field lengths is a
    // different number and would let a scaffolding-heavy result pass.
    const passages = Array.from({ length: 30 }, (_, i) => makePassage(i + 1));
    const result = makeResult(passages);
    assert.ok(
      serialised(result) > MAX_RESULT_CHARS,
      `precondition: fixture must exceed the ceiling, got ${serialised(result)}`
    );

    const trimmed = fitResultToBudget(result);

    assert.ok(
      serialised(trimmed) <= MAX_RESULT_CHARS,
      `trimmed result is ${serialised(trimmed)}, still over ${MAX_RESULT_CHARS}`
    );
    assert.ok(
      trimmed.passages.length < passages.length,
      "it must have dropped at least one passage"
    );
    // Still valid JSON with the required fields intact.
    const round = JSON.parse(JSON.stringify(trimmed)) as SemanticSearchResult;
    for (const key of [
      "query",
      "semantic",
      "strong_match",
      "total_hits",
      "filtered_out",
      "retrievers",
      "passages",
    ]) {
      assert.ok(key in round, `trimmed result lost ${key}`);
    }
    assert.equal(round.query, "a query");
    assert.deepEqual(round.retrievers, ["vector", "lexical"]);
  });

  test("reports WHAT IT DROPPED, rather than trimming silently", () => {
    // The whole reason `trimmed` exists: a caller that asked for 30 and got 6
    // cannot otherwise tell whether the corpus held 6 or 24 were removed.
    const passages = Array.from({ length: 30 }, (_, i) => makePassage(i + 1));
    const result = makeResult(passages);
    const trimmed = fitResultToBudget(result);

    assert.ok(trimmed.trimmed, "a trimmed result must say so");
    assert.equal(trimmed.trimmed.reason, "result_size");
    assert.equal(trimmed.trimmed.limit_chars, MAX_RESULT_CHARS);
    assert.equal(
      trimmed.trimmed.dropped,
      passages.length - trimmed.passages.length,
      "the reported drop count must equal the real difference"
    );
  });

  test("KEEPS THE HEAD -- it pops the tail, never the best passage", () => {
    // Passages arrive ranked. Trimming from the front would discard the best
    // hit while keeping a worse one, which is the failure mode that makes a
    // bound actively harmful.
    const passages = Array.from({ length: 30 }, (_, i) => makePassage(i + 1));
    const trimmed = fitResultToBudget(makeResult(passages));

    assert.equal(trimmed.passages[0].id, 1, "the top-ranked passage must survive");
    // And the survivors are a PREFIX of the input, in order.
    assert.deepEqual(
      trimmed.passages.map((p) => p.id),
      passages.slice(0, trimmed.passages.length).map((p) => p.id)
    );
  });

  test("IS A NO-OP WHEN THE RESULT ALREADY FITS", () => {
    // The test that catches an over-eager trimmer. A function that always
    // trims is exactly as wrong as one that never does, and it is harder to
    // notice because the output still validates.
    const passages = Array.from({ length: 3 }, (_, i) => makePassage(i + 1));
    const result = makeResult(passages);
    assert.ok(serialised(result) <= MAX_RESULT_CHARS, "precondition: fixture fits");

    const out = fitResultToBudget(result);

    assert.equal(out.passages.length, 3, "nothing may be dropped from a result that fits");
    assert.deepEqual(
      out.passages.map((p) => p.id),
      [1, 2, 3],
      "order must be untouched"
    );
    assert.equal(out.trimmed, undefined, "a no-op must not claim it trimmed");
    // Identity, not just equality: a no-op should return the object it was given.
    assert.equal(out, result, "the untouched result must be the same object");
  });

  test("an EMPTY passages list is returned untouched", () => {
    const result = makeResult([]);
    const out = fitResultToBudget(result);
    assert.equal(out.passages.length, 0);
    assert.equal(out.trimmed, undefined);
    assert.equal(out, result);
  });

  test("THE CONTAINMENT INVARIANT SURVIVES TRIMMING", () => {
    // A live discipline in this repo: the returned passage must contain the
    // phrase its `highlights` quotes. A past defect returned a passage that did
    // not contain the phrase it highlighted. Trimming drops whole passages, so
    // it must not be able to break this -- asserted rather than assumed.
    const passages = Array.from({ length: 30 }, (_, i) =>
      makePassage(i + 1, { phrase: `needle${i + 1}` })
    );
    const trimmed = fitResultToBudget(makeResult(passages));

    assert.ok(trimmed.passages.length > 0, "fixture must survive with some passages");
    for (const passage of trimmed.passages) {
      assert.ok(passage.highlights, `passage ${passage.id} lost its highlight`);
      assert.ok(
        passage.passage.toLowerCase().includes(passage.highlights.toLowerCase()),
        `passage ${passage.id} does not contain its own highlighted phrase "${passage.highlights}"`
      );
    }
  });

  test("a SINGLE oversized passage is still refused rather than shipped whole", () => {
    // The documented terminal case: the loop pops until empty. If one passage
    // alone exceeds the ceiling, the honest outcome is an empty list carrying
    // `trimmed`, not a 30,000-character payload the client will mangle.
    const huge = makePassage(1, { window: 60_000 });
    const result = makeResult([huge]);
    assert.ok(serialised(result) > MAX_RESULT_CHARS, "precondition");

    const trimmed = fitResultToBudget(result);

    assert.equal(trimmed.passages.length, 0);
    assert.ok(trimmed.trimmed, "it must report that it dropped everything");
    assert.equal(trimmed.trimmed.dropped, 1);
  });
});
