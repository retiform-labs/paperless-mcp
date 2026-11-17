import assert from "node:assert/strict";
import { test } from "node:test";
import { z } from "zod";
import {
  MAX_PAPERLESS_FILTER_VALUES,
  paperlessFiltersSchema,
} from "./documentQuery";

interface ValidationIssue {
  code: string;
  /** A failing union nests one array of issues per branch it tried. */
  errors?: ValidationIssue[][];
}

/**
 * Counts every validation issue reachable from a parse error, including the
 * per-branch issues a failing union nests under its own errors field.
 *
 * WHY the recursion matters: zod reports a union failure as ONE top-level
 * invalid_union issue no matter how many elements failed inside a branch. A
 * flat issues.length therefore reads the same small number whether the array
 * was capped or not, and a test built on it passes while testing nothing.
 * Descending into the branches is what turns the number into a measurement.
 */
function countIssues(issues: ValidationIssue[]): number {
  let total = 0;
  for (const issue of issues) {
    total += 1;
    for (const branch of issue.errors ?? []) {
      total += countIssues(branch);
    }
  }
  return total;
}

function issuesFor(schema: z.ZodType, input: unknown): number {
  const result = schema.safeParse(input);
  assert.equal(result.success, false, "expected the input to be rejected");
  const issues = (result.error as unknown as { issues: ValidationIssue[] })
    .issues;
  return countIssues(issues);
}

/** An array of the given length where every element is the wrong type. */
function wrongTypedArray(length: number): unknown[] {
  return Array.from({ length }, () => ({ not: "a scalar" }));
}

// CWE-770 (GHSA-mrhx-c7xc-m3w5): zod's array validator pushes one issue object
// per failing element with no cap, so an oversized array of wrong-typed values
// is an unbounded-memory path. Measured on zod 4.6.7 against the UNBOUNDED
// union this schema used to declare: 100,000 elements produced 400,005 issues
// in about 293 ms and grew the heap by 114 MB. The cap below turns the same
// input into 7 issues in under a millisecond.
//
// The length check only became a real defence at zod 4.6.6. Before that the
// element loop ran first, so .max() still reported 2,000,001 issues on a
// 2,000,000-element input instead of short-circuiting. This test therefore also
// pins the zod floor: dropping the dependency back to the pre-fix line fails
// here even though the schema text is unchanged.
test("rejects an oversized array without accumulating one issue per element", () => {
  const ELEMENTS = 100_000;

  // POSITIVE CONTROL. The counter this test relies on must be able to report one
  // issue per element, or a small number below proves nothing at all. A bare
  // array walks the same uncapped path zod used to walk for our union, so it
  // measures the upper bound of what this channel can report.
  const controlIssues = issuesFor(z.array(z.string()), wrongTypedArray(ELEMENTS));
  assert.equal(
    controlIssues,
    ELEMENTS,
    "positive control failed: the issue counter cannot report one issue per element, so the bounded assertion below would be vacuous"
  );

  // THE DEFENCE. One filter value holding an __in-style list of wrong-typed
  // elements. A capped validator rejects on length before walking elements.
  const issues = issuesFor(paperlessFiltersSchema, {
    id__in: wrongTypedArray(ELEMENTS),
  });

  assert.ok(
    issues < 100,
    "a " +
      ELEMENTS +
      "-element array produced " +
      issues +
      " validation issues; the array cap is not short-circuiting the element loop (the control channel above reports up to " +
      ELEMENTS +
      ")"
  );

  // The assertion above only means something while the input is genuinely far
  // past the cap. If the cap ever grew past this size the test would silently
  // stop exercising the defence, so the relationship is asserted rather than
  // assumed.
  assert.ok(
    ELEMENTS > MAX_PAPERLESS_FILTER_VALUES * 10,
    "the oversized input must exceed the cap by an order of magnitude for this test to mean anything"
  );
});

// The cap has to reject oversized lists without rejecting legitimate ones.
// Without this, a schema that rejected everything would satisfy the test above.
test("accepts a filter list at the cap and rejects one element past it", () => {
  const atCap = Array.from({ length: MAX_PAPERLESS_FILTER_VALUES }, (_, i) => i);
  assert.equal(
    paperlessFiltersSchema.safeParse({ id__in: atCap }).success,
    true,
    "a list exactly at the cap must be accepted"
  );

  const overCap = [...atCap, MAX_PAPERLESS_FILTER_VALUES];
  assert.equal(
    paperlessFiltersSchema.safeParse({ id__in: overCap }).success,
    false,
    "a list one element past the cap must be rejected"
  );
});

// The scalar alternatives are what make this a union at all. If a bound broke
// one of them the tool would start rejecting ordinary filters, so each shape is
// asserted to still validate.
test("still accepts every scalar filter value shape", () => {
  for (const value of ["a-string", 42, true]) {
    assert.equal(
      paperlessFiltersSchema.safeParse({ some_filter: value }).success,
      true,
      "scalar filter value " + JSON.stringify(value) + " must be accepted"
    );
  }
});

// The string bound must reject rather than silently truncate: an over-long
// filter has to fail loudly here, not get clipped and sent upstream.
test("rejects an over-long string filter value", () => {
  assert.equal(
    paperlessFiltersSchema.safeParse({ title__icontains: "x".repeat(1025) })
      .success,
    false,
    "a filter string past the documented maximum must be rejected"
  );
  assert.equal(
    paperlessFiltersSchema.safeParse({ title__icontains: "x".repeat(1024) })
      .success,
    true,
    "a filter string exactly at the maximum must be accepted"
  );
});
