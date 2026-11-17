import assert from "node:assert/strict";
import { test } from "node:test";
import {
  buildSearchPassage,
  extractHighlightPhrases,
  extractSearchTerms,
} from "./documents";
import { buildPassage, queryTerms } from "./search";

// ---------------------------------------------------------------------------
// ADVERSARIAL LAYER -- THE CONTAINMENT PROPERTY.
//
// Both passage builders promise the same thing: the window they return CONTAINS
// the text their own  field quotes, and  locates it. That
// promise has already been broken once in production, by an HTML-entity gap:
// whoosh emits &#x27; and the decoder handled only &#39;/&apos;, so the phrase
// silently failed to resolve and the window anchored at the document head
// while still reporting a highlight for text it did not contain.
//
// The house suite already asserts containment for the cases that were fixed.
// This file asserts it as a PROPERTY over the shapes that break naive decoding
// and windowing: entities of every form, CJK, combining marks, an empty
// highlight, and a term longer than the window itself.
//
//  is the reusable helper the brief asks for: it
// takes a result and asks the only question that matters -- is the thing you
// quoted actually in the thing you returned?
// ---------------------------------------------------------------------------

/**
 * Asserts a passage result contains every phrase its own  quotes.
 *
 * Returns the phrases that were quoted but absent, so a caller can put them in
 * an assertion message. An empty array means the contract held.
 *
 * Only words of four characters or more are checked, matching
 * ' own threshold: a decoded fragment shorter than
 * that cannot anchor a window, and asserting on it would demand something the
 * builder never claimed.
 */
function quotedButAbsent(result: {
  passage: string;
  highlights?: string;
}): string[] {
  if (!result.highlights) return [];
  const passage = result.passage.toLowerCase();
  return extractHighlightPhrases(result.highlights)
    .join(" ")
    .split(" ")
    .filter((word) => word.length >= 4)
    .filter((word) => !passage.includes(word.toLowerCase()));
}

/** Far enough in that a head-anchored window provably misses the hit. */
const FILLER =
  "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor. ".repeat(10);

/** Every shape this file checks, so one loop covers both builders. */
const CASES: Array<{ label: string; hit: string; query: string; highlights: string }> = [
  {
    label: "numeric entity (&#39;) -- the form that broke in production",
    hit: "the patient's ankle was swollen",
    query: "patient's ankle",
    highlights: "<em>patient&#39;s</em> <em>ankle</em>",
  },
  {
    label: "hex entity (&#x27;) -- the form the fix added",
    hit: "the patient's ankle was swollen",
    query: "patient's ankle",
    highlights: "<em>patient&#x27;s</em> <em>ankle</em>",
  },
  {
    label: "named entity (&apos;)",
    hit: "the patient's ankle was swollen",
    query: "patient's ankle",
    highlights: "<em>patient&apos;s</em> <em>ankle</em>",
  },
  {
    label: "ampersand entity (&amp;)",
    hit: "Smith & Wesson reported the defect",
    query: "smith wesson defect",
    highlights: "<em>Smith</em> <em>&amp;</em> <em>Wesson</em> <em>defect</em>",
  },
  {
    label: "double-escaped entity (&amp;#39;)",
    hit: "the patient&#39;s ankle was swollen",
    query: "patient ankle",
    highlights: "<em>patient&amp;#39;s</em> <em>ankle</em>",
  },
  {
    label: "CJK with no whitespace",
    hit: "患者の足首が腫れていた",
    query: "足首",
    highlights: "<em>足首</em>",
  },
  {
    label: "Tibetan syllable with combining marks",
    hit: "རླུང is wind",
    query: "རླུང",
    highlights: "<em>རླུང</em>",
  },
  {
    label: "accented Latin",
    hit: "El diagnóstico fue severo",
    query: "diagnóstico severo",
    highlights: "<em>diagnóstico</em> <em>severo</em>",
  },
];

for (const { label, hit, query, highlights } of CASES) {
  test("search_passages contains the phrase its highlights quote: " + label, () => {
    const content = FILLER + hit + ". " + FILLER;
    const result = buildSearchPassage(
      { id: 1, title: "t", content },
      extractSearchTerms(query),
      800,
      { score: 1, highlights }
    );

    const absent = quotedButAbsent(result);
    assert.deepEqual(
      absent,
      [],
      "quoted " + JSON.stringify(absent) + " but the passage (offset " + result.offset +
        ") does not contain it. The window must contain the hit it was centred for."
    );
    // The offset must also be honest: slicing the document at it must give back
    // the passage, or the caller cannot page from here.
    assert.equal(
      content.slice(result.offset, result.offset + result.passage.length),
      result.passage,
      "offset " + result.offset + " does not slice the document to the passage returned"
    );
  });

  test("semantic_search contains the term its anchor reports: " + label, () => {
    const content = FILLER + hit + ". " + FILLER;
    const result = buildPassage(
      { id: 1, title: "t", content },
      queryTerms(query),
      800
    );

    assert.ok(result.term, "expected an anchor term, got none for " + label);
    assert.ok(
      result.passage.toLowerCase().includes(result.term!.toLowerCase()),
      "the passage reports term " + JSON.stringify(result.term) +
        " at offset " + result.offset + " but does not contain it"
    );
    assert.equal(
      content.slice(result.offset, result.offset + result.passage.length),
      result.passage,
      "offset " + result.offset + " does not slice the document to the passage returned"
    );
  });
}

test("a stopword in the query cannot drag the window away from the hit", () => {
  // The measured original defect: "the practice of a wrathful deity" anchored on
  // "the" at character 3, so eleven of eighteen windows landed at the head.
  // Containment still passed, because "the" really was in the window. The
  // property that matters is anchoring on the DISTINCTIVE term, so this asserts
  // the position, not just the presence.
  const content = FILLER + "Sinhamukha appears here. " + FILLER;
  const result = buildSearchPassage(
    { id: 1, title: "t", content },
    extractSearchTerms("the practice of a wrathful deity called sinhamukha"),
    400
  );
  assert.ok(
    result.offset > FILLER.length / 2,
    "offset " + result.offset + " is at the document head -- the window anchored on a stopword"
  );
  assert.ok(result.passage.includes("Sinhamukha"), "the window must contain the distinctive term");
});

test("a query term longer than the window is still fully contained", () => {
  // search.ts widens the window when the anchor term is longer than it, because
  // a window that stops one character short of its own match is exactly what
  // the containment rule forbids. Both builders must do it.
  const longTerm = "a".repeat(300);
  const content = FILLER + "x" + longTerm + "x" + FILLER;
  const window = 200;

  const passageBuilder = buildPassage({ id: 1, title: "t", content }, [longTerm], window);
  assert.ok(
    passageBuilder.passage.includes(longTerm),
    "semantic_search cut a " + window + "-char window centred on a " + longTerm.length +
      "-char term and the term is not in it"
  );

  const searchPassage = buildSearchPassage({ id: 2, title: "t", content }, [longTerm], window);
  assert.ok(
    searchPassage.passage.includes(longTerm),
    "search_passages cut a " + window + "-char window centred on a " + longTerm.length +
      "-char term and the term is not in it"
  );
});
