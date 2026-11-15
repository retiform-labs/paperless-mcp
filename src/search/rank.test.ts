import assert from "node:assert/strict";
import { test } from "node:test";
import { fuse, type Hit } from "./rank";

function hit(documentId: number, position: number, score: number): Hit {
  return {
    documentId,
    position,
    text: `text-${documentId}-${position}`,
    score,
  };
}

// Reciproc rank fusion at k=60: 1/61 = 0.016393442622950821, 1/62 = 1/62.
const K60_RANK1 = 1 / 61;
const K60_RANK2 = 1 / 62;
const K60_RANK3 = 1 / 63;

test("a vector-only hit is retained with matched meaning", () => {
  const result = fuse([hit(1, 0, 0.9)], []);

  assert.equal(result.length, 1);
  assert.equal(result[0].documentId, 1);
  assert.equal(result[0].matched, "meaning");
  assert.equal(result[0].bestSimilarity, 0.9);
  assert.equal(result[0].passage, "text-1-0");
});

test("a document in both lists outranks one in only one", () => {
  // Doc 1 is 2nd by meaning and 1st by words. Doc 2 is 1st by meaning only.
  const vector = [hit(2, 0, 0.95), hit(1, 5, 0.90)];
  const text = [hit(1, 7, 1.2)];

  const result = fuse(vector, text);

  assert.equal(result[0].documentId, 1);
  assert.equal(result[0].matched, "both");
  assert.equal(result[1].documentId, 2);
  assert.equal(result[1].matched, "meaning");
});

test("the arithmetic is the hand-computed k=60 sum", () => {
  // Doc 1: rank 2 in vector, rank 1 in text  -> 1/62 + 1/61
  // Doc 2: rank 1 in vector only             -> 1/61
  const vector = [hit(2, 0, 0.95), hit(1, 5, 0.90)];
  const text = [hit(1, 7, 1.2)];

  const result = fuse(vector, text);

  assert.equal(result.length, 2);
  const doc1 = result.find((r) => r.documentId === 1)!;
  const doc2 = result.find((r) => r.documentId === 2)!;
  assert.equal(doc1.score, K60_RANK2 + K60_RANK1);
  assert.equal(doc2.score, K60_RANK1);
  // And the fused doc must actually be ahead, not merely present.
  assert.ok(doc1.score > doc2.score);
});

test("a single-list hit scores exactly 1/(k+rank) at default k", () => {
  assert.equal(fuse([hit(1, 0, 0.9)], [])[0].score, K60_RANK1);
  assert.equal(fuse([hit(1, 0, 0.9)], [], 10)[0].score, 1 / 11);
});

test("a document with several hits in one list counts only its best rank", () => {
  const result = fuse([hit(1, 0, 0.9), hit(1, 1, 0.8), hit(1, 2, 0.7)], [hit(2, 0, 1.0)]);

  const doc1 = result.find((r) => r.documentId === 1)!;
  assert.equal(doc1.score, K60_RANK1);
});

test("best rank is used on the text side too, not just the vector side", () => {
  // Doc 1 appears 3 times in the TEXT list at ranks 2, 3 and 4, and not at all
  // in vector. Only its best rank (2) counts: 1/62. If the code took the last
  // rank it would score 1/64 instead.
  const text = [hit(9, 0, 1.0), hit(1, 1, 0.9), hit(1, 2, 0.8), hit(1, 3, 0.7)];

  const result = fuse([], text);

  const doc1 = result.find((r) => r.documentId === 1)!;
  assert.equal(doc1.score, K60_RANK2);
  assert.equal(doc1.matched, "words");
});

test("three documents: both beats meaning-only beats words-only", () => {
  // vector: [doc3, doc1]   text: [doc1, doc2]
  // doc1 -> 1/62 + 1/61  (both)
  // doc3 -> 1/61         (meaning)
  // doc2 -> 1/62         (words)
  const vector = [hit(3, 0, 0.9), hit(1, 1, 0.8)];
  const text = [hit(1, 2, 1.1), hit(2, 3, 1.0)];

  const result = fuse(vector, text);

  assert.deepEqual(
    result.map((r) => [r.documentId, r.matched]),
    [
      [1, "both"],
      [3, "meaning"],
      [2, "words"],
    ]
  );
  assert.equal(result[0].score, K60_RANK2 + K60_RANK1);
  assert.equal(result[1].score, K60_RANK1);
  assert.equal(result[2].score, K60_RANK2);
});

test("matched is both only when the document genuinely appears in both lists", () => {
  const result = fuse([hit(1, 0, 0.9), hit(2, 1, 0.8)], [hit(1, 0, 1.0)]);

  const doc1 = result.find((r) => r.documentId === 1)!;
  const doc2 = result.find((r) => r.documentId === 2)!;
  assert.equal(doc1.matched, "both");
  assert.equal(doc2.matched, "meaning");
  assert.ok(result.every((r) => r.matched !== "words"));
});

test("a text-only hit survives with matched words and no similarity", () => {
  const result = fuse([], [hit(3, 0, 1.0)]);

  assert.equal(result.length, 1);
  assert.equal(result[0].documentId, 3);
  assert.equal(result[0].matched, "words");
  assert.equal(result[0].bestSimilarity, null);
  assert.equal(result[0].passage, "text-3-0");
});

test("passage comes from the vector hit when the document has one", () => {
  const result = fuse([hit(1, 0, 0.9)], [hit(1, 1, 1.0)]);
  assert.equal(result[0].passage, "text-1-0");
});

test("passage falls back to the text hit when there is no vector hit", () => {
  const result = fuse([], [hit(1, 4, 1.0)]);
  assert.equal(result[0].passage, "text-1-4");
});

test("bestSimilarity is the vector score, not the text score", () => {
  const result = fuse([hit(1, 0, 0.42)], [hit(1, 1, 99.0)]);
  assert.equal(result[0].bestSimilarity, 0.42);
});

test("both lists empty gives an empty result", () => {
  assert.deepEqual(fuse([], []), []);
});

test("results are sorted strictly descending by score", () => {
  const vector = [hit(1, 0, 0.9), hit(2, 1, 0.8), hit(3, 2, 0.7)];
  const text = [hit(3, 3, 1.0), hit(1, 4, 0.9)];

  const result = fuse(vector, text);
  const scores = result.map((r) => r.score);
  const descending = [...scores].sort((a, b) => b - a);
  assert.deepEqual(scores, descending);
});

test("k is honoured as a parameter across the whole sum", () => {
  const vector = [hit(2, 0, 0.95), hit(1, 5, 0.90)];
  const text = [hit(1, 7, 1.2)];

  const withK = (k: number) => {
    const r = fuse(vector, text, k);
    return {
      doc1: r.find((x) => x.documentId === 1)!.score,
      doc2: r.find((x) => x.documentId === 2)!.score,
    };
  };

  assert.deepEqual(withK(0), { doc1: 1 / 2 + 1 / 1, doc2: 1 / 1 });
  assert.deepEqual(withK(60), { doc1: 1 / 62 + 1 / 61, doc2: 1 / 61 });
});
