// Reciprocal rank fusion of a vector (meaning) list and a text (words) list.
//
// Ported verbatim from jaimemachado/paperless-mcp (src/search/rank.ts at
// 50f7beec0d36804cab8fbcaa7b47f85e562e8061), with one structural change: their
// only import was `Hit` from ./store, and store.ts does not exist yet in this
// branch. Hit is declared and exported here instead, so this module compiles
// and tests on its own. The integration lap should either re-point this to
// ./store or keep this as the single definition -- see the port report.
//
// A document's score is the sum of 1/(k + bestRank) over each list it appears
// in, using its BEST rank in that list. A document found only by meaning (the
// vector list) must stay in the results -- that is the whole point of fusing
// rather than intersecting.

export interface Hit {
  documentId: number;
  position: number;
  text: string;
  score: number;
}

export interface Fused {
  documentId: number;
  score: number;
  passage: string;
  matched: "meaning" | "words" | "both";
  bestSimilarity: number | null;
}

export function fuse(vector: Hit[], text: Hit[], k = 60): Fused[] {
  const byDocument = new Map<
    number,
    { score: number; vector?: Hit; text?: Hit }
  >();

  function addRanks(hits: Hit[], side: "vector" | "text") {
    const bestRank = new Map<number, number>();
    hits.forEach((hit, index) => {
      const rank = index + 1;
      if (!bestRank.has(hit.documentId)) bestRank.set(hit.documentId, rank);
    });
    for (const hit of hits) {
      const entry = byDocument.get(hit.documentId) ?? { score: 0 };
      byDocument.set(hit.documentId, entry);
      if (side === "vector" && !entry.vector) entry.vector = hit;
      if (side === "text" && !entry.text) entry.text = hit;
    }
    for (const [documentId, rank] of bestRank) {
      const entry = byDocument.get(documentId)!;
      entry.score += 1 / (k + rank);
    }
  }

  addRanks(vector, "vector");
  addRanks(text, "text");

  return [...byDocument.entries()]
    .map(([documentId, entry]) => {
      const matched: Fused["matched"] =
        entry.vector && entry.text
          ? "both"
          : entry.vector
          ? "meaning"
          : "words";
      const best = entry.vector ?? entry.text!;
      return {
        documentId,
        score: entry.score,
        passage: best.text,
        matched,
        bestSimilarity: entry.vector ? entry.vector.score : null,
      };
    })
    .sort((a, b) => b.score - a.score);
}
