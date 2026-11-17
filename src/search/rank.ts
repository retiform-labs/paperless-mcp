// Reciprocal rank fusion of a vector (meaning) list and a text (words) list.
//
// Ported verbatim from jaimemachado/paperless-mcp (src/search/rank.ts at
// 50f7beec0d36804cab8fbcaa7b47f85e562e8061). Their only import was `Hit` from
// ./store; when this file was written store.ts did not exist on this branch, so
// Hit was declared here to let the module compile and test alone. That
// declaration is now GONE and store.ts owns Hit -- the merge of the two vector
// shards put two structurally identical live definitions in one tree, which is
// a defect whether or not it typechecks. The re-export below is kept so the
// existing `import { fuse, type Hit } from "./rank"` call sites keep working
// against the single definition.
//
// A document's score is the sum of 1/(k + bestRank) over each list it appears
// in, using its BEST rank in that list. A document found only by meaning (the
// vector list) must stay in the results -- that is the whole point of fusing
// rather than intersecting.

import type { Hit } from "./store";

export type { Hit };

export interface Fused {
  documentId: number;
  score: number;
  passage: string;
  matched: "meaning" | "words" | "both";
  bestSimilarity: number | null;
}

// `k` is the RRF dampening constant: it settles how much a document's rank
// matters relative to how many lists agree about it. A document both arms hold
// at rank D scores 2/(k+D), while a document only ONE arm found at rank 1
// scores 1/(k+1). So a dual-arm document outranks a perfect single-arm hit for
// every D < k+2.
//
// At the upstream default of k=60 that crossover sits at D=62 -- meaning any
// document both arms happened to return in their top 62 outranked a document
// one arm had ranked FIRST. k=60 is the figure from the original RRF paper,
// where candidate lists ran to thousands; it was inherited here verbatim from
// jaimemachado/paperless-mcp and never tuned for a corpus whose arms return at
// most a few hundred candidates and whose top 10 is what a caller reads.
//
// Measured on the live index (eval/bakeoff.mjs, 20 ground-truthed cases,
// 24 strategies, reproducible run to run): k=60 scores 15/20 at depth 10 while
// every k of 10 or below scores 18/20. Two cases carry the whole difference --
// a document ONE arm ranked first fell to rank 11 and to rank 52 under k=60.
// Which value below 60 is best is inside this fixture's noise (10, 5, 2, 1 and
// 0 all tie at 18/20), so k=10 is chosen for being the smallest change that
// removes the pathology while keeping rank order meaningful across the pool
// rather than collapsing it to bare reciprocal rank. Do not read 18 as "the
// best available number"; read it as "k no longer fights the ranking".
export const DEFAULT_FUSE_K = 10;

export function fuse(vector: Hit[], text: Hit[], k = DEFAULT_FUSE_K): Fused[] {
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
