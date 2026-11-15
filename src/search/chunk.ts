// Text chunking for the paperless vector index.
//
// Algorithm ported from jaimemachado/paperless-mcp (src/search/chunk.ts at
// 50f7beec0d36804cab8fbcaa7b47f85e562e8061), kept near-verbatim. Two deliberate
// changes, both measured against our corpus:
//
//   1. metadataText() emitted its five scaffolding labels in SPANISH (title,
//      type, sender, tags, date) at position 0 of every document, so those
//      five tokens were embedded AND lexical-indexed across an English corpus.
//      English now. Same structure, same ordering. No Spanish literal is kept
//      in this file, not even in a comment, so a grep for the bug cannot hit
//      this header and read like the bug survived.
//   2. MAX_CHUNKS 1000 -> 4000. 21 live documents exceed 2M chars (75,133,944
//      chars total) and lost ~56% of their body at 1000 chunks; at 4000 no live
//      document truncates, at a cost of ~20,700 extra chunks (~7% of the index).
//
// The stride is CHUNK_SIZE - CHUNK_OVERLAP = 1800, so consecutive chunks share
// exactly the trailing 200 characters of the previous one.

export const CHUNK_SIZE = 2000;
export const CHUNK_OVERLAP = 200;
export const MAX_CHUNKS = 4000;
const SKIPPED_TAG_PREFIXES = ["ai-", "via-"];

export interface ChunkResult {
  chunks: string[];
  truncated: boolean;
}

// A chunk has 2000 characters, about 512 tokens, and the overlap is 200
// characters. A document has at most maxChunks chunks (MAX_CHUNKS by default).
//
// truncated is true only when the cap was reached WITHOUT the text ending. A
// document that lands exactly on the cap is not truncated.
export function chunkText(text: string, maxChunks = MAX_CHUNKS): ChunkResult {
  const normalized = text.replace(/\s+/g, " ").trim();
  if (normalized === "") {
    return { chunks: [], truncated: false };
  }

  const chunks: string[] = [];
  let start = 0;
  let reachedEnd = false;
  while (start < normalized.length && chunks.length < maxChunks) {
    chunks.push(normalized.slice(start, start + CHUNK_SIZE));
    if (start + CHUNK_SIZE >= normalized.length) {
      reachedEnd = true;
      break;
    }
    start += CHUNK_SIZE - CHUNK_OVERLAP;
  }

  const truncated = !reachedEnd && chunks.length >= maxChunks;
  return { chunks, truncated };
}

export function metadataText(
  doc: { title: string; created: string },
  names: { correspondent?: string; documentType?: string; tags: string[] }
): string {
  const tags = names.tags.filter(
    (tag) => !SKIPPED_TAG_PREFIXES.some((prefix) => tag.startsWith(prefix))
  );

  const parts = [`Title: ${doc.title}.`];
  if (names.documentType) {
    parts.push(`Type: ${names.documentType}.`);
  }
  if (names.correspondent) {
    parts.push(`Sender: ${names.correspondent}.`);
  }
  parts.push(`Tags: ${tags.join(", ")}.`);
  parts.push(`Date: ${doc.created}.`);

  return parts.join(" ");
}
