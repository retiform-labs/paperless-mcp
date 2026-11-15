import assert from "node:assert/strict";
import { test } from "node:test";
import {
  CHUNK_OVERLAP,
  CHUNK_SIZE,
  MAX_CHUNKS,
  chunkText,
  metadataText,
} from "./chunk";

// The stride is the arithmetic the whole overlap guarantee rests on. If these
// two drift apart, consecutive chunks stop sharing a boundary and nothing else
// in this file would notice.
test("the stride is CHUNK_SIZE - CHUNK_OVERLAP", () => {
  assert.equal(CHUNK_SIZE, 2000);
  assert.equal(CHUNK_OVERLAP, 200);
  assert.equal(CHUNK_SIZE - CHUNK_OVERLAP, 1800);
});

test("the default cap is 4000 chunks", () => {
  assert.equal(MAX_CHUNKS, 4000);
});

test("an empty string gives no chunks and is not truncated", () => {
  assert.deepEqual(chunkText(""), { chunks: [], truncated: false });
});

test("whitespace-only text gives no chunks and is not truncated", () => {
  assert.deepEqual(chunkText("  \n\t  \n "), { chunks: [], truncated: false });
});

test("whitespace normalisation collapses newlines, tabs and runs of spaces", () => {
  const input = "line one\n\n\ttabbed   spaced\r\nline two";
  assert.deepEqual(chunkText(input), {
    chunks: ["line one tabbed spaced line two"],
    truncated: false,
  });
});

test("a short document produces exactly one chunk", () => {
  const result = chunkText("a short document");
  assert.equal(result.chunks.length, 1);
  assert.equal(result.chunks[0], "a short document");
  assert.equal(result.truncated, false);
});

test("a document of exactly CHUNK_SIZE does not produce a spurious second chunk", () => {
  const text = "x".repeat(CHUNK_SIZE);
  const result = chunkText(text);
  assert.equal(result.chunks.length, 1);
  assert.equal(result.chunks[0].length, CHUNK_SIZE);
  assert.equal(result.truncated, false);
});

test("one character past CHUNK_SIZE does start a second chunk", () => {
  const text = "x".repeat(CHUNK_SIZE + 1);
  const result = chunkText(text);
  assert.equal(result.chunks.length, 2);
  // The second chunk starts at the stride (1800), NOT at 2000, so it runs from
  // 1800 to the end: 2001 - 1800 = 201 characters -- 200 of overlap plus the
  // one character that spilled past the first chunk.
  assert.equal(result.chunks[1].length, CHUNK_SIZE + 1 - (CHUNK_SIZE - CHUNK_OVERLAP));
  assert.equal(result.chunks[1].length, 201);
  assert.equal(result.truncated, false);
});

test("consecutive chunks share exactly CHUNK_OVERLAP characters", () => {
  // A PERIODIC string would make this assertion unfalsifiable: the stride 1800
  // is a multiple of 10, so "abcdefghij".repeat(n) matches itself under any
  // shift by a multiple of 10 and a wrong overlap would still pass. So build a
  // non-repeating filler: a long prime cycle that shares no factor with 1800.
  const cycle: string[] = [];
  for (let i = 0; i < 71; i += 1) cycle.push(String.fromCharCode(97 + (i % 26)) + (i % 7));
  let text = "";
  while (text.length < 6000) text += cycle.join("");
  text = text.slice(0, 6000);

  const result = chunkText(text);
  assert.equal(result.chunks.length, 4);
  for (let i = 1; i < result.chunks.length; i += 1) {
    const tail = result.chunks[i - 1].slice(-CHUNK_OVERLAP);
    const head = result.chunks[i].slice(0, CHUNK_OVERLAP);
    assert.equal(tail.length, CHUNK_OVERLAP);
    assert.equal(head, tail, `chunk ${i} does not start with chunk ${i - 1}'s tail`);
  }
  // And prove the filler is genuinely non-periodic at this stride: chunk 0's
  // tail must NOT equal chunk 0's first 200 chars, so an accidental match
  // cannot be explained by repetition.
  assert.notEqual(result.chunks[0].slice(-CHUNK_OVERLAP), result.chunks[0].slice(0, CHUNK_OVERLAP));
});

test("the overlap is genuinely content from the document, not padding", () => {
  // A unique marker placed to land inside the overlap window.
  const marker = "OVERLAPMARKER";
  const prefix = "a".repeat(CHUNK_SIZE - CHUNK_OVERLAP + 50); // 1850
  const text = prefix + marker + "b".repeat(500);
  const result = chunkText(text);
  assert.equal(result.chunks.length, 2);
  // The marker sits at 1850..1863, which is inside chunk 0 and inside chunk 1
  // (chunk 1 starts at 1800). Both must contain it.
  assert.ok(result.chunks[0].includes(marker));
  assert.ok(result.chunks[1].includes(marker));
});

test("a document that ends exactly on the cap boundary is not truncated", () => {
  // 2 chunks: 0..2000 then 1800..3800, and 3800 is exactly the end.
  const result = chunkText("y".repeat(3800), 2);
  assert.equal(result.chunks.length, 2);
  assert.equal(result.truncated, false);
});

test("one character past the cap boundary IS truncated", () => {
  const result = chunkText("y".repeat(3801), 2);
  assert.equal(result.chunks.length, 2);
  assert.equal(result.truncated, true);
});

test("the cap is honoured and truncation is flagged with a small maxChunks", () => {
  const result = chunkText("y".repeat(5000 * 1800), 5);
  assert.equal(result.chunks.length, 5);
  assert.equal(result.truncated, true);
});

test("the default cap of 4000 is honoured on a document that needs more", () => {
  // 5000 chunks' worth of text: the default cap must cut it at 4000 and say so.
  const result = chunkText("y".repeat(5000 * 1800));
  assert.equal(result.chunks.length, MAX_CHUNKS);
  assert.equal(result.chunks.length, 4000);
  assert.equal(result.truncated, true);
});

test("a document that needs exactly the default cap is not truncated", () => {
  // 4000 chunks exactly: the last one starts at 3999*1800 and runs to +2000.
  const result = chunkText("y".repeat(3999 * 1800 + 2000));
  assert.equal(result.chunks.length, MAX_CHUNKS);
  assert.equal(result.truncated, false);
});

// --- metadataText: the Spanish regression ------------------------------------
//
// Before the port, metadataText emitted its five scaffolding labels in Spanish
// at position 0 of every document, so those tokens were both embedded and
// lexical-indexed across a whole English corpus. These tests are the pin.

test("metadataText emits English scaffolding labels", () => {
  const text = metadataText(
    { title: "Power bill", created: "2026-01-15" },
    {
      correspondent: "Duke Energy",
      documentType: "Invoice",
      tags: ["Contract", "ai-processed", "via-mcp"],
    }
  );
  assert.ok(text.includes("Title:"), `expected "Title:" in: ${text}`);
  assert.ok(text.includes("Type:"), `expected "Type:" in: ${text}`);
  assert.ok(text.includes("Sender:"), `expected "Sender:" in: ${text}`);
  assert.ok(text.includes("Tags:"), `expected "Tags:" in: ${text}`);
  assert.ok(text.includes("Date:"), `expected "Date:" in: ${text}`);
});

test("metadataText contains no Spanish label - the Título regression", () => {
  const text = metadataText(
    { title: "Power bill", created: "2026-01-15" },
    { correspondent: "Duke Energy", documentType: "Invoice", tags: ["Contract"] }
  );
  assert.ok(!text.includes("Título"), `"Título" leaked into: ${text}`);
  assert.ok(!text.includes("Titulo"), `"Titulo" leaked into: ${text}`);
  assert.ok(!text.includes("Tipo"), `"Tipo" leaked into: ${text}`);
  assert.ok(!text.includes("Remitente"), `"Remitente" leaked into: ${text}`);
  assert.ok(!text.includes("Etiquetas"), `"Etiquetas" leaked into: ${text}`);
  assert.ok(!text.includes("Fecha"), `"Fecha" leaked into: ${text}`);
});

test("metadataText keeps the original structure and ordering", () => {
  const text = metadataText(
    { title: "Power bill", created: "2026-01-15" },
    { correspondent: "Duke Energy", documentType: "Invoice", tags: ["Contract"] }
  );
  assert.equal(
    text,
    "Title: Power bill. Type: Invoice. Sender: Duke Energy. Tags: Contract. Date: 2026-01-15."
  );
});

test("metadataText skips ai- and via- tags", () => {
  const text = metadataText(
    { title: "Note", created: "2026-02-01" },
    { tags: ["Contract", "ai-processed", "via-mcp"] }
  );
  assert.ok(text.includes("Tags: Contract."));
  assert.ok(!text.includes("ai-processed"));
  assert.ok(!text.includes("via-mcp"));
});

test("metadataText drops Type and Sender when they are missing", () => {
  const text = metadataText(
    { title: "Note", created: "2026-02-01" },
    { tags: [] }
  );
  assert.equal(text, "Title: Note. Tags: . Date: 2026-02-01.");
  assert.ok(!text.includes("Type:"));
  assert.ok(!text.includes("Sender:"));
});
