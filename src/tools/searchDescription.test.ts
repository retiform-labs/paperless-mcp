import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";

// __dirname, not import.meta: this repo compiles to CommonJS and tsconfig has
// no ESM module target, so import.meta is a type error here.
const root = join(__dirname, "..", "..");
const searchSrc = readFileSync(join(root, "src/tools/search.ts"), "utf8");
const readme = readFileSync(join(root, "README.md"), "utf8");

// The tool description IS the contract a client receives -- it ships in
// tools/list and an agent writes control flow against it. The defect this
// pins: `strong_match`'s description promised "false when nothing scored
// above the similarity floor" long after the code had been reduced to
// `passages.length > 0`, so the false branch was unreachable and a caller
// would have written `if (!strong_match) requery()` against a lie.
//
// A CODE test could not catch this: the code was correct. Only a text
// assertion catches a stale contract.
test("no surface still promises the 0.45 similarity floor for strong_match", () => {
  const dead = [
    /strong_match is false when nothing scored above/i,
    /cleared the similarity floor \(0\.45\)/i,
    /A `false` here means everything returned is weak/i,
  ];
  for (const [name, text] of [["search.ts", searchSrc], ["README.md", readme]] as const) {
    for (const re of dead) {
      assert.ok(!re.test(text), `${name} still carries the retired claim: ${re}`);
    }
  }
});

test("the honest description states what strong_match actually means", () => {
  assert.match(
    searchSrc,
    /strong_match is true exactly when at least one passage was returned/,
    "the live tool description must say the real condition"
  );
  assert.match(
    readme,
    /coarse "did anything come back" flag/,
    "the README must not describe a quality judgement it cannot make"
  );
});

// The same class, one field over: `highlights` was documented as "always
// absent on this path" after P2 started returning it.
test("the README does not claim highlights is always absent", () => {
  assert.ok(
    !/highlights` is always absent/i.test(readme),
    "P2 added highlights to this path; the docs must not say otherwise"
  );
});

// And the anchor: documented as "the first verbatim query term" both before
// and after the fix that changed it, which is how a stale doc survives a
// behaviour change.
test("the README describes the length-based anchor, not query order", () => {
  assert.ok(
    !/anchored on the first verbatim query term/i.test(readme),
    "the anchor is chosen by term LENGTH since ef12937; 'first' was the defect"
  );
  assert.match(readme, /longest matching query term/i);
});
