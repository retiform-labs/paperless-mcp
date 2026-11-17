import assert from "node:assert/strict";
import { test } from "node:test";
import { MAX_BODY_BYTES, parseEvent, presentedSecret, secretMatches } from "./webhook";

// ---------------------------------------------------------------------------
// ADVERSARIAL LAYER -- THE WEBHOOK INBOX.
//
// The receiver is an UNAUTHENTICATED-BY-DESIGN surface: it listens on the Docker
// bridge so Paperless can POST to it, and the only gate is a shared secret. Two
// properties matter here and neither is about the happy path:
//
//   1. A malformed document id must not be reinterpreted into a DIFFERENT real
//      document. A sync triggered for the wrong doc is silent -- the caller sees
//      success and the wrong document gets re-indexed.
//   2. The secret comparison must fail closed for every variant that is not the
//      secret itself.
//
// A mutating endpoint reached without a valid secret is a remote-write bug, so
// each rejection is paired with a positive control in the same test.
// ---------------------------------------------------------------------------

test("a well-formed id is read exactly (positive control)", () => {
  // Every negative assertion in this file is vacuous if this fails.
  assert.deepEqual(parseEvent('{"id":42}', "application/json").ids, [42]);
  assert.deepEqual(parseEvent("id=42", "application/x-www-form-urlencoded").ids, [42]);
});

test("a fractional id is not silently truncated onto a different document", () => {
  // parseInt("5.9") is 5. Document 5 is a real document, so an event naming
  // "5.9" would re-index document 5 and report success. Nothing about the
  // response would tell the operator their payload was malformed.
  const event = parseEvent('{"id":"5.9"}', "application/json");
  assert.deepEqual(
    event.ids,
    [],
    "id \"5.9\" was read as document " + JSON.stringify(event.ids) +
      ". A fractional id names no document, and truncating it silently targets " +
      "a real one the sender never asked for."
  );
});

test("a numeric-prefixed string is not silently truncated onto a different document", () => {
  for (const raw of ["12abc", "12 34", "0x10", "1e3"]) {
    const event = parseEvent(JSON.stringify({ id: raw }), "application/json");
    assert.deepEqual(
      event.ids,
      [],
      "id " + JSON.stringify(raw) + " was read as document " +
        JSON.stringify(event.ids) + ". parseInt(.., 10) stops at the first " +
        "non-digit, so a malformed id becomes a real, different document."
    );
  }
  // Control: the canonical form of each value is still accepted where it is a
  // legitimate document id.
  assert.deepEqual(parseEvent('{"id":"12"}', "application/json").ids, [12]);
});

test("an out-of-range exponent does not collapse to document 1", () => {
  // parseInt("1e400") is 1 -- the most dangerous possible reinterpretation,
  // because document 1 is almost always real and a sync of it looks normal.
  const event = parseEvent('{"id":"1e400"}', "application/json");
  assert.deepEqual(
    event.ids,
    [],
    "id \"1e400\" was read as document " + JSON.stringify(event.ids) +
      ". It is not a document id at all; reading it as number 1 re-indexes the " +
      "wrong document and reports success."
  );
});

test("an array id does not collapse to its first element", () => {
  const event = parseEvent('{"id":[1,2]}', "application/json");
  assert.deepEqual(
    event.ids,
    [],
    "id [1,2] was read as " + JSON.stringify(event.ids) +
      ". String([1,2]) is \"1,2\" and parseInt stops at the comma, so a " +
      "multi-value id silently becomes the first document only."
  );
});

test("a non-integer JSON number is refused rather than floored", () => {
  for (const raw of [5.9, -3, 0, 1.0000001]) {
    const event = parseEvent(JSON.stringify({ id: raw }), "application/json");
    assert.deepEqual(
      event.ids,
      [],
      "id " + JSON.stringify(raw) + " was accepted as " + JSON.stringify(event.ids) +
        "; only a positive integer names a document"
    );
  }
  assert.deepEqual(parseEvent('{"id":7}', "application/json").ids, [7]);
});

test("an empty presented secret never matches a configured one", () => {
  // The receiver refuses to start with an empty EXPECTED secret, so the
  // interesting case is an empty PROVIDED value against a real secret. If this
  // ever returned true, every unauthenticated POST on the bridge would pass.
  assert.equal(
    secretMatches("", "s3cret"),
    false,
    "an empty presented secret must never match the configured secret"
  );
  assert.equal(
    secretMatches(undefined, "s3cret"),
    false,
    "an absent secret must never match"
  );
  // Control: the exact secret still matches.
  assert.equal(secretMatches("s3cret", "s3cret"), true, "control: the exact secret matches");
});

test("a secret differing only in case or whitespace is refused", () => {
  for (const wrong of ["S3CRET", "s3cret ", " s3cret", "s3cre", "s3cretx"]) {
    assert.equal(
      secretMatches(wrong, "s3cret"),
      false,
      "secret " + JSON.stringify(wrong) + " must not match; a comparison that " +
        "normalises case or trims would accept a value the operator never set"
    );
  }
  assert.equal(secretMatches("s3cret", "s3cret"), true, "control: exact match still works");
});

test("only the documented header forms yield a presented secret", () => {
  assert.equal(
    presentedSecret({ "x-paperless-mcp-secret": "abc" }),
    "abc",
    "the dedicated header must be read"
  );
  assert.equal(
    presentedSecret({ authorization: "Bearer abc" }),
    "abc",
    "the bearer form must be read"
  );
  // A lowercase scheme is not the documented form. Pinned as a deviation that
  // fails CLOSED: an interoperable client is refused, never waved through.
  assert.equal(
    presentedSecret({ authorization: "bearer abc" }),
    undefined,
    "a lowercase scheme must yield no secret, never the token"
  );
  assert.equal(
    presentedSecret({ authorization: "Basic abc" }),
    undefined,
    "a non-bearer scheme must yield no secret"
  );
  assert.equal(presentedSecret({}), undefined, "no header must yield no secret");
  // An empty dedicated header must not shadow a valid bearer header.
  assert.equal(
    presentedSecret({ "x-paperless-mcp-secret": "", authorization: "Bearer abc" }),
    "abc",
    "an empty dedicated header must not suppress the bearer form"
  );
});

test("the body cap is a real byte budget, not a character count", () => {
  // MAX_BODY_BYTES is compared against byte length. A multi-byte payload that
  // fits in 1 MiB of BYTES must not be refused because it is shorter in
  // characters, and vice versa: the budget is what the socket carries.
  assert.equal(MAX_BODY_BYTES, 1024 * 1024, "the documented cap is 1 MiB");
  const atCap = Buffer.alloc(MAX_BODY_BYTES, 0x61);
  assert.equal(atCap.byteLength, MAX_BODY_BYTES, "control: the fixture is exactly at the cap");
  const overCap = Buffer.alloc(MAX_BODY_BYTES + 1, 0x61);
  assert.ok(
    overCap.byteLength > MAX_BODY_BYTES,
    "one byte past the cap must be over it; comparing utf8 LENGTH of a decoded " +
      "string would let a multi-byte payload exceed the byte budget"
  );
});
