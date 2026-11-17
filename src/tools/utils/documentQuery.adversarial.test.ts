import assert from "node:assert/strict";
import { test } from "node:test";
import {
  DOCUMENT_QUERY_PAPERLESS_FILTER_KEYS,
  buildDocumentQueryString,
  customFieldQuerySchema,
  paperlessFiltersSchema,
} from "./documentQuery";

// ---------------------------------------------------------------------------
// ADVERSARIAL LAYER -- the custom-field query parser and the paperless_filters
// escape hatch.
//
//  is the one argument a caller can hand us a RECURSIVE
// structure through, and  walks it with no bound. These
// tests exist because the happy-path suite proves the parser accepts what it
// should, and none of it proves what it does with a hostile shape.
//
// Measured on this tree,  on a merely-deep
// structure (a few thousand nested one-element groups) throws
//  rather than returning a
// validation failure -- and the same shape reaches  over a
// real MCP handshake, where it surfaces as . A stack overflow is
// not a validation error: it is the process losing its own call stack to an
// argument it could have refused.
// ---------------------------------------------------------------------------

/** A single valid leaf clause. */
const LEAF = ["tag", "eq", "x"] as const;

/**
 * Wrap a leaf in  nested one-element AND groups.
 *
 * One-element groups are used deliberately: they are the cheapest way to add a
 * level of recursion for the least bytes, so a caller attacking the parser
 * reaches the stack limit with a small request body.
 */
function nestedGroup(depth: number): unknown {
  let node: unknown = [...LEAF];
  for (let index = 0; index < depth; index += 1) node = ["AND", [node]];
  return node;
}

test("a deeply nested custom_field_query fails validation instead of exhausting the stack", () => {
  // 5000 is comfortably past the measured overflow point on this build, and far
  // past any legitimate query: a real custom-field query has a handful of
  // groups. The assertion is on the SHAPE of the failure, not on a threshold --
  // either the schema refuses it or it accepts it, but the parse must not throw
  // a RangeError out of the validator.
  const hostile = nestedGroup(5000);

  let success: boolean | undefined;
  let thrown: Error | undefined;
  try {
    success = customFieldQuerySchema.safeParse(hostile).success;
  } catch (error) {
    thrown = error as Error;
  }

  assert.equal(
    thrown,
    undefined,
    "the schema threw " + (thrown ? thrown.constructor.name + " (" + thrown.message + ")" : "") +
      " on a nested custom_field_query. A recursive validator with no depth bound lets a caller " +
      "replace a validation error with a stack overflow."
  );
  assert.equal(
    success,
    false,
    "a 5000-deep custom_field_query must be rejected as invalid, not accepted"
  );
});

test("an oversized clause list is refused rather than turned into a huge GET", () => {
  // A group may hold any number of clauses, and the validated tree becomes a GET
  // query string -- so without a cap the CALLER picks how large a URL we send.
  // The failure then appears as a transport error, not as a bad argument.
  const tooMany = Array.from({ length: 2000 }, (_, index) => ["tag", "eq", String(index)]);
  const refused = customFieldQuerySchema.safeParse(["OR", tooMany]);
  assert.equal(
    refused.success,
    false,
    "a 2000-clause group validated; the clause count is caller-controlled and " +
      "there is no upper bound on the query string that follows"
  );

  // The largest accepted group must still produce a URL a server will take.
  const atCap = Array.from({ length: 64 }, (_, index) => ["tag", "eq", String(index)]);
  const accepted = customFieldQuerySchema.safeParse(["OR", atCap]);
  assert.equal(accepted.success, true, "a 64-clause group is a legitimate filter");
  const queryString = buildDocumentQueryString({
    custom_field_query: accepted.data as never,
  });
  assert.ok(
    queryString.length < 8192,
    "the largest accepted custom_field_query produced " + queryString.length +
      " chars of query string, which is past what a default URL limit will carry"
  );
});

test("paperless_filters refuses keys outside the documented set", () => {
  // The allowlist is the whole defence between a caller-chosen string and a
  // Django ORM lookup on the far side. Each of these keys is a real hazard if
  // it ever reaches the query string: prototype keys can corrupt plain-object
  // handling, and an arbitrary key lets a caller pick an unadvertised lookup.
  for (const key of ["__proto__", "constructor", "prototype", "evil__param"]) {
    assert.throws(
      () => buildDocumentQueryString({ paperless_filters: { [key]: "x" } }),
      /Unsupported paperless_filters key/,
      "paperless_filters must reject the key " + JSON.stringify(key)
    );
  }
});

test("a value cannot smuggle a second query parameter past the encoder", () => {
  // Values are set through URLSearchParams, so separators MUST be
  // percent-encoded. A raw CRLF or a raw ampersand would let a value append a
  // parameter the tool never advertised.
  const hostile = [
    { key: "title__icontains", value: "a&tags__id=1" },
    // Built from char codes so no quoting layer can turn this into a real newline.
    { key: "title__icontains", value: "a" + String.fromCharCode(13, 10) + "X-Injected: 1" },
    { key: "title__icontains", value: "a#frag" },
  ];
  for (const { key, value } of hostile) {
    const queryString = buildDocumentQueryString({
      paperless_filters: { [key]: value },
    });
    const search = new URLSearchParams(queryString.replace(/^\?/, ""));
    assert.equal(
      [...search.keys()].length,
      1,
      "value " + JSON.stringify(value) + " produced " + search.toString() +
        " -- it added a parameter that was never asked for"
    );
    assert.equal(search.get(key), value, "the value must round-trip unaltered");
  }
});

test("the same filter cannot be supplied twice through different routes", () => {
  //  maps to tags__id. Supplying tags__id again via paperless_filters
  // would emit the parameter twice, and which one Paperless honours is an
  // implementation detail -- so the pair must be refused, not resolved.
  assert.throws(
    () => buildDocumentQueryString({ tag: 3, paperless_filters: { tags__id: 4 } }),
    /Duplicate filter/,
    "a filter reachable both first-class and via paperless_filters must be rejected"
  );
});

test("an unknown group operator is rejected rather than treated as a field name", () => {
  // NOT only AND/OR. A three-element leaf whose first entry is a group operator
  // must not fall through and become [fieldName, operator, value].
  for (const operator of ["NOT", "XOR", "and", "or"]) {
    const result = customFieldQuerySchema.safeParse([
      operator,
      [["tag", "eq", "x"]],
    ]);
    assert.equal(
      result.success,
      false,
      "custom_field_query accepted " + JSON.stringify(operator) + " as a group operator"
    );
  }
});
