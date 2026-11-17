import assert from "node:assert/strict";
import { test } from "node:test";
import type express from "express";
import { getBearerToken, sendUnauthorized } from "./server";

// ---------------------------------------------------------------------------
// ADVERSARIAL LAYER -- THE AUTH BOUNDARY.
//
// getBearerToken is the ONLY thing between an unauthenticated HTTP client and a
// Paperless API token. Everything downstream trusts its answer, so a false
// positive here is a full compromise and a false negative is an outage.
//
// Every "does not authenticate" assertion below sits beside a positive control
// in the SAME test. A negative-only test passes just as happily when the harness
// is broken and never ran the code, so the control is what makes the red
// meaningful.
// ---------------------------------------------------------------------------

/** Minimal express.Request stand-in: only the fields the helper reads. */
function makeReq(headers: Record<string, string | string[] | undefined>): express.Request {
  return { headers } as unknown as express.Request;
}

const STRICT = { allowAnonymous: false } as const;
const ANON = { allowAnonymous: true, fallbackToken: "server-token" } as const;

test("a well-formed Bearer token authenticates (positive control)", () => {
  // If this ever fails, every negative assertion in this file is vacuous.
  const token = getBearerToken(makeReq({ authorization: "Bearer client-token" }), STRICT);
  assert.equal(token, "client-token", "the control case must yield the client token");
});

test("a missing Authorization header does not authenticate", () => {
  const token = getBearerToken(makeReq({}), STRICT);
  assert.equal(token, undefined, "no header must not authenticate");
  // Control: the identical request WITH a header does authenticate.
  assert.equal(
    getBearerToken(makeReq({ authorization: "Bearer client-token" }), STRICT),
    "client-token",
    "control: the same code path must authenticate when the header is present"
  );
});

test("an empty Bearer token does not authenticate", () => {
  // 'Bearer ' slices to the empty string. The empty string is falsy, so
  // index.ts'  treats it as absent -- but a caller reading
  // only this helper would see a value. Pin the falsiness, not just undefined.
  const token = getBearerToken(makeReq({ authorization: "Bearer " }), STRICT);
  assert.ok(!token, "'Bearer ' with no token must not yield a usable credential (got " +
    JSON.stringify(token) + ")");
  assert.equal(
    getBearerToken(makeReq({ authorization: "Bearer x" }), STRICT),
    "x",
    "control: a one-character token is still returned"
  );
});

test("a wrong auth scheme does not authenticate", () => {
  for (const header of ["Basic dXNlcjpwYXNz", "Token abc", "abc", "Bearerabc"]) {
    const token = getBearerToken(makeReq({ authorization: header }), STRICT);
    assert.equal(
      token,
      undefined,
      "header " + JSON.stringify(header) + " must not authenticate"
    );
  }
  assert.equal(
    getBearerToken(makeReq({ authorization: "Bearer abc" }), STRICT),
    "abc",
    "control: the correct scheme with a space does authenticate"
  );
});

test("a lowercase scheme is refused -- and therefore fails closed", () => {
  // RFC 7235 makes the auth scheme case-INSENSITIVE, so  is a legal
  // Authorization header. This server only accepts the exact capitalisation.
  // That is a spec deviation, but it is the SAFE direction: an interoperable
  // client gets a 401, never an unauthenticated request. Pinned so a future
  // 'fix' cannot quietly turn it into a passthrough.
  assert.equal(
    getBearerToken(makeReq({ authorization: "bearer client-token" }), STRICT),
    undefined,
    "a lowercase scheme must not fall through to 'no credential' as 'authenticated'"
  );
  assert.equal(
    getBearerToken(makeReq({ authorization: "Bearer client-token" }), STRICT),
    "client-token",
    "control: the canonical capitalisation works"
  );
});

test("two Authorization headers cannot produce a valid credential by concatenation", () => {
  // Express joins repeated headers with ', '. The joined string must not be
  // interpreted as a bearer token, or a client could append a second header to
  // smuggle a value past a proxy that inspected only the first.
  const joined = getBearerToken(
    makeReq({ authorization: "Basic x, Bearer smuggled" }),
    STRICT
  );
  assert.equal(joined, undefined, "a joined header must not authenticate");

  // And when the FIRST header is the Bearer one, what comes back is a mangled
  // string, not the clean token -- so the downstream Paperless call fails
  // rather than authenticating as the smuggled value.
  const mangled = getBearerToken(
    makeReq({ authorization: "Bearer first, Bearer second" }),
    STRICT
  );
  assert.notEqual(mangled, "second", "the second header's value must not become the token");
  assert.notEqual(mangled, "first", "neither must the first, intact");
  assert.equal(
    getBearerToken(makeReq({ authorization: "Bearer clean" }), STRICT),
    "clean",
    "control: a single header yields a clean token"
  );
});

test("a token with surrounding whitespace is not silently trimmed into validity", () => {
  // Trimming would be a silent reinterpretation of a credential. Whatever the
  // policy, the helper must not invent a token the client did not send: a padded
  // value must not equal the unpadded one.
  const padded = getBearerToken(makeReq({ authorization: "Bearer token " }), STRICT);
  assert.notEqual(padded, "token", "a trailing space must not be trimmed away");
  assert.equal(
    getBearerToken(makeReq({ authorization: "Bearer token" }), STRICT),
    "token",
    "control: the unpadded form is returned verbatim"
  );
});

test("--no-auth substitutes the server token and never yields a usable 'no credential'", () => {
  // The legacy escape hatch is NOT 'no auth'. It is 'the server's token stands in
  // for an absent client token'. The distinction is the whole security property:
  // a request with no header must still arrive at Paperless carrying a token.
  assert.equal(
    getBearerToken(makeReq({}), ANON),
    "server-token",
    "with allowAnonymous the server token must be substituted"
  );
  // A client-supplied token still WINS over the fallback -- otherwise --no-auth
  // would silently override a properly authenticated caller.
  assert.equal(
    getBearerToken(makeReq({ authorization: "Bearer client-token" }), ANON),
    "client-token",
    "an explicit client token must take precedence over the fallback"
  );
  // And with anonymous allowed but NO fallback configured, there is nothing to
  // authenticate as, so the answer must be undefined rather than an empty string
  // that a truthiness check somewhere else might read as present.
  assert.equal(
    getBearerToken(makeReq({}), { allowAnonymous: true }),
    undefined,
    "anonymous with no fallback token must yield no credential"
  );
  // Control: the same no-header request under the STRICT policy is refused.
  assert.equal(
    getBearerToken(makeReq({}), STRICT),
    undefined,
    "control: strict mode refuses the identical request"
  );
});

test("strict mode never falls back to the server token", () => {
  // The v2.0.0 change is precisely this: a server-configured token must not be
  // usable by a caller that did not present one. Passing a fallbackToken while
  // allowAnonymous is false must change nothing.
  const token = getBearerToken(
    makeReq({}),
    { allowAnonymous: false, fallbackToken: "server-token" }
  );
  assert.equal(
    token,
    undefined,
    "a fallbackToken must be inert unless allowAnonymous is true, or v2.0.0's " +
      "per-request token requirement is not actually enforced"
  );
});

test("sendUnauthorized answers 401 with a challenge and no configuration hints", () => {
  const recorded: { status?: number; headers: Record<string, string>; body?: unknown } = {
    headers: {},
  };
  const res = {
    status(code: number) {
      recorded.status = code;
      return this;
    },
    set(name: string, value: string) {
      recorded.headers[name.toLowerCase()] = value;
      return this;
    },
    json(body: unknown) {
      recorded.body = body;
      return this;
    },
  } as unknown as express.Response;

  sendUnauthorized(res);

  assert.equal(recorded.status, 401, "the rejection status must be 401");
  assert.equal(
    recorded.headers["www-authenticate"],
    'Bearer realm="paperless-mcp"',
    "a 401 must carry a WWW-Authenticate challenge so a conforming client knows the scheme"
  );
  assert.deepEqual(
    recorded.body,
    { error: "unauthorized" },
    "the body must be minimal -- sendUnauthorized' own comment promises the wire " +
      "response carries no configuration hints"
  );
  // The operator guidance belongs on stderr, never in the response.
  const bodyText = JSON.stringify(recorded.body);
  for (const hint of ["--no-auth", "Bearer <paperless-ngx-api-token>", "Authorization"]) {
    assert.ok(
      !bodyText.includes(hint),
      "the 401 body leaked operator guidance (" + hint + "), which tells an " +
        "unauthenticated caller how the deployment is configured"
    );
  }
});
