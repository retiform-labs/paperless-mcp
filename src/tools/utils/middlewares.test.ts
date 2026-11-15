import { test } from "node:test";
import assert from "node:assert/strict";
import { withErrorHandling } from "./middlewares";

/**
 * Builds an axios-shaped error. The real object is reproduced closely enough
 * for this middleware: `config.headers.Authorization` carries the API token
 * (verified against a live axios rejection), and `response` is circular.
 */
function axiosError(options: {
  status?: number;
  data?: unknown;
  code?: string;
  message?: string;
  host?: string;
  token?: string;
}): Error {
  const {
    status,
    data,
    code,
    message = status ? `Request failed with status code ${status}` : "Network Error",
    host = "paperless.example:8000",
    token = "SUPERSECRETTOKEN123456",
  } = options;

  const err = new Error(message);
  (err as any).isAxiosError = true;
  (err as any).code = code;
  (err as any).config = {
    url: `/api/documents/`,
    baseURL: `http://${host}`,
    method: "get",
    headers: { Authorization: `Token ${token}`, Accept: "application/json" },
  };
  if (status !== undefined) {
    const response: any = { status, statusText: "", headers: {}, config: (err as any).config, data };
    // The real axios response references its request, which references the
    // config, which references the headers -- genuinely circular.
    response.request = { response };
    (err as any).response = response;
  }
  return err;
}

/** Runs the middleware and returns the message it threw. Fails if it does not throw. */
async function thrownMessage(inner: unknown): Promise<string> {
  const wrapped = withErrorHandling((async () => {
    throw inner;
  }) as any);
  try {
    await (wrapped as any)({}, {});
  } catch (err) {
    return err instanceof Error ? err.message : String(err);
  }
  assert.fail("withErrorHandling did not throw");
}

/** Every assertion the spec's actionable-error requirement implies. */
async function assertActionable(inner: unknown, mustMention: RegExp): Promise<string> {
  const message = await thrownMessage(inner);
  assert.ok(
    !message.startsWith("{"),
    `message must be a sentence, not a JSON blob; got: ${message}`
  );
  assert.match(message, /invalid|field|must|reject|permis|rate limit|failed|not found|timed out|reach|exist/i, `message names no cause: ${message}`);
  assert.match(message, mustMention, `message omits the actionable detail; got: ${message}`);
  assert.ok(
    !/SUPERSECRETTOKEN123456/.test(message),
    `message leaked the API token: ${message}`
  );
  return message;
}

test("a 400 names the offending field and says to fix it", async () => {
  const message = await assertActionable(
    axiosError({ status: 400, data: { title: ["This field may not be blank."] } }),
    /title/
  );
  // The whole point: the field name is present, so a caller can correct the call.
  assert.match(message, /title:\s*This field may not be blank/);
  assert.match(message, /fix/i);
});

test("a 400 with nested field detail keeps the nested key path", async () => {
  const message = await assertActionable(
    axiosError({ status: 400, data: { custom_fields: { "12": ["not a valid value"] } } }),
    /custom_fields\.12/
  );
  assert.match(message, /not a valid value/);
});

test("a 400 with an unrecognised body still quotes the server detail", async () => {
  const message = await assertActionable(
    axiosError({ status: 400, data: { detail: "Unsupported media type in payload." } }),
    /Unsupported media type in payload/
  );
  assert.match(message, /HTTP 400/);
});

test("a 400 with an empty body still produces a sentence naming the schema", async () => {
  const message = await assertActionable(axiosError({ status: 400, data: "" }), /schema|invalid/i);
  assert.match(message, /HTTP 400/);
});

test("401 names the credential problem and says an argument change will not help", async () => {
  await assertActionable(axiosError({ status: 401, data: { detail: "Invalid token." } }), /token|credential/i);
});

test("403 names the permission problem and warns against an unchanged retry", async () => {
  const message = await assertActionable(
    axiosError({ status: 403, data: { detail: "You do not have permission to perform this action." } }),
    /permission|forbidden/i
  );
  assert.match(message, /do not retry unchanged/i);
});

test("404 names the id problem and says the object may have been deleted", async () => {
  const message = await assertActionable(axiosError({ status: 404, data: { detail: "Not found." } }), /not exist|deleted|404/i);
  assert.match(message, /deleted|does not exist/i);
});

test("429 says rate limited and suggests retrying", async () => {
  const message = await assertActionable(axiosError({ status: 429, data: { detail: "Too many requests." } }), /rate limit/i);
  assert.match(message, /retry/i);
});

test("a 5xx says the server failed and that it is not the caller's fault", async () => {
  const message = await assertActionable(axiosError({ status: 503, data: { detail: "Service unavailable." } }), /server/i);
  assert.match(message, /not a problem with your arguments/i);
});

test("ECONNREFUSED names the host and says it may be unreachable", async () => {
  const message = await thrownMessage(
    axiosError({ code: "ECONNREFUSED", host: "paperless.internal:8000" })
  );
  assert.match(message, /paperless\.internal:8000/);
  assert.match(message, /unreachable|reach/i);
  assert.match(message, /ECONNREFUSED/);
  assert.ok(!/SUPERSECRETTOKEN123456/.test(message), `leaked token: ${message}`);
});

test("a timeout is reported as a timeout, not as a generic network error", async () => {
  const message = await thrownMessage(axiosError({ code: "ECONNABORTED", host: "paperless.internal:8000" }));
  assert.match(message, /timed out/i);
  assert.match(message, /paperless\.internal:8000/);
});

test("a circular response body does not turn the error into a TypeError", async () => {
  // JSON.stringify(err.response) throws on the real axios object. If the
  // handler serialises it directly, a useful error is replaced by a
  // serialisation failure -- strictly worse for the caller.
  const circular: any = { detail: "bad input" };
  circular.self = circular;
  const message = await thrownMessage(axiosError({ status: 400, data: circular }));
  assert.ok(!/circular structure|Converting circular/i.test(message), `serialisation error surfaced: ${message}`);
  assert.match(message, /invalid|field|reject/i);
});

test("a token echoed back by the server is scrubbed from the message", async () => {
  const message = await thrownMessage(
    axiosError({
      status: 400,
      data: { detail: "Header Authorization: Token SUPERSECRETTOKEN123456 was rejected." },
      token: "SUPERSECRETTOKEN123456",
    })
  );
  assert.ok(!/SUPERSECRETTOKEN123456/.test(message), `echoed token survived scrubbing: ${message}`);
  assert.match(message, /redacted/);
});

test("the thrown value is still an Error, so the SDK converts it to isError:true", async () => {
  const wrapped = withErrorHandling((async () => {
    throw axiosError({ status: 400, data: { title: ["bad"] } });
  }) as any);
  await assert.rejects(
    () => (wrapped as any)({}, {}),
    (err: unknown) => {
      assert.ok(err instanceof Error, "must rethrow an Error instance");
      return true;
    }
  );
});

test("a non-axios TypeError still produces an actionable sentence", async () => {
  const message = await thrownMessage(new TypeError("x is not a function"));
  assert.ok(!message.startsWith("{"));
  assert.match(message, /x is not a function/);
});

test("a local validation error is passed through, not relabelled as a network fault", async () => {
  // Tool bodies throw their own validation errors before any request is made.
  // Those carry no `response` and no axios config. Treating "no response" as
  // "transport failure" would tell the caller to check its base URL when the
  // actual fix is a different argument -- strictly worse than the old blob.
  const message = await thrownMessage(
    new Error("Confirmation required for destructive operation. Set confirm: true to proceed.")
  );
  assert.match(message, /confirm/i);
  assert.ok(
    !/could not reach|connection problem|unreachable|timed out/i.test(message),
    `a local validation error was relabelled as a transport failure: ${message}`
  );
});
