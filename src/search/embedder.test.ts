import assert from "node:assert/strict";
import { afterEach, describe, test } from "node:test";
import {
  EmbeddingContextLengthError,
  EmbeddingUnavailableError,
  MIN_SPLITTABLE_CHARS,
  OllamaEmbedder,
  PREFIXES,
  PrefixModeMismatchError,
  embeddingsFromBody,
  graphemeBoundaries,
  isContextLengthRejection,
  parseEmbeddingEndpoint,
  renormalisedMean,
  resolveEmbeddingEndpoint,
  splitForEmbedding,
  assertPrefixModeMatches,
} from "./embedder";

const realFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = realFetch;
});

interface CapturedRequest {
  url: string;
  body: { model: string; input: string[] };
  headers: Record<string, string>;
  /** The fetch init, so a test can assert on transport options like redirect. */
  init: RequestInit;
}

function stubFetch(
  respond: (req: CapturedRequest) => { status?: number; body?: unknown; text?: string }
): CapturedRequest[] {
  const captured: CapturedRequest[] = [];
  globalThis.fetch = (async (url: string | URL, init?: RequestInit) => {
    const headers = (init?.headers ?? {}) as Record<string, string>;
    const request: CapturedRequest = {
      url: String(url),
      body: JSON.parse(String(init?.body ?? "{}")),
      headers,
      init: init ?? {},
    };
    captured.push(request);
    const result = respond(request);
    const status = result.status ?? 200;
    // `text` exists so an error-body test can return a non-JSON payload -- an
    // error page is often HTML, and the client must survive one.
    if (result.text !== undefined) {
      return new Response(result.text, {
        status,
        headers: { "Content-Type": "text/plain" },
      });
    }
    return new Response(JSON.stringify(result.body), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;
  return captured;
}

function vectorOfWidth(width: number, head = 0.5): number[] {
  return Array.from({ length: width }, (_, index) => (index === 0 ? head : 0));
}

// ── Endpoint validation ────────────────────────────────────────────────────
// Two layers, tested separately because they answer different questions:
//   parseEmbeddingEndpoint  -- is this URL admissible at all?
//   localOnly               -- is it admissible for a deployment that refuses
//                              to send text off the host?
// A gate that only passes its happy path is untested, so every case below is a
// negative one first.

describe("parseEmbeddingEndpoint — always-on hygiene", () => {
  test("accepts loopback and RFC1918 with no flag set", () => {
    for (const url of [
      "http://127.0.0.1:11434",
      "http://localhost:11434",
      "http://[::1]:11434",
      "http://127.0.0.1:11434",
      "http://192.0.2.10:11434",
      "http://172.16.0.1:11434",
      "https://192.0.2.20:11434/v1",
    ]) {
      assert.doesNotThrow(() => parseEmbeddingEndpoint(url), url);
    }
  });

  test("ACCEPTS a DNS name and a public host when local-only is off", () => {
    // ⚠️ INVERTED. This was "refuses a public DNS name" / "refuses a real remote
    // provider", asserting that `example.com` and `api.openai.com` throw
    // unconditionally. That assertion was an opinionated deployment policy compiled into a
    // library published for general use: a third-party install pointing at a
    // hosted provider has done nothing wrong. The restriction still exists and
    // is still tested below -- it is now a flag the deployment sets rather than
    // a rule the package imposes.
    for (const url of [
      "http://example.com:11434",
      "https://api.openai.com/v1",
      "https://embeddings.example.net/v1/embeddings",
      "http://8.8.8.8:11434",
    ]) {
      assert.doesNotThrow(() => parseEmbeddingEndpoint(url), url);
    }
  });

  test("refuses a non-http protocol, flag or no flag", () => {
    for (const localOnly of [false, true]) {
      assert.throws(
        () => parseEmbeddingEndpoint("file:///etc/passwd", { localOnly }),
        /must use http: or https:/,
        `localOnly=${localOnly}`
      );
      assert.throws(
        () => parseEmbeddingEndpoint("ftp://127.0.0.1:21", { localOnly }),
        /must use http: or https:/,
        `localOnly=${localOnly}`
      );
    }
  });

  test("refuses an unparseable URL", () => {
    assert.throws(
      () => parseEmbeddingEndpoint("not a url at all"),
      /not a valid URL/
    );
  });

  test("refuses an octet above 255", () => {
    // The WHATWG URL parser rejects this before our allowlist ever runs, so the
    // message is the parse failure rather than the allowlist refusal. Asserting
    // on the message would pin us to which check happens to fire first; what
    // matters is that the endpoint is refused. Measured, not assumed.
    assert.throws(() => parseEmbeddingEndpoint("http://999.0.0.1:11434"));
  });
});

describe("parseEmbeddingEndpoint — the local-only restriction", () => {
  test("refuses a public DNS name", () => {
    assert.throws(
      () => parseEmbeddingEndpoint("http://example.com", { localOnly: true }),
      /not a literal loopback or RFC1918 address/
    );
  });

  test("refuses a real remote provider, which is the whole point of the flag", () => {
    assert.throws(
      () => parseEmbeddingEndpoint("https://api.example.com/v1", { localOnly: true }),
      /not a literal loopback or RFC1918 address/
    );
    assert.throws(
      () => parseEmbeddingEndpoint("https://api.openai.com/v1", { localOnly: true }),
      /not a literal loopback or RFC1918 address/
    );
  });

  test("still accepts loopback and RFC1918 with the flag on", () => {
    for (const url of [
      "http://127.0.0.1:11434",
      "http://192.0.2.10:11434",
      "http://127.0.0.1:11434",
    ]) {
      assert.doesNotThrow(() => parseEmbeddingEndpoint(url, { localOnly: true }), url);
    }
  });

  test("refuses a public literal address", () => {
    assert.throws(
      () => parseEmbeddingEndpoint("http://8.8.8.8:11434", { localOnly: true }),
      /public address/
    );
  });

  test("refuses 172.15 and 172.32, which sit outside the RFC1918 window", () => {
    for (const url of ["http://172.15.1.1:11434", "http://172.32.1.1:11434"]) {
      assert.throws(
        () => parseEmbeddingEndpoint(url, { localOnly: true }),
        /public address/,
        url
      );
    }
  });

  test("refuses 0.0.0.0 with an explanation naming the real fix", () => {
    // A server BINDS 0.0.0.0 on this box, so the obvious naive check would be
    // wrong in the opposite direction. Dialing the unspecified address is still
    // not a destination.
    assert.throws(
      () => parseEmbeddingEndpoint("http://0.0.0.0:11434", { localOnly: true }),
      /unspecified address/
    );
  });

  test("the flag defaults to OFF", () => {
    // The default is a claim, so it gets a test: `parseEmbeddingEndpoint(url)`
    // and `parseEmbeddingEndpoint(url, {})` must behave like localOnly:false.
    assert.doesNotThrow(() => parseEmbeddingEndpoint("http://example.com"));
    assert.doesNotThrow(() => parseEmbeddingEndpoint("http://example.com", {}));
  });
});

describe("resolveEmbeddingEndpoint — where the request goes", () => {
  test("an origin with no path falls back to Ollama's native route", () => {
    assert.equal(
      resolveEmbeddingEndpoint(new URL("http://127.0.0.1:11434")),
      "http://127.0.0.1:11434/api/embed"
    );
  });

  test("a trailing slash does not double the separator", () => {
    assert.equal(
      resolveEmbeddingEndpoint(new URL("http://127.0.0.1:11434/")),
      "http://127.0.0.1:11434/api/embed"
    );
  });

  test("a configured path gets /embeddings appended", () => {
    assert.equal(
      resolveEmbeddingEndpoint(new URL("https://api.openai.com/v1")),
      "https://api.openai.com/v1/embeddings"
    );
  });

  test("a full endpoint path is used verbatim, not appended twice", () => {
    // This is the case a naive `+ "/embeddings"` breaks: the caller who read the
    // docs and pasted the complete URL would get `.../embeddings/embeddings`.
    assert.equal(
      resolveEmbeddingEndpoint(new URL("https://host/v1/embeddings")),
      "https://host/v1/embeddings"
    );
    assert.equal(
      resolveEmbeddingEndpoint(new URL("https://host/v1/embeddings/")),
      "https://host/v1/embeddings"
    );
  });
});

describe("embeddingsFromBody — both reply shapes", () => {
  test("reads Ollama's native {embeddings: [[...]]}", () => {
    const vector = vectorOfWidth(4, 0.25);
    assert.deepEqual(embeddingsFromBody({ embeddings: [vector] }), [vector]);
  });

  test("reads the OpenAI-compatible {data: [{embedding: [...]}]}", () => {
    const vector = vectorOfWidth(4, 0.25);
    assert.deepEqual(
      embeddingsFromBody({ object: "list", data: [{ object: "embedding", embedding: vector }] }),
      [vector]
    );
  });

  test("returns undefined for a body that is neither shape", () => {
    // ⚠️ `{data: []}` is deliberately NOT in this list, and it was when the test
    // was first written. An empty `data` array IS the OpenAI shape, just an
    // empty one, exactly as `{embeddings: []}` is an empty native one -- so a
    // parser that answered `undefined` for it would be inconsistent with
    // itself. The refusal of an empty reply belongs one layer up, where the
    // count check compares against the number of texts sent, and that is
    // asserted separately below ("a mismatched embeddings count is refused").
    for (const body of [null, undefined, 42, "text", {}, { data: [{}] }, { embeddings: 5 }]) {
      assert.equal(embeddingsFromBody(body), undefined, JSON.stringify(body));
    }
  });

  test("an empty array in either envelope parses to an empty list, not undefined", () => {
    assert.deepEqual(embeddingsFromBody({ embeddings: [] }), []);
    assert.deepEqual(embeddingsFromBody({ data: [] }), []);
  });

  test("a data array with a non-array embedding on any entry is not usable", () => {
    // Half a reply is not a reply: returning the good half would silently
    // shorten the corpus.
    assert.equal(
      embeddingsFromBody({ data: [{ embedding: [1, 2] }, { embedding: "nope" }] }),
      undefined
    );
  });
});

describe("OllamaEmbedder — validation is in the constructor", () => {
  test("refuses to construct against a remote host under localOnly", () => {
    assert.throws(
      () =>
        new OllamaEmbedder({
          url: "https://api.example.com/v1",
          model: "nomic-embed-text",
          prefixMode: "nomic",
          localOnly: true,
        }),
      /not a literal loopback or RFC1918 address/
    );
  });

  test("CONSTRUCTS against a remote host when localOnly is not set", () => {
    // ⚠️ INVERTED. This asserted that construction against a remote provider
    // throws. It was the policy, not the library: a published MCP server whose
    // constructor refuses every hosted embedding provider is unusable for the
    // general case it advertises.
    assert.doesNotThrow(
      () =>
        new OllamaEmbedder({
          url: "https://api.openai.com/v1",
          model: "text-embedding-3-small",
          prefixMode: "none",
          apiKey: "sk-not-a-real-key",
        })
    );
  });

  test("refuses to construct with an empty model", () => {
    assert.throws(
      () =>
        new OllamaEmbedder({
          url: "http://127.0.0.1:11434",
          model: "  ",
          prefixMode: "nomic",
        }),
      /EMBEDDING_MODEL must not be empty/
    );
  });

  test("refuses an unknown prefix mode", () => {
    assert.throws(
      () =>
        new OllamaEmbedder({
          url: "http://127.0.0.1:11434",
          model: "nomic-embed-text",
          prefixMode: "openai" as never,
        }),
      /Unknown embedding prefix mode/
    );
  });

  test("refuses a non-positive batch size instead of silently defaulting", () => {
    for (const batchSize of [0, -1, 1.5]) {
      assert.throws(
        () =>
          new OllamaEmbedder({
            url: "http://127.0.0.1:11434",
            model: "nomic-embed-text",
            prefixMode: "nomic",
            batchSize,
          }),
        /batch size must be a positive whole number/,
        String(batchSize)
      );
    }
  });
});

describe("credentials", () => {
  test("NO auth header is attached when no key is configured", async () => {
    // ⚠️ INVERTED IN SCOPE, NOT IN ASSERTION. The old test was "carries no
    // credential: no auth header is ever attached", and it asserted that a
    // request carries exactly one header, `content-type`. The absence half is
    // kept verbatim below; what changed is that the absence is now a
    // consequence of there being no key rather than of there being no code.
    const captured = stubFetch(() => ({
      body: { embeddings: [vectorOfWidth(768)] },
    }));
    const embedder = new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "nomic-embed-text",
      prefixMode: "nomic",
    });
    await embedder.embedQuery("anything");
    assert.equal(captured.length, 1);
    const headerNames = Object.keys(captured[0].headers).map((name) =>
      name.toLowerCase()
    );
    for (const forbidden of ["authorization", "x-api-key", "api-key", "token"]) {
      assert.ok(
        !headerNames.includes(forbidden),
        `an embedding request must not carry ${forbidden}`
      );
    }
    // Asserted as an exact set, not a length: a header added later should break
    // this deliberately so someone decides whether it may be there.
    assert.deepEqual(headerNames, ["content-type"]);
  });

  test("the key IS sent as Authorization: Bearer when configured", async () => {
    const captured = stubFetch(() => ({
      body: { embeddings: [vectorOfWidth(768)] },
    }));
    const embedder = new OllamaEmbedder({
      url: "https://api.openai.com/v1",
      model: "text-embedding-3-small",
      prefixMode: "none",
      apiKey: "sk-test-key-123",
    });
    await embedder.embedQuery("anything");
    assert.equal(captured.length, 1);
    assert.equal(
      captured[0].headers.Authorization,
      "Bearer sk-test-key-123"
    );
  });

  test("the key never appears in an error message", async () => {
    // The failure path is the one that gets logged. A credential in a
    // diagnostic string outlives the request that used it.
    const key = "sk-super-secret-value-that-must-not-leak";
    for (const respond of [
      // A non-2xx answer.
      () => ({ status: 401, body: { error: "unauthorized" } }),
      // A transport throw.
      () => {
        throw new Error("ECONNREFUSED");
      },
      // A body of the wrong shape.
      () => ({ body: { nonsense: true } }),
    ]) {
      stubFetch(respond as never);
      const embedder = new OllamaEmbedder({
        url: "https://api.openai.com/v1",
        model: "text-embedding-3-small",
        prefixMode: "none",
        apiKey: key,
      });
      await assert.rejects(
        () => embedder.embedQuery("anything"),
        (err: Error) => {
          assert.ok(
            !err.message.includes(key),
            `the API key leaked into: ${err.message}`
          );
          assert.ok(!String(err.stack ?? "").includes(key), "key leaked into the stack");
          return true;
        }
      );
    }
  });

  test("the key is invisible to JSON, spread and Object.keys", async () => {
    // ⚠️ THIS TEST FOUND A REAL DEFECT IN ITS OWN FIRST IMPLEMENTATION.
    //
    // The field was `private readonly apiKey?: string`. `private` is erased at
    // compile time -- there is no runtime protection at all -- so the key was
    // an own ENUMERABLE property and `JSON.stringify(embedder)` carried it. The
    // KDoc two lines above it promised the opposite. The field is now a true ES
    // private (`#apiKey`), which compiles to a WeakMap slot and is invisible to
    // every reflective surface.
    stubFetch(() => ({ body: { embeddings: [vectorOfWidth(768)] } }));
    const key = "sk-another-secret";
    const embedder = new OllamaEmbedder({
      url: "https://api.openai.com/v1",
      model: "m",
      prefixMode: "none",
      apiKey: key,
    });
    await embedder.embedQuery("x");
    assert.ok(!JSON.stringify(embedder).includes(key), "JSON.stringify leaked the key");
    assert.ok(!JSON.stringify({ ...embedder }).includes(key), "spread leaked the key");
    assert.ok(!Object.keys(embedder).some((k) => k.toLowerCase().includes("key")));
    assert.ok(!Object.getOwnPropertyNames(embedder).includes("#apiKey"));
    // And it is still USED, so the privacy is not achieved by dropping it.
    assert.ok(!JSON.stringify(embedder).includes("Bearer"));
  });
});

describe("prefix handling", () => {
  test("the nomic mode applies the nomic templates, not the E5 ones", () => {
    assert.deepEqual(PREFIXES.nomic, {
      document: "search_document: ",
      query: "search_query: ",
    });
    // The reference implementation hardcoded this onto a nomic model.
    assert.notEqual(PREFIXES.nomic.query, PREFIXES.e5.query);
  });

  test("the none mode prefixes nothing", () => {
    assert.deepEqual(PREFIXES.none, { document: "", query: "" });
  });

  test("documents get the document prefix and queries get the query prefix", async () => {
    const captured = stubFetch((request) => ({
      body: {
        embeddings: request.body.input.map(() => vectorOfWidth(768)),
      },
    }));
    const embedder = new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "nomic-embed-text",
      prefixMode: "nomic",
    });
    await embedder.embedDocuments(["alpha document"]);
    await embedder.embedQuery("beta query");
    assert.deepEqual(captured[0].body.input, ["search_document: alpha document"]);
    assert.deepEqual(captured[1].body.input, ["search_query: beta query"]);
  });

  test("the e5 mode uses passage/query", async () => {
    const captured = stubFetch((request) => ({
      body: { embeddings: request.body.input.map(() => vectorOfWidth(768)) },
    }));
    const embedder = new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "e5-large",
      prefixMode: "e5",
    });
    await embedder.embedDocuments(["alpha"]);
    await embedder.embedQuery("beta");
    assert.deepEqual(captured[0].body.input, ["passage: alpha"]);
    assert.deepEqual(captured[1].body.input, ["query: beta"]);
  });

  test("the none mode sends the text untouched", async () => {
    const captured = stubFetch((request) => ({
      body: { embeddings: request.body.input.map(() => vectorOfWidth(768)) },
    }));
    const embedder = new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "some-model",
      prefixMode: "none",
    });
    await embedder.embedDocuments(["alpha"]);
    assert.deepEqual(captured[0].body.input, ["alpha"]);
  });

  test("refuses a query whose mode differs from the stored mode", () => {
    assert.doesNotThrow(() => assertPrefixModeMatches("nomic", "nomic"));
    assert.throws(
      () => assertPrefixModeMatches("nomic", "e5"),
      PrefixModeMismatchError
    );
    assert.throws(
      () => assertPrefixModeMatches("none", "nomic"),
      /returns quietly worse results|built with/
    );
  });
});

describe("transport failures", () => {
  test("a network throw becomes EmbeddingUnavailableError", async () => {
    globalThis.fetch = (async () => {
      throw new Error("ECONNREFUSED");
    }) as unknown as typeof fetch;
    const embedder = new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "nomic-embed-text",
      prefixMode: "nomic",
    });
    await assert.rejects(
      () => embedder.embedQuery("x"),
      (err: unknown) => {
        assert.ok(err instanceof EmbeddingUnavailableError);
        assert.match(String(err), /did not answer at http:\/\/127\.0\.0\.1:11434\/api\/embed/);
        return true;
      }
    );
  });

  test("a non-2xx status becomes EmbeddingUnavailableError", async () => {
    stubFetch(() => ({ status: 500, body: { error: "boom" } }));
    const embedder = new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "nomic-embed-text",
      prefixMode: "nomic",
    });
    await assert.rejects(() => embedder.embedQuery("x"), /status 500/);
  });

  test("a redirect is refused rather than followed — the local-only guard's request-time half", async () => {
    // The gate validates the endpoint ONCE, in the constructor. Without
    // `redirect: "error"`, a 307 from the local endpoint would forward the body
    // -- restricted document text -- to whatever host the redirect names, and
    // the gate would still read as satisfied. So the assertion is on the
    // transport option itself, not on a status code: the failure has to happen
    // BEFORE the request is forwarded, which means it must be fetch that
    // refuses, and undici refuses when told to.
    let followedTo: string | undefined;
    const captured: CapturedRequest[] = [];
    globalThis.fetch = (async (url: string | URL, init?: RequestInit) => {
      captured.push({
        url: String(url),
        body: JSON.parse(String(init?.body ?? "{}")),
        headers: (init?.headers ?? {}) as Record<string, string>,
        init: init ?? {},
      });
      // Emulate undici: it only follows when the option permits it.
      if (init?.redirect !== "error") {
        followedTo = "http://attacker.example/collect";
      }
      return new Response("moved", {
        status: 307,
        headers: { Location: "http://attacker.example/collect" },
      });
    }) as unknown as typeof fetch;

    const embedder = new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "nomic-embed-text",
      prefixMode: "nomic",
    });

    // The option must be present, on the one outbound fetch.
    assert.equal(captured.length, 0);
    await assert.rejects(
      () => embedder.embedQuery("confidential sample text"),
      (err: unknown) => {
        assert.ok(err instanceof EmbeddingUnavailableError);
        assert.match(String(err), /307/);
        return true;
      }
    );
    assert.equal(captured.length, 1);
    assert.equal(captured[0].init.redirect, "error");
    // The proof that the body was NOT forwarded.
    assert.equal(followedTo, undefined);
  });

  test("probe() exercises a real embedding, not a bare GET", async () => {
    // A probe that only pings the server's root would pass against an Ollama
    // that is up with the model un-pulled, and then fail on the first real
    // request -- after `force:true` has already deleted the index. So the probe
    // must hit the same endpoint with the same model as the real work.
    const captured = stubFetch(() => ({ body: { embeddings: [vectorOfWidth(768)] } }));
    const embedder = new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "nomic-embed-text",
      prefixMode: "nomic",
    });
    await embedder.probe();
    assert.equal(captured.length, 1);
    assert.match(captured[0].url, /\/api\/embed/);
    assert.equal(captured[0].body.model, "nomic-embed-text");
    // A constant, never document text: a liveness check must not put restricted
    // material on the wire. Sent UNPREFIXED deliberately -- the probe asks "will
    // this model embed at all", and a model that fails only on a prefix is not
    // a failure mode anyone has observed. The prefix is validated at build time
    // by the metadata assertion, not by the liveness probe.
    assert.deepEqual(captured[0].body.input, ["ping"]);
  });

  test("probe() rejects when the model is not pulled, even though the server is up", async () => {
    stubFetch(() => ({ status: 404, body: { error: "model not found" } }));
    const embedder = new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "nomic-embed-text",
      prefixMode: "nomic",
    });
    await assert.rejects(() => embedder.probe(), EmbeddingUnavailableError);
  });

  test("a mismatched embeddings count is refused rather than truncated", async () => {
    stubFetch(() => ({ body: { embeddings: [] } }));
    const embedder = new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "nomic-embed-text",
      prefixMode: "nomic",
    });
    await assert.rejects(() => embedder.embedQuery("x"), /mismatched embeddings array/);
  });

  test("a wrong-width vector is refused, naming the model as the suspect", async () => {
    stubFetch(() => ({ body: { embeddings: [vectorOfWidth(1024)] } }));
    const embedder = new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "wrong-model",
      prefixMode: "nomic",
    });
    await assert.rejects(
      () => embedder.embedQuery("x"),
      /expected 768 dimensions[\s\S]*EMBEDDING_MODEL/
    );
  });
});

describe("batching", () => {
  test("defaults to sixteen per request when no batchSize is given", async () => {
    const captured = stubFetch((request) => ({
      body: { embeddings: request.body.input.map(() => vectorOfWidth(768)) },
    }));
    const embedder = new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "nomic-embed-text",
      prefixMode: "nomic",
    });
    const texts = Array.from({ length: 20 }, (_, index) => `doc ${index}`);
    const result = await embedder.embedDocuments(texts);
    assert.equal(result.length, 20);
    assert.deepEqual(
      captured.map((request) => request.body.input.length),
      [16, 4]
    );
  });

  test("a configured batchSize governs the request count: 7 texts at 3 is 3,3,1", async () => {
    // ⚠️ THE WIRING TEST FOR PAPERLESS_MCP_SYNC_BATCH_SIZE, AND IT IS NOT A TEST
    // OF THE OPTION. A test that passes 16 and asserts 16 would pass against a
    // module constant that never reads the option at all, which is exactly the
    // defect being fixed: the variable was read and range-checked in config.ts
    // and consumed by nothing. The numbers here are deliberately not the
    // default and not a divisor of the text count, so a hardcoded 16 produces
    // [7] rather than [3,3,1].
    const captured = stubFetch((request) => ({
      body: { embeddings: request.body.input.map(() => vectorOfWidth(768)) },
    }));
    const embedder = new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "nomic-embed-text",
      prefixMode: "nomic",
      batchSize: 3,
    });
    const texts = Array.from({ length: 7 }, (_, index) => `doc ${index}`);
    const result = await embedder.embedDocuments(texts);
    assert.equal(result.length, 7);
    assert.deepEqual(
      captured.map((request) => request.body.input.length),
      [3, 3, 1]
    );
  });

  test("a batchSize larger than the input makes exactly one request", async () => {
    const captured = stubFetch((request) => ({
      body: { embeddings: request.body.input.map(() => vectorOfWidth(768)) },
    }));
    const embedder = new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "nomic-embed-text",
      prefixMode: "nomic",
      batchSize: 64,
    });
    await embedder.embedDocuments(["a", "b", "c"]);
    assert.equal(captured.length, 1);
    assert.equal(captured[0].body.input.length, 3);
  });

  test("an empty document list makes no request at all", async () => {
    const captured = stubFetch(() => ({ body: { embeddings: [] } }));
    const embedder = new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "nomic-embed-text",
      prefixMode: "nomic",
    });
    assert.deepEqual(await embedder.embedDocuments([]), []);
    assert.equal(captured.length, 0);
  });
});

describe("an http error carries the server's own explanation", () => {
  test("a rejected request surfaces the server's message, not just the status", async () => {
    // A bare "status 400" is undiagnosable. The whole cost of the probe that
    // found this was re-deriving by hand what the server would have said.
    stubFetch(() => ({
      status: 400,
      body: { error: "the input length exceeds the context length" },
    }));
    const embedder = new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "nomic-embed-text",
      prefixMode: "none",
    });
    await assert.rejects(
      () => embedder.embedQuery("x"),
      /status 400[\s\S]*input length exceeds the context length/
    );
  });

  test("a long error body is truncated to one line", async () => {
    // An error page can be an entire HTML document. An unbounded string inside
    // an exception message is its own defect.
    const long = `<html>\n<body>${"reason ".repeat(400)}</body></html>`;
    stubFetch(() => ({ status: 422, text: long }));
    const embedder = new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "m",
      prefixMode: "none",
    });
    let message = "";
    try {
      await embedder.embedQuery("x");
      assert.fail("expected a rejection");
    } catch (err) {
      message = String(err);
    }
    assert.match(message, /status 422/);
    assert.ok(!message.includes("\n"), "the detail must be collapsed to one line");
    // 300 characters of detail plus the surrounding sentence.
    assert.ok(message.length < 500, `message was ${message.length} chars`);
  });

  test("a configured API key never appears in an http error", async () => {
    const key = "sk-live-do-not-print";
    stubFetch(() => ({ status: 401, body: { error: "invalid api key" } }));
    const embedder = new OllamaEmbedder({
      url: "https://api.example.com/v1",
      model: "m",
      prefixMode: "none",
      apiKey: key,
    });
    let message = "";
    try {
      await embedder.embedQuery("x");
      assert.fail("expected a rejection");
    } catch (err) {
      message = String(err);
    }
    assert.match(message, /status 401/);
    assert.ok(!message.includes(key), "the credential leaked into the error");
    assert.ok(!message.includes("Bearer"), "the authorization header leaked");
  });
});

describe("both reply shapes, end to end through the client", () => {
  test("an OpenAI-shaped server and an Ollama-shaped server yield the same vectors", async () => {
    // The sniff is asserted at the parser level above; this proves the CLIENT
    // reaches the parser with both, which is what a real provider swap depends
    // on. Same numbers in, same numbers out, different envelopes.
    const expected = vectorOfWidth(768, 0.125);
    const native = stubFetch(() => ({ body: { embeddings: [expected] } }));
    const viaNative = await new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "m",
      prefixMode: "none",
    }).embedQuery("q");
    assert.equal(native.length, 1);
    assert.deepEqual(viaNative, expected);

    const openai = stubFetch(() => ({
      body: { object: "list", data: [{ object: "embedding", embedding: expected }] },
    }));
    const viaOpenAi = await new OllamaEmbedder({
      url: "https://api.openai.com/v1",
      model: "m",
      prefixMode: "none",
      apiKey: "k",
    }).embedQuery("q");
    assert.equal(openai.length, 1);
    assert.deepEqual(viaOpenAi, expected);
    assert.deepEqual(viaNative, viaOpenAi);
  });

  test("an unrecognised body is an EmbeddingUnavailableError naming both shapes", async () => {
    stubFetch(() => ({ body: { predictions: [[1, 2, 3]] } }));
    const embedder = new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "m",
      prefixMode: "none",
      dimensions: 3,
    });
    await assert.rejects(
      () => embedder.embedQuery("q"),
      (err: Error) => {
        assert.ok(err instanceof EmbeddingUnavailableError);
        assert.match(err.message, /OpenAI-compatible/);
        assert.match(err.message, /Ollama/);
        return true;
      }
    );
  });
});

describe("transport options", () => {
  test("redirect is always 'error', on every endpoint", async () => {
    // Generic hygiene, not policy: a 307 would let any endpoint forward the
    // request body to a host the configuration never named.
    for (const url of ["http://127.0.0.1:11434", "https://api.openai.com/v1"]) {
      const captured = stubFetch(() => ({
        body: { embeddings: [vectorOfWidth(768)] },
      }));
      const embedder = new OllamaEmbedder({
        url,
        model: "m",
        prefixMode: "none",
        apiKey: "k",
      });
      await embedder.embedQuery("q");
      assert.equal(captured[0].init.redirect, "error", url);
    }
  });
});

// ── Over-long inputs: split and retry ──────────────────────────────────────
// The run this exists for died at document 1,727 of 2,466 with a 400 whose body
// read "the input length exceeds the context length". It is a TOKEN budget
// being reported against a CHARACTER limit: a chunk of 2,000 characters of dot
// leaders tokenises past a 2,048-token window while 2,000 characters of prose
// uses about a quarter of it. `truncate: true` was measured and does not help.
//
// So the fix cannot be a smaller character budget -- that would re-chunk the
// whole corpus and still fail on the next document with a denser one. It has to
// be a reaction to the REJECTION. Every test below therefore fails against a
// client that treats this 400 like any other.

/** A server that refuses any single input longer than `limit` characters. */
function tokenWindowServer(limit: number) {
  return (request: CapturedRequest) => {
    const overLong = request.body.input.some((text) => text.length > limit);
    if (overLong) {
      return {
        status: 400,
        body: { error: "the input length exceeds the context length" },
      };
    }
    // Length-keyed axis, so a test can tell WHICH text produced a vector and
    // therefore prove the order survived the split.
    return {
      body: {
        embeddings: request.body.input.map((text) => {
          const vector = new Array<number>(768).fill(0);
          vector[text.length % 768] = 1;
          return vector;
        }),
      },
    };
  };
}

function unitNorm(vector: number[]): number {
  return Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
}

describe("a context-length rejection is split, not treated as an outage", () => {
  test("the whole batch survives and stays in order when one text is too long", async () => {
    // Without the split this rejects: the first request contains the over-long
    // text, the client raises EmbeddingUnavailableError, and every remaining
    // text in the batch is lost. That is precisely how 739 documents were
    // skipped by a failure that was about 1,727 of them.
    const limit = 300;
    const captured = stubFetch(tokenWindowServer(limit));
    const embedder = new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "nomic-embed-text",
      prefixMode: "none",
      batchSize: 16,
    });
    const short = Array.from({ length: 15 }, (_, index) =>
      String.fromCharCode(97 + index).repeat(40 + index)
    );
    const overLong = "d".repeat(900);
    const result = await embedder.embedDocuments([...short, overLong]);

    assert.equal(result.length, 16, "every text must get a vector");
    // The short texts were never split, so each keeps its length-keyed axis and
    // its position. A split that reordered the batch would put a different axis
    // at these indices.
    for (let index = 0; index < short.length; index += 1) {
      const axis = short[index].length % 768;
      assert.equal(result[index][axis], 1, `text ${index} lost its vector`);
    }
    assert.ok(
      captured.length > 1,
      "the client must have retried in halves rather than giving up"
    );
  });

  test("both halves of a split text contribute to the one vector", async () => {
    // ⚠️ THIS TEST EXISTS BECAUSE A WEAKER ONE PASSED A DELIBERATE DEFECT.
    //
    // An earlier version asserted only `result.length === 1` and `norm === 1`.
    // Mutating the code to embed the left half and SILENTLY DROP the right half
    // left both assertions green -- one vector, unit length, plausible-looking.
    // That is precisely the "silently drop content" failure the whole design is
    // trying to make impossible, so the test has to see BOTH halves.
    //
    // The marker server returns a vector whose components say WHICH markers the
    // text contained. A correct mean has both; a dropped tail has only the one.
    const captured: CapturedRequest[] = [];
    globalThis.fetch = (async (url: string | URL, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body ?? "{}"));
      captured.push({
        url: String(url),
        body,
        headers: {},
        init: init ?? {},
      });
      if (body.input.some((text: string) => text.length > 300)) {
        return new Response(
          JSON.stringify({ error: "the input length exceeds the context length" }),
          { status: 400, headers: { "Content-Type": "application/json" } }
        );
      }
      return new Response(
        JSON.stringify({
          embeddings: body.input.map((text: string) => {
            const vector = new Array<number>(768).fill(0);
            if (text.includes("ALPHA")) vector[0] = 1;
            if (text.includes("OMEGA")) vector[1] = 1;
            return vector;
          }),
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      );
    }) as typeof fetch;

    const embedder = new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "nomic-embed-text",
      prefixMode: "none",
    });
    // The two markers land on opposite sides of every split point, so a correct
    // combination is the only thing that can produce both components.
    const text = "ALPHA ".repeat(100) + "OMEGA ".repeat(100);
    const result = await embedder.embedDocuments([text]);

    assert.equal(result.length, 1, "one chunk stays one row");
    assert.equal(result[0].length, 768);
    assert.ok(
      Math.abs(unitNorm(result[0]) - 1) < 1e-9,
      "a cosine store compares direction, so the combined vector must be unit length"
    );
    assert.ok(
      result[0][0] > 0,
      "the ALPHA half contributed nothing -- the head was dropped"
    );
    assert.ok(
      result[0][1] > 0,
      "the OMEGA half contributed nothing -- the tail was silently dropped"
    );
    assert.ok(captured.some((request) => (request.body.input as string[]).some(
      (text: string) => text.includes("OMEGA")
    )), "the tail half must actually have been sent");
  });

  test("an unrelated 400 is still an outage and is NOT retried", async () => {
    // Splitting on every 400 would turn one visible failure into a cascade of
    // invisible ones, and would hammer a server that is rejecting us for a
    // reason a smaller request cannot fix.
    const captured = stubFetch(() => ({
      status: 400,
      body: { error: "model 'nomic-embed-text' not found, try pulling it first" },
    }));
    const embedder = new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "nomic-embed-text",
      prefixMode: "none",
    });
    await assert.rejects(
      () => embedder.embedDocuments(["a", "b", "c", "d"]),
      (err: unknown) =>
        err instanceof EmbeddingUnavailableError &&
        !(err instanceof EmbeddingContextLengthError)
    );
    assert.equal(captured.length, 1, "no retry, no split, one request");
  });

  test("a 5xx is an outage and is NOT retried", async () => {
    const captured = stubFetch(() => ({ status: 503, body: {} }));
    const embedder = new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "nomic-embed-text",
      prefixMode: "none",
    });
    await assert.rejects(
      () => embedder.embedDocuments(["a", "b"]),
      EmbeddingUnavailableError
    );
    assert.equal(captured.length, 1);
  });

  test("a fragment at the floor fails the document whole instead of dropping it", async () => {
    // ⚠️ THE HONEST-FAILURE TEST. A shorter-than-floor fragment means the
    // model window is smaller than this index was built against, and splitting
    // again produces non-words. The alternative -- quietly skipping the chunk
    // -- leaves an index that reports a document it holds only part of, and no
    // count anywhere would show it.
    const captured: CapturedRequest[] = [];
    globalThis.fetch = (async (url: string | URL, init?: RequestInit) => {
      captured.push({
        url: String(url),
        body: JSON.parse(String(init?.body ?? "{}")),
        headers: {},
        init: init ?? {},
      });
      return new Response(
        JSON.stringify({ error: "the input length exceeds the context length" }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }) as typeof fetch;
    const embedder = new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "nomic-embed-text",
      prefixMode: "none",
    });
    // ⚠️ THE ASSERTION NAMES THE FLOOR, NOT JUST THE FAILURE. A weaker version
    // asserted only that SOME EmbeddingUnavailableError mentioning "no content
    // was dropped" was raised -- and a mutation that DELETED the floor check
    // still passed, because recursion ran down to a one-character fragment and
    // threw the different "no boundary to split at" error, which reads the same
    // from the outside. Deleting a safety check must not be invisible.
    await assert.rejects(
      () => embedder.embedDocuments(["x".repeat(MIN_SPLITTABLE_CHARS * 4)]),
      (err: unknown) =>
        err instanceof EmbeddingUnavailableError &&
        new RegExp(
          `splitting down to the ${MIN_SPLITTABLE_CHARS}-character floor`
        ).test((err as Error).message) &&
        /No content was dropped/.test((err as Error).message)
    );
    // Bounded: it split down to the floor and then stopped. An unbounded
    // recursion would keep issuing requests forever.
    assert.ok(captured.length < 32, `recursion must terminate (${captured.length})`);
  });

  test("the split path preserves the document prefix on both halves", async () => {
    // The prefix is a MEANING carrier for nomic ("search_document: "). Applying
    // it once at the top and then splitting would send bare fragments, whose
    // vectors live in a different region of the space than the queries.
    const captured = stubFetch(tokenWindowServer(300));
    const embedder = new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "nomic-embed-text",
      prefixMode: "nomic",
    });
    await embedder.embedDocuments(["word ".repeat(400)]);
    assert.ok(captured.length >= 3);
    for (const request of captured) {
      for (const text of request.body.input) {
        assert.ok(
          text.startsWith(PREFIXES.nomic.document),
          "every dispatched fragment must carry the document prefix"
        );
      }
    }
  });

  test("splits are reported as counts and lengths, never as text", async () => {
    // The sync logs this, so it is a surface document content could reach. It
    // carries a length and a reason; there is no field a text could hide in.
    stubFetch(tokenWindowServer(300));
    const embedder = new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "nomic-embed-text",
      prefixMode: "none",
    });
    await embedder.embedDocuments(["word ".repeat(400)]);
    const splits = embedder.lastSplits;
    assert.ok(splits.length > 0, "the fallback fired, so it must be visible");
    for (const event of splits) {
      assert.deepEqual(
        Object.keys(event).sort(),
        ["chars", "depth", "reason", "texts"],
        "a split event must expose only counts and lengths"
      );
      assert.ok(["batch", "text"].includes(event.reason));
    }
  });

  test("a query is never split, so a too-long query still fails loudly", async () => {
    // A query is the user's own words. Silently embedding the mean of two
    // halves of their sentence would answer a question they did not ask.
    stubFetch(() => ({
      status: 400,
      body: { error: "the input length exceeds the context length" },
    }));
    const embedder = new OllamaEmbedder({
      url: "http://127.0.0.1:11434",
      model: "nomic-embed-text",
      prefixMode: "none",
    });
    await assert.rejects(() => embedder.embedQuery("x".repeat(9000)), EmbeddingUnavailableError);
  });
});

describe("split boundaries never cut a grapheme", () => {
  test("graphemeBoundaries counts a stacked Tibetan syllable as one unit", () => {
    // U+0F40 (TIBETAN LETTER KA, category Lo) + U+0F74 (vowel sign U, category
    // Mn) is one syllable in two code points. This is why a naive slice is not
    // merely inelegant on this corpus -- it produces a base with a stranded
    // vowel sign, which is not a word, embedded as if it were.
    const syllable = "\u0F40\u0F74";
    assert.deepEqual(graphemeBoundaries(syllable), [0, 2]);
  });

  test("graphemeBoundaries does not split a run of two syllables", () => {
    // The two-base fixture is the one that matters: a segmenter that glued an
    // unbounded run of combining marks would report one cluster for any length,
    // and the splitter below would then have no boundary to use at all.
    const text = "\u0F40\u0F74".repeat(2);
    assert.deepEqual(graphemeBoundaries(text), [0, 2, 4]);
  });

  test("a whitespace-free Tibetan run splits between syllables, not inside one", () => {
    // One ASCII character of padding makes the midpoint FALL INSIDE a cluster:
    // boundaries are 0,1,3,5..., so a naive slice at 200 lands between a base
    // and its vowel sign. Every grapheme boundary here is odd above 1.
    const text = "a" + "\u0F40\u0F74".repeat(200);
    assert.equal(text.length % 2, 1, "the fixture must have an odd length");
    const midpoint = Math.floor(text.length / 2);
    assert.ok(
      graphemeBoundaries(text).includes(midpoint) === false,
      "the midpoint must NOT be a grapheme boundary, or the test proves nothing"
    );
    const halves = splitForEmbedding(text);
    assert.ok(halves, "a 401-character text must be splittable");
    for (const half of halves) {
      assert.ok(
        !/^\p{M}/u.test(half),
        "a fragment must never begin with a combining mark"
      );
    }
    assert.equal(halves[0].length + halves[1].length, text.length);
  });

  test("an unbreakable run still reports undefined rather than looping", () => {
    // One character, no boundary of any kind. The caller throws; it does not
    // recurse forever on a text it cannot shorten.
    assert.equal(splitForEmbedding("x"), undefined);
  });
});

describe("helpers the split path is built from", () => {
  test("isContextLengthRejection keys on the server text, and only on a 400", () => {
    const body = "the input length exceeds the context length";
    assert.equal(isContextLengthRejection(400, body), true);
    assert.equal(isContextLengthRejection(400, body.toUpperCase()), true);
    assert.equal(isContextLengthRejection(500, body), false);
    assert.equal(isContextLengthRejection(401, body), false);
    assert.equal(isContextLengthRejection(400, "something else entirely"), false);
  });

  test("renormalisedMean returns direction at unit length", () => {
    const a = new Array<number>(768).fill(0);
    a[0] = 1;
    const b = new Array<number>(768).fill(0);
    b[1] = 1;
    const mean = renormalisedMean(a, b);
    assert.ok(Math.abs(unitNorm(mean) - 1) < 1e-12);
    assert.ok(Math.abs(mean[0] - mean[1]) < 1e-12, "both halves weigh the same");
  });

  test("renormalisedMean refuses opposed halves rather than storing a dead row", () => {
    // A zero vector has no direction: it would sit in the index matching
    // nothing, indistinguishable from a chunk with no content.
    const a = new Array<number>(768).fill(0);
    a[0] = 1;
    const b = new Array<number>(768).fill(0);
    b[0] = -1;
    assert.throws(() => renormalisedMean(a, b), EmbeddingUnavailableError);
  });

  test("renormalisedMean refuses mismatched widths", () => {
    assert.throws(
      () => renormalisedMean([1, 0], [1, 0, 0]),
      /different widths/
    );
  });
});
