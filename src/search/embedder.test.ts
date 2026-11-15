import assert from "node:assert/strict";
import { afterEach, describe, test } from "node:test";
import {
  EmbeddingUnavailableError,
  OllamaEmbedder,
  PREFIXES,
  PrefixModeMismatchError,
  assertLocalEmbeddingEndpoint,
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
}

function stubFetch(
  respond: (req: CapturedRequest) => { status?: number; body: unknown }
): CapturedRequest[] {
  const captured: CapturedRequest[] = [];
  globalThis.fetch = (async (url: string | URL, init?: RequestInit) => {
    const headers = (init?.headers ?? {}) as Record<string, string>;
    const request: CapturedRequest = {
      url: String(url),
      body: JSON.parse(String(init?.body ?? "{}")),
      headers,
    };
    captured.push(request);
    const result = respond(request);
    const status = result.status ?? 200;
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

// ── The local-only guard ────────────────────────────────────────────────────────
// A gate that only passes its happy path is untested. Every assertion below is
// a negative case first.

describe("assertLocalEmbeddingEndpoint — the local-only guard", () => {
  test("accepts loopback", () => {
    for (const url of ["http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434"]) {
      assert.doesNotThrow(() => assertLocalEmbeddingEndpoint(url), url);
    }
  });

  test("accepts literal RFC1918 addresses", () => {
    for (const url of [
      "http://127.0.0.1:11434",
      "http://192.0.2.10:11434",
      "http://172.16.0.1:11434",
      "http://172.31.255.254:11434",
      "https://192.0.2.20:11434/v1",
    ]) {
      assert.doesNotThrow(() => assertLocalEmbeddingEndpoint(url), url);
    }
  });

  test("refuses a public DNS name", () => {
    assert.throws(
      () => assertLocalEmbeddingEndpoint("http://example.com"),
      /not a literal loopback or RFC1918 address/
    );
  });

  test("refuses a real remote provider, which is the whole point", () => {
    assert.throws(
      () => assertLocalEmbeddingEndpoint("https://api.example.com/v1"),
      /not a literal loopback or RFC1918 address/
    );
    assert.throws(
      () => assertLocalEmbeddingEndpoint("https://api.openai.com/v1"),
      /not a literal loopback or RFC1918 address/
    );
  });

  test("refuses a public literal address", () => {
    assert.throws(
      () => assertLocalEmbeddingEndpoint("http://8.8.8.8:11434"),
      /public address/
    );
  });

  test("refuses 172.15 and 172.32, which sit outside the RFC1918 window", () => {
    for (const url of ["http://172.15.1.1:11434", "http://172.32.1.1:11434"]) {
      assert.throws(() => assertLocalEmbeddingEndpoint(url), /public address/, url);
    }
  });

  test("refuses 0.0.0.0 with an explanation naming the real fix", () => {
    // Ollama BINDS 0.0.0.0 on this box, so the obvious naive check would be
    // wrong in the opposite direction. Dialing the unspecified address is still
    // not a destination.
    assert.throws(
      () => assertLocalEmbeddingEndpoint("http://0.0.0.0:11434"),
      /unspecified address/
    );
  });

  test("refuses a non-http protocol", () => {
    assert.throws(
      () => assertLocalEmbeddingEndpoint("file:///etc/passwd"),
      /must use http: or https:/
    );
    assert.throws(
      () => assertLocalEmbeddingEndpoint("ftp://127.0.0.1:21"),
      /must use http: or https:/
    );
  });

  test("refuses an unparseable URL", () => {
    assert.throws(
      () => assertLocalEmbeddingEndpoint("not a url at all"),
      /not a valid URL/
    );
  });

  test("refuses an octet above 255", () => {
    // The WHATWG URL parser rejects this before our allowlist ever runs, so the
    // message is the parse failure rather than the allowlist refusal. Asserting
    // on the message would pin us to which check happens to fire first; what
    // matters is that the endpoint is refused. Measured, not assumed.
    assert.throws(() => assertLocalEmbeddingEndpoint("http://999.0.0.1:11434"));
  });
});

describe("OllamaEmbedder — the gate is in the constructor", () => {
  test("refuses to construct against a remote host", () => {
    assert.throws(
      () =>
        new OllamaEmbedder({
          url: "https://api.example.com/v1",
          model: "nomic-embed-text",
          prefixMode: "nomic",
        }),
      /not a literal loopback or RFC1918 address/
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

  test("carries no credential: no auth header is ever attached", async () => {
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
    assert.equal(headerNames.length, 1);
    assert.equal(headerNames[0], "content-type");
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
  test("splits documents into batches of sixteen", async () => {
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
