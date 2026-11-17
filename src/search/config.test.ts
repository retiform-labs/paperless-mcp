import assert from "node:assert/strict";
import { homedir } from "node:os";
import { join } from "node:path";
import { describe, test } from "node:test";
import {
  DEFAULT_EMBEDDING_MODEL,
  DEFAULT_EMBEDDING_URL,
  DEFAULT_MAX_CHUNKS,
  DEFAULT_PREFIX_MODE,
  DEFAULT_SYNC_INTERVAL_MS,
  defaultIndexDir,
  expandHome,
  indexPath,
  searchConfig,
  webhookRefusal,
  CHUNK_ROWID_STRIDE,
  MAX_CHUNKS_LIMIT,
} from "./config";

describe("searchConfig", () => {
  test("returns undefined when nothing is configured", () => {
    assert.equal(searchConfig({}), undefined);
  });

  test("treats a blank value as unset, so whitespace does not enable the feature", () => {
    assert.equal(searchConfig({ PAPERLESS_MCP_EMBEDDING_MODEL: "   " }), undefined);
  });

  test("any single variable enables the feature and everything else defaults", () => {
    const config = searchConfig({ PAPERLESS_MCP_INDEX_DIR: "/tmp/idx" });
    assert.ok(config);
    assert.equal(config.embeddingUrl, DEFAULT_EMBEDDING_URL);
    assert.equal(config.model, DEFAULT_EMBEDDING_MODEL);
    assert.equal(config.prefixMode, DEFAULT_PREFIX_MODE);
    assert.equal(config.maxChunks, DEFAULT_MAX_CHUNKS);
    assert.equal(config.syncIntervalMs, DEFAULT_SYNC_INTERVAL_MS);
    assert.equal(config.indexDir, "/tmp/idx");
  });

  test("background sync is off by default, never on", () => {
    const config = searchConfig({ PAPERLESS_MCP_INDEX_DIR: "/tmp/idx" });
    assert.equal(config?.syncIntervalMs, 0);
  });

  test("a namespaced variable beats its generic fallback", () => {
    const config = searchConfig({
      PAPERLESS_MCP_EMBEDDING_MODEL: "nomic-embed-text",
      EMBEDDING_MODEL: "mxbai-embed-large",
    });
    assert.equal(config?.model, "nomic-embed-text");
  });

  test("a generic fallback still works on its own", () => {
    const config = searchConfig({ EMBEDDING_MODEL: "mxbai-embed-large" });
    assert.equal(config?.model, "mxbai-embed-large");
  });

  test("rejects an unknown prefix mode instead of silently defaulting", () => {
    // A silent fallback here would change every stored vector without a word.
    assert.throws(
      () => searchConfig({ PAPERLESS_MCP_EMBEDDING_PREFIX_MODE: "openai" }),
      /must be one of nomic, none, e5/
    );
  });

  test("accepts each documented prefix mode", () => {
    for (const mode of ["nomic", "none", "e5"] as const) {
      const config = searchConfig({ PAPERLESS_MCP_EMBEDDING_PREFIX_MODE: mode });
      assert.equal(config?.prefixMode, mode);
    }
  });

  test("rejects a non-numeric chunk bound", () => {
    assert.throws(
      () => searchConfig({ PAPERLESS_MCP_MAX_CHUNKS: "many" }),
      /must be a number/
    );
  });

  test("rejects a fractional chunk bound", () => {
    assert.throws(
      () => searchConfig({ PAPERLESS_MCP_MAX_CHUNKS: "12.5" }),
      /must be a whole number/
    );
  });

  test("rejects a chunk bound of zero", () => {
    // The INTENT of this test is unchanged: zero is a mistake, not a ceiling,
    // and clamping it would hide it. Only the upper number in the message moved,
    // because 100000 was itself the bug -- the rowid stride is 8192, so 8191 is
    // the largest chunk count the store can encode. The old number was a
    // promise config could not keep: it accepted a value that `chunkRowid`
    // would only reject hours into a build, on document two thousand.
    assert.throws(
      () => searchConfig({ PAPERLESS_MCP_MAX_CHUNKS: "0" }),
      /must be between 1 and 8191/
    );
  });

  test("clamps an over-large chunk bound to what the stride can encode", () => {
    // Above the ceiling is a CAPABILITY question, not a mistake: the caller
    // asked for more chunks than the index can address, which means "as many as
    // possible". It must clamp rather than throw, and it must clamp to the
    // stride-derived bound rather than to a second literal.
    const config = searchConfig({ PAPERLESS_MCP_MAX_CHUNKS: "100000" });
    assert.equal(config?.maxChunks, MAX_CHUNKS_LIMIT);
    assert.equal(config?.maxChunks, CHUNK_ROWID_STRIDE - 1);
  });

  test("a chunk bound inside the encodable range is taken as given", () => {
    // The clamp must not swallow ordinary values: only the above-ceiling case
    // is a ceiling rather than a mistake.
    const config = searchConfig({ PAPERLESS_MCP_MAX_CHUNKS: "4000" });
    assert.equal(config?.maxChunks, 4000);
  });

  test("accepts a sync interval of exactly zero as an explicit off switch", () => {
    const config = searchConfig({ PAPERLESS_MCP_SYNC_INTERVAL_MS: "0" });
    assert.equal(config?.syncIntervalMs, 0);
  });

  test("expands a leading tilde in the index directory", () => {
    const config = searchConfig({ PAPERLESS_MCP_INDEX_DIR: "~/my-index" });
    assert.equal(config?.indexDir, join(homedir(), "my-index"));
  });

  test("defaults the index directory under the user's data home", () => {
    assert.equal(
      defaultIndexDir(),
      join(homedir(), ".local", "share", "paperless-mcp", "index")
    );
  });

  test("indexPath appends a single file inside the directory", () => {
    const config = searchConfig({ PAPERLESS_MCP_INDEX_DIR: "/tmp/idx" });
    assert.ok(config);
    assert.equal(indexPath(config), join("/tmp/idx", "index.db"));
  });
});

describe("embedding endpoint policy and credentials", () => {
  test("local-only is OFF when the variable is absent", () => {
    // The default is a claim about behaviour, so it gets an assertion. An
    // install pointing at a hosted provider must not have to fight a policy it
    // never asked for.
    const config = searchConfig({ PAPERLESS_MCP_EMBEDDING_URL: "https://api.example.com/v1" });
    assert.equal(config?.embeddingLocalOnly, false);
  });

  test("local-only turns on from the variable, in either accepted spelling", () => {
    for (const raw of ["true", "TRUE", "1"]) {
      const config = searchConfig({ PAPERLESS_MCP_EMBEDDING_LOCAL_ONLY: raw });
      assert.equal(config?.embeddingLocalOnly, true, raw);
    }
  });

  test("local-only turns off explicitly, in either accepted spelling", () => {
    for (const raw of ["false", "FALSE", "0"]) {
      const config = searchConfig({ PAPERLESS_MCP_EMBEDDING_LOCAL_ONLY: raw });
      assert.equal(config?.embeddingLocalOnly, false, raw);
    }
  });

  test("a mistyped flag REFUSES rather than defaulting to off", () => {
    // ⚠️ The failure mode this prevents is the bad one: an operator who set a
    // safety flag, mistyped it, got the default, and was told by a clean
    // startup that the flag was in effect. A permissive boolean parser is a
    // silent lie about configuration.
    for (const raw of ["yes", "no", "on", "off", "ture", "2"]) {
      assert.throws(
        () => searchConfig({ PAPERLESS_MCP_EMBEDDING_LOCAL_ONLY: raw }),
        /must be true or false/,
        raw
      );
    }
  });

  test("an API key alone does NOT enable the feature", () => {
    // A credential is a modifier, not a request. An environment that happens to
    // export a provider key must not silently switch on semantic search for an
    // install that never configured an index.
    assert.equal(searchConfig({ PAPERLESS_MCP_EMBEDDING_API_KEY: "sk-x" }), undefined);
  });

  test("the API key is carried alongside the rest of the config", () => {
    const config = searchConfig({
      PAPERLESS_MCP_EMBEDDING_URL: "https://api.example.com/v1",
      PAPERLESS_MCP_EMBEDDING_API_KEY: "sk-abc",
    });
    assert.equal(config?.embeddingApiKey, "sk-abc");
  });

  test("no key means undefined, not an empty string", () => {
    // An empty string would produce `Authorization: Bearer ` — a header that
    // exists, is malformed, and changes what the server does.
    const config = searchConfig({ PAPERLESS_MCP_EMBEDDING_URL: "http://127.0.0.1:11434" });
    assert.equal(config?.embeddingApiKey, undefined);
  });
});

describe("expandHome", () => {
  test("leaves an absolute path alone", () => {
    assert.equal(expandHome("/var/lib/x"), "/var/lib/x");
  });

  test("expands a bare tilde", () => {
    assert.equal(expandHome("~"), homedir());
  });

  test("does not expand a tilde that is not a path prefix", () => {
    assert.equal(expandHome("/tmp/~backup"), "/tmp/~backup");
  });
});

describe("webhook configuration", () => {
  test("a secret on its own enables NOTHING", () => {
    // A secret is a modifier on the port, never an enabler of the feature. Set
    // alone it must not turn on semantic search, and it must certainly not
    // cause a socket: an install that upgrades has to acquire nothing it did
    // not ask for.
    assert.equal(searchConfig({ PAPERLESS_MCP_WEBHOOK_SECRET: "abc" }), undefined);
    assert.equal(searchConfig({ PAPERLESS_MCP_WEBHOOK_HOST: "0.0.0.0" }), undefined);
    assert.equal(searchConfig({ PAPERLESS_MCP_WEBHOOK_DEBOUNCE_MS: "500" }), undefined);
  });

  test("with no port configured the resolved value is zero, so nothing binds", () => {
    // Any OTHER search variable enables the feature; the receiver must still be
    // off, because the port is the only thing that opens a socket.
    const config = searchConfig({ PAPERLESS_MCP_INDEX_DIR: "/tmp/idx" });
    assert.ok(config);
    assert.equal(config.webhookPort, 0);
    assert.equal(webhookRefusal(config), undefined);
  });

  test("a port with no secret is REFUSED, and the reason names both variables", () => {
    const config = searchConfig({ PAPERLESS_MCP_WEBHOOK_PORT: "8899" });
    assert.ok(config);
    assert.equal(config.webhookPort, 8899);
    const refusal = webhookRefusal(config);
    assert.ok(refusal, "a port without a secret must be refused");
    assert.match(refusal, /PAPERLESS_MCP_WEBHOOK_SECRET/);
    assert.match(refusal, /PAPERLESS_MCP_WEBHOOK_PORT/);
  });

  test("a port with a secret is accepted, and the default bind is loopback", () => {
    const config = searchConfig({
      PAPERLESS_MCP_WEBHOOK_PORT: "8899",
      PAPERLESS_MCP_WEBHOOK_SECRET: "s3cret",
    });
    assert.ok(config);
    assert.equal(webhookRefusal(config), undefined);
    // Loopback by default: widening the bind has to be a decision, because an
    // endpoint that writes to a local index should not be reachable off-host
    // unless the operator said so.
    assert.equal(config.webhookHost, "127.0.0.1");
  });

  test("the bind address and the debounce window are honoured when set", () => {
    const config = searchConfig({
      PAPERLESS_MCP_WEBHOOK_PORT: "8899",
      PAPERLESS_MCP_WEBHOOK_SECRET: "s3cret",
      PAPERLESS_MCP_WEBHOOK_HOST: "0.0.0.0",
      PAPERLESS_MCP_WEBHOOK_DEBOUNCE_MS: "250",
    });
    assert.ok(config);
    assert.equal(config.webhookHost, "0.0.0.0");
    assert.equal(config.webhookDebounceMs, 250);
  });

  test("a zero debounce is legal, so immediate-on-arrival is reachable", () => {
    const config = searchConfig({
      PAPERLESS_MCP_WEBHOOK_PORT: "8899",
      PAPERLESS_MCP_WEBHOOK_SECRET: "s3cret",
      PAPERLESS_MCP_WEBHOOK_DEBOUNCE_MS: "0",
    });
    assert.ok(config);
    assert.equal(config.webhookDebounceMs, 0);
  });

  test("an out-of-range port is rejected rather than silently coerced", () => {
    assert.throws(() =>
      searchConfig({
        PAPERLESS_MCP_WEBHOOK_PORT: "70000",
        PAPERLESS_MCP_WEBHOOK_SECRET: "s3cret",
      })
    );
  });

  test("an out-of-range debounce is rejected", () => {
    assert.throws(() =>
      searchConfig({
        PAPERLESS_MCP_WEBHOOK_PORT: "8899",
        PAPERLESS_MCP_WEBHOOK_SECRET: "s3cret",
        PAPERLESS_MCP_WEBHOOK_DEBOUNCE_MS: "-1",
      })
    );
  });
});
