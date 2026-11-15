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
