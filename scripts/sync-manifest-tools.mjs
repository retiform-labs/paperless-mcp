// The manifest carried a HAND-MAINTAINED `tools[]` and it drifted exactly the
// way the version field did before 2.3.6: 13 entries against 48 on the wire,
// with every semantic tool — `semantic_search`, `search_passages`,
// `embedding_status`, `sync_embeddings` — absent. A DXT host renders the
// manifest, so a capability missing here is a capability the marketplace does
// not know exists, no matter what the npm wire serves.
//
// The fix is the same principle that fixed the version: package.json (here,
// the running SERVER) is the single source, and this stamps it into the
// manifest before the bundle is packed. A list derived from the artifact
// cannot disagree with the artifact.
//
// Runs against build/index.js over stdio JSON-RPC, so it needs a build first —
// `prepack` already builds, and this is sequenced after the build in that chain.
import { spawn } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const entry = join(root, "build", "index.js");
const manifestPath = join(root, "manifest.json");

// The server validates nothing at startup beyond needing a URL to construct
// its client, so a placeholder reaches tools/list without touching a real
// instance. No request is ever issued: tools/list is answered from the
// registered handlers.
//
// ⚠️ THE SEARCH TOOLS ARE CONDITIONALLY REGISTERED. `registerSearchTools`
// returns early unless `context.isConfigured`, so a manifest derived from a
// bare environment lists 45 tools and silently omits `semantic_search`,
// `search_passages`, `embedding_status` and `sync_embeddings` -- which is the
// exact defect this script exists to fix, reintroduced by the instrument.
// Setting one search variable makes the gate true; the index itself is never
// opened, because tools/list does not touch it.
const env = {
  ...process.env,
  PAPERLESS_URL: process.env.PAPERLESS_URL ?? "http://127.0.0.1:1",
  PAPERLESS_API_KEY: process.env.PAPERLESS_API_KEY ?? "manifest-sync-placeholder",
  PAPERLESS_MCP_EMBEDDING_URL:
    process.env.PAPERLESS_MCP_EMBEDDING_URL ?? "http://127.0.0.1:1",
};

function rpc(child, id, method, params) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error(`timeout waiting for ${method}`)),
      20000,
    );
    const onData = (chunk) => {
      for (const line of String(chunk).split("\n")) {
        const text = line.trim();
        if (!text.startsWith("{")) continue;
        let msg;
        try {
          msg = JSON.parse(text);
        } catch {
          continue;
        }
        if (msg.id !== id) continue;
        clearTimeout(timer);
        child.stdout.off("data", onData);
        if (msg.error) reject(new Error(`${method}: ${msg.error.message}`));
        else resolve(msg.result);
      }
    };
    child.stdout.on("data", onData);
    child.stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
  });
}

const child = spawn(process.execPath, [entry], { env, stdio: ["pipe", "pipe", "pipe"] });
let stderr = "";
child.stderr.on("data", (c) => {
  stderr += String(c);
});

try {
  await rpc(child, 1, "initialize", {
    protocolVersion: "2024-11-05",
    capabilities: {},
    clientInfo: { name: "manifest-sync", version: "1.0.0" },
  });
  child.stdin.write(
    JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }) + "\n",
  );
  const { tools } = await rpc(child, 2, "tools/list", {});
  if (!Array.isArray(tools) || tools.length === 0) {
    throw new Error("tools/list returned no tools -- refusing to write an empty list");
  }

  // name + description only. The manifest schema documents exactly these two
  // for a tool entry, and copying inputSchema wholesale would ship a large
  // duplicate of what the MCP handshake already delivers.
  const next = tools
    .map((t) => ({ name: t.name, description: t.description ?? "" }))
    .sort((a, b) => a.name.localeCompare(b.name));

  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  const before = Array.isArray(manifest.tools) ? manifest.tools.length : 0;
  const beforeNames = new Set((manifest.tools ?? []).map((t) => t.name));
  const added = next.filter((t) => !beforeNames.has(t.name)).map((t) => t.name);
  const removed = (manifest.tools ?? [])
    .map((t) => t.name)
    .filter((n) => !next.some((t) => t.name === n));

  // Idempotent: rewrite only when the derived list differs, so a no-op run
  // leaves the file's bytes (and its mtime) alone.
  const differs =
    JSON.stringify(next) !== JSON.stringify(manifest.tools ?? []);
  if (!differs) {
    console.log(`manifest.json tools already current (${next.length})`);
  } else {
    manifest.tools = next;
    writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + "\n");
    console.log(`manifest.json tools ${before} -> ${next.length}`);
    if (added.length) console.log(`  added: ${added.join(", ")}`);
    if (removed.length) console.log(`  removed: ${removed.join(", ")}`);
  }
} catch (error) {
  console.error(
    `sync-manifest-tools: ${error instanceof Error ? error.message : String(error)}`,
  );
  if (stderr.trim()) console.error(stderr.trim().slice(0, 2000));
  process.exit(1);
} finally {
  child.stdin.end();
  child.kill();
}
