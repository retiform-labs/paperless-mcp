// Stage-7 wire probe: spawn the INSTALLED package and exercise it.
// Run from a clean-room install so the bytes are the ones a stranger receives.
//   node /REDACTED/src/paperless-mcp/eval/wire-probe.mjs
import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";

// ⚠️ createRequire(import.meta.url) resolves against the SCRIPT's location, not
// the cwd -- so this probe would have imported the dev tree and proved nothing
// about the installed artifact, the exact substitution stage 7 exists to prevent.
// Anchoring on cwd makes the resolution follow whoever is running it.
const require = createRequire(join(process.cwd(), "noop.js"));
const entry = require.resolve("@retiform/paperless-mcp/build/index.js");
const pkgPath = join(dirname(entry), "..", "package.json");
const { version } = JSON.parse(readFileSync(pkgPath, "utf8"));

const child = spawn(process.execPath, [entry], { stdio: ["pipe", "pipe", "pipe"] });
let buf = "";
let stderr = "";
const waiting = new Map();
child.stderr.on("data", (d) => {
  stderr += String(d);
});
child.stdout.on("data", (d) => {
  buf += String(d);
  for (const line of buf.split("\n")) {
    const t = line.trim();
    if (!t.startsWith("{")) continue;
    let m;
    try {
      m = JSON.parse(t);
    } catch {
      continue;
    }
    if (waiting.has(m.id)) {
      waiting.get(m.id)(m);
      waiting.delete(m.id);
    }
  }
});
const rpc = (id, method, params) =>
  new Promise((res, rej) => {
    const timer = setTimeout(() => rej(new Error(`timeout ${method}`)), 30000);
    waiting.set(id, (m) => {
      clearTimeout(timer);
      res(m);
    });
    child.stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
  });

try {
  await rpc(1, "initialize", {
    protocolVersion: "2024-11-05",
    capabilities: {},
    clientInfo: { name: "wire-probe", version: "1" },
  });
  child.stdin.write(
    JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }) + "\n",
  );

  const list = await rpc(2, "tools/list", {});
  const names = list.result.tools.map((t) => t.name);
  console.log(`INSTALLED_VERSION=${version}`);
  console.log(`TOOLS=${names.length}`);
  for (const want of ["semantic_search", "search_passages", "embedding_status", "sync_embeddings"]) {
    console.log(`  ${want}=${names.includes(want) ? "PRESENT" : "MISSING"}`);
  }

  const call = await rpc(3, "tools/call", {
    name: "semantic_search",
    arguments: { query: "ZQZTYSKFNGVDPR", limit: 3 },
  });
  const r = JSON.parse(call.result.content[0].text);
  console.log(`SEMANTIC=${r.semantic} TOTAL_HITS=${r.total_hits}`);
  console.log(`RETRIEVERS=${(r.retrievers ?? []).join(",")}`);
  const ids = r.passages.map((p) => p.id);
  console.log(`PASSAGE_IDS=${ids.join(",")}`);
  const hit = r.passages.find((p) => p.id === 2784);
  console.log(`EXACT_DOC_PRESENT=${hit ? "YES" : "NO"}`);
  if (hit) {
    console.log(`  rank=${ids.indexOf(2784) + 1} matched=${hit.matched} score=${hit.score.toFixed(4)}`);
    console.log(`  highlights=${JSON.stringify(hit.highlights ?? null)}`);
    console.log(`  passage_len=${hit.passage.length} content_length=${hit.content_length}`);
  }
  // Degradation path: a tool that must never invent results.
  const bad = await rpc(4, "tools/call", {
    name: "semantic_search",
    arguments: { query: "", limit: 3 },
  });
  console.log(`EMPTY_QUERY_ISERROR=${bad.result?.isError === true}`);
} catch (e) {
  console.log(`PROBE_FAILED=${e.message}`);
  if (stderr.trim()) console.log(stderr.trim().slice(0, 800));
  process.exitCode = 1;
} finally {
  child.stdin.end();
  child.kill();
}
