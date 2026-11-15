/* One-shot full corpus build for the semantic search index. */
import { searchConfig, indexPath } from "../src/search/config";
import { PaperlessAPI } from "../src/api/PaperlessAPI";
import { SearchIndex } from "../src/search/store";
import { OllamaEmbedder } from "../src/search/embedder";
import { IndexSync } from "../src/search/sync";

function log(msg: string) {
  console.log(`[${new Date().toISOString()}] ${msg}`);
}

async function main() {
  const config = searchConfig(process.env);
  if (!config) {
    console.error("No search config: set PAPERLESS_MCP_EMBEDDING_URL etc.");
    process.exit(2);
  }
  const baseUrl = process.env.PAPERLESS_URL;
  const token = process.env.PAPERLESS_API_KEY;
  if (!baseUrl || !token) {
    console.error("PAPERLESS_URL / PAPERLESS_API_KEY required");
    process.exit(2);
  }

  const path = indexPath(config);
  log(`index path: ${path}`);
  log(`embedder: ${config.embeddingUrl} model=${config.model} prefix=${config.prefixMode}`);
  log(`maxChunks: ${config.maxChunks}`);

  const api = new PaperlessAPI(baseUrl, token);
  const store = new SearchIndex(path);
  store.open();
  log(`store opened; fts5=${store.lexicalAvailable} before=${JSON.stringify(store.counts())}`);

  const embedder = new OllamaEmbedder({
    url: config.embeddingUrl,
    model: config.model,
    prefixMode: config.prefixMode,
  });

  // Probe BEFORE any destructive step, matching the fixed force:true ordering.
  log("probing embedder with a real embedding...");
  const t0 = Date.now();
  const probe = await embedder.embedQuery("ping");
  log(`probe ok: ${probe.length} dims in ${Date.now() - t0} ms`);

  const runner = new IndexSync({ api, store, embedder, maxChunks: config.maxChunks });

  log("sync starting");
  const started = Date.now();
  let tick = setInterval(() => {
    try {
      log(`  ...${Math.round((Date.now() - started) / 1000)}s ${JSON.stringify(store.counts())}`);
    } catch (e) { log(`  ...counts failed: ${(e as Error).message}`); }
  }, 30000);

  try {
    const report = await runner.run();
    clearInterval(tick);
    const secs = Math.round((Date.now() - started) / 1000);
    log(`DONE in ${secs}s`);
    log(`report: ${JSON.stringify(report)}`);
    log(`final counts: ${JSON.stringify(store.counts())}`);
    log(`meta: ${JSON.stringify(store.meta() ?? "n/a")}`);
    store.close();
    log("BUILD_OK");
  } catch (err) {
    clearInterval(tick);
    console.error(`BUILD_FAILED: ${(err as Error).message}`);
    console.error((err as Error).stack);
    try { store.close(); } catch {}
    process.exit(1);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
