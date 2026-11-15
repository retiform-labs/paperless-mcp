/* One-shot full corpus build for the semantic search index. */
import { searchConfig, indexPath } from "../src/search/config";
import { PaperlessAPI } from "../src/api/PaperlessAPI";
import { SearchIndex } from "../src/search/store";
import { OllamaEmbedder } from "../src/search/embedder";
import { IndexSync } from "../src/search/sync";

function log(msg: string) {
  console.log(`[${new Date().toISOString()}] ${msg}`);
}

/**
 * The exit code a finished run must produce.
 *
 * ⚠️ EXTRACTED SO IT CAN BE FALSIFIED. This script runs `main()` on import, so
 * nothing in the suite can execute it -- which meant the decision "does a
 * partial build count as success?" had no test at all, and a deliberate
 * mutation that silenced it stayed green. A one-line pure function is the
 * smallest seam that makes the answer checkable.
 *
 * The distinction is not cosmetic. The run this replaces died at 1,727 of 2,466
 * documents; the index was usable but missing 739, and the wrapper reported
 * success. A caller that cannot tell "complete" from "partial" will keep
 * serving an index that silently returns nothing for a third of the corpus.
 */
export function exitCodeForReport(report: { failed: number }): number {
  return report.failed > 0 ? 3 : 0;
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

    // ⚠️ THREE OUTCOMES, THREE EXIT CODES, AND THEY ARE NOT INTERCHANGEABLE.
    //
    // The run this replaces ended its systemd wrapper with a bare
    // `echo "EXIT=$?"`, which always exits 0 -- so `systemctl is-active` said
    // inactive and the unit reported success while the process had in fact
    // exited 1. A wrapper that cannot fail is worse than no wrapper: it makes
    // the log the only instrument, and nobody reads the log of a run that
    // reported success.
    //
    // A partial build is its own outcome because the index is USABLE but
    // INCOMPLETE, and reporting that as success is how a missing 739 documents
    // goes unnoticed until a search silently returns nothing for them.
    const exitCode = exitCodeForReport(report);
    if (exitCode !== 0) {
      console.error(
        `BUILD_PARTIAL: ${report.failed} document(s) failed; the index is ` +
          `usable but incomplete. Counts: ${JSON.stringify(store.counts())}`
      );
      process.exit(exitCode);
    }
    log("BUILD_OK");
    process.exit(exitCode);
  } catch (err) {
    clearInterval(tick);
    console.error(`BUILD_FAILED: ${(err as Error).message}`);
    console.error((err as Error).stack);
    try { store.close(); } catch {}
    process.exit(1);
  }
}

// ⚠️ THE GUARD IS LOAD-BEARING. Without it, importing this module to test
// `exitCodeForReport` would start a corpus build as a side effect -- a test that
// launches a fifty-minute job is not a test, and the failure would look like a
// timeout rather than like a missing guard.
if (require.main === module) {
  main().catch((e) => { console.error(e); process.exit(1); });
}
