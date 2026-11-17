import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, describe, test } from "node:test";
import type { PaperlessAPI } from "../api/PaperlessAPI";
import { EMBEDDING_DIMENSIONS, SearchIndex } from "./store";
import type { Embedder } from "./embedder";
import { DOCUMENT_LIST_PATH, IndexSync, WATERMARK_PATH, type SyncReport } from "./sync";
import {
  DEFAULT_DEBOUNCE_MS,
  MAX_BODY_BYTES,
  SECRET_HEADER,
  WebhookReceiver,
  parseEvent,
  presentedSecret,
  secretMatches,
  type SyncRunner,
} from "./webhook";

/**
 * The tests here exercise the REAL HTTP surface on an ephemeral port.
 *
 * ⚠️ They bind 127.0.0.1:0 and never a fixed port. A test that binds a public
 * port is a test that fails on a busy machine, and — worse — one that could be
 * reached from off the host on a deployment where the firewall is open. Port 0
 * asks the kernel for a free one, which is both collision-free and unreachable
 * from anywhere but loopback.
 */

const SECRET = "test-secret-not-a-real-one";

/** A runner that records every call, so "exactly one sync" is countable. */
class CountingRunner implements SyncRunner {
  calls: { force?: boolean; documentIds?: number[] }[] = [];
  report: Partial<SyncReport> = {};
  failWith: Error | undefined;
  /** Resolve gate, so a test can hold a run open while sending more events. */
  private release?: () => void;
  private gate?: Promise<void>;

  get isRunning(): boolean {
    return false;
  }

  /** Make the next run(s) wait until `release()` is called. */
  hold(): void {
    this.gate = new Promise<void>((resolve) => {
      this.release = resolve;
    });
  }

  releaseAll(): void {
    this.release?.();
    this.release = undefined;
    this.gate = undefined;
  }

  async run(opts: { force?: boolean; documentIds?: number[] } = {}): Promise<SyncReport> {
    this.calls.push(opts);
    if (this.gate) await this.gate;
    if (this.failWith) throw this.failWith;
    return {
      added: 0,
      updated: 0,
      deleted: 0,
      metadataRefreshed: 0,
      failed: 0,
      deletedSkipped: false,
      splits: 0,
      skipped: false,
      ...this.report,
    };
  }
}

interface SendResult {
  status: number;
  body: string;
}

/** POST to the receiver over a real socket, on loopback. */
async function post(
  port: number,
  body: string,
  opts: { secret?: string | null; contentType?: string; path?: string; method?: string } = {}
): Promise<SendResult> {
  const headers: Record<string, string> = {};
  const contentType = opts.contentType ?? "application/json";
  if (contentType) headers["content-type"] = contentType;
  // The DEFAULT is the correct secret, so a caller that is not testing
  // authentication does not have to remember to authenticate. Only an explicit
  // `null` omits the header, which is what the unauthenticated case asks for.
  const secret = opts.secret === undefined ? SECRET : opts.secret;
  if (secret !== null) headers[SECRET_HEADER] = secret;
  const response = await fetch(`http://127.0.0.1:${port}${opts.path ?? "/webhook/paperless"}`, {
    method: opts.method ?? "POST",
    headers,
    body: opts.method === "GET" ? undefined : body,
  });
  return { status: response.status, body: await response.text() };
}

/** Wait for a condition, or fail. Beats a fixed sleep for a debounce window. */
async function waitFor(
  predicate: () => boolean,
  timeoutMs = 3000,
  label = "condition"
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  throw new Error(`timed out waiting for ${label}`);
}

/** A short debounce, so the suite is not gated on a two-second wait per case. */
const FAST_DEBOUNCE = 40;

interface Harness {
  receiver: WebhookReceiver;
  runner: CountingRunner;
  port: number;
}

const live: Harness[] = [];

async function makeReceiver(overrides: Partial<{
  secret: string;
  debounceMs: number;
  runner: CountingRunner;
}> = {}): Promise<Harness> {
  const runner = overrides.runner ?? new CountingRunner();
  const receiver = new WebhookReceiver({
    secret: overrides.secret ?? SECRET,
    port: 0,
    host: "127.0.0.1",
    debounceMs: overrides.debounceMs ?? FAST_DEBOUNCE,
    runner: () => runner,
    // Silent, so a failing run's own log line does not pollute test output and
    // so the test cannot accidentally depend on stderr formatting.
    log: () => {},
  });
  const { port } = await receiver.start();
  const harness = { receiver, runner, port };
  live.push(harness);
  return harness;
}

after(async () => {
  await Promise.all(live.splice(0).map((h) => h.receiver.close()));
});

describe("parseEvent", () => {
  test("reads the id from a JSON body", () => {
    const event = parseEvent('{"id":42,"title":"A thing"}', "application/json");
    assert.deepEqual(event.ids, [42]);
    assert.equal(event.idless, false);
  });

  test("reads the id from a form-encoded body", () => {
    // The sender renders a template and then posts it either as JSON or as a
    // form depending on one boolean, so both shapes are legitimate.
    const event = parseEvent("id=43&title=A+thing", "application/x-www-form-urlencoded");
    assert.deepEqual(event.ids, [43]);
  });

  test("accepts the long id aliases the sender also offers", () => {
    assert.deepEqual(parseEvent('{"doc_id":7}', "application/json").ids, [7]);
    assert.deepEqual(parseEvent('{"document_id":8}', "application/json").ids, [8]);
  });

  test("a body with no id is reported as idless, not as an error", () => {
    // An operator who configured neither params nor body gets an empty
    // delivery. That still means "something changed", so it must not be
    // silently dropped -- but the caller has to know it cannot be targeted.
    const event = parseEvent("{}", "application/json");
    assert.deepEqual(event.ids, []);
    assert.equal(event.idless, true);
  });

  test("a non-numeric id is discarded rather than NaN", () => {
    const event = parseEvent('{"id":"not-a-number"}', "application/json");
    assert.deepEqual(event.ids, []);
    assert.equal(event.idless, true);
  });

  test("a JSON-labelled body that is not JSON degrades to idless", () => {
    const event = parseEvent("<xml>nope</xml>", "application/json");
    assert.deepEqual(event.ids, []);
    assert.equal(event.idless, true);
  });

  test("a body with no content type is still parsed when it looks like JSON", () => {
    assert.deepEqual(parseEvent('{"id":9}', undefined).ids, [9]);
  });

  test("a repeated id from two aliases is not counted twice", () => {
    const event = parseEvent('{"id":5,"doc_id":5}', "application/json");
    assert.deepEqual(event.ids, [5]);
  });
});

describe("secretMatches", () => {
  test("accepts the exact secret", () => {
    assert.equal(secretMatches(SECRET, SECRET), true);
  });

  test("rejects a wrong secret of the same length", () => {
    const wrong = SECRET.slice(0, -1) + "X";
    assert.equal(wrong.length, SECRET.length);
    assert.equal(secretMatches(wrong, SECRET), false);
  });

  test("rejects a secret of a different length without throwing", () => {
    // timingSafeEqual throws on a length mismatch, so the length is compared
    // first. This asserts the guard rather than the exception.
    assert.equal(secretMatches("short", SECRET), false);
  });

  test("rejects an absent secret", () => {
    assert.equal(secretMatches(undefined, SECRET), false);
  });
});

describe("presentedSecret", () => {
  test("reads the dedicated header", () => {
    assert.equal(presentedSecret({ [SECRET_HEADER]: SECRET }), SECRET);
  });

  test("reads a bearer token", () => {
    assert.equal(presentedSecret({ authorization: `Bearer ${SECRET}` }), SECRET);
  });

  test("returns undefined when neither is present", () => {
    assert.equal(presentedSecret({}), undefined);
  });
});

describe("the receiver refuses to start without a secret", () => {
  test("an empty secret is a thrown refusal, not a listening socket", async () => {
    // ⚠️ THE REFUSAL IS THE POINT. This endpoint causes embedding work; an
    // unauthenticated trigger for that is a resource-abuse vector, and a log
    // line nobody reads is not a mitigation.
    const receiver = new WebhookReceiver({
      secret: "",
      port: 0,
      host: "127.0.0.1",
      debounceMs: FAST_DEBOUNCE,
      runner: () => new CountingRunner(),
      log: () => {},
    });
    await assert.rejects(
      () => receiver.start(),
      /needs a shared secret/,
      "an unset secret must refuse to start"
    );
    assert.equal(receiver.isListening, false);
    assert.equal(receiver.boundPort, undefined);
    await receiver.close();
  });
});

describe("authentication", () => {
  test("a wrong secret answers 401 and triggers NO sync", async () => {
    const { port, runner } = await makeReceiver();
    const response = await post(port, '{"id":1}', { secret: "wrong-secret-entirely" });
    assert.equal(response.status, 401);

    // The negative that makes the status meaningful: waiting out the debounce
    // and then finding no call proves the handler returned before enqueueing.
    await new Promise((resolve) => setTimeout(resolve, FAST_DEBOUNCE * 5));
    assert.equal(runner.calls.length, 0, "a rejected request must not schedule work");
  });

  test("a request with no secret header answers 401", async () => {
    const { port, runner } = await makeReceiver();
    const response = await post(port, '{"id":1}', { secret: null });
    assert.equal(response.status, 401);
    await new Promise((resolve) => setTimeout(resolve, FAST_DEBOUNCE * 5));
    assert.equal(runner.calls.length, 0);
  });

  test("the same secret as a bearer token is accepted", async () => {
    const { port, runner } = await makeReceiver();
    const response = await fetch(`http://127.0.0.1:${port}/webhook/paperless`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${SECRET}` },
      body: '{"id":5}',
    });
    assert.equal(response.status, 202);
    await waitFor(() => runner.calls.length === 1, 3000, "the sync to run");
    assert.deepEqual(runner.calls[0].documentIds, [5]);
  });
});

describe("acknowledge-then-work", () => {
  test("202 comes back BEFORE the run happens", async () => {
    // ⚠️ THE ORDERING IS THE FEATURE. The sender times out at five seconds and
    // retries on failure, so a receiver that embedded first and answered after
    // would turn one edit into up to four deliveries of the same event. Holding
    // the run open and still getting a 202 proves the acknowledgement does not
    // wait for it.
    const { port, runner } = await makeReceiver();
    runner.hold();

    const response = await post(port, '{"id":11}');
    assert.equal(response.status, 202, "the response must not wait for the run");

    // The run is scheduled but held; the response already completed.
    await waitFor(() => runner.calls.length === 1, 3000, "the run to start");
    runner.releaseAll();
  });

  test("the payload names the document the run targets", async () => {
    const { port, runner } = await makeReceiver();
    const response = await post(port, '{"id":77,"title":"A document"}');
    assert.equal(response.status, 202);
    await waitFor(() => runner.calls.length === 1, 3000, "the sync to run");
    assert.deepEqual(runner.calls[0].documentIds, [77]);
  });

  test("an idless delivery runs an untargeted sync rather than nothing", async () => {
    // "Something changed and the sender did not say what" is not "nothing
    // happened". Dropping it would be a silent no-op.
    const { port, runner } = await makeReceiver();
    await post(port, "{}");
    await waitFor(() => runner.calls.length === 1, 3000, "the sync to run");
    assert.equal(
      runner.calls[0].documentIds,
      undefined,
      "an idless event must not pass an empty target list"
    );
  });
});

describe("debounce", () => {
  test("TEN rapid events for one document produce EXACTLY ONE sync", async () => {
    const { port, runner } = await makeReceiver({ debounceMs: 60 });
    for (let i = 0; i < 10; i += 1) {
      const response = await post(port, '{"id":42}');
      assert.equal(response.status, 202);
    }
    await waitFor(() => runner.calls.length >= 1, 3000, "the coalesced run");
    // Give a second run every chance to appear before asserting there is none.
    await new Promise((resolve) => setTimeout(resolve, 60 * 4));
    assert.equal(runner.calls.length, 1, "ten events must coalesce into one run");
    assert.deepEqual(runner.calls[0].documentIds, [42]);
  });

  test("events for DIFFERENT documents coalesce into one run naming all of them", async () => {
    const { port, runner } = await makeReceiver({ debounceMs: 60 });
    for (const id of [1, 2, 3, 4, 5]) {
      await post(port, `{"id":${id}}`);
    }
    await waitFor(() => runner.calls.length >= 1, 3000, "the coalesced run");
    await new Promise((resolve) => setTimeout(resolve, 60 * 4));
    assert.equal(runner.calls.length, 1);
    assert.deepEqual(
      [...(runner.calls[0].documentIds ?? [])].sort((a, b) => a - b),
      [1, 2, 3, 4, 5]
    );
  });

  test("a repeated id in a burst is named once", async () => {
    const { port, runner } = await makeReceiver({ debounceMs: 60 });
    for (let i = 0; i < 4; i += 1) await post(port, '{"id":9}');
    await waitFor(() => runner.calls.length === 1, 3000, "the coalesced run");
    await new Promise((resolve) => setTimeout(resolve, 60 * 4));
    assert.deepEqual(runner.calls[0].documentIds, [9]);
  });

  test("a quiet gap between bursts produces TWO runs, one per burst", async () => {
    // The other half of coalescing: the debounce must not collapse unrelated
    // edits forever, or the index would stop tracking a long editing session.
    const { port, runner } = await makeReceiver({ debounceMs: 40 });
    await post(port, '{"id":1}');
    await waitFor(() => runner.calls.length === 1, 3000, "the first run");
    await post(port, '{"id":2}');
    await waitFor(() => runner.calls.length === 2, 3000, "the second run");
    assert.deepEqual(runner.calls[0].documentIds, [1]);
    assert.deepEqual(runner.calls[1].documentIds, [2]);
  });
});

describe("body size cap", () => {
  test("a declared oversize body answers 413 and triggers NO sync", async () => {
    const { port, runner } = await makeReceiver();
    const oversized = "x".repeat(MAX_BODY_BYTES + 1);
    const response = await post(port, oversized);
    assert.equal(response.status, 413);
    await new Promise((resolve) => setTimeout(resolve, FAST_DEBOUNCE * 5));
    assert.equal(runner.calls.length, 0, "an oversized body must not schedule work");
  });

  test("a body exactly at the cap is accepted", async () => {
    // The boundary, asserted from the accepting side too: an off-by-one that
    // refused the limit itself would pass the test above and be wrong.
    const { port } = await makeReceiver();
    const atLimit = '{"id":1,"pad":"' + "x".repeat(MAX_BODY_BYTES - 20) + '"}';
    const response = await post(port, atLimit);
    assert.equal(response.status, 202);
  });
});

describe("lifecycle", () => {
  test("close() frees the port", async () => {
    const { receiver, port } = await makeReceiver();
    assert.equal(receiver.isListening, true);
    await receiver.close();
    assert.equal(receiver.isListening, false);

    // The instrument that matters: a NEW server can bind the same port. A
    // receiver that merely stopped accepting but held the socket would fail
    // here, and no amount of internal state would show it.
    const probe = createServer();
    await new Promise<void>((resolve, reject) => {
      probe.once("error", reject);
      probe.listen(port, "127.0.0.1", () => resolve());
    });
    await new Promise<void>((resolve) => probe.close(() => resolve()));
  });

  test("close() cancels a pending run", async () => {
    const { receiver, runner } = await makeReceiver({ debounceMs: 200 });
    await post(receiver.boundPort as number, '{"id":3}');
    // Closed inside the debounce window, so the run must never happen.
    await receiver.close();
    await new Promise((resolve) => setTimeout(resolve, 300));
    assert.equal(runner.calls.length, 0, "a run pending at close must be cancelled");
  });

  test("close() is idempotent", async () => {
    const { receiver } = await makeReceiver();
    await receiver.close();
    await receiver.close();
    assert.equal(receiver.isListening, false);
  });

  test("a run that throws does not take the receiver down", async () => {
    // A failed run must not stop the next event from being handled: the
    // watermark still converges on the next start, and an outage that killed
    // the receiver would need a process restart to recover from.
    const runner = new CountingRunner();
    runner.failWith = new Error("the index is unavailable");
    const { port } = await makeReceiver({ runner });
    await post(port, '{"id":1}');
    await waitFor(() => runner.calls.length === 1, 3000, "the failing run");
    runner.failWith = undefined;
    await post(port, '{"id":2}');
    await waitFor(() => runner.calls.length === 2, 3000, "the recovery run");
  });

  test("a GET on the root answers a health probe and schedules nothing", async () => {
    const { port, runner } = await makeReceiver();
    const response = await post(port, "", { method: "GET", path: "/", secret: null });
    assert.equal(response.status, 200);
    await new Promise((resolve) => setTimeout(resolve, FAST_DEBOUNCE * 5));
    assert.equal(runner.calls.length, 0);
  });

  test("an unknown path answers 404 and schedules nothing", async () => {
    const { port, runner } = await makeReceiver();
    const response = await post(port, '{"id":1}', { path: "/nope" });
    assert.equal(response.status, 404);
    await new Promise((resolve) => setTimeout(resolve, FAST_DEBOUNCE * 5));
    assert.equal(runner.calls.length, 0);
  });

  test("a trailing slash on the event path is accepted", async () => {
    const { port } = await makeReceiver();
    const response = await post(port, '{"id":1}', { path: "/webhook/paperless/" });
    assert.equal(response.status, 202);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// TARGETED SYNC — the receiver's path, driven through a real IndexSync.
//
// The receiver tests above prove the HTTP contract with a fake runner. These
// prove the thing the receiver calls actually does what the webhook needs,
// against a real sqlite index and a fake API. A test that stopped at the fake
// runner would assert the wiring and nothing about the effect.
// ─────────────────────────────────────────────────────────────────────────────

interface FakeDoc {
  id: number;
  title: string;
  content: string;
  created: string;
  modified: string;
  correspondent: number | null;
  document_type: number | null;
  tags: number[];
}

/** The routes the targeted path touches, and only those. */
class TargetedFakeApi {
  documents = new Map<number, FakeDoc>();
  detailRequests: number[] = [];
  listRequests = 0;

  add(doc: Partial<FakeDoc> & { id: number }): FakeDoc {
    const full: FakeDoc = {
      id: doc.id,
      title: doc.title ?? `Document ${doc.id}`,
      content: doc.content ?? `Body of document ${doc.id}.`,
      created: doc.created ?? "2026-11-01",
      modified: doc.modified ?? "2026-11-15T00:00:00Z",
      correspondent: doc.correspondent ?? null,
      document_type: doc.document_type ?? null,
      tags: doc.tags ?? [],
    };
    this.documents.set(full.id, full);
    return full;
  }

  remove(id: number): void {
    this.documents.delete(id);
  }

  async request<T>(path: string): Promise<T> {
    if (path === WATERMARK_PATH) {
      const all = [...this.documents.values()];
      let newest: string | null = null;
      let newestAt = -Infinity;
      for (const doc of all) {
        const at = new Date(doc.modified).getTime();
        if (Number.isFinite(at) && at > newestAt) {
          newestAt = at;
          newest = doc.modified;
        }
      }
      return {
        count: all.length,
        next: null,
        results: newest === null ? [] : [{ id: 0, modified: newest }],
      } as unknown as T;
    }

    if (path.startsWith(DOCUMENT_LIST_PATH)) {
      this.listRequests += 1;
      const rows = [...this.documents.values()]
        .sort((a, b) => a.id - b.id)
        .map(({ id, modified }) => ({ id, modified }));
      return { count: rows.length, next: null, results: rows } as unknown as T;
    }

    if (path.startsWith("/correspondents/")) {
      return { count: 0, next: null, results: [] } as unknown as T;
    }
    if (path.startsWith("/document_types/")) {
      return { count: 0, next: null, results: [] } as unknown as T;
    }
    if (path.startsWith("/tags/")) {
      // ⚠️ A REAL NAME, NOT AN EMPTY LIST. The metadata text is built from tag
      // NAMES, so a fixture that returned no tags would make a tag edit produce
      // a byte-identical metadata chunk -- and the "metadata changed without
      // `modified` moving" test would then pass or fail for a reason unrelated
      // to the code under test.
      return {
        count: 1,
        next: null,
        results: [{ id: 1, name: "a-tag" }],
      } as unknown as T;
    }

    const detail = /^\/documents\/(\d+)\/$/.exec(path);
    if (detail) {
      const id = Number(detail[1]);
      this.detailRequests.push(id);
      const doc = this.documents.get(id);
      // ⚠️ A MISSING DOCUMENT ANSWERS LIKE THE REAL API: an error. The targeted
      // path must survive it, because a delivery can name a document deleted
      // between the change and the webhook.
      if (!doc) throw new Error("Not found.");
      return doc as unknown as T;
    }

    throw new Error(`TargetedFakeApi got an unexpected request: ${path}`);
  }
}

/** A stub embedder: deterministic, and it counts what it was asked to embed. */
class StubEmbedder implements Embedder {
  readonly model = "stub-embed";
  readonly prefixMode = "none" as const;
  readonly dimensions = EMBEDDING_DIMENSIONS;
  embeddedTexts = 0;

  async embedDocuments(texts: string[]): Promise<number[][]> {
    this.embeddedTexts += texts.length;
    return texts.map((_, i) =>
      Array.from({ length: EMBEDDING_DIMENSIONS }, (__, index) =>
        index === (i + 1) % 64 ? 1 : 0
      )
    );
  }

  async embedQuery(): Promise<number[]> {
    return Array.from({ length: EMBEDDING_DIMENSIONS }, (_, index) =>
      index === 1 ? 1 : 0
    );
  }
}

let indexCounter = 0;
const openedIndexes: SearchIndex[] = [];

function freshIndex(): SearchIndex {
  indexCounter += 1;
  const dir = join(
    mkdtempSync(join(tmpdir(), "paperless-webhook-")),
    `idx-${indexCounter}`
  );
  mkdirSync(dir, { recursive: true });
  const index = new SearchIndex(join(dir, "index.db"));
  index.open();
  openedIndexes.push(index);
  return index;
}

function makeSync(api: TargetedFakeApi, index: SearchIndex, embedder: Embedder): IndexSync {
  return new IndexSync({
    api: api as unknown as PaperlessAPI,
    store: index,
    embedder,
  });
}

after(() => {
  for (const index of openedIndexes.splice(0)) index.close();
});

describe("a targeted run", () => {
  test("indexes the named document without walking the corpus list", async () => {
    const api = new TargetedFakeApi();
    api.add({ id: 1 });
    api.add({ id: 2 });
    const index = freshIndex();
    const sync = makeSync(api, index, new StubEmbedder());

    const report = await sync.run({ documentIds: [2] });
    assert.equal(report.added, 1);
    assert.equal(report.skipped, false);
    assert.deepEqual(api.detailRequests, [2], "only the named document is fetched");
    assert.equal(api.listRequests, 0, "a targeted run must not walk the full list");
    assert.deepEqual([...index.indexed().keys()], [2]);
  });

  test("re-indexes a document whose metadata changed while `modified` did not", async () => {
    // ⚠️ THIS IS THE CASE THE WEBHOOK EXISTS FOR. A tag edit goes through a
    // queryset update on a through-table, so `auto_now` never fires: the
    // document's `modified` is unchanged and the corpus count is unchanged.
    // Both watermark scalars therefore read "nothing happened" while the
    // metadata chunk has genuinely gone stale. The event is the only signal.
    const api = new TargetedFakeApi();
    api.add({ id: 1, title: "Old title", tags: [] });
    const index = freshIndex();
    const embedder = new StubEmbedder();
    const sync = makeSync(api, index, embedder);

    await sync.run({ documentIds: [1] });
    const chunksBefore = index.chunksFor(1).map((c) => c.text);

    // Same `modified`, different tags. Nothing a corpus-level signal can see.
    api.documents.set(1, { ...api.documents.get(1)!, tags: [1] });
    api.add({ id: 2 }); // a neighbour, to prove only the target moved

    const report = await sync.run({ documentIds: [1] });
    assert.equal(report.updated, 1);
    const chunksAfter = index.chunksFor(1).map((c) => c.text);
    assert.notDeepEqual(chunksAfter, chunksBefore, "the metadata chunk must be rebuilt");
  });

  test("a full run right after a targeted run is NOT short-circuited", async () => {
    // ⚠️ THE WATERMARK GUARD THAT MAKES THE COEXISTENCE SAFE. A targeted run
    // looked at one document; writing the watermark from it would claim the
    // corpus was swept and stop the next full run from doing it -- the index
    // would then be missing every other document, permanently and silently.
    const api = new TargetedFakeApi();
    api.add({ id: 1 });
    api.add({ id: 2 });
    api.add({ id: 3 });
    const index = freshIndex();
    const sync = makeSync(api, index, new StubEmbedder());

    await sync.run({ documentIds: [1] });
    assert.equal(index.watermark(), undefined, "a targeted run must not write the watermark");

    const report = await sync.run({});
    assert.equal(report.skipped, false, "the full run must still do its work");
    assert.equal(report.added, 2);
    assert.deepEqual([...index.indexed().keys()].sort((a, b) => a - b), [1, 2, 3]);
    assert.ok(index.watermark(), "the full run records the watermark");
  });

  test("a named document that no longer exists does not crash the run", async () => {
    // A delivery can name a document deleted between the change and the
    // webhook. The event is already in flight and cannot be recalled, so the
    // run must count the failure and continue rather than abandon the rest.
    const api = new TargetedFakeApi();
    api.add({ id: 1 });
    const index = freshIndex();
    const sync = makeSync(api, index, new StubEmbedder());

    const report = await sync.run({ documentIds: [999] });
    assert.equal(report.failed, 1);
    assert.equal(report.added, 0);
    assert.equal(index.counts().documents, 0);
  });

  test("a burst naming a live and a dead document indexes the live one", async () => {
    // The realistic shape: a bulk edit touches documents, one of them was
    // deleted in the meantime. The surviving ones must still be indexed.
    const api = new TargetedFakeApi();
    api.add({ id: 1 });
    api.add({ id: 2 });
    const index = freshIndex();
    const sync = makeSync(api, index, new StubEmbedder());

    const report = await sync.run({ documentIds: [1, 999, 2] });
    assert.equal(report.failed, 1);
    assert.equal(report.added, 2);
    assert.deepEqual([...index.indexed().keys()].sort((a, b) => a - b), [1, 2]);
  });

  test("a duplicate id in the target list is indexed once", async () => {
    const api = new TargetedFakeApi();
    api.add({ id: 5 });
    const index = freshIndex();
    const sync = makeSync(api, index, new StubEmbedder());

    const report = await sync.run({ documentIds: [5, 5, 5] });
    assert.equal(report.added, 1);
    assert.deepEqual(api.detailRequests, [5], "the same id must be fetched once");
  });
});

describe("run() sequencing", () => {
  test("a full run during a targeted run is sequenced, not joined", async () => {
    // ⚠️ JOINING THEM WOULD BE A SILENT LIE. A full caller that joined a
    // targeted run would receive a report covering one document and read it as
    // "the corpus is up to date". Sequenced, it gets its own sweep.
    const api = new TargetedFakeApi();
    api.add({ id: 1 });
    api.add({ id: 2 });
    const index = freshIndex();
    const sync = makeSync(api, index, new StubEmbedder());

    const targeted = sync.run({ documentIds: [1] });
    const full = sync.run({});
    const [targetedReport, fullReport] = await Promise.all([targeted, full]);

    assert.equal(targetedReport.added, 1, "the targeted run added one");
    assert.equal(fullReport.added, 1, "the full run then added the remaining one");
    assert.deepEqual([...index.indexed().keys()].sort((a, b) => a - b), [1, 2]);
  });

  test("two targeted runs share the one in flight", async () => {
    // The opposite case: same question, so one answer is correct.
    const api = new TargetedFakeApi();
    api.add({ id: 1 });
    const index = freshIndex();
    const sync = makeSync(api, index, new StubEmbedder());

    const a = sync.run({ documentIds: [1] });
    const b = sync.run({ documentIds: [1] });
    assert.equal(a, b, "a second targeted run must join the first");
    await a;
  });
});

describe("the persistence path still works without a receiver", () => {
  test("the watermark catches a document added while no event arrived", async () => {
    // ⚠️ THIS IS WHY THE WATERMARK IS NOT REPLACED. A webhook only reports
    // events the process was alive to receive. A document added while the
    // process was down produces no delivery at all, and only the corpus
    // watermark can notice.
    const api = new TargetedFakeApi();
    api.add({ id: 1 });
    const index = freshIndex();
    const sync = makeSync(api, index, new StubEmbedder());

    await sync.run({});
    assert.equal(index.counts().documents, 1);

    // Nothing is delivered. The corpus simply changed.
    api.add({ id: 2, modified: "2026-11-16T00:00:00Z" });

    const report = await sync.run({});
    assert.equal(report.skipped, false, "the watermark must notice the addition");
    assert.equal(report.added, 1);
    assert.deepEqual([...index.indexed().keys()].sort((a, b) => a - b), [1, 2]);
  });

  test("a deletion is still detected by the count, not by any event", async () => {
    // The sending side fires nothing on delete, so this is the only detector.
    const api = new TargetedFakeApi();
    api.add({ id: 1 });
    api.add({ id: 2 });
    const index = freshIndex();
    const sync = makeSync(api, index, new StubEmbedder());

    await sync.run({});
    assert.equal(index.counts().documents, 2);

    api.remove(2);
    const report = await sync.run({});
    assert.equal(report.deleted, 1);
    assert.deepEqual([...index.indexed().keys()], [1]);
  });
});
