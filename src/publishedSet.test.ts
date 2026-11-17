import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync, readdirSync, statSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, relative, sep } from "node:path";
import { test } from "node:test";

// Two defects reached consumers through the published tarball, and both were
// invisible in the source tree because the tree is not what ships.
//
//  1. A scratch harness file was committed at the repo root carrying an
//     absolute readFileSync path to a secrets-bearing env file under a
//     developer home directory (the paperless index build settings). It was
//     inside the shipped .dxt.
//  2. the build output shipped 58 compiled test files (29 .test.js plus 29
//     .test.d.ts) because files was set to include build. tsconfig.build.json
//     now excludes them.
//
// Both are properties of the PUBLISHED SET, so the assertions run against the
// published set. The listing is derived by compiling with tsconfig.build.json
// into a temp outDir rather than by running npm pack, because pack triggers
// prepack: rm -rf build, a full rebuild, sync-manifest (which MUTATES the
// tracked manifest.json) and dxt-pack. A test that dirties the worktree of the
// thing it is testing is not a test.
//
// A MINIMUM FILE COUNT is asserted so that a compile which silently emits
// nothing, or a temp dir that is not where we think, FAILS instead of
// vacuously satisfying every no-bad-file clause. A test that passes because it
// could not measure is worse than no test.

const root = join(__dirname, "..");

interface ShippedFile {
  path: string;
  text: string;
}

function walk(dir: string, base: string, out: string[]): void {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) walk(full, base, out);
    else out.push(relative(base, full).split(sep).join("/"));
  }
}

/**
 * The files a consumer receives, with their text.
 *
 * Compiles with tsconfig.build.json into a scratch outDir, reads the emitted
 * tree, then removes it. The temp dir is deleted in a finally so a failed
 * assertion cannot leave 500 MB behind.
 */
function publishedFiles(): ShippedFile[] {
  const out = mkdtempSync(join(tmpdir(), "paperless-published-"));
  try {
    execFileSync(
      process.execPath,
      [
        join(root, "node_modules", ".bin", "tsc"),
        "-p",
        join(root, "tsconfig.build.json"),
        // BOTH flags are required. tsconfig.json sets declarationDir to ./build,
        // which overrides outDir for .d.ts emission: passing only --outDir moved
        // the 32 .js files into the temp dir and wrote all 32 .d.ts files into
        // the real ./build of the worktree. The instrument then measured HALF
        // the published set and every .test.d.ts assertion passed vacuously --
        // the exact failure mode the floor assertion caught.
        "--outDir",
        out,
        "--declarationDir",
        out,
      ],
      { cwd: root, stdio: "pipe" }
    );

    const paths: string[] = [];
    walk(out, out, paths);

    // The floor. 32 source files compile to 32 js plus 32 d.ts, so anything
    // near zero means the compile did not happen where this function looked.
    assert.ok(
      paths.length >= 40,
      "compile emitted only " +
        paths.length +
        " files -- the published set was not measured, so every assertion below would pass vacuously"
    );

    return paths.map((p) => ({ path: p, text: readFileSync(join(out, p), "utf8") }));
  } finally {
    rmSync(out, { recursive: true, force: true });
  }
}

// DEFECT PINNED: 29 .test.js and 29 .test.d.ts shipped inside build for every
// consumer to download, because files included build and nothing excluded the
// tests sitting beside the code they cover. The exclusion now lives in
// tsconfig.build.json.
test("the published set contains no compiled test artifacts", () => {
  const shipped = publishedFiles();

  const offenders = shipped
    .map((f) => f.path)
    .filter((p) => /\.test\.(js|d\.ts)$/.test(p) || /\.spec\.(js|d\.ts)$/.test(p))
    .sort();
  assert.deepEqual(
    offenders,
    [],
    "the published set contains " +
      offenders.length +
      " compiled test file(s):\n  " +
      offenders.join("\n  ") +
      "\nConsumers download these and they serve no purpose in the package.\n" +
      "Fix: keep the exclusion in tsconfig.build.json."
  );

  // Test scaffolding travels with the tests. A mock or fixture directory that
  // ships is the same defect wearing a different extension -- this is the
  // residual the __mocks__ glob missed, because the real directory is
  // src/test/mocks and the glob named src/**/__mocks__/**.
  const scaffolding = shipped
    .map((f) => f.path)
    .filter((p) => /(^|\/)(test|tests|mocks|__mocks__|__tests__|fixtures)\//.test(p))
    .sort();
  assert.deepEqual(
    scaffolding,
    [],
    "the published set contains " +
      scaffolding.length +
      " test-scaffolding file(s):\n  " +
      scaffolding.join("\n  ") +
      "\nExcluding one directory name is not excluding the class."
  );
});

// DEFECT PINNED: a scratch harness committed at the repo root contained
// readFileSync with an absolute path into a developer home directory, naming a
// file that holds index build settings on that machine, and it travelled inside
// the shipped .dxt. Nothing in the source tree looked wrong; an absolute path is
// only a defect in a file that ships.
//
// The scan covers TWO surfaces, because the tarball is not the only thing users
// receive: the DXT bundles from the tracked tree, so a path that is tracked is
// publishable even when it does not compile into build.
const SLASH = "/";
const HOME_PREFIX = SLASH + "home" + SLASH;
const FORBIDDEN: { pattern: RegExp; label: string }[] = [
  { pattern: new RegExp(HOME_PREFIX + "guan" + SLASH), label: "a developer home directory" },
  { pattern: new RegExp(HOME_PREFIX + "userland" + SLASH), label: "a developer home directory" },
  {
    pattern: new RegExp("\\b192\\." + "168\\." + "8\\.\\d+"),
    label: "a private LAN address",
  },
  {
    pattern: new RegExp(HOME_PREFIX + "[a-z][a-z0-9_-]*" + SLASH + "\\.(paperless|mcp)"),
    label: "an absolute path to a dotfile under a home directory",
  },
];

function scan(label: string, entries: { path: string; text: string }[]): void {
  const hits: string[] = [];
  for (const entry of entries) {
    for (const rule of FORBIDDEN) {
      if (rule.pattern.test(entry.text)) {
        hits.push(entry.path + "  ->  " + rule.label);
      }
    }
  }
  assert.deepEqual(
    hits,
    [],
    label +
      " contains " +
      hits.length +
      " absolute home or LAN path(s):\n  " +
      hits.join("\n  ") +
      "\nA path to a developer machine leaks a username, a layout, or both, and\n" +
      "points a consumer at a file that does not exist for them."
  );
}

test("the published build output contains no absolute home or LAN paths", () => {
  const shipped = publishedFiles();
  scan("the published build output", shipped);
});

test("the tracked tree contains no absolute home or LAN paths", () => {
  // git ls-files is the committed set, which is what the DXT bundles and what a
  // reviewer reads. Untracked scratch files are deliberately out of scope:
  // eval/queries.json and eval/live.mjs are private and never committed, while
  // eval/results/*.log IS tracked and therefore IS scanned.
  const paths = execFileSync("git", ["ls-files"], { cwd: root, encoding: "utf8" })
    .split("\n")
    .map((p) => p.trim())
    .filter((p) => p.length > 0);

  assert.ok(
    paths.length >= 80,
    "git ls-files returned only " +
      paths.length +
      " files -- the tracked set was not measured, so this scan would pass vacuously"
  );

  const entries: { path: string; text: string }[] = [];
  for (const p of paths) {
    // Binary files are skipped explicitly rather than silently: reading a PNG as
    // utf8 yields replacement characters and cannot produce a false hit, but it
    // also cannot produce a true one, so the exclusion is stated here.
    if (/\.(png|jpg|jpeg|gif|webp|ico|pdf|zip|gz|tgz|woff2?|ttf|otf)$/i.test(p)) continue;
    try {
      const size = statSync(join(root, p)).size;
      if (size > 4_000_000) continue;
      entries.push({ path: p, text: readFileSync(join(root, p), "utf8") });
    } catch {
      // A tracked path that is not readable in the working tree still counts as
      // publishable. Fail rather than skip: silence here would hide a file that
      // cannot be scanned but will be shipped.
      assert.fail("tracked file could not be read for scanning: " + p);
    }
  }

  assert.ok(entries.length >= 60, "only " + entries.length + " tracked files were actually read");
  scan("the tracked tree", entries);
});
