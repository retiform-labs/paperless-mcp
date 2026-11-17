import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, cpSync, mkdtempSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { test } from "node:test";

// The DXT manifest carries its own copy of the version and changesets only
// bumps package.json, so the two drifted on every bot-driven release: 2.3.3
// shipped a 2.3.2 manifest, 2.3.5 shipped a 2.3.4 one. scripts/sync-manifest-version.mjs
// derives it at pack time; this pins both halves of that guarantee.

const root = join(__dirname, "..");

function readManifest(): Record<string, unknown> {
  return JSON.parse(readFileSync(join(root, "manifest.json"), "utf8"));
}

// The tracked manifest version MUST equal package.json, and this is a real
// invariant rather than a style preference.
//
// An earlier revision of this file declined that assertion, reasoning that the
// changesets Version PR bumps package.json only, so the tracked pair is
// legitimately unequal between a version bump and the next pack. That reasoning
// was measured and is FALSE. prepack runs sync-manifest, sync-manifest rewrites
// manifest.json from package.json, and CI then requires git status --porcelain
// to be EMPTY after npm pack --dry-run. Measured directly: package.json 2.4.1
// with manifest.json 2.4.0 packs to 2.4.1 and leaves a modified manifest.json,
// which that gate fails on. A mismatched tracked pair cannot reach a publish --
// it stops the build first. Equality is the state CI demands, so asserting it
// here catches a version bump that forgot the manifest in the SUITE instead of
// in a release job.
//
// What is pinned below: the tracked pair is equal, AND the mechanism that keeps
// it equal still runs at pack time.

test("sync-manifest-version stamps package.json over a stale manifest version", () => {
  const dir = mkdtempSync(join(tmpdir(), "manifest-sync-"));
  try {
    cpSync(join(root, "package.json"), join(dir, "package.json"));
    const manifest = readManifest();
    manifest.version = "0.0.0-stale";
    writeFileSync(join(dir, "manifest.json"), JSON.stringify(manifest, null, 2));
    cpSync(join(root, "scripts"), join(dir, "scripts"), { recursive: true });
    cpSync(join(root, "src"), join(dir, "src"), { recursive: true });

    execFileSync(process.execPath, [join(dir, "scripts", "sync-manifest-version.mjs")], {
      stdio: "pipe",
    });

    const pkg = JSON.parse(readFileSync(join(dir, "package.json"), "utf8"));
    const after = JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8"));
    assert.equal(after.version, pkg.version, "sync did not stamp package.json version");
    assert.notEqual(after.version, "0.0.0-stale", "stale literal survived the sync");
    assert.equal(after.dxt_version, manifest.dxt_version, "sync clobbered dxt_version");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("sync-manifest-version is idempotent", () => {
  const dir = mkdtempSync(join(tmpdir(), "manifest-idem-"));
  try {
    cpSync(join(root, "package.json"), join(dir, "package.json"));
    cpSync(join(root, "manifest.json"), join(dir, "manifest.json"));
    cpSync(join(root, "scripts"), join(dir, "scripts"), { recursive: true });

    const run = () =>
      execFileSync(process.execPath, [join(dir, "scripts", "sync-manifest-version.mjs")], {
        stdio: "pipe",
      });

    run();
    const once = readFileSync(join(dir, "manifest.json"), "utf8");
    run();
    const twice = readFileSync(join(dir, "manifest.json"), "utf8");
    assert.equal(twice, once, "second run changed the manifest");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// DEFECT PINNED: 2.3.3 shipped a 2.3.2 manifest, 2.3.5 a 2.3.4, 2.3.6 a 2.3.5.
// changesets bumps package.json; nothing bumped the manifest, so every bot-driven
// release published a manifest one version behind the package it described.
test("the tracked manifest version equals package.json", () => {
  const pkg = JSON.parse(readFileSync(join(root, "package.json"), "utf8")) as { version?: string };
  const manifest = readManifest() as { version?: unknown };

  assert.equal(typeof pkg.version, "string", "package.json has no version");
  assert.equal(
    manifest.version,
    pkg.version,
    "manifest.json version " +
      String(manifest.version) +
      " != package.json version " +
      String(pkg.version) +
      "\nA published manifest would advertise the wrong version to a DXT host.\n" +
      "Fix: npm run sync-manifest, then commit manifest.json with the version bump."
  );
});

// The equality above is only durable if the mechanism that produces it still
// runs. Pinning the wiring means deleting the sync step from prepack, or
// dropping a script from sync-manifest, fails HERE rather than shipping a
// drifting manifest on the next release.
test("prepack still regenerates the manifest from package.json", () => {
  const pkg = JSON.parse(readFileSync(join(root, "package.json"), "utf8")) as {
    scripts?: Record<string, string>;
  };
  const scripts = pkg.scripts ?? {};

  assert.ok(
    typeof scripts.prepack === "string",
    "package.json has no prepack script -- nothing regenerates the manifest at publish time"
  );
  assert.match(
    scripts.prepack,
    /sync-manifest/,
    "prepack does not run sync-manifest, so a version bump will not reach manifest.json"
  );

  assert.ok(
    typeof scripts["sync-manifest"] === "string",
    "package.json has no sync-manifest script"
  );
  assert.match(
    scripts["sync-manifest"],
    /sync-manifest-version\.mjs/,
    "sync-manifest no longer runs sync-manifest-version.mjs"
  );
  // The tools[] half drifted too (13 entries against 48 registered tools).
  assert.match(
    scripts["sync-manifest"],
    /sync-manifest-tools\.mjs/,
    "sync-manifest no longer runs sync-manifest-tools.mjs, so tools[] can drift again"
  );
});

// DEFECT PINNED: `npm pack` could dirty the tracked tree. CI now runs
// `npm pack --dry-run` and then requires `git status --porcelain` to be EMPTY,
// so a publish can never silently rewrite a committed file. That gate is a CI
// invariant; what is testable here is the property the gate depends on -- the
// generation scripts must be IDEMPOTENT, and must not even touch the file when
// there is nothing to change.
//
// Byte-identity alone is too weak. A script that rewrites an identical byte
// sequence would pass a content assertion and still churn the mtime, which is
// what makes `git status` (and every downstream cache) see a change. So the
// assertions are: bytes identical, AND the inode is not touched on a no-op run.
test("sync-manifest-version does not rewrite the manifest when already in sync", () => {
  const dir = mkdtempSync(join(tmpdir(), "manifest-nochurn-"));
  try {
    cpSync(join(root, "package.json"), join(dir, "package.json"));
    cpSync(join(root, "manifest.json"), join(dir, "manifest.json"));
    cpSync(join(root, "scripts"), join(dir, "scripts"), { recursive: true });

    const manifestPath = join(dir, "manifest.json");
    const before = readFileSync(manifestPath, "utf8");
    const mtimeBefore = statSync(manifestPath).mtimeMs;

    const out = execFileSync(
      process.execPath,
      [join(dir, "scripts", "sync-manifest-version.mjs")],
      { stdio: "pipe", encoding: "utf8" }
    );

    const after = readFileSync(manifestPath, "utf8");
    assert.equal(after, before, "an in-sync run changed the manifest bytes");

    assert.match(
      out,
      /already/,
      "an in-sync run did not report that the version was already correct -- it may have rewritten the file"
    );

    // The no-churn property. An in-sync run must not open the file for writing.
    assert.equal(
      statSync(manifestPath).mtimeMs,
      mtimeBefore,
      "an in-sync run touched the manifest mtime, which dirties git status and fails the CI pack gate"
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
