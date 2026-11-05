import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, cpSync, mkdtempSync, rmSync } from "node:fs";
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

// NOTE: there is deliberately no assertion that the TRACKED manifest matches
// package.json. The changesets Version PR bumps package.json only, so that
// state is legitimately unequal between a version bump and the next pack; a
// test on it would fail for a correct reason. What is pinned below is the
// mechanism - prepack stamps the manifest from the package version - which is
// the property the published artifact actually depends on.

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
