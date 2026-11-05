// The DXT manifest carried its own copy of the version, and changesets only
// bumps package.json, so the two drifted on every bot-driven release: 2.3.3
// shipped a 2.3.2 manifest and 2.3.5 shipped a 2.3.4 one. Declaring a version
// twice lets it drift, so package.json is the single source and this stamps it
// into the manifest before the bundle is packed.
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const Q = String.fromCharCode(34);
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const pkg = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
const manifestPath = join(root, "manifest.json");
const raw = readFileSync(manifestPath, "utf8");

const key = Q + "version" + Q;
const at = raw.indexOf(key);
if (at === -1) {
  console.error("sync-manifest-version: no version field in manifest.json");
  process.exit(1);
}

const colon = raw.indexOf(":", at);
const open = raw.indexOf(Q, colon);
const close = raw.indexOf(Q, open + 1);
const current = raw.slice(open + 1, close);

if (current === pkg.version) {
  console.log("manifest.json version already " + pkg.version);
} else {
  writeFileSync(manifestPath, raw.slice(0, open + 1) + pkg.version + raw.slice(close));
  console.log("manifest.json version " + current + " -> " + pkg.version);
}
