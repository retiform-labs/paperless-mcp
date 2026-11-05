---
"@retiform/paperless-mcp": patch
---

Derive the DXT manifest version from `package.json` at pack time.

The manifest carried its own copy of the version, and `changeset publish` bumps
only `package.json`, so the two drifted on every bot-driven release: 2.3.3
shipped a bundle declaring 2.3.2, and 2.3.5 shipped one declaring 2.3.4. The
bundle is inside the tarball, so every install of those versions downloaded a
manifest that disagreed with the package around it.

`prepack` now stamps the manifest from the package version before the bundle is
built, so the two cannot diverge. Two regression tests pin the mechanism: one
drives the script over a deliberately stale manifest and asserts it is corrected
without disturbing other fields, one asserts the script is idempotent.

Verified on the built artifact rather than the source tree: the 2.3.6 tarball
carries `2.3.6` in both the package and the embedded bundle, where the 2.3.5
tarball carries `2.3.5` and `2.3.4`.
