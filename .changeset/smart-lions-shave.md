---
"@retiform/paperless-mcp": patch
---

Harden the surfaces this package ships: input bounds, a body-size fix, an error
path that leaked internals, and the preflight gate that was missing.

- **Bound four unbounded input paths.** `custom_field_query` had no recursion limit and no clause cap -- a deeply nested group exhausted the stack and surfaced as a transport error instead of a validation error. It now has a depth bound of 8 and a clause cap of 64. `paperless_filters` values are capped at 1024 characters and 500 entries, matching the limits Paperless-ngx declares on the same fields. Mail-rule strings are capped at 256 characters, which is Paperless' own `maxLength` rather than a number chosen here.

- **Refuse hostile leaf values, not just bad structure.** `custom_field_query` leaf values must be a primitive or an array of primitives. An object, an `undefined`, a nested array, a function or a symbol was previously accepted and coerced into a GET parameter.

- **Raise the request body bound in `--http` mode.** express' 100 kB default meant a legal base64 upload -- a document at the 100 MB `MAX_FILE_SIZE_BYTES` ceiling arrives as ~133 MB of JSON -- was refused by the transport before any tool ran. The new bound is 140 MB, overridable with `PAPERLESS_MCP_MAX_BODY_BYTES`.

- **Stop disclosing stack traces to unauthenticated callers.** `express.json()` runs before the auth middleware, so a malformed body reached express' default error handler, which returns absolute filesystem paths and module names. The server now answers with a fixed minimal body and preserves the status code.

- **Fix `search_passages` widening the window around the wrong phrase.** The window was widened by the longest query term rather than by the highlighted phrase that resolved it, so a narrow multi-word hit could return a passage that did not contain the phrase its `highlights` field quoted.

- **Generate the `manifest.json` `tools` array from the server.** It was hand-maintained and listed 13 of the 48 tools the server registers, so a DXT host saw a third of the surface.

- **Fix `parseDocumentId` coercing non-decimal ids.** `0x10` was read as 16 and `1e3` as 1000 -- one id in, a different real document out.

- **Stop shipping the test mocks in `build/`.** `tsconfig.build.json` excluded `src/**/__mocks__/**`, but the directory is `src/test/mocks`.

- **Move to zod 4.6.7.** Clears CVE-2023-54404 / GHSA-mrhx-c7xc-m3w5 (CWE-770, CVSS 7.5), which affects `<4.6.6`. Note the pair with the bounds above: the advisory is about a union expanding without bound, and in 4.4.3 a member-level `.max()` does not mitigate it because the length check runs after the element loop.

- **Add the opsec preflight gate this repo was missing.** Runs over the staged tree, the commit message, and for releases the full history.

- **Pin the e2e matrix to a specific Paperless-ngx patch.** Tracking `latest` made a red PR check unattributable: it passed on v3.4.2 and failed on v3.5.0 with no change on our side. Drift detection moves to a scheduled job that opens an issue.
