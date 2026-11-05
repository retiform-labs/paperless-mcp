---
"@retiform/paperless-mcp": patch
---

Update transitive dependencies to clear published security advisories.

 reported 11 advisories (1 critical, 5 high, 5 moderate)
against the locked tree. All but one are now resolved by moving within existing
ranges, so  is unchanged and only the lockfile moves:

- proxy-addr 2.0.7 -> 2.0.8 (critical: IP spoofing via IPv4-mapped IPv6)
- body-parser 2.3.0 -> 2.4.0 (high: DoS via trailing bytes after compressed body)
- follow-redirects 1.16.0 -> 1.16.2 (high: credential headers following redirects)
- path-to-regexp 8.2.0 -> 8.4.2 (high: ReDoS via sequential optional groups)
- @modelcontextprotocol/sdk 1.30.0 -> 1.32.2 (high, direct)
- express 5.2.1 -> 5.3.0, axios 1.20.0 -> 1.21.0

One advisory remains and is deliberately not fixed: zod is held at 4.4.3 because
versions from 4.5.0 onward collapse unions of unchecked primitives into an
array-form , which strict MCP clients reject and silently drop the whole
tool. That regression is pinned by , which fails on the
bump. The advisory (memory exhaustion while validating oversized constrained
collections) is a DoS reachable only from input the caller already controls, and
trading it for a schema the clients cannot read is the worse exchange. Revisit
when zod's converter stops emitting the array form.

No source changes. 141/141 tests pass;  clean; the build is green
run twice in a dirty tree.
