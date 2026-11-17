# scripts/

## Opsec preflight — the gate before a public release

`scripts/opsec-preflight.sh` audits **what would actually ship** against a
private pattern set, and refuses to report clean if it cannot load its rules.

    bash scripts/opsec-preflight.sh              # audit the published set
    bash scripts/opsec-preflight.sh --self-test  # prove all three exit codes
    bash scripts/opsec-preflight.sh --quiet      # only failures

### Exit codes — the third is the point

| code | meaning |
|---|---|
| `0` | clean |
| `1` | a pattern hit — something sensitive is in the published set |
| `2` | **misconfiguration** — pattern file missing, unreadable or empty, or the published set could not be enumerated completely |

Exit `2` exists because a gate that cannot load its rules, or cannot see what it
is meant to check, must **never** report clean. A missing pattern file is not a
pass; it is a broken gate, and it says so.

### One-time setup on a new machine

The pattern file lives **outside the repository** on purpose. A tracked pattern
file is itself the leak: it hands a reader the complete target list in the one
place they would look for it. Only the categories and the loader are in the
script.

    bash scripts/opsec-preflight.sh --init-patterns

That writes a skeleton to `~/.config/retiform/opsec-patterns` with mode `600`.
It is a **skeleton** — the `codename` line is a literal placeholder
(`REPLACE_WITH_PRIVATE_CODENAME`) and must be edited before the gate is
trustworthy. Override the location with `$OPSEC_PATTERNS`:

    OPSEC_PATTERNS=/secure/path/patterns bash scripts/opsec-preflight.sh

File format — one rule per line, `#` at line start is a comment:

    <category><whitespace><regex>

The regex is used verbatim with `grep -P` and is case-insensitive, so **you**
control the anchoring. Two rules that matter:

* A short private name needs boundaries: `(?<!\p{L})name(?!\p{L})`.
  Do **not** use `\b` — it treats `_` as a word character and will miss a name
  joined by underscores.
* Order rules most-specific-first. The first rule that matches decides the
  category printed.

### Categories

| category | covers |
|---|---|
| `home-path` | absolute home directories — this class leaked twice in this repo |
| `internal-host` | the fleet's own private ranges and host addresses |
| `credential` | API keys, bearer tokens, private-key blocks, and `password=` / `_TOKEN=` followed by a real literal value |
| `codename` | private project names that must never appear publicly |
| `personal-marker` | intimate emoji and endearments |
| `local-artifact` | `.serena/`, `.env.local`, `local.properties`, key material |

### What is audited, and why it is the published set

The gate runs `npm pack --dry-run --json --ignore-scripts` and greps only the
files npm reports. A tree scan reports clean while one file inside `files[]`
leaks — which is the whole failure mode this exists to catch.

Three consequences worth knowing:

* **`--ignore-scripts` is required.** This repo's `prepack` runs
  `rm -rf build && npm run build && npm run sync-manifest && npm run dxt-pack`,
  which **rewrites `manifest.json`**. A gate observes; it never builds. A gate
  that builds its own subject can also build itself clean.
* **Archives in the set are expanded and their members scanned.**
  `paperless-mcp.dxt` carries `CHANGELOG.md`, `AGENTS.md`, `CLAUDE.md` and the
  compiled tree — **none of which are in npm `files[]`**. The number of members
  scanned and the number of vendored members skipped is printed on every run.
* **Vendored `node_modules/` members are skipped and counted.** `password`
  inside axios's basic-auth helper is a library implementation, not a secret we
  published. The count is printed so a vendoring mistake stays visible.

If `build/` is absent the enumerated set is incomplete (3 files instead of 68).
The gate detects that against `package.json` `files[]` and the declared `main`,
and **exits 2** rather than certifying a subset.

`git ls-files` is used **only** if `npm pack` is unavailable, and the output
says so loudly — it is not the published set.

### A preflight log is itself a leak surface

On a hit the gate prints `path:line: [category] matched <first-4>…(N chars)`.
It never prints the matched value. Logs get pasted into chat, kept in CI and
mailed around; a report that quotes the secret is a second copy of the leak.

### Where the gate runs, and where it must not

**The gate is LOCAL ONLY. It is deliberately not wired into `prepublishOnly`.**

The pattern file holds vocabulary that must never be published -- the same restricted-practice terms the gate exists to keep out of the tree. It therefore cannot be committed, and a GitHub Actions runner has no copy of it. Wiring the gate into `prepublishOnly` makes publishing *structurally impossible* in CI: the gate correctly refuses to report clean without its rules, npm aborts, and the release fails about twenty seconds in. That is the gate working and the wiring being wrong, and it happened once -- see the 2.4.1 release run.

| context | gate | why |
|---|---|---|
| local, before `git push` | the `pre-push` hook | patterns are present |
| local, before `npm publish` | `npm run release:gate` | patterns are present |
| CI (`changesets/action`) | **not run** | patterns cannot exist there |

`prepublishOnly` runs `npm run build` only. The pre-push hook remains the enforcement point that matters, because the hook sees the **commits**, and a leak in a commit is the leak that reaches the registry even when the working tree is clean.

If you ever do need the gate in CI, it has to come from a repository secret written to a mode-600 temp path with `OPSEC_PATTERNS` pointing at it -- and that trade puts the private vocabulary into a second system. Prefer the hook.
