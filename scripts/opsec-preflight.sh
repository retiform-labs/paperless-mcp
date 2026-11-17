#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# opsec-preflight — the mechanical gate before a PUBLIC npm release.
#
# WHY THIS FILE DOES NOT CONTAIN THE PATTERNS
#   This script ships in a public repository. A tracked pattern file IS the
#   leak: it hands a reader the complete target list in the one place they
#   would look for it. So the rules live OUTSIDE the repo, at
#   ~/.config/retiform/opsec-patterns (mode 600), overridable with
#   $OPSEC_PATTERNS. Only CATEGORIES and the LOADER are in this file.
#
# CATEGORIES THE PATTERN FILE MUST COVER   (setup: scripts/README.md)
#   1. home-path        absolute home directories  /home/<user>/
#                       — this exact class leaked twice in this repo already
#   2. internal-host    RFC1918 ranges, mesh addresses, private hostnames
#   3. credential       API keys, bearer tokens, private-key blocks,
#                       password= / secret= / _KEY= / _TOKEN=
#   4. codename         private project names that must never appear publicly
#   5. personal-marker  intimate emoji, endearments, relationship terms
#   6. local-artifact   .serena/ .env local.properties key material
#
# PATTERN FILE FORMAT
#   <category><whitespace><regex>      one rule per line
#   '#' at line start = comment; blank lines ignored.
#   The regex is used verbatim with `grep -P`, so the AUTHOR controls
#   anchoring. A short codename needs boundaries (?<!\p{L})…(?!\p{L}); a
#   path fragment does not. Do not rely on \b — it treats '_' as a word
#   character and will miss a codename joined by underscores.
#
# THREE EXIT CODES — the third is the important one
#   0  clean
#   1  a pattern hit — something sensitive is in the published set
#   2  MISCONFIGURATION — pattern file missing/unreadable/empty, or the
#      published set could not be enumerated completely. A gate that cannot
#      load its rules, or cannot see what it is meant to check, must NEVER
#      report clean. That refusal is the whole point of this script.
#
# WHAT IS AUDITED: THE PUBLISHED SET, NOT THE WORKING TREE
#   `npm pack --dry-run --json` enumerates exactly what would ship; only
#   those paths are grepped. A tree scan reports clean while one file inside
#   `files[]` leaks. Three consequences worth knowing:
#     • --ignore-scripts is REQUIRED. This repo's `prepack` runs
#       `rm -rf build && build && sync-manifest && dxt-pack` — it MUTATES
#       manifest.json. A gate observes; it never builds.
#     • An ARCHIVE in the set (paperless-mcp.dxt: 18 MB, 3.6k members) is
#       expanded and its MEMBERS scanned. The .dxt carries files that are NOT
#       in npm's `files[]` — CHANGELOG.md, AGENTS.md, CLAUDE.md — so grepping
#       the set without opening archives misses that whole surface.
#     • If `build/` is absent the set is INCOMPLETE (3 files, not 68). The
#       gate refuses with exit 2 rather than certifying a subset.
#   `git ls-files` is a fallback ONLY, and it says so loudly in the output.
#
# USAGE
#   scripts/opsec-preflight.sh               # audit the publish set
#   scripts/opsec-preflight.sh --self-test   # prove all three exit codes
#   scripts/opsec-preflight.sh --init-patterns   # write a skeleton file
#   scripts/opsec-preflight.sh --quiet       # only failures
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

DEFAULT_PATTERNS="${XDG_CONFIG_HOME:-$HOME/.config}/retiform/opsec-patterns"
PATTERNS_FILE="${OPSEC_PATTERNS:-$DEFAULT_PATTERNS}"
QUIET=0
MODE=check
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

say() { [ "$QUIET" -eq 1 ] || printf '%s\n' "$*"; }
die() { printf 'opsec-preflight: %s\n' "$*" >&2; exit 2; }

ROOT_OVERRIDE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --self-test)     MODE=self-test ;;
    --init-patterns) MODE=init ;;
    --quiet)         QUIET=1 ;;
    --root)          ROOT_OVERRIDE="${2:-}"; shift ;;
    -h|--help)       sed -n '2,63p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)               printf 'opsec-preflight: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done

# ── REDACTION ─────────────────────────────────────────────────────────────
# A preflight log is itself a leak surface: it gets pasted into chat, kept in
# CI logs, and mailed around. So a hit prints the CATEGORY and a SHAPE, never
# the value. first-4 + length is enough to locate it in the file and useless
# to whoever reads the log.
redact() {
  local v="$1" n=${#1}
  if [ "$n" -le 8 ]; then printf '****(%d chars)' "$n"
  else printf '%s…(%d chars)' "${v:0:4}" "$n"; fi
}

# ── PATTERN LOADING ───────────────────────────────────────────────────────
PAT_CAT=(); PAT_RE=()
load_patterns() {
  local f="$1" line cat re
  [ -e "$f" ] || die "pattern file not found: $f
     Refusing to report clean (exit 2). A gate that cannot load its rules
     must never pass. See scripts/README.md for the one-time setup."
  [ -r "$f" ] || die "pattern file not readable: $f — refusing to report clean (exit 2)."
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|'#'*) continue ;; esac
    cat="${line%%[[:space:]]*}"
    re="${line#"$cat"}"
    re="${re#"${re%%[![:space:]]*}"}"          # strip leading whitespace only
    [ -n "$re" ] || die "malformed rule (no regex after category '$cat') in $f"
    PAT_CAT+=("$cat"); PAT_RE+=("$re")
  done < "$f"
  [ "${#PAT_RE[@]}" -gt 0 ] || die "pattern file is empty: $f
     A gate with no rules must never pass (exit 2)."
}

build_combined() {
  local parts=() i
  for i in "${!PAT_RE[@]}"; do parts+=("(?:${PAT_RE[$i]})"); done
  local IFS='|'; printf '%s' "${parts[*]}"
}

# Which category owns this match. First rule that matches wins, so order the
# file most-specific-first: a codename containing 'password' should report as
# the codename, not as a credential.
category_of() {
  local m="$1" i
  for i in "${!PAT_RE[@]}"; do
    if printf '%s' "$m" | grep -qiP -- "${PAT_RE[$i]}" 2>/dev/null; then
      printf '%s' "${PAT_CAT[$i]}"; return 0
    fi
  done
  printf 'unknown'
}

# ── THE PUBLISHED SET ─────────────────────────────────────────────────────
# `npm pack --dry-run --json` is the ONLY authoritative answer to "what ships".
# --ignore-scripts is mandatory here, not tidiness: this repo's `prepack` runs
# `rm -rf build && npm run build && npm run sync-manifest && npm run dxt-pack`,
# which REWRITES manifest.json. A gate must observe, never mutate — and a gate
# that builds its own subject can also build itself clean.
#
# Returns "PACK" or "FALLBACK" on stdout; writes "abs<TAB>rel" lines to $2.
enumerate_set() {
  local dir="$1" out="$2" tmpjson="$3" pack_rc=0
  ( cd "$dir" && npm pack --dry-run --json --ignore-scripts ) >"$tmpjson" 2>/dev/null || pack_rc=$?
  if [ "$pack_rc" -ne 0 ] || ! grep -q '"files"' "$tmpjson" 2>/dev/null; then
    printf 'FALLBACK'; return 0
  fi
  node -e '
    const fs = require("fs");
    const j = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
    const root = process.argv[2];
    for (const f of ((j[0] && j[0].files) || []))
      process.stdout.write(root + "/" + f.path + "\t" + f.path + "\n");
  ' "$tmpjson" "$dir" > "$out"
  printf 'PACK'
}

# Completeness gate #1: every literal root in package.json `files[]` must
# actually appear in the enumerated set. Without this the gate silently
# certifies a SUBSET — measured concretely in this repo: with build/ absent
# the set is 3 files instead of 68, and all three are clean. A clean verdict
# about 4% of the payload is the exact failure this script exists to prevent.
assert_complete() {
  local dir="$1" list="$2" roots missing=""
  roots="$(node -e '
    const fs = require("fs");
    const p = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
    for (const f of (p.files || [])) process.stdout.write(f + "\n");
  ' "$dir/package.json" 2>/dev/null || true)"
  while IFS= read -r root; do
    [ -n "$root" ] || continue
    case "$root" in *'*'*) continue ;; esac   # glob entry — cannot assert directly
    if cut -f2 "$list" | grep -qxF "$root"; then continue; fi
    if cut -f2 "$list" | grep -qF -- "$root/"; then continue; fi
    missing="$missing$root
"
  done <<EOF
$roots
EOF
  [ -z "$missing" ] || { printf '%s' "$missing"; return 1; }
  return 0
}

# Completeness gate #2, independent of `files[]`: a package that declares a
# compiled entrypoint must SHIP that entrypoint. Catches the case where
# `files[]` itself is wrong, which gate #1 by construction cannot see.
assert_entrypoint() {
  local dir="$1" list="$2" main
  main="$(node -e '
    const fs = require("fs");
    const p = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
    process.stdout.write(p.main || "");
  ' "$dir/package.json" 2>/dev/null || true)"
  [ -n "$main" ] || return 0
  if cut -f2 "$list" | grep -qxF "$main"; then return 0; fi
  printf '%s' "$main"; return 1
}

# ── SCANNING ──────────────────────────────────────────────────────────────
# Archives inside the set are EXPANDED and their members scanned. Not optional
# polish: paperless-mcp.dxt ships ~3.6k members carrying CHANGELOG.md,
# AGENTS.md, CLAUDE.md and the compiled tree — none of which are in npm's
# `files[]`. A set scan that steps over archives checks the wrapper and calls
# it the parcel.
is_archive() {
  case "$1" in
    *.dxt|*.zip|*.tgz|*.jar|*.whl) return 0 ;;
  esac
  return 1
}

# scan_text <abs> <display>  → prints "  path:line: [cat] matched REDACTED"
# Returns 1 if any hit survived the allowlist, 0 otherwise. stdout is for
# reporting only, so a caller can capture it without losing the signal.
scan_text() {
  local abs="$1" disp="$2" hits ln rest cat tok filtered
  # -i is NOT optional. A codename appears capitalised in prose ("Mneme") and
  # lowercased in code ("mneme"); a case-sensitive sweep reports clean while
  # the name sits in the README. Learned the hard way in a sibling repo: the
  # first draft used grep -E and silently missed a lowercase fixture tag.
  hits="$(grep -aniP -- "$COMBO" "$abs" 2>/dev/null || true)"
  [ -n "$hits" ] || return 0
  if [ -n "$ALLOW_RE" ]; then
    # The allowlist is matched against "<disp>:<line>:<content>", never against
    # the bare line. File-scoped on purpose: an entry for one file's safety
    # advice must not be able to suppress a hit in another file. The first cut
    # filtered the line alone, which made a path-bearing ALLOW_RE unmatchable
    # — a guard that could never fire, which is the same as no guard.
    # Accumulate WITHOUT a trailing blank record. The first cut appended a
    # newline per kept line into a string that already ended with one, so the
    # later `while read` loop saw an extra empty line and reported it as a hit
    # with an empty match — a phantom finding invented by the filter itself.
    filtered=""
    while IFS= read -r one; do
      [ -n "$one" ] || continue
      if printf '%s:%s' "$disp" "$one" | grep -qP -- "$ALLOW_RE" 2>/dev/null; then
        continue
      fi
      filtered="${filtered:+$filtered$'\n'}$one"
    done <<EOF
$hits
EOF
    hits="$filtered"
    [ -n "$hits" ] || return 0
  fi
  while IFS= read -r one; do
    [ -n "$one" ] || continue
    ln="${one%%:*}"; rest="${one#*:}"
    case "$ln" in ''|*[!0-9]*) continue ;; esac
    cat="$(category_of "$rest")"
    # Redact the MATCHED TOKEN, not the line: a line-length tells the reader
    # nothing about the secret and looks like a deliberate obfuscation.
    tok="$(printf '%s' "$rest" | grep -oiP -- "$COMBO" 2>/dev/null | head -1 || true)"
    [ -n "$tok" ] || tok="$rest"
    printf '  %s:%s: [%s] matched %s\n' "$disp" "$ln" "$cat" "$(redact "$tok")"
  done <<EOF
$hits
EOF
  return 1   # inverted on purpose: 1 = "found something"
}

# scan_archive <abs> <display> → returns 1 if ANY member hit, 0 if clean,
# 2 if the archive could not be expanded (an unscanned archive must not pass).
scan_archive() {
  local abs="$1" disp="$2" xdir n=0 member result=0 vendored=0
  xdir="$(mktemp -d)"
  if ! unzip -qq -o "$abs" -d "$xdir" >/dev/null 2>&1; then
    rm -rf "$xdir"
    printf '  %s: [local-artifact] archive could not be expanded — NOT SCANNED\n' "$disp"
    return 2
  fi
  while IFS= read -r member; do
    [ -n "$member" ] || continue
    # Vendored third-party code is not our authorship and is not a leak signal:
    # `password` inside axios's basic-auth helper is a library implementation,
    # not a secret we published. Excluded and COUNTED — silence would hide a
    # vendoring mistake, which is the thing worth seeing.
    case "/$member" in
      */node_modules/*) vendored=$((vendored + 1)); continue ;;
    esac
    n=$((n + 1))
    if scan_text "$xdir/$member" "$disp::$member"; then :; else result=1; fi
  done <<EOF
$( cd "$xdir" && find . -type f -printf '%P\n' 2>/dev/null || true )
EOF
  rm -rf "$xdir"
  say "  ($n members of $(basename "$disp") scanned; $vendored vendored under node_modules/ skipped)"
  return "$result"
}

run_check() {
  local dir="$1" label="${2:-publish set}"
  local tmp out json src miss ep count abs disp n
  tmp="$(mktemp -d)"; out="$tmp/list"; json="$tmp/pack.json"
  src="$(enumerate_set "$dir" "$out" "$json")"

  if [ "$src" = "FALLBACK" ]; then
    printf '⚠  opsec-preflight: `npm pack --dry-run --json` UNAVAILABLE.\n'
    printf '   Falling back to `git ls-files` — THIS IS NOT THE PUBLISHED SET.\n'
    printf '   Files in package.json "files" that are uncommitted, and files\n'
    printf '   excluded by .npmignore, are both mis-modelled by this fallback.\n'
    ( cd "$dir" && git ls-files ) > "$tmp/names"
    paste <(sed "s|^|$dir/|" "$tmp/names") "$tmp/names" > "$out"
  else
    miss="$(assert_complete "$dir" "$out" || true)"
    if [ -n "$miss" ]; then
      printf 'opsec-preflight: INCOMPLETE PUBLISH SET — declared in files[] but absent:\n'
      printf '     %s\n' "$miss"
      printf '  Refusing to report clean (exit 2). Build first, then re-run.\n'
      rm -rf "$tmp"; return 2
    fi
    ep="$(assert_entrypoint "$dir" "$out" || true)"
    if [ -n "$ep" ]; then
      printf 'opsec-preflight: entrypoint %s not in publish set — refusing (exit 2).\n' "$ep"
      rm -rf "$tmp"; return 2
    fi
  fi

  count="$(wc -l < "$out" | tr -d ' ')"
  say "── $label: $count files (source: $src) ──"

  local hits=0 bad=0 arc_rc=0
  while IFS=$'\t' read -r abs disp; do
    [ -n "${disp:-}" ] || continue
    [ -f "$abs" ] || continue
    if is_archive "$disp"; then
      arc_rc=0
      scan_archive "$abs" "$disp" || arc_rc=$?
      [ "$arc_rc" -eq 0 ] || hits=1
      [ "$arc_rc" -eq 2 ] && bad=1
    else
      scan_text "$abs" "$disp" || hits=1
    fi
  done < "$out"

  rm -rf "$tmp"
  if [ "$hits" -eq 1 ]; then
    printf '\nopsec-preflight: FAILED — sensitive material in the publish set.\n'
    return 1
  fi
  if [ "$bad" -eq 1 ]; then
    printf '\nopsec-preflight: FAILED — an archive could not be scanned (exit 2).\n'
    return 2
  fi
  return 0
}

# ── SELF-TEST ─────────────────────────────────────────────────────────────
# Proves all three exit codes against a THROWAWAY fixture, never against the
# live repo. The canary is assembled from two halves so this file does not
# literally contain a string its own pattern set would flag — the gate must
# still be able to scan itself.
run_self_test() {
  local tmp canary rc fails=0
  tmp="$(mktemp -d)"
  canary="CANARY""_SELFTEST_9f3a2b"

  printf 'opsec-preflight self-test\n'
  printf 'fixture: %s\n\n' "$tmp"

  # A pattern file that exists ONLY for this test, so the test cannot pass by
  # accident on a machine with no real pattern file.
  printf '%s\n' "personal-marker ${canary}" > "$tmp/patterns"
  chmod 600 "$tmp/patterns"

  # ── case 1: a hit must be found, named, and REDACTED ────────────────────
  mkdir -p "$tmp/repo1/build"
  printf '{"name":"fixture","version":"1.0.0","main":"build/index.js","files":["build"]}\n' > "$tmp/repo1/package.json"
  printf 'const k = "%s";\n' "$canary" > "$tmp/repo1/build/index.js"
  printf 'clean line\n' > "$tmp/repo1/build/other.js"
  out="$(OPSEC_PATTERNS="$tmp/patterns" bash "$SELF" --quiet --root "$tmp/repo1" 2>&1)" && rc=0 || rc=$?
  if [ "$rc" -eq 1 ]; then
    printf '✓ case 1: hit detected → exit 1\n'
    printf '%s\n' "$out" | sed 's/^/    /'
    case "$out" in
      *"build/index.js"*) printf '    ✓ correct file named\n' ;;
      *) printf '    ✗ WRONG FILE named\n'; fails=$((fails+1)) ;;
    esac
    case "$out" in
      *"$canary"*) printf '    ✗ SECRET LEAKED verbatim into the report\n'; fails=$((fails+1)) ;;
      *) printf '    ✓ value redacted in the report\n' ;;
    esac
    case "$out" in
      *"[personal-marker]"*) printf '    ✓ category printed\n' ;;
      *) printf '    ✗ category missing\n'; fails=$((fails+1)) ;;
    esac
  else
    printf '✗ case 1: expected exit 1, got %s\n%s\n' "$rc" "$out"; fails=$((fails+1))
  fi

  # ── case 2: a clean set must pass ───────────────────────────────────────
  rm -rf "$tmp/repo1/build"; mkdir -p "$tmp/repo1/build"
  printf 'const k = 1;\n' > "$tmp/repo1/build/index.js"
  printf 'clean line\n' > "$tmp/repo1/build/other.js"
  out="$(OPSEC_PATTERNS="$tmp/patterns" bash "$SELF" --quiet --root "$tmp/repo1" 2>&1)" && rc=0 || rc=$?
  if [ "$rc" -eq 0 ]; then printf '✓ case 2: clean set → exit 0\n'
  else printf '✗ case 2: expected exit 0, got %s\n%s\n' "$rc" "$out"; fails=$((fails+1)); fi

  # ── case 3: THE IMPORTANT ONE — no pattern file must REFUSE ─────────────
  out="$(OPSEC_PATTERNS="$tmp/does-not-exist" bash "$SELF" --quiet --root "$tmp/repo1" 2>&1)" && rc=0 || rc=$?
  if [ "$rc" -eq 2 ]; then
    printf '✓ case 3: missing pattern file → exit 2 (refuses to report clean)\n'
    printf '%s\n' "$out" | sed 's/^/    /'
  else
    printf '✗ case 3: expected exit 2, got %s — A GATE THAT CANNOT LOAD ITS RULES PASSED\n' "$rc"
    printf '%s\n' "$out" | sed 's/^/    /'
    fails=$((fails+1))
  fi

  # ── case 4: an EMPTY pattern file must also refuse ──────────────────────
  : > "$tmp/empty-patterns"
  out="$(OPSEC_PATTERNS="$tmp/empty-patterns" bash "$SELF" --quiet --root "$tmp/repo1" 2>&1)" && rc=0 || rc=$?
  if [ "$rc" -eq 2 ]; then printf '✓ case 4: empty pattern file → exit 2\n'
  else printf '✗ case 4: expected exit 2, got %s\n' "$rc"; fails=$((fails+1)); fi

  # ── case 5: an INCOMPLETE set must refuse, not certify the subset ───────
  # package.json declares build/ but build/ is missing → npm pack returns 2
  # files, both clean. Passing here would be the exact defect this gate exists
  # to prevent, so it MUST exit 2.
  mkdir -p "$tmp/repo2"
  printf '{"name":"fixture2","version":"1.0.0","main":"build/index.js","files":["build"]}\n' > "$tmp/repo2/package.json"
  out="$(OPSEC_PATTERNS="$tmp/patterns" bash "$SELF" --quiet --root "$tmp/repo2" 2>&1)" && rc=0 || rc=$?
  if [ "$rc" -eq 2 ]; then printf '✓ case 5: incomplete publish set → exit 2\n'
  else printf '✗ case 5: expected exit 2, got %s — certified a subset as clean\n' "$rc"; printf '%s\n' "$out" | sed 's/^/    /'; fails=$((fails+1)); fi

  rm -rf "$tmp"
  printf '\n'
  if [ "$fails" -eq 0 ]; then
    printf 'self-test: ALL PASSED — exit 0 is clean, exit 1 is a hit, exit 2 is a refusal.\n'
    return 0
  fi
  printf 'self-test: %s FAILURE(S)\n' "$fails"
  return 1
}

# ── INIT PATTERN FILE (skeleton, no secrets) ──────────────────────────────
init_patterns() {
  local f="$1"
  if [ -e "$f" ]; then printf 'opsec-preflight: %s already exists — not overwriting.\n' "$f"; exit 0; fi
  mkdir -p "$(dirname "$f")"
  cat > "$f" <<'SKELETON'
# Retiform opsec pattern set — PRIVATE, never committed, never shipped.
# Format:  <category><whitespace><regex>   (one rule per line, # = comment)
# The regex is used verbatim with `grep -P`. YOU control the anchoring:
# short private names need (?<!\p{L})…(?!\p{L}); paths and IPs do not.
# ORDER MATTERS: the first rule that matches decides the printed category,
# so put the most specific rules first.

# ── 1. home-path — absolute home dirs. Leaked twice in this repo already.
home-path        /home/[a-z][a-z0-9_-]*/
home-path        /Users/[A-Za-z][A-Za-z0-9_-]*/

# ── 2. internal-host — RFC1918 + mesh + private hostnames. Replace the
#       hostname list with YOUR fleet's names.
internal-host    (?<![0-9])192\.168\.[0-9]+\.[0-9]+(?![0-9])
internal-host    (?<![0-9])10\.[0-9]+\.[0-9]+\.[0-9]+(?![0-9])
internal-host    (?<![0-9])172\.(1[6-9]|2[0-9]|3[01])\.[0-9]+\.[0-9]+(?![0-9])

# ── 3. credential — shapes only; never list a real key here.
credential       -----BEGIN [A-Z ]*PRIVATE KEY-----
credential       (?i)\b(bearer)[[:space:]]+[A-Za-z0-9._-]{20,}
credential       (?i)\b(password|passwd|secret|api[_-]?key|access[_-]?token)[[:space:]]*[:=][[:space:]]*['"]?[^[:space:]'"]{6,}
credential       (?i)\b[A-Z0-9_]*(_KEY|_TOKEN|_SECRET|_PASSWORD)[[:space:]]*=
credential       (?i)\b(sk|pk|ghp|gho|ghs|xox[baprs])[-_][A-Za-z0-9]{16,}

# ── 4. codename — private project names. Boundaries REQUIRED, and add '\*'
#       only if a name must also match inside a compound token.
codename         (?<!\p{L})REPLACE_WITH_PRIVATE_CODENAME(?!\p{L})

# ── 5. personal-marker — intimate emoji and endearments. Never ship these.
personal-marker  (?<!\p{L})(babe|boo|sweetheart|darling|my love)(?!\p{L})
personal-marker  [\x{1F48B}\x{1F618}\x{1F970}\x{1F633}]

# ── 6. local-artifact — editor/agent/secret state that must never ship.
local-artifact   \.serena/
local-artifact   (?<!\p{L})\.env(\.[a-z]+)?(?!\p{L})
local-artifact   local\.properties
local-artifact   \.aibtc/
SKELETON
  chmod 600 "$f"
  printf 'opsec-preflight: wrote skeleton pattern file → %s (mode 600)\n' "$f"
  printf 'EDIT IT: the codename line is a placeholder and must be replaced.\n'
  exit 0
}

# ── MAIN ──────────────────────────────────────────────────────────────────
# --root is used only by --self-test and by fixtures; the normal invocation
# audits the repo that contains this script.
case "$MODE" in
  init)      init_patterns "$PATTERNS_FILE" ;;
  self-test) run_self_test; exit $? ;;
esac

if [ -n "$ROOT_OVERRIDE" ]; then
  REPO_ROOT="$( cd "$ROOT_OVERRIDE" && pwd )"
else
  REPO_ROOT="$( cd "$(dirname "$SELF")/.." && pwd )"
fi

load_patterns "$PATTERNS_FILE"
COMBO="$(build_combined)"

# Allowlist — verified-benign lines only, each with its reason. Kept SHORT on
# purpose: an allowlist that grows without justification is how a real leak
# gets waved through. Every entry is matched against "path:line:content", so it
# is scoped to ONE file and ONE concern; it can never suppress a codename or a
# credential hit in a different file.
ALLOW_RE='paperless-mcp\.dxt::CLAUDE\.md:[0-9]+:.*Environment files'
#   CLAUDE.md:148 reads "Do not commit: Environment files (.env, .env.local,
#   .env.*.local)". That is documentation TELLING a reader not to commit
#   secrets — flagging it is a false positive, and a gate that cries wolf on
#   its own safety advice is a gate people learn to ignore. Verified by reading
#   the line; the same sentence appears in AGENTS.md:28 for bare `.env`, which
#   the rule no longer matches because it requires a path shape.
say "opsec-preflight: ${#PAT_RE[@]} rules from $PATTERNS_FILE"
# NOTE: must not be a bare call — with `set -e`, a non-zero return would end
# the script before $? could be captured, and the exit code IS the contract.
rc=0
run_check "$REPO_ROOT" || rc=$?
exit "$rc"
