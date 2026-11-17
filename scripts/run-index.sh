#!/usr/bin/env bash
# Run the corpus index build and PROPAGATE ITS EXIT CODE.
#
# ⚠️ WHY THIS FILE EXISTS. The previous wrapper ended with
#
#     npx ts-node scripts/build-index.ts > "$log" 2>&1
#     echo "EXIT=$?"
#
# and `echo` succeeds. So the last command in the unit exited 0 no matter what
# the build did: `systemctl is-active` reported inactive while the process had
# in fact exited 1, and the only evidence of the failure was a log nobody had a
# reason to read, because the run had reported success. A wrapper that cannot
# fail is worse than no wrapper.
#
# The rule this encodes: the LAST command in a runner must be the thing whose
# status you want, or must re-raise it. Capturing `$?` and then echoing it is
# the bug -- the capture is right, the echo overwrites it.
#
# Exit codes, so a caller can act without parsing:
#   0  complete
#   1  failed (embedding outage, unrecoverable error)
#   2  misconfigured (missing env file, missing npx)
#   3  partial (some documents failed; the index is usable but incomplete)

set -uo pipefail

ENV_FILE="${PAPERLESS_INDEX_ENV:-$HOME/.paperless-index-build.env}"
LOG_FILE="${PAPERLESS_INDEX_LOG:-/tmp/paperless-index-build.log}"
NODE_BIN_DIR="${NODE_BIN_DIR:-$HOME/.nvm/versions/node/v24.21.0/bin}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ ! -f "$ENV_FILE" ]; then
  echo "run-index: no env file at $ENV_FILE" >&2
  exit 2
fi

export PATH="$NODE_BIN_DIR:$PATH"
if ! command -v npx >/dev/null 2>&1; then
  echo "run-index: npx not on PATH (looked in $NODE_BIN_DIR)" >&2
  exit 2
fi

# shellcheck disable=SC1090
set -a
. "$ENV_FILE"
set +a

cd "$PROJECT_DIR" || { echo "run-index: cannot enter $PROJECT_DIR" >&2; exit 2; }

echo "run-index: building, log at $LOG_FILE" >&2
npx ts-node scripts/build-index.ts > "$LOG_FILE" 2>&1
rc=$?

# Append the code to the log for readers, but DO NOT let tee's status become
# the script's. `$rc` is the build's; the exit below is the script's.
echo "EXIT=$rc" >> "$LOG_FILE"
echo "run-index: finished with $rc" >&2
exit "$rc"
