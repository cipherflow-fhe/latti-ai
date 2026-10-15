#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
export LATTI_BROWSER_ARTIFACTS="${LATTI_BROWSER_ARTIFACTS:-$(mktemp -d /tmp/latti-workflow.XXXXXX)}"
printf 'Artifacts: %s\n' "$LATTI_BROWSER_ARTIFACTS"
dbus-run-session -- xvfb-run -a bash -c '
  /usr/bin/python3 tests/directory-portal.py > "$LATTI_BROWSER_ARTIFACTS/portal.log" 2>&1 &
  portal_pid=$!
  trap '\''kill "$portal_pid" 2>/dev/null || true'\'' EXIT
  node tests/workflow.mjs
'
