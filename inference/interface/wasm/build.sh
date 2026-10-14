#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p dist
CGO_ENABLED=0 GOOS=js GOARCH=wasm go build -trimpath -ldflags='-s -w' -o dist/latti_client.wasm ./cmd/latti-client-wasm
rm -f dist/wasm_exec.js
cp "$(go env GOROOT)/lib/wasm/wasm_exec.js" dist/wasm_exec.js
cp src/index.js src/worker.js src/csv.js src/index.d.ts dist/
sha256sum dist/latti_client.wasm dist/wasm_exec.js dist/worker.js dist/index.js > SHA256SUMS
