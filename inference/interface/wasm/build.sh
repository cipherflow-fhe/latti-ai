#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p dist
go run ./cmd/profile ../../lattisense/frontend/parameter.json src/default-profile.json
node build-info.mjs
runtime_version="$(node -p "JSON.parse(require('fs').readFileSync('package.json', 'utf8')).version")"
build_id="$(node --input-type=module -e "import {BUILD_INFO} from './src/build-info.js'; process.stdout.write(BUILD_INFO.build)")"
CGO_ENABLED=0 GOOS=js GOARCH=wasm go build -trimpath -ldflags="-s -w -X main.runtimeVersion=$runtime_version -X main.buildID=$build_id" -o dist/latti_client.wasm ./cmd/latti-client-wasm
rm -f dist/wasm_exec.js
cp "$(go env GOROOT)/lib/wasm/wasm_exec.js" dist/wasm_exec.js
cp src/index.js src/worker.js src/csv.js src/zip.js src/index.d.ts src/build-info.js src/io.js src/default-profile.json dist/
cp src/example.html dist/
node --input-type=module -e 'import {cp,rm} from "node:fs/promises"; await rm("dist/tasks",{recursive:true,force:true}); await cp("src/tasks","dist/tasks",{recursive:true,filter:p=>!p.split("/").includes("__pycache__")});'
node build-manifest.mjs
