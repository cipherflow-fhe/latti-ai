#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
output_dir="${1:-$PWD}"
mkdir -p "$output_dir"
output_dir="$(cd -- "$output_dir" && pwd)"
npm run build
package_version="$(node -p "JSON.parse(require('fs').readFileSync('package.json', 'utf8')).version")"
package_name="latti-client-wasm-${package_version}"
stage_dir="$(mktemp -d "${TMPDIR:-/tmp}/latti-client-wasm.XXXXXX")"
trap 'rm -rf "$stage_dir"' EXIT
mkdir -p "$stage_dir/$package_name"
cp -a dist README.md SHA256SUMS package.json "$stage_dir/$package_name/"
(cd "$stage_dir/$package_name" && sha256sum -c SHA256SUMS)
rm -f "$output_dir/$package_name.zip"
(cd "$stage_dir" && zip -qr "$output_dir/$package_name.zip" "$package_name")
cp "$output_dir/$package_name.zip" "$output_dir/WASM-$package_version.zip"
printf '%s\n' "$output_dir/$package_name.zip" "$output_dir/WASM-$package_version.zip"
