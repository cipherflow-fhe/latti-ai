#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")"
output_dir="${1:-$PWD}"
mkdir -p "$output_dir"

npm run build
package_version="$(node -p "JSON.parse(require('fs').readFileSync('package.json', 'utf8')).version")"
package_name="latti-client-wasm-${package_version}"
stage_dir="$(mktemp -d "${TMPDIR:-/tmp}/latti-client-wasm.XXXXXX")"
trap 'rm -rf "$stage_dir"' EXIT

mkdir -p "$stage_dir/$package_name"
cp -a dist README.md SHA256SUMS package.json "$stage_dir/$package_name/"
rm -f "$output_dir/$package_name.zip"
(cd "$stage_dir" && zip -qr "$output_dir/$package_name.zip" "$package_name")
printf '%s\n' "$output_dir/$package_name.zip"

task_source="../../../examples/test_cifar10/task/client"
mkdir -p "$stage_dir/tasks/resnet20/client"
cp "$task_source/"{task_config.json,ckks_parameter.json,preprocess.js,postprocess.js} "$stage_dir/tasks/resnet20/client/"
rm -f "$output_dir/latti-client-tasks.zip"
(cd "$stage_dir" && zip -qr "$output_dir/latti-client-tasks.zip" tasks)
printf '%s\n' "$output_dir/latti-client-tasks.zip"

cp -a ../../../../datasets/CIFAR-10/dataset "$stage_dir/dataset"
rm -f "$output_dir/latti-dataset-cifar10.zip"
(cd "$stage_dir" && zip -qr "$output_dir/latti-dataset-cifar10.zip" dataset)
printf '%s\n' "$output_dir/latti-dataset-cifar10.zip"
