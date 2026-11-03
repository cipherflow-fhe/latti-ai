#!/usr/bin/env bash
# Activate the latti-ai Python environment for compiler/generator commands.
# Retina-DKD remains a model/data workspace; its virtualenv is not suitable
# for latti-ai ONNX/compiler imports.

set -euo pipefail

LattiRoot="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$LattiRoot/.venv/bin/activate"

export PYTHONPATH="$LattiRoot:$LattiRoot/inference:$LattiRoot/inference/lattisense:$LattiRoot/training/model_compiler${PYTHONPATH:+:$PYTHONPATH}"

echo "Using Python: $(command -v python)"
python - <<'PY'
import sys
print(f"Python: {sys.executable}")
import numpy
import onnx
print(f"NumPy: {numpy.__version__}")
print(f"ONNX: {onnx.__version__}")
PY
