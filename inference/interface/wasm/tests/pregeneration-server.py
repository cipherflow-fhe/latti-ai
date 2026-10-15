"""Import the production browser eval file using the existing native server."""

import json
import sys
import time
from pathlib import Path

import latti_inference

artifacts, repo = map(Path, sys.argv[1:])
server = latti_inference.InferenceServer(str(repo / 'examples/test_cifar10/task/server'), True, 0)
server.import_eval_context_file(str(artifacts / 'N16QP1546H192H32/public/eval_context.bin'))
server.load_model()
start = time.monotonic()
output = server.evaluate({'input': (artifacts / 'input.ct.bin').read_bytes()})
elapsed = time.monotonic() - start
(artifacts / 'server-output.bin').write_bytes(output['output'])
plaintext = server.evaluate_plaintext({'input': str(artifacts / 'resnet20/img.csv')})['output']
(artifacts / 'server.json').write_text(json.dumps({'plaintext': plaintext, 'gpuEvaluateSeconds': elapsed}, indent=2))
