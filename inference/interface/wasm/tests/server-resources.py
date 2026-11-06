"""Two real server owners: busy release, continued inference, then pool reload."""

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import zipfile

import latti_inference
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=Path, required=True)
parser.add_argument('--keys', type=Path, required=True)
parser.add_argument('--workflow', type=Path, required=True)
parser.add_argument('--report', type=Path, required=True)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()
config = json.loads((args.task / 'client/task_config.json').read_text())
case = json.loads((args.workflow / 'cases.json').read_text())[0]
reference = json.loads((args.workflow / 'native.json').read_text())[0]['plaintext']
with zipfile.ZipFile(args.workflow / 'input.zip') as archive:
    inputs = {name: archive.read(case['case_id'] + ('.ct' if len(config['task_input_id']) == 1 else '__' + name + '.ct'))
              for name in config['task_input_id']}
client = latti_inference.InferenceClient(str(args.task / 'client'))
client.load_full_context((args.keys / 'full_context.bin').read_bytes())


def memory_mib():
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
    return sum(int(row.split(',')[1]) for row in rows.splitlines() if int(row.split(',')[0]) == os.getpid())


def create():
    server = latti_inference.InferenceServer(str(args.task / 'server'), True, args.gpu)
    server.import_eval_context_file(str(args.keys / 'eval_context.bin'))
    server.load_model()
    return server


errors = []


def evaluate(server):
    output = config['task_output_id'][0]
    actual = client.decrypt(server.evaluate(inputs))[output].output
    error = float(np.max(np.abs(np.asarray(actual) - reference)))
    assert error < 0.2, error
    errors.append(error)


first, second = create(), create()
try:
    first.release_device()
except RuntimeError as error:
    assert 'RESOURCE_BUSY' in str(error), str(error)
else:
    raise AssertionError('A second loaded owner must prevent global pool release')
# Both loaded states survive rejection and can still perform real inference.
evaluate(first)
evaluate(second)
loaded_mib = memory_mib()
del second
gc.collect()
first.release_device()
released_mib = memory_mib()
assert loaded_mib - released_mib > 1024, (loaded_mib, released_mib)
first.release_device()  # idempotent
first.load_model()
evaluate(first)
first.release_device()
report = {'twoLoadedOwnersBusy': True, 'bothInferAfterBusy': True,
          'reloadUsesRetainedHost': True, 'maxAbsoluteErrors': errors,
          'loadedProcessGpuMiB': loaded_mib, 'releasedProcessGpuMiB': released_mib}
args.report.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
