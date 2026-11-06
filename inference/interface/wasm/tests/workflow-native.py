"""Independent native inference/ZIP/accuracy oracle for the browser runner."""

import ast
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

mode, artifact_arg, repo_arg = sys.argv[1:]
work, repo = Path(artifact_arg), Path(repo_arg)
task = Path(os.environ['LATTI_WORKFLOW_TASK'])
keys = Path(os.environ['LATTI_WORKFLOW_KEYS'])
model = os.environ.get('LATTI_WORKFLOW_MODEL', 'cifar10')
data = Path(os.environ.get('LATTI_WORKFLOW_DATA', str(work / 'data')))

if mode == 'prepare':
    if model == 'cifar10' and 'LATTI_WORKFLOW_DATA' not in os.environ:
        from torchvision import transforms
        from torchvision.datasets import CIFAR10

        dataset = CIFAR10(os.environ['LATTI_CIFAR10_DATA'], train=False, download=False)
        source = repo / 'examples/test_cifar10/train.py'
        loader = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'get_cifar10_loaders')
        scope = {'transforms': transforms, 'input_shape': [3, 32, 32]}
        for node in loader.body:
            if isinstance(node, ast.Assign) and node.targets[0].id in ('normalize', 'test_transform'):
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
        (data / 'images').mkdir(parents=True)
        (data / 'input').mkdir()
        cases = []
        for index in range(int(os.environ.get('LATTI_WORKFLOW_CASES', '3'))):
            image, label = dataset[index]
            name = f'cifar10_{index:05d}'
            image.save(data / 'images' / (name + '.png'))
            np.savetxt(data / 'input' / (name + '.csv'), scope['test_transform'](image).numpy().reshape(1, -1), delimiter=',', fmt='%.17g')
            cases.append({'case_id': name, 'label': label, 'image': 'images/' + name + '.png', 'inputs': {'input': 'input/' + name + '.csv'}})
        (data / 'cases.json').write_text(json.dumps(cases) + '\n')
    cases = json.loads((data / 'cases.json').read_text())
    config = json.loads((task / 'client/task_config.json').read_text())
    assert cases and len({c['case_id'] for c in cases}) == len(cases)
    for case in cases:
        assert set(case['inputs']) == set(config['task_input_id'])
        assert all((data / p).is_file() for p in case['inputs'].values())
        assert 0 <= case['label'] < (10 if model == 'cifar10' else 2)
    (work / 'cases.json').write_text(json.dumps(cases) + '\n')
elif mode == 'evaluate':
    import latti_inference

    cases = json.loads((work / 'cases.json').read_text())
    config = json.loads((task / 'client/task_config.json').read_text())
    inputs, output = config['task_input_id'], config['task_output_id'][0]
    server = latti_inference.InferenceServer(str(task / 'server'), True, int(os.environ.get('LATTI_GPU_DEVICE', '0')))
    server.import_eval_context_file(str(keys / 'eval_context.bin'))
    server.load_model()
    client = latti_inference.InferenceClient(str(task / 'client'))
    client.load_full_context((keys / 'full_context.bin').read_bytes())
    reference = []
    with zipfile.ZipFile(work / 'input.zip') as incoming, zipfile.ZipFile(work / 'output.zip', 'w', allowZip64=True) as outgoing:
        expected = [case['case_id'] + ('.ct' if len(inputs) == 1 else '__' + name + '.ct') for case in cases for name in inputs]
        assert sorted(incoming.namelist()) == sorted(expected) and len(incoming.namelist()) == len(set(expected))
        assert incoming.testzip() is None
        for case in cases:
            encrypted = {name: incoming.read(case['case_id'] + ('.ct' if len(inputs) == 1 else '__' + name + '.ct')) for name in inputs}
            start = time.monotonic()
            result = server.evaluate(encrypted)
            elapsed = time.monotonic() - start
            # Production bytes are passed unchanged; no scalar metadata patches.
            with outgoing.open(case['case_id'] + '.result.bin', 'w', force_zip64=True) as file:
                file.write(result[output])
            native = client.decrypt(result)[output].output
            plaintext = server.evaluate_plaintext({name: str(data / case['inputs'][name]) for name in inputs})[output]
            error = float(np.max(np.abs(np.asarray(native) - plaintext)))
            assert error < 0.2, (case['case_id'], error)
            reference.append({'case_id': case['case_id'], 'label': case['label'], 'nativeLogits': native, 'plaintext': plaintext, 'maxAbsoluteError': error, 'gpuEvaluateSeconds': elapsed})
            print(json.dumps(reference[-1]), flush=True)
    # The same host keys must survive device release and model reconstruction.
    server.release_device()
    server.load_model()
    first = cases[0]
    with zipfile.ZipFile(work / 'input.zip') as incoming:
        encrypted = {name: incoming.read(first['case_id'] + ('.ct' if len(inputs) == 1 else '__' + name + '.ct')) for name in inputs}
    repeated = client.decrypt(server.evaluate(encrypted))[output].output
    reload_error = float(np.max(np.abs(np.asarray(repeated) - reference[0]['plaintext'])))
    assert reload_error < 0.2, reload_error
    (work / 'reload.json').write_text(json.dumps({'hostKeysRetained': True, 'maxAbsoluteError': reload_error}) + '\n')
    server.release_device()
    (work / 'native.json').write_text(json.dumps(reference, indent=2) + '\n')
elif mode == 'check':
    reference = {r['case_id']: r for r in json.loads((work / 'native.json').read_text())}
    predictions = [json.loads(line) for line in (work / 'predictions.jsonl').read_text().splitlines()]
    assert set(reference) == {p['case_id'] for p in predictions} and len(predictions) == len(reference)
    correct = 0
    for record in predictions:
        expected = reference[record['case_id']]
        values = np.asarray(record['logits'])
        assert len(values) == (10 if model == 'cifar10' else 2) and np.isfinite(values).all()
        assert np.max(np.abs(values - expected['nativeLogits'])) < 1e-6
        prediction = int(np.argmax(values))
        assert record['prediction'] == prediction and record['label'] == expected['label']
        assert record['correct'] == (prediction == expected['label'])
        correct += prediction == expected['label']
    (work / 'checked.json').write_text(json.dumps({'total': len(predictions), 'correct': correct, 'accuracy': correct / len(predictions)}, indent=2) + '\n')
else:
    raise ValueError(mode)
