"""Prepare real CIFAR-10 images and validate the browser workflow with the GPU server."""

import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

mode, artifact_arg, repo_arg, keys_arg = sys.argv[1:]
artifacts, repo, keys = map(Path, (artifact_arg, repo_arg, keys_arg))
work = artifacts / 'cifar10'

if mode == 'prepare':
    from torchvision.datasets import CIFAR10

    dataset = CIFAR10(os.environ.get('LATTI_CIFAR10_DATA', str(repo.parent / 'data')), train=False, download=False)
    images = work / 'images'
    images.mkdir(parents=True)
    task = work / 'client'
    task.mkdir()
    task_source = Path(os.environ.get('LATTI_WORKFLOW_TASK', str(repo / 'examples/test_cifar10/task/client')))
    shutil.copyfile(task_source / 'task_config.json', task / 'task_config.json')
    secret = work / 'N16QP1546H192H32/secret'
    secret.mkdir(parents=True)
    shutil.copyfile(keys / 'secret/full_context.bin', secret / 'full_context.bin')
    (secret / 'full_context.bin').chmod(0o600)
    with (work / 'label.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['filename', 'label'])
        for index in range(3):
            image, label = dataset[index]
            name = f'cifar10_{index:05d}.png'
            image.save(images / name)
            writer.writerow([name, label])
    print('Prepared three CIFAR-10 test images and ground-truth labels', flush=True)
elif mode == 'evaluate':
    import latti_inference

    results = work / 'encrypted-results'
    results.mkdir()
    server = latti_inference.InferenceServer(str(repo / 'examples/test_cifar10/task/server'), True, 0)
    server.import_eval_context_file(str(keys / 'public/eval_context.bin'))
    server.load_model()
    reference = []
    for input_file in sorted((work / 'encrypted-inputs').glob('*.ct.bin')):
        start = time.monotonic()
        output = server.evaluate({'input': input_file.read_bytes()})
        elapsed = time.monotonic() - start
        (results / input_file.name).write_bytes(output['output'])
        plaintext = server.evaluate_plaintext(
            {'input': str(work / 'csv' / (input_file.name.removesuffix('.ct.bin') + '.csv'))}
        )['output']
        reference.append({'name': input_file.name, 'plaintext': plaintext, 'gpuEvaluateSeconds': elapsed})
    (work / 'native.json').write_text(json.dumps(reference, indent=2))
elif mode == 'check':
    import numpy as np

    reference = json.loads((work / 'native.json').read_text())
    labels = {row['filename']: int(row['label']) for row in csv.DictReader((work / 'label.csv').open())}
    report_rows = list(csv.DictReader((work / 'reports/predictions.csv').open()))
    checks = []
    correct = 0
    for index, sample in enumerate(reference):
        name = sample['name'].removesuffix('.ct.bin')
        values = np.loadtxt(work / 'logits' / (name + '.csv'), delimiter=',').reshape(-1)
        assert len(values) == 10 and np.isfinite(values).all()
        error = float(np.max(np.abs(values - sample['plaintext'])))
        assert error < 0.1, (name, error)
        prediction = int(np.argmax(values))
        filename = name + '.png'
        matched = int(prediction == labels[filename])
        assert report_rows[index] == {
            'filename': filename,
            'label': str(labels[filename]),
            'prediction': str(prediction),
            'correct': str(matched),
        }
        correct += matched
        checks.append({**sample, 'maxAbsoluteError': error, 'prediction': prediction, 'label': labels[filename]})
    assert len(reference) == len(report_rows) == len(labels) == 3
    (work / 'checked.json').write_text(
        json.dumps(
            {'samples': checks, 'total': len(labels), 'correct': correct, 'accuracy': correct / len(labels)}, indent=2
        )
    )
    print('GPU logits, argmax, predictions.csv and accuracy independently verified', flush=True)
else:
    raise ValueError(mode)
