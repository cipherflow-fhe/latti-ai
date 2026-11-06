"""Prepare Retina's three CSV inputs with the supplied offline task script.

Samples are independent JSONL records with case_id, image, disk, lesion, label.
Segmentation and clinical processing use the existing task tools unchanged.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'samples', 'xlsx', 'task-client', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    task = json.loads((args.task_client / 'task_config.json').read_text())
    features = task['task_input_param']
    if set(task['task_input_id']) != {'input_0', 'input_1', 'input_2'} or features['input_0']['channel'] != 5 or features['input_0']['shape'] != [512, 512] or features['input_1']['channel'] != 2 or features['input_2']['channel'] != 3:
        raise ValueError('Expected the current Retina 5-channel/2-factor/3-factor task')
    spec = importlib.util.spec_from_file_location('retina_task_preprocessing', args.reference)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    ids, non_inv, invasive = reference.read_xlsx_5(str(args.xlsx))
    args.output.mkdir(parents=True, exist_ok=False)
    for name in features:
        (args.output / name).mkdir()
    cases, seen = [], set()
    for line in args.samples.read_text().splitlines():
        sample = json.loads(line)
        case = sample['case_id']
        key = case.casefold()
        if not case or '/' in case or '\\' in case or '__' in case or case in ('.', '..') or key in seen or sample['label'] not in (0, 1):
            raise ValueError('Invalid or duplicate sample: ' + case)
        seen.add(key)
        paths = [Path(sample[k]) for k in ('image', 'disk', 'lesion')]
        patients = [reference.get_id(path.name) for path in paths]
        if len(set(patients)) != 1 or patients[0] not in ids:
            raise ValueError('Image/mask/clinical patient mismatch: ' + case)
        index = ids[patients[0]]
        image = reference.preprocess_image_with_seg(*(str(path) for path in paths), 512, True)
        values = {'input_0': image, 'input_1': np.asarray(non_inv[index]).reshape(1, -1), 'input_2': np.asarray(invasive[index]).reshape(1, -1)}
        for name, value in values.items():
            if not np.isfinite(value).all():
                raise ValueError('Non-finite feature: ' + case + '/' + name)
            np.savetxt(args.output / name / (case + '.csv'), value, delimiter=',', fmt='%.17g')
        cases.append({'case_id': case, 'label': sample['label'], 'inputs': {name: name + '/' + case + '.csv' for name in features}, 'sourceSHA256': {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in zip(('image', 'disk', 'lesion'), paths)}})
        print(case, flush=True)
    if not cases:
        raise ValueError('Empty samples')
    (args.output / 'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    (args.output / 'preprocessing.json').write_text(json.dumps({'referenceSHA256': hashlib.sha256(args.reference.read_bytes()).hexdigest(), 'xlsxSHA256': hashlib.sha256(args.xlsx.read_bytes()).hexdigest(), 'normalization': True, 'cases': len(cases)}, indent=2) + '\n')


if __name__ == '__main__':
    main()
