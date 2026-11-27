"""Independent raw preprocessing/full-ONNX oracle and native ciphertext transport."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
import onnx
import onnxruntime as ort

repo = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(repo), str(repo / 'tools')]
if os.environ.get('LATTI_BATCH_ORACLE_DEPS'):
    sys.path.insert(0, os.environ['LATTI_BATCH_ORACLE_DEPS'])
from evaluate_onnx_csv import expand_polynomials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['inputs', 'prepare-runner', 'zip-results', 'compare'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--workflow', type=Path)
    args = parser.parse_args()
    root = args.root
    samples = json.loads((root / 'prepared/inputs.json').read_text())['samples']
    if args.operation == 'prepare-runner':
        incoming = root / 'native-inputs'
        incoming.mkdir(exist_ok=False)
        with zipfile.ZipFile(root / 'input.zip') as zipped:
            expected = {
                s['case_id'] + '__' + name + '.ct'
                for s in samples
                for name in json.loads((root / 'bundle/task_config.json').read_text())['task_input_param']
            }
            if set(zipped.namelist()) != expected or len(zipped.namelist()) != len(expected):
                raise ValueError('Input ciphertext case/feature set differs')
            zipped.extractall(incoming)
        names = json.loads((root / 'bundle/task_config.json').read_text())['task_input_param']
        manifest = dict(
            samples=[
                dict(case_id=s['case_id'], inputs={name: s['case_id'] + '__' + name + '.ct' for name in names})
                for s in samples
            ]
        )
        (incoming / 'batch.json').write_text(json.dumps(manifest, indent=2) + '\n')
    elif args.operation == 'zip-results':
        records = json.loads((root / 'ciphertexts/batch_manifest.json').read_text())
        if [s['case_id'] for s in records['samples']] != [s['case_id'] for s in samples]:
            raise ValueError('Runner case order differs')
        with zipfile.ZipFile(root / 'output.zip', 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as zipped:
            for record in records['samples']:
                zipped.write(root / 'ciphertexts' / record['file'], record['file'])
    elif args.operation == 'inputs':
        # xlrd 1.2 reads XLSX; Python >=3.9 removed its legacy XML alias.
        import xml.etree.ElementTree as ET

        if not hasattr(ET.ElementTree, 'getiterator'):
            ET.ElementTree.getiterator = ET.ElementTree.iter
        path = Path('/LattiAI-Task-Resources/retina_degree8_02/task/client/gen_csv/gen_client_inputs_seg.py')
        spec = importlib.util.spec_from_file_location('reference_preprocessing', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ids, non_inv, invasive = module.read_xlsx_5(str(args.raw / 'risk_factor_5.xlsx'))
        checks = []
        for sample in samples:
            case = sample['case_id']
            image = case + '.jpg'
            patient = module.get_id(image)
            index = ids[patient]
            image_values = module.preprocess_image_with_seg(
                str(args.raw / 'images' / image),
                str(args.raw / 'disk' / (case + '_disk.jpg')),
                str(args.raw / 'lesion' / (case + '_lesion.jpg')),
                512,
                True,
            )
            references = {
                'input_0': image_values.reshape(-1),
                'input_1': np.array(non_inv[index]).reshape(-1),
                'input_2': np.array(invasive[index]).reshape(-1),
            }
            report = {}
            for entry in sample['inputs']:
                actual = np.loadtxt(root / 'prepared' / entry['csv'], delimiter=',').reshape(-1)
                expected = references[entry['name']]
                if actual.size != expected.size or not np.isfinite(actual).all():
                    raise ValueError('Preprocessing shape/finite check failed')
                maximum = float(np.max(np.abs(actual - expected)))
                report[entry['name']] = maximum
                if maximum > 1e-6:
                    raise ValueError(f'{case}/{entry["name"]}: JS/Python preprocessing differs by {maximum}')
            checks.append(dict(case_id=case, max_abs_error=report))
        (root / 'preprocessing-check.json').write_text(
            json.dumps(dict(status='passed', samples=checks), indent=2) + '\n'
        )
    else:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 8
        model = expand_polynomials(onnx.load(args.model))
        session = ort.InferenceSession(
            model.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider']
        )
        names = [v.name for v in session.get_outputs()]
        if len(names) != 1:
            raise ValueError('Retina must have one full-model output')
        reports = []
        reference = {}
        for sample in samples:
            feeds = {
                entry['name']: np.loadtxt(root / 'prepared' / entry['csv'], delimiter=',', dtype=np.float32).reshape(
                    entry['shape']
                )
                for entry in sample['inputs']
            }
            expected = session.run(names, feeds)[0].reshape(-1)
            actual = np.loadtxt(root / 'results' / (sample['case_id'] + '.csv'), delimiter=',').reshape(-1)
            if expected.size != actual.size or not np.isfinite(actual).all():
                raise ValueError('Output coverage failed')
            error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
            reports.append(
                dict(
                    case_id=sample['case_id'],
                    max_abs_error=error,
                    pass_=error < 0.1,
                    reference=expected.tolist(),
                    actual=actual.tolist(),
                )
            )
            reference[sample['case_id']] = expected.tolist()
        maximum = max(r['max_abs_error'] for r in reports)
        result = dict(
            status='passed' if maximum < 0.1 else 'failed',
            max_abs_error=maximum,
            tolerance=0.1,
            count=len(reports),
            samples=reports,
        )
        if args.workflow:
            import csv
            from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

            labels = {
                row['filename'].rsplit('.', 1)[0]: int(row['label'])
                for row in csv.DictReader((args.raw / 'label.csv').open())
            }
            actual_labels = [labels[r['case_id']] for r in reports]
            predicted = [int(r['actual'][1] > r['actual'][0]) for r in reports]
            margins = [r['actual'][1] - r['actual'][0] for r in reports]
            tn, fp, fn, tp = confusion_matrix(actual_labels, predicted, labels=[0, 1]).ravel()
            independent = dict(
                accuracy=float(accuracy_score(actual_labels, predicted)),
                specificity=float(tn / (tn + fp)),
                sensitivity=float(tp / (tp + fn)),
                auc=float(roc_auc_score(actual_labels, margins)),
            )
            workflow = json.loads(args.workflow.read_text())
            for name, value in independent.items():
                if abs(workflow['statistics'][name] - value) > 1e-12:
                    raise ValueError('Postprocess metric differs: ' + name)
            result['statistics'] = independent
            result['statistics_oracle'] = 'sklearn.metrics'
        (root / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
        (root / 'reference.json').write_text(json.dumps(reference, indent=2) + '\n')
        if maximum >= 0.1:
            raise ValueError('Batch maximum absolute error must be below 0.1')
        print(json.dumps(result))


if __name__ == '__main__':
    main()
