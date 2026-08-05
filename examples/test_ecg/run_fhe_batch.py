import os
import re
import json
import argparse
import subprocess
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report

"""
FHE Batch Inference and Evaluation Module.

This module executes Fully Homomorphic Encryption (FHE) inference on a batch of ECG samples,
parses encrypted and plaintext verification results from the FHE binary output, and performs
comprehensive evaluation comparing FHE predictions with baseline plaintext predictions. It calculates
classification metrics (accuracy, confusion matrix, precision, recall, F1-score) for both modes
and measures prediction consistency between encrypted and plaintext inference, providing detailed
analysis for validating FHE-based privacy-preserving ECG classification.
"""

CLASS_NAMES = ['normal', 'abnormal']

EXAMPLE_DIR = Path(__file__).resolve().parent

def parse_vector_from_verification_table(text: str, field_name: str):
    """
      Parse vector from Verification table.
      field_name: 'Encrypted' or 'Plaintext'
      Returns: list[float]
    """
    lines = text.splitlines()
    values = {}

    row_pattern = re.compile(
        r'^\s*(\d+)\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$'
    )

    in_table = False
    for line in lines:
        if 'Abs Error' in line and 'Encrypted' in line and 'Plaintext' in line:
            in_table = True
            continue

        if in_table:
            if line.strip().startswith('---'):
                continue
            if 'Max absolute error:' in line:
                break

            m = row_pattern.match(line)
            if m:
                idx = int(m.group(1))
                enc_val = float(m.group(2))
                plain_val = float(m.group(3))

                if field_name == 'Encrypted':
                    values[idx] = enc_val
                elif field_name == 'Plaintext':
                    values[idx] = plain_val
                else:
                    raise ValueError(f'Unknown field_name: {field_name}')

    if not values:
        raise ValueError(f'Unable to parse {field_name} vector from the Verification form')

    return [values[i] for i in sorted(values.keys())]


def calc_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    report = classification_report(
        y_true, y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True
    )

    return {
        'accuracy': float(acc),
        'confusion_matrix': cm,
        'per_class': {
            'normal': {
                'precision': float(precision[0]),
                'recall': float(recall[0]),
                'f1': float(f1[0]),
                'support': int(support[0]),
            },
            'abnormal': {
                'precision': float(precision[1]),
                'recall': float(recall[1]),
                'f1': float(f1[1]),
                'support': int(support[1]),
            }
        },
        'macro_f1': float(report['macro avg']['f1-score']),
        'weighted_f1': float(report['weighted avg']['f1-score']),
    }


def main():
    parser = argparse.ArgumentParser()
    default_task_dir = EXAMPLE_DIR / 'runs' / 'exp_over' / 'task'
    parser.add_argument('--task-dir', type=str, default=str(default_task_dir))
    parser.add_argument('--binary', type=str, default='./build/examples/inference')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--input-subdir', type=str, default='client_batch400')
    args = parser.parse_args()

    task_dir = Path(args.task_dir)
    binary = Path(args.binary)
    if not binary.exists():
        raise FileNotFoundError(f'Cannot find inference binary: {binary}')

    client_dir = task_dir / args.input_subdir
    inputs_dir = client_dir / 'inputs'

    sample_list_path = client_dir / 'sample_list.json'
    plaintext_path = client_dir / 'plaintext_predictions.json'

    if not sample_list_path.exists():
        raise FileNotFoundError(f'Cannot find sample list: {sample_list_path}')
    if not plaintext_path.exists():
        raise FileNotFoundError(f'Cannot find plaintext prediction file: {plaintext_path}')

    sample_list = json.loads(sample_list_path.read_text(encoding='utf-8'))
    plaintext_items = json.loads(plaintext_path.read_text(encoding='utf-8'))

    plain_map = {int(item['local_id']): item for item in plaintext_items}

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(args.threads)
    env['OPENBLAS_NUM_THREADS'] = str(args.threads)
    env['MKL_NUM_THREADS'] = str(args.threads)
    env['NUMEXPR_NUM_THREADS'] = str(args.threads)

    logs_dir = client_dir / 'fhe_logs'
    logs_dir.mkdir(parents=True, exist_ok=True)

    fhe_items = []

    y_true = []
    y_plain = []
    y_fhe = []

    for item in sample_list:
        local_id = int(item['local_id'])
        global_index = int(item['global_index'])
        true_label = int(item['true_label_id'])
        csv_path = Path(item['csv_path'])

        if not csv_path.exists():
            raise FileNotFoundError(f'Cannot find input CSV: {csv_path}')

        cmd = [
            str(binary),
            '--task-dir', str(task_dir),
            '--input', str(csv_path),
            '--verify'
        ]

        print(f'Running local_id={local_id:02d}, global_index={global_index}, file={csv_path.name}')
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env
        )

        output = proc.stdout
        (logs_dir / f'sample_{local_id:02d}.log').write_text(output, encoding='utf-8')

        if proc.returncode != 0:
            raise RuntimeError(f'FHE inference failed: local_id={local_id}, return_code={proc.returncode}')

        enc_logits = parse_vector_from_verification_table(output, 'Encrypted')
        plain_logits_from_verify = parse_vector_from_verification_table(output, 'Plaintext')

        fhe_pred = int(np.argmax(np.array(enc_logits)))
        plain_pred = int(plain_map[local_id]['baseline_pred_id'])
        plain_logits_saved = plain_map[local_id]['logits']

        fhe_item = {
            'local_id': local_id,
            'global_index': global_index,
            'true_label_id': true_label,
            'true_label_name': CLASS_NAMES[true_label],
            'plaintext_pred_id': plain_pred,
            'plaintext_pred_name': CLASS_NAMES[plain_pred],
            'fhe_pred_id': fhe_pred,
            'fhe_pred_name': CLASS_NAMES[fhe_pred],
            'encrypted_logits': enc_logits,
            'plaintext_logits_from_verify': plain_logits_from_verify,
            'plaintext_logits_saved': plain_logits_saved,
            'csv_path': str(csv_path),
            'log_path': str(logs_dir / f'sample_{local_id:02d}.log'),
        }
        fhe_items.append(fhe_item)

        y_true.append(true_label)
        y_plain.append(plain_pred)
        y_fhe.append(fhe_pred)

    plaintext_metrics = calc_metrics(y_true, y_plain)
    fhe_metrics = calc_metrics(y_true, y_fhe)

    consistency = float(np.mean(np.array(y_plain) == np.array(y_fhe)))

    summary = {
        'num_samples': len(y_true),
        'plaintext_metrics': plaintext_metrics,
        'fhe_metrics': fhe_metrics,
        'plaintext_fhe_prediction_consistency': consistency,
    }

    with open(client_dir / 'fhe_predictions.json', 'w', encoding='utf-8') as f:
        json.dump(fhe_items, f, ensure_ascii=False, indent=2)

    with open(client_dir / 'fhe_compare_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('\n========== Plaintext Metrics ==========')
    print(json.dumps(plaintext_metrics, ensure_ascii=False, indent=2))

    print('\n========== FHE Metrics ==========')
    print(json.dumps(fhe_metrics, ensure_ascii=False, indent=2))

    print('\n========== Consistency ==========')
    print(json.dumps({
        'plaintext_fhe_prediction_consistency': consistency
    }, ensure_ascii=False, indent=2))

    print('\nSave File:')
    print(client_dir / 'fhe_predictions.json')
    print(client_dir / 'fhe_compare_summary.json')


if __name__ == '__main__':
    main()
