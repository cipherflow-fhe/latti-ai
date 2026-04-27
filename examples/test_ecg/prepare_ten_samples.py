import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
"""
Balanced Batch Sample Preparation Module.

This module selects a balanced batch of ECG samples (default: 5 normal + 5 abnormal)
from the validation or test set, exports them as individual CSV files for batch FHE inference,
and generates baseline plaintext predictions for all samples. It produces a sample list
(sample_list.json), plaintext prediction results (plaintext_predictions.json), and individual
sample tensors for subsequent encrypted inference verification and consistency analysis.
"""

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test_ecg.model import build_model

CLASS_NAMES = ['normal', 'abnormal']


def softmax_np(x: np.ndarray):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def load_baseline_model(ckpt_path: str, num_classes: int = 2, model_name: str = 'two_conv'):
    device = torch.device('cpu')
    model = build_model(num_classes=num_classes, model_name=model_name)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)

    model.eval()
    return model


@torch.no_grad()
def predict_one(model, x: np.ndarray):
    inp = torch.from_numpy(x).unsqueeze(0)   # [1,1,16,16]
    logits = model(inp).cpu().numpy()[0]
    probs = softmax_np(logits)
    pred = int(np.argmax(logits))
    return logits, probs, pred


def select_balanced_indices(y: np.ndarray, normal_count: int, abnormal_count: int):
    idx_normal = np.where(y == 0)[0]
    idx_abnormal = np.where(y == 1)[0]

    if len(idx_normal) < normal_count:
        raise ValueError(f'normal 样本不够，需要 {normal_count}，实际 {len(idx_normal)}')
    if len(idx_abnormal) < abnormal_count:
        raise ValueError(f'abnormal 样本不够，需要 {abnormal_count}，实际 {len(idx_abnormal)}')

    selected = list(idx_normal[:normal_count]) + list(idx_abnormal[:abnormal_count])
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-dir', type=str, default='./test_ecg/processed_over_1to1')
    parser.add_argument('--baseline-ckpt', type=str, default='./test_ecg/runs/exp_over009/model/train_baseline.pth')
    parser.add_argument('--task-dir', type=str, default='./test_ecg/runs/exp_over009/task')
    parser.add_argument('--model-name', type=str, default='two_conv')
    parser.add_argument('--dataset-split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--normal-count', type=int, default=5)
    parser.add_argument('--abnormal-count', type=int, default=5)
    parser.add_argument('--output-subdir', type=str, default='client_batch10')
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    task_dir = Path(args.task_dir)
    client_dir = task_dir / args.output_subdir
    inputs_dir = client_dir / 'inputs'
    inputs_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_split == 'val':
        X = np.load(processed_dir / 'X_val.npy').astype(np.float32)
        y = np.load(processed_dir / 'y_val.npy').astype(np.int64)
    else:
        X = np.load(processed_dir / 'X_test.npy').astype(np.float32)
        y = np.load(processed_dir / 'y_test.npy').astype(np.int64)

    indices = select_balanced_indices(y, args.normal_count, args.abnormal_count)

    model = load_baseline_model(
        ckpt_path=args.baseline_ckpt,
        num_classes=2,
        model_name=args.model_name
    )

    sample_items = []
    plaintext_items = []

    for local_id, global_idx in enumerate(indices):
        x = X[global_idx]
        label = int(y[global_idx])

        if x.shape != (1, 16, 16):
            raise ValueError(f'样本形状必须是 (1,16,16)，当前是 {x.shape}，global_idx={global_idx}')

        csv_name = f'sample_{local_id:02d}_idx_{global_idx}_label_{label}.csv'
        csv_path = inputs_dir / csv_name

        np.savetxt(csv_path, x[0], delimiter=',', fmt='%.10f')

        logits, probs, pred = predict_one(model, x)

        sample_meta = {
            'local_id': int(local_id),
            'global_index': int(global_idx),
            'true_label_id': int(label),
            'true_label_name': CLASS_NAMES[label],
            'csv_path': str(csv_path),
        }

        plain_pred = {
            'local_id': int(local_id),
            'global_index': int(global_idx),
            'true_label_id': int(label),
            'true_label_name': CLASS_NAMES[label],
            'baseline_pred_id': int(pred),
            'baseline_pred_name': CLASS_NAMES[pred],
            'logits': logits.tolist(),
            'probabilities': probs.tolist(),
            'csv_path': str(csv_path),
        }

        sample_items.append(sample_meta)
        plaintext_items.append(plain_pred)

        np.save(client_dir / f'sample_{local_id:02d}_tensor.npy', x)

    with open(client_dir / 'sample_list.json', 'w', encoding='utf-8') as f:
        json.dump(sample_items, f, ensure_ascii=False, indent=2)

    with open(client_dir / 'plaintext_predictions.json', 'w', encoding='utf-8') as f:
        json.dump(plaintext_items, f, ensure_ascii=False, indent=2)

    summary = {
        'dataset_split': args.dataset_split,
        'normal_count': args.normal_count,
        'abnormal_count': args.abnormal_count,
        'total_samples': len(indices),
        'processed_dir': str(processed_dir),
        'baseline_ckpt': args.baseline_ckpt,
        'task_dir': str(task_dir),
        'output_dir': str(client_dir),
        'model_name': args.model_name,
    }

    with open(client_dir / 'prepare_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('========== Plaintext Batch Ready ==========')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'sample_list      : {client_dir / "sample_list.json"}')
    print(f'plaintext_preds  : {client_dir / "plaintext_predictions.json"}')
    print(f'inputs_dir       : {inputs_dir}')


if __name__ == '__main__':
    main()
