import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
"""
Plaintext Sample Selection and Baseline Prediction Module.

This module selects a single ECG sample from the validation set, exports it as a CSV file
for FHE inference testing, and generates baseline plaintext predictions using the trained model.
It produces metadata (sample_meta.json), baseline prediction results (baseline_plaintext.json),
and the sample tensor for subsequent encrypted inference verification and comparison.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-dir', type=str, default='./test_ecg/processed_over_1to1')
    parser.add_argument('--baseline-ckpt', type=str, default='./test_ecg/runs/exp_over009/model/train_baseline.pth')
    parser.add_argument('--task-dir', type=str, default='./test_ecg/runs/exp_over009/task')
    parser.add_argument('--model-name', type=str, default='two_conv')
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--output-name', type=str, default='ecg_input.csv')
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    task_dir = Path(args.task_dir)
    client_dir = task_dir / 'client'
    client_dir.mkdir(parents=True, exist_ok=True)

    X_val = np.load(processed_dir / 'X_val.npy').astype(np.float32)
    y_val = np.load(processed_dir / 'y_val.npy').astype(np.int64)

    idx = args.sample_index
    if idx < 0 or idx >= len(X_val):
        raise IndexError(f'sample-index 越界: {idx}, 合法范围 [0, {len(X_val)-1}]')

    x = X_val[idx]
    y = int(y_val[idx])

    if x.shape != (1, 16, 16):
        raise ValueError(f'样本形状必须是 (1,16,16)，当前是 {x.shape}')

    csv_path = client_dir / args.output_name
    np.savetxt(csv_path, x[0], delimiter=',', fmt='%.10f')

    model = load_baseline_model(args.baseline_ckpt, num_classes=2, model_name=args.model_name)
    logits, probs, pred = predict_one(model, x)

    meta = {
        'sample_index': int(idx),
        'true_label_id': y,
        'true_label_name': CLASS_NAMES[y],
        'baseline_pred_id': pred,
        'baseline_pred_name': CLASS_NAMES[pred],
        'csv_path': str(csv_path),
        'baseline_ckpt': args.baseline_ckpt,
    }

    result = {
        'logits': logits.tolist(),
        'probabilities': probs.tolist(),
        'argmax_class_id': pred,
        'argmax_class_name': CLASS_NAMES[pred],
    }

    with open(client_dir / 'sample_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(client_dir / 'baseline_plaintext.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    np.save(client_dir / 'sample_tensor.npy', x)

    print('========== Plaintext Sample Ready ==========')
    print(f'sample_index      : {idx}')
    print(f'true_label        : {y} ({CLASS_NAMES[y]})')
    print(f'baseline_pred     : {pred} ({CLASS_NAMES[pred]})')
    print(f'csv_path          : {csv_path}')
    print(f'baseline_result   : {client_dir / "baseline_plaintext.json"}')
    print(f'meta_json         : {client_dir / "sample_meta.json"}')
    print('logits            :', logits)
    print('probabilities     :', probs)


if __name__ == '__main__':
    main()
