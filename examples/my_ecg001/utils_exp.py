import os
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_csv_row(csv_path, fieldnames, row_dict):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


class EarlyStopping:
    def __init__(self, mode='max', patience=5, min_delta=0.0):
        if mode not in ['max', 'min']:
            raise ValueError("mode must be 'max' or 'min'")
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, current):
        if self.best is None:
            self.best = current
            return False

        improved = False
        if self.mode == 'max':
            improved = current > self.best + self.min_delta
        else:
            improved = current < self.best - self.min_delta

        if improved:
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.should_stop = True

        return self.should_stop


def save_checkpoint(state: dict, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f'[Saved] {path}')


def load_checkpoint(model, path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)
    print(f'[Loaded] {path}')
    return ckpt


def compute_class_weights(y_path: str, num_classes: int):
    y = np.load(y_path).astype(np.int64)
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)