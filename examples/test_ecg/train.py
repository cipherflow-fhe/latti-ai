import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

"""
ECG Model Training and Export Module.

This module implements the complete training pipeline for ECG classification models,
including baseline training and FHE-compatible polynomial model conversion. It supports
training with class-weighted cross-entropy loss, automatic best model checkpointing,
and export to ONNX and H5 formats for FHE deployment. The polynomial activation replacement
enables privacy-preserving inference using Fully Homomorphic Encryption while maintaining
classification accuracy.
"""

#  Add project root directory to sys.path
FILE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(FILE_ROOT) not in sys.path:
    sys.path.insert(0, str(FILE_ROOT))

from dataset import ECGNpyDataset
from model import build_model

from training.nn_tools import (
    replace_activation_with_poly,
    replace_maxpool_with_avgpool,
    export_to_onnx,
    fuse_and_export_h5,
)
from training.nn_tools.activations import RangeNormPoly2d, Simple_Polyrelu


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_poly_module(name: str):
    if name == 'RangeNormPoly2d':
        return RangeNormPoly2d
    if name == 'Simple_Polyrelu':
        return Simple_Polyrelu
    raise ValueError(f'不支持的 poly module: {name}')


def compute_class_weights(y_train_path: str, num_classes: int):
    y = np.load(y_train_path).astype(np.int64)
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    pbar = tqdm(loader, desc='Train', leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_num += x.size(0)

        pbar.set_postfix(loss=f'{loss.item():.4f}')

    return total_loss / total_num, total_correct / total_num


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    pbar = tqdm(loader, desc='Eval ', leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * x.size(0)
        total_correct += (preds == y).sum().item()
        total_num += x.size(0)

    return total_loss / total_num, total_correct / total_num


def save_checkpoint(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f'[Saved] {path}')


def load_checkpoint(model, path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    print(f'[Loaded] {path}')
    return ckpt


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--processed-dir', type=str, default='./test_ecg/processed_over_1to1')
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--model-name', type=str, default='two_conv',
                        choices=['tiny_cnn', 'tiny_cnn8', 'two_conv', 'mlp_head'])

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--torch-num-threads', type=int, default=1)

    parser.add_argument('--output-dir', type=str, default='./test_ecg/runs/exp_over009/model')
    parser.add_argument('--input-dir', type=str, default='./test_ecg/runs/exp_over009/model')
    parser.add_argument('--export-dir', type=str, default='./test_ecg/runs/exp_over009/task/server')
    parser.add_argument('--pretrained', type=str, default='')

    parser.add_argument('--poly_model_convert', action='store_true')
    parser.add_argument('--degree', type=int, default=4)
    parser.add_argument('--upper-bound', type=float, default=3.0)
    parser.add_argument(
        '--poly-module',
        type=str,
        default='RangeNormPoly2d',
        choices=['RangeNormPoly2d', 'Simple_Polyrelu']
    )

    parser.add_argument('--input-shape', type=int, nargs=3, default=[1, 16, 16])

    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(args.torch_num_threads)

    output_dir = Path(args.output_dir)
    input_dir = Path(args.input_dir)
    export_dir = Path(args.export_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cpu')
    print(f'[Device] {device}')
    print(f'[Model] 使用模型：{args.model_name}')

    train_set = ECGNpyDataset(
        x_path=os.path.join(args.processed_dir, 'X_train.npy'),
        y_path=os.path.join(args.processed_dir, 'y_train.npy')
    )
    val_set = ECGNpyDataset(
        x_path=os.path.join(args.processed_dir, 'X_val.npy'),
        y_path=os.path.join(args.processed_dir, 'y_val.npy')
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )

    model = build_model(num_classes=args.num_classes, model_name=args.model_name)

    if args.pretrained:
        load_checkpoint(model, args.pretrained, device)

    if args.poly_model_convert:
        poly_cls = get_poly_module(args.poly_module)

        print('[Info] replace maxpool -> avgpool')
        replace_maxpool_with_avgpool(model)

        print('[Info] replace relu -> poly activation')
        replace_activation_with_poly(
            model,
            old_cls=nn.ReLU,
            new_module_factory=poly_cls,
            upper_bound=args.upper_bound,
            degree=args.degree
        )

    model = model.to(device)

    class_weights = compute_class_weights(
        y_train_path=os.path.join(args.processed_dir, 'y_train.npy'),
        num_classes=args.num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = -1.0
    best_state = None
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f'\n===== Epoch {epoch}/{args.epochs} =====')
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f'Epoch {epoch}:')
        print(f'  train_loss={train_loss:.6f}, train_acc={train_acc:.6f}')
        print(f'  val_loss  ={val_loss:.6f}, val_acc  ={val_acc:.6f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_val_acc': best_val_acc,
                'args': vars(args)
            }

    elapsed = time.time() - start_time
    print(f'\nTraining finished in {elapsed / 60:.2f} min, best_val_acc={best_val_acc:.6f}')

    if args.poly_model_convert:
        best_ckpt_path = output_dir / 'train_poly.pth'
    else:
        best_ckpt_path = output_dir / 'train_baseline.pth'

    save_checkpoint(best_state, str(best_ckpt_path))

    model.load_state_dict(best_state['model'])
    model.eval()

    if args.poly_model_convert:
        onnx_path = output_dir / 'trained_poly.onnx'
        h5_path = export_dir / 'model_parameters.h5'
        c, h, w = args.input_shape
        input_size = (1, c, h, w)

        print(f'[Export ONNX] {onnx_path}')
        export_to_onnx(
            model,
            save_path=str(onnx_path),
            input_size=input_size,
            opset_version=13,
            dynamic_batch=False,
            remove_identity=True,
            save_h5=False,
            verbose=True
        )

        print(f'[Export H5] {h5_path}')
        fuse_and_export_h5(
            model,
            h5_path=str(h5_path),
            upper_bound=args.upper_bound,
            degree=args.degree,
            eps=1e-3,
            verbose=True
        )

        print('\n[Done] Poly model export finished.')
        print(f'  PTH : {best_ckpt_path}')
        print(f'  ONNX: {onnx_path}')
        print(f'  H5  : {h5_path}')


if __name__ == '__main__':
    main()
