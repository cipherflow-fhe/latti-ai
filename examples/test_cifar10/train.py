# Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Train ResNet20 on CIFAR-10 (baseline or poly-ReLU).

Baseline (standard ReLU):
  python train.py --epochs 200

Poly-ReLU (replace ReLU with RangeNormPoly2d, fine-tune, export):
  python train.py --poly_model_convert --pretrained train_baseline.pth --epochs 10

The --poly_model_convert flag enables ReLU -> RangeNormPoly2d replacement and
triggers ONNX + fused-H5 export after training.
"""

import logging
import sys
import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from model import resnet20

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


def get_cifar10_loaders(data_dir, batch_size, num_workers=2):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)
    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)
    return running_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description='Train ResNet20 on CIFAR-10')
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--pretrained', default=None, help='path to .pth checkpoint')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--input-dir', default='./input')
    parser.add_argument('--output-dir', default='./output')
    parser.add_argument('--export-dir', default=None, help='directory for fused H5 (default: same as --output-dir)')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=0, help='-1 for CPU')
    parser.add_argument('--lr-milestones', type=int, nargs='+', default=[100, 150])
    parser.add_argument('--lr-gamma', type=float, default=0.1)

    # Poly-ReLU options
    parser.add_argument('--poly_model_convert', action='store_true', help='replace ReLU with RangeNormPoly2d')
    parser.add_argument('--upper-bound', type=float, default=3.0, help='normalization upper bound for RangeNormPoly2d')
    parser.add_argument('--degree', type=int, default=4, choices=[2, 4, 8])
    parser.add_argument(
        '--input-shape',
        type=int,
        nargs='+',
        default=None,
        help='C H W input shape for ONNX export, e.g. --input-shape 3 32 32',
    )
    args = parser.parse_args()

    if args.poly_model_convert and args.input_shape is None:
        parser.error('--input-shape is required when --poly_model_convert is enabled, e.g. --input-shape 3 32 32')

    if args.poly_model_convert:
        args.output_dir = args.input_dir
    device = torch.device(f'cuda:{args.gpu}') if args.gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    export_dir = args.export_dir or args.output_dir
    os.makedirs(export_dir, exist_ok=True)

    train_loader, test_loader = get_cifar10_loaders(args.data_dir, args.batch_size, args.num_workers)

    # Build model
    model = resnet20()
    if args.pretrained:
        log.info(f'Loading pretrained: {args.pretrained}')
        ckpt = torch.load(args.pretrained, map_location='cpu')
        sd = ckpt.get('state_dict', ckpt)
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)

    # Optionally replace ReLU -> RangeNormPoly2d
    if args.poly_model_convert:
        from training.nn_tools import (
            replace_activation_with_poly,
            export_to_onnx,
            fuse_and_export_h5,
            replace_maxpool_with_avgpool,
        )
        from training.nn_tools.replace import count_activations
        from training.nn_tools.activations import RangeNormPoly2d

        n_maxpool = count_activations(model, nn.MaxPool2d)
        replace_maxpool_with_avgpool(model)
        n_avgpool = count_activations(model, nn.AvgPool2d)
        log.info(f'Device: {device}  |  MaxPool2d {n_maxpool} -> AvgPool2d {n_avgpool} ')
        n_relu = count_activations(model, nn.ReLU)
        replace_activation_with_poly(model, old_cls=nn.ReLU, upper_bound=args.upper_bound, degree=args.degree)
        n_poly = count_activations(model, RangeNormPoly2d)
        log.info(f'Device: {device}  |  ReLU {n_relu} -> Poly {n_poly} (ub={args.upper_bound}, deg={args.degree})')
    else:
        n_params = sum(p.numel() for p in model.parameters())
        log.info(f'Device: {device}  |  Params: {n_params:,}')

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)

    tag = 'train_poly' if args.poly_model_convert else 'train_baseline'
    best_acc = 0.0
    best_path = os.path.join(args.output_dir, f'{tag}.pth')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        mark = '*' if test_acc > best_acc else ' '
        log.info(
            f'[{epoch:3d}/{args.epochs}] '
            f'lr={optimizer.param_groups[0]["lr"]:.4f}  '
            f'train {train_loss:.4f}/{train_acc:.2f}%  '
            f'test {test_loss:.4f}/{test_acc:.2f}% {mark}  '
            f'{time.time() - t0:.1f}s'
        )

        if test_acc > best_acc:
            best_acc = test_acc
            save_dict = {'epoch': epoch, 'state_dict': model.state_dict(), 'best_acc': best_acc}
            if args.poly_model_convert:
                save_dict.update(upper_bound=args.upper_bound, degree=args.degree)
            torch.save(save_dict, best_path)

    log.info(f'Best accuracy: {best_acc:.2f}%  ->  {best_path}')

    # Export ONNX + fused H5 (poly mode only)
    if args.poly_model_convert:
        ckpt = torch.load(best_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        model.eval()

        onnx_path = os.path.join(args.output_dir, 'trained_poly.onnx')
        export_to_onnx(
            model,
            save_path=onnx_path,
            input_size=tuple([1, *args.input_shape]),
            dynamic_batch=False,
        )
        log.info(f'ONNX saved: {onnx_path}')

        h5_path = os.path.join(export_dir, 'model_parameters.h5')
        fuse_and_export_h5(model, h5_path=h5_path, upper_bound=args.upper_bound, degree=args.degree, eps=1e-3)
        log.info(f'Fused H5 saved: {h5_path}')


if __name__ == '__main__':
    main()
