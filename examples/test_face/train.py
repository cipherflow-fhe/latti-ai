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
"""Train MobileNetV2 FaceNet and optionally export an FHE-friendly model."""

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from model.facenet_mobilenetv2 import FaceEmbeddingExportWrapper, FaceNetMobileNetV2
from utils.dataloader import FacenetDataset, dataset_collate, get_num_classes
from utils.losses import combined_facenet_loss

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


DEFAULT_ANNOTATION_PATH = '/home/zhongy/facenet-pytorch-change/cls_train_2.txt'


def load_annotation_lines(annotation_path):
    with open(annotation_path, 'r') as f:
        return f.readlines()


def build_transforms(train=True):
    ops = []
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(ops)


def build_model(args, num_classes):
    return FaceNetMobileNetV2(
        num_classes=num_classes,
        embedding_size=args.embedding_size,
        width_mult=args.width_mult,
        dropout_keep_prob=args.dropout_keep_prob,
        use_embedding_bn=not args.no_embedding_bn,
        normalize_embedding=args.normalize_embedding,
    )


def build_dataloaders(args, annotation_lines, num_classes):
    lines = annotation_lines[:]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(lines)

    num_val = int(len(lines) * args.val_split)
    if args.val_split > 0 and len(lines) > 1:
        num_val = max(1, num_val)
    train_lines = lines[num_val:]
    val_lines = lines[:num_val]

    if len(train_lines) == 0:
        raise ValueError('No training samples after train/val split')

    triplet_batch_size = args.batch_size // 3
    train_dataset = FacenetDataset(
        args.input_shape,
        train_lines,
        num_classes,
        random=True,
        transform=build_transforms(train=True),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=triplet_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=dataset_collate,
    )

    val_loader = None
    if len(val_lines) > 0:
        try:
            val_dataset = FacenetDataset(
                args.input_shape,
                val_lines,
                num_classes,
                random=False,
                transform=build_transforms(train=False),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=triplet_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=dataset_collate,
            )
        except ValueError as exc:
            log.warning('Skipping validation loader: %s', exc)

    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, device, args, epoch):
    model.train()
    ce_loss_fn = nn.CrossEntropyLoss()
    triplet_batch_size = args.batch_size // 3
    total_loss = 0.0
    total_triplet = 0.0
    total_ce = 0.0
    total_acc = 0.0
    steps = 0

    pbar = tqdm(loader, desc=f'Train {epoch}/{args.epochs}', leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        embeddings, logits = model(images, mode='train')
        loss, tri_loss, ce_loss = combined_facenet_loss(
            embeddings,
            logits,
            labels,
            triplet_batch_size,
            ce_loss_fn,
            triplet_weight=args.triplet_loss_weight,
            ce_weight=args.ce_loss_weight,
            margin=args.triplet_margin,
        )
        loss.backward()
        optimizer.step()

        acc = torch.mean((torch.argmax(logits, dim=-1) == labels).float())
        total_loss += loss.item()
        total_triplet += tri_loss.item()
        total_ce += ce_loss.item()
        total_acc += acc.item()
        steps += 1
        pbar.set_postfix(
            loss=f'{total_loss / steps:.4f}',
            tri=f'{total_triplet / steps:.4f}',
            ce=f'{total_ce / steps:.4f}',
            acc=f'{total_acc / steps:.4f}',
        )

    if steps == 0:
        raise ValueError('Training loader produced zero batches; reduce --batch-size or disable drop_last')

    return {
        'loss': total_loss / steps,
        'triplet_loss': total_triplet / steps,
        'ce_loss': total_ce / steps,
        'accuracy': total_acc / steps,
    }


@torch.no_grad()
def evaluate(model, loader, device, args, epoch):
    if loader is None:
        return None

    model.eval()
    ce_loss_fn = nn.CrossEntropyLoss()
    triplet_batch_size = args.batch_size // 3
    total_loss = 0.0
    total_triplet = 0.0
    total_ce = 0.0
    total_acc = 0.0
    steps = 0

    pbar = tqdm(loader, desc=f'Val {epoch}/{args.epochs}', leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        embeddings, logits = model(images, mode='train')
        loss, tri_loss, ce_loss = combined_facenet_loss(
            embeddings,
            logits,
            labels,
            triplet_batch_size,
            ce_loss_fn,
            triplet_weight=args.triplet_loss_weight,
            ce_weight=args.ce_loss_weight,
            margin=args.triplet_margin,
        )
        acc = torch.mean((torch.argmax(logits, dim=-1) == labels).float())
        total_loss += loss.item()
        total_triplet += tri_loss.item()
        total_ce += ce_loss.item()
        total_acc += acc.item()
        steps += 1
        pbar.set_postfix(
            loss=f'{total_loss / steps:.4f}',
            tri=f'{total_triplet / steps:.4f}',
            ce=f'{total_ce / steps:.4f}',
            acc=f'{total_acc / steps:.4f}',
        )

    if steps == 0:
        return None

    return {
        'loss': total_loss / steps,
        'triplet_loss': total_triplet / steps,
        'ce_loss': total_ce / steps,
        'accuracy': total_acc / steps,
    }


def convert_to_poly_model(model, args):
    from training.nn_tools import replace_activation_with_poly, replace_maxpool_with_avgpool
    from training.nn_tools.activations import PolyAct, RangeNormPoly2d
    from training.nn_tools.replace import count_activations

    poly_cls = RangeNormPoly2d if args.poly_module == 'RangeNormPoly2d' else PolyAct

    n_maxpool = count_activations(model, nn.MaxPool2d)
    replace_maxpool_with_avgpool(model)
    n_relu6 = count_activations(model, nn.ReLU6)
    replace_activation_with_poly(
        model,
        old_cls=nn.ReLU6,
        new_module_factory=poly_cls,
        upper_bound=args.upper_bound,
        degree=args.degree,
    )
    log.info(
        'Poly convert: MaxPool2d=%d, ReLU6=%d, poly=%s, upper_bound=%s, degree=%s',
        n_maxpool,
        n_relu6,
        poly_cls.__name__,
        args.upper_bound,
        args.degree,
    )
    return model


def export_model(model, args):
    from training.nn_tools import export_to_onnx, fuse_and_export_h5, replace_general_avgpool_with_depthwise_conv

    input_size = tuple([1, *args.input_shape])
    model.eval()
    export_model = FaceEmbeddingExportWrapper(model)
    replace_general_avgpool_with_depthwise_conv(export_model, input_size=input_size)

    onnx_path = os.path.join(args.output_dir, 'trained_poly.onnx' if args.poly_model_convert else 'trained.onnx')
    export_to_onnx(export_model, save_path=onnx_path, input_size=input_size, dynamic_batch=False)

    export_dir = args.export_dir or args.output_dir
    os.makedirs(export_dir, exist_ok=True)
    h5_path = os.path.join(export_dir, 'model_parameters.h5')
    fuse_and_export_h5(export_model, h5_path=h5_path, upper_bound=args.upper_bound, degree=args.degree, eps=1e-3)
    log.info('Exported ONNX=%s H5=%s', onnx_path, h5_path)


def load_pretrained(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt.get('net', ckpt))
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    log.info('Loaded checkpoint: %s', checkpoint_path)
    if missing:
        log.info('Missing keys: %d', len(missing))
    if unexpected:
        log.info('Unexpected keys: %d', len(unexpected))


def save_checkpoint(model, optimizer, epoch, metrics, path):
    torch.save(
        {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'metrics': metrics,
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Train MobileNetV2 FaceNet')
    parser.add_argument('--annotation-path', default=DEFAULT_ANNOTATION_PATH)
    parser.add_argument('--num-classes', type=int, default=None)
    parser.add_argument('--pretrained', default=None, help='path to checkpoint')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=24, help='effective batch size; must be a multiple of 3')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0, help='-1 for CPU')
    parser.add_argument('--seed', type=int, default=10101)
    parser.add_argument('--output-dir', default='./output')
    parser.add_argument('--export-dir', default=None)
    parser.add_argument('--input-shape', type=int, nargs=3, default=[3, 256, 256], metavar=('C', 'H', 'W'))

    parser.add_argument('--width-mult', type=float, default=1.0)
    parser.add_argument('--embedding-size', type=int, default=128)
    parser.add_argument('--dropout-keep-prob', type=float, default=0.5)
    parser.add_argument('--no-embedding-bn', action='store_true')
    parser.add_argument('--normalize-embedding', action='store_true')

    parser.add_argument('--triplet-margin', type=float, default=0.2)
    parser.add_argument('--triplet-loss-weight', type=float, default=1.0)
    parser.add_argument('--ce-loss-weight', type=float, default=1.0)

    parser.add_argument('--poly_model_convert', action='store_true', help='replace MobileNetV2 ReLU6 with polynomial activation')
    parser.add_argument('--poly-module', default='RangeNormPoly2d', choices=['RangeNormPoly2d', 'PolyAct'])
    parser.add_argument('--upper-bound', type=float, default=3.0)
    parser.add_argument('--degree', type=int, default=4, choices=[2, 4, 8])
    parser.add_argument('--export', action='store_true', help='export ONNX/H5 after training')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}') if args.gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    if args.export_dir:
        os.makedirs(args.export_dir, exist_ok=True)

    if args.batch_size % 3 != 0:
        raise ValueError('batch-size must be a multiple of 3 for Facenet triplet training')

    log.info('annotation_path=%s', args.annotation_path)
    log.info('input_shape=%s device=%s', args.input_shape, device)

    annotation_lines = load_annotation_lines(args.annotation_path)
    num_classes = args.num_classes or get_num_classes(args.annotation_path)
    model = build_model(args, num_classes)
    if args.pretrained:
        load_pretrained(model, args.pretrained)
    if args.poly_model_convert:
        model = convert_to_poly_model(model, args)
    model = model.to(device)

    train_loader, val_loader = build_dataloaders(args, annotation_lines, num_classes)
    optimizer = optim.Adam(model.parameters(), args.lr, betas=(args.momentum, 0.999), weight_decay=args.weight_decay)

    best_score = None
    best_path = os.path.join(args.output_dir, 'best.pth')
    last_path = os.path.join(args.output_dir, 'last.pth')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args, epoch)
        val_metrics = evaluate(model, val_loader, device, args, epoch)
        score = val_metrics['loss'] if val_metrics is not None else train_metrics['loss']

        save_checkpoint(model, optimizer, epoch, val_metrics or train_metrics, last_path)
        if best_score is None or score < best_score:
            best_score = score
            save_checkpoint(model, optimizer, epoch, val_metrics or train_metrics, best_path)
            mark = '*'
        else:
            mark = ' '

        val_text = 'no-val'
        if val_metrics is not None:
            val_text = f"val {val_metrics['loss']:.4f}/{val_metrics['accuracy']:.4f}"
        log.info(
            '[%3d/%d] train %.4f/%.4f  %s %s  %.1fs',
            epoch,
            args.epochs,
            train_metrics['loss'],
            train_metrics['accuracy'],
            val_text,
            mark,
            time.time() - t0,
        )

    log.info('Best checkpoint: %s', best_path)
    if args.export:
        load_pretrained(model, best_path)
        model = model.to(device)
        export_model(model, args)


if __name__ == '__main__':
    main()
