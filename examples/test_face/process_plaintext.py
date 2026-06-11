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
import argparse
import sys
from pathlib import Path

TEST_FACE_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_FACE_DIR.parents[1]
sys.path.insert(0, str(TEST_FACE_DIR))
sys.path.insert(0, str(REPO_ROOT))

# ruff: noqa: E402
# isort: off
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model.facenet_mobilenetv2 import FaceNetMobileNetV2
from training.nn_tools import replace_activation_with_poly, replace_maxpool_with_avgpool
from training.nn_tools.activations import PolyAct, RangeNormPoly2d
from utils.dataloader import parse_annotation_lines
# isort: on


DEFAULT_CHECKPOINT = TEST_FACE_DIR / 'output_poly' / 'last.pth'
DEFAULT_INPUT_SHAPE = (3, 256, 256)
DEFAULT_INPUT_CSV_PATH = TEST_FACE_DIR / 'task' / 'client' / 'img.csv'
DEFAULT_QUERY_EMBEDDING_PATH = TEST_FACE_DIR / 'query_embedding.csv'
DEFAULT_GALLERY_EMBEDDING_PATH = TEST_FACE_DIR / 'gallery_embedding.csv'
DEFAULT_EMBEDDING_A_PATH = DEFAULT_QUERY_EMBEDDING_PATH
DEFAULT_EMBEDDING_B_PATH = DEFAULT_GALLERY_EMBEDDING_PATH


def resize_image(image, size, letterbox_image=True):
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image
    return image.resize((w, h), Image.BICUBIC)


def load_checkpoint_state_dict(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get(
        'state_dict', checkpoint.get('net', checkpoint))
    return {key.replace('module.', ''): value for key, value in state_dict.items()}


def infer_num_classes(state_dict):
    for key, value in state_dict.items():
        if key.endswith('classifier.weight'):
            return value.shape[0]
    raise ValueError(
        'Cannot infer num_classes from checkpoint classifier.weight')


def build_model(state_dict, poly_model_convert=True, poly_module='RangeNormPoly2d', upper_bound=3.0, degree=4):
    model = FaceNetMobileNetV2(num_classes=infer_num_classes(
        state_dict), normalize_embedding=False)
    if poly_model_convert:
        replace_maxpool_with_avgpool(model)
        poly_cls = RangeNormPoly2d if poly_module == 'RangeNormPoly2d' else PolyAct
        replace_activation_with_poly(
            model,
            old_cls=nn.ReLU,
            new_module_factory=poly_cls,
            upper_bound=upper_bound,
            degree=degree,
        )
        initialize_poly_buffers(model, DEFAULT_INPUT_SHAPE)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'Missing keys: {len(missing)}')
    if unexpected:
        print(f'Unexpected keys: {len(unexpected)}')
    return model.eval()


def initialize_poly_buffers(model, input_shape):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *input_shape), mode='predict')
    model.train(was_training)


def read_image(path, input_shape=DEFAULT_INPUT_SHAPE):
    c, h, w = input_shape
    if c != 3:
        raise ValueError(f'Only 3-channel image input is supported, got {c}')
    image = Image.open(path).convert('RGB')
    try:
        image = resize_image(image, [w, h], letterbox_image=True)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        return transform(image).unsqueeze(0)
    finally:
        image.close()


def read_csv(path, input_shape=DEFAULT_INPUT_SHAPE):
    with open(path, 'r') as file:
        values = [float(item) for item in file.read().replace(
            '\n', ',').split(',') if item.strip()]
    data = np.asarray(values, dtype=np.float32).reshape(input_shape)
    return torch.from_numpy(data).float().unsqueeze(0)


def read_input(path, input_shape=DEFAULT_INPUT_SHAPE):
    suffix = Path(path).suffix.lower()
    if suffix == '.csv':
        return read_csv(path, input_shape)
    return read_image(path, input_shape)


def write_input_csv(path, input_tensor):
    values = input_tensor.squeeze(0).detach().cpu().numpy().reshape(-1)
    with open(path, 'w') as file:
        file.write(','.join(str(float(value)) for value in values))
        file.write('\n')


def export_input_csv(input_path, output_path=DEFAULT_INPUT_CSV_PATH, input_shape=DEFAULT_INPUT_SHAPE):
    input_tensor = read_input(input_path, input_shape)
    write_input_csv(output_path, input_tensor)


class ImagePathDataset(Dataset):
    def __init__(self, image_paths, input_shape=DEFAULT_INPUT_SHAPE):
        self.image_paths = list(image_paths)
        self.input_shape = input_shape

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        return read_image(image_path, self.input_shape).squeeze(0), image_path


@torch.no_grad()
def compute_embedding(model, input_tensor, device):
    return model(input_tensor.to(device), mode='predict')


def write_embedding(path, embedding):
    values = embedding.detach().cpu().numpy().reshape(-1)
    with open(path, 'w') as file:
        file.write(','.join(str(float(value)) for value in values))
        file.write('\n')


def compute_single_embedding(model, path, device, input_shape=DEFAULT_INPUT_SHAPE,
                             embedding_output=DEFAULT_EMBEDDING_A_PATH):
    input_tensor = read_input(path, input_shape)
    embedding = compute_embedding(model, input_tensor, device)
    write_embedding(embedding_output, embedding)
    norm2 = torch.sum(embedding * embedding, dim=1).cpu().numpy()[0]
    return float(norm2)


def compute_distance(model, path_a, path_b, device, input_shape=DEFAULT_INPUT_SHAPE,
                     embedding_output_a=DEFAULT_EMBEDDING_A_PATH,
                     embedding_output_b=DEFAULT_EMBEDDING_B_PATH):
    input_a = read_input(path_a, input_shape)
    input_b = read_input(path_b, input_shape)
    embedding_a = compute_embedding(model, input_a, device)
    embedding_b = compute_embedding(model, input_b, device)
    write_embedding(embedding_output_a, embedding_a)
    write_embedding(embedding_output_b, embedding_b)
    embedding_a_normed = F.normalize(embedding_a, p=2, dim=1)
    embedding_b_normed = F.normalize(embedding_b, p=2, dim=1)
    return torch.sqrt(torch.sum((embedding_a_normed - embedding_b_normed) ** 2, dim=1)).cpu().numpy()[0]


@torch.no_grad()
def stat_embedding_norm2_min_max(annotation_path, checkpoint=DEFAULT_CHECKPOINT, gpu=-1, batch_size=64,
                                 num_workers=4, poly_model_convert=True, margin_ratio=0.0,
                                 input_shape=DEFAULT_INPUT_SHAPE):
    device = torch.device(
        f'cuda:{gpu}') if gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
    state_dict = load_checkpoint_state_dict(checkpoint)
    model = build_model(state_dict, poly_model_convert=poly_model_convert).to(device)

    with open(annotation_path, 'r') as file:
        image_paths, _, _ = parse_annotation_lines(file.readlines())
    image_paths = sorted(set(str(path) for path in image_paths))

    loader = DataLoader(
        ImagePathDataset(image_paths, input_shape),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    norm2_values = []
    norm2_paths = []
    for images, paths in tqdm(loader, desc='Stat embedding norm2'):
        embeddings = compute_embedding(model, images, device)
        norm2 = torch.sum(embeddings * embeddings, dim=1)
        norm2_values.append(norm2.cpu().numpy())
        norm2_paths.extend(paths)

    norm2_values = np.concatenate(norm2_values)
    norm2_paths = np.asarray(norm2_paths, dtype=object)
    finite_mask = np.isfinite(norm2_values)
    finite_norm2 = norm2_values[finite_mask]
    finite_paths = norm2_paths[finite_mask]
    nonfinite_count = int(norm2_values.size - finite_norm2.size)
    if finite_norm2.size == 0:
        raise ValueError('No finite embedding norm2 values found')

    raw_min_idx = int(np.argmin(finite_norm2))
    raw_max_idx = int(np.argmax(finite_norm2))
    raw_norm2_min = float(finite_norm2[raw_min_idx])
    raw_norm2_max = float(finite_norm2[raw_max_idx])
    min_path = finite_paths[raw_min_idx]
    max_path = finite_paths[raw_max_idx]

    p01, p50, p95, p99, p999 = np.percentile(finite_norm2, [0.1, 50, 95, 99, 99.9])
    norm2_min = float(p01)
    norm2_max = float(p999)

    if margin_ratio > 0:
        norm2_min *= max(0.0, 1.0 - margin_ratio)
        norm2_max *= 1.0 + margin_ratio

    print(f'nonfinite_norm2_count={nonfinite_count}')
    print(f'raw_norm2_min={raw_norm2_min} path={min_path}')
    print(f'raw_norm2_p001={p01}')
    print(f'raw_norm2_p50={p50}')
    print(f'raw_norm2_p95={p95}')
    print(f'raw_norm2_p99={p99}')
    print(f'raw_norm2_p999={p999}')
    print(f'raw_norm2_max={raw_norm2_max} path={max_path}')
    return norm2_min, norm2_max


def run_single_plaintext(arg, checkpoint=DEFAULT_CHECKPOINT, gpu=-1, poly_model_convert=True,
                         embedding_output=DEFAULT_EMBEDDING_A_PATH):
    device = torch.device(
        f'cuda:{gpu}') if gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
    state_dict = load_checkpoint_state_dict(checkpoint)
    model = build_model(
        state_dict, poly_model_convert=poly_model_convert).to(device)
    return compute_single_embedding(
        model,
        arg,
        device,
        embedding_output=embedding_output,
    )


def run_plaintext(arg1, arg2, checkpoint=DEFAULT_CHECKPOINT, gpu=-1, poly_model_convert=True,
                  embedding_output_a=DEFAULT_EMBEDDING_A_PATH,
                  embedding_output_b=DEFAULT_EMBEDDING_B_PATH):
    device = torch.device(
        f'cuda:{gpu}') if gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
    state_dict = load_checkpoint_state_dict(checkpoint)
    model = build_model(
        state_dict, poly_model_convert=poly_model_convert).to(device)
    return compute_distance(
        model,
        arg1,
        arg2,
        device,
        embedding_output_a=embedding_output_a,
        embedding_output_b=embedding_output_b,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run plaintext face preprocessing, embedding inference, or distance.')
    parser.add_argument('input_a', nargs='?')
    parser.add_argument('input_b', nargs='?')
    parser.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT))
    parser.add_argument('--gpu', type=int, default=-1, help='-1 for CPU')
    parser.add_argument('--export-input-csv')
    parser.add_argument('--inference-output')
    parser.add_argument('--query-output')
    parser.add_argument('--gallery-output')
    parser.add_argument('--embedding-output-a', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--embedding-output-b', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--no-poly-model-convert', action='store_true')
    parser.add_argument('--stat-norm2-min-max', action='store_true')
    parser.add_argument('--annotation-path')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--margin-ratio', type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stat_norm2_min_max:
        if args.annotation_path is None:
            raise ValueError('--annotation-path is required with --stat-norm2-min-max')
        norm2_min, norm2_max = stat_embedding_norm2_min_max(
            args.annotation_path,
            checkpoint=args.checkpoint,
            gpu=args.gpu,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            poly_model_convert=not args.no_poly_model_convert,
            margin_ratio=args.margin_ratio,
        )
        print(f'norm2_min={norm2_min}')
        print(f'norm2_max={norm2_max}')
        return

    if args.input_a is None:
        raise ValueError('input_a is required unless --stat-norm2-min-max is set')
    if args.export_input_csv is not None:
        export_input_csv(args.input_a, args.export_input_csv)
        print(f'input_csv={args.export_input_csv}')
        return

    if args.input_b is None:
        inference_output = args.inference_output or args.embedding_output_a or str(DEFAULT_QUERY_EMBEDDING_PATH)
        norm2 = run_single_plaintext(
            args.input_a,
            checkpoint=args.checkpoint,
            gpu=args.gpu,
            poly_model_convert=not args.no_poly_model_convert,
            embedding_output=inference_output,
        )
        print(f'inference_output={inference_output}')
        print(f'norm2={norm2}')
        return

    query_output = args.query_output or args.embedding_output_a or str(DEFAULT_QUERY_EMBEDDING_PATH)
    gallery_output = args.gallery_output or args.embedding_output_b or str(DEFAULT_GALLERY_EMBEDDING_PATH)
    distance = run_plaintext(
        args.input_a,
        args.input_b,
        checkpoint=args.checkpoint,
        gpu=args.gpu,
        poly_model_convert=not args.no_poly_model_convert,
        embedding_output_a=query_output,
        embedding_output_b=gallery_output,
    )
    print(distance)


if __name__ == '__main__':
    main()
