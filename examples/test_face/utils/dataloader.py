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
"""Facenet triplet dataloader.

Annotation format follows /home/zhongy/facenet-pytorch-change:

    <class_id>;<image_path>

The dataset samples anchor/positive/negative triplets and collates them as
[anchors, positives, negatives], matching the original training method.
"""

import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def cvt_color(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    return image.convert('RGB')


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


def parse_annotation_lines(annotation_lines):
    paths = []
    labels = []
    class_to_paths = defaultdict(list)

    for line in annotation_lines:
        line = line.strip()
        if not line:
            continue
        label_text, image_path = line.split(';', 1)
        label = int(label_text)
        image_path = image_path.strip()
        paths.append(image_path)
        labels.append(label)
        class_to_paths[label].append(image_path)

    return np.array(paths, dtype=np.object_), np.array(labels), dict(class_to_paths)


def get_hw(input_shape):
    if len(input_shape) == 3 and input_shape[0] in (1, 3):
        return input_shape[1], input_shape[2]
    if len(input_shape) >= 2:
        return input_shape[0], input_shape[1]
    raise ValueError(f'Invalid input_shape: {input_shape}')


def get_num_classes(annotation_path):
    with open(annotation_path, 'r') as f:
        _, labels, _ = parse_annotation_lines(f.readlines())
    if len(labels) == 0:
        raise ValueError(f'No samples found in annotation file: {annotation_path}')
    return int(np.max(labels)) + 1


class FacenetDataset(Dataset):
    """Anchor/positive/negative sampler.

    Based on /home/zhongy/facenet-pytorch-change/utils/dataloader.py.
    """

    def __init__(self, input_shape, annotation_lines, num_classes, random=True, transform=None):
        self.input_shape = input_shape
        self.lines = annotation_lines
        self.num_classes = num_classes
        self.random = random
        self.transform = transform
        self.paths, self.labels, self.class_to_paths = parse_annotation_lines(annotation_lines)
        self.length = len(self.paths)
        self.positive_classes = [label for label, paths in self.class_to_paths.items() if len(paths) >= 2]
        self.available_classes = [label for label, paths in self.class_to_paths.items() if len(paths) >= 1]

        if len(self.positive_classes) == 0:
            raise ValueError('FacenetDataset requires at least one class with two images for anchor/positive sampling')
        if len(self.available_classes) < 2:
            raise ValueError('FacenetDataset requires at least two classes for negative sampling')

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.random:
            anchor_label = int(np.random.choice(self.positive_classes))
            anchor_paths = self.class_to_paths[anchor_label]
            anchor_path, positive_path = np.random.choice(anchor_paths, 2, replace=False)

            negative_classes = [label for label in self.available_classes if label != anchor_label]
            negative_label = int(np.random.choice(negative_classes))
            negative_path = np.random.choice(self.class_to_paths[negative_label])
        else:
            anchor_label = self.positive_classes[index % len(self.positive_classes)]
            anchor_paths = self.class_to_paths[anchor_label]
            anchor_path = anchor_paths[index % len(anchor_paths)]
            positive_path = anchor_paths[(index + 1) % len(anchor_paths)]

            negative_classes = [label for label in self.available_classes if label != anchor_label]
            negative_label = negative_classes[index % len(negative_classes)]
            negative_path = self.class_to_paths[negative_label][index % len(self.class_to_paths[negative_label])]

        images = torch.stack(
            [
                self.load_image(anchor_path),
                self.load_image(positive_path),
                self.load_image(negative_path),
            ],
            dim=0,
        )
        labels = torch.tensor([anchor_label, anchor_label, negative_label], dtype=torch.long)
        return images, labels

    def load_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)

        height, width = get_hw(self.input_shape)
        with Image.open(image_path) as image:
            image = cvt_color(image)
            image = resize_image(image, [width, height], letterbox_image=True)
            if self.transform is not None:
                return self.transform(image)
            array = np.asarray(image, dtype='float32') / 255.0
            array = np.transpose(array, [2, 0, 1])
            return torch.from_numpy(array)


class LFWDataset(Dataset):
    """Placeholder for optional LFW pair evaluation."""

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            'TODO: port LFWDataset from /home/zhongy/facenet-pytorch-change/utils/dataloader.py if LFW eval is needed.'
        )


def dataset_collate(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch], dim=0)

    anchors = images[:, 0, :, :, :]
    positives = images[:, 1, :, :, :]
    negatives = images[:, 2, :, :, :]
    images = torch.cat([anchors, positives, negatives], dim=0).float()

    anchor_labels = labels[:, 0]
    positive_labels = labels[:, 1]
    negative_labels = labels[:, 2]
    labels = torch.cat([anchor_labels, positive_labels, negative_labels], dim=0).long()
    return images, labels
