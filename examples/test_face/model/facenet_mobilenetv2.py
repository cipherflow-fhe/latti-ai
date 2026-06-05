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
"""Facenet-style wrapper around the test_imagenet MobileNetV2 backbone."""

import torch.nn as nn
import torch.nn.functional as F

from .mobilenetv2 import mobilenetv2


class FaceNetMobileNetV2(nn.Module):
    """MobileNetV2 backbone with Facenet-style embedding and classifier heads.

    This is a skeleton mirroring the train/predict interface from
    /home/zhongy/facenet-pytorch-change/nets/facenet.py.
    """

    def __init__(
        self,
        num_classes,
        embedding_size=128,
        width_mult=1.0,
        dropout_keep_prob=0.5,
        use_embedding_bn=True,
        normalize_embedding=False,
    ):
        super().__init__()
        self.backbone = mobilenetv2(num_classes=num_classes, width_mult=width_mult)
        self.embedding_size = embedding_size
        self.use_embedding_bn = use_embedding_bn
        self.normalize_embedding = normalize_embedding

        flat_shape = self.backbone.classifier.in_features
        self.dropout = nn.Dropout(1 - dropout_keep_prob)
        self.post_pool_activation = nn.ReLU(inplace=True)
        self.embedding = nn.Linear(flat_shape, embedding_size, bias=False)
        self.embedding_bn = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward_features(self, x):
        x = self.backbone.features(x)
        x = self.backbone.conv(x)
        x = self.backbone.avgpool(x)
        x = self.post_pool_activation(x)
        return x.view(x.size(0), -1)

    def forward_embedding(self, x, apply_dropout=False, apply_bn=False):
        x = self.forward_features(x)
        if apply_dropout:
            x = self.dropout(x)
        x = self.embedding(x)
        if apply_bn and self.use_embedding_bn:
            x = self.embedding_bn(x)
        if self.normalize_embedding:
            x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x, mode='train'):
        if mode == 'predict':
            return self.forward_embedding(x, apply_dropout=False, apply_bn=True)

        if mode != 'train':
            raise ValueError(f'Unsupported mode: {mode}')

        embedding = self.forward_embedding(x, apply_dropout=True, apply_bn=False)
        logits = self.classifier(embedding)
        return embedding, logits


class FaceEmbeddingExportWrapper(nn.Module):
    """Single-output wrapper for embedding-only ONNX export."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # TODO: confirm whether encrypted inference should export raw embedding,
        # BN-applied embedding, or normalized embedding.
        return self.model(x, mode='predict')
