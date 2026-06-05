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
"""Facenet loss functions."""

import torch


def triplet_loss(embeddings, batch_size, margin=0.2):
    """Semi-hard triplet loss.

    Expected layout matches /home/zhongy/facenet-pytorch-change:
    - embeddings[:batch_size] are anchors
    - embeddings[batch_size:2 * batch_size] are positives
    - embeddings[2 * batch_size:] are negatives
    """
    if embeddings.size(0) < 3 * batch_size:
        raise ValueError(f'Expected at least {3 * batch_size} embeddings, got {embeddings.size(0)}')

    anchor = embeddings[:batch_size]
    positive = embeddings[batch_size : 2 * batch_size]
    negative = embeddings[2 * batch_size : 3 * batch_size]

    pos_dist = torch.sqrt(torch.sum(torch.pow(anchor - positive, 2), dim=-1) + 1e-12)
    neg_dist = torch.sqrt(torch.sum(torch.pow(anchor - negative, 2), dim=-1) + 1e-12)

    basic_loss = pos_dist - neg_dist + margin
    hard_triplets = basic_loss > 0
    if torch.any(hard_triplets):
        return torch.mean(basic_loss[hard_triplets])
    return torch.sum(basic_loss * 0.0)


def combined_facenet_loss(
    embeddings,
    logits,
    labels,
    batch_size,
    ce_loss_fn,
    triplet_weight=1.0,
    ce_weight=1.0,
    margin=0.2,
):
    """Triplet + classification loss used by the training loop."""
    tri_loss = triplet_loss(embeddings, batch_size, margin=margin)
    ce_loss = ce_loss_fn(logits, labels)
    return triplet_weight * tri_loss + ce_weight * ce_loss, tri_loss, ce_loss
