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

"""Unit tests for pipeline._active_slots / pipeline._infer_slots.

Phase 1 of the latti-ai sparse-bootstrap E2E plan. Verifies the
log_slots auto-detection pass picks the smallest power-of-two slot
count covering every bootstrap input under Definition A
(shape × pack_num // skip²) — and falls back to N/2 when the result
already equals dense full-packing or when there is no bootstrap.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slot_inference import _active_slots, _infer_slots  # noqa: E402


def _write_graph(server_dir: Path, layers, features) -> None:
    """Write a synthetic nn_layers_ct_0.json under server_dir."""
    nn = {
        'feature': features,
        'layer': {l['id']: l for l in layers},
        'input_feature': [],
        'output_feature': [],
    }
    with open(server_dir / 'nn_layers_ct_0.json', 'w') as f:
        json.dump(nn, f)


class TestActiveSlots(unittest.TestCase):
    def test_dim2_full_packing(self):
        # CIFAR10's typical bootstrap input shape — fully packed at N=2^16.
        feat = {'dim': 2, 'shape': [32, 32], 'pack_num': 32, 'skip': [1, 1]}
        self.assertEqual(_active_slots(feat), 32768)

    def test_dim2_tiny_target(self):
        # Tiny-MLP target: log_slots=8 → 256 active slots.
        feat = {'dim': 2, 'shape': [8, 8], 'pack_num': 4, 'skip': [1, 1]}
        self.assertEqual(_active_slots(feat), 256)

    def test_dim2_with_skip(self):
        # skip[0]*skip[1] = 4 divides the per-ct slot count.
        feat = {'dim': 2, 'shape': [16, 16], 'pack_num': 16, 'skip': [2, 2]}
        self.assertEqual(_active_slots(feat), 1024)

    def test_dim1(self):
        feat = {'dim': 1, 'shape': [128], 'pack_num': 4, 'skip': 1}
        self.assertEqual(_active_slots(feat), 512)

    def test_dim0(self):
        feat = {'dim': 0, 'pack_num': 32, 'skip': 1}
        self.assertEqual(_active_slots(feat), 32)

    def test_dim0_skip(self):
        feat = {'dim': 0, 'pack_num': 32, 'skip': 4}
        self.assertEqual(_active_slots(feat), 8)


class TestInferSlots(unittest.TestCase):
    def test_no_bootstrap_returns_dense(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_graph(
                tmp_path,
                layers=[{'id': 'L0', 'type': 'conv2d', 'feature_input': ['f0'], 'feature_output': ['f1']}],
                features={
                    'f0': {'dim': 2, 'shape': [16, 16], 'pack_num': 16, 'skip': [1, 1]},
                    'f1': {'dim': 2, 'shape': [16, 16], 'pack_num': 16, 'skip': [1, 1]},
                },
            )
            self.assertEqual(_infer_slots(tmp_path, n=16384), 8192)

    def test_single_sparse_log_slots_8(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_graph(
                tmp_path,
                layers=[{'id': 'B0', 'type': 'bootstrapping', 'feature_input': ['f0'], 'feature_output': ['f1']}],
                features={
                    'f0': {'dim': 2, 'shape': [8, 8], 'pack_num': 4, 'skip': [1, 1]},
                    'f1': {'dim': 2, 'shape': [8, 8], 'pack_num': 4, 'skip': [1, 1]},
                },
            )
            self.assertEqual(_infer_slots(tmp_path, n=65536), 256)

    def test_multi_bootstrap_takes_max(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_graph(
                tmp_path,
                layers=[
                    {'id': 'B0', 'type': 'bootstrapping', 'feature_input': ['f0'], 'feature_output': ['f0r']},
                    {'id': 'B1', 'type': 'bootstrapping', 'feature_input': ['f1'], 'feature_output': ['f1r']},
                ],
                features={
                    'f0': {'dim': 2, 'shape': [8, 8], 'pack_num': 4, 'skip': [1, 1]},
                    'f0r': {'dim': 2, 'shape': [8, 8], 'pack_num': 4, 'skip': [1, 1]},
                    'f1': {'dim': 2, 'shape': [16, 16], 'pack_num': 8, 'skip': [1, 1]},
                    'f1r': {'dim': 2, 'shape': [16, 16], 'pack_num': 8, 'skip': [1, 1]},
                },
            )
            # max(256, 2048) = 2048 → log_slots = 11
            self.assertEqual(_infer_slots(tmp_path, n=65536), 2048)

    def test_log_slots_floor_at_8(self):
        # active_slots = 1 would round to log_slots=0. HEonGPU itself supports
        # log_slots in [2, 14], but the Python frontend's _gen_wfft_index_map
        # panics with a negative shift count below log_slots=8 at default
        # CTS/STC depth, so _infer_slots clamps to 8 (256 slots).
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_graph(
                tmp_path,
                layers=[{'id': 'B0', 'type': 'bootstrapping', 'feature_input': ['f0'], 'feature_output': ['f0r']}],
                features={
                    'f0': {'dim': 0, 'pack_num': 1, 'skip': 1},
                    'f0r': {'dim': 0, 'pack_num': 1, 'skip': 1},
                },
            )
            self.assertEqual(_infer_slots(tmp_path, n=65536), 256)  # log_slots=8

    def test_clamps_to_dense_when_fully_packed(self):
        # CIFAR10-shaped: 32×32×32 = 32768 = N/2 → dense fallback.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_graph(
                tmp_path,
                layers=[{'id': 'B0', 'type': 'bootstrapping', 'feature_input': ['f0'], 'feature_output': ['f0r']}],
                features={
                    'f0': {'dim': 2, 'shape': [32, 32], 'pack_num': 32, 'skip': [1, 1]},
                    'f0r': {'dim': 2, 'shape': [32, 32], 'pack_num': 32, 'skip': [1, 1]},
                },
            )
            self.assertEqual(_infer_slots(tmp_path, n=65536), 32768)

    def test_real_cifar10_graph(self):
        # Spot-check against the committed CIFAR10 example: must remain dense.
        latti_ai_root = Path(__file__).resolve().parents[3]
        cifar10_server = latti_ai_root / 'examples' / 'test_cifar10' / 'task' / 'server'
        if (cifar10_server / 'nn_layers_ct_0.json').exists():
            self.assertEqual(_infer_slots(cifar10_server, n=65536), 32768)

    def test_real_mnist_graph(self):
        # MNIST has no bootstrap; must return dense fallback.
        latti_ai_root = Path(__file__).resolve().parents[3]
        mnist_server = latti_ai_root / 'examples' / 'test_mnist' / 'task' / 'server'
        if (mnist_server / 'nn_layers_ct_0.json').exists():
            self.assertEqual(_infer_slots(mnist_server, n=16384), 8192)


if __name__ == '__main__':
    unittest.main()
