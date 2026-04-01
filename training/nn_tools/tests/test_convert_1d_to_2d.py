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
"""Tests for 1D → 2D model conversion and 1D-aware activation replacement."""

import copy
import os
import sys
import tempfile
import unittest

import torch
import torch.nn as nn

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from nn_tools.activations import RangeNormPoly2d, RangeNormPoly1d
from nn_tools.replace import (
    replace_activation_with_poly,
    replace_maxpool_with_avgpool,
    convert_1d_to_2d,
    _has_1d_modules,
    prepare_for_fhe,
)
from nn_tools.export import export_to_onnx


class SimpleConv1dModel(nn.Module):
    """Conv1d + BN + ReLU + AvgPool + Flatten + Linear."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(32)
        self.act = nn.ReLU()
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 64, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Conv1dWithMaxPool(nn.Module):
    """Conv1d + MaxPool1d (for testing maxpool replacement)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(x))


class Conv1dBlock(nn.Module):
    """A block with Conv1d + BN + ReLU for testing context-aware replacement."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MixedModel(nn.Module):
    """Model with both 1D and 2D blocks (edge case)."""

    def __init__(self):
        super().__init__()
        self.block_1d = Conv1dBlock(8, 16)
        self.block_2d = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

    def forward(self, x):
        # 1D part
        x = self.block_1d(x)
        # Reshape to 2D
        x = x.unsqueeze(2)
        x = self.block_2d(x)
        return x


class TestReplace1DActivation(unittest.TestCase):
    """Test that replace_activation_with_poly chooses the right factory per context."""

    def test_1d_context_gets_rangenormpoly1d(self):
        model = Conv1dBlock(8, 16)
        replace_activation_with_poly(model, old_cls=nn.ReLU)
        acts = [m for m in model.modules() if isinstance(m, (RangeNormPoly1d, RangeNormPoly2d))]
        self.assertEqual(len(acts), 1)
        self.assertIsInstance(acts[0], RangeNormPoly1d)

    def test_2d_context_gets_rangenormpoly2d(self):
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        replace_activation_with_poly(model, old_cls=nn.ReLU)
        acts = [m for m in model.modules() if isinstance(m, (RangeNormPoly1d, RangeNormPoly2d))]
        self.assertEqual(len(acts), 1)
        self.assertIsInstance(acts[0], RangeNormPoly2d)

    def test_mixed_model_gets_both(self):
        model = MixedModel()
        replace_activation_with_poly(model, old_cls=nn.ReLU)
        poly1d = [m for m in model.modules() if isinstance(m, RangeNormPoly1d)]
        poly2d = [m for m in model.modules() if isinstance(m, RangeNormPoly2d)]
        self.assertEqual(len(poly1d), 1, 'Expected 1 RangeNormPoly1d in 1D block')
        self.assertEqual(len(poly2d), 1, 'Expected 1 RangeNormPoly2d in 2D block')

    def test_maxpool1d_replaced_with_avgpool1d(self):
        model = Conv1dWithMaxPool()
        replace_maxpool_with_avgpool(model)
        self.assertFalse(any(isinstance(m, nn.MaxPool1d) for m in model.modules()))
        self.assertTrue(any(isinstance(m, nn.AvgPool1d) for m in model.modules()))


class TestConvert1dTo2d(unittest.TestCase):
    """Test convert_1d_to_2d numerical correctness and structure."""

    def test_numerical_equivalence(self):
        """Output of the converted 2D model must match the original 1D model."""
        model = SimpleConv1dModel()
        model.eval()

        # Deep copy before in-place conversion
        model_ref = copy.deepcopy(model)
        x_1d = torch.randn(2, 16, 128)

        with torch.no_grad():
            out_1d = model_ref(x_1d)

        model_2d = convert_1d_to_2d(model)
        model_2d.eval()

        with torch.no_grad():
            out_2d = model_2d(x_1d)  # wrapper handles unsqueeze

        self.assertTrue(
            torch.allclose(out_1d, out_2d, atol=1e-5),
            f'Max diff: {(out_1d - out_2d).abs().max().item():.2e}',
        )

    def test_no_1d_modules_remain(self):
        model = SimpleConv1dModel()
        model_2d = convert_1d_to_2d(model)
        self.assertFalse(
            _has_1d_modules(model_2d),
            'Converted model should not contain any 1D modules',
        )

    def test_has_2d_modules(self):
        model = SimpleConv1dModel()
        model_2d = convert_1d_to_2d(model)
        inner = model_2d.model  # unwrap
        self.assertTrue(any(isinstance(m, nn.Conv2d) for m in inner.modules()))
        self.assertTrue(any(isinstance(m, nn.BatchNorm2d) for m in inner.modules()))
        self.assertTrue(any(isinstance(m, nn.AvgPool2d) for m in inner.modules()))

    def test_conv2d_kernel_shape(self):
        model = SimpleConv1dModel()
        model_2d = convert_1d_to_2d(model)
        conv2d = next(m for m in model_2d.modules() if isinstance(m, nn.Conv2d))
        self.assertEqual(conv2d.kernel_size, (1, 3))
        self.assertEqual(conv2d.stride, (1, 1))
        self.assertEqual(conv2d.padding, (0, 1))

    def test_rangenormpoly1d_to_2d(self):
        """RangeNormPoly1d should be converted to RangeNormPoly2d."""
        model = Conv1dBlock(8, 16)
        replace_activation_with_poly(model, old_cls=nn.ReLU)
        self.assertTrue(any(isinstance(m, RangeNormPoly1d) for m in model.modules()))

        # Run a forward pass to initialize lazy buffers
        model.eval()
        with torch.no_grad():
            model(torch.randn(1, 8, 64))

        model_2d = convert_1d_to_2d(model)
        self.assertFalse(any(isinstance(m, RangeNormPoly1d) for m in model_2d.modules()))
        self.assertTrue(any(isinstance(m, RangeNormPoly2d) for m in model_2d.modules()))

    def test_onnx_export(self):
        """Converted model should be exportable to ONNX."""
        model = SimpleConv1dModel()
        model_2d = convert_1d_to_2d(model)
        model_2d.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, 'test.onnx')
            export_to_onnx(
                model_2d,
                save_path=onnx_path,
                input_size=(1, 16, 128),
                save_h5=False,
                verbose=False,
            )
            self.assertTrue(os.path.exists(onnx_path))

            import onnx

            onnx_model = onnx.load(onnx_path)
            op_types = {n.op_type for n in onnx_model.graph.node}
            # Should not contain any 1D-specific ops
            self.assertNotIn('Conv1d', op_types)
            # Should contain Conv (ONNX uses "Conv" for both 1D/2D, but kernel shape should be 2D)
            self.assertIn('Conv', op_types)

    def test_prepare_for_fhe_with_1d_model(self):
        """prepare_for_fhe should work on a 1D model (activation replacement is 1D-aware)."""
        model = Conv1dBlock(8, 16)
        prepare_for_fhe(model, input_size=(1, 8, 64))
        # Should have RangeNormPoly1d (not 2d) since model is 1D
        self.assertTrue(any(isinstance(m, RangeNormPoly1d) for m in model.modules()))
        self.assertFalse(any(isinstance(m, RangeNormPoly2d) for m in model.modules()))


class TestConvert1dTo2dWithPoly(unittest.TestCase):
    """End-to-end: prepare_for_fhe -> convert_1d_to_2d -> export_to_onnx."""

    def test_full_pipeline(self):
        model = Conv1dBlock(8, 16)

        # Step 1: FHE preparation (replaces ReLU -> RangeNormPoly1d, MaxPool -> AvgPool)
        prepare_for_fhe(model, input_size=(1, 8, 64))

        # Deep copy before in-place conversion
        model_ref = copy.deepcopy(model)
        model_ref.eval()
        x = torch.randn(1, 8, 64)
        with torch.no_grad():
            out_1d = model_ref(x)

        # Step 2: Convert 1D -> 2D
        model_2d = convert_1d_to_2d(model)
        model_2d.eval()
        with torch.no_grad():
            out_2d = model_2d(x)

        # Squeeze the extra height dim for comparison: (B, C, 1, L) -> (B, C, L)
        out_2d_squeezed = out_2d.squeeze(2)

        self.assertEqual(out_1d.shape, out_2d_squeezed.shape)
        self.assertTrue(
            torch.allclose(out_1d, out_2d_squeezed, atol=1e-5),
            f'Max diff: {(out_1d - out_2d_squeezed).abs().max().item():.2e}',
        )

        # Step 3: Export to ONNX
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, 'poly_1d.onnx')
            export_to_onnx(
                model_2d,
                save_path=onnx_path,
                input_size=(1, 8, 64),
                save_h5=False,
                verbose=False,
            )
            self.assertTrue(os.path.exists(onnx_path))


if __name__ == '__main__':
    unittest.main()
