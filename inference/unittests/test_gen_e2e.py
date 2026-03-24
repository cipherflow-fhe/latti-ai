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

"""
End-to-end test data generator: Model Compiler → FHE Instruction Generation → Test Manifest.

This script creates a closed-loop test pipeline:
1. Create a single-layer PyTorch model
2. Export to ONNX → JSON
3. Run the full model compiler pipeline (run_pipeline)
4. Generate FHE instructions (gen_custom_task)
5. Write test_manifest.json for C++ test consumption

Usage:
    python test_gen_e2e.py                          # Generate all test data
    python test_gen_e2e.py TestConv2D.test_conv_1ch_s1  # Generate specific test
"""

import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
base_path = project_root / 'build' / 'inference' / 'hetero_e2e'

sys.path.insert(0, str(project_root / 'training' / 'model_compiler'))
sys.path.insert(0, str(project_root / 'training'))
sys.path.insert(0, str(project_root / 'inference'))

from model_export.onnx_to_json import onnx_to_json
from pipeline import run_pipeline
from model_generator.deploy_cmds import gen_custom_task, set_param


def generate_via_compiler(
    model: nn.Module,
    input_size: tuple,
    output_dir: Path,
    style: str = 'ordinary',
    test_name: str = '',
    extra_manifest: dict = None,
):
    """Run a model through the full compiler pipeline and generate FHE instructions.

    Args:
        model: PyTorch model (single-layer).
        input_size: Input tensor shape, e.g. (1, 32, 32, 32).
        output_dir: Where to write the compiled output (hetero_e2e/test_name/...).
        style: 'ordinary' or 'multiplexed'.
        test_name: Human-readable test name for the manifest.
        extra_manifest: Additional manifest fields (layer-specific params).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        onnx_path = tmpdir / 'model.onnx'
        json_path = tmpdir / 'model.json'

        # Step 1: Export PyTorch → ONNX
        dummy_input = torch.randn(*input_size)
        torch.onnx.export(
            model.eval(),
            dummy_input,
            str(onnx_path),
            input_names=['input_0'],
            output_names=['output'],
            opset_version=18,
            training=torch.onnx.TrainingMode.EVAL,
            verbose=False,
        )

        # Step 2: ONNX → JSON
        onnx_to_json(str(onnx_path), str(json_path), style)

        # Step 3: Run full compiler pipeline
        # This calls dump_graph() internally, producing:
        #   output_dir/task/server/nn_layers_ct_0.json
        #   output_dir/task/server/task_config.json
        #   output_dir/task/server/ckks_parameter.json
        graph, score = run_pipeline(
            num_experiments=1,
            input_file_path=json_path,
            output_dir=output_dir,
            temperature=0.0,
            num_workers=1,
            style=style,
            graph_type='btp',
        )

    server_path = output_dir / 'task' / 'server'

    # Step 4: Generate FHE instructions (mega_ag)
    gen_custom_task(str(server_path), use_gpu=True, style=style)

    # Step 5: Write test_manifest.json
    write_manifest(graph, server_path, style, test_name, extra_manifest)

    print(f'  [OK] {test_name} -> {server_path}')
    return graph


def write_manifest(graph, server_path: Path, style: str, test_name: str, extra: dict = None):
    """Write test_manifest.json with all parameters needed by the C++ test."""
    from components import FeatureNode, ComputeNode, config
    import networkx as nx

    features = {}
    for node in graph.dag.nodes:
        if not isinstance(node, FeatureNode):
            continue
        attrs = graph.dag.nodes[node]
        f_info = {
            'dim': node.dim,
            'channel': node.channel,
            'level': attrs.get('level', 0),
            'pack_num': attrs.get('pack_num', 0),
        }
        if node.dim in (1, 2):
            f_info['shape'] = node.shape
            f_info['skip'] = attrs.get('skip', [1, 1])
        if node.dim == 0:
            skip_val = attrs.get('skip', [1])
            f_info['skip'] = (
                skip_val if isinstance(skip_val, int) else skip_val[0] if isinstance(skip_val, list) else skip_val
            )
        features[node.node_id] = f_info

    layers = {}
    for node in graph.dag.nodes:
        if not isinstance(node, ComputeNode):
            continue
        l_info = {
            'type': node.layer_type,
            'channel_input': node.channel_input,
            'channel_output': node.channel_output,
        }
        if hasattr(node, 'kernel_shape'):
            l_info['kernel_shape'] = node.kernel_shape
        if hasattr(node, 'stride'):
            l_info['stride'] = node.stride
        if hasattr(node, 'groups'):
            l_info['groups'] = getattr(node, 'groups', 1)
        if hasattr(node, 'order') and node.order > 0:
            l_info['order'] = node.order
        layers[node.layer_id] = l_info

    # Identify input/output features
    input_nodes = [n for n in graph.dag.nodes if isinstance(n, FeatureNode) and graph.dag.in_degree(n) == 0]
    output_nodes = [n for n in graph.dag.nodes if isinstance(n, FeatureNode) and graph.dag.out_degree(n) == 0]

    manifest = {
        'test_name': test_name,
        'N': config.fhe_param.poly_modulus_degree,
        'style': style,
        'input_features': [n.node_id for n in input_nodes],
        'output_features': [n.node_id for n in output_nodes],
        'features': features,
        'layers': layers,
        'error_threshold': {
            'max_error_ratio': 0.05,
            'rmse_ratio': 0.01,
        },
    }
    if extra:
        manifest.update(extra)

    with open(server_path / 'test_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)


# ─── Test Models ──────────────────────────────────────────────────────────────


class ConvModel(nn.Module):
    """Single conv2d layer for testing."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=True,
        )

    def forward(self, x):
        return self.conv(x)


class DepthwiseConvModel(nn.Module):
    """Depthwise conv2d layer for testing (groups=in_channels)."""

    def __init__(self, channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=channels,
            bias=True,
        )

    def forward(self, x):
        return self.conv(x)


class ConvPolyReluModel(nn.Module):
    """Conv + PolyReLU for testing activation through compiler."""

    def __init__(self, channels):
        super().__init__()
        from nn_tools.activations import RangeNormPoly2d

        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.act = RangeNormPoly2d(num_features=channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class ConvReshapeDenseModel(nn.Module):
    """Conv → Reshape → Dense for testing FC through compiler."""

    def __init__(self, in_ch, spatial, out_features):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False)
        flat_size = in_ch * spatial * spatial
        self.dense = nn.Linear(flat_size, out_features, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x


# ─── Test Cases ───────────────────────────────────────────────────────────────


class TestConv2D(unittest.TestCase):
    """Conv2D tests going through the full compiler pipeline."""

    def test_conv_1ch_s1(self):
        """Single-channel conv2d with stride=1, various input/kernel shapes."""
        configs = [{'input_shape': (s, s), 'kernel_shape': k} for s in [4, 8, 16, 32, 64] for k in [1, 3, 5]]
        for cfg in configs:
            s = cfg['input_shape'][0]
            k = cfg['kernel_shape']
            with self.subTest(input_shape=cfg['input_shape'], kernel_shape=k):
                model = ConvModel(1, 1, k, stride=1)
                out_dir = base_path / f'conv2d_1ch_s1_i{s}_k{k}'
                generate_via_compiler(
                    model=model,
                    input_size=(1, 1, s, s),
                    output_dir=out_dir,
                    style='ordinary',
                    test_name=f'conv2d_1ch_s1_i{s}_k{k}',
                )

    def test_conv_mch_s1(self):
        """Multi-channel conv2d with stride=1, 32x32 input, 3x3 kernel."""
        n_ins = [1, 3, 4, 16]
        n_outs = [1, 3, 4, 32]
        for n_in in n_ins:
            for n_out in n_outs:
                with self.subTest(n_in=n_in, n_out=n_out):
                    model = ConvModel(n_in, n_out, 3, stride=1)
                    out_dir = base_path / f'conv2d_mch_s1_cin{n_in}_cout{n_out}'
                    generate_via_compiler(
                        model=model,
                        input_size=(1, n_in, 32, 32),
                        output_dir=out_dir,
                        style='ordinary',
                        test_name=f'conv2d_mch_s1_cin{n_in}_cout{n_out}',
                    )

    def test_conv_mch_s2(self):
        """Multi-channel conv2d with stride=2, 32x32 input, 3x3 kernel."""
        n_ins = [1, 4, 16]
        n_outs = [1, 4, 32]
        for n_in in n_ins:
            for n_out in n_outs:
                with self.subTest(n_in=n_in, n_out=n_out):
                    model = ConvModel(n_in, n_out, 3, stride=2)
                    out_dir = base_path / f'conv2d_mch_s2_cin{n_in}_cout{n_out}'
                    generate_via_compiler(
                        model=model,
                        input_size=(1, n_in, 32, 32),
                        output_dir=out_dir,
                        style='ordinary',
                        test_name=f'conv2d_mch_s2_cin{n_in}_cout{n_out}',
                    )

    def test_mux_conv_varied_stride(self):
        """Multiplexed conv2d with different strides."""
        configs = [{'n_in': n, 'n_out': n, 'stride': s} for n in [4, 8, 32] for s in [1, 2]]
        for cfg in configs:
            with self.subTest(**cfg):
                model = ConvModel(cfg['n_in'], cfg['n_out'], 3, stride=cfg['stride'])
                tag = f'mux_conv_s{cfg["stride"]}_c{cfg["n_in"]}'
                out_dir = base_path / tag
                generate_via_compiler(
                    model=model,
                    input_size=(1, cfg['n_in'], 32, 32),
                    output_dir=out_dir,
                    style='multiplexed',
                    test_name=tag,
                )

    def test_mux_dw_s2(self):
        """Multiplexed depthwise conv2d with stride=2."""
        for n_ch in [4, 8, 32]:
            with self.subTest(n_ch=n_ch):
                model = DepthwiseConvModel(n_ch, 3, stride=2)
                tag = f'mux_dw_s2_c{n_ch}'
                out_dir = base_path / tag
                generate_via_compiler(
                    model=model,
                    input_size=(1, n_ch, 32, 32),
                    output_dir=out_dir,
                    style='multiplexed',
                    test_name=tag,
                )


class TestActivation(unittest.TestCase):
    """Activation layer tests going through the full compiler pipeline."""

    def test_conv_polyrelu(self):
        """Conv + PolyReLU through compiler."""
        model = ConvPolyReluModel(32)
        out_dir = base_path / 'conv_polyrelu_32ch'
        generate_via_compiler(
            model=model,
            input_size=(1, 32, 32, 32),
            output_dir=out_dir,
            style='ordinary',
            test_name='conv_polyrelu_32ch',
        )


class TestDense(unittest.TestCase):
    """Dense (FC) layer tests going through the full compiler pipeline."""

    def test_conv_reshape_dense(self):
        """Conv → Reshape → Dense through compiler."""
        model = ConvReshapeDenseModel(3, 32, 32)
        out_dir = base_path / 'conv_reshape_dense'
        generate_via_compiler(
            model=model,
            input_size=(1, 3, 32, 32),
            output_dir=out_dir,
            style='ordinary',
            test_name='conv_reshape_dense',
        )


if __name__ == '__main__':
    unittest.main()
