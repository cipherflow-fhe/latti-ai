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

import unittest
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

import os

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent.parent
sys.path.append(str(script_dir.parent))
sys.path.append(str(script_dir.parent.parent))

from nn_tools.export import export_to_onnx, fuse_and_export_h5
from model_export.onnx_to_json import onnx_to_json
from pipeline import run_pipeline
from components import (
    LayerAbstractGraph,
    FeatureNode,
    config,
    ComputeNode,
    ConvComputeNode,
    UpsampleComputeNode,
    SpatialComputeNode,
    ActivationComputeNode,
    PoolComputeNode,
)
import nn_modules
import networkx as nx
import transforms


def check_level_cost(graph: LayerAbstractGraph) -> bool:
    """
    Check that for each compute node: output_level - input_level == level_cost.

    Returns True if all compute nodes satisfy the constraint, False otherwise.
    """
    result = True
    for node in graph.dag.nodes:
        if not isinstance(node, ComputeNode) or node.layer_type in ['drop_level', 'bootstrapping']:
            continue
        level_cost = graph.dag.nodes[node].get('level_cost')
        if level_cost is None:
            continue
        preds: list[FeatureNode] = list(graph.dag.predecessors(node))
        succs: list[FeatureNode] = list(graph.dag.successors(node))
        if not preds or not succs:
            continue
        input_level = max(graph.dag.nodes[p]['level'] for p in preds)
        output_level = graph.dag.nodes[succs[0]]['level']
        if input_level - output_level != level_cost:
            print(
                f'[check_level_cost] FAIL: {node.layer_id} ({node.layer_type}): '
                f'input_level({input_level}) - output_level({output_level}) = '
                f'{input_level - output_level}, expected level_cost={level_cost}'
            )
            result = False
    return result


def check_multi_input_level_skip_aligned(graph: LayerAbstractGraph) -> bool:
    """
    Check that for each compute node with multiple input FeatureNodes,
    all inputs have the same skip and level.

    Returns True if all such nodes satisfy the constraint, False otherwise.
    """
    result = True
    for node in graph.dag.nodes:
        if not isinstance(node, ComputeNode):
            continue
        preds: list[FeatureNode] = list(graph.dag.predecessors(node))
        if len(preds) < 2:
            continue
        base_level = graph.dag.nodes[preds[0]]['level']
        base_skip = graph.dag.nodes[preds[0]]['skip'][:2]
        for p in preds[1:]:
            p_level = graph.dag.nodes[p]['level']
            p_skip = graph.dag.nodes[p]['skip'][:2]
            if p_level != base_level:
                print(
                    f'[check_multi_input_consistency] FAIL level: {node.layer_id} ({node.layer_type}): '
                    f'{preds[0].node_id} level={base_level} vs {p.node_id} level={p_level}'
                )
                result = False
            if p_skip != base_skip:
                print(
                    f'[check_multi_input_consistency] FAIL skip: {node.layer_id} ({node.layer_type}): '
                    f'{preds[0].node_id} skip={base_skip} vs {p.node_id} skip={p_skip}'
                )
                result = False
    return result


def check_feature_scale(graph: LayerAbstractGraph):
    all_nodes_in_topo_sort = list(nx.topological_sort(graph.dag))
    for node in all_nodes_in_topo_sort:
        if not isinstance(node, ComputeNode):
            continue
        preds = list(graph.dag.predecessors(node))
        succs = list(graph.dag.successors(node))
        if not preds or not succs:
            continue
        assert all(p.scale == preds[0].scale for p in preds), (
            f'[calculate_feture_scale_for_test] preds scale mismatch at {node.layer_id}: {[p.scale for p in preds]}'
        )
        f_node = preds[0]
        out_node = succs[0]
        if node.layer_type in config.absorbable_layers:
            out_node.scale = f_node.scale * node.weight_scale
        elif node.layer_type == 'mult_coeff':
            out_node.scale = f_node.scale * (1 / node.coeff)
        elif node.layer_type == 'avgpool2d':
            if (not node.is_adaptive_avgpool) and (not node.is_big_size):
                out_node.scale = f_node.scale
            else:
                out_node.scale = f_node.scale * (node.kernel_shape[0] * node.kernel_shape[1])
        else:
            out_node.scale = f_node.scale

    output_nodes = [node for node, out_deg in graph.dag.out_degree() if out_deg == 0 and isinstance(node, FeatureNode)]
    return all(math.isclose(node.scale, 1.0) for node in output_nodes)


def check_dropped_levels_per_subgraph(graph: LayerAbstractGraph) -> bool:
    """
    For every linear subgraph, verify that the sum of level_cost values on all
    drop_level nodes does not exceed config.fhe_param.max_level + 2.

    Returns True if all subgraphs satisfy the constraint, False otherwise.
    """
    subgraphs = transforms.split_graph_to_linear_subgraph(graph.dag)
    result = True
    for sub in subgraphs:
        total_dropped = sum(
            sub.nodes[node].get('level_cost', 0)
            for node in sub.nodes
            if isinstance(node, ComputeNode) and node.layer_type == 'drop_level'
        )
        if total_dropped > config.fhe_param.max_level + 2:
            print(
                f'[check_dropped_levels_per_subgraph] FAIL: subgraph total dropped levels '
                f'{total_dropped} > config.fhe_param.max_level + 2 = {config.fhe_param.max_level + 2}'
            )
            result = False
    return result


def check_reshape_sp_info_propagation(graph: LayerAbstractGraph) -> bool:
    """
    Check that every 2D->0D reshape node correctly propagates sp_info from its input:
      output.sp_info.skip[i]         == input node's skip[i]
      output.sp_info.shape[i]        == input node's shape[i]
      output.sp_info.invalid_fill[i] == input node's invalid_fill[i]

    Returns True if all reshape nodes satisfy the constraint, False otherwise.
    """
    result = True
    for node in graph.dag.nodes:
        if not (isinstance(node, ComputeNode) and node.layer_type == 'reshape'):
            continue
        inp = list(graph.dag.predecessors(node))[0]
        out = list(graph.dag.successors(node))[0]
        if not (inp.dim == 2 and out.dim == 0):
            continue
        if out.sp_info['skip'] != graph.dag.nodes[inp]['skip']:
            print(
                f'[check_reshape_sp_info_propagation] FAIL skip: {node.layer_id}: '
                f'output sp_info.skip={out.sp_info["skip"]} != input skip={graph.dag.nodes[inp]["skip"]}'
            )
            result = False
        if out.sp_info['shape'] != inp.shape:
            print(
                f'[check_reshape_sp_info_propagation] FAIL shape: {node.layer_id}: '
                f'output sp_info.shape={out.sp_info["shape"]} != input shape={inp.shape}'
            )
            result = False
        if out.sp_info['invalid_fill'] != inp.invalid_fill:
            print(
                f'[check_reshape_sp_info_propagation] FAIL invalid_fill: {node.layer_id}: '
                f'output sp_info.invalid_fill={out.sp_info["invalid_fill"]} != input invalid_fill={inp.invalid_fill}'
            )
            result = False
        expected_skip = [math.prod(out.sp_info['skip']) * math.prod(out.sp_info['shape'])]
        if graph.dag.nodes[out]['skip'] != expected_skip:
            print(
                f'[check_reshape_sp_info_propagation] FAIL output skip: {node.layer_id}: '
                f'output skip={graph.dag.nodes[out]["skip"]} != '
                f'prod(sp_info.skip)*prod(sp_info.shape)={expected_skip}'
            )
            result = False
    return result


def check_2d_invalid_fill_propagation(graph: LayerAbstractGraph) -> bool:
    """
    Check that every 2D->2D compute node sets the output invalid_fill correctly:

    ordinary packing:
      output.invalid_fill == graph.dag.nodes[input]['skip']

    multiplexed packing:
      - ConvComputeNode or non-adaptive PoolComputeNode: output.invalid_fill == [1, 1]
      - adaptive PoolComputeNode:                        output.invalid_fill == compute_node.stride
      - other (e.g. simple poly):                        output.invalid_fill == input.invalid_fill

    Returns True if all such nodes satisfy the constraint, False otherwise.
    """
    result = True
    for node in graph.dag.nodes:
        if not isinstance(node, ComputeNode):
            continue
        preds: list[FeatureNode] = list(graph.dag.predecessors(node))
        succs: list[FeatureNode] = list(graph.dag.successors(node))
        if not preds or not succs:
            continue
        inp, out = preds[0], succs[0]
        if not (inp.dim == 2 and out.dim == 2):
            continue

        if config.style == 'ordinary':
            expected = graph.dag.nodes[inp]['skip']
        elif isinstance(node, (ConvComputeNode,)) or (
            isinstance(node, PoolComputeNode) and not node.is_adaptive_avgpool
        ):
            expected = [1, 1]
        elif isinstance(node, PoolComputeNode) and node.is_adaptive_avgpool:
            expected = node.stride
        else:
            expected = inp.invalid_fill

        if out.invalid_fill != expected:
            print(
                f'[check_2d_invalid_fill_propagation] FAIL: {node.layer_id} ({node.layer_type}): '
                f'output.invalid_fill={out.invalid_fill} != expected={expected}'
            )
            result = False
    return result


class CompilerTestBase(unittest.TestCase):
    temp_onnx_path = script_dir / 'temp.onnx'
    temp_json_path = script_dir / 'temp.json'
    e2e_base_path = project_root / 'build' / 'inference' / 'hetero_e2e'

    def _check_concat_input_ordering(self, graph: LayerAbstractGraph, raw_json_path):
        """Assert every concat2d node in *graph* preserves the input ordering from *raw_json_path*."""
        raw_graph = LayerAbstractGraph.from_json(str(raw_json_path))

        raw_concat_inputs: dict[str, list[str]] = {}
        for node in raw_graph.dag.nodes:
            if isinstance(node, ComputeNode) and node.layer_type == 'concat2d':
                raw_concat_inputs[node.layer_id] = [p.node_id for p in raw_graph.dag.predecessors(node)]

        for node in graph.dag.nodes:
            if not (isinstance(node, ComputeNode) and node.layer_type == 'concat2d'):
                continue
            if node.layer_id not in raw_concat_inputs:
                continue
            compiled_ids = [
                p.node_id
                for p in sorted(
                    graph.dag.predecessors(node),
                    key=lambda p: graph.dag.edges[p, node]['input_index'],
                )
            ]
            self.assertEqual(
                compiled_ids,
                raw_concat_inputs[node.layer_id],
                f'concat {node.layer_id} input order changed after compilation',
            )

    def _export_and_compile(
        self,
        model,
        input_size,
        style='ordinary',
        graph_type='btp',
        replace=True,
        **export_kwargs,
    ):
        if replace:
            from nn_tools import prepare_for_fhe

            prepare_for_fhe(model, input_size=input_size)

        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=input_size,
            dynamic_batch=False,
            save_h5=False,
            **export_kwargs,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, style)
        graph, score = run_pipeline(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
            style=style,
            graph_type=graph_type,
        )
        return graph, score

    def _export_compile_and_deploy(
        self,
        model,
        input_size,
        test_name,
        style='ordinary',
        replace=True,
        **export_kwargs,
    ):
        """Full E2E pipeline: compile model and generate all files for C++ inference test.

        Produces a complete task directory at e2e_base_path/test_name/task/ containing:
          - client/: ckks_parameter.json, task_config.json, input CSV(s)
          - server/: ckks_parameter.json, task_config.json, nn_layers_ct_0.json,
                     model_parameters.h5, mega_ag.json, ergs/, task_signature.json
        """
        if replace:
            from nn_tools import prepare_for_fhe

            prepare_for_fhe(model, input_size=input_size)

        output_dir = self.e2e_base_path / test_name
        output_dir.mkdir(parents=True, exist_ok=True)

        temp_onnx = output_dir / 'temp.onnx'
        temp_json = output_dir / 'temp.json'

        # Step 1: Export PyTorch → ONNX (no h5 yet)
        export_to_onnx(
            model,
            save_path=temp_onnx,
            input_size=input_size,
            dynamic_batch=False,
            save_h5=False,
            **export_kwargs,
        )

        # Step 2: ONNX → JSON
        onnx_to_json(temp_onnx, temp_json, style)

        # Step 3: Compile (produces task/server/ and task/client/)
        graph, score = run_pipeline(
            num_experiments=1,
            input_file_path=temp_json,
            output_dir=output_dir,
            temperature=0.0,
            num_workers=1,
            style=style,
            graph_type='btp',
        )

        server_dir = output_dir / 'task' / 'server'
        client_dir = output_dir / 'task' / 'client'

        # Step 4: Export model weights to h5
        h5_path = server_dir / 'model_parameters.h5'
        fuse_and_export_h5(model, str(h5_path), verbose=False)

        # Step 6: Read pack_style and param_name from configs
        with open(client_dir / 'task_config.json', 'r') as f:
            task_config = json.load(f)
        pack_style = task_config.get('pack_style', style)

        with open(client_dir / 'ckks_parameter.json', 'r') as f:
            ckks_config = json.load(f)
        first_param = next(iter(ckks_config.values()))
        param_name = first_param.get('param_name', '')

        # Step 7: Generate mega_ag instructions
        # Run in subprocess to avoid global state pollution in lattisense frontend
        import subprocess

        gen_script = (
            f'import sys;'
            f'sys.path.insert(0,"{project_root}");'
            f'sys.path.insert(0,"{project_root / "inference"}");'
            f'from inference.model_generator.deploy_cmds import gen_custom_task;'
            f'gen_custom_task("{server_dir}",param_name="{param_name}",use_gpu=True,style="{pack_style}")'
        )
        result = subprocess.run([sys.executable, '-c', gen_script], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'gen_custom_task failed:\n{result.stderr}')

        # Step 8: Generate random input CSV(s)
        for input_name, input_param in task_config['task_input_param'].items():
            dim = input_param['dim']
            channel = input_param['channel']
            csv_path = client_dir / f'{input_name}.csv'
            if dim == 2:
                h, w = input_param['shape']
                data = np.random.uniform(-1, 1, (channel, h * w))
                np.savetxt(csv_path, data, delimiter=',', fmt='%.6f')
            elif dim == 1:
                (length,) = input_param['shape']
                data = np.random.uniform(-1, 1, (channel, length))
                np.savetxt(csv_path, data, delimiter=',', fmt='%.6f')
            elif dim == 0:
                data = np.random.uniform(-1, 1, (channel,))
                np.savetxt(csv_path, data.reshape(1, -1), delimiter=',', fmt='%.6f')
            else:
                raise ValueError(f'Unsupported input dim={dim} for input "{input_name}"')

        # Cleanup temp files
        temp_onnx.unlink(missing_ok=True)
        temp_json.unlink(missing_ok=True)

        print(f'  [E2E] {test_name} -> {server_dir}')
        return graph, score


class TestSingleLayer(CompilerTestBase):
    @unittest.skip('1D bug: transforms.py IndexError on succ.shape[1]')
    def test_single_conv1d(self):
        model = nn_modules.SingleConv1d()
        graph, score = self._export_and_compile(model, (1, 32, 64))

        self.assertEqual(
            max(graph.dag.nodes[feature]['level'] for feature in graph.dag.nodes if isinstance(feature, FeatureNode)), 1
        )

    @unittest.skip('1D bug: transforms.py IndexError on succ.shape[1]')
    def test_single_act1d(self):
        model = nn_modules.SingleAct1d()
        graph, score = self._export_and_compile(model, (1, 32, 64))

        self.assertEqual(
            max(graph.dag.nodes[feature]['level'] for feature in graph.dag.nodes if isinstance(feature, FeatureNode)), 3
        )

    def test_single_avgpool_big_size(self):
        model = nn_modules.SingleAvgpool()
        graph, score = self._export_and_compile(model, (1, 32, 256, 256))
        self.assertEqual(check_feature_scale(graph), True)

    def test_single_maxpool(self):
        model = nn_modules.SingleMaxpool()
        graph, score = self._export_and_compile(model, (1, 32, 64, 64), replace=False)

    def test_single_reshape(self):
        model = nn_modules.SingleReshape()
        graph, score = self._export_and_compile(model, (1, 16, 4, 4))

    def test_single_mult_ceoff(self):
        model = nn_modules.SingleMultCoeff()
        graph, score = self._export_and_compile(model, (1, 16, 4, 4))
        self.assertEqual(check_feature_scale(graph), True)

    def test_single_conv_with_stride_big_size(self):
        model = nn_modules.SingleConv(2)
        graph, score = self._export_and_compile(model, (1, 32, 256, 256), style='multiplexed')
        res = None
        for node in graph.dag.nodes:
            if isinstance(node, ComputeNode):
                input = list(graph.dag.predecessors(node))[0]
                output = list(graph.dag.successors(node))[0]
                if graph.dag.nodes[output]['skip'] == [1, 1]:
                    res = True
                    break
        self.assertEqual(res, True)

    def test_single_dw_conv_big_size(self):
        model = nn_modules.DepthwiseConv(channels=32, stride=2)
        graph, score = self._export_and_compile(model, (1, 32, 256, 256), style='multiplexed')
        self.assertTrue(any(node.is_big_size for node in graph.dag.nodes if isinstance(node, ConvComputeNode)))


class TestLayerInteraction(CompilerTestBase):
    def test_mismatched_scale(self):
        model = nn_modules.MismatchedScale()
        graph, score = self._export_and_compile(model, (1, 32, 64, 64))

    def test_conv_and_convtranspose(self):
        model = nn_modules.ConvAndConvTransposeBlock()
        graph, score = self._export_and_compile(model, (1, 32, 16, 16), style='multiplexed')
        self.assertTrue(
            any(node.upsample_factor_in == [2, 2] for node in graph.dag.nodes if isinstance(node, ConvComputeNode))
        )
        self.assertTrue(
            any(node.zero_skip == [2, 2] for node in graph.dag.nodes if isinstance(node, ActivationComputeNode))
        )

    def test_conv_and_convtranspose_big_size(self):
        model = nn_modules.ConvAndConvTransposeBlock()
        graph, score = self._export_and_compile(model, (1, 32, 256, 256), style='multiplexed')
        self.assertFalse(
            any(node.upsample_factor_in == [2, 2] for node in graph.dag.nodes if isinstance(node, ConvComputeNode))
        )
        self.assertTrue(any(node.is_big_size for node in graph.dag.nodes if isinstance(node, ConvComputeNode)))

    def test_conv_and_upsample(self):
        model = nn_modules.ConvAndUpsample()
        graph, score = self._export_and_compile(model, (1, 32, 64, 64), style='multiplexed', do_constant_folding=True)
        res = False
        for node in graph.dag.nodes:
            if isinstance(node, ComputeNode) and node.layer_type == 'resize':
                input = list(graph.dag.predecessors(node))[0]
                output = list(graph.dag.successors(node))[0]
                if graph.dag.nodes[output]['skip'][0] == graph.dag.nodes[input]['skip'][0] / node.upsample_factor[0]:
                    res = True
        self.assertEqual(res, True)

    @unittest.skip('will be supported very soon')
    def test_conv_reshape_dense(self):
        model = nn_modules.ConvReshapeAndDense()
        graph, score = self._export_and_compile(model, (1, 3, 32, 32), do_constant_folding=True)
        res = None
        for node in graph.dag.nodes:
            if isinstance(node, ComputeNode) and node.layer_type == 'reshape':
                output = list(graph.dag.successors(node))[0]
                if output.sp_info['skip'][0] == 2:
                    res = True
                    break
        self.assertEqual(res, True)
        self.assertEqual(check_reshape_sp_info_propagation(graph), True)
        self.assertEqual(check_2d_invalid_fill_propagation(graph), True)

    @unittest.skip('will be supported very soon')
    def test_conv_reshape_two_dense(self):
        model = nn_modules.ConvReshapeAndTwoDense()
        graph, score = self._export_and_compile(model, (1, 3, 32, 32), do_constant_folding=True)
        self.assertEqual(check_reshape_sp_info_propagation(graph), True)
        self.assertEqual(check_2d_invalid_fill_propagation(graph), True)
        # Verify 0D -> 0D propagation: the feature between dense0 and dense1 inherits skip
        for node in graph.dag.nodes:
            if isinstance(node, ComputeNode) and node.layer_type == 'fc0':
                inp = list(graph.dag.predecessors(node))[0]
                out = list(graph.dag.successors(node))[0]

                self.assertEqual(
                    graph.dag.nodes[out]['skip'],
                    graph.dag.nodes[inp]['skip'],
                    '0D -> 0D: dense output skip should equal input skip',
                )

    def test_two_dense(self):
        """FC-FC: graph input is 0d; all feature node skips should equal 2**floor(log2(N)/2)."""
        model = nn_modules.TwoDense()
        graph, score = self._export_and_compile(model, (1, 64), do_constant_folding=True)

        N = config.fhe_param.poly_modulus_degree
        expected_skip = 2 ** math.floor(math.log2(N) / 2)

        for node in graph.dag.nodes:
            if isinstance(node, FeatureNode):
                self.assertEqual(
                    graph.dag.nodes[node]['skip'],
                    [expected_skip],
                    f'Feature node {node} skip should be [{expected_skip}], got {graph.dag.nodes[node]["skip"]}',
                )

    def test_conv_avgpool_reshape_dense(self):
        model = nn_modules.ConvAvgpoolReshapeAndDense()
        graph, score = self._export_and_compile(model, (1, 3, 64, 64), style='multiplexed', do_constant_folding=True)
        res = None
        from components import PoolComputeNode

        for node in graph.dag.nodes:
            if isinstance(node, PoolComputeNode):
                input = list(graph.dag.predecessors(node))[0]
                output = list(graph.dag.successors(node))[0]
                if (
                    output.shape[0] == input.shape[0] / node.stride[0]
                    and graph.dag.nodes[output]['skip'][0] == graph.dag.nodes[input]['skip'][0] * node.stride[0]
                ):
                    res = True
                    break
        self.assertEqual(res, True)
        self.assertEqual(check_dropped_levels_per_subgraph(graph), True)
        self.assertEqual(check_reshape_sp_info_propagation(graph), True)
        self.assertEqual(check_2d_invalid_fill_propagation(graph), True)


class TestCompiler(CompilerTestBase):
    def test_conv_series_with_stride(self):
        model = nn_modules.ConvSeriesWithStride()
        graph, score = self._export_and_compile(model, (1, 32, 256, 256), style='multiplexed')

        self.assertLessEqual(
            max(graph.dag.nodes[feature]['level'] for feature in graph.dag.nodes if isinstance(feature, FeatureNode)),
            config.fhe_param.max_level,
        )
        self.assertEqual(check_dropped_levels_per_subgraph(graph), True)

    def test_mult_coeff_series(self):
        model = nn_modules.MultCoeffSeries()
        graph, score = self._export_and_compile(model, (1, 32, 256, 256), style='multiplexed')
        self.assertEqual(check_feature_scale(graph), True)
        self.assertEqual(
            max(graph.dag.nodes[feature]['level'] for feature in graph.dag.nodes if isinstance(feature, FeatureNode)),
            1,
        )
        self.assertEqual(check_dropped_levels_per_subgraph(graph), True)

    def test_conv_and_mult_coeff_series(self):
        model = nn_modules.ConvAndMultCoeffSeries()
        graph, score = self._export_and_compile(model, (1, 32, 256, 256), style='multiplexed')

        self.assertEqual(
            max(graph.dag.nodes[feature]['level'] for feature in graph.dag.nodes if isinstance(feature, FeatureNode)),
            5,
        )
        self.assertEqual(check_dropped_levels_per_subgraph(graph), True)

    def test_resnet_20(self):
        from nn_tools import prepare_for_fhe
        from nn_tools.activations import Simple_Polyrelu
        from resnet import resnet20

        model = resnet20()
        prepare_for_fhe(model, poly_module=Simple_Polyrelu, input_size=(1, 3, 32, 32))

        graph, score = self._export_and_compile(model, (1, 3, 32, 32), style='multiplexed', replace=False)
        self.assertEqual(check_level_cost(graph), True)
        self.assertEqual(check_multi_input_level_skip_aligned(graph), True)
        self.assertEqual(check_dropped_levels_per_subgraph(graph), True)

    def test_intertwined_with_coeff(self):
        model = nn_modules.IntertwinedWithCoeff()
        graph, score = self._export_and_compile(model, (1, 32, 64, 64))
        self.assertEqual(check_dropped_levels_per_subgraph(graph), True)

    # @unittest.skip('Not supported yet')
    def test_multiple_inputs(self):
        model = nn_modules.MutipleInputs()
        input_sizes = [(1, 32, 64, 64)] * model.n_inputs
        graph, score = self._export_and_compile(model, input_sizes)

    # @unittest.skip('Not supported yet')
    def test_multiple_outputs(self):
        model = nn_modules.MutipleOutputs()
        graph, score = self._export_and_compile(model, (1, 32, 64, 64))

    def test_pack_num_ordinary(self):
        model = nn_modules.ConvSeriesWithStride()
        graph, score = self._export_and_compile(model, (1, 32, 64, 64))

        for node in graph.dag.nodes:
            if isinstance(node, FeatureNode):
                attrs = graph.dag.nodes[node]
                self.assertIn('pack_num', attrs)
                self.assertGreater(attrs['pack_num'], 0)
                if node.dim == 0:
                    expected = math.ceil(
                        config.fhe_param.poly_modulus_degree
                        / 2
                        / (
                            attrs['virtual_shape'][0]
                            * attrs['virtual_shape'][1]
                            * attrs['virtual_skip'][0]
                            * attrs['virtual_skip'][1]
                        )
                    )
                else:
                    expected = math.ceil(
                        config.fhe_param.poly_modulus_degree
                        / 2
                        / (node.shape[0] * node.shape[1] * attrs['skip'][0] * attrs['skip'][1])
                    )
                self.assertEqual(attrs['pack_num'], expected)
        self.assertEqual(check_dropped_levels_per_subgraph(graph), True)

    def test_pack_num_multiplexed(self):
        model = nn_modules.ConvSeriesWithStride()
        graph, score = self._export_and_compile(model, (1, 32, 256, 256), style='multiplexed')

        for node in graph.dag.nodes:
            if isinstance(node, FeatureNode):
                attrs = graph.dag.nodes[node]
                self.assertIn('pack_num', attrs)
                self.assertGreater(attrs['pack_num'], 0)
                if node.dim == 0:
                    expected = math.ceil(
                        config.fhe_param.poly_modulus_degree
                        / 2
                        / (
                            attrs['virtual_shape'][0]
                            * attrs['virtual_shape'][1]
                            * attrs['virtual_skip'][0]
                            * attrs['virtual_skip'][1]
                        )
                    )
                else:
                    expected = math.ceil(config.fhe_param.poly_modulus_degree / 2 / (node.shape[0] * node.shape[1]))
                self.assertEqual(attrs['pack_num'], expected)
        self.assertEqual(check_dropped_levels_per_subgraph(graph), True)

    def test_split_skip_connection(self):
        model = nn_modules.SkipConnection()
        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 64, 64]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

        from pipeline import prepare_graph
        from transforms import split_graph_to_linear_subgraph

        config.style = 'ordinary'
        config.graph_type = 'btp'
        raw_graph = LayerAbstractGraph.from_json(self.temp_json_path)
        pt_graph = prepare_graph(raw_graph)
        subs = split_graph_to_linear_subgraph(pt_graph.dag)
        self.assertEqual(len(subs), 2)

    def test_conv_concat_conv(self):
        """Shared-input concat structure with final add:
        concat1: [conv1_out, conv2_out] → 16ch
        concat2: [conv2_out, conv3_out, conv4_out] → 16ch  (conv2_out shared)
        add:     concat1_out + concat2_out
        """
        model = nn_modules.ConvConcatConv()
        graph, score = self._export_and_compile(model, (1, 16, 32, 32))
        # temp_json_path still holds the pre-pipeline JSON at this point
        self._check_concat_input_ordering(graph, self.temp_json_path)


class TestCompilerErrors(CompilerTestBase):
    """Tests that verify the compiler raises the correct errors on invalid inputs."""

    def _export_only(self, model, input_size):
        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=input_size,
            dynamic_batch=False,
            save_h5=False,
        )

    def test_wrong_padding(self):
        self._export_only(nn_modules.WrongPadding(), (1, 32, 64, 64))
        with self.assertRaisesRegex(ValueError, r'Unsupported padding value: \[0, 0, 0, 0\]'):
            onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

    def test_wrong_dilation(self):
        self._export_only(nn_modules.WrongDilation(), (1, 32, 64, 64))
        with self.assertRaisesRegex(ValueError, r'Unsupported dilation value: \[2, 2\]'):
            onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

    def test_wrong_groups(self):
        self._export_only(nn_modules.WrongGroups(), (1, 32, 64, 64))
        with self.assertRaisesRegex(ValueError, r'Unsupported groups value: 2'):
            onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

    def test_unreplaced_relu(self):
        self._export_only(nn_modules.SingleRelu(), (1, 32, 64, 64))
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')
        with self.assertRaisesRegex(ValueError, r'Relu2d is not supported in current mode'):
            run_pipeline(
                num_experiments=1,
                input_file_path=self.temp_json_path,
                output_dir=script_dir,
                temperature=0.0,
                num_workers=1,
                style='ordinary',
                graph_type='btp',
            )


class TestE2E(CompilerTestBase):
    """Generate E2E test data for C++ inference tests.

    Each test compiles a model and produces a complete task directory
    (h5 weights, mega_ag instructions, input CSVs) under
    build/inference/hetero_e2e/<test_name>/.

    Run the C++ test_e2e binary afterwards to verify encrypted vs plaintext
    inference consistency.
    """

    # ── Helper for common assertions ──

    def _max_feature_level(self, graph):
        return max(graph.dag.nodes[f]['level'] for f in graph.dag.nodes if isinstance(f, FeatureNode))

    def _has_bootstrapping(self, graph):
        return any(isinstance(n, ComputeNode) and n.layer_type == 'bootstrapping' for n in graph.dag.nodes)

    # ── Single layer tests (poly_n=8192, ≤5 levels) ──

    def test_e2e_single_conv(self):
        """1 Conv = 1 level → poly_n=8192, no BTP."""
        model = nn_modules.SingleConv()
        graph, score = self._export_compile_and_deploy(model, (1, 32, 8, 8), 'single_conv')
        self.assertEqual(self._max_feature_level(graph), 1)
        self.assertEqual(config.fhe_param.poly_modulus_degree, 8192)

    def test_e2e_single_act(self):
        """1 Act = 3 levels → poly_n=8192, no BTP."""
        model = nn_modules.SingleAct()
        graph, score = self._export_compile_and_deploy(model, (1, 32, 8, 8), 'single_act')
        self.assertEqual(self._max_feature_level(graph), 3)
        self.assertEqual(config.fhe_param.poly_modulus_degree, 8192)

    def test_e2e_single_avgpool(self):
        """1 Avgpool → poly_n=8192, no BTP."""
        model = nn_modules.SingleAvgpool()
        graph, score = self._export_compile_and_deploy(model, (1, 32, 8, 8), 'single_avgpool', style='multiplexed')
        self.assertTrue(check_feature_scale(graph))

    def test_e2e_single_dense(self):
        """1 Dense → poly_n=8192, no BTP."""
        model = nn_modules.SingleDense()
        graph, score = self._export_compile_and_deploy(model, (1, 64), 'single_dense')
        self.assertIsNotNone(graph)

    @unittest.skip('Reshape-only model has no FHE computation for gen_custom_task')
    def test_e2e_single_reshape(self):
        """Reshape → poly_n=8192, no BTP."""
        model = nn_modules.SingleReshape()
        self._export_compile_and_deploy(model, (1, 16, 4, 4), 'single_reshape')

    def test_e2e_single_add(self):
        """Add two inputs → poly_n=8192, no BTP."""
        model = nn_modules.SingleAdd()
        graph, score = self._export_compile_and_deploy(
            model, [(1, 32, 8, 8), (1, 32, 8, 8)], 'single_add', input_names=['x0', 'x1']
        )
        self.assertIsNotNone(graph)

    # ── Layer interaction tests (poly_n=8192) ──

    def test_e2e_conv_batchnorm(self):
        """Conv + BatchNorm (BN absorbed into conv weights) → poly_n=8192, no BTP."""
        model = nn_modules.ConvWithBatchNorms()
        graph, score = self._export_compile_and_deploy(model, (1, 32, 8, 8), 'conv_batchnorm')
        self.assertEqual(self._max_feature_level(graph), 1)

    def test_e2e_conv_reshape_dense(self):
        """Conv → Reshape → Dense pipeline."""
        model = nn_modules.ConvReshapeAndDense()
        self._export_compile_and_deploy(model, (1, 3, 32, 32), 'conv_reshape_dense', do_constant_folding=True)

    def test_e2e_conv_avgpool_reshape_dense(self):
        """Conv → Avgpool → Reshape → Dense, multiplexed."""
        model = nn_modules.ConvAvgpoolReshapeAndDense()
        graph, score = self._export_compile_and_deploy(
            model, (1, 3, 64, 64), 'conv_avgpool_reshape_dense', style='multiplexed', do_constant_folding=True
        )
        self.assertTrue(check_dropped_levels_per_subgraph(graph))

    # ── No-BTP tests (poly_n=8192) ──

    def test_e2e_poly_n_8192(self):
        """1 Conv + 1 Act = 4 levels → poly_n=8192, no BTP."""
        model = nn_modules.PolyDegreeN8192()
        graph, score = self._export_compile_and_deploy(model, (1, 32, 8, 8), 'poly_n_8192')
        self.assertEqual(config.fhe_param.poly_modulus_degree, 8192)
        self.assertEqual(config.fhe_param.max_level, 5)
        self.assertFalse(self._has_bootstrapping(graph))

    # ── No-BTP tests (poly_n=16384, 6-9 levels) ──

    def test_e2e_conv_act(self):
        """3 Conv + 1 Act = 6 levels → poly_n=16384, no BTP."""
        model = nn_modules.PolyDegreeN16384()
        graph, score = self._export_compile_and_deploy(model, (1, 32, 8, 8), 'conv_act')
        self.assertEqual(config.fhe_param.poly_modulus_degree, 16384)
        self.assertEqual(config.fhe_param.max_level, 9)
        self.assertFalse(self._has_bootstrapping(graph))
        self.assertTrue(check_dropped_levels_per_subgraph(graph))

    def test_e2e_resnet_basic_block(self):
        """2 Conv + 2 BN + 2 Act + Add = 8 levels → poly_n=16384, no BTP."""
        model = nn_modules.ResNetBasicBlock(32, 32)
        graph, score = self._export_compile_and_deploy(model, (1, 32, 8, 8), 'resnet_basic_block')
        self.assertTrue(check_level_cost(graph))
        self.assertTrue(check_multi_input_level_skip_aligned(graph))
        self.assertTrue(check_dropped_levels_per_subgraph(graph))

    # ── No-BTP tests (poly_n=32768, 10-17 levels) ──

    def test_e2e_poly_n_32768(self):
        """4 Conv + 2 Act = 10 levels → poly_n=32768, no BTP."""
        model = nn_modules.PolyDegreeN32768()
        graph, score = self._export_compile_and_deploy(model, (1, 32, 8, 8), 'poly_n_32768')
        self.assertEqual(config.fhe_param.poly_modulus_degree, 32768)
        self.assertEqual(config.fhe_param.max_level, 17)
        self.assertFalse(self._has_bootstrapping(graph))
        self.assertTrue(check_dropped_levels_per_subgraph(graph))

    # ── No-BTP tests (poly_n=65536, 18-33 levels) ──

    def test_e2e_poly_n_65536_no_btp(self):
        """6 Conv + 4 Act = 18 levels → poly_n=65536, no BTP."""
        model = nn_modules.PolyDegreeN65536NoBtp()
        graph, score = self._export_compile_and_deploy(model, (1, 32, 8, 8), 'poly_n_65536_no_btp')
        self.assertEqual(config.fhe_param.poly_modulus_degree, 65536)
        self.assertEqual(config.fhe_param.max_level, 33)
        self.assertFalse(self._has_bootstrapping(graph))
        self.assertTrue(check_dropped_levels_per_subgraph(graph))

    # ── BTP tests (poly_n=65536, >33 levels) ──

    def test_e2e_btp(self):
        """4 Conv + 10 Act = 34 levels → poly_n=65536, BTP."""
        model = nn_modules.PolyDegreeNBtp()
        graph, score = self._export_compile_and_deploy(model, (1, 32, 8, 8), 'btp')
        self.assertEqual(config.fhe_param.poly_modulus_degree, 65536)
        self.assertEqual(config.fhe_param.max_level, 9)
        self.assertTrue(self._has_bootstrapping(graph))
        self.assertTrue(check_dropped_levels_per_subgraph(graph))

    def test_e2e_conv_series(self):
        """Deep conv chain, requires BTP."""
        model = nn_modules.ConvSeries()
        graph, score = self._export_compile_and_deploy(model, (1, 32, 8, 8), 'conv_series')
        self.assertEqual(self._max_feature_level(graph), config.fhe_param.max_level)
        self.assertTrue(check_dropped_levels_per_subgraph(graph))

    def test_e2e_act_series(self):
        """Deep activation chain, requires BTP."""
        model = nn_modules.ActSeries()
        graph, score = self._export_compile_and_deploy(model, (1, 32, 8, 8), 'act_series')
        self.assertEqual(self._max_feature_level(graph), config.fhe_param.max_level)
        self.assertTrue(check_dropped_levels_per_subgraph(graph))

    @unittest.skip('check_dropped_levels_per_subgraph assertion failure after dev merge')
    def test_e2e_intertwined(self):
        """Multi-branch graph with add. Tests BTP with complex topology."""
        model = nn_modules.Intertwined()
        graph, score = self._export_compile_and_deploy(model, (1, 32, 8, 8), 'intertwined')
        self.assertTrue(check_dropped_levels_per_subgraph(graph))

    @unittest.skip('should be absorbed after compilation')
    def test_e2e_intertwined_with_coeff(self):
        """Multi-branch graph with add + mult_scalar. Tests BTP with scale ops."""
        model = nn_modules.IntertwinedWithCoeff()
        self._export_compile_and_deploy(model, (1, 32, 8, 8), 'intertwined_with_coeff')

    # ── Big-size tests (256×256 input) ──

    def test_e2e_single_avgpool_big_size(self):
        """Avgpool with big_size input (256×256), multiplexed style."""
        model = nn_modules.SingleAvgpool()
        graph, score = self._export_compile_and_deploy(
            model, (1, 5, 256, 256), 'single_avgpool_big_size', style='multiplexed'
        )
        self.assertTrue(check_feature_scale(graph))

    def test_e2e_single_conv_with_stride_big_size(self):
        """Conv stride=2 with big_size input (256×256), multiplexed."""
        model = nn_modules.SingleConv(2)
        graph, score = self._export_compile_and_deploy(
            model, (1, 32, 128, 128), 'single_conv_with_stride_big_size', style='multiplexed'
        )
        self.assertIsNotNone(graph)

    def test_e2e_dw_conv_big_size(self):
        """Depthwise conv stride=2 with big_size input (256x256), multiplexed."""
        model = nn_modules.DepthwiseConv(channels=32, stride=2)
        graph, score = self._export_compile_and_deploy(
            model, (1, 32, 256, 256), 'dw_conv_big_size', style='multiplexed'
        )
        self.assertIsNotNone(graph)

    def test_e2e_conv_series_with_stride(self):
        """Deep conv chain with strides, big_size (256×256), multiplexed."""
        model = nn_modules.ConvSeriesWithStride()
        graph, score = self._export_compile_and_deploy(
            model, (1, 32, 256, 256), 'conv_series_with_stride', style='multiplexed'
        )
        self.assertTrue(check_dropped_levels_per_subgraph(graph))

    # ── Operator-level migration from test_fhe_layers_hetero ──

    def test_e2e_conv_mch_s1(self):
        """Multi-channel conv, stride=1. Covers conv_mch_s1."""
        model = nn_modules.MultiChannelConv(in_channels=3, out_channels=16, stride=1)
        graph, score = self._export_compile_and_deploy(model, (1, 3, 32, 32), 'conv_mch_s1')
        self.assertIsNotNone(graph)

    def test_e2e_conv_mch_s2(self):
        """Multi-channel conv, stride=2. Covers conv_mch_s2."""
        model = nn_modules.MultiChannelConv(in_channels=3, out_channels=16, stride=2)
        graph, score = self._export_compile_and_deploy(model, (1, 3, 32, 32), 'conv_mch_s2')
        self.assertIsNotNone(graph)

    def test_e2e_depthwise_conv_s1(self):
        """Depthwise conv, stride=1. Covers dw_32ch_s1."""
        model = nn_modules.DepthwiseConv(channels=32, stride=1)
        graph, score = self._export_compile_and_deploy(model, (1, 32, 32, 32), 'depthwise_conv_s1')
        self.assertIsNotNone(graph)

    def test_e2e_depthwise_conv_s2(self):
        """Depthwise conv, stride=2. Covers dw_4ch_s2."""
        model = nn_modules.DepthwiseConv(channels=4, stride=2)
        graph, score = self._export_compile_and_deploy(model, (1, 4, 32, 32), 'depthwise_conv_s2')
        self.assertIsNotNone(graph)

    def test_e2e_two_fc(self):
        """Conv → Flatten → FC → FC. Covers fc_fc_0d."""
        model = nn_modules.ConvReshapeTwoFC()
        graph, score = self._export_compile_and_deploy(model, (1, 3, 32, 32), 'two_fc', do_constant_folding=True)
        self.assertIsNotNone(graph)

    def test_e2e_mux_conv_large_channel(self):
        """Large-channel conv triggering multiplexed. Covers mux_conv_varied_*."""
        model = nn_modules.MuxConvLargeChannel()
        graph, score = self._export_compile_and_deploy(
            model, (1, 32, 32, 32), 'mux_conv_large_channel', style='multiplexed'
        )
        self.assertIsNotNone(graph)

    def test_e2e_conv1d(self):
        """Conv1d E2E. Covers conv1d."""
        model = nn_modules.SingleConv1dE2E()
        graph, score = self._export_compile_and_deploy(model, (1, 4, 64), 'conv1d_e2e')
        self.assertIsNotNone(graph)

    # ── New layer migration from refactor/linghm ──

    def test_e2e_concat(self):
        """Two conv branches concatenated. Covers concat_layer."""
        model = nn_modules.ConcatModel()
        graph, score = self._export_compile_and_deploy(model, (1, 3, 32, 32), 'concat_e2e')
        self.assertIsNotNone(graph)

    def test_e2e_uneven_concat(self):
        """Two conv branches with uneven channels concatenated. Covers concat_layer uneven path."""
        model = nn_modules.UnevenConcatModel()
        graph, score = self._export_compile_and_deploy(model, (1, 3, 32, 32), 'uneven_concat')
        self.assertIsNotNone(graph)

    def test_e2e_conv_upsample(self):
        """Conv stride=2 + nearest upsample. Covers upsample_layer / upsample_nearest_layer."""
        model = nn_modules.ConvUpsampleE2E()
        graph, score = self._export_compile_and_deploy(
            model, (1, 32, 64, 64), 'conv_upsample_e2e', style='multiplexed', do_constant_folding=True
        )
        self.assertIsNotNone(graph)

    def test_e2e_avgpool_stride4(self):
        """Avgpool with stride=4. Covers avgpool2d_layer varied strides."""
        model = nn_modules.AvgpoolVariedStride(stride=4)
        graph, score = self._export_compile_and_deploy(model, (1, 32, 32, 32), 'avgpool_stride4', style='multiplexed')
        self.assertIsNotNone(graph)

    def test_multiple_outputs(self):
        """input."""
        model = nn_modules.MutipleOutputs()
        graph, score = self._export_compile_and_deploy(model, (1, 32, 64, 64), 'multiple_outputs', style='multiplexed')
        self.assertIsNotNone(graph)

    def test_multiple_inputs(self):
        """output."""
        model = nn_modules.MutipleInputs()
        input_sizes = [(1, 32, 64, 64)] * model.n_inputs
        graph, score = self._export_compile_and_deploy(model, input_sizes, 'multiple_inputs', style='multiplexed')
        self.assertIsNotNone(graph)

    def test_e2e_conv_concat_conv(self):
        """Shared-input concat structure with final add."""
        model = nn_modules.ConvConcatConv()
        graph, score = self._export_compile_and_deploy(model, (1, 16, 32, 32), 'conv_concat_conv')
        self.assertIsNotNone(graph)

    def test_e2e_general_avgpool(self):
        """General avgpool (kernel_size=3, stride=2) replaced with depthwise conv for FHE."""
        model = nn_modules.GeneralAvgpool(kernel_size=3, stride=2, padding=1)
        graph, score = self._export_compile_and_deploy(
            model,
            (1, 32, 8, 8),
            'general_avgpool',
            style='multiplexed',
        )
        self.assertIsNotNone(graph)


if __name__ == '__main__':
    unittest.main()
