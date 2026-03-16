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
import math
import sys
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
sys.path.append(str(script_dir.parent))
sys.path.append(str(script_dir.parent.parent))

from nn_tools.export import export_to_onnx
from model_export.onnx_to_json import onnx_to_json
from pipeline import init_config_with_args, run_pipeline
from components import (
    LayerAbstractGraph,
    FeatureNode,
    config,
    ComputeNode,
    ConvComputeNode,
    UpsampleComputeNode,
    SpatialComputeNode,
    ActivationComputeNode,
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


class CompilerTestBase(unittest.TestCase):
    temp_onnx_path = script_dir / 'temp.onnx'
    temp_json_path = script_dir / 'temp.json'

    def _export_and_compile(
        self,
        model,
        input_size,
        style='ordinary',
        graph_type='btp',
        **export_kwargs,
    ):
        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=input_size,
            dynamic_batch=False,
            save_h5=False,
            **export_kwargs,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, style)
        init_config_with_args(style=style, graph_type=graph_type)
        graph, score = run_pipeline(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )
        return graph, score


class TestCompiler(CompilerTestBase):
    def test_single_conv(self):
        model = nn_modules.SingleConv()
        graph, score = self._export_and_compile(model, (1, 32, 64, 64))

        self.assertEqual(
            max(graph.dag.nodes[feature]['level'] for feature in graph.dag.nodes if isinstance(feature, FeatureNode)), 1
        )

    def test_single_act(self):
        model = nn_modules.SingleAct()
        graph, score = self._export_and_compile(model, (1, 32, 64, 64))

        self.assertEqual(
            max(graph.dag.nodes[feature]['level'] for feature in graph.dag.nodes if isinstance(feature, FeatureNode)), 3
        )

    def test_single_avgpool(self):
        model = nn_modules.SingleAvgpool()
        graph, score = self._export_and_compile(model, (1, 32, 64, 64))
        self.assertEqual(check_feature_scale(graph), True)

    def test_single_avgpool_big_size(self):
        model = nn_modules.SingleAvgpool()
        graph, score = self._export_and_compile(model, (1, 32, 256, 256))
        self.assertEqual(check_feature_scale(graph), True)

    def test_single_maxpool(self):
        model = nn_modules.SingleMaxpool()
        graph, score = self._export_and_compile(model, (1, 32, 64, 64))

    def test_single_dense(self):
        model = nn_modules.SingleDense()
        graph, score = self._export_and_compile(model, (1, 64))

    def test_single_reshape(self):
        model = nn_modules.SingleReshape()
        graph, score = self._export_and_compile(model, (1, 16, 4, 4))

    def test_single_mult_ceoff(self):
        model = nn_modules.SingleMultCoeff()
        graph, score = self._export_and_compile(model, (1, 16, 4, 4))
        self.assertEqual(check_feature_scale(graph), True)

    def test_single_add(self):
        model = nn_modules.SingleAdd()
        graph, score = self._export_and_compile(model, [(1, 32, 64, 64), (1, 32, 64, 64)], input_names=['x0', 'x1'])

    def test_conv_with_batchnorms(self):
        model = nn_modules.ConvWithBatchNorms()
        graph, score = self._export_and_compile(model, (1, 32, 64, 64))

        self.assertEqual(
            max(graph.dag.nodes[feature]['level'] for feature in graph.dag.nodes if isinstance(feature, FeatureNode)), 1
        )

    def test_conv_series(self):
        model = nn_modules.ConvSeries()
        graph, score = self._export_and_compile(model, (1, 32, 64, 64))

        self.assertEqual(
            max(graph.dag.nodes[feature]['level'] for feature in graph.dag.nodes if isinstance(feature, FeatureNode)),
            config.fhe_param.max_level,
        )
        self.assertEqual(check_dropped_levels_per_subgraph(graph), True)

    def test_act_series(self):
        model = nn_modules.ActSeries()
        graph, score = self._export_and_compile(model, (1, 32, 64, 64))

        self.assertEqual(
            max(graph.dag.nodes[feature]['level'] for feature in graph.dag.nodes if isinstance(feature, FeatureNode)),
            config.fhe_param.max_level,
        )
        self.assertEqual(check_dropped_levels_per_subgraph(graph), True)

    def test_conv_series_with_stride(self):
        model = nn_modules.ConvSeriesWithStride()
        graph, score = self._export_and_compile(model, (1, 32, 256, 256), style='multiplexed')

        self.assertEqual(
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

    def test_resnet_basic_block(self):
        model = nn_modules.ResNetBasicBlock(32, 32)
        graph, score = self._export_and_compile(model, (1, 32, 64, 64))
        self.assertEqual(check_level_cost(graph), True)
        self.assertEqual(check_multi_input_level_skip_aligned(graph), True)
        self.assertEqual(check_dropped_levels_per_subgraph(graph), True)

    def test_resnet_20(self):
        import torch.nn as nn
        from training.nn_tools.activations import RangeNormPoly2d, Simple_Polyrelu
        from training.nn_tools import (
            export_to_onnx,
            fuse_and_export_h5,
            replace_activation_with_poly,
            replace_maxpool_with_avgpool,
        )
        from resnet import resnet20

        model = resnet20()

        replace_maxpool_with_avgpool(model)
        replace_activation_with_poly(
            model,
            old_cls=nn.ReLU,
            new_module_factory=Simple_Polyrelu,
            upper_bound=3.0,
            degree=4,
        )

        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 3, 32, 32]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'multiplexed')

        init_config_with_args(style='multiplexed', graph_type='btp')
        graph, score = run_pipeline(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )
        self.assertEqual(check_level_cost(graph), True)
        self.assertEqual(check_multi_input_level_skip_aligned(graph), True)
        self.assertEqual(check_dropped_levels_per_subgraph(graph), True)

    def test_mismatched_scale(self):
        model = nn_modules.MismatchedScale()
        graph, score = self._export_and_compile(model, (1, 32, 64, 64))

    def test_intertwined(self):
        model = nn_modules.Intertwined()
        graph, score = self._export_and_compile(model, (1, 32, 64, 64))
        self.assertEqual(check_dropped_levels_per_subgraph(graph), True)

    def test_intertwined_with_coeff(self):
        model = nn_modules.IntertwinedWithCoeff()
        graph, score = self._export_and_compile(model, (1, 32, 64, 64))
        self.assertEqual(check_dropped_levels_per_subgraph(graph), True)

    @unittest.skip('Not supported yet')
    def test_multiple_inputs(self):
        model = nn_modules.MutipleInputs()
        graph, score = self._export_and_compile(model, (1, 32, 64, 64))

    @unittest.skip('Not supported yet')
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

        init_config_with_args(style='ordinary', graph_type='btp')
        from pipeline import prepare_graph
        from transforms import split_graph_to_linear_subgraph

        raw_graph = LayerAbstractGraph.from_json(self.temp_json_path)
        pt_graph = prepare_graph(raw_graph)
        subs = split_graph_to_linear_subgraph(pt_graph.dag)
        self.assertEqual(len(subs), 2)

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

    def test_conv_reshape_dense(self):
        model = nn_modules.ConvReshapeAndDense()
        graph, score = self._export_and_compile(model, (1, 3, 32, 32), do_constant_folding=True)
        res = None
        for node in graph.dag.nodes:
            if isinstance(node, ComputeNode) and node.layer_type == 'reshape':
                input = list(graph.dag.predecessors(node))[0]
                output = list(graph.dag.successors(node))[0]
                if (
                    graph.dag.nodes[output]['virtual_shape'][0] == 16
                    and graph.dag.nodes[output]['virtual_skip'][0] == 2
                ):
                    res = True
                    break
        self.assertEqual(res, True)

    def test_conv_avgpool_reshape_dense(self):
        model = nn_modules.ConvAvgpoolReshapeAndDense()
        graph, score = self._export_and_compile(model, (1, 3, 64, 64), style='multiplexed', do_constant_folding=True)
        res = None
        from components import PoolComputeNode

        for node in graph.dag.nodes:
            if isinstance(node, PoolComputeNode):
                input = list(graph.dag.predecessors(node))[0]
                output = list(graph.dag.successors(node))[0]
                if output.shape == input.shape and graph.dag.nodes[output]['skip'] == graph.dag.nodes[input]['skip']:
                    res = True
                    break
        self.assertEqual(res, True)
        self.assertEqual(check_dropped_levels_per_subgraph(graph), True)

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
        init_config_with_args(style='ordinary', graph_type='btp')
        with self.assertRaisesRegex(ValueError, r'Relu2d is not supported in current mode'):
            run_pipeline(
                num_experiments=1,
                input_file_path=self.temp_json_path,
                output_dir=script_dir,
                temperature=0.0,
                num_workers=1,
            )


class TestPolyDegree(CompilerTestBase):
    """Verify that models with specific depth profiles select the correct poly_n and mode.

    Level costs (ordinary style):
        Conv  (stride=1): 1
        Act   (RangeNormPoly2d, order=4): ceil(log2(4)) + 1 = 3

    Non-BTP poly_n → max_level map: {8192: 5, 16384: 9, 32768: 17, 65536: 33}
    BTP poly_n: 65536, max_level: 9 (bootstrapping inserted between segments)
    """

    def test_no_btp_poly_n_8192(self):
        """1 Conv + 1 Act = 4 levels; fits poly_n=8192 (max_level=5); no-BTP mode."""
        model = nn_modules.PolyDegreeN8192()
        graph, score = self._export_and_compile(model, (1, 32, 8, 8))
        # No-BTP selects the smallest poly_n whose max_level accommodates the graph.
        self.assertEqual(config.fhe_param.poly_modulus_degree, 8192)
        self.assertEqual(config.fhe_param.max_level, 5)
        self.assertIsNotNone(graph)
        # No bootstrapping nodes should be present.
        self.assertFalse(any(isinstance(n, ComputeNode) and n.layer_type == 'bootstrapping' for n in graph.dag.nodes))

    def test_no_btp_poly_n_16384(self):
        """3 Conv + 1 Act = 6 levels; exceeds poly_n=8192 (max 5), fits poly_n=16384 (max 9); no-BTP mode."""
        model = nn_modules.PolyDegreeN16384()
        graph, score = self._export_and_compile(model, (1, 32, 8, 8))
        self.assertEqual(config.fhe_param.poly_modulus_degree, 16384)
        self.assertEqual(config.fhe_param.max_level, 9)
        self.assertIsNotNone(graph)
        self.assertFalse(any(isinstance(n, ComputeNode) and n.layer_type == 'bootstrapping' for n in graph.dag.nodes))
        self.assertEqual(check_dropped_levels_per_subgraph(graph), True)

    def test_no_btp_poly_n_32768(self):
        """4 Conv + 2 Act = 10 levels; exceeds poly_n=16384 (max 9), fits poly_n=32768 (max 17); no-BTP mode."""
        model = nn_modules.PolyDegreeN32768()
        graph, score = self._export_and_compile(model, (1, 32, 8, 8))
        self.assertEqual(config.fhe_param.poly_modulus_degree, 32768)
        self.assertEqual(config.fhe_param.max_level, 17)
        self.assertIsNotNone(graph)
        self.assertFalse(any(isinstance(n, ComputeNode) and n.layer_type == 'bootstrapping' for n in graph.dag.nodes))
        self.assertEqual(check_dropped_levels_per_subgraph(graph), True)

    def test_no_btp_poly_n_65536(self):
        """6 Conv + 4 Act = 18 levels; exceeds poly_n=32768 (max 17), fits poly_n=65536 (max 33); no-BTP mode."""
        model = nn_modules.PolyDegreeN65536NoBtp()
        graph, score = self._export_and_compile(model, (1, 32, 8, 8))
        self.assertEqual(config.fhe_param.poly_modulus_degree, 65536)
        self.assertEqual(config.fhe_param.max_level, 33)
        self.assertIsNotNone(graph)
        self.assertFalse(any(isinstance(n, ComputeNode) and n.layer_type == 'bootstrapping' for n in graph.dag.nodes))
        self.assertEqual(check_dropped_levels_per_subgraph(graph), True)

    def test_btp_poly_n_65536(self):
        """4 Conv + 10 Act = 34 levels; exceeds all non-BTP limits → BTP mode, poly_n=65536, max_level=9."""
        model = nn_modules.PolyDegreeNBtp()
        graph, score = self._export_and_compile(model, (1, 32, 8, 8))
        self.assertEqual(config.fhe_param.poly_modulus_degree, 65536)
        self.assertEqual(config.fhe_param.max_level, 9)
        self.assertIsNotNone(graph)
        # BTP mode must insert bootstrapping nodes.
        self.assertTrue(any(isinstance(n, ComputeNode) and n.layer_type == 'bootstrapping' for n in graph.dag.nodes))
        self.assertEqual(check_dropped_levels_per_subgraph(graph), True)


if __name__ == '__main__':
    unittest.main()
