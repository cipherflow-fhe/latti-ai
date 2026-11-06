# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""PCM BERT parameter semantics and compiler JSON regression checks."""

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import ComputeNode, FeatureNode, LayerAbstractGraph, config
from transforms import _append_softmax_normalize
from training.nn_tools.feature_mat_h5 import _export_add_pt, _export_parcpmm, _export_pdmupperpoly


class TestPcmBert(unittest.TestCase):
    def test_scaled_softmax_preserves_fixed_newton_iterations(self):
        previous = config.__dict__.copy()
        self.addCleanup(lambda: (config.__dict__.clear(), config.__dict__.update(previous)))
        for style in ('par_block_col_major', 'par_diagonal_pack'):
            config.mat_pack_style = style
            for signal_scale, values_scale in ((1, 1), (2, 1), (512, 1), (512, 512), (65536, 65536)):
                with self.subTest(style=style, signal_scale=signal_scale):
                    # Tiny thor-wide denominators do not converge to 1/b in
                    # 15 iterations. Compare with those exact finite iterations.
                    values = np.array([[1e-8, 2e-8, 4e-8], [0.01, 0.02, 0.03]])
                    graph = LayerAbstractGraph()
                    x = FeatureNode('x', 2, 1, shape=[2, 3])
                    graph.dag.add_node(x, level=12)
                    output = _append_softmax_normalize(
                        graph, 'normalize', x, signal_scale, 15, values_scale=values_scale)
                    actual = {x: values * values_scale}
                    for node in nx.topological_sort(graph.dag):
                        if node is x:
                            continue
                        inputs = sorted(graph.dag.predecessors(node),
                                        key=lambda p: graph.dag.edges[p, node].get('input_index', 0))
                        args = [actual[p] for p in inputs]
                        if isinstance(node, FeatureNode):
                            result = args[0]
                        elif node.layer_type.endswith('headcolsum'):
                            result = np.repeat(args[0].sum(axis=-1, keepdims=True), 3, axis=-1)
                        elif node.layer_type.endswith('gamma'):
                            result = args[0] * node.scalar_value
                        elif node.layer_type.endswith('invinit'):
                            result = 2 - args[0]
                        elif node.layer_type in ('pcmgeneralpoly', 'pdmupperpoly'):
                            result = np.polynomial.polynomial.polyval(args[0], node.coefficients)
                        elif node.layer_type.endswith('inviter'):
                            result = args[0] * (2 - args[1] * args[0])
                        elif node.layer_type.endswith('ctmul'):
                            result = args[0] * args[1]
                        else:
                            self.fail(node.layer_type)
                        actual[node] = result
                    denominator = values.sum(axis=-1, keepdims=True)
                    inverse = np.ones_like(denominator)
                    for _ in range(15):
                        inverse *= 2 - denominator * inverse
                    np.testing.assert_allclose(actual[output], values * inverse, rtol=2e-14, atol=1e-16)

    def test_bert_weight_direction_and_single_q_scaling(self):
        # Include a nonsymmetric square Q, actual FFN expansion and two logits.
        for n_in, n_out, multiplier in ((64, 64, 1 / math.sqrt(64)), (768, 3072, 1), (768, 2, 1)):
            with self.subTest(n_in=n_in, n_out=n_out):
                weight = np.arange(n_out * n_in, dtype='float64').reshape(n_out, n_in) / (n_in * n_out)
                bias = np.arange(n_out, dtype='float64') / n_out
                layer = dict(type='parcpmm', feature_input=['x'], feature_output=['y'],
                             weight_path='weight', bias_path='bias', weight_transposed=True,
                             weight_multiplier=multiplier, bias_multiplier=multiplier)
                features = dict(x={'shape': [44, n_in], 'data_type': 'feature_mat'},
                                y={'shape': [44, n_out], 'data_type': 'feature_mat'})
                out = {}
                _export_parcpmm('linear', layer, features, dict(weight=weight, bias=bias), out)
                np.testing.assert_array_equal(out['weight'], weight.T * multiplier)
                np.testing.assert_array_equal(out['bias'], bias * multiplier)
                x = np.linspace(-0.1, 0.2, n_in)
                np.testing.assert_allclose(x @ out['weight'] + out['bias'],
                                           (x @ weight.T + bias) * multiplier, atol=1e-12)

    def test_vit_fused_qkv_direction_and_scaling(self):
        width, heads = 12, 3
        weight = np.arange(width * 3 * width, dtype='float64').reshape(width, 3 * width)
        bias = np.arange(3 * width, dtype='float64')
        features = dict(x={'shape': [7, width], 'head_shape': [7, width // heads], 'data_type': 'feature_mat'},
                        y={'shape': [7, width], 'head_shape': [7, width // heads], 'data_type': 'feature_mat'})
        for part, index in (('q', 0), ('k', 1), ('v', 2)):
            with self.subTest(part=part):
                layer = dict(type='parcpmm', feature_input=['x'], feature_output=['y'],
                             weight_path=f'attn.{part}.weight', bias_path=f'attn.{part}.bias')
                out = {}
                _export_parcpmm('attn', layer, features,
                                {'attn.qkv.weight': weight, 'attn.qkv.bias': bias}, out)
                factor = 1 / math.sqrt(width // heads) if part == 'q' else 1
                np.testing.assert_array_equal(out[layer['weight_path']], np.split(weight, 3, axis=1)[index] * factor)
                np.testing.assert_array_equal(out[layer['bias_path']], np.split(bias, 3)[index] * factor)

    def test_pcm_constants_and_coefficient_axis(self):
        features = dict(x={'shape': [44, 3072], 'data_type': 'feature_mat'})
        out = {}
        layer = dict(type='pcmgeneralpoly', feature_input=['x'], order=3,
                     weight_path='coeff', coefficients=[0.2, -0.3, 0, 0.4])
        _export_pdmupperpoly('poly', layer, features, out)
        np.testing.assert_array_equal(out['coeff'], np.repeat(np.array([[0.2], [-0.3], [0], [0.4]]), 3072, axis=1))
        layer = dict(type='pcm_add_pt', feature_input=['x'], weight_path='constant', value=-0.25)
        _export_add_pt('add', layer, features, {}, out)
        np.testing.assert_array_equal(out['constant'], np.full((44, 3072), -0.25))

    def test_json_round_trip_and_ordered_binary_inputs(self):
        unary = ('pcmgeneralpoly', 'pcmmulsquare', 'pcmheadcolsum', 'pcminvinit', 'pcm_add_pt')
        binary = ('pcminviter', 'pcmctmul', 'pcmpolymultct')
        layers = {}
        features = {}
        for typ in unary + binary + ('pcminit', 'pcmgs', 'pcmstats'):
            a, b, y = (typ + suffix for suffix in ('_a', '_b', '_y'))
            for key in (a, b, y):
                features[key] = dict(dim=2, shape=[11, 54], head_shape=[11, 9],
                                     data_type='feature_mat', ckks_parameter_id='param0')
            layer = dict(type=typ, feature_input=[b, a] if typ in binary + ('pcmgs',) else [a],
                         feature_output=[y])
            if typ == 'pcmgeneralpoly':
                layer.update(order=3, coefficients=[0.2, -0.3, 0, 0.4], weight_path='poly', coeffs_path='poly')
            elif typ == 'pcm_add_pt':
                layer.update(value=-0.25, weight_path='constant')
            elif typ == 'pcminit':
                layer.update(coeffs=[0.8, -0.2, 0.03], c0=0.8, c1=-0.2, c2=0.03)
            elif typ == 'pcmstats':
                layer.update(epsilon=1e-5, inv_var=0.7, input_scale=0.9, min_var=0.1, max_var=2.0)
            layers[typ] = layer
        with tempfile.TemporaryDirectory(prefix='pcm-bert-json-', dir='/tmp') as tmp:
            src, dst = Path(tmp) / 'source.json', Path(tmp) / 'roundtrip.json'
            src.write_text(json.dumps(dict(feature=features, layer=layers)))
            graph = LayerAbstractGraph.from_json(src)
            for node in graph.dag:
                if isinstance(node, FeatureNode):
                    graph.dag.nodes[node].update(level=6, pack_num=1)
                elif isinstance(node, ComputeNode):
                    # Reverse insertion order: serialization must use input_index.
                    edges = [(p, dict(graph.dag.edges[p, node])) for p in graph.dag.predecessors(node)]
                    for p, _ in edges:
                        graph.dag.remove_edge(p, node)
                    for p, attrs in reversed(edges):
                        graph.dag.add_edge(p, node, **attrs)
            graph.to_json({}, str(dst))
            result = json.loads(dst.read_text())
            for key, layer in layers.items():
                for attr, value in layer.items():
                    self.assertEqual(result['layer'][key][attr], value, (key, attr))
            for key, feature in features.items():
                self.assertEqual(result['feature'][key]['head_shape'], feature['head_shape'])


if __name__ == '__main__':
    unittest.main()
