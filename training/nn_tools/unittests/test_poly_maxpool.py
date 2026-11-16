"""Plaintext replacement, gradients and ONNX/compiler boundary regression tests."""
import io
import unittest

import numpy as np
import onnx
import torch
from torch import nn

from training.nn_tools import PolyMaxPool2d, prepare_for_fhe, replace_maxpool_with_poly
from training.model_export.operations import FeatureNode, format_id
from training.model_export.operations.MaxPool import MaxPoolComputeNode
from inference.model_generator.layers.poly_maxpool_stages import _maxpool_stage_coefficients


class PolyMaxPoolTest(unittest.TestCase):
    def test_plaintext_matches_runtime_rounds_and_padding(self):
        rng = np.random.default_rng(42)
        for alpha in (7, 14):
            # Negative inputs expose incorrect zero padding at image edges.
            x = rng.uniform(-.7, -.05, (1, 2, 6, 8))
            pool = PolyMaxPool2d(alpha=alpha, upper_bound=.1, B=1.).double().eval()
            pool.running_max.fill_(.8)
            result = pool(torch.from_numpy(x)).detach().numpy()
            reference = np.empty((1, 2, 3, 4))
            for row in range(3):
                for col in range(4):
                    taps = []
                    for kr in range(3):
                        for kc in range(3):
                            r, c = 2*row+kr-1, 2*col+kc-1
                            if not (0 <= r < 6 and 0 <= c < 8):
                                r, c = 2*row, 2*col
                            taps.append(x[..., r, c] * .1 / .8)
                    depth = 0
                    while len(taps) > 1:
                        reduced = []
                        stages = _maxpool_stage_coefficients(alpha, depth, 1.)['stages']
                        for i in range(0, len(taps)-1, 2):
                            a, b = taps[i:i+2]
                            sign = a-b
                            for stage in stages:
                                sign = sum(coef * sign**(2*j+1) for j, coef in enumerate(stage))
                            reduced.append(a+b+(a-b)*sign)
                        if len(taps) % 2:
                            reduced.append(2*taps[-1])
                        taps = reduced
                        depth += 1
                    reference[..., row, col] = taps[0] * .8/.1/2**depth
            np.testing.assert_allclose(result, reference, rtol=1e-9, atol=1e-9)
            self.assertGreater(np.max(np.abs(result - nn.functional.max_pool2d(torch.from_numpy(x), 3, 2, 1).numpy())), 1e-6)

    def test_prepare_replaces_and_allows_finetuning(self):
        model = nn.Sequential(nn.MaxPool2d(3, 2, 1))
        prepare_for_fhe(model, maxpool_alpha=14, maxpool_upper_bound=.1, maxpool_B=1.)
        self.assertIsInstance(model[0], PolyMaxPool2d)
        x = torch.linspace(-.5, .5, 64, dtype=torch.double).reshape(1, 1, 8, 8).requires_grad_()
        model.double()
        model(x).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertAlmostEqual(model[0].running_max.item(), .5)
        model.eval()
        bound = model[0].running_max.clone()
        torch.autograd.gradcheck(model, (x,), atol=1e-4)
        torch.testing.assert_close(model[0].running_max, bound)

    def test_export_preserves_approximation_and_bound(self):
        pool = PolyMaxPool2d(alpha=14, upper_bound=.1).eval()
        pool.running_max.fill_(2.5)
        data = io.BytesIO()
        torch.onnx.export(pool, torch.zeros(1, 2, 8, 8), data, opset_version=13, dynamo=False,
                          input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
        graph = onnx.load_model_from_string(data.getvalue()).graph
        node = next(n for n in graph.node if n.op_type == 'PolyMaxPool2d')
        self.assertEqual(node.domain, 'nn_tools')
        features = {format_id(name): FeatureNode(format_id(name), 2, 2, 1., [1, 1], 'param0', shape)
                    for name, shape in ((node.input[0], [8, 8]), (node.output[0], [4, 4]))}
        layer = MaxPoolComputeNode.from_onnx_node(node, features).to_json()
        self.assertEqual(layer['alpha'], 14)
        self.assertAlmostEqual(layer['upper_bound'], .1)
        self.assertEqual(layer['running_max'], 2.5)

    def test_replacement_preserves_eval_and_rejects_unsupported_semantics(self):
        model = nn.Sequential(nn.MaxPool2d(3, 2, 1)).eval()
        replace_maxpool_with_poly(model)
        self.assertFalse(model[0].training)
        for options in ({'ceil_mode': True}, {'return_indices': True}, {'dilation': 2}):
            with self.assertRaises(ValueError):
                replace_maxpool_with_poly(nn.Sequential(nn.MaxPool2d(3, 2, 1, **options)))
