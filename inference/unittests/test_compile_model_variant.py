# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""The exported operator schema selects BERT semantics without a config label."""
import json
import tempfile
import unittest
from pathlib import Path

import onnx
from onnx import helper

from training.run_compile import infer_model_type


class CompileModelVariantTest(unittest.TestCase):
    def test_onnx_attention_variants_and_tokenwise_subgraphs(self):
        for op, attrs, expected in (
            ('CustomMultiHeadAttention', {'exp_coefficients': '1,1'}, 'bert'),
            ('CustomMultiHeadAttention', {'attn_act': 'polyactrn'}, ''),
            ('CustomGELU', {}, 'bert'),
            ('CustomTanh', {}, 'bert'),
            ('Conv', {}, ''),
        ):
            with self.subTest(op=op, attrs=attrs), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'arbitrary_name.onnx'
                graph = helper.make_graph([helper.make_node(op, ['x'], ['y'], **attrs)], 'g', [], [])
                onnx.save(helper.make_model(graph), path)
                self.assertEqual(infer_model_type(path), expected)

    def test_preconverted_json_preserves_operator_semantics(self):
        for layer, expected in (
            ({'type': 'CustomMultiHeadAttention', 'model_type': 'bert'}, 'bert'),
            ({'type': 'CustomGELU'}, 'bert'),
            ({'type': 'CustomMultiHeadAttention'}, ''),
            ({'type': 'conv2d'}, ''),
        ):
            with self.subTest(layer=layer), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'pt.json'
                path.write_text(json.dumps({'layer': {'node': layer}}))
                self.assertEqual(infer_model_type(path), expected)


if __name__ == '__main__':
    unittest.main()
