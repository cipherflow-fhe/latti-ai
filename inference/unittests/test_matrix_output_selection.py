# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Preserve the original ONNX CLS selection through packed tokenwise layers."""
import json
import tempfile
import unittest
from pathlib import Path
import onnx
from onnx import helper, TensorProto
from training.run_compile import write_matrix_output_selections


class MatrixOutputSelectionTest(unittest.TestCase):
    def model(self, suffix):
        nodes = [helper.make_node('Linear', ['input', 'w', 'b'], ['selected'],
                                  name='pooler', domain='custom_domain', select_first_token=1), *suffix]
        return helper.make_model(helper.make_graph(nodes, 'cls',
            [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 44, 4])],
            [helper.make_tensor_value_info('logits', TensorProto.FLOAT, [1, 2])]))

    def test_selection_reaches_output_configs_through_tokenwise_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = self.model([helper.make_node('CustomTanh', ['selected'], ['tanh']),
                                helper.make_node('Linear', ['tanh', 'w2', 'b2'], ['logits'])])
            onnx.save(model, root/'model.onnx')
            for role in ['server', 'client', 'provisioner', 'runner']:
                (root/role).mkdir()
                (root/role/'task_config.json').write_text(json.dumps({'task_output_param':
                    {'logits': {'data_type': 'feature_mat', 'shape': [2, 44]}}}))
            write_matrix_output_selections(root/'model.onnx', root)
            for path in root.glob('*/task_config.json'):
                param = json.loads(path.read_text())['task_output_param']['logits']
                self.assertTrue(param['select_first_token'])
                self.assertEqual(param['shape'], [2, 44])

    def test_selection_cannot_cross_token_mixing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            onnx.save(self.model([helper.make_node('CustomMultiHeadAttention', ['selected'], ['logits'])]),
                      root/'model.onnx')
            with self.assertRaisesRegex(ValueError, 'Cannot defer CLS selection'):
                write_matrix_output_selections(root/'model.onnx', root)


if __name__ == '__main__':
    unittest.main()
