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
import sys
import os
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..'))
sys.path.append(os.path.join(script_dir, '..', '..'))

from nn_tools.export import export_to_onnx
from model_export.onnx_to_json import onnx_to_json
from graph_partition_dp import init_config_with_args, compile_model_btp, run_parallel
from components import LayerAbstractGraph, FeatureNode
import nn_modules


class TestCompiler(unittest.TestCase):
    temp_onnx_path = os.path.join(script_dir, 'temp.onnx')
    temp_json_path = os.path.join(script_dir, 'temp.json')

    def test_nn0(self):
        nn = nn_modules.NN0()
        export_to_onnx(
            nn,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 64, 64]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

        init_config_with_args(poly_n=65536, style='ordinary', graph_type='btp')
        score, graph = compile_model_btp(
            input_file_path=Path(self.temp_json_path),
            output_dir=Path(script_dir),
        )

        self.assertEqual(
            max(graph.dag.nodes[feature]['level'] for feature in graph.dag.nodes if isinstance(feature, FeatureNode)), 1
        )

    def test_nn1(self):
        nn = nn_modules.NN1()
        export_to_onnx(
            nn,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 64, 64]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

        init_config_with_args(poly_n=65536, style='ordinary', graph_type='btp')
        score, graph = compile_model_btp(
            input_file_path=Path(self.temp_json_path),
            output_dir=Path(script_dir),
        )

        self.assertEqual(
            max(graph.dag.nodes[feature]['level'] for feature in graph.dag.nodes if isinstance(feature, FeatureNode)), 3
        )

    def test_resnet_basic_block(self):
        nn = nn_modules.ResNetBasicBlock(32, 32)
        export_to_onnx(
            nn,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 64, 64]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

        init_config_with_args(poly_n=65536, style='ordinary', graph_type='btp')
        # run_parallel(
        #     num_experiments=1,
        #     input_file_path=Path(temp_json_path),
        #     output_dir=Path(script_dir),
        #     temperature=1.0,
        #     num_workers=1,
        # )
        score, graph = compile_model_btp(
            input_file_path=Path(self.temp_json_path),
            output_dir=Path(script_dir),
        )
        print(graph)


if __name__ == '__main__':
    unittest.main()
