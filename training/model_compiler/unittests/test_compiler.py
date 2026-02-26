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
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
sys.path.append(str(script_dir.parent))
sys.path.append(str(script_dir.parent.parent))

from nn_tools.export import export_to_onnx
from model_export.onnx_to_json import onnx_to_json
from graph_partition_dp import init_config_with_args, run_parallel, config
from components import LayerAbstractGraph, FeatureNode
import nn_modules


class TestCompiler(unittest.TestCase):
    temp_onnx_path = script_dir / 'temp.onnx'
    temp_json_path = script_dir / 'temp.json'

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
        graph, score = run_parallel(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
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
        graph, score = run_parallel(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )

        self.assertEqual(
            max(graph.dag.nodes[feature]['level'] for feature in graph.dag.nodes if isinstance(feature, FeatureNode)), 3
        )

    def test_nn2(self):
        nn = nn_modules.NN2()
        export_to_onnx(
            nn,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 64, 64]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

        init_config_with_args(poly_n=65536, style='ordinary', graph_type='btp')
        graph, score = run_parallel(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )

        self.assertEqual(
            max(graph.dag.nodes[feature]['level'] for feature in graph.dag.nodes if isinstance(feature, FeatureNode)),
            config['MAX_LEVEL'],
        )

    def test_nn3(self):
        nn = nn_modules.NN3()
        export_to_onnx(
            nn,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 64, 64]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

        init_config_with_args(poly_n=65536, style='ordinary', graph_type='btp')
        graph, score = run_parallel(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )

        self.assertEqual(
            max(graph.dag.nodes[feature]['level'] for feature in graph.dag.nodes if isinstance(feature, FeatureNode)),
            config['MAX_LEVEL'],
        )

    def test_nn4(self):
        nn = nn_modules.NN4()
        export_to_onnx(
            nn,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 256, 256]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'multiplexed')

        init_config_with_args(poly_n=65536, style='multiplexed', graph_type='btp')
        graph, score = run_parallel(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )

        self.assertEqual(
            max(graph.dag.nodes[feature]['level'] for feature in graph.dag.nodes if isinstance(feature, FeatureNode)),
            config['MAX_LEVEL'],
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
        graph, score = run_parallel(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )

    def test_wrong_padding(self):
        nn = nn_modules.WrongPadding()
        export_to_onnx(
            nn,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 64, 64]),
            dynamic_batch=False,
            save_h5=False,
        )

        with self.assertRaisesRegex(ValueError, r'Unsupported padding value: \[0, 0, 0, 0\]'):
            onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

    def test_wrong_dilation(self):
        nn = nn_modules.WrongDilation()
        export_to_onnx(
            nn,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 64, 64]),
            dynamic_batch=False,
            save_h5=False,
        )

        with self.assertRaisesRegex(ValueError, r'Unsupported dilation value: \[2, 2\]'):
            onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

    def test_wrong_groups(self):
        nn = nn_modules.WrongGroups()
        export_to_onnx(
            nn,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 64, 64]),
            dynamic_batch=False,
            save_h5=False,
        )

        with self.assertRaisesRegex(ValueError, r'Unsupported groups value: 2'):
            onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')


if __name__ == '__main__':
    unittest.main()
