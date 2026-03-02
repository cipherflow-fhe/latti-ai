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
from pipeline import init_config_with_args, run_pipeline
from components import LayerAbstractGraph, FeatureNode, config
import nn_modules
from processor import check_level_cost, check_multi_input_level_skip_aligned


class TestCompiler(unittest.TestCase):
    temp_onnx_path = script_dir / 'temp.onnx'
    temp_json_path = script_dir / 'temp.json'

    def test_single_conv(self):
        model = nn_modules.SingleConv()
        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 64, 64]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

        init_config_with_args(poly_n=65536, style='ordinary', graph_type='btp')
        graph, score = run_pipeline(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )

        self.assertEqual(
            max(graph.dag.nodes[feature]['level'] for feature in graph.dag.nodes if isinstance(feature, FeatureNode)), 1
        )

    def test_single_act(self):
        model = nn_modules.SingleAct()
        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 64, 64]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

        init_config_with_args(poly_n=65536, style='ordinary', graph_type='btp')
        graph, score = run_pipeline(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )

        self.assertEqual(
            max(graph.dag.nodes[feature]['level'] for feature in graph.dag.nodes if isinstance(feature, FeatureNode)), 3
        )

    def test_single_avgpool(self):
        model = nn_modules.SingleAvgpool()
        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 64, 64]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

        init_config_with_args(poly_n=65536, style='ordinary', graph_type='btp')
        graph, score = run_pipeline(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )

    def test_single_maxpool(self):
        model = nn_modules.SingleMaxpool()
        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 64, 64]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

        init_config_with_args(poly_n=65536, style='ordinary', graph_type='btp')
        graph, score = run_pipeline(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )

    def test_single_dense(self):
        model = nn_modules.SingleDense()
        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 64]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

        init_config_with_args(poly_n=65536, style='ordinary', graph_type='btp')
        graph, score = run_pipeline(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )

    def test_single_reshape(self):
        model = nn_modules.SingleReshape()
        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 16, 4, 4]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

        init_config_with_args(poly_n=65536, style='ordinary', graph_type='btp')
        graph, score = run_pipeline(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )

    def test_single_mult_ceoff(self):
        model = nn_modules.SingleMultCoeff()
        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 16, 4, 4]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

        init_config_with_args(poly_n=65536, style='ordinary', graph_type='btp')
        graph, score = run_pipeline(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )

    def test_single_add(self):
        model = nn_modules.SingleAdd()
        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=[(1, 32, 64, 64), (1, 32, 64, 64)],
            input_names=['x0', 'x1'],
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

        init_config_with_args(poly_n=65536, style='ordinary', graph_type='btp')
        graph, score = run_pipeline(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )

    def test_conv_series(self):
        model = nn_modules.ConvSeries()
        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 64, 64]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

        init_config_with_args(poly_n=65536, style='ordinary', graph_type='btp')
        graph, score = run_pipeline(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )

        self.assertEqual(
            max(graph.dag.nodes[feature]['level'] for feature in graph.dag.nodes if isinstance(feature, FeatureNode)),
            config.max_level,
        )

    def test_act_series(self):
        model = nn_modules.ActSeries()
        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 64, 64]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

        init_config_with_args(poly_n=65536, style='ordinary', graph_type='btp')
        graph, score = run_pipeline(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )

        self.assertEqual(
            max(graph.dag.nodes[feature]['level'] for feature in graph.dag.nodes if isinstance(feature, FeatureNode)),
            config.max_level,
        )

    def test_conv_series_with_stride(self):
        model = nn_modules.ConvSeriesWithStride()
        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 256, 256]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'multiplexed')

        init_config_with_args(poly_n=65536, style='multiplexed', graph_type='btp')
        graph, score = run_pipeline(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )

        self.assertEqual(
            max(graph.dag.nodes[feature]['level'] for feature in graph.dag.nodes if isinstance(feature, FeatureNode)),
            config.max_level,
        )

    def test_mult_coeff_series(self):
        model = nn_modules.MultCoeffSeries()
        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 256, 256]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'multiplexed')

        init_config_with_args(poly_n=65536, style='multiplexed', graph_type='btp')
        graph, score = run_pipeline(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )

        self.assertEqual(
            max(graph.dag.nodes[feature]['level'] for feature in graph.dag.nodes if isinstance(feature, FeatureNode)),
            1,
        )

    def test_conv_and_mult_coeff_series(self):
        model = nn_modules.ConvAndMultCoeffSeries()
        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 256, 256]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'multiplexed')

        init_config_with_args(poly_n=65536, style='multiplexed', graph_type='btp')
        graph, score = run_pipeline(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )

        self.assertEqual(
            max(graph.dag.nodes[feature]['level'] for feature in graph.dag.nodes if isinstance(feature, FeatureNode)),
            5,
        )

    def test_resnet_basic_block(self):
        model = nn_modules.ResNetBasicBlock(32, 32)
        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 64, 64]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

        init_config_with_args(poly_n=65536, style='ordinary', graph_type='btp')
        graph, score = run_pipeline(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )
        self.assertEqual(check_level_cost(graph), True)
        self.assertEqual(check_multi_input_level_skip_aligned(graph), True)

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

        init_config_with_args(poly_n=65536, style='multiplexed', graph_type='btp')
        graph, score = run_pipeline(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )
        self.assertEqual(check_level_cost(graph), True)
        self.assertEqual(check_multi_input_level_skip_aligned(graph), True)

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

        init_config_with_args(poly_n=65536, style='multiplexed', graph_type='btp')
        graph, score = run_pipeline(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )

    def test_mismatched_scale(self):
        model = nn_modules.MismatchedScale()
        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 64, 64]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

        init_config_with_args(poly_n=65536, style='ordinary', graph_type='btp')
        graph, score = run_pipeline(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )

    def test_intertwined(self):
        model = nn_modules.Intertwined()
        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 64, 64]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

        init_config_with_args(poly_n=65536, style='ordinary', graph_type='btp')
        graph, score = run_pipeline(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )

    def test_intertwined_with_coeff(self):
        model = nn_modules.IntertwinedWithCoeff()
        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 64, 64]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

        init_config_with_args(poly_n=65536, style='ordinary', graph_type='btp')
        graph, score = run_pipeline(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )

    @unittest.skip('Not supported yet')
    def test_multiple_inputs(self):
        model = nn_modules.MutipleInputs()
        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 64, 64]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

        init_config_with_args(poly_n=65536, style='ordinary', graph_type='btp')
        graph, score = run_pipeline(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )

    @unittest.skip('Not supported yet')
    def test_multiple_outputs(self):
        model = nn_modules.MutipleOutputs()
        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 64, 64]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

        init_config_with_args(poly_n=65536, style='ordinary', graph_type='btp')
        graph, score = run_pipeline(
            num_experiments=1,
            input_file_path=self.temp_json_path,
            output_dir=script_dir,
            temperature=0.0,
            num_workers=1,
        )

    def test_wrong_padding(self):
        model = nn_modules.WrongPadding()
        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 64, 64]),
            dynamic_batch=False,
            save_h5=False,
        )

        with self.assertRaisesRegex(ValueError, r'Unsupported padding value: \[0, 0, 0, 0\]'):
            onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

    def test_wrong_dilation(self):
        model = nn_modules.WrongDilation()
        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 64, 64]),
            dynamic_batch=False,
            save_h5=False,
        )

        with self.assertRaisesRegex(ValueError, r'Unsupported dilation value: \[2, 2\]'):
            onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

    def test_wrong_groups(self):
        model = nn_modules.WrongGroups()
        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 64, 64]),
            dynamic_batch=False,
            save_h5=False,
        )

        with self.assertRaisesRegex(ValueError, r'Unsupported groups value: 2'):
            onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

    def test_unreplaced_relu(self):
        model = nn_modules.SingleRelu()
        export_to_onnx(
            model,
            save_path=self.temp_onnx_path,
            input_size=tuple([1, 32, 64, 64]),
            dynamic_batch=False,
            save_h5=False,
        )
        onnx_to_json(self.temp_onnx_path, self.temp_json_path, 'ordinary')

        init_config_with_args(poly_n=65536, style='ordinary', graph_type='btp')
        with self.assertRaisesRegex(ValueError, r'Relu2d is not supported in current mode'):
            graph, score = run_pipeline(
                num_experiments=1,
                input_file_path=self.temp_json_path,
                output_dir=script_dir,
                temperature=0.0,
                num_workers=1,
            )


if __name__ == '__main__':
    unittest.main()
