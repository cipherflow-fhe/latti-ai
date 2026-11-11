# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components import FHE_PARAMS, N16QP1546H192H32, PN15Default, ComputeNode, FeatureNode, LayerAbstractGraph, config
import pipeline
from inference.model_generator.deploy_cmds import set_param


class FheProfileTest(unittest.TestCase):
    def setUp(self):
        previous = config.__dict__.copy()
        self.addCleanup(lambda: (config.__dict__.clear(), config.__dict__.update(previous)))

    def test_n16_metadata_reaches_generated_graph(self):
        fhe = N16QP1546H192H32
        param = set_param(fhe.name)
        self.assertEqual(FHE_PARAMS[fhe.name].to_dict()['secret_hamming_weight'], 192)
        self.assertEqual((param.log_n, param.log_max_slots(), param.max_level), (16, 15, 9))
        self.assertEqual(param.q, fhe.q)
        self.assertEqual(param.p, fhe.p)
        self.assertEqual(param.to_json_dict()['ring_type'], 0)
        self.assertEqual(param.to_json_dict()['secret_hamming_weight'], 192)

    def test_scoring_copy_preserves_n16_metadata(self):
        from graph_partition_dp import generate_param_dict_for_graph
        from score import encrypted_parameter_bytes

        config.fhe_param = N16QP1546H192H32
        copied = generate_param_dict_for_graph()['param0']
        self.assertEqual(copied.to_dict(), N16QP1546H192H32.to_dict())
        self.assertEqual(encrypted_parameter_bytes(copied, 9), 10486239)

    def test_n16_storage_calibration_matches_native_serialization(self):
        import copy
        from score import encrypted_parameter_bytes

        self.assertEqual(encrypted_parameter_bytes(N16QP1546H192H32, 0), 1048911)
        self.assertEqual(encrypted_parameter_bytes(N16QP1546H192H32, 9), 10486239)
        with self.assertRaisesRegex(ValueError, 'Uncalibrated encrypted parameter level'):
            encrypted_parameter_bytes(N16QP1546H192H32, 10)
        changed = copy.deepcopy(N16QP1546H192H32)
        changed.q[0] += 1
        with self.assertRaisesRegex(ValueError, 'No encrypted parameter size calibration'):
            encrypted_parameter_bytes(changed, 0)

    def test_ci_default_still_uses_32768_real_slots(self):
        param = set_param(PN15Default.name)
        self.assertEqual((param.log_n, param.log_max_slots()), (15, 15))
        self.assertEqual(param.to_json_dict()['ring_type'], 1)
        self.assertEqual(param.to_json_dict()['secret_hamming_weight'], 0)

    def test_selection_limits_no_bootstrap_candidates(self):
        config.fhe_param_name = N16QP1546H192H32.name
        config.set_btp_scale = None
        with (
            patch.object(pipeline, 'set_block_shape'),
            patch.object(pipeline, 'prepare_graph'),
            patch.object(pipeline, 'process_with_no_btp', return_value=None) as process,
        ):
            self.assertFalse(pipeline.try_no_btp(LayerAbstractGraph())[0])
        self.assertEqual(process.call_count, 1)
        self.assertEqual(config.fhe_param.name, N16QP1546H192H32.name)

    def test_n16_uses_slot_capacity_for_automatic_block_shape(self):
        graph = LayerAbstractGraph()
        graph.dag.add_edge(FeatureNode('input', 2, 1, shape=[512, 512]), ComputeNode('op', 'square', 1, 1))
        config.model_type = ''
        pipeline.set_block_shape(N16QP1546H192H32, graph)
        self.assertEqual(config.block_shape, [128, 256])


if __name__ == '__main__':
    unittest.main()
