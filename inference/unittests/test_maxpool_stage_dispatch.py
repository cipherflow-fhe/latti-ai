"""Dense output packing and stage-boundary regression checks."""
import unittest
from unittest.mock import patch

import numpy as np

from inference.model_generator.layers import poly_maxpool_stages as stages
from inference.unittests.test_interleaved_maxpool_reduction import Cipher
from training.model_compiler.maxpool_plan import build_maxpool_plan


class StageDispatchTest(unittest.TestCase):
    def test_parameter_manifest_matches_polynomial_scales(self):
        import json
        from types import SimpleNamespace
        import inference.lattisense.frontend.custom_task as custom_task
        from inference.model_generator.layers.poly2d import Poly2D

        plan = build_maxpool_plan(
            {'kernel_shape': [3, 3], 'stride': [2, 2], 'padding': [1, 1]}, [256, 256])
        # Nonuniform moduli expose accidental substitution of q[level] for
        # the Stockmeyer coefficient scale.
        param = SimpleNamespace(n=65536, scale=2.**40,
                                q=[2.**40 + 4096 * i for i in range(10)])
        for round_index in range(4):
            declarations = {}

            def make_pt(name, count, descriptor):
                self.assertNotIn(name, declarations)
                declarations[name] = dict(descriptor, count=count)
                return [name] * count

            with patch.object(custom_task, 'g_param', param):
                stages.bind_maxpool_stage_plaintexts(
                    f'reduce_round_{round_index + 1}', plan, 3, 32768, 9, 2., make_pt)
                for prefix, level in [('p0', 9), ('p1', 6)]:
                    poly = Poly2D([128, 128], 7, [1, 1], 1)
                    poly.init_stockmeyer(level)
                    for degree in range(8):
                        desc = declarations[f'{prefix}_degree_{degree}']
                        self.assertEqual(desc['level'], poly.cached_stockmeyer_level_order[degree])
                        self.assertEqual(desc['scale'], poly.cached_stockmeyer_coeff_scale[degree])
                        self.assertEqual(desc['count'], 3)
            self.assertEqual(json.loads(json.dumps(declarations)), declarations)
            bound = (1 if round_index == 0 else 1.05) * 2**round_index
            self.assertAlmostEqual(declarations['p0_degree_1']['value'], 7.30445164958251 / (2 * bound))
        with self.assertRaisesRegex(ValueError, 'positive calibrated'):
            stages.bind_maxpool_stage_plaintexts('restore', plan, 3, 32768, 9, 0., make_pt)

    def test_pack_restore_odd_channel_count(self):
        # Retina output plane size, including the partially filled last CT.
        plane, slots, channels = 128 * 128, 32768, 3
        plan = build_maxpool_plan(
            {'kernel_shape': [3, 3], 'stride': [2, 2], 'padding': [1, 1]}, [256, 256],
        )
        mask = np.zeros(slots)
        mask[:plane] = 1
        nodes = [Cipher(np.r_[np.full(plane, (ch + 1) * 16.), np.full(plane, 999.)], 9)
                 for ch in range(channels)]
        with patch.object(stages, 'mult', lambda ct, pt: Cipher(ct.values * pt, ct.level)), \
             patch.object(stages, 'rescale', lambda ct: Cipher(ct.values, ct.level - 1)), \
             patch.object(stages, 'rotate_cols', lambda ct, steps: [Cipher(np.roll(ct.values, -steps[0]), ct.level)]), \
             patch.object(stages, 'add', lambda a, b: Cipher(a.values + b.values, a.level)):
            packed = stages.call_maxpool_stage('repack', nodes, plan, channels, slots,
                {'select': [mask] * channels})
            self.assertEqual(len(packed), 2)
            restored = stages.call_maxpool_stage('restore', packed, plan, channels, slots,
                {'restore': [1 / 16] * 2})
        np.testing.assert_array_equal(restored[0].values[:plane], 1)
        np.testing.assert_array_equal(restored[0].values[plane:], 2)
        np.testing.assert_array_equal(restored[1].values[:plane], 3)
        np.testing.assert_array_equal(restored[1].values[plane:], 0)
        self.assertEqual([ct.level for ct in restored], [7, 7])

    def test_fused_backend_rejects_internal_bootstrap(self):
        from inference.model_generator.deploy_cmds import _collapse_maxpool_stage_layers
        config = {'layer': {
            'pool_normalize_taps': {'type': 'maxpool2d_normalize_taps',
                'feature_input': ['x'], 'feature_output': ['taps']},
            'btp': {'type': 'bootstrapping', 'feature_input': ['taps'], 'feature_output': ['fresh']},
        }}
        with self.assertRaisesRegex(NotImplementedError, 'cannot be collapsed'):
            _collapse_maxpool_stage_layers(config)
        self.assertIn('pool_normalize_taps', config['layer'])


if __name__ == '__main__':
    unittest.main()
