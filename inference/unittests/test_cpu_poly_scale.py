"""Regression coverage for runtime coefficient scale dependencies in CPU BSGS."""

import json
import tempfile
import unittest
from pathlib import Path

from inference.model_generator.deploy_cmds import gen_custom_task


class TestCpuPolyScale(unittest.TestCase):
    def test_activation_boundaries(self):
        for use_gpu in (False, True):
            for lazy in (False, True):
                with self.subTest(use_gpu=use_gpu, lazy=lazy), tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp)
                    feature = dict(dim=2, channel=1, shape=[2, 2], skip=[1, 1], pack_num=1)
                    config = {
                        'input_feature': ['x'], 'output_feature': ['z'],
                        'feature': {name: dict(feature, level=level)
                                    for name, level in [('x', 10), ('y', 7), ('z', 4)]},
                        'layer': {
                            name: dict(type='polyact', channel_input=1, channel_output=1,
                                       feature_input=[src], feature_output=[dst], order=4)
                            for name, src, dst in [('act1', 'x', 'y'), ('act2', 'y', 'z')]
                        },
                    }
                    (path / 'task_config.json').write_text('{}')
                    (path / 'nn_layers_ct_0.json').write_text(json.dumps(config))
                    gen_custom_task(tmp, param_name='PN15Default', use_gpu=use_gpu, lazy=lazy)
                    graph = json.loads((path / 'compiled_mega_ag.json').read_text())
                    ops = [op for group in graph['compute'].values() for op in group['ops']]
                    self.assertFalse(any(op['type'] == 'normalize_scale' for op in ops))
                    encodes = [op for op in ops if op['type'] == 'encode_pt'
                               and op.get('attributes', {}).get('runtime_bsgs_scale')]
                    self.assertEqual(len(encodes), 0 if use_gpu else 10)
                    for op in encodes:
                        attrs = op['attributes']
                        self.assertNotEqual(graph['data'][op['inputs'][0]]['type'], 'ct')
                        for key in ('power_input', 'giant_input'):
                            if key in attrs:
                                ref = op['inputs'][attrs[key]]
                                self.assertEqual(graph['data'][ref]['type'], 'ct')
                                self.assertNotIn(ref, op['outputs'])
                        self.assertEqual('power_input' in attrs, attrs['i'] not in (0, 3))
                        self.assertEqual('giant_input' in attrs, attrs['i'] >= 3)



if __name__ == '__main__':
    unittest.main()
