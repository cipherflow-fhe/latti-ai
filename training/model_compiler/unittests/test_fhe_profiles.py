# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components import PN15Default, config
from inference.model_generator.deploy_cmds import set_param


class FheProfileTest(unittest.TestCase):
    def setUp(self):
        previous = config.__dict__.copy()
        self.addCleanup(lambda: (config.__dict__.clear(), config.__dict__.update(previous)))

    def test_ci_default_still_uses_32768_real_slots(self):
        param = set_param(PN15Default.name)
        self.assertEqual((param.log_n, param.log_max_slots()), (15, 15))
        self.assertEqual(param.to_json_dict()['ring_type'], 1)


if __name__ == '__main__':
    unittest.main()
