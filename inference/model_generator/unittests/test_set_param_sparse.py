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

"""Unit tests for deploy_cmds.set_param sparse-vs-dense factory selection.

Phase 2 of the latti-ai sparse-bootstrap E2E plan. Verifies that
set_param picks CkksBtpParam.create_sparse_param(log_slots) when
slots < N/2 and falls back to create_default_param() otherwise.
The Galois set augmentation (sparse Trace rotations) is then a
consequence of lattisense's set_slots() / rotations_for_bootstrapping()
chain — covered by integration smoke tests, not here.
"""

import sys
import unittest
from pathlib import Path

_LATTI_AI_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_LATTI_AI_ROOT))
sys.path.insert(0, str(_LATTI_AI_ROOT / 'inference' / 'lattisense'))

from inference.model_generator import deploy_cmds  # noqa: E402
from inference.lattisense.frontend.custom_task import CkksBtpParam, CkksParam  # noqa: E402


class TestSetParamSparseFactory(unittest.TestCase):
    def setUp(self):
        # Defensive: clear any global FHE param state between tests.
        from inference.lattisense.frontend import custom_task

        custom_task.g_param = None

    def _get_global_param(self):
        from inference.lattisense.frontend import custom_task

        return custom_task.g_param

    def test_btp_default_when_slots_none(self):
        deploy_cmds.set_param('N16QP1546H192H32', slots=None)
        p = self._get_global_param()
        self.assertIsInstance(p, CkksBtpParam)
        # Default param means full packing: slots == n // 2.
        self.assertEqual(p.slots, p.n // 2)

    def test_btp_default_when_slots_equals_n_half(self):
        deploy_cmds.set_param('N16QP1546H192H32', slots=32768)
        p = self._get_global_param()
        self.assertEqual(p.slots, p.n // 2)

    def test_btp_sparse_when_slots_below_n_half(self):
        deploy_cmds.set_param('N16QP1546H192H32', slots=4096)
        p = self._get_global_param()
        self.assertIsInstance(p, CkksBtpParam)
        self.assertEqual(p.slots, 4096)
        self.assertTrue(p.is_sparse())

    def test_btp_sparse_at_log_slots_8(self):
        deploy_cmds.set_param('N16QP1546H192H32', slots=256)
        p = self._get_global_param()
        self.assertEqual(p.slots, 256)
        self.assertTrue(p.is_sparse())

    def test_sparse_param_emits_trace_rotations(self):
        # Direct check: rotations_for_bootstrapping must include 2^i for
        # i ∈ [log_slots, log_n - 1) when sparse.
        deploy_cmds.set_param('N16QP1546H192H32', slots=4096)
        p = self._get_global_param()
        rots = set(p.rotations_for_bootstrapping())
        for i in range(12, 15):  # log_slots=12, log_n=16 → range(12, 15)
            self.assertIn(1 << i, rots, f'missing Trace rotation 2^{i}={1 << i}')

    def test_non_btp_param_ignores_slots(self):
        # Non-btp params have no bootstrap → slots arg is irrelevant.
        deploy_cmds.set_param('PN14QP438', slots=256)
        p = self._get_global_param()
        self.assertIsInstance(p, CkksParam)
        # CkksParam.create_custom_param doesn't expose `slots` so just
        # verify it didn't crash.


if __name__ == '__main__':
    unittest.main()
