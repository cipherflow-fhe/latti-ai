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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from inference.lattisense.frontend.custom_task import *
from inference.model_generator.layers.poly_relu_base import PolyReluBase

op_class = 'PolyRelu0D'


class PolyRelu0D(PolyReluBase):
    """PolyRelu for Feature0DEncrypted.

    Slot layout: channel ch at slot ch * skip (remaining slots zero-padded).
      n_channel_per_ct = N/2 / skip
    """

    def __init__(self, order: int, skip: int, n_channel_per_ct: int):
        """
        Args:
            order:            polynomial order
            skip:             ciphertext skip (channel ch at slot ch * skip)
            n_channel_per_ct: number of channels packed per ciphertext
        """
        self.order = order
        self.skip = skip
        self.n_channel_per_ct = n_channel_per_ct

    def get_fhe_op_count_bsgs_feature0d(self, n_ct: int, level: int) -> dict[int, dict[str, int]]:
        """Count FHE primitive operations in call_bsgs_feature0d() for n_ct input ciphertexts, grouped by level.

        call_bsgs_feature0d() delegates entirely to _run_bsgs_core(), so the op
        count is identical to PolyReluBase.get_fhe_op_count().
        See PolyReluBase.get_fhe_op_count() for the detailed breakdown.
        """
        return self.get_fhe_op_count(n_ct, level)

    def call_bsgs_feature0d(self, x: list, weight_pt):
        """BSGS with pre-computed weight plaintexts (eager mode).

        weight_pt shape: [order+1][n_ct]
        """
        return self._run_bsgs_core(x, lambda idx, x_idx: weight_pt[idx][x_idx])

    def call_bsgs_feature0d_lazy(self, x: list, poly_data_source, layer_id: str = ''):
        """BSGS with on-demand weight generation via custom_compute (lazy mode)."""
        weight_cache = {}

        def get_weight(idx, x_idx):
            key = (idx, x_idx)
            if key not in weight_cache:
                w_pt = CkksPlaintextRingtNode(f'encode_pt_0d_{layer_id}_{idx}_{x_idx}')
                custom_compute(
                    inputs=[poly_data_source],
                    output=w_pt,
                    type='encode_pt',
                    attributes={
                        'op_class': op_class,
                        'type': 'weight_pt_feature0d',
                        'i': idx,
                        'j': x_idx,
                    },
                )
                weight_cache[key] = w_pt
            return weight_cache[key]

        return self._run_bsgs_core(x, get_weight)
