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


op_class = 'PolyRelu1DLayer'


class PolyRelu1D(PolyReluBase):
    """PolyRelu for Feature1DEncrypted.

    Two packing modes, matching Feature1DEncrypted::pack / pack_multiplexed:

      Mode 1 — skip pack (wasted slots):
        n_channel_per_ct = N/2 / (shape * skip)
        channel ch (CT-local), position i → slot = ch * shape * skip + i * skip

      Mode 2 — multiplexed/interleaved pack (no wasted slots):
        n_channel_per_ct = (N/2 / (shape * skip)) * skip
        channel j (CT-local), position i → slot = (j/skip)*shape*skip + i*skip + (j%skip)

    The BSGS computation graph is identical to 2D/0D; only weight plaintext
    layout differs (handled in C++ generate_weight_pt_for_bsgs).
    """

    def __init__(self, shape: int, order: int, skip: int, n_channel_per_ct: int):
        """
        Args:
            shape:            spatial length of the 1D feature
            order:            polynomial order
            skip:             skip of the Feature1DEncrypted
            n_channel_per_ct: channels per ciphertext (mode 1: N/2/(shape*skip),
                              mode 2: N/2/shape)
        """
        self.order = order
        self.shape1d = shape
        self.skip1d = skip
        self.n_channel_per_ct = n_channel_per_ct

    # ------------------------------------------------------------------
    # Mode 1 — skip pack
    # ------------------------------------------------------------------

    def get_fhe_op_count_bsgs_skip(self, n_ct: int, level: int) -> dict[int, dict[str, int]]:
        """Count FHE primitive operations in call_bsgs_skip() for n_ct input ciphertexts, grouped by level.

        call_bsgs_skip() delegates entirely to _run_bsgs_core(), so the op
        count is identical to PolyReluBase.get_fhe_op_count().
        See PolyReluBase.get_fhe_op_count() for the detailed breakdown.
        """
        return self.get_fhe_op_count(n_ct, level)

    def call_bsgs_skip(self, x: list, weight_pt):
        """BSGS with pre-computed weight plaintexts, skip-pack mode (eager).

        weight_pt shape: [order+1][n_ct]
        """
        return self._run_bsgs_core(x, lambda idx, x_idx: weight_pt[idx][x_idx])

    def call_bsgs_skip_lazy(self, x: list, poly_data_source, layer_id: str = ''):
        """BSGS with on-demand weight generation, skip-pack mode (lazy)."""
        weight_cache = {}

        def get_weight(idx, x_idx):
            key = (idx, x_idx)
            if key not in weight_cache:
                w_pt = CkksPlaintextRingtNode(f'encode_pt_1d_skip_{layer_id}_{idx}_{x_idx}')
                custom_compute(
                    inputs=[poly_data_source],
                    output=w_pt,
                    type='encode_pt',
                    attributes={
                        'op_class': op_class,
                        'type': 'weight_pt_skip',
                        'i': idx,
                        'j': x_idx,
                    },
                )
                weight_cache[key] = w_pt
            return weight_cache[key]

        return self._run_bsgs_core(x, get_weight)

    # ------------------------------------------------------------------
    # Mode 2 — multiplexed/interleaved pack
    # ------------------------------------------------------------------

    def get_fhe_op_count_bsgs_mux(self, n_ct: int, level: int) -> dict[int, dict[str, int]]:
        """Count FHE primitive operations in call_bsgs_mux() for n_ct input ciphertexts, grouped by level.

        call_bsgs_mux() delegates entirely to _run_bsgs_core(), so the op
        count is identical to PolyReluBase.get_fhe_op_count().
        The only difference from call_bsgs_skip() is the weight plaintext layout
        (handled in C++), which does not affect the FHE op graph.
        See PolyReluBase.get_fhe_op_count() for the detailed breakdown.
        """
        return self.get_fhe_op_count(n_ct, level)

    def call_bsgs_mux(self, x: list, weight_pt):
        """BSGS with pre-computed weight plaintexts, multiplexed mode (eager).

        weight_pt shape: [order+1][n_ct]
        """
        return self._run_bsgs_core(x, lambda idx, x_idx: weight_pt[idx][x_idx])

    def call_bsgs_mux_lazy(self, x: list, poly_data_source, layer_id: str = ''):
        """BSGS with on-demand weight generation, multiplexed mode (lazy)."""
        weight_cache = {}

        def get_weight(idx, x_idx):
            key = (idx, x_idx)
            if key not in weight_cache:
                w_pt = CkksPlaintextRingtNode(f'encode_pt_1d_mux_{layer_id}_{idx}_{x_idx}')
                custom_compute(
                    inputs=[poly_data_source],
                    output=w_pt,
                    type='encode_pt',
                    attributes={
                        'op_class': op_class,
                        'type': 'weight_pt_mux',
                        'i': idx,
                        'j': x_idx,
                    },
                )
                weight_cache[key] = w_pt
            return weight_cache[key]

        return self._run_bsgs_core(x, get_weight)
