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
import math
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from inference.lattisense.frontend.custom_task import *


op_class = 'ParBlockColMajorAddPt'


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


class ParBlockColMajorAddPt:
    """Python model-generator counterpart of C++ ParBlockColMajorAddPt.

    Adds a plaintext matrix B to a ciphertext matrix A in par_block_col_major
    format.  Result = A + B.

    B is of shape (m, total_dim) where total_dim = n_heads * cols_per_head.
    Level consumption: 0 (plaintext addition does not consume a level).
    """

    def __init__(self, shape: tuple, block_size: int, n_heads: int, n_slot: int):
        """
        Args:
            shape:      (m, total_dim) — full matrix shape.
            block_size: d, block size.
            n_heads:    number of interleaved heads.
            n_slot:     N/2 (number of CKKS slots per ciphertext).
        """
        self.m = shape[0]
        total_dim = shape[1]
        self.d = block_size
        self.n_heads = n_heads
        self.n_slot = n_slot

        assert total_dim % n_heads == 0
        self.cols_per_head = total_dim // n_heads

        self.n_h_padded = _next_pow2(n_heads)

        if n_slot >= self.n_h_padded * self.d * self.d:
            self.S = self.n_h_padded
            self.chunk_size = self.n_h_padded * self.d * self.d
            self.G = 1
        else:
            self.S = n_slot // (self.d * self.d)
            self.chunk_size = n_slot
            if self.S == 1:
                self.n_h_padded = self.n_heads
            self.G = self.n_h_padded // self.S

        self.num_chunks = n_slot // self.chunk_size
        self.num_block_rows = math.ceil(self.m / self.d)
        self.num_block_cols = math.ceil(self.cols_per_head / self.d)
        self.total_cts = self.num_block_rows * self.num_block_cols * self.G

    # ------------------------------------------------------------------ #
    #  Public call — pre-declared weights                                 #
    # ------------------------------------------------------------------ #

    def call(self, A_cts: list, pt_nodes: list) -> list:
        """Build computation DAG: A + B (plaintext addition).

        Args:
            A_cts:    input CkksCiphertextNode list, total_cts entries.
            pt_nodes: list of total_cts CkksPlaintextRingtNode.
        Returns:
            list of output CkksCiphertextNode.
        """
        assert len(A_cts) == self.total_cts
        assert len(pt_nodes) == self.total_cts
        return [add(ct, pt) for ct, pt in zip(A_cts, pt_nodes)]

    # ------------------------------------------------------------------ #
    #  Custom-compute — inline weight declaration (lazy path)            #
    # ------------------------------------------------------------------ #

    def call_custom_compute(self, A_cts: list, data_source) -> list:
        """Build computation DAG with inline plaintext nodes via custom_compute."""
        # PT nodes must follow block_col_major ordering: bj -> bi -> g
        pt_nodes = []
        for bj in range(self.num_block_cols):
            for bi in range(self.num_block_rows):
                for g in range(self.G):
                    pt_node = CkksPlaintextRingtNode(f'encode_pt_{bi}_{bj}_{g}')
                    custom_compute(
                        inputs=[data_source],
                        output=pt_node,
                        type='encode_pt',
                        attributes={
                            'op_class': op_class,
                            'type': 'add_pt',
                            'bi': bi,
                            'bj': bj,
                            'g': g,
                        },
                    )
                    pt_nodes.append(pt_node)
        return self.call(A_cts, pt_nodes)

    # ------------------------------------------------------------------ #
    #  FHE operation count                                                #
    # ------------------------------------------------------------------ #

    def get_fhe_op_count(self, level: int) -> dict:
        """Count FHE primitive operations grouped by level.

        Per output CT: 1 add_plain (no level consumed).
        """
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})
        ops[level]['add'] += self.total_cts
        return dict(ops)
