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
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from inference.lattisense.frontend.custom_task import *

DEFAULT_SCALE = 2**31

ENC_TOSHARE_SCALE = 2**28


def f_equal(a, b):
    eps = 1e-8
    if abs(b) < eps:
        return abs(a - b) < eps
    else:
        return abs((a - b) / b) < eps


op_class = 'MultScalarLayer'


class MultScalarLayer:
    def __init__(self):
        return

    def make_pt_nodes(self, layer_id, n_input_nodes):
        """Return weight_pt list with n_input_nodes elements."""
        return [CkksPlaintextRingtNode(f'mult_scalar_{layer_id}_{i}') for i in range(n_input_nodes)]

    def call_custom_compute(self, x: list, conv_data_source) -> list:
        """Lazy path: generate encode_pt nodes on-demand."""
        result = []
        for i in range(len(x)):
            w_pt = CkksPlaintextRingtNode(f'encode_pt_mult_scalar_{i}')
            custom_compute(
                inputs=[conv_data_source],
                output=w_pt,
                type='encode_pt',
                attributes={'op_class': op_class, 'type': 'weight_pt', 'i': i},
            )
            mult_res = mult(x[i], w_pt)
            result.append(rescale(mult_res))
        return result

    def call_encode_ringt_compute(self, x: list, layer_id: str, scale: float):
        result = []
        sources = []
        for i, x_i in enumerate(x):
            source = CustomDataNode(type='mult_scalar_encode_ringt_data_source', id=f'{layer_id}_{i}')
            weight_pt = encode_ringt(source, scale, output_id=f'encode_ringt_{layer_id}_{i}')
            sources.append(source)
            result.append(rescale(mult(x_i, weight_pt)))
        return result, sources

    def get_fhe_op_count(self, n_ct: int, level: int) -> dict[int, dict[str, int]]:
        """Count FHE primitive operations in call() for n_ct input ciphertexts, grouped by level.

        Per ct: 1 mult (ct-pt) + 1 rescale  [level → level-1]
        """
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})
        lv = level

        ops[lv]['mult_plain'] += n_ct
        ops[lv]['rescale'] += n_ct
        lv -= 1

        return dict(ops)

    def call(self, x1: list[DataNode], weight_pt: list[DataNode]):
        result: list[DataNode] = list()

        for i in range(len(x1)):
            mult_res = mult(x1[i], weight_pt[i])
            mult_res_scale = rescale(mult_res)
            result.append(mult_res_scale)
        return result
