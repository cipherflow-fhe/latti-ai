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

"""Generate GPU instruction files for CKKS primitive-op benchmark.

One instruction directory per (op, N, level), named e.g.:
  ckks_add_n16384_l3/
  ckks_mult_relin_n8192_l1/

Level ranges (matching score.py timing tables):
  add, rotate     : level 0 ~ max_level
  mult_plain, mult, rescale : level 1 ~ max_level

max_level per N (from parameter.json):
  N=8192  → 5
  N=16384 → 9
  N=32768 → 17
  N=65536 → 33

Usage:
  python benchmark_ckks_ops_gpu.py
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_here, '..', '..')
sys.path.insert(0, _project_root)
sys.path.append(os.path.join(_here, '..', 'lattisense'))

from frontend.custom_task import (
    Argument,
    CkksCiphertextNode,
    CkksPlaintextNode,
    Param,
    add,
    mult,
    mult_relin,
    process_custom_task,
    rescale,
    rotate_cols,
    set_fhe_param,
)
from training.model_compiler.components import PN16QP1761


def make_param(n: int) -> Param:
    """Return the correct Param for a given N.

    N=65536 uses PN16QP1761 (34 Q, 4 P, max_level=33) to match the C++ side's
    CkksParameter::create_parameter(65536).  All other N values use the default
    parameter.json entries.
    """
    if n == 65536:
        return Param.create_ckks_custom_param(n=n, q=PN16QP1761.q, p=PN16QP1761.p)
    return Param.create_ckks_default_param(n=n)


N_OPS_DEFAULT = 16
N_OPS_LARGE = 16  # N=65536: fewer ops per batch (larger polynomials)

# (N, max_level, n_ops)
N_CONFIGS = [
    (8192, 5, N_OPS_DEFAULT),
    (16384, 9, N_OPS_DEFAULT),
    (32768, 17, N_OPS_DEFAULT),
    (65536, 33, N_OPS_LARGE),
]


def outpath(op_name: str, n: int, level: int) -> str:
    return os.path.join(_here, f'{op_name}_n{n}_l{level}')


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def gen_add(n: int, level: int, n_ops: int):
    param = make_param(n)
    set_fhe_param(param)
    xs = [CkksCiphertextNode(f'x_{i}', level) for i in range(n_ops)]
    ys = [CkksCiphertextNode(f'y_{i}', level) for i in range(n_ops)]
    zs = [add(xs[i], ys[i], f'z_{i}') for i in range(n_ops)]
    process_custom_task(
        input_args=[Argument('xs', xs), Argument('ys', ys)],
        output_args=[Argument('zs', zs)],
        output_instruction_path=outpath('ckks_add', n, level),
    )


def gen_rotate(n: int, level: int, n_ops: int):
    param = make_param(n)
    set_fhe_param(param)
    xs = [CkksCiphertextNode(f'x_{i}', level) for i in range(n_ops)]
    ys = [rotate_cols(xs[i], 1, f'y_{i}') for i in range(n_ops)]
    process_custom_task(
        input_args=[Argument('xs', xs)],
        output_args=[Argument('ys', ys)],
        output_instruction_path=outpath('ckks_rotate', n, level),
    )


def gen_mult_plain(n: int, level: int, n_ops: int):
    param = make_param(n)
    set_fhe_param(param)
    xs = [CkksCiphertextNode(f'x_{i}', level) for i in range(n_ops)]
    ys = [CkksPlaintextNode(f'y_{i}', level) for i in range(n_ops)]
    zs = [mult(xs[i], ys[i], f'z_{i}') for i in range(n_ops)]
    process_custom_task(
        input_args=[Argument('xs', xs), Argument('ys', ys)],
        output_args=[Argument('zs', zs)],
        output_instruction_path=outpath('ckks_mult_plain', n, level),
    )


def gen_mult(n: int, level: int, n_ops: int):
    param = make_param(n)
    set_fhe_param(param)
    xs = [CkksCiphertextNode(f'x_{i}', level) for i in range(n_ops)]
    ys = [CkksCiphertextNode(f'y_{i}', level) for i in range(n_ops)]
    zs = [mult_relin(xs[i], ys[i], f'z_{i}') for i in range(n_ops)]
    process_custom_task(
        input_args=[Argument('xs', xs), Argument('ys', ys)],
        output_args=[Argument('zs', zs)],
        output_instruction_path=outpath('ckks_mult_relin', n, level),
    )


def gen_rescale(n: int, level: int, n_ops: int):
    param = make_param(n)
    set_fhe_param(param)
    xs = [CkksCiphertextNode(f'x_{i}', level) for i in range(n_ops)]
    ys = [rescale(xs[i], f'y_{i}') for i in range(n_ops)]
    process_custom_task(
        input_args=[Argument('xs', xs)],
        output_args=[Argument('ys', ys)],
        output_instruction_path=outpath('ckks_rescale', n, level),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    for n, max_lv, n_ops in N_CONFIGS:
        print(f'N={n}  (max_level={max_lv}, n_ops={n_ops})')

        # add & rotate: level 0 ~ max_lv
        for lv in range(0, max_lv + 1):
            gen_add(n, lv, n_ops)
            gen_rotate(n, lv, n_ops)
            print(f'  level {lv}: add, rotate')

        # mult_plain, mult, rescale: level 1 ~ max_lv
        for lv in range(1, max_lv + 1):
            gen_mult_plain(n, lv, n_ops)
            gen_mult(n, lv, n_ops)
            gen_rescale(n, lv, n_ops)
            print(f'  level {lv}: mult_plain, mult, rescale')

    print('Done.')
