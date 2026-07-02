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


def naf_weight(n: int) -> int:
    """Return the number of non-zero digits in the NAF (Non-Adjacent Form) of |n|.

    This matches the get_glk_col() / convert2naf() decomposition used in
    custom_task.py: each non-zero NAF digit corresponds to one RotateColUnit
    primitive inside rotate_cols(). Use this to convert a rotate_cols call count
    into an accurate primitive-rotate count when steps are not powers of 2.
    """
    n = abs(n)
    xh = n >> 1
    x3 = n + xh
    c = xh ^ x3
    return bin(x3 & c).count('1') + bin(xh & c).count('1')


def memory_from_pt_counts(counts: dict[str, int], bytes_per_plaintext: int = 0) -> dict[str, int]:
    """Build a plaintext-memory summary from plaintext-node counts."""
    result = dict(counts)
    total = sum(counts.values())
    result['total'] = total
    result['bytes_per_plaintext'] = bytes_per_plaintext
    result['bytes'] = total * bytes_per_plaintext
    return result
