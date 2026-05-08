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

"""log_slots auto-detection for sparse-bootstrap-aware compilation.

Phase 1 of the latti-ai sparse-bootstrap E2E plan. Kept as a zero-dep
module (stdlib only) so unit tests can import it without bringing in
the rest of the compiler stack (torch / onnx / tqdm / networkx).
"""

import json
import math
from pathlib import Path


def _active_slots(feat: dict) -> int:
    """Definition A — tightest: physical occupied slots per ciphertext.

    Uses the feature's shape, pack_num, and skip as written in the graph.
    The graph already accounts for block_shape splitting on big-size
    features, so this returns the slot count of a single ciphertext at
    that bootstrap site.
    """
    pack_num = int(feat.get('pack_num', 1))
    dim = int(feat['dim'])
    if dim == 0:
        skip = int(feat.get('skip', 1))
        return max(1, pack_num // max(1, skip))
    if dim == 1:
        length = int(feat['shape'][0])
        skip = int(feat.get('skip', 1))
        return max(1, length * (pack_num // max(1, skip)))
    if dim == 2:
        h, w = int(feat['shape'][0]), int(feat['shape'][1])
        skip = feat.get('skip', [1, 1])
        s_div = max(1, int(skip[0]) * int(skip[1]))
        return max(1, h * w * (pack_num // s_div))
    raise ValueError(f'unsupported feat dim: {dim}')


def _infer_slots(server_dir: Path, n: int) -> int:
    """Walk bootstrap nodes in nn_layers_ct_0.json and return the smallest
    power-of-two slot count covering every bootstrap input under
    Definition A. Returns ``n // 2`` (dense full-packing) when there is
    no bootstrap or when the inferred coverage already equals ``n // 2``.

    The HEonGPU sparse bootstrap floor is ``log_slots >= 2`` (lattisense
    feat/sparse-bootstrap, project memory `project_sparse_bootstrap_log_slots_floor`).
    """
    nn_path = Path(server_dir) / 'nn_layers_ct_0.json'
    with open(nn_path) as f:
        nn = json.load(f)

    btp_input_slots = [
        _active_slots(nn['feature'][fid])
        for layer in nn['layer'].values()
        if layer['type'] == 'bootstrapping'
        for fid in layer['feature_input']
    ]
    if not btp_input_slots:
        return n // 2

    raw = max(btp_input_slots)
    # Floor at log_slots=8: the Python frontend's `_gen_wfft_index_map`
    # panics with a negative shift count below this with default CTS/STC depth.
    log_slots = max(8, math.ceil(math.log2(max(2, raw))))
    if (1 << log_slots) >= n // 2:
        return n // 2
    return 1 << log_slots
