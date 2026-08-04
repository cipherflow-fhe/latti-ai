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

from inference.lattisense.frontend.custom_task import (
    CkksCiphertextNode,
    CkksPlaintextMulNode,
    CkksPlaintextNode,
    CkksPlaintextRingtNode,
    add,
    mult,
    mult_relin,
    rescale,
)


PlainInputNode = CkksPlaintextNode | CkksPlaintextRingtNode | CkksPlaintextMulNode
ParamInputNode = CkksCiphertextNode | PlainInputNode


def is_plaintext_input(node: ParamInputNode) -> bool:
    return isinstance(node, (CkksPlaintextNode, CkksPlaintextRingtNode, CkksPlaintextMulNode))


def multiply_with_encrypted_param(x: ParamInputNode, weight_ct: CkksCiphertextNode) -> CkksCiphertextNode:
    if isinstance(x, CkksCiphertextNode):
        return rescale(mult_relin(x, weight_ct))
    if is_plaintext_input(x):
        return rescale(mult(weight_ct, x))
    raise ValueError(f'Unsupported encrypted parameter input node type: {type(x)!r}')


def accumulate_encrypted_param_terms(xs: list[ParamInputNode], weights: list[CkksCiphertextNode]) -> CkksCiphertextNode:
    if len(xs) != len(weights):
        raise ValueError(f'Input/weight term count mismatch: {len(xs)} vs {len(weights)}')
    if not xs:
        raise ValueError('Encrypted parameter accumulation requires at least one term')
    result = multiply_with_encrypted_param(xs[0], weights[0])
    for x, weight in zip(xs[1:], weights[1:]):
        result = add(result, multiply_with_encrypted_param(x, weight))
    return result


def require_no_plaintext_input_rotation(layer_name: str, input_is_plaintext: bool, rotations_needed: bool) -> None:
    if input_is_plaintext and rotations_needed:
        raise ValueError(
            f'{layer_name} with plaintext graph input requires plaintext rotation support or pre-rotated '
            'encrypted parameters, which is not implemented in server_provisioned_runner Phase 1.'
        )
