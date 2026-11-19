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
"""nn_tools: activation replacement and ONNX export utilities.

Replace standard activations (e.g. ReLU) with polynomial approximations
suitable for encrypted inference, and export models to ONNX / H5 formats.
"""

from importlib import import_module

__all__ = [
    # Activations
    'RangeNorm2d',
    'PolyAct',
    'PolyActRNPoly',
    'RangeNormPoly2d',
    'FHELayerNorm',
    # Modules
    'DepthwiseAvgPool2d',
    # Replace
    'replace_activation',
    'replace_activation_with_poly',
    'replace_maxpool_with_avgpool',
    'replace_layernorm_with_fhe',
    'replace_general_avgpool_with_depthwise_conv',
    'prepare_for_fhe',
    # Export
    'export_to_onnx',
    'save_onnx_weights_to_h5',
    'remove_identity_nodes',
    'load_h5_weights',
    'fuse_and_export_h5',
    'export_h5_from_onnx',
]

__version__ = '1.0.0'


def __getattr__(name):
    # ONNX compilation must not import the optional PyTorch training stack.
    if name in ('RangeNorm2d', 'PolyAct', 'PolyActRNPoly', 'RangeNormPoly2d', 'FHELayerNorm'):
        module = 'activations'
    elif name == 'DepthwiseAvgPool2d':
        module = 'modules'
    elif name in (
        'replace_activation',
        'replace_activation_with_poly',
        'replace_maxpool_with_avgpool',
        'replace_layernorm_with_fhe',
        'replace_general_avgpool_with_depthwise_conv',
        'prepare_for_fhe',
    ):
        module = 'replace'
    elif name in __all__:
        module = 'export'
    else:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module(f'.{module}', __name__), name)
    globals()[name] = value
    return value
