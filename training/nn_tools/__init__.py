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

from .activations import RangeNorm2d, Simple_Polyrelu, RangeNormPoly2d
from .replace import replace_activation, replace_activation_with_poly, replace_maxpool_with_avgpool
from .export import (
    export_to_onnx,
    save_onnx_weights_to_h5,
    remove_identity_nodes,
    load_h5_weights,
    fuse_and_export_h5,
)

__all__ = [
    # Activations
    'RangeNorm2d',
    'Simple_Polyrelu',
    'RangeNormPoly2d',
    # Replace
    'replace_activation',
    'replace_activation_with_poly',
    'replace_maxpool_with_avgpool',
    # Export
    'export_to_onnx',
    'save_onnx_weights_to_h5',
    'remove_identity_nodes',
    'load_h5_weights',
    'fuse_and_export_h5',
]

__version__ = '1.0.0'
