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
import os

# Add mega_ag_generator to path for importing frontend module
script_dir = os.path.dirname(os.path.abspath(__file__))
mega_ag_generator_dir = os.path.join(script_dir, '../../lattisense')
sys.path.insert(0, mega_ag_generator_dir)

from frontend.custom_task import *

DEFAULT_SCALE = 2**31

ENC_TOSHARE_SCALE = 2**28


def f_equal(a, b):
    eps = 1e-8
    if abs(b) < eps:
        return abs(a - b) < eps
    else:
        return abs((a - b) / b) < eps


class MultScalarLayer:
    def __init__(self):
        # self.target_scale = target_scale
        return

    def call(self, x1: list[DataNode], pt_scale1: DataNode):
        result: list[list[DataNode]] = list()

        for i in range(len(x1)):
            mult_res = mult(x1[i], pt_scale1)
            mult_res_scale = rescale(mult_res)
            result.append(mult_res_scale)
        return result
