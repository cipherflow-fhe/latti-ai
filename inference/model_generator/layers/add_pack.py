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


class AddLayer:
    def __init__(self):
        return

    def call(
        self,
        x1: list[DataNode],
        x2: list[DataNode],
        scale1: int,
        scale2: int,
        pt_scale1: DataNode = None,
        pt_scale2: DataNode = None,
    ):
        result: list[DataNode] = list()
        if scale1 == 1.0 and scale2 == 1.0:
            res = add(x1[i], x2[i])
            result.append(res)
        return result
