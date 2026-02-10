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

    def mult_and_add(self, x1: DataNode, x2: DataNode, pt_scale1: DataNode, pt_scale2: DataNode = None):
        mult_res1 = mult(x1, pt_scale1)
        mult_res1 = rescale(mult_res1)
        if pt_scale2 is not None:
            mult_res2 = mult(x2, pt_scale2)
            mult_res2 = rescale(mult_res2)
            if mult_res2.level < mult_res1.level:
                drop_level_n = mult_res1.level - mult_res2.level
                mult_res1 = drop_level(mult_res1, drop_level_n)
            elif mult_res1.level < mult_res2.level:
                drop_level_n = mult_res2.level - mult_res1.level
                mult_res2 = drop_level(mult_res2, drop_level_n)
            return add(mult_res1, mult_res2)
        else:
            if x2.level < mult_res1.level:
                drop_level_n = mult_res1.level - x2.level
                mult_res1 = drop_level(mult_res1, drop_level_n)
            elif mult_res1.level < x2.level:
                drop_level_n = x2.level - mult_res1.level
                x2 = drop_level(x2, drop_level_n)
            return add(mult_res1, x2)

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
            for i in range(len(x1)):
                if x2[i].level < x1[i].level:
                    drop_level_n = x1[i].level - x2[i].level
                    temp = drop_level(x1[i], drop_level_n)
                    res = add(temp, x2[i])
                elif x2[i].level > x1[i].level:
                    drop_level_n = x2[i].level - x1[i].level
                    temp = drop_level(x2[i], drop_level_n)
                    res = add(temp, x1[i])
                elif x2[i].level == x1[i].level:
                    res = add(x1[i], x2[i])
                result.append(res)
        elif scale1 == 1.0 and scale2 != 1.0:
            for i in range(len(x1)):
                result.append(self.mult_and_add(x2[i], x1[i], pt_scale2))
        elif scale1 != 1.0 and scale2 == 1.0:
            for i in range(len(x1)):
                result.append(self.mult_and_add(x1[i], x2[i], pt_scale1))
        else:
            for i in range(len(x1)):
                result.append(self.mult_and_add(x1[i], x2[i], pt_scale1, pt_scale2))
        return result
