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
import numpy as np


class Avgpool_layer:
    def __init__(self, stride, shape, channel=1, skip=[1, 1]):
        self.stride = stride
        self.shape = shape
        self.skip = skip
        self.channel = channel

        if shape[0] & (shape[0] - 1) != 0 or shape[1] & (shape[1] - 1) != 0:
            raise ValueError(f"shape must be powers of 2, got: [{shape[0]}, {shape[1]}]")
        if stride[0] & (stride[0] - 1) != 0 or stride[1] & (stride[1] - 1) != 0:
            raise ValueError(f"stride must be powers of 2, got: [{stride[0]}, {stride[1]}]")
        if skip[0] & (skip[0] - 1) != 0 or skip[1] & (skip[1] - 1) != 0:
            raise ValueError(f"skip must be powers of 2, got: [{skip[0]}, {skip[1]}]")

    def call(self, x: list[DataNode]):
        res: list[DataNode] = list()
        for i in range(len(x)):
            rr = x[i]
            for j in range(1, self.stride[0]):
                ri = rotate_cols(x[i], [j * self.shape[0]])[0]
                rr = add(rr, ri)
            step = self.stride[0]
            while step > 1:
                step = int(step)
                ri = rotate_cols(rr, [step // 2])[0]
                rr = add(rr, ri)
                step /= 2
            res.append(rr)
        return res

    def run_adaptive_avgpool(self, x: list[DataNode], n: int):
        # n: number of valid slots in a ciphertext
        x_size = len(x)

        # If ciphertext slots are not full, need to rotate to fill them
        n_rot = int(np.ceil(n / 2 / (self.channel * self.shape[0] * self.shape[1])))

        log2_shape_0 = int(np.ceil(np.log2(self.shape[0])))
        log2_shape_1 = int(np.ceil(np.log2(self.shape[1])))

        result = []
        for idx in range(0, x_size):
            res = x[idx]
            for i in range(log2_shape_0 - 1, 0 - 1, -1):
                ct_tmp = rotate_cols(res, (2**i) * self.shape[0] * self.skip[0] * self.skip[1])
                res = add(res, ct_tmp[0])

            for j in range(log2_shape_1 - 1, 0 - 1, -1):
                ct_tmp = rotate_cols(res, (2**j) * self.skip[1])
                res = add(res, ct_tmp[0])

            for r in range(0, int(np.log2(n_rot))):
                res = add(res, rotate_cols(res, (2**r) * self.channel * self.shape[0] * self.shape[1])[0])
            result.append(res)
        return result
