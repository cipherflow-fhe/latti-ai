/*
 * Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <functional>
#include <vector>
#include "layer.h"
#include "../data_structs/feature0d.h"

class SoftmaxLayer : public Layer {
public:
    using KernelFn =
        std::function<std::vector<ls::CkksCiphertext>(ls::CkksContext&, const std::vector<ls::CkksCiphertext>&)>;

    explicit SoftmaxLayer(const ls::CkksParameter& param_in,
                          uint32_t n_classes = 0,
                          uint32_t input_level = 0,
                          KernelFn kernel = nullptr);

    void set_kernel(KernelFn kernel);
    void prepare_offline_args(uint32_t n_classes, uint32_t input_level);
    std::vector<ls::CkksCiphertext> run_core(ls::CkksContext& ctx, const std::vector<ls::CkksCiphertext>& x) const;
    Feature0DEncrypted run(ls::CkksContext& ctx, const Feature0DEncrypted& x) const;
    Array<double, 1> run_plaintext(const Array<double, 1>& x) const;

    std::vector<ls::CkksPlaintextRingt> pt_quarter;
    std::vector<ls::CkksPlaintextRingt> pt_inv_classes;
    std::vector<ls::CkksPlaintextMul> exp_c5;
    std::vector<ls::CkksPlaintext> exp_c4;
    std::vector<ls::CkksPlaintext> exp_c3;
    std::vector<ls::CkksPlaintext> exp_c2;
    std::vector<ls::CkksPlaintext> exp_c1;
    std::vector<ls::CkksPlaintext> exp_c0;
    std::vector<ls::CkksPlaintextMul> recip_c3;
    std::vector<ls::CkksPlaintext> recip_c2;
    std::vector<ls::CkksPlaintext> recip_c1;
    std::vector<ls::CkksPlaintext> recip_c0;

private:
    KernelFn kernel_;
    uint32_t n_classes_ = 0;
    uint32_t input_level_ = 0;
};
