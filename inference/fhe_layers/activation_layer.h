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

#include <vector>
#include "layer.h"
#include "../data_structs/feature2d.h"
#include "../data_structs/feature0d.h"

class SquareLayer : public Layer {
public:
    SquareLayer(const ls::CkksParameter& param_in);
    virtual std::vector<ls::CkksCiphertext> call(ls::CkksContext& ctx, const std::vector<ls::CkksCiphertext>& x);
    virtual Feature2DEncrypted call(ls::CkksContext& ctx, const Feature2DEncrypted& x);
    virtual Feature0DEncrypted call(ls::CkksContext& ctx, const Feature0DEncrypted& x);

    template <int dim> Array<double, dim> run_plaintext(const Array<double, dim>& x) {
        return x.apply([](double e) { return e * e; });
    }
};
