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
#include "layer.h"
#include "data_structs/feature2d.h"
#include "util.h"

class AddLayer : public Layer {
public:
    AddLayer(const ls::CkksParameter& param_in);
    double target_ckks_scale = DEFAULT_SCALE;
    Feature2DEncrypted run(ls::CkksContext& ctx, const Feature2DEncrypted& x0, const Feature2DEncrypted& x1);
    Array<double, 3> run_plaintext(const Array<double, 3>& x0, const Array<double, 3>& x1);
    void
    add(ls::CkksContext* ctx, const Feature2DEncrypted& x0, const Feature2DEncrypted& x1, Feature2DEncrypted& result);
};

class DropLevelLayer : public Layer {
public:
    DropLevelLayer();
    void
    run(ls::CkksContext& ctx, const Feature2DEncrypted& x, Feature2DEncrypted& result, int level_in, int level_out);
};
