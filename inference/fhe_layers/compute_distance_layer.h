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
#include <vector>
#include "layer.h"
#include "../data_structs/feature0d.h"

class ComputeDistanceLayer : public Layer {
public:
    ComputeDistanceLayer(const ls::CkksParameter& param_in,
                         uint32_t dim,
                         double norm2_min,
                         double norm2_max,
                         int nr_iterations);

    void prepare_weight(const std::vector<double>& gallery, uint32_t level);

    Feature0DEncrypted run(ls::CkksBtpContext& ctx, const Feature0DEncrypted& query) const;

    double run_plaintext(const std::vector<double>& query, const std::vector<double>& gallery) const;

private:
    uint32_t dim_;
    double norm2_min_;
    double norm2_max_;
    int nr_iterations_;
    ls::CkksPlaintext gallery_pt_;
    bool has_gallery_pt_ = false;
};
