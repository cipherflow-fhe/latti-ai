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
#include "feature.h"

class Feature1DEncrypted : public FeatureEncrypted {
public:
    Feature1DEncrypted(ls::CkksContext* context_in, int ct_level, uint32_t skip_in = 1);
    virtual void pack(Array<double, 2>& feature_mg, bool is_symmetric = false, double scale_in = DEFAULT_SCALE);
    virtual Array<double, 2> unpack() const;
    virtual void
    pack_multiplexed(const Array<double, 2>& feature_mg, bool is_symmetric = false, double scale_in = DEFAULT_SCALE);
    virtual Array<double, 2> unpack_multiplexed() const;
    uint32_t shape = 0;
    uint32_t skip = 0;
    std::vector<ls::CkksCiphertext> data;
    std::vector<ls::CkksCompressedCiphertext> data_compress;
};
