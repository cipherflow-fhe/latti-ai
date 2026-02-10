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

#include <stdio.h>
#include <cstdint>
#include <string>
#include <vector>
#include "data_structs/constants.h"
#include "feature.h"

enum class DecryptTypeTiny { RESHAPE, SPARSE };

class FeatureEncrypted_tiny {
public:
    FeatureEncrypted_tiny();
    virtual ~FeatureEncrypted_tiny();
    CkksContext* context;
    uint32_t dim;
    uint32_t n_channel;
    uint32_t n_channel_per_ct;
    uint32_t level;
};

class Feature2DEncrypted_tiny : public FeatureEncrypted_tiny {
public:
    Feature2DEncrypted_tiny(CkksContext* context_in);
    virtual void pack(std::vector<std::vector<std::vector<double>>>& feature_mg, bool is_symmetric = false);
    void pack1(std::vector<std::vector<std::vector<double>>>& feature_mg, int pack, bool is_symmetric = false);
    virtual std::vector<std::vector<std::vector<double>>> unpack();

    uint32_t shape[2];
    uint32_t skip[2];

    std::vector<cxx_sdk_v2::CkksCiphertext> data_handle;
};

class Feature0DEncrypted_tiny : public FeatureEncrypted_tiny {
public:
    Feature0DEncrypted_tiny(CkksContext* context_in);
    void pack(std::vector<double>& feature_mg, bool is_symmetric = false);
    void pack1(std::vector<double>& feature_mg, bool is_symmetric = false);
    void pack(std::vector<double>& feature_mg, uint32_t pack_num, uint32_t level);
    std::vector<double> unpack(DecryptTypeTiny dec_type, int pack_num);

    uint32_t pack_type;
    uint32_t skip;
    std::vector<CkksCiphertext> data_handle;
};
CkksCiphertext drop_level_to(CkksCiphertext& x, CkksContext& ctx, int level_pre, int level_next);