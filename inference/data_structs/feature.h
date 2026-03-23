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
#include <array>
#include <cxx_sdk_v2/cxx_fhe_task.h>
#include "../common.h"

using namespace std;
using namespace cxx_sdk_v2;

enum class PackType { MultChannelPack, SinglePack, MultiplexedPack, ParMultiplexedPack, InterleavedDecompositionPack };

enum class DecryptType { RESHAPE, SPARSE };
enum class ExecuteType { FPGA, SDK };

class FeatureEncrypted {
public:
    CkksContext* context = nullptr;
    uint32_t dim = 0;
    uint32_t n_channel = 0;
    uint32_t n_channel_per_ct = 0;
    uint32_t level = 0;
    uint32_t matmul_block_size = 0;
    double ckks_scale = 0;
    double multiplier = 0;

    FeatureEncrypted();
    virtual ~FeatureEncrypted();

    virtual void deserialize(const Bytes& bytes) {};
};

class FeatureShare {
public:
    FeatureShare(uint64_t q, int s);

    uint64_t ring_mod;
    int scale_ord;
    Array<uint64_t, 1> data;
};

int64_t gen_random_for_share(int r_bitlength);
double uint64_to_double(uint64_t input, double scale, uint64_t ring_mod);
uint64_t double_to_uint64(double input, double scale, uint64_t ring_mod);

void parallel_for(int n, int n_thread, CkksContext& context, const function<void(CkksContext&, int)>& fn);

void parallel_for(int n, int n_thread, CkksBtpContext& context, const function<void(CkksBtpContext&, int)>& fn);

void parallel_for_with_extra_level_context(int n,
                                           int n_thread,
                                           CkksContext& context,
                                           const function<void(CkksContext&, CkksContext&, int)>& fn);

inline CkksCiphertext drop_level_to(const CkksCiphertext& x, CkksContext& ctx, int level_pre, int level_next) {
    int level_diff = level_pre - level_next;
    assert(level_diff > 0);
    CkksCiphertext res = ctx.drop_level(x);
    for (int i = 1; i < level_diff; i++) {
        res = ctx.drop_level(res);
    }
    return res;
}

// Include dimension-specific feature headers so that including "feature.h"
// continues to provide all Feature*Encrypted and Feature*Share types.
#include "feature0d.h"
#include "feature1d.h"
#include "feature2d.h"
