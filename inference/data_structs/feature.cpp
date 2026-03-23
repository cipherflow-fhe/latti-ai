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

#include "feature.h"
#include "util.h"
#include <sstream>

using namespace std;
using namespace cxx_sdk_v2;

int32_t bitlength = RING_MOD_BIT;

Bytes save_ct(const CkksCiphertext& ct, const CkksParameter& param_h) {
    auto vec = ct.serialize(param_h);
    return vec;
}
CkksCiphertext load_ct(Bytes& vec) {
    auto y_ct = CkksCiphertext::deserialize(vec);
    return y_ct;
}

int64_t gen_random_for_share(int r_bitlength) {
    const int rand_bitlength = 16;
    int n_rand = div_ceil(r_bitlength, rand_bitlength);
    uint64_t mask = (1 << (r_bitlength % rand_bitlength)) - 1;
    int64_t result = rand() & mask;
    mask = (1 << rand_bitlength) - 1;
    for (int i = 1; i < n_rand; i++) {
        result = (result << rand_bitlength) + (rand() & mask);
    }
    if (rand() % 2 == 1) {
        result = -result;
    }
    return result;
}

void parallel_for(int n, int n_thread, CkksContext& context, const function<void(CkksContext&, int)>& fn) {
    int n_group = div_ceil(n, n_thread);
    context.resize_copies(n_thread);
#pragma omp parallel for num_threads(n_thread)
    for (int thread_idx = 0; thread_idx < n_thread; thread_idx++) {
        CkksContext& context_copy = context.get_copy(thread_idx);
        for (int group_idx = 0; group_idx < n_group; group_idx++) {
            int i = group_idx * n_thread + thread_idx;
            if (i >= n) {
                continue;
            }
            fn(context_copy, i);
        }
    }
}

void parallel_for(int n, int n_thread, CkksBtpContext& context, const function<void(CkksBtpContext&, int)>& fn) {
    int n_group = div_ceil(n, n_thread);
    context.resize_copies(n_thread);
#pragma omp parallel for num_threads(n_thread)
    for (int thread_idx = 0; thread_idx < n_thread; thread_idx++) {
        CkksBtpContext& context_copy = context.get_copy(thread_idx);
        for (int group_idx = 0; group_idx < n_group; group_idx++) {
            int i = group_idx * n_thread + thread_idx;
            if (i >= n) {
                continue;
            }
            fn(context_copy, i);
        }
    }
}

void parallel_for_with_extra_level_context(int n,
                                           int n_thread,
                                           CkksContext& context,
                                           const function<void(CkksContext&, CkksContext&, int)>& fn) {
    int n_group = div_ceil(n, n_thread);
    context.resize_copies(n_thread);
    CkksContext& extra_level_context = context.get_extra_level_context();
    extra_level_context.resize_copies(n_thread);
#pragma omp parallel for num_threads(n_thread)
    for (int thread_idx = 0; thread_idx < n_thread; thread_idx++) {
        CkksContext& context_copy = context.get_copy(thread_idx);
        CkksContext& extra_level_context_copy = extra_level_context.get_copy(thread_idx);
        for (int group_idx = 0; group_idx < n_group; group_idx++) {
            int i = group_idx * n_thread + thread_idx;
            if (i >= n) {
                continue;
            }
            fn(context_copy, extra_level_context_copy, i);
        }
    }
}

FeatureEncrypted::FeatureEncrypted() : ckks_scale{DEFAULT_SCALE}, multiplier{1.0} {}

FeatureEncrypted::~FeatureEncrypted() {}

FeatureShare::FeatureShare(uint64_t q, int s) {
    ring_mod = q;
    scale_ord = s;
}
