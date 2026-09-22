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
using namespace lattisense;

int32_t bitlength = RING_MOD_BIT;

Bytes save_ct(const CkksCiphertext& ct, const CkksParameter& param_h) {
    auto vec = ct.serialize();
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

// lattisense v6 移除了 CkksContext::get_copy / resize_copies，FheContext 禁止拷贝。
// 暂时串行化，保证功能正确；等 lattisense 提供线程安全模型后再恢复并行。
void parallel_for(int n, int n_thread, CkksContext& context, const function<void(CkksContext&, int)>& fn) {
    (void)n_thread;
    for (int i = 0; i < n; i++) {
        fn(context, i);
    }
}

// lattisense v6 移除了 get_extra_level_context()。
// v6 中 level 是 ciphertext/plaintext 自身的属性，不挂在 context 上，
// 因此 extra_level_context 语义上等同于主 context。先用同一个 context 传两次。
void parallel_for_with_extra_level_context(int n,
    int n_thread,
    CkksContext& context,
    const function<void(CkksContext&, CkksContext&, int)>& fn) {
    (void)n_thread;
    for (int i = 0; i < n; i++) {
        fn(context, context, i);
    }
}

FeatureEncrypted::FeatureEncrypted() {}

FeatureEncrypted::~FeatureEncrypted() {}

FeatureShare::FeatureShare(uint64_t q, int s) {
    ring_mod = q;
    scale_ord = s;
}
