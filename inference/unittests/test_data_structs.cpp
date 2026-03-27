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

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "data_structs/feature.h"
#include "util.h"
#include "ut_util.h"

using namespace std;
using namespace cxx_sdk_v2;

TEST_CASE("Feature2DEncrypted serialization", "") {
    CkksParameter paramter = CkksParameter::create_parameter(8192);
    CkksContext context = CkksContext::create_random_context(paramter);

    uint32_t shape[] = {32, 32};
    uint32_t skip[] = {1, 1};
    uint32_t n_channel = 4;
    uint32_t n_channel_per_ct = 4;

    auto x_mg = gen_random_array<3>({4, shape[0], shape[1]}, 1);
    Feature2DEncrypted x_e(&context, 1);
    x_e.pack_multiple_channel(x_mg);

    auto ss = x_e.serialize();
    Feature2DEncrypted y_e(&context, 1);
    y_e.deserialize(ss);
    auto y_mg = y_e.unpack_multiple_channel();

    auto compare_res = compare(x_mg, y_mg);
    REQUIRE(compare_res.max_error < 1.0e-3 * compare_res.max_abs);
}
