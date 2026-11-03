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

#include <cxx_sdk_v2/cxx_fhe_task.h>

#include <cstdint>
#include <vector>

namespace fhe_api {

using namespace fhe_ops_lib;

/**
 * Encode a coefficient-domain (ring-t) plaintext with the given scale.
 *
 * The v6 SDK no longer exposes CkksContext::encode_ringt(); instead the
 * plaintext carries its own metadata and CkksContext::encode() dispatches on
 * it. log_slots defaults to the ring's maximum (N slots for the
 * conjugate-invariant ring, N/2 for the standard ring).
 */
inline CkksPlaintext encode_ringt(CkksContext& ctx, const std::vector<double>& values, double scale) {
    const auto& param = ctx.parameter();
    CkksPlaintext pt(param, plaintext_metadata(/*is_ringt=*/true,
                                               /*is_batched=*/true,
                                               /*level=*/0,
                                               /*log_slots=*/param.log_max_slots(),
                                               /*scale=*/scale));
    ctx.encode(values, pt);
    return pt;
}

/**
 * Encode a slot-domain plaintext at the given level and scale.
 *
 * The old API returned a CkksPlaintext; the v6 encode() takes an in/out
 * plaintext, so this helper preserves the old call shape.
 */
inline CkksPlaintext encode(CkksContext& ctx, const std::vector<double>& values, int level, double scale) {
    const auto& param = ctx.parameter();
    CkksPlaintext pt(param, plaintext_metadata(/*is_ringt=*/false,
                                               /*is_batched=*/true,
                                               level,
                                               param.log_max_slots(),
                                               scale,
                                               /*is_ntt=*/true));
    ctx.encode(values, pt);
    return pt;
}

}  // namespace fhe_api
