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
#include "poly_relu_base.h"
#include "../data_structs/feature1d.h"

// PolyRelu for Feature1DEncrypted.
//
// Two packing modes, matching Feature1DEncrypted::pack / pack_multiplexed:
//
//   Mode 1 — skip pack (wasted slots):
//     n_channel_per_ct = N/2 / (shape * skip)
//     channel ch (CT-local), position i → slot = ch * shape * skip + i * skip
//     Remaining (skip-1) slots per position are zero-padded.
//
//   Mode 2 — multiplexed/interleaved pack (no wasted slots):
//     n_channel_per_ct = (N/2 / (shape * skip)) * skip
//     channel j (CT-local), position i → slot = (j/skip)*shape*skip + i*skip + (j%skip)
//     Each block of (shape*skip) slots carries `skip` channels interleaved.
//
// Weight shape: [order+1, n_channel]
// Weight is per-channel (same value broadcast to all `shape` spatial positions).
class PolyRelu1D : public PolyReluBase {
public:
    // skip_in   : skip of the Feature1DEncrypted
    // shape_in  : shape of the Feature1DEncrypted
    // For mode 1: n_channel_per_ct = N/2 / (shape*skip)
    // For mode 2: call prepare_weight_bsgs_mux / prepare_weight_bsgs_mux_lazy instead
    PolyRelu1D(const ls::CkksParameter& param_in,
               Array<double, 2>&& weight_in,
               uint32_t level_in,
               int order_in,
               int skip_in,
               int shape_in);

    void prepare_weight() override {
        prepare_weight_bsgs();
    }
    void prepare_weight_lazy() override {
        prepare_weight_bsgs_lazy();
    }

    // Mode 1 — skip pack
    void prepare_weight_bsgs();
    void prepare_weight_bsgs_lazy();

    // Mode 2 — multiplexed/interleaved pack
    void prepare_weight_bsgs_mux();
    void prepare_weight_bsgs_mux_lazy();

    ls::CkksPlaintextRingt generate_weight_pt_for_bsgs(ls::CkksContext& ctx, int idx, int ct_idx) const override;

    // run uses whichever mode was prepared
    Feature1DEncrypted run(ls::CkksContext& ctx, const Feature1DEncrypted& x);
    Array<double, 2> run_plaintext(const Array<double, 2>& x);

    int skip;
    int shape;
    bool is_multiplexed = false;

private:
    int shape_with_skip;  // = shape * skip
    // n_channel_per_ct for mode 2 (larger than mode 1 by factor skip)
    int n_channel_per_ct_mux;

    ls::CkksPlaintextRingt generate_weight_pt_skip1d(ls::CkksContext& ctx, int idx, int ct_idx) const;
    ls::CkksPlaintextRingt generate_weight_pt_mux1d(ls::CkksContext& ctx, int idx, int ct_idx) const;
};
