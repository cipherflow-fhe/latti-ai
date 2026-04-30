/*
 * Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "layer.h"
#include "../data_structs/feature0d.h"

class SoftmaxLayer : public Layer {
public:
    SoftmaxLayer(const ls::CkksParameter& param_in, uint32_t num_channels, int level_in,
                 double scale_in, uint32_t skip = 1);

    Feature0DEncrypted run(ls::CkksContext& ctx, const Feature0DEncrypted& x);

private:
    uint32_t num_channels_;
    uint32_t skip_;
    uint32_t n_slots_;
    uint32_t n_channel_per_ct_;
    double scale_;
    int level_;

    std::vector<double> exp_coeffs_;
    std::vector<double> inv_coeffs_;

    ls::CkksCiphertext poly_exp(ls::CkksContext& ctx, const ls::CkksCiphertext& x);
    ls::CkksCiphertext poly_inv(ls::CkksContext& ctx, const ls::CkksCiphertext& x);
    ls::CkksCiphertext rotate_sum(ls::CkksContext& ctx, const ls::CkksCiphertext& ct,
                                  uint32_t step, uint32_t n_terms);
    ls::CkksCiphertext broadcast(ls::CkksContext& ctx, const ls::CkksCiphertext& ct, uint32_t n_slots);
};
