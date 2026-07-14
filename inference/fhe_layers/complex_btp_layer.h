/*
 * Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#pragma once

#include <utility>
#include <vector>

#include "layer.h"
#include "../data_structs/feature0d.h"
#include "../data_structs/feature1d.h"
#include "../data_structs/feature2d.h"
#include "../data_structs/feature_mat.h"

/**
 * Refresh a complex CKKS lane and split it into its real and imaginary lanes.
 *
 * The input is z = a + i*b.  The layer first computes z/2, consuming one
 * multiplicative level, and then bootstraps it.  If r = BTP(z/2), then:
 *
 *   a = r + conj(r)
 *   b = (r - conj(r)) / i
 *
 * Therefore no multiplication by two is needed after bootstrapping.
 */
class ComplexBtpLayer : public Layer {
public:
    explicit ComplexBtpLayer(const ls::CkksParameter& param_in, bool use_complex_btp = true);

    // Generate the ring-t plaintext used by the mega-ag graph to multiply
    // the input by 1/2 before rescaling and bootstrapping.
    ls::CkksPlaintextRingt generate_half_pt(ls::CkksContext& ctx, int input_level) const;

    // Return {a, b} for a vector of ciphertexts encoding a + i*b.
    std::pair<std::vector<ls::CkksCiphertext>, std::vector<ls::CkksCiphertext>>
    run(ls::CkksBtpContext& ctx, const std::vector<ls::CkksCiphertext>& packed) const;

    // Construct a + i*b from two aligned ciphertext vectors, then run the
    // complex BTP path.  The construction itself does not consume a level.
    std::pair<std::vector<ls::CkksCiphertext>, std::vector<ls::CkksCiphertext>>
    run(ls::CkksBtpContext& ctx,
        const std::vector<ls::CkksCiphertext>& a,
        const std::vector<ls::CkksCiphertext>& b) const;

    std::pair<Feature0DEncrypted, Feature0DEncrypted> run(ls::CkksBtpContext& ctx,
                                                          const Feature0DEncrypted& packed) const;
    std::pair<Feature1DEncrypted, Feature1DEncrypted> run(ls::CkksBtpContext& ctx,
                                                          const Feature1DEncrypted& packed) const;
    std::pair<Feature2DEncrypted, Feature2DEncrypted> run(ls::CkksBtpContext& ctx,
                                                          const Feature2DEncrypted& packed) const;
    std::pair<FeatureMatEncrypted, FeatureMatEncrypted> run(ls::CkksBtpContext& ctx,
                                                            const FeatureMatEncrypted& packed) const;

private:
    ls::CkksCiphertext scale_by_half(ls::CkksBtpContext& ctx, const ls::CkksCiphertext& input) const;

    bool use_complex_btp_ = true;
};
