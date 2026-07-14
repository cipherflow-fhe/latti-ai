/*
 * Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#pragma once

#include <vector>

#include "layer.h"
#include "../data_structs/feature0d.h"
#include "../data_structs/feature1d.h"
#include "../data_structs/feature2d.h"
#include "../data_structs/feature_mat.h"

/** Pack two aligned real ciphertext lanes into one complex lane: z = a + i*b. */
class ComplexPackLayer : public Layer {
public:
    explicit ComplexPackLayer(const ls::CkksParameter& param_in);

    std::vector<ls::CkksCiphertext>
    run(ls::CkksContext& ctx, const std::vector<ls::CkksCiphertext>& a, const std::vector<ls::CkksCiphertext>& b) const;

    ls::CkksCiphertext run(ls::CkksContext& ctx, const ls::CkksCiphertext& a, const ls::CkksCiphertext& b) const;

    Feature0DEncrypted run(ls::CkksContext& ctx, const Feature0DEncrypted& a, const Feature0DEncrypted& b) const;
    Feature1DEncrypted run(ls::CkksContext& ctx, const Feature1DEncrypted& a, const Feature1DEncrypted& b) const;
    Feature2DEncrypted run(ls::CkksContext& ctx, const Feature2DEncrypted& a, const Feature2DEncrypted& b) const;
    FeatureMatEncrypted run(ls::CkksContext& ctx, const FeatureMatEncrypted& a, const FeatureMatEncrypted& b) const;
};
