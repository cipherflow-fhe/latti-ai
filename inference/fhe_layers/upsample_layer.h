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
#include <vector>
#include <cstdint>
#include <iostream>
#include "layer.h"
#include "util.h"
#include "data_structs/feature2d.h"

class UpsampleLayer : public Layer {
public:
    UpsampleLayer(const ls::CkksParameter& param_in,
                  const Duo& stride_in,
                  const Duo& upsample_factor_in,
                  const int& level_in,
                  const int& n_channel_in,
                  const int& n_channel_per_ct_in);
    virtual void prepare_data();
    Feature2DEncrypted run(ls::CkksContext& ctx, const Feature2DEncrypted& x);
    Array<double, 3> upsample_with_zero(const Array<double, 3>& x);

protected:
    Duo stride;
    Duo upsample_factor;
    ls::CkksCiphertext zero_vector_encrypted;
    ls::CkksPlaintext zero_vector_encoded;
    int n_channel_per_ct;
    int n_channel;
};
