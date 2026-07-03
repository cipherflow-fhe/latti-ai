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

#include "layer.h"
#include <map>
#include <vector>

struct MatPolyPowerInfo {
    int depth;
    int level;
    double scale;
    int decomp_a;
    int decomp_b;
    bool computed;
};

class MatPolyBase : public Layer {
public:
    MatPolyBase(const ls::CkksParameter& param_in, Array<double, 2>&& coeffs_in, uint32_t level_in, int order_in);

    static int compute_stockmeyer_level_cost(int order);

protected:
    int N = 0;
    int order = 0;
    Array<double, 2> coeffs_;

    void init_stockmeyer();
    void compute_stockmeyer_power_info();
    void compute_coefficient_scales_stockmeyer(std::map<int, double>& coeff_scale, std::map<int, int>& level_order);

    std::vector<double> modulus;

    int stockmeyer_baby_steps = 4;
    int stockmeyer_n_baby_polys = 0;
    std::map<int, MatPolyPowerInfo> stockmeyer_powers;
    std::vector<double> stockmeyer_baby_poly_output_scale;
    std::vector<int> stockmeyer_baby_poly_output_level;
    int stockmeyer_output_level = 0;
    std::map<int, double> cached_stockmeyer_coeff_scale;
    std::map<int, int> cached_stockmeyer_level_order;
    bool stockmeyer_initialized = false;
};
