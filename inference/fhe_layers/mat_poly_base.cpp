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

#include "mat_poly_base.h"
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <utility>

using namespace std;
using namespace lattisense;

namespace {

int ceil_log2_int(int n) {
    int depth = 0;
    int span = 1;
    while (span < n) {
        span <<= 1;
        depth++;
    }
    return depth;
}

int next_power_of_two_int(int n) {
    int span = 1;
    while (span < n) {
        span <<= 1;
    }
    return span;
}

int stockmeyer_baby_poly_count(int order) {
    return (order + 4) / 4;
}

}  // namespace

MatPolyBase::MatPolyBase(const CkksParameter& param_in, Array<double, 2>&& coeffs_in, uint32_t level_in, int order_in)
    : Layer(param_in), order(order_in), coeffs_(move(coeffs_in)) {
    level_ = level_in;
    N = param_in.get_n();
}

int MatPolyBase::compute_stockmeyer_level_cost(int order) {
    if (order <= 0 || order >= 64) {
        throw invalid_argument("MatPolyBase Stockmeyer supports only order < 64");
    }
    if (order < 2) {
        return 1;
    }

    int n_baby_polys = stockmeyer_baby_poly_count(order);
    return 2 + ceil_log2_int(n_baby_polys);
}

void MatPolyBase::init_stockmeyer() {
    if (stockmeyer_initialized) {
        return;
    }

    if (order <= 0 || order >= 64) {
        throw invalid_argument("MatPolyBase Stockmeyer supports only order < 64");
    }

    int level_cost = compute_stockmeyer_level_cost(order);
    if ((int)level_ < level_cost) {
        throw invalid_argument("MatPolyBase Stockmeyer input level is too low for order " + to_string(order));
    }

    modulus.clear();
    for (int i = 0; i <= (int)level_; i++) {
        modulus.push_back(param_.get_q(i));
    }

    stockmeyer_baby_steps = 4;
    stockmeyer_n_baby_polys = stockmeyer_baby_poly_count(order);
    stockmeyer_output_level = (int)level_ - level_cost;

    compute_stockmeyer_power_info();
    compute_coefficient_scales_stockmeyer(cached_stockmeyer_coeff_scale, cached_stockmeyer_level_order);
    stockmeyer_initialized = true;
}

void MatPolyBase::compute_stockmeyer_power_info() {
    double S = param_.get_default_scale();
    stockmeyer_powers.clear();

    stockmeyer_powers[1] = {0, (int)level_, S, 0, 0, true};

    auto add_square_power = [&](int power, int half_power) {
        const auto& half = stockmeyer_powers.at(half_power);
        int result_level = half.level - 1;
        if (result_level < 0 || half.level >= (int)modulus.size()) {
            throw invalid_argument("MatPolyBase Stockmeyer power level is out of range");
        }
        double result_scale = (half.scale / modulus[half.level]) * half.scale;
        stockmeyer_powers[power] = {half.depth + 1, result_level, result_scale, half_power, half_power, true};
    };

    int max_power = 1;
    if (order >= 2) {
        max_power = 2;
    }
    if (stockmeyer_n_baby_polys > 1) {
        int tree_span = next_power_of_two_int(stockmeyer_n_baby_polys);
        max_power = std::max(max_power, 2 * tree_span);
    }

    for (int power = 2; power <= max_power; power <<= 1) {
        add_square_power(power, power / 2);
    }
}

void MatPolyBase::compute_coefficient_scales_stockmeyer(std::map<int, double>& coeff_scale,
                                                        std::map<int, int>& level_order) {
    coeff_scale.clear();
    level_order.clear();

    double S = param_.get_default_scale();
    double A1 = stockmeyer_powers.at(1).scale;

    stockmeyer_baby_poly_output_scale.assign(stockmeyer_n_baby_polys, 0.0);
    stockmeyer_baby_poly_output_level.assign(stockmeyer_n_baby_polys, -1);

    int Lout = stockmeyer_output_level;
    int tree_span = next_power_of_two_int(stockmeyer_n_baby_polys);

    function<void(int, int, int, int, double)> assign_targets = [&](int start, int span, int actual_count,
                                                                    int target_level, double target_scale) {
        if (actual_count <= 0) {
            return;
        }
        if (span == 1) {
            stockmeyer_baby_poly_output_level[start] = target_level;
            stockmeyer_baby_poly_output_scale[start] = target_scale;
            return;
        }

        int half = span / 2;
        int left_count = std::min(actual_count, half);
        int right_count = actual_count - left_count;
        assign_targets(start, half, left_count, target_level, target_scale);

        if (right_count > 0) {
            int combine_power = stockmeyer_baby_steps * half;
            const auto& power_info = stockmeyer_powers.at(combine_power);
            int right_level = target_level + 1;
            double right_scale = (target_scale / power_info.scale) * param_.get_q(right_level);
            assign_targets(start + half, half, right_count, right_level, right_scale);
        }
    };

    assign_targets(0, tree_span, stockmeyer_n_baby_polys, Lout, S);

    for (int j = 0; j < stockmeyer_n_baby_polys; j++) {
        int target_level = stockmeyer_baby_poly_output_level[j];
        double target_scale = stockmeyer_baby_poly_output_scale[j];
        if (target_level < 0 || target_level > (int)level_) {
            throw invalid_argument("MatPolyBase Stockmeyer baby polynomial target level is out of range");
        }

        int base = j * stockmeyer_baby_steps;
        if (base <= order) {
            level_order[base] = target_level;
            coeff_scale[base] = target_scale;
        }
        if (base + 1 <= order) {
            if (target_level + 1 > (int)level_) {
                throw invalid_argument("MatPolyBase Stockmeyer linear coefficient target level is out of range");
            }
            level_order[base + 1] = target_level + 1;
            coeff_scale[base + 1] = (target_scale / A1) * param_.get_q(target_level + 1);
        }
        if (base + 2 <= order) {
            if (target_level + 1 > (int)level_) {
                throw invalid_argument("MatPolyBase Stockmeyer quadratic coefficient target level is out of range");
            }
            double A2 = stockmeyer_powers.at(2).scale;
            level_order[base + 2] = target_level + 1;
            coeff_scale[base + 2] = (target_scale / A2) * param_.get_q(target_level + 1);
        }
        if (base + 3 <= order) {
            if (target_level + 2 > (int)level_) {
                throw invalid_argument("MatPolyBase Stockmeyer cubic coefficient target level is out of range");
            }
            double A2 = stockmeyer_powers.at(2).scale;
            level_order[base + 3] = target_level + 2;
            coeff_scale[base + 3] =
                ((target_scale / A2) * param_.get_q(target_level + 1) / A1) * param_.get_q(target_level + 2);
        }
    }
}
