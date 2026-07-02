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
#include <cmath>
#include <functional>
#include <limits>
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

void MatPolyBase::compute_all_powers() {
    powers[1] = {0, (int)level_, param_.get_default_scale(), 0, 0, true};
    for (int n = 2; n <= order; n++) {
        compute_power(n);
    }
}

void MatPolyBase::compute_power(int n) {
    if (powers.find(n) != powers.end() && powers[n].computed) {
        return;
    }

    int best_depth = std::numeric_limits<int>::max();
    int best_a = 1, best_b = n - 1;

    for (int a = 1; a <= n / 2; a++) {
        int b = n - a;
        if (powers.find(a) == powers.end()) {
            compute_power(a);
        }
        if (powers.find(b) == powers.end()) {
            compute_power(b);
        }

        int depth = std::max(powers[a].depth, powers[b].depth) + 1;
        if (depth < best_depth) {
            best_depth = depth;
            best_a = a;
            best_b = b;
        } else if (depth == best_depth && std::abs(a - b) < std::abs(best_a - best_b)) {
            best_a = a;
            best_b = b;
        }
    }

    int result_level = std::min(powers[best_a].level, powers[best_b].level) - 1;
    double result_scale = powers[best_a].scale * powers[best_b].scale;
    if (result_level >= 0 && result_level + 1 < (int)modulus.size()) {
        result_scale = result_scale / modulus[result_level + 1];
    }

    powers[n] = {best_depth, result_level, result_scale, best_a, best_b, true};
}

MatPolyPowerInfo MatPolyBase::get_power_info(int n) const {
    auto it = powers.find(n);
    if (it != powers.end()) {
        return it->second;
    }
    return {-1, -1, 0.0, 0, 0, false};
}

int MatPolyBase::compute_bsgs_level_cost(int order) {
    if (order <= 1) {
        return 1;
    }

    int baby_steps = (int)ceil(sqrt(order + 1));
    int giant_steps = (int)ceil((double)(order + 1) / baby_steps);

    struct PInfo {
        int depth;
        int a;
        int b;
    };

    std::map<int, PInfo> pinfo;
    pinfo[1] = {0, 0, 0};
    for (int n = 2; n <= order; n++) {
        int best_d = std::numeric_limits<int>::max();
        int best_a = 1, best_b = n - 1;
        for (int a = 1; a <= n / 2; a++) {
            int b = n - a;
            int d = std::max(pinfo[a].depth, pinfo[b].depth) + 1;
            if (d < best_d || (d == best_d && std::abs(a - b) < std::abs(best_a - best_b))) {
                best_d = d;
                best_a = a;
                best_b = b;
            }
        }
        pinfo[n] = {best_d, best_a, best_b};
    }

    std::set<int> required, to_compute;
    for (int i = 1; i <= baby_steps; i++) {
        required.insert(i);
        to_compute.insert(i);
    }
    for (int g = 1; g < giant_steps; g++) {
        int gp = g * baby_steps;
        if (gp <= order) {
            required.insert(gp);
            to_compute.insert(gp);
        }
    }

    std::function<void(int)> add_deps = [&](int n) {
        if (n <= 1) {
            return;
        }
        if (pinfo[n].a > 1) {
            to_compute.insert(pinfo[n].a);
            add_deps(pinfo[n].a);
        }
        if (pinfo[n].b > 1) {
            to_compute.insert(pinfo[n].b);
            add_deps(pinfo[n].b);
        }
    };
    for (int p : std::set<int>(required)) {
        add_deps(p);
    }

    std::map<int, int> pd;
    pd[1] = 0;
    for (int n : to_compute) {
        if (n <= 1) {
            continue;
        }
        pd[n] = std::max(pd[pinfo[n].a], pd[pinfo[n].b]) + 1;
    }

    int max_d = 0;
    for (int p : required) {
        max_d = std::max(max_d, pd[p]);
    }
    return max_d + 1;
}

int MatPolyBase::compute_stockmeyer_level_cost(int order) {
    if (order < 0 || order >= 64) {
        throw invalid_argument("MatPolyBase Stockmeyer supports only order < 64");
    }
    if (order < 2) {
        return 1;
    }

    int n_baby_polys = stockmeyer_baby_poly_count(order);
    return 2 + ceil_log2_int(n_baby_polys);
}

void MatPolyBase::init_bsgs() {
    if (bsgs_initialized) {
        return;
    }

    baby_steps = (int)ceil(sqrt(order + 1));
    bsgs_giant_steps = (int)ceil((double)(order + 1) / baby_steps);

    modulus.clear();
    for (int i = 0; i <= (int)level_; i++) {
        modulus.push_back(param_.get_q(i));
    }

    powers.clear();
    compute_all_powers();
    determine_required_powers_bsgs();
    compute_coefficient_scales_bsgs(cached_bsgs_coeff_scale, cached_bsgs_level_order);
    bsgs_initialized = true;
}

void MatPolyBase::determine_required_powers_bsgs() {
    required_powers.clear();

    for (int i = 1; i <= baby_steps; i++) {
        required_powers.insert(i);
    }

    for (int g = 1; g < bsgs_giant_steps; g++) {
        int giant_power = g * baby_steps;
        if (giant_power <= order) {
            required_powers.insert(giant_power);
        }
    }
}

void MatPolyBase::compute_coefficient_scales_bsgs(std::map<int, double>& coeff_scale, std::map<int, int>& level_order) {
    double S = param_.get_default_scale();

    int max_depth = 0;
    int max_power_level = level_;
    for (int p : required_powers) {
        MatPolyPowerInfo info = get_power_info(p);
        if (info.depth > max_depth) {
            max_depth = info.depth;
            max_power_level = info.level;
        }
    }
    bsgs_output_level = max_power_level - 1;

    baby_poly_output_scale.resize(bsgs_giant_steps);
    baby_poly_output_level.resize(bsgs_giant_steps);

    for (int g = 0; g < bsgs_giant_steps; g++) {
        if (g == 0) {
            baby_poly_output_scale[g] = S;
            baby_poly_output_level[g] = bsgs_output_level;
        } else {
            int giant_power = g * baby_steps;
            if (giant_power > order) {
                break;
            }

            MatPolyPowerInfo gp_info = get_power_info(giant_power);
            int level_mult = bsgs_output_level + 1;
            baby_poly_output_level[g] = level_mult;
            baby_poly_output_scale[g] = S * param_.get_q(level_mult) / gp_info.scale;
        }

        int start_idx = g * baby_steps;
        int end_idx = std::min(start_idx + baby_steps - 1, order);
        double target_scale = baby_poly_output_scale[g];
        int target_level = baby_poly_output_level[g];

        for (int idx = start_idx; idx <= end_idx; idx++) {
            int baby_step = idx - start_idx;
            if (baby_step == 0) {
                level_order[idx] = target_level;
                coeff_scale[idx] = target_scale;
            } else {
                MatPolyPowerInfo x_info = get_power_info(baby_step);
                coeff_scale[idx] = target_scale * param_.get_q(target_level + 1) / x_info.scale;
                level_order[idx] = target_level + 1;
            }
        }
    }
}

void MatPolyBase::init_stockmeyer() {
    if (stockmeyer_initialized) {
        return;
    }

    if (order < 0 || order >= 64) {
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
        double result_scale = half.scale * half.scale / modulus[half.level];
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
            double right_scale = target_scale * param_.get_q(right_level) / power_info.scale;
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
            coeff_scale[base + 1] = target_scale * param_.get_q(target_level + 1) / A1;
        }
        if (base + 2 <= order) {
            if (target_level + 1 > (int)level_) {
                throw invalid_argument("MatPolyBase Stockmeyer quadratic coefficient target level is out of range");
            }
            double A2 = stockmeyer_powers.at(2).scale;
            level_order[base + 2] = target_level + 1;
            coeff_scale[base + 2] = target_scale * param_.get_q(target_level + 1) / A2;
        }
        if (base + 3 <= order) {
            if (target_level + 2 > (int)level_) {
                throw invalid_argument("MatPolyBase Stockmeyer cubic coefficient target level is out of range");
            }
            double A2 = stockmeyer_powers.at(2).scale;
            level_order[base + 3] = target_level + 2;
            coeff_scale[base + 3] =
                target_scale * param_.get_q(target_level + 1) * param_.get_q(target_level + 2) / (A2 * A1);
        }
    }
}
