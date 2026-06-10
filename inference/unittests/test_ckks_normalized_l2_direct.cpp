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

#include <fhe_ops_lib/fhe_lib_v2.h>
#include "fhe_layers/compute_distance_layer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace fhe_ops_lib;
using namespace std;

namespace {

double dot_product(const vector<double>& a, const vector<double>& b) {
    REQUIRE(a.size() == b.size());
    double ret = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        ret += a[i] * b[i];
    }
    return ret;
}

double l2_norm_squared(const vector<double>& x) {
    return dot_product(x, x);
}

vector<double> normalize_vector(const vector<double>& x) {
    const double norm = sqrt(l2_norm_squared(x));
    REQUIRE(norm > 0.0);
    vector<double> ret(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        ret[i] = x[i] / norm;
    }
    return ret;
}

double inverse_sqrt_newton_raphson(double a, double a_min, double a_max, int iterations) {
    REQUIRE(a > 0.0);
    REQUIRE(a_min > 0.0);
    REQUIRE(a_max >= a_min);
    REQUIRE(iterations >= 0);

    double x = 0.5 * (1.0 / sqrt(a_min) + 1.0 / sqrt(a_max));
    for (int i = 0; i < iterations; ++i) {
        x = 1.5 * x - 0.5 * a * x * x * x;
    }
    return x;
}

double exact_normalized_l2_distance_squared(const vector<double>& a, const vector<double>& b) {
    const double a_norm = sqrt(l2_norm_squared(a));
    const double b_norm = sqrt(l2_norm_squared(b));
    const double cos = dot_product(a, b) / (a_norm * b_norm);
    return 2.0 - 2.0 * cos;
}

pair<double, double> calibrate_norm2_range(const vector<vector<double>>& vectors) {
    REQUIRE(!vectors.empty());
    double lo = l2_norm_squared(vectors.front());
    double hi = lo;
    for (const auto& v : vectors) {
        const double norm2 = l2_norm_squared(v);
        lo = min(lo, norm2);
        hi = max(hi, norm2);
    }
    return {lo, hi};
}

vector<double> single_slot(double value) {
    return {value};
}

vector<double> read_embedding_txt(const string& path) {
    ifstream file(path);
    REQUIRE(file.is_open());
    string text((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    replace(text.begin(), text.end(), ',', ' ');

    vector<double> values;
    istringstream stream(text);
    double value;
    while (stream >> value) {
        values.push_back(value);
    }
    REQUIRE(!values.empty());
    return values;
}

CkksCiphertext encrypt_vector(CkksContext& ctx, const vector<double>& values, int level, double scale) {
    auto pt = ctx.encode(values, level, scale);
    return ctx.encrypt_asymmetric(pt);
}

double decrypt_first_slot(CkksContext& ctx, const CkksCiphertext& ct) {
    cout<<"last_ct_level="<<ct.get_level()<<endl;
    auto pt = ctx.decrypt(ct);
    auto values = ctx.decode(pt);
    REQUIRE(!values.empty());
    return values.front();
}

CkksCiphertext sum_slots(CkksContext& ctx, CkksCiphertext ct, int dim) {
    REQUIRE(dim > 0);
    for (int step = 1; step < dim; step <<= 1) {
        auto rotated = ctx.rotate(ct, step);
        ct = ctx.add(ct, rotated);
    }
    return ct;
}

CkksCiphertext multiply_and_rescale(CkksContext& ctx, const CkksCiphertext& lhs, const CkksCiphertext& rhs, double scale) {
    auto product = ctx.mult(lhs, rhs);
    auto relined = ctx.relinearize(product);
    return ctx.rescale(relined, scale);
}

CkksCiphertext multiply_plain_and_rescale(CkksContext& ctx,
                                          const CkksCiphertext& ct,
                                          const vector<double>& values,
                                          double scale) {
    auto pt = ctx.encode(values, ct.get_level(), scale);
    auto product = ctx.mult_plain(ct, pt);
    return ctx.rescale(product, scale);
}

CkksCiphertext add_plain_scalar(CkksContext& ctx, const CkksCiphertext& ct, double value, double scale) {
    auto pt = ctx.encode(single_slot(value), ct.get_level(), scale);
    return ctx.add_plain(ct, pt);
}

CkksCiphertext drop_to_level(CkksContext& ctx, const CkksCiphertext& ct, int target_level) {
    REQUIRE(ct.get_level() >= target_level);
    const int levels = ct.get_level() - target_level;
    if (levels == 0) {
        return ct.copy();
    }
    return ctx.drop_level(ct, levels);
}

CkksCiphertext inverse_sqrt_one_iteration(CkksContext& ctx,
                                          const CkksCiphertext& a_ct,
                                          double a_min,
                                          double a_max,
                                          double scale) {
    const double x0 = 0.5 * (1.0 / sqrt(a_min) + 1.0 / sqrt(a_max));
    auto term = multiply_plain_and_rescale(ctx, a_ct, single_slot(-0.5 * x0 * x0 * x0), scale);
    return add_plain_scalar(ctx, term, 1.5 * x0, scale);
}

CkksCiphertext bootstrap_ciphertext(CkksBtpContext& ctx, const CkksCiphertext& ct, double scale) {
    auto input_scale = ct.get_scale();
    auto boot_ct = ct.copy();
    boot_ct.set_scale(scale);
    boot_ct = ctx.bootstrap(boot_ct);
    boot_ct.set_scale(input_scale);
    return boot_ct;
}

CkksCiphertext bootstrap_if_needed(CkksBtpContext& ctx, const CkksCiphertext& ct, double scale) {
    if (ct.get_level() - 3 > 0) {
        return ct.copy();
    }
    cout << "boots input_level=" << ct.get_level() << endl;
    auto boot_ct = bootstrap_ciphertext(ctx, ct, scale);
    cout << "boots output_level=" << boot_ct.get_level() << endl;
    return boot_ct;
}

CkksCiphertext inverse_sqrt_next_iteration(CkksBtpContext& ctx,
                                           const CkksCiphertext& a_ct,
                                           const CkksCiphertext& x_ct,
                                           double scale) {
    cout << "next_iteration enter x_level=" << x_ct.get_level() << " a_level=" << a_ct.get_level() << endl;
    auto x_work_ct = bootstrap_if_needed(ctx, x_ct, scale);
    if (x_work_ct.get_level() > a_ct.get_level()) {
        x_work_ct = drop_to_level(ctx, x_work_ct, a_ct.get_level());
    }
    auto x2_ct = multiply_and_rescale(ctx, x_work_ct, x_work_ct, scale);
    auto a_aligned_ct = drop_to_level(ctx, a_ct, x2_ct.get_level());
    auto ax2_ct = multiply_and_rescale(ctx, a_aligned_ct, x2_ct, scale);
    auto factor_ct = multiply_plain_and_rescale(ctx, ax2_ct, single_slot(-0.5), scale);
    factor_ct = add_plain_scalar(ctx, factor_ct, 1.5, scale);
    auto x_aligned_ct = drop_to_level(ctx, x_work_ct, factor_ct.get_level());
    auto ret = multiply_and_rescale(ctx, x_aligned_ct, factor_ct, scale);
    cout << "next_iteration exit x_level=" << ret.get_level() << endl;
    return ret;
}

CkksCiphertext inverse_sqrt_iterations(CkksBtpContext& ctx,
                                       const CkksCiphertext& a_ct,
                                       double a_min,
                                       double a_max,
                                       int iterations,
                                       double scale) {
    REQUIRE(iterations >= 1);
    auto x_ct = inverse_sqrt_one_iteration(ctx, a_ct, a_min, a_max, scale);
    cout << "iteration 1 exit x_level=" << x_ct.get_level() << " a_level=" << a_ct.get_level() << endl;
    for (int i = 1; i < iterations; ++i) {
        x_ct = inverse_sqrt_next_iteration(ctx, a_ct, x_ct, scale);
    }
    return x_ct;
}

}  // namespace

TEST_CASE("direct CKKS normalized L2 distance with plaintext-normalized gallery", "[ckks][distance]") {
    constexpr size_t dim = 128;
    constexpr int nr_iterations = 3;

    CkksBtpParameter btp_param = CkksBtpParameter::create_parameter();
    auto& param = btp_param.get_ckks_parameter();
    REQUIRE(param.get_n() == 65536);
    CkksBtpContext ctx = CkksBtpContext::create_random_context(btp_param);
    ctx.gen_rotation_keys();

    const double scale = param.get_default_scale();
    const int level = param.get_max_level();
    cout << "max_level=" << level << endl;
    REQUIRE(level >= 4 * nr_iterations - 1);

    const string query_path = string(SOURCE_PATH) + "/../examples/test_face/query_embedding.txt";
    const string gallery_path = string(SOURCE_PATH) + "/../examples/test_face/gallery_embedding.txt";
    const vector<double> query = read_embedding_txt(query_path);
    const vector<double> gallery = read_embedding_txt(gallery_path);
    REQUIRE(query.size() == dim);
    REQUIRE(gallery.size() == dim);

    const double query_norm2 = l2_norm_squared(query);
    cout<<"query_norm="<<query_norm2<<endl;
    const double norm2_min = 21.9;
    const double norm2_max = 75.2;
    const vector<double> gallery_normed = normalize_vector(gallery);

    vector<double> scaled_gallery_normed(dim);
    for (size_t i = 0; i < dim; ++i) {
        scaled_gallery_normed[i] = 2.0 * gallery_normed[i];
    }

    auto query_ct = encrypt_vector(ctx, query, level, scale);
    query_ct = ctx.drop_level(query_ct,15);
    cout<<"level="<<query_ct.get_level()<<endl;

    const auto compute_start = chrono::high_resolution_clock::now();
    auto dot2_ct = multiply_plain_and_rescale(ctx, query_ct, scaled_gallery_normed, scale);
    dot2_ct = sum_slots(ctx, std::move(dot2_ct), dim);
    cout << "dot2_level=" << dot2_ct.get_level() << endl;

    auto norm2_ct = multiply_and_rescale(ctx, query_ct, query_ct, scale);
    norm2_ct = sum_slots(ctx, std::move(norm2_ct), dim);
    cout << "norm2_level=" << norm2_ct.get_level() << endl;

    auto rsqrt_ct = inverse_sqrt_iterations(ctx, norm2_ct, norm2_min, norm2_max, nr_iterations, scale);
    auto dot2_aligned_ct = drop_to_level(ctx, dot2_ct, rsqrt_ct.get_level());
    auto cos2_ct = multiply_and_rescale(ctx, dot2_aligned_ct, rsqrt_ct, scale);

    auto dist2_ct = multiply_plain_and_rescale(ctx, cos2_ct, single_slot(-1.0), scale);
    dist2_ct = add_plain_scalar(ctx, dist2_ct, 2.0, scale);
    const auto compute_end = chrono::high_resolution_clock::now();
    const auto compute_ms = chrono::duration_cast<chrono::milliseconds>(compute_end - compute_start).count();
    cout << "encrypted_compute_time_ms=" << compute_ms << endl;

    const double encrypted_dist2 = decrypt_first_slot(ctx, dist2_ct);
    const double query_rsqrt = inverse_sqrt_newton_raphson(l2_norm_squared(query), norm2_min, norm2_max, nr_iterations);
    const double approx_dist2 = 2.0 - 2.0 * dot_product(query, gallery_normed) * query_rsqrt;
    const double exact_dist2 = exact_normalized_l2_distance_squared(query, gallery);
    cout<<"exact_dist2="<<exact_dist2<<endl;
    cout<<"encrypted_dist2="<<encrypted_dist2<<endl;
    REQUIRE(abs(encrypted_dist2 - approx_dist2) < 1.0e-2);
    REQUIRE(abs(approx_dist2 - exact_dist2) < 1.0e-2);
}

TEST_CASE("ComputeDistanceLayer CKKS normalized L2 distance", "[ckks][distance][fhe_layers]") {
    constexpr size_t dim = 128;
    constexpr int nr_iterations = 3;

    CkksBtpParameter btp_param = CkksBtpParameter::create_parameter();
    auto& param = btp_param.get_ckks_parameter();
    REQUIRE(param.get_n() == 65536);
    CkksBtpContext ctx = CkksBtpContext::create_random_context(btp_param);
    ctx.gen_rotation_keys();

    const double scale = param.get_default_scale();
    const int level = param.get_max_level();
    const double norm2_min = 21.9;
    const double norm2_max = 75.2;

    const string query_path = string(SOURCE_PATH) + "/../examples/test_face/query_embedding.txt";
    const string gallery_path = string(SOURCE_PATH) + "/../examples/test_face/gallery_embedding.txt";
    const vector<double> query = read_embedding_txt(query_path);
    const vector<double> gallery = read_embedding_txt(gallery_path);
    REQUIRE(query.size() == dim);
    REQUIRE(gallery.size() == dim);

    const vector<double> gallery_normed = normalize_vector(gallery);
    auto query_ct = encrypt_vector(ctx, query, level, scale);
    query_ct = ctx.drop_level(query_ct, 15);

    Feature0DEncrypted query_feature(&ctx, query_ct.get_level());
    query_feature.data.push_back(std::move(query_ct));
    query_feature.dim = 0;
    query_feature.n_channel = dim;
    query_feature.n_channel_per_ct = ctx.get_parameter().get_n() / 2;
    query_feature.skip = 1;
    query_feature.level = query_feature.data[0].get_level();

    ComputeDistanceLayer layer(param, dim, norm2_min, norm2_max, nr_iterations);
    layer.prepare_weight(gallery_normed, query_feature.level);
    const auto compute_start = chrono::high_resolution_clock::now();
    auto output = layer.run(ctx, query_feature);
    const auto compute_end = chrono::high_resolution_clock::now();
    const auto compute_ms = chrono::duration_cast<chrono::milliseconds>(compute_end - compute_start).count();
    cout << "compute_distance_layer_time_ms=" << compute_ms << endl;

    REQUIRE(output.data.size() == 1);
    REQUIRE(output.n_channel == 1);
    REQUIRE(output.skip == 1);

    const double encrypted_dist2 = decrypt_first_slot(ctx, output.data[0]);
    const double query_rsqrt = inverse_sqrt_newton_raphson(l2_norm_squared(query), norm2_min, norm2_max, nr_iterations);
    const double approx_dist2 = 2.0 - 2.0 * dot_product(query, gallery_normed) * query_rsqrt;
    const double exact_dist2 = layer.run_plaintext(query, gallery_normed);
    cout << "layer_exact_dist2=" << exact_dist2 << endl;
    cout << "layer_encrypted_dist2=" << encrypted_dist2 << endl;
    REQUIRE(abs(encrypted_dist2 - approx_dist2) < 1.0e-2);
    REQUIRE(abs(approx_dist2 - exact_dist2) < 1.0e-2);
}
