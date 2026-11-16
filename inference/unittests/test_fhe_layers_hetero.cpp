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

#include <tuple>
#include <math.h>
#include <vector>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <iomanip>
#include <utility>
#include <limits>
#include <random>

#include "data_structs/feature.h"
#include "fhe_layers/conv2d_packed_layer.h"
#include "fhe_layers/poly_relu2d.h"
#include "fhe_layers/poly2d.h"
#include "fhe_layers/poly_relu_base.h"
#include "fhe_layers/poly_relu1d.h"
#include "fhe_layers/multiplexed_conv2d_pack_layer.h"
#include "fhe_layers/multiplexed_conv2d_pack_layer_depthwise.h"
#include "fhe_layers/activation_layer.h"
#include "fhe_layers/conv2d_depthwise.h"
#include "fhe_layers/dense_packed_layer.h"
#include "fhe_layers/reshape_layer.h"
#include "fhe_layers/par_block_col_major_transpose.h"
#include "fhe_layers/par_block_col_major_ccmm.h"
#include "fhe_layers/par_block_col_major_cpmm.h"
#include "fhe_layers/par_lower_diag_pcmm.h"
#include "fhe_layers/par_lower_diag_ccmm.h"
#include "fhe_layers/par_lower_diag_transpose.h"
#include "fhe_layers/par_block_col_major_add_pt.h"
#include "fhe_layers/par_lower_diagonal_add_pt.h"
#include "fhe_layers/conv1d_packed_layer.h"
#include "fhe_layers/multiplexed_conv1d_pack_layer.h"
#include "fhe_layers/multiplexed_conv1d_depthwise_pack_layer.h"
#include "fhe_layers/inverse_multiplexed_conv2d_layer.h"
#include "fhe_layers/inverse_multiplexed_conv2d_layer_depthwise.h"
#include "fhe_layers/interleaved_seed.h"
#include "fhe_layers/add_layer.h"
#include "fhe_layers/avgpool2d_layer.h"
#include "fhe_layers/avgpool1d_layer.h"
#include "fhe_layers/concat_layer.h"
#include "fhe_layers/mult_scaler.h"
#include "fhe_layers/upsample_layer.h"
#include "fhe_layers/upsample_nearest_layer.h"
#include "fhe_layers/block_col_major_layernorm.h"
#include "fhe_layers/par_block_col_major_layernorm.h"
#include "fhe_layers/par_upper_diagonal_layernorm.h"
#include "fhe_layers/block_col_major_polyactrn.h"
#include "fhe_layers/par_block_col_major_polyactrn.h"
#include "fhe_layers/par_upper_diagonal_polyact.h"
#include "fhe_layers/par_upper_diagonal_poly_mult_ct.h"
#include "fhe_layers/par_upper_diagonal_poly.h"
#include "fhe_layers/par_upper_diagonal_softmax.h"
#include "data_structs/feature_mat.h"
#include "fhe_layers/batch_dense_packed_layer.h"
#include "ut_util.h"
#include <fhe_ops_lib/utils.h>
#include <cxx_sdk_v2/cxx_fhe_task.h>
#include <cxx_sdk_v2/cxx_argument.h>
#include <lattisense/lib/nlohmann/json.hpp>
#include <any>
#include <algorithm>
#include <unordered_map>

using namespace std;
using namespace lattisense;
using namespace fhe_ops_lib;
namespace fs = std::filesystem;

fs::path base_path = "../hetero";

static void check_kernel_seed_recovery(
    CkksContext& ctx, int level, int cost, uint32_t kernels, Duo shape, Duo skip, Duo subgrid,
    const std::function<CkksPlaintext(int)>& generate_seed,
    const std::function<CkksPlaintext(int)>& generate_legacy,
    const std::function<CkksPlaintext(int, bool)>& generate_mask,
    const std::function<int(int)>& canonical_index = {}) {
    const auto& param = ctx.parameter();
    const double scale = param.q()[level] * 0.625;
    for (uint32_t k = 0; k < kernels; ++k) {
        INFO("kernel=" << k << " level=" << level);
        const uint32_t canonical = canonical_index ? canonical_index(k) : k;
        const uint32_t residue = canonical % prod(subgrid);
        auto seed = generate_seed(canonical / prod(subgrid));
        auto select = generate_mask(k, false);
        const auto& seed_pt_1 = seed;
        std::vector<double> seed_values_1;
        ctx.decode(seed_pt_1, seed_values_1);
        auto encoded_seed_1 = fhe_api::encode(ctx, seed_values_1, level + cost, seed_pt_1.scale());
        auto restored = ctx.encrypt(std::move(encoded_seed_1));
        restored = ctx.rescale(ctx.mult(restored, select));
        const uint32_t residues[2] = {residue / subgrid[1], residue % subgrid[1]};
        const uint32_t units[2] = {skip[0] * shape[1] * skip[1], skip[1]};
        for (int dim = 0; dim < 2; ++dim) {
            for (uint32_t bit = 0; (1u << bit) < subgrid[dim]; ++bit) {
                int step = (1u << bit) * units[dim];
                if (((residues[dim] >> bit) & 1) == 0) step = -step;
                restored = ctx.add(restored, ctx.rotate(restored, step));
            }
        }
        if (cost == 2) {
            auto boundary = generate_mask(k, true);
            restored = ctx.rescale(ctx.mult(restored, boundary));
        }
        REQUIRE(restored.level() == level);
        REQUIRE(restored.scale() == Approx(scale).epsilon(1e-10));
        const auto& seed_pt_2 = generate_legacy(k);
        std::vector<double> seed_values_2;
        ctx.decode(seed_pt_2, seed_values_2);
        auto encoded_seed_2 = fhe_api::encode(ctx, seed_values_2, level, seed_pt_2.scale());
        std::vector<double> expected; ctx.decode(std::move(encoded_seed_2), expected);
        std::vector<double> actual; ctx.decode(ctx.decrypt(restored), actual);
        double error = 0.0;
        for (size_t i = 0; i < expected.size(); ++i) error = std::max(error, std::abs(expected[i] - actual[i]));
        REQUIRE(error < 1e-5);
    }
}

static vector<string> read_arg_names(const fs::path& project_path) {
    ifstream f(project_path / "task_signature.json");
    auto sig = nlohmann::json::parse(f);
    vector<string> names;
    for (const auto& entry : sig["online"]) {
        names.push_back(entry["id"].get<string>());
    }
    return names;
}

struct TaskMetrics {
    std::string test_name;
    std::string task_config;
    int n;
    std::string processor_type;
    double execution_time_ms;
};

std::string extract_task_config(const fs::path& project_path, const fs::path& base_path) {
    auto rel = fs::relative(project_path, base_path);
    return rel.parent_path().string();  // removes the trailing "server" component
}

class MetricsCollector {
public:
    static void add_metrics(const TaskMetrics& metrics) {
        get_instance().metrics_.push_back(metrics);
    }

    static void save_to_csv(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "open file failed: " << filename << std::endl;
            return;
        }

        file << "name, parameter, N, mode, execution time (ms)\n";

        for (const auto& metric : get_instance().metrics_) {
            file << metric.test_name << "," << metric.task_config << "," << metric.n << "," << metric.processor_type
                 << "," << std::fixed << std::setprecision(2) << metric.execution_time_ms << "\n";
        }

        file.close();
        std::cout << "result saved to: " << filename << std::endl;
    }

private:
    static MetricsCollector& get_instance() {
        static MetricsCollector instance;
        return instance;
    }

    std::vector<TaskMetrics> metrics_;
};

class ProcessorCpu;

#ifdef INFERENCE_SDK_ENABLE_GPU
class ProcessorGpu;
#endif

class ProcessorFpga;

struct SharedHeteroResources {
    static SharedHeteroResources& get() {
        static SharedHeteroResources instance;
        return instance;
    }
    const int N = 32768;
    const int n_slot = N;
    CkksParameter param;
    CkksContext context;

private:
    SharedHeteroResources()
        : param(CkksParameter::create_parameter(static_cast<int>(std::log2(N)), RingType::ConjugateInvariant)), context(CkksContext::create_random_context(param)) {
        context.gen_rotation_keys();
    }
    SharedHeteroResources(const SharedHeteroResources&) = delete;
    SharedHeteroResources& operator=(const SharedHeteroResources&) = delete;
};

struct UpperDiagonalHeteroResources {
    static UpperDiagonalHeteroResources& get() {
        static UpperDiagonalHeteroResources instance;
        return instance;
    }
    const int N = 8192;
    const int n_slot = N;
    CkksParameter param;
    CkksContext context;

private:
    UpperDiagonalHeteroResources()
        : param(CkksParameter::create_parameter(static_cast<int>(std::log2(N)), RingType::ConjugateInvariant)), context(CkksContext::create_random_context(param)) {
        context.gen_rotation_keys();
    }
    UpperDiagonalHeteroResources(const UpperDiagonalHeteroResources&) = delete;
    UpperDiagonalHeteroResources& operator=(const UpperDiagonalHeteroResources&) = delete;
};

TEST_CASE("depthwise public masks preserve source positions and rescale", "[depthwise_mask]") {
    auto& resources = SharedHeteroResources::get();
    auto& ctx = resources.context;
    const auto& param = resources.param;
    const uint32_t slots = param.max_slots();
    const int level = 3;
    const Duo shape{8, 8}, stride{2, 2};
    for (Duo skip : std::vector<Duo>{{1, 1}, {1, 2}, {2, 1}, {2, 2}}) {
        const uint32_t channels = 4 * prod(skip);
        const uint32_t n_out = channels + 1;  // A second group with a single valid channel.
        MultiplexedConv2DPackedLayerDepthwise layer(
            param, shape, gen_random_array<4>({n_out, 1, 3, 3}, 0.1),
            gen_random_array<1>({n_out}, 0.1), stride, skip, channels, level);
        for (uint32_t channel : std::vector<uint32_t>{0, 1, channels - 1, channels}) {
            const uint32_t ct_idx = channel / channels, local = channel % channels;
            INFO("skip=" << str(skip) << " ct_idx=" << ct_idx << " local=" << local);
            const auto source = MultiplexedConv2DPackedLayerDepthwise::mask_tensor_for_config(
                shape, skip, stride, channels, slots, ct_idx, local);
            const auto& seed_pt_3 = layer.generate_mask_pt_for_indices(ctx, ct_idx, local);
            std::vector<double> seed_values_3;
            ctx.decode(seed_pt_3, seed_values_3);
            auto encoded_seed_3 = fhe_api::encode(ctx, seed_values_3, level - 1, seed_pt_3.scale());
            std::vector<double> legacy; ctx.decode(std::move(encoded_seed_3), legacy);
            REQUIRE(source.size() == slots);
            // Decode rounds the q(L-1)-scaled ring polynomial; compare its
            // binary support independently below, allowing CKKS encode noise here.
            for (uint32_t j = 0; j < slots; ++j) REQUIRE(legacy[j] == Approx(source[j]).margin(1e-7));

            // Use the channel reorder step from the compute path to check the mask's
            // source positions independently of its generation/encoding helper.
            const int aligned = ct_idx * channels + (local / skip[1]) * skip[1];
            const int dst_block = aligned / prod(skip * stride);
            const int dst_phase = aligned % prod(skip * stride);
            const int src_block = aligned / prod(skip), src_phase = aligned % prod(skip);
            const int step = -((dst_block - src_block) * static_cast<int>(prod(skip) * prod(shape)) +
                (dst_phase / static_cast<int>(stride[1] * skip[1]) - src_phase / static_cast<int>(skip[1])) *
                    static_cast<int>(skip[1] * shape[1]) +
                dst_phase % static_cast<int>(stride[1] * skip[1]) - src_phase % static_cast<int>(skip[1]));
            auto target = layer.select_tensor(channel % (channels * prod(stride)));
            target.resize(slots, 0.0);
            std::vector<double> values(slots);
            for (uint32_t j = 0; j < slots; ++j) values[j] = (static_cast<int>(j % 29) - 14) * 0.03125;
            const auto input = ctx.encrypt(fhe_api::encode(ctx, values, level - 1, param.default_scale()));
            const auto mask = fhe_api::encode_ringt(ctx, source, param.q()[level - 1]);
            auto result = ctx.rescale(ctx.mult(input, mask));
            REQUIRE(result.level() == level - 2);
            REQUIRE(result.scale() == Approx(input.scale()).epsilon(1e-10));
            result = ctx.rotate(result, step);
            std::vector<double> actual; ctx.decode(ctx.decrypt(result), actual);
            double error = 0;
            for (uint32_t j = 0; j < slots; ++j) {
                const uint32_t src = (static_cast<int>(j) + step + static_cast<int>(slots)) % slots;
                REQUIRE(source[src] == target[j]);
                error = std::max(error, std::abs(actual[j] - target[j] * values[src]));
            }
            REQUIRE(error < 1e-5);
        }
    }
}

TEST_CASE("depthwise masks select logical channels in rectangular layouts", "[depthwise_layout]") {
    for (Duo shape : std::vector<Duo>{{4, 8}, {8, 4}}) {
        for (Duo skip : std::vector<Duo>{{1, 2}, {2, 1}, {2, 4}}) {
            const Duo stride{2, 2};
            const uint32_t channels = 2 * prod(skip), slots = channels * prod(shape);
            const uint32_t width = shape[1] * skip[1];
            for (uint32_t channel = 0; channel < 2 * channels + 1; ++channel) {
                const uint32_t local = channel % channels;
                const auto actual = MultiplexedConv2DPackedLayerDepthwise::mask_tensor_for_config(
                    shape, skip, stride, channels, slots, channel / channels, local);
                std::vector<double> expected(slots, 0.0);
                for (uint32_t row = 0; row < shape[0]; row += stride[0]) {
                    for (uint32_t col = 0; col < shape[1]; col += stride[1]) {
                        const uint32_t block = local / prod(skip), phase = local % prod(skip);
                        const uint32_t index = block * prod(shape * skip) +
                            (row * skip[0] + phase / skip[1]) * width + col * skip[1] + phase % skip[1];
                        expected[index] = 1.0;
                    }
                }
                INFO("shape=" << str(shape) << " skip=" << str(skip) << " channel=" << channel);
                REQUIRE(actual == expected);
            }
        }
    }
}

TEST_CASE("depthwise joint seeds restore distinct channel groups and kernels", "[joint_seed]") {
    auto& resources = SharedHeteroResources::get();
    auto& ctx = resources.context;
    const auto& param = resources.param;
    const Duo shape{8, 8}, skip{1, 2}, subgrid{2, 2};
    const int level = 2;
    for (Duo kernel : std::vector<Duo>{{1, 1}, {3, 3}}) {
        const int channels_per_ct = 2, channels = 5, groups = 3;
        auto weight = gen_random_array<4>({channels, 1, kernel[0], kernel[1]}, 0.1);
        auto bias = gen_random_array<1>({channels}, 0.1);
        MultiplexedConv2DPackedLayerDepthwise joint(param, shape, std::move(weight), std::move(bias),
            {1, 1}, skip, channels_per_ct, level, 0.625, subgrid, true);
        joint.prepare_weight_lazy();
        REQUIRE(joint.joint_seed_weights());
        const int requests = groups * prod(kernel);
        REQUIRE(joint.seed_weight_count() == div_ceil(requests, prod(subgrid)));
        check_kernel_seed_recovery(ctx, level, joint.seed_weight_level_cost(), requests, shape, skip, subgrid,
            [&](int q) { return joint.generate_weight_pt_for_seed_indices(ctx, 0, q); },
            [&](int u) { return joint.generate_weight_pt_for_indices(ctx, u / prod(kernel), u % prod(kernel)); },
            [&](int u, bool boundary) { return joint.generate_mask_pt_for_seed_indices(
                ctx, u % prod(subgrid), u % prod(kernel), boundary); });
        REQUIRE_THROWS(joint.generate_weight_pt_for_seed_indices(ctx, 1, 0));
        REQUIRE_THROWS(joint.generate_weight_pt_for_seed_indices(ctx, 0, joint.seed_weight_count()));
    }
}

TEST_CASE("dense tile seeds preserve sparse weights and bias before fold", "[dense_tile]") {
    auto& resources = SharedHeteroResources::get();
    auto& ctx = resources.context;
    const auto& param = resources.param;
    const Duo shape{2, 2}, skip{32, 32}, fill{2, 2};
    const int slots = param.max_slots(), level = 2, outputs = 5, inputs = 129 * 4;
    auto weight = gen_random_array<2>({outputs, inputs}, 0.1);
    auto bias = gen_random_array<1>({outputs}, 0.1);
    DensePackedLayer layer(param, std::move(weight), std::move(bias), 2, level, 0, 0.625);
    layer.prepare_weight_for_2d_multiplexed_lazy(shape, skip, fill);
    // Slot capacity determines the block count; the last output group is partial.
    const int blocks_per_ct = slots / prod(shape * skip);
    const int columns = blocks_per_ct, rows = div_ceil(outputs, blocks_per_ct);
    layer.validate_tile_geometry(shape, skip, fill, slots, {rows, columns}, level);
    REQUIRE_THROWS(layer.validate_tile_geometry(shape, skip, fill, slots, {rows, columns + 1}, level));
    for (bool bias_seed : {false, true}) {
        const int count = bias_seed ? rows : rows * columns;
        const int target_level = level - (bias_seed ? 1 : 0);
        const double scale = bias_seed ? param.default_scale() : param.q()[level] * 0.625;
        for (int u = 0; u < count; ++u) {
            INFO("bias=" << bias_seed << " request=" << u);
            const int sub = u % prod(fill);
            const int delta = interleaved_seed::tile_offset(shape, skip, fill, sub);
            auto seed_pt = layer.generate_tile_seed(ctx, u / prod(fill), bias_seed);
            const auto& seed_pt_4 = seed_pt;
            std::vector<double> seed_values_4;
            ctx.decode(seed_pt_4, seed_values_4);
            auto encoded_seed_4 = fhe_api::encode(ctx, seed_values_4, target_level + 1, seed_pt_4.scale());
            auto seed = ctx.encrypt(std::move(encoded_seed_4));
            const auto seed_bytes = seed.serialize();
            const auto mask = interleaved_seed::tile_mask(shape, skip, fill, slots, sub, bias_seed);
            auto mask_pt = fhe_api::encode_ringt(ctx, mask, param.q()[target_level + 1]);
            auto restored = ctx.rescale(ctx.mult(seed, mask_pt));
            if (delta) restored = ctx.rotate(restored, delta);
            REQUIRE(seed.serialize() == seed_bytes);
            REQUIRE(restored.level() == target_level);
            REQUIRE(restored.scale() == Approx(scale).epsilon(1e-10));
            const auto expected = bias_seed ? layer.generate_bias_values_mult_pack(u) :
                layer.generate_weight_values_mult_pack(u / columns, u % columns);
            std::vector<double> actual; ctx.decode(ctx.decrypt(restored), actual);
            double error = 0.0;
            for (int i = 0; i < slots; ++i) error = std::max(error, std::abs(actual[i] - expected[i]));
            REQUIRE(error < 1e-5);
            if (bias_seed) {
                auto folded = restored.copy();
                auto legacy_folded = ctx.encrypt(fhe_api::encode(ctx, expected, target_level, scale));
                for (int fold = slots / blocks_per_ct; fold > 1; fold /= 2) folded = ctx.add(folded, ctx.rotate(folded, fold / 2));
                for (int fold = slots / blocks_per_ct; fold > 1; fold /= 2)
                    legacy_folded = ctx.add(legacy_folded, ctx.rotate(legacy_folded, fold / 2));
                std::vector<double> folded_values; ctx.decode(ctx.decrypt(folded), folded_values);
                std::vector<double> legacy_values; ctx.decode(ctx.decrypt(legacy_folded), legacy_values);
                for (int block = 0; block < blocks_per_ct; ++block) {
                    const int i = block * slots / blocks_per_ct;
                    CAPTURE(folded_values[i], legacy_values[i], expected[i]);
                    // Summing a full block also sums CKKS noise in zero slots.
                    // Both the original and tile-recovered sparse bias must
                    // remain a single bias, rather than a broadcast times 4096.
                    REQUIRE(std::abs(legacy_values[i] - expected[i]) < 1e-4);
                    REQUIRE(std::abs(folded_values[i] - expected[i]) < 1e-4);
                }
            }
        }
        REQUIRE_THROWS(layer.generate_tile_seed(ctx, div_ceil(count, prod(fill)), bias_seed));
        const auto& seed_pt_5 = layer.generate_tile_seed(ctx, (count - 1) / prod(fill), bias_seed);
        std::vector<double> seed_values_5;
        ctx.decode(seed_pt_5, seed_values_5);
        auto encoded_seed_5 = fhe_api::encode(ctx, seed_values_5, target_level + 1, seed_pt_5.scale());
        std::vector<double> tail; ctx.decode(std::move(encoded_seed_5), tail);
        for (int sub = (count - 1) % prod(fill) + 1; sub < static_cast<int>(prod(fill)); ++sub) {
            const auto mask = interleaved_seed::tile_mask(shape, skip, fill, slots, sub, bias_seed);
            double maximum = 0.0;
            for (int i = 0; i < slots; ++i) if (mask[i]) maximum = std::max(maximum, std::abs(tail[i]));
            REQUIRE(maximum < 1e-6);
        }
    }
}

TEST_CASE("0D dense tiles preserve slot zero and ignore invalid input slots", "[dense_tile]") {
    auto& resources = SharedHeteroResources::get();
    auto& ctx = resources.context;
    const auto& param = resources.param;
    const uint32_t slots = param.max_slots(), inputs = 3, outputs = 5;
    const int level = 1;
    auto weight = gen_random_array<2>({outputs, inputs}, 0.1);
    auto bias = gen_random_array<1>({outputs}, 0.1);
    std::vector<double> expected(outputs);
    for (uint32_t out = 0; out < outputs; ++out) {
        expected[out] = bias.get(out);
        for (uint32_t in = 0; in < inputs; ++in) expected[out] += weight.get(out, in) * (in + 1) * 0.25;
    }
    DensePackedLayer layer(param, std::move(weight), std::move(bias), 1, level, 0);
    layer.prepare_weight_0d_skip_lazy(slots);
    REQUIRE(layer.normal_dense);
    layer.validate_tile_geometry({1, 1}, {1, slots}, {1, slots}, slots, {outputs, inputs}, level);
    const auto& seed_pt_6 = layer.generate_tile_seed(ctx, 0, false);
    std::vector<double> seed_values_6;
    ctx.decode(seed_pt_6, seed_values_6);
    auto encoded_seed_6 = fhe_api::encode(ctx, seed_values_6, level + 1, seed_pt_6.scale());
    auto weights = ctx.encrypt(std::move(encoded_seed_6));
    const auto& seed_pt_7 = layer.generate_tile_seed(ctx, 0, true);
    std::vector<double> seed_values_7;
    ctx.decode(seed_pt_7, seed_values_7);
    auto encoded_seed_7 = fhe_api::encode(ctx, seed_values_7, level, seed_pt_7.scale());
    auto biases = ctx.encrypt(std::move(encoded_seed_7));
    const auto restore = [&](const CkksCiphertext& seed, uint32_t sub, bool bias_seed) {
        const int target = level - bias_seed;
        auto mask = fhe_api::encode_ringt(ctx, interleaved_seed::tile_mask({1, 1}, {1, slots}, {1, slots},
            slots, sub, bias_seed), param.q()[target + 1]);
        auto restored = ctx.rescale(ctx.mult(seed, mask));
        if (sub) restored = ctx.rotate(restored, sub);
        REQUIRE(restored.level() == target);
        const auto legacy = bias_seed ? layer.generate_bias_0d_pt_for_index(ctx, sub) :
            layer.generate_weight_0d_pt_for_indices(ctx, sub / inputs, sub % inputs);
        const auto& seed_pt_8 = legacy;
        std::vector<double> seed_values_8;
        ctx.decode(seed_pt_8, seed_values_8);
        auto encoded_seed_8 = fhe_api::encode(ctx, seed_values_8, target, seed_pt_8.scale());
        std::vector<double> wanted; ctx.decode(std::move(encoded_seed_8), wanted);
        std::vector<double> actual; ctx.decode(ctx.decrypt(restored), actual);
        double error = 0;
        for (uint32_t i = 0; i < slots; ++i) error = std::max(error, std::abs(wanted[i] - actual[i]));
        REQUIRE(error < 1e-5);
        return restored;
    };
    for (uint32_t out = 0; out < outputs; ++out) {
        CkksCiphertext sum;
        for (uint32_t in = 0; in < inputs; ++in) {
            std::vector<double> values(slots, 0.75);  // Deliberately nonzero invalid positions.
            values[0] = (in + 1) * 0.25;
            auto w = restore(weights, out * inputs + in, false);
            auto term = ctx.mult(w, fhe_api::encode_ringt(ctx, values, param.default_scale()));
            if (in) sum = ctx.add(sum, term);
            else sum = std::move(term);
        }
        sum = ctx.rescale(sum);
        sum = ctx.add(sum, restore(biases, out, true));
        std::vector<double> actual; ctx.decode(ctx.decrypt(sum), actual);
        REQUIRE(actual[0] == Approx(expected[out]).margin(1e-5));
        double invalid = 0;
        for (uint32_t i = 1; i < slots; ++i) invalid = std::max(invalid, std::abs(actual[i]));
        REQUIRE(invalid < 1e-5);
    }
}

TEST_CASE("polynomial channel seeds keep degree scale and tail support", "[poly_seed]") {
    auto& resources = SharedHeteroResources::get();
    auto& ctx = resources.context;
    const auto& param = resources.param;
    for (bool big : {false, true}) {
        const Duo shape = big ? Duo{128, 128} : Duo{16, 16};
        const Duo expansion = big ? Duo{2, 1} : Duo{1, 1};
        const Duo physical = shape / expansion;
        const Duo skip = big ? Duo{1, 1} : Duo{2, 2};
        const Duo stride = big ? Duo{16, 16} : Duo{2, 2};
        const int slots = param.max_slots(), groups = big ? 6 : 5;
        const int channels_per_ct = big ? 1 : slots / prod(shape), level = 5;
        const int channels = big ? 3 : 4 * channels_per_ct + 1;
        PolyRelu2D layer(param, shape, 4, gen_random_array<2>({5, static_cast<uint64_t>(channels)}, 0.1),
                         skip, channels_per_ct, level, {1, 1}, expansion);
        layer.prepare_weight_bsgs_lazy();
        const std::vector<int> targets{2, 3, 3, 3, 4};
        std::vector<double> degree_scales;
        for (int degree = 0; degree <= 4; ++degree) {
            const int target = targets[degree];
            layer.validate_seed_geometry(physical, skip, channels_per_ct / prod(skip), slots, groups, degree, target);
            REQUIRE_THROWS(layer.validate_seed_geometry(physical, skip, channels_per_ct / prod(skip), slots, groups, degree, target + 1));
            for (int u = 0; u < groups; ++u) {
                INFO("big=" << big << " degree=" << degree << " channel_group=" << u);
                const auto& seed_pt_9 = layer.generate_weight_pt_for_bsgs(ctx, degree, u);
                std::vector<double> seed_values_9;
                ctx.decode(seed_pt_9, seed_values_9);
                auto encoded_seed_9 = fhe_api::encode(ctx, seed_values_9, target, seed_pt_9.scale());
                const auto reference = std::move(encoded_seed_9);
                const double scale = ctx.encrypt(reference).scale();
                if (!u) degree_scales.push_back(scale);
                const auto& seed_pt_10 = layer.generate_channel_seed(ctx, degree, u / prod(stride), stride);
                std::vector<double> seed_values_10;
                ctx.decode(seed_pt_10, seed_values_10);
                auto encoded_seed_10 = fhe_api::encode(ctx, seed_values_10, target + 1, seed_pt_10.scale());
                auto seed = ctx.encrypt(std::move(encoded_seed_10));
                REQUIRE(seed.scale() == Approx(scale).epsilon(1e-10));
                const auto saved_seed = seed.serialize();
                const int sub = u % prod(stride);
                auto values = interleaved_seed::mask(physical, skip, channels_per_ct / prod(skip), slots, stride, sub, {0, 0}, false);
                auto mask = fhe_api::encode_ringt(ctx, values, param.q()[target + 1]);
                auto restored = ctx.rescale(ctx.mult(seed, mask));
                const int units[2] = {static_cast<int>(skip[0] * physical[1] * skip[1]), static_cast<int>(skip[1])};
                const int residues[2] = {sub / static_cast<int>(stride[1]), sub % static_cast<int>(stride[1])};
                for (int dim = 0; dim < 2; ++dim)
                    for (int bit = 0; (1u << bit) < stride[dim]; ++bit) {
                        int step = units[dim] * (1 << bit);
                        restored = ctx.add(restored, ctx.rotate(restored, (residues[dim] & (1 << bit)) ? step : -step));
                    }
                REQUIRE(seed.serialize() == saved_seed);
                REQUIRE(restored.level() == target);
                REQUIRE(restored.scale() == Approx(scale).epsilon(1e-10));
                std::vector<double> actual, expected;
                ctx.decode(ctx.decrypt(restored), actual);
                ctx.decode(reference, expected);
                double error = 0;
                for (int i = 0; i < slots; ++i) error = std::max(error, std::abs(actual[i] - expected[i]));
                // Broadcasting 256 subgrids amplifies CKKS noise at this test parameter scale.
                REQUIRE(error < (big ? 1e-4 : 1e-5));
            }
            REQUIRE_THROWS(layer.generate_channel_seed(ctx, degree, div_ceil(groups, prod(stride)), stride));
            const auto& seed_pt_11 = layer.generate_channel_seed(ctx, degree, (groups - 1) / prod(stride), stride);
            std::vector<double> seed_values_11;
            ctx.decode(seed_pt_11, seed_values_11);
            auto encoded_seed_11 = fhe_api::encode(ctx, seed_values_11, target + 1, seed_pt_11.scale());
            std::vector<double> tail; ctx.decode(std::move(encoded_seed_11), tail);
            for (int sub = (groups - 1) % prod(stride) + 1; sub < static_cast<int>(prod(stride)); ++sub) {
                const auto mask = interleaved_seed::mask(physical, skip, channels_per_ct / prod(skip), slots, stride, sub, {0, 0}, false);
                double maximum = 0;
                for (int i = 0; i < slots; ++i) if (mask[i]) maximum = std::max(maximum, std::abs(tail[i]));
                REQUIRE(maximum < 1e-6);
            }
        }
        REQUIRE(std::abs(degree_scales[0] / degree_scales[3] - 1.0) > 1e-8);
    }
}

TEST_CASE("inverse canonical seeds preserve each execution boundary", "[inverse_canonical]") {
    auto& resources = SharedHeteroResources::get();
    auto& ctx = resources.context;
    const auto& param = resources.param;
    const Duo block{8, 16}, input = block * 4, kernel{3, 3}, subgrid{2, 2};
    Array<int, 1> padding({2});
    padding.set(0, -1);
    padding.set(1, -1);
    const auto canonical = [&](int e) {
        const auto k = interleaved_seed::inverse_kernel_position(kernel, {2, 2}, {2, 2}, {1, 1}, e).kernel;
        return k[0] * kernel[1] + k[1];
    };
    InverseMultiplexedConv2DLayer layer(param, input, gen_random_array<4>({2, 3, 3, 3}, 0.1),
        gen_random_array<1>({2}, 0.1), padding, {2, 2}, block, 2, 0.625, subgrid, true);
    layer.prepare_weight_lazy();
    REQUIRE(layer.seed_weight_kernel_count() == 3);
    check_kernel_seed_recovery(ctx, 2, 2, 36, block, {1, 1}, subgrid,
        [&](int q) { return layer.generate_weight_pt_for_seed_indices(ctx, 1, 2, q); },
        [&](int e) { return layer.generate_weight_pt_for_indices(ctx, 1, 2, e); },
        [&](int e, bool boundary) { return layer.generate_mask_pt_for_seed_indices(ctx, canonical(e) % 4, e, boundary); },
        canonical);
    REQUIRE_THROWS(layer.generate_weight_pt_for_seed_indices(ctx, 1, 2, 3));
    InverseMultiplexedConv2DLayerDepthwise depthwise(param, input, gen_random_array<4>({3, 1, 3, 3}, 0.1),
        gen_random_array<1>({3}, 0.1), padding, {2, 2}, block, 2, 0.625, subgrid, true);
    depthwise.prepare_weight_lazy();
    REQUIRE(depthwise.seed_weight_kernel_count() == 3);
    check_kernel_seed_recovery(ctx, 2, 2, 36, block, {1, 1}, subgrid,
        [&](int q) { return depthwise.generate_weight_pt_for_seed_indices(ctx, 2, q); },
        [&](int e) { return depthwise.generate_weight_pt_for_indices(ctx, 2, e); },
        [&](int e, bool boundary) { return depthwise.generate_mask_pt_for_seed_indices(ctx, canonical(e) % 4, e, boundary); },
        canonical);
}

TEST_CASE("multiplexed joint seeds cross output input and kernel axes", "[multiplexed_joint]") {
    auto& resources = SharedHeteroResources::get();
    auto& ctx = resources.context;
    const auto& param = resources.param;
    const Duo shape{8, 8}, skip{1, 2}, subgrid{4, 2};
    for (const Duo kernel : std::vector<Duo>{{1, 1}, {3, 3}}) {
        MultiplexedConv2DPackedLayer layer(param, shape,
            gen_random_array<4>({5, 3, kernel[0], kernel[1]}, 0.1), gen_random_array<1>({5}, 0.1),
            {1, 1}, skip, 8, 2, 0.625, {1, 1}, subgrid, true);
        layer.prepare_weight_for_post_skip_rotation_lazy();
        REQUIRE(layer.joint_seed_weights());
        const int kernels = prod(kernel), requests = 2 * 2 * kernels;
        const int cost = kernel == Duo{1, 1} ? 1 : 2;
        REQUIRE(layer.seed_weight_count() == div_ceil(requests, 8));
        check_kernel_seed_recovery(ctx, 2, cost, requests, shape, skip, subgrid,
            [&](int q) { return layer.generate_weight_pt_for_seed_indices(ctx, 0, q, 0); },
            [&](int u) { return layer.generate_weight_pt_for_indices(ctx, u / (2 * kernels), (u / kernels) % 2, u % kernels); },
            [&](int u, bool boundary) {
                const auto mask = MultiplexedConv2DPackedLayer::seed_mask_tensor_for_config(
                    shape, skip, {1, 1}, 8, {1, 1}, subgrid[0], subgrid[1], u % 8, u % kernels, kernel, boundary);
                return fhe_api::encode_ringt(ctx, mask, param.q()[2 + cost - static_cast<int>(boundary)]);
            });
        REQUIRE_THROWS(layer.generate_weight_pt_for_seed_indices(ctx, 0, layer.seed_weight_count(), 0));
    }
}

TEST_CASE("kernel seed weights restore encrypted values and preserve levels", "[kernel_seed]") {
    auto& resources = SharedHeteroResources::get();
    auto& ctx = resources.context;
    const auto& param = resources.param;
    const Duo subgrid{4, 2};
    for (const Duo kernel : std::vector<Duo>{{1, 1}, {3, 3}, {1, 5}}) {
        for (int level : {0, 1, 3}) {
            INFO("kernel=" << str(kernel));
            const Duo shape{8, 8}, skip{1, 2};
            auto dw_weight = gen_random_array<4>({5, 1, kernel[0], kernel[1]}, 0.1);
            MultiplexedConv2DPackedLayerDepthwise dw(
                param, shape, move(dw_weight), gen_random_array<1>({5}, 0.1),
                {1, 1}, skip, (param.max_slots()) / prod(shape), level, 0.625, subgrid);
            dw.prepare_weight_lazy();
            check_kernel_seed_recovery(ctx, level, dw.seed_weight_level_cost(), prod(kernel), shape, skip, subgrid,
                [&](int k) { return dw.generate_weight_pt_for_seed_indices(ctx, 0, k); },
                [&](int k) { return dw.generate_weight_pt_for_indices(ctx, 0, k); },
                [&](int k, bool boundary) { return dw.generate_mask_pt_for_seed_indices(ctx, k % prod(subgrid), k, boundary); });

            const Duo block{8, 16}, input = block * 4;
            Array<int, 1> padding({2});
            padding.set(0, -1);
            padding.set(1, -1);
            InverseMultiplexedConv2DLayer inverse(
                param, input, gen_random_array<4>({2, 3, kernel[0], kernel[1]}, 0.1),
                gen_random_array<1>({2}, 0.1), padding, {2, 2}, block, level, 0.625, subgrid);
            inverse.prepare_weight_lazy();
            check_kernel_seed_recovery(ctx, level, inverse.seed_weight_level_cost(), prod(kernel) * 4,
                block, {1, 1}, subgrid,
                [&](int k) { return inverse.generate_weight_pt_for_seed_indices(ctx, 1, 2, k); },
                [&](int k) { return inverse.generate_weight_pt_for_indices(ctx, 1, 2, k); },
                [&](int k, bool boundary) { return inverse.generate_mask_pt_for_seed_indices(ctx, k % prod(subgrid), k, boundary); });

            InverseMultiplexedConv2DLayerDepthwise inverse_dw(
                param, input, gen_random_array<4>({3, 1, kernel[0], kernel[1]}, 0.1),
                gen_random_array<1>({3}, 0.1), padding, {2, 2}, block, level, 0.625, subgrid);
            inverse_dw.prepare_weight_lazy();
            check_kernel_seed_recovery(ctx, level, inverse_dw.seed_weight_level_cost(), prod(kernel) * 4,
                block, {1, 1}, subgrid,
                [&](int k) { return inverse_dw.generate_weight_pt_for_seed_indices(ctx, 2, k); },
                [&](int k) { return inverse_dw.generate_weight_pt_for_indices(ctx, 2, k); },
                [&](int k, bool boundary) { return inverse_dw.generate_mask_pt_for_seed_indices(ctx, k % prod(subgrid), k, boundary); });
        }
    }
}

TEST_CASE("kernel seed C++ eager and lazy convolution match legacy", "[kernel_seed]") {
    auto& resources = SharedHeteroResources::get();
    auto& ctx = resources.context;
    const auto& param = resources.param;
    const int level = 3;
    auto check_run = [&](auto& packed, auto& legacy, Feature2DEncrypted& input) {
        legacy.prepare_weight();
        const auto expected = legacy.run(ctx, input);
        for (bool lazy : {false, true}) {
            INFO("lazy=" << lazy);
            if (lazy) {
                packed.weight_pt.clear();
                packed.bias_pt.clear();
                packed.prepare_weight_lazy();
            } else {
                packed.prepare_weight();
            }
            const auto actual = packed.run(ctx, input);
            REQUIRE(actual.level == expected.level);
            REQUIRE(actual.data.size() == expected.data.size());
            for (size_t k = 0; k < actual.data.size(); ++k) {
                REQUIRE(actual.data[k].level() == expected.data[k].level());
                std::vector<double> a; ctx.decode(ctx.decrypt(actual.data[k]), a);
                std::vector<double> e; ctx.decode(ctx.decrypt(expected.data[k]), e);
                double error = 0.0;
                for (size_t i = 0; i < a.size(); ++i) error = std::max(error, std::abs(a[i] - e[i]));
                REQUIRE(error < 1e-5);
            }
        }
    };
    for (Duo stride : std::vector<Duo>{{1, 1}, {2, 2}}) {
        const Duo shape{16, 16}, skip{1, 1};
        auto weight = gen_random_array<4>({5, 1, 3, 3}, 0.1);
        auto bias = gen_random_array<1>({5}, 0.1);
        const auto channels_per_ct = (param.max_slots()) / prod(shape);
        MultiplexedConv2DPackedLayerDepthwise packed(param, shape, weight.copy(), bias.copy(), stride, skip,
                                                    channels_per_ct, level, 1.0, {4, 4});
        MultiplexedConv2DPackedLayerDepthwise legacy(param, shape, weight.copy(), bias.copy(), stride, skip,
                                                    channels_per_ct, level);
        Feature2DEncrypted input(&ctx, level, skip);
        input.pack_multiplexed(gen_random_array<3>({5, shape[0], shape[1]}, 1.0), param.default_scale());
        check_run(packed, legacy, input);
    }
    for (Duo stride : std::vector<Duo>{{2, 2}, {4, 4}}) {
        const Duo block{64, 128}, shape = block * 2;
        Array<int, 1> padding({2});
        padding.set(0, -1);
        padding.set(1, -1);
        auto weight = gen_random_array<4>({2, 1, 3, 3}, 0.1);
        auto bias = gen_random_array<1>({2}, 0.1);
        InverseMultiplexedConv2DLayer packed(param, shape, weight.copy(), bias.copy(), padding, stride, block,
                                            level, 1.0, {4, 4});
        InverseMultiplexedConv2DLayer legacy(param, shape, weight.copy(), bias.copy(), padding, stride, block, level);
        Feature2DEncrypted input(&ctx, level, {1, 1});
        input.pack_interleaved(gen_random_array<3>({1, shape[0], shape[1]}, 1.0), block, {2, 2}, param.default_scale());
        check_run(packed, legacy, input);
        InverseMultiplexedConv2DLayerDepthwise packed_dw(param, shape, weight.copy(), bias.copy(), padding, stride, block,
                                                        level, 1.0, {4, 4});
        InverseMultiplexedConv2DLayerDepthwise legacy_dw(param, shape, weight.copy(), bias.copy(), padding, stride, block, level);
        Feature2DEncrypted dw_input(&ctx, level, {1, 1});
        dw_input.pack_interleaved(gen_random_array<3>({2, shape[0], shape[1]}, 1.0), block, {2, 2}, param.default_scale());
        check_run(packed_dw, legacy_dw, dw_input);
    }
}

template <typename T> class HeteroFixture {
public:
    HeteroFixture()
        : N{SharedHeteroResources::get().N}, n_slot{SharedHeteroResources::get().n_slot},
          param{SharedHeteroResources::get().param}, context{SharedHeteroResources::get().context}, level(3),
          min_level{0}, max_level{param.max_level()}, default_scale{param.default_scale()} {}

    ~HeteroFixture() {
        MetricsCollector::save_to_csv("hetero_performance_results.csv");
    }

    uint64_t run(const fs::path& project_path, const vector<CxxVectorArgument>& cxx_args) {
        auto start_time = std::chrono::high_resolution_clock::now();
        uint64_t result;
        std::string processor_type;
        std::unordered_map<std::string, ExecutorFunc> executors;
        executors["encode_pt"] = [](ExecutionContext& exec_ctx, std::unordered_map<NodeId, std::any>& local_data,
                                    const ComputeNode& self) {
            const auto& attrs = self.custom_prop->attributes;
            auto* ctx = exec_ctx.get_arithmetic_context<CkksContext>();
            const auto op_class = attrs.at("op_class").get<string>();
            if (attrs.at("type") != "normalization_pt" ||
                (op_class != "Avgpool1DLayer" && op_class != "Avgpool2DLayer"))
                throw runtime_error("unsupported encode_pt in pooling fixture");
            const auto divisor = attrs.at("divisor").get<uint32_t>();
            const auto level = attrs.at("level").get<uint32_t>();
            auto pt = op_class == "Avgpool1DLayer" ? Avgpool1DLayer::normalization_pt(*ctx, divisor, level)
                                                   : Avgpool2DLayer::normalization_pt(*ctx, divisor, level);
            local_data[self.output_nodes[0]->id] = make_shared<CkksPlaintext>(move(pt));
        };

        if constexpr (is_same_v<T, ProcessorCpu>) {
            processor_type = "CPU";
            FheTaskCpu fhe_task(project_path.string());
            fhe_task.bind_custom_executors(executors);
            result = fhe_task.run(&context, cxx_args);
#ifdef INFERENCE_SDK_ENABLE_GPU
        } else if constexpr (is_same_v<T, ProcessorGpu>) {
            processor_type = "GPU";
            FheTaskGpu fhe_task(project_path.string());
            fhe_task.bind_custom_executors(executors);
            result = fhe_task.run(&context, cxx_args);
#endif
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "[" << processor_type << "] execution time: " << duration.count() << " ms" << std::endl;

        TaskMetrics metrics;
        metrics.processor_type = processor_type;
        metrics.task_config = extract_task_config(project_path, base_path);
        metrics.n = N;
        metrics.execution_time_ms = duration.count();
        metrics.test_name = Catch::getCurrentContext().getResultCapture()->getCurrentTestName();

        MetricsCollector::add_metrics(metrics);

        return result;
    }

protected:
    int N;
    int n_slot;
    CkksParameter& param;
    CkksContext& context;
    int level;
    int min_level;
    int max_level;
    double default_scale;
};

#ifdef INFERENCE_SDK_ENABLE_GPU
using HeteroProcessors = tuple<ProcessorCpu, ProcessorGpu>;
#else
using HeteroProcessors = tuple<ProcessorCpu>;
#endif
using UpperDiagonalGraphProcessors = tuple<ProcessorCpu>;

#define FOR_EACH_SECTION(var_decl, range, section_name)                                                                \
    for (var_decl : range)                                                                                             \
    SECTION(section_name)

#define FOR_EACH_SECTION_IF(var_decl, range, condition, section_name)                                                  \
    for (var_decl : range)                                                                                             \
        if (!(condition)) {                                                                                            \
        } else                                                                                                         \
            SECTION(section_name)

static uint32_t next_pow2_u32(uint32_t x) {
    uint32_t p = 1;
    while (p < x)
        p <<= 1;
    return p;
}

static uint32_t
par_diagonal_total_cts(const CkksParameter& param, Duo shape, Duo head_shape, uint32_t n_heads, bool is_transposed) {
    uint32_t n_prepad = is_transposed ? head_shape[1] : head_shape[0];
    uint32_t m_prepad = is_transposed ? head_shape[0] : head_shape[1];
    uint32_t H = next_pow2_u32(n_heads);
    uint32_t m = next_pow2_u32(m_prepad);
    uint32_t n = next_pow2_u32(n_prepad);
    uint32_t segment_len = H * n;
    uint32_t c = (param.max_slots()) / segment_len;
    uint32_t cts_per_mb = m / c;
    uint32_t packed_extent = n_heads * m_prepad;
    uint32_t packed_total = is_transposed ? shape[0] : shape[1];
    return div_ceil(packed_total, packed_extent) * cts_per_mb;
}

static Array<double, 2> unpack_par_diagonal_output(CkksContext& context,
                                                   int level,
                                                   const vector<CkksCiphertext>& out_cts,
                                                   Duo shape,
                                                   Duo head_shape,
                                                   uint32_t n_heads,
                                                   bool is_lower,
                                                   bool is_transposed) {
    FeatureMatEncrypted out_enc(&context, level);
    out_enc.shape = shape;
    out_enc.head_shape = head_shape;
    out_enc.matmul_block_size = next_pow2_u32(is_transposed ? head_shape[0] : head_shape[1]);
    for (const auto& ct : out_cts)
        out_enc.data.push_back(ct.copy());
    return out_enc.par_diagonal_unpack(n_heads, head_shape, is_lower, is_transposed);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "par_lower_diag_transpose_roundtrip",
                               "[fhe_layers][par_lower_diag_transpose]",
                               HeteroProcessors) {
    int init_level = 2;
    uint32_t n_heads = 3;
    uint32_t head_dim = 64;
    vector<uint32_t> n_prepads = {65, 128};

    FOR_EACH_SECTION(uint32_t n_prepad, n_prepads, "n_prepad=" + to_string(n_prepad)) {
        fs::path project_path = base_path / "CKKS_par_lower_diag_transpose" / ("n_prepad_" + to_string(n_prepad)) /
                                ("level_" + to_string(init_level)) / "server";

        if (!fs::exists(project_path / "mega_ag.json"))
            return;

        Array<double, 2> X_T = gen_random_array<2>({n_heads * head_dim, n_prepad}, 1.0);

        FeatureMatEncrypted input_enc(&this->context, init_level);
        input_enc.par_diagonal_pack(X_T, n_heads, {head_dim, n_prepad}, true, true,
                                    this->param.default_scale());

        ParLowerDiagTranspose layer(this->param, Duo{head_dim, n_prepad}, n_heads, head_dim, init_level);
        layer.prepare_weight();
        Array<double, 2> expected = layer.run_plaintext(X_T);

        vector<CkksCiphertext> in_cts, out_cts;
        for (auto& ct : input_enc.data)
            in_cts.push_back(ct.copy());
        uint32_t n_out =
            par_diagonal_total_cts(this->param, {n_prepad, n_heads * head_dim}, {n_prepad, head_dim}, n_heads, false);
        for (uint32_t i = 0; i < n_out; i++)
            out_cts.emplace_back(this->context.parameter(), init_level - 1);
            out_cts.back().set_scale(this->param.default_scale());

        auto arg_names = read_arg_names(project_path);
        vector<CxxVectorArgument> cxx_args;
        for (const auto& name : arg_names) {
            if (name == "input")
                cxx_args.push_back({name, &in_cts});
            else if (name == "transpose_mask_pt")
                cxx_args.push_back({name, &layer.transpose_mask_pt_});
            else if (name == "output")
                cxx_args.push_back({name, &out_cts});
        }
        this->run(project_path, cxx_args);

        Array<double, 2> actual =
            unpack_par_diagonal_output(this->context, init_level - 1, out_cts, {n_prepad, n_heads * head_dim},
                                       {n_prepad, head_dim}, n_heads, true, false);
        print_double_message(actual.to_array_1d().data(), "actual", 10);
        print_double_message(expected.to_array_1d().data(), "expected", 10);
        auto comparison = compare(expected, actual);
        REQUIRE(comparison.max_error < 5.0e-2 * comparison.max_abs);
        REQUIRE(comparison.rmse < 1.0e-2 * comparison.rms);
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "par_lower_diag_pcmm_roundtrip",
                               "[fhe_layers][par_lower_diag_pcmm]",
                               HeteroProcessors) {
    int init_level = 3;
    uint32_t n_prepad = 65;
    uint32_t n_heads = 3;
    uint32_t head_dim = 64;
    uint32_t d_prepad = n_heads * head_dim;

    struct Case {
        uint32_t w_rows;
        uint32_t w_cols;
        const char* mode;
    };

    vector<Case> cases = {
        {d_prepad, d_prepad, "SQUARE"},
        {d_prepad, 4 * d_prepad, "EXPAND"},
        {4 * d_prepad, d_prepad, "REDUCE"},
    };

    for (const auto& cfg : cases) {
        INFO(cfg.mode);
        std::string mode_dir = cfg.mode == std::string("SQUARE") ? "square" :
                               cfg.mode == std::string("EXPAND") ? "expand" :
                                                                   "reduce";
        fs::path project_path =
            base_path / "CKKS_par_lower_diag_pcmm" / mode_dir / ("level_" + to_string(init_level)) / "server";
        if (!fs::exists(project_path / "mega_ag.json"))
            return;

        Array<double, 2> X_T = gen_random_array<2>({cfg.w_rows, n_prepad}, 1.0);
        Array<double, 2> W_T = gen_random_array<2>({cfg.w_cols, cfg.w_rows}, 0.1);
        Array<double, 1> bias = gen_random_array<1>({cfg.w_cols}, 0.1);

        FeatureMatEncrypted X_enc(&this->context, init_level);
        X_enc.par_diagonal_pack(X_T, n_heads, {head_dim, n_prepad}, true, true, this->param.default_scale());

        ParLowerDiagPCMM layer(this->param, Duo{cfg.w_rows, n_prepad}, n_heads, head_dim, W_T, init_level, bias.copy());
        layer.prepare_weight();
        Array<double, 2> expected = layer.run_plaintext(X_T);

        vector<CkksCiphertext> in_cts, out_cts;
        for (auto& ct : X_enc.data)
            in_cts.push_back(ct.copy());
        uint32_t n_out =
            par_diagonal_total_cts(this->param, {cfg.w_cols, n_prepad}, {head_dim, n_prepad}, n_heads, true);
        for (uint32_t i = 0; i < n_out; i++)
            out_cts.emplace_back(this->context.parameter(), init_level - 2);
            out_cts.back().set_scale(this->param.default_scale());

        auto arg_names = read_arg_names(project_path);
        vector<CxxVectorArgument> cxx_args;
        for (const auto& name : arg_names) {
            if (name == "input")
                cxx_args.push_back({name, &in_cts});
            else if (name == "pt_A")
                cxx_args.push_back({name, &layer.pt_A_});
            else if (name == "mask_wrap_pt")
                cxx_args.push_back({name, &layer.mask_wrap_pt_});
            else if (name == "bias_pt")
                cxx_args.push_back({name, &layer.bias_pt_});
            else if (name == "output")
                cxx_args.push_back({name, &out_cts});
        }
        this->run(project_path, cxx_args);

        Array<double, 2> actual = unpack_par_diagonal_output(
            this->context, init_level - 2, out_cts, {cfg.w_cols, n_prepad}, {head_dim, n_prepad}, n_heads, true, true);
        print_double_message(actual.to_array_1d().data(), "actual", 10);
        print_double_message(expected.to_array_1d().data(), "expected", 10);
        auto comparison = compare(expected, actual);
        REQUIRE(comparison.max_error < 5.0e-2 * comparison.max_abs);
        REQUIRE(comparison.rmse < 1.0e-2 * comparison.rms);
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "par_lower_diag_ccmm_kqt_roundtrip",
                               "[fhe_layers][par_lower_diag_ccmm][kqt]",
                               HeteroProcessors) {
    int ccmm_level = 3;
    uint32_t n_heads = 3;
    uint32_t head_dim = 64;
    uint32_t n_prepad = 97;
    fs::path project_path =
        base_path / "CKKS_par_lower_diag_ccmm" / "kqt" / ("level_" + to_string(ccmm_level)) / "server";
    if (!fs::exists(project_path / "mega_ag.json"))
        return;

    Array<double, 2> A = gen_random_array<2>({n_prepad, n_heads * head_dim}, 0.5);
    Array<double, 2> B = gen_random_array<2>({n_heads * head_dim, n_prepad}, 0.5);

    FeatureMatEncrypted A_enc(&this->context, ccmm_level);
    FeatureMatEncrypted B_enc(&this->context, ccmm_level);
    A_enc.par_diagonal_pack(A, n_heads, {n_prepad, head_dim}, true, false, this->param.default_scale());
    B_enc.par_diagonal_pack(B, n_heads, {head_dim, n_prepad}, true, true, this->param.default_scale());

    ParLowerDiagCCMM layer(this->param, Duo{n_prepad, head_dim}, Duo{head_dim, n_prepad}, n_heads, head_dim,
                           ccmm_level);
    layer.prepare_weight();
    Array<double, 2> expected = layer.run_plaintext(A, B);

    vector<CkksCiphertext> A_cts, B_cts, out_cts;
    for (auto& ct : A_enc.data)
        A_cts.push_back(ct.copy());
    for (auto& ct : B_enc.data)
        B_cts.push_back(ct.copy());
    uint32_t n_out =
        par_diagonal_total_cts(this->param, {n_heads * n_prepad, n_prepad}, {n_prepad, n_prepad}, n_heads, true);
    for (uint32_t i = 0; i < n_out; i++)
        out_cts.emplace_back(this->context.parameter(), ccmm_level - 3);
        out_cts.back().set_scale(this->param.default_scale());

    auto arg_names = read_arg_names(project_path);
    vector<CxxVectorArgument> cxx_args;
    for (const auto& name : arg_names) {
        if (name == "A_input")
            cxx_args.push_back({name, &A_cts});
        else if (name == "B_input")
            cxx_args.push_back({name, &B_cts});
        else if (name == "replication_mask_pt")
            cxx_args.push_back({name, &layer.replication_mask_pt_});
        else if (name == "kqt_route_pt")
            cxx_args.push_back({name, &layer.kqt_route_pt_});
        else if (name == "output")
            cxx_args.push_back({name, &out_cts});
    }
    this->run(project_path, cxx_args);

    Array<double, 2> actual =
        unpack_par_diagonal_output(this->context, ccmm_level - 3, out_cts, {n_heads * n_prepad, n_prepad},
                                   {n_prepad, n_prepad}, n_heads, true, true);
    print_double_message(actual.to_array_1d().data(), "actual", 10);
    print_double_message(expected.to_array_1d().data(), "expected", 10);
    auto comparison = compare(expected, actual);
    REQUIRE(comparison.max_error < 5.0e-2 * comparison.max_abs);
    REQUIRE(comparison.rmse < 1.0e-2 * comparison.rms);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "par_lower_diag_ccmm_ordinary_roundtrip",
                               "[fhe_layers][par_lower_diag_ccmm][ordinary]",
                               HeteroProcessors) {
    int init_level = 3;
    uint32_t n_heads = 3;
    uint32_t head_dim = 64;
    uint32_t n_prepad = 97;
    fs::path project_path =
        base_path / "CKKS_par_lower_diag_ccmm" / "ordinary" / ("level_" + to_string(init_level)) / "server";
    if (!fs::exists(project_path / "mega_ag.json"))
        return;

    Array<double, 2> A = gen_random_array<2>({n_heads * head_dim, n_prepad}, 0.5);
    Array<double, 2> B = gen_random_array<2>({n_heads * n_prepad, n_prepad}, 0.5);

    FeatureMatEncrypted A_enc(&this->context, init_level);
    FeatureMatEncrypted B_enc(&this->context, init_level);
    A_enc.par_diagonal_pack(A, n_heads, {head_dim, n_prepad}, true, true, this->param.default_scale());
    B_enc.par_diagonal_pack(B, n_heads, {n_prepad, n_prepad}, true, true, this->param.default_scale());

    ParLowerDiagCCMM layer(this->param, Duo{head_dim, n_prepad}, Duo{n_prepad, n_prepad}, n_heads, head_dim,
                           init_level);
    layer.prepare_weight();
    Array<double, 2> expected = layer.run_plaintext(A, B);

    vector<CkksCiphertext> A_cts, B_cts, out_cts;
    for (auto& ct : A_enc.data)
        A_cts.push_back(ct.copy());
    for (auto& ct : B_enc.data)
        B_cts.push_back(ct.copy());
    uint32_t n_out =
        par_diagonal_total_cts(this->param, {n_heads * head_dim, n_prepad}, {head_dim, n_prepad}, n_heads, true);
    for (uint32_t i = 0; i < n_out; i++)
        out_cts.emplace_back(this->context.parameter(), init_level - 3);
        out_cts.back().set_scale(this->param.default_scale());

    auto arg_names = read_arg_names(project_path);
    vector<CxxVectorArgument> cxx_args;
    for (const auto& name : arg_names) {
        if (name == "A_input")
            cxx_args.push_back({name, &A_cts});
        else if (name == "B_input")
            cxx_args.push_back({name, &B_cts});
        else if (name == "replication_mask_pt")
            cxx_args.push_back({name, &layer.replication_mask_pt_});
        else if (name == "ordinary_route_pt")
            cxx_args.push_back({name, &layer.ordinary_route_pt_});
        else if (name == "output")
            cxx_args.push_back({name, &out_cts});
    }
    this->run(project_path, cxx_args);

    Array<double, 2> actual =
        unpack_par_diagonal_output(this->context, init_level - 3, out_cts, {n_heads * head_dim, n_prepad},
                                   {head_dim, n_prepad}, n_heads, true, true);
    print_double_message(actual.to_array_1d().data(), "actual", 10);
    print_double_message(expected.to_array_1d().data(), "expected", 10);
    auto comparison = compare(expected, actual);
    REQUIRE(comparison.max_error < 5.0e-2 * comparison.max_abs);
    REQUIRE(comparison.rmse < 1.0e-2 * comparison.rms);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "par_lower_diagonal_add_pt_cpp_roundtrip",
                               "[fhe_layers][par_lower_diagonal_add_pt]",
                               HeteroProcessors) {
    const uint32_t total_rows = 137;
    const uint32_t n_prepad = 197;
    const uint32_t n_heads = 3;
    const uint32_t m_prepad = 20;
    const uint32_t init_level = 2;
    Duo shape = {total_rows, n_prepad};
    Duo head_shape = {m_prepad, n_prepad};
    fs::path project_path =
        base_path / "CKKS_par_lower_diagonal_add_pt" / ("level_" + to_string(init_level)) / "server";
    if (!fs::exists(project_path / "mega_ag.json"))
        return;

    auto A = gen_random_array<2>({total_rows, n_prepad}, 0.5);
    auto B = gen_random_array<2>({total_rows, n_prepad}, 0.5);

    FeatureMatEncrypted A_enc(&this->context, init_level);
    A_enc.par_diagonal_pack(A, n_heads, head_shape, true, true, this->param.default_scale());

    ParLowerDiagonalAddPt layer(this->param, shape, head_shape, n_heads, init_level, B.copy());
    layer.precompute_pts();
    Array<double, 2> expected = layer.run_plaintext(A);

    vector<CkksCiphertext> in_cts, out_cts;
    for (auto& ct : A_enc.data)
        in_cts.push_back(ct.copy());
    uint32_t n_out = par_diagonal_total_cts(this->param, shape, head_shape, n_heads, true);
    for (uint32_t i = 0; i < n_out; i++)
        out_cts.emplace_back(this->context.parameter(), init_level);
        out_cts.back().set_scale(this->param.default_scale());

    auto arg_names = read_arg_names(project_path);
    vector<CxxVectorArgument> cxx_args;
    for (const auto& name : arg_names) {
        if (name == "input")
            cxx_args.push_back({name, &in_cts});
        else if (name == "add_pt")
            cxx_args.push_back({name, &layer.pt_});
        else if (name == "output")
            cxx_args.push_back({name, &out_cts});
    }
    this->run(project_path, cxx_args);

    Array<double, 2> actual =
        unpack_par_diagonal_output(this->context, init_level, out_cts, shape, head_shape, n_heads, true, true);
    print_double_message(actual.to_array_1d().data(), "actual", 10);
    print_double_message(expected.to_array_1d().data(), "expected", 10);
    auto comparison = compare(expected, actual);
    CAPTURE(comparison.max_error, comparison.max_abs, comparison.rmse, comparison.rms);
    REQUIRE(comparison.max_error < 5.0e-2 * comparison.max_abs);
    REQUIRE(comparison.rmse < 1.0e-2 * comparison.rms);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "par_upper_diagonal_ln_stats_cpp_roundtrip",
                               "[fhe_layers][par_upper_diagonal_layernorm][stats]",
                               HeteroProcessors) {
    const uint32_t n_prepad = 65;
    const uint32_t total_cols = 73;
    const uint32_t n_heads = 3;
    const uint32_t m_prepad = 20;
    const int init_level = 4;
    const double eps = 1e-5;
    const double var_std_bound = 4.0;
    const double inv_var = 1.0 / (var_std_bound * var_std_bound);
    Duo shape = {n_prepad, total_cols};
    Duo head_shape = {n_prepad, m_prepad};
    fs::path project_path =
        base_path / "CKKS_par_upper_diagonal_layernorm" / "stats" / ("level_" + to_string(init_level)) / "server";
    if (!fs::exists(project_path / "mega_ag.json"))
        return;
    auto X_mat = gen_random_array<2>({n_prepad, total_cols}, 0.5);
    FeatureMatEncrypted X_enc(&this->context, init_level);
    X_enc.par_diagonal_pack(X_mat, n_heads, head_shape, false, false, this->param.default_scale());

    ParUpperDiagonalLNStats layer(this->param, shape, head_shape, n_heads, init_level, eps, inv_var);
    layer.prepare_weight();
    Array<double, 2> expected = layer.run_plaintext(X_mat);

    vector<CkksCiphertext> in_cts, out_cts;
    for (auto& ct : X_enc.data)
        in_cts.push_back(ct.copy());
    uint32_t n_out = par_diagonal_total_cts(this->param, shape, head_shape, n_heads, false);
    for (uint32_t i = 0; i < n_out; i++)
        out_cts.emplace_back(this->context.parameter(), init_level - 3);
        out_cts.back().set_scale(this->param.default_scale());

    auto arg_names = read_arg_names(project_path);
    vector<CxxVectorArgument> cxx_args;
    for (const auto& name : arg_names) {
        if (name == "input")
            cxx_args.push_back({name, &in_cts});
        else if (name == "h0_mask_pt")
            cxx_args.push_back({name, &layer.h0_mask_pt_});
        else if (name == "inv_n_pt")
            cxx_args.push_back({name, &layer.inv_n_pt_});
        else if (name == "iv_pt")
            cxx_args.push_back({name, &layer.iv_pt_});
        else if (name == "eps_add_pt")
            cxx_args.push_back({name, &layer.eps_add_pt_});
        else if (name == "output")
            cxx_args.push_back({name, &out_cts});
    }
    this->run(project_path, cxx_args);

    Array<double, 2> actual =
        unpack_par_diagonal_output(this->context, init_level - 3, out_cts, shape, head_shape, n_heads, false, false);
    print_double_message(actual.to_array_1d().data(), "actual", 10);
    print_double_message(expected.to_array_1d().data(), "expected", 10);
    auto comparison = compare(expected, actual);
    CAPTURE(comparison.max_error, comparison.max_abs, comparison.rmse, comparison.rms);
    REQUIRE(comparison.max_error < 5.0e-2 * comparison.max_abs);
    REQUIRE(comparison.rmse < 1.0e-2 * comparison.rms);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "par_upper_diagonal_ln_xcentered_cpp_roundtrip",
                               "[fhe_layers][par_upper_diagonal_layernorm][xcentered]",
                               HeteroProcessors) {
    const uint32_t n_prepad = 65;
    const uint32_t total_cols = 73;
    const uint32_t n_heads = 3;
    const uint32_t m_prepad = 20;
    const int init_level = 2;
    Duo shape = {n_prepad, total_cols};
    Duo head_shape = {n_prepad, m_prepad};
    fs::path project_path =
        base_path / "CKKS_par_upper_diagonal_layernorm" / "xcentered" / ("level_" + to_string(init_level)) / "server";
    if (!fs::exists(project_path / "mega_ag.json"))
        return;
    auto X_mat = gen_random_array<2>({n_prepad, total_cols}, 0.5);
    FeatureMatEncrypted X_enc(&this->context, init_level);
    X_enc.par_diagonal_pack(X_mat, n_heads, head_shape, false, false, this->param.default_scale());

    ParUpperDiagonalLNXCentered layer(this->param, shape, head_shape, n_heads, init_level);
    layer.prepare_weight();
    Array<double, 2> expected = layer.run_plaintext(X_mat);

    vector<CkksCiphertext> in_cts, out_cts;
    for (auto& ct : X_enc.data)
        in_cts.push_back(ct.copy());
    uint32_t n_out = par_diagonal_total_cts(this->param, shape, head_shape, n_heads, false);
    for (uint32_t i = 0; i < n_out; i++)
        out_cts.emplace_back(this->context.parameter(), init_level - 2);
        out_cts.back().set_scale(this->param.default_scale());

    auto arg_names = read_arg_names(project_path);
    vector<CxxVectorArgument> cxx_args;
    for (const auto& name : arg_names) {
        if (name == "input")
            cxx_args.push_back({name, &in_cts});
        else if (name == "h0_mask_pt")
            cxx_args.push_back({name, &layer.h0_mask_pt_});
        else if (name == "inv_n_pt")
            cxx_args.push_back({name, &layer.inv_n_pt_});
        else if (name == "output")
            cxx_args.push_back({name, &out_cts});
    }
    this->run(project_path, cxx_args);

    Array<double, 2> actual =
        unpack_par_diagonal_output(this->context, init_level - 2, out_cts, shape, head_shape, n_heads, false, false);
    print_double_message(actual.to_array_1d().data(), "actual", 10);
    print_double_message(expected.to_array_1d().data(), "expected", 10);
    auto comparison = compare(expected, actual);
    CAPTURE(comparison.max_error, comparison.max_abs, comparison.rmse, comparison.rms);
    REQUIRE(comparison.max_error < 5.0e-2 * comparison.max_abs);
    REQUIRE(comparison.rmse < 1.0e-2 * comparison.rms);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "par_upper_diagonal_ln_minimax_cpp_roundtrip",
                               "[fhe_layers][par_upper_diagonal_layernorm][minimax]",
                               HeteroProcessors) {
    const uint32_t n_prepad = 65;
    const uint32_t total_cols = 73;
    const uint32_t n_heads = 3;
    const uint32_t m_prepad = 20;
    const int input_level = 2;
    const double c0 = 6.19067182;
    const double c1 = -16.15885111;
    const double c2 = 11.52830778;
    Duo shape = {n_prepad, total_cols};
    Duo head_shape = {n_prepad, m_prepad};
    fs::path project_path =
        base_path / "CKKS_par_upper_diagonal_layernorm" / "minimax" / ("level_" + to_string(input_level)) / "server";
    if (!fs::exists(project_path / "mega_ag.json"))
        return;
    auto a_mat = gen_random_array<2>({n_prepad, total_cols}, 1.0);
    FeatureMatEncrypted a_input(&this->context, input_level);
    a_input.par_diagonal_pack(a_mat, n_heads, head_shape, false, false, this->param.default_scale());

    ParUpperDiagonalLNMinimaxInit layer(this->param, shape, head_shape, n_heads, input_level, c0, c1, c2);
    layer.prepare_weight();
    Array<double, 2> expected = layer.run_plaintext(a_mat);

    vector<CkksCiphertext> in_cts, out_cts;
    for (auto& ct : a_input.data)
        in_cts.push_back(ct.copy());
    uint32_t n_out = par_diagonal_total_cts(this->param, shape, head_shape, n_heads, false);
    for (uint32_t i = 0; i < n_out; i++)
        out_cts.emplace_back(this->context.parameter(), input_level - 2);
        out_cts.back().set_scale(this->param.default_scale());

    auto arg_names = read_arg_names(project_path);
    vector<CxxVectorArgument> cxx_args;
    for (const auto& name : arg_names) {
        if (name == "input")
            cxx_args.push_back({name, &in_cts});
        else if (name == "c0_add_pt")
            cxx_args.push_back({name, &layer.c0_add_pt_});
        else if (name == "c1_pt")
            cxx_args.push_back({name, &layer.c1_pt_});
        else if (name == "c2_norm_pt")
            cxx_args.push_back({name, &layer.c2_norm_pt_});
        else if (name == "output")
            cxx_args.push_back({name, &out_cts});
    }
    this->run(project_path, cxx_args);

    Array<double, 2> actual =
        unpack_par_diagonal_output(this->context, input_level - 2, out_cts, shape, head_shape, n_heads, false, false);
    print_double_message(actual.to_array_1d().data(), "actual", 10);
    print_double_message(expected.to_array_1d().data(), "expected", 10);
    auto comparison = compare(expected, actual);
    CAPTURE(comparison.max_error, comparison.max_abs, comparison.rmse, comparison.rms);
    REQUIRE(comparison.max_error < 5.0e-2 * comparison.max_abs);
    REQUIRE(comparison.rmse < 1.0e-2 * comparison.rms);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "par_upper_diagonal_ln_goldschmidt_cpp_roundtrip",
                               "[fhe_layers][par_upper_diagonal_layernorm][goldschmidt]",
                               HeteroProcessors) {
    const uint32_t n_prepad = 65;
    const uint32_t total_cols = 73;
    const uint32_t n_heads = 3;
    const uint32_t m_prepad = 20;
    const int input_level = 3;
    Duo shape = {n_prepad, total_cols};
    Duo head_shape = {n_prepad, m_prepad};
    fs::path project_path = base_path / "CKKS_par_upper_diagonal_layernorm" / "goldschmidt" /
                            ("level_" + to_string(input_level)) / "server";
    if (!fs::exists(project_path / "mega_ag.json"))
        return;
    auto y_mat = gen_random_array<2>({n_prepad, total_cols}, 1.0);
    auto a_mat = gen_random_array<2>({n_prepad, total_cols}, 1.0);
    FeatureMatEncrypted y_input(&this->context, input_level);
    FeatureMatEncrypted a_input(&this->context, input_level);
    y_input.par_diagonal_pack(y_mat, n_heads, head_shape, false, false, this->param.default_scale());
    a_input.par_diagonal_pack(a_mat, n_heads, head_shape, false, false, this->param.default_scale());

    ParUpperDiagonalLNGoldschmidt layer(this->param, shape, head_shape, n_heads, input_level);
    layer.prepare_weight();
    Array<double, 2> expected = layer.run_plaintext(y_mat, a_mat);

    vector<CkksCiphertext> y_cts, a_cts, out_cts;
    for (auto& ct : y_input.data)
        y_cts.push_back(ct.copy());
    for (auto& ct : a_input.data)
        a_cts.push_back(ct.copy());
    uint32_t n_out = par_diagonal_total_cts(this->param, shape, head_shape, n_heads, false);
    for (uint32_t i = 0; i < n_out; i++)
        out_cts.emplace_back(this->context.parameter(), input_level - 3);
        out_cts.back().set_scale(this->param.default_scale());

    auto arg_names = read_arg_names(project_path);
    vector<CxxVectorArgument> cxx_args;
    for (const auto& name : arg_names) {
        if (name == "y_input")
            cxx_args.push_back({name, &y_cts});
        else if (name == "a_input")
            cxx_args.push_back({name, &a_cts});
        else if (name == "three_pt")
            cxx_args.push_back({name, &layer.three_pt_});
        else if (name == "half_norm_pt")
            cxx_args.push_back({name, &layer.half_norm_pt_});
        else if (name == "output")
            cxx_args.push_back({name, &out_cts});
    }
    this->run(project_path, cxx_args);

    Array<double, 2> actual =
        unpack_par_diagonal_output(this->context, input_level - 3, out_cts, shape, head_shape, n_heads, false, false);
    print_double_message(actual.to_array_1d().data(), "actual", 10);
    print_double_message(expected.to_array_1d().data(), "expected", 10);
    auto comparison = compare(expected, actual);
    CAPTURE(comparison.max_error, comparison.max_abs, comparison.rmse, comparison.rms);
    REQUIRE(comparison.max_error < 5.0e-2 * comparison.max_abs);
    REQUIRE(comparison.rmse < 1.0e-2 * comparison.rms);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "par_upper_diagonal_ln_affine_cpp_roundtrip",
                               "[fhe_layers][par_upper_diagonal_layernorm][affine]",
                               HeteroProcessors) {
    const uint32_t n_prepad = 65;
    const uint32_t total_cols = 73;
    const uint32_t n_heads = 3;
    const uint32_t m_prepad = 20;
    const int y_level = 2;
    const double inv_std = 0.25;
    Duo shape = {n_prepad, total_cols};
    Duo head_shape = {n_prepad, m_prepad};
    fs::path project_path =
        base_path / "CKKS_par_upper_diagonal_layernorm" / "affine" / ("level_" + to_string(y_level)) / "server";
    if (!fs::exists(project_path / "mega_ag.json"))
        return;

    auto x_centered_mat = gen_random_array<2>({n_prepad, total_cols}, 1.0);
    auto y_mat = gen_random_array<2>({n_prepad, total_cols}, 1.0);
    auto gamma = gen_random_array<1>({total_cols}, 0.5);
    auto beta = gen_random_array<1>({total_cols}, 0.1);
    FeatureMatEncrypted x_centered_input(&this->context, y_level - 1);
    FeatureMatEncrypted y_input(&this->context, y_level);
    x_centered_input.par_diagonal_pack(x_centered_mat, n_heads, head_shape, false, false,
                                       this->param.default_scale());
    y_input.par_diagonal_pack(y_mat, n_heads, head_shape, false, false, this->param.default_scale());

    ParUpperDiagonalLNAffine layer(this->param, shape, head_shape, n_heads, y_level, inv_std, gamma.copy(),
                                   beta.copy());
    layer.prepare_weight();
    Array<double, 2> expected = layer.run_plaintext(x_centered_mat, y_mat);

    vector<CkksCiphertext> x_cts, y_cts, out_cts;
    for (auto& ct : x_centered_input.data)
        x_cts.push_back(ct.copy());
    for (auto& ct : y_input.data)
        y_cts.push_back(ct.copy());
    uint32_t n_out = par_diagonal_total_cts(this->param, shape, head_shape, n_heads, false);
    for (uint32_t i = 0; i < n_out; i++)
        out_cts.emplace_back(this->context.parameter(), y_level - 2);
        out_cts.back().set_scale(this->param.default_scale());

    auto arg_names = read_arg_names(project_path);
    vector<CxxVectorArgument> cxx_args;
    for (const auto& name : arg_names) {
        if (name == "x_centered")
            cxx_args.push_back({name, &x_cts});
        else if (name == "y_input")
            cxx_args.push_back({name, &y_cts});
        else if (name == "gamma_pt")
            cxx_args.push_back({name, &layer.gamma_pt_});
        else if (name == "beta_add_pt")
            cxx_args.push_back({name, &layer.beta_add_pt_});
        else if (name == "output")
            cxx_args.push_back({name, &out_cts});
    }
    this->run(project_path, cxx_args);

    Array<double, 2> actual =
        unpack_par_diagonal_output(this->context, y_level - 2, out_cts, shape, head_shape, n_heads, false, false);
    print_double_message(actual.to_array_1d().data(), "actual", 10);
    print_double_message(expected.to_array_1d().data(), "expected", 10);
    auto comparison = compare(expected, actual);
    CAPTURE(comparison.max_error, comparison.max_abs, comparison.rmse, comparison.rms);
    REQUIRE(comparison.max_error < 5.0e-2 * comparison.max_abs);
    REQUIRE(comparison.rmse < 1.0e-2 * comparison.rms);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "par_upper_diagonal_polyact_gamma_cpp_roundtrip",
                               "[fhe_layers][par_upper_diagonal_polyact][gamma]",
                               HeteroProcessors) {
    const uint32_t n_prepad = 65;
    const uint32_t total_cols = 73;
    const uint32_t n_heads = 3;
    const uint32_t m_prepad = 20;
    const uint32_t init_level = 1;
    Duo shape = {n_prepad, total_cols};
    Duo head_shape = {n_prepad, m_prepad};
    fs::path project_path =
        base_path / "CKKS_par_upper_diagonal_polyact" / "gamma" / ("level_" + to_string(init_level)) / "server";
    if (!fs::exists(project_path / "mega_ag.json"))
        return;

    auto X_mat = gen_random_array<2>({n_prepad, total_cols}, 0.5);
    auto gamma_vals = gen_random_array<1>({total_cols}, 0.5);
    FeatureMatEncrypted X_enc(&this->context, init_level);
    X_enc.par_diagonal_pack(X_mat, n_heads, head_shape, false, false, this->param.default_scale());

    ParUpperDiagonalPolyActRNGamma layer(this->param, shape, head_shape, n_heads, init_level, gamma_vals.copy());
    layer.prepare_weight();
    Array<double, 2> expected = layer.run_plaintext(X_mat);

    vector<CkksCiphertext> in_cts, out_cts;
    for (auto& ct : X_enc.data)
        in_cts.push_back(ct.copy());
    uint32_t n_out = par_diagonal_total_cts(this->param, shape, head_shape, n_heads, false);
    for (uint32_t i = 0; i < n_out; i++)
        out_cts.emplace_back(this->context.parameter(), init_level - 1);
        out_cts.back().set_scale(this->param.default_scale());

    auto arg_names = read_arg_names(project_path);
    vector<CxxVectorArgument> cxx_args;
    for (const auto& name : arg_names) {
        if (name == "input")
            cxx_args.push_back({name, &in_cts});
        else if (name == "gamma_pt")
            cxx_args.push_back({name, &layer.gamma_pt_});
        else if (name == "output")
            cxx_args.push_back({name, &out_cts});
    }
    this->run(project_path, cxx_args);

    Array<double, 2> actual =
        unpack_par_diagonal_output(this->context, init_level - 1, out_cts, shape, head_shape, n_heads, false, false);
    print_double_message(actual.to_array_1d().data(), "actual", 10);
    print_double_message(expected.to_array_1d().data(), "expected", 10);
    auto comparison = compare(expected, actual);
    CAPTURE(comparison.max_error, comparison.max_abs, comparison.rmse, comparison.rms);
    REQUIRE(comparison.max_error < 5.0e-2 * comparison.max_abs);
    REQUIRE(comparison.rmse < 1.0e-2 * comparison.rms);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "par_upper_diagonal_polyact_poly_cpp_roundtrip",
                               "[fhe_layers][par_upper_diagonal_polyact][poly]",
                               HeteroProcessors) {
    const uint32_t n_prepad = 65;
    const uint32_t total_cols = 73;
    const uint32_t n_heads = 3;
    const uint32_t m_prepad = 20;
    const uint32_t init_level = 3;
    const uint32_t degree = 4;
    Duo shape = {n_prepad, total_cols};
    Duo head_shape = {n_prepad, m_prepad};
    fs::path project_path = base_path / "CKKS_par_upper_diagonal_polyact" / "poly" / ("degree_" + to_string(degree)) /
                            ("level_" + to_string(init_level)) / "server";
    if (!fs::exists(project_path / "mega_ag.json"))
        return;

    auto X_mat = gen_random_array<2>({n_prepad, total_cols}, 0.5);
    auto coeffs = gen_random_array<2>({degree + 1, total_cols}, 0.2);
    FeatureMatEncrypted X_enc(&this->context, init_level);
    X_enc.par_diagonal_pack(X_mat, n_heads, head_shape, false, false, this->param.default_scale());

    ParUpperDiagonalPolyActRNPoly layer(this->param, shape, head_shape, n_heads, init_level, coeffs.copy(), degree);
    layer.prepare_weight();
    Array<double, 2> expected = layer.run_plaintext(X_mat);

    vector<CkksCiphertext> in_cts, out_cts;
    for (auto& ct : X_enc.data)
        in_cts.push_back(ct.copy());
    uint32_t n_out = par_diagonal_total_cts(this->param, shape, head_shape, n_heads, false);
    for (uint32_t i = 0; i < n_out; i++)
        out_cts.emplace_back(this->context.parameter(), init_level - 3);
        out_cts.back().set_scale(this->param.default_scale());

    auto arg_names = read_arg_names(project_path);
    vector<CxxVectorArgument> cxx_args;
    for (const auto& name : arg_names) {
        if (name == "input")
            cxx_args.push_back({name, &in_cts});
        else if (name == "c0_add_pt")
            cxx_args.push_back({name, &layer.c0_add_pt_});
        else if (name == "c1_pt")
            cxx_args.push_back({name, &layer.c1_pt_});
        else if (name == "c2_pt")
            cxx_args.push_back({name, &layer.c2_pt_});
        else if (name == "c3_pt")
            cxx_args.push_back({name, &layer.c3_pt_});
        else if (name == "c4_pt")
            cxx_args.push_back({name, &layer.c4_pt_});
        else if (name == "output")
            cxx_args.push_back({name, &out_cts});
    }
    this->run(project_path, cxx_args);

    Array<double, 2> actual =
        unpack_par_diagonal_output(this->context, init_level - 3, out_cts, shape, head_shape, n_heads, false, false);
    print_double_message(actual.to_array_1d().data(), "actual", 10);
    print_double_message(expected.to_array_1d().data(), "expected", 10);
    auto comparison = compare(expected, actual);
    CAPTURE(comparison.max_error, comparison.max_abs, comparison.rmse, comparison.rms);
    REQUIRE(comparison.max_error < 5.0e-2 * comparison.max_abs);
    REQUIRE(comparison.rmse < 1.0e-2 * comparison.rms);
}

// Test-only conjugate-invariant parameter chain with enough levels for the
// deep three-stage polynomial tests. The default PN15 chain keeps only 13 Q
// primes (max_level=12), while these tests need 18 levels. All moduli are
// 1 mod 4N for N=2^15, as required by the conjugate-invariant ring.
static CkksParameter make_long_ci_parameter() {
    static const vector<uint64_t> q = {
        0x7FFFFFFFE0001, 0x10000140001, 0xFFFFE80001, 0xFFFFC40001, 0x100003E0001, 0x10000500001,
        0xFFFFB20001,    0xFFFF940001,    0xFFFF8A0001, 0xFFFF820001, 0xFFFF780001, 0x10000960001,
        0x10000A40001,   0x10000B60001,   0x10000CE0001, 0x10000DE0001, 0x100010A0001, 0x10001680001,
    };
    static const vector<uint64_t> p = {
        0x4000000120001,
        0x3FFFFFFD20001,
    };
    return CkksParameter::create_custom_parameter(15, 40, q, p, RingType::ConjugateInvariant);
}

TEST_CASE("par_upper_diagonal_poly_stockmeyer_polygelu_cpp", "[fhe_layers][par_upper_diagonal_poly][stockmeyer]") {
    const uint32_t N = 1 << 15;
    CkksParameter param = make_long_ci_parameter();
    const int init_level = param.max_level();
    CkksContext context = CkksContext::create_random_context(param);

    const uint32_t n_slot = param.max_slots();
    const uint32_t n_prepad = 128;
    const uint32_t n_heads = 2;
    const uint32_t m_prepad = 128;
    const uint32_t total_cols = 256;
    const uint32_t order = 15;
    const double scale = 64.0;
    const double f2_input_scale = 0.5;
    const double f3_input_scale = 0.5;
    const double default_scale = param.default_scale();
    Duo shape = {n_prepad, total_cols};
    Duo head_shape = {n_prepad, m_prepad};

    REQUIRE(n_slot == n_prepad * total_cols);
    REQUIRE(init_level > 0);

    const vector<double> p1_desc = {
        -0.029080343483084341, 2.3066773634990557e-17,  0.45471038861073326, -2.8078925777913385e-16,
        -2.8915411182152724,   1.2572521416511905e-15,  9.6019913623544397,  -2.3997776216928771e-15,
        -17.785134844150317,   1.4296496501542554e-15,  18.198630605563167,  5.618746855114301e-16,
        -9.6389986430588284,   -2.7157761908506778e-16, 2.6150871121857149,  -2.1483406699330675e-17,
    };
    const vector<double> p2_desc = {
        -22.772929128163085, -9.9238473151061153e-13, 133.52575642060197, 5.1885599061549671e-12,
        -322.28293092520312, -1.057124065245549e-11,  412.46567940639022, 1.0572474934837157e-11,
        -300.32201839910158, -5.3476054574106177e-12, 124.10645061245243, 1.2506387010969551e-12,
        -27.696116404655168, -1.0228470304160076e-13, 3.4795291737254996, 9.2762077723673713e-16,
    };
    const vector<double> p3_desc = {
        -3.2153402192444749,  -1.9791353470822114e-13, 17.416597794210833,  5.9631920964000009e-13,
        -36.618586273953056,  -6.1161725269934055e-13, 38.03096482489083,   2.6665573637468773e-13,
        -20.470163356627168,  -1.136498515409159e-13,  5.5037834886270582,  8.0623177269952888e-14,
        -0.86849898556709171, -1.9246732444228731e-14, 0.72131651367427774, -1.0400333128983701e-15,
    };

    auto to_increasing = [](const vector<double>& desc) { return vector<double>(desc.rbegin(), desc.rend()); };
    const vector<double> p1 = to_increasing(p1_desc);
    const vector<double> p2 = to_increasing(p2_desc);
    const vector<double> p3 = to_increasing(p3_desc);

    auto eval_poly = [](double x, const vector<double>& coeffs) {
        double result = 0.0;
        for (auto it = coeffs.rbegin(); it != coeffs.rend(); ++it) {
            result = result * x + *it;
        }
        return result;
    };

    auto make_coeff_matrix = [&](const vector<double>& coeffs) {
        Array<double, 2> result({static_cast<uint64_t>(coeffs.size()), static_cast<uint64_t>(total_cols)});
        for (uint32_t k = 0; k < coeffs.size(); k++) {
            for (uint32_t j = 0; j < total_cols; j++) {
                result.set(k, j, coeffs[k]);
            }
        }
        return result;
    };

    Array<double, 2> X_mat({n_prepad, total_cols});
    for (uint32_t idx = 0; idx < n_slot; idx++) {
        double x = -128.0 + 256.0 * static_cast<double>(idx) / static_cast<double>(n_slot - 1);
        X_mat.set(idx / total_cols, idx % total_cols, x);
    }

    Array<double, 2> expected({n_prepad, total_cols});
    for (uint32_t i = 0; i < n_prepad; i++) {
        for (uint32_t j = 0; j < total_cols; j++) {
            double x = X_mat.get(i, j);
            double f1 = eval_poly(x / scale, p1);
            double half_tanh = eval_poly(f1 / f2_input_scale, p2);
            double refined = eval_poly(half_tanh / f3_input_scale, p3);
            expected.set(i, j, x * (0.5 + refined));
        }
    }

    FeatureMatEncrypted X_enc(&context, init_level);
    X_enc.par_diagonal_pack(X_mat, n_heads, head_shape, false, false, default_scale);

    auto make_scalar_gamma = [&](double scalar) {
        Array<double, 1> gamma({total_cols});
        for (uint32_t col = 0; col < total_cols; col++) {
            gamma.set(col, scalar);
        }
        return gamma;
    };

    auto multiply_by_scalar = [&](const FeatureMatEncrypted& input, double scalar) {
        ParUpperDiagonalPolyActRNGamma scalar_layer(param, shape, head_shape, n_heads, input.level,
                                                    make_scalar_gamma(scalar));
        scalar_layer.prepare_weight();
        return scalar_layer.run(context, input);
    };

    FeatureMatEncrypted X_scaled = multiply_by_scalar(X_enc, 1.0 / scale);
    REQUIRE(X_scaled.level == init_level - 1);

    ParUpperDiagonalPoly poly1(param, shape, head_shape, n_heads, X_scaled.level, make_coeff_matrix(p1), order);
    poly1.prepare_weight_stockmeyer();
    FeatureMatEncrypted f1_enc = poly1.run_stockmeyer(context, X_scaled);
    REQUIRE(f1_enc.level == init_level - 5);

    FeatureMatEncrypted f1_scaled_for_poly2 = multiply_by_scalar(f1_enc, 1.0 / f2_input_scale);
    REQUIRE(f1_scaled_for_poly2.level == init_level - 6);

    ParUpperDiagonalPoly poly2(param, shape, head_shape, n_heads, f1_scaled_for_poly2.level, make_coeff_matrix(p2),
                               order);
    poly2.prepare_weight_stockmeyer();
    FeatureMatEncrypted half_tanh_enc = poly2.run_stockmeyer(context, f1_scaled_for_poly2);
    REQUIRE(half_tanh_enc.level == init_level - 10);

    FeatureMatEncrypted half_tanh_scaled_for_poly3 = multiply_by_scalar(half_tanh_enc, 1.0 / f3_input_scale);
    REQUIRE(half_tanh_scaled_for_poly3.level == init_level - 11);

    ParUpperDiagonalPoly poly3(param, shape, head_shape, n_heads, half_tanh_scaled_for_poly3.level,
                               make_coeff_matrix(p3), order);
    poly3.prepare_weight_stockmeyer();
    FeatureMatEncrypted half_tanh_refined = poly3.run_stockmeyer(context, half_tanh_scaled_for_poly3);
    REQUIRE(half_tanh_refined.level == init_level - 15);

    FeatureMatEncrypted X_for_gelu = X_enc.drop_level(X_enc.level - half_tanh_refined.level);
    REQUIRE(X_for_gelu.level == half_tanh_refined.level);
    ParUpperDiagonalPolyMultCt gelu_layer(param, shape, head_shape, n_heads, half_tanh_refined.level);
    gelu_layer.prepare_weight();
    FeatureMatEncrypted gelu_enc = gelu_layer.run(context, half_tanh_refined, X_for_gelu);
    REQUIRE(gelu_enc.level == init_level - 17);
    CAPTURE(gelu_enc.data[0].scale(), default_scale);
    REQUIRE(fabs(gelu_enc.data[0].scale() - default_scale) <= default_scale * 1.0e-12);

    Array<double, 2> output_mg = gelu_enc.par_diagonal_unpack(n_heads, head_shape, false, false);
    Array<double, 2> plain_output = expected.copy();
    auto output_mg_1d = output_mg.to_array_1d();
    auto plain_output_1d = plain_output.to_array_1d();
    print_double_message(output_mg_1d.data(), "output_mg", 10);
    print_double_message(plain_output_1d.data(), "plain_output", 10);
    print_double_message(output_mg_1d.data() + output_mg_1d.size() - 10, "output_mg_last", 10);
    print_double_message(plain_output_1d.data() + plain_output_1d.size() - 10, "plain_output_last", 10);

    auto compare_result = compare(plain_output, output_mg);
    std::cout << "max_error=" << compare_result.max_error << " max_abs=" << compare_result.max_abs
              << " rmse=" << compare_result.rmse << " rms=" << compare_result.rms << std::endl;
    CAPTURE(compare_result.max_error, compare_result.max_abs, compare_result.rmse, compare_result.rms);
    REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
    REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
}

TEST_CASE("par_upper_diagonal_poly_stockmeyer_polytanh_cpp", "[fhe_layers][par_upper_diagonal_poly][stockmeyer]") {
    const uint32_t N = 1 << 15;
    CkksParameter param = CkksParameter::create_parameter(static_cast<int>(std::log2(N)), RingType::ConjugateInvariant);
    const int init_level = 9;
    CkksContext context = CkksContext::create_random_context(param);

    const uint32_t n_slot = param.max_slots();
    const uint32_t n_prepad = 128;
    const uint32_t n_heads = 2;
    const uint32_t m_prepad = 128;
    const uint32_t total_cols = 256;
    const uint32_t order = 15;
    const int stockmeyer_level_cost = PolyBase::compute_stockmeyer_level_cost(order);
    const double input_scale = 40.0;
    const double default_scale = param.default_scale();
    Duo shape = {n_prepad, total_cols};
    Duo head_shape = {n_prepad, m_prepad};

    REQUIRE(n_slot == n_prepad * total_cols);
    REQUIRE(init_level >= 1 + 2 * stockmeyer_level_cost);

    const vector<double> p1_desc = {
        -7.14529052e+03, -7.76519925e+01, 2.74279201e+04,  2.45150249e+02,  -4.25793697e+04, -3.01953016e+02,
        3.42189880e+04,  1.82989351e+02,  -1.51158283e+04, -5.64098990e+01, 3.58757327e+03,  8.17596753e+00,
        -4.13341496e+02, -4.29024545e-01, 1.95056729e+01,  2.06201784e-03,
    };
    const vector<double> p2_desc = {
        -9.02573450e-03, -1.12320034e-04, 1.08762008e-01,  7.96793166e-04,  -5.41327356e-01, -1.42873183e-03,
        1.46476749e+00,  -2.22416152e-03, -2.43259032e+00, 1.17381072e-02,  2.74974898e+00,  -1.77631073e-02,
        -2.38934873e+00, 1.30194294e-02,  2.02874846e+00,  -4.08442578e-03,
    };

    auto to_increasing = [](const vector<double>& desc) { return vector<double>(desc.rbegin(), desc.rend()); };
    auto rescale_coeffs_for_input_multiplier = [](vector<double> coeffs, double input_multiplier) {
        double degree_scale = 1.0;
        for (double& coeff : coeffs) {
            coeff *= degree_scale;
            degree_scale /= input_multiplier;
        }
        return coeffs;
    };
    const vector<double> p1 = rescale_coeffs_for_input_multiplier(to_increasing(p1_desc), 2.0);
    const vector<double> p2 = to_increasing(p2_desc);

    auto make_coeff_matrix = [&](const vector<double>& coeffs) {
        Array<double, 2> result({static_cast<uint64_t>(coeffs.size()), static_cast<uint64_t>(total_cols)});
        for (uint32_t k = 0; k < coeffs.size(); k++) {
            for (uint32_t j = 0; j < total_cols; j++) {
                result.set(k, j, coeffs[k]);
            }
        }
        return result;
    };

    auto make_scalar_gamma = [&](double scalar) {
        Array<double, 1> gamma({total_cols});
        for (uint32_t col = 0; col < total_cols; col++) {
            gamma.set(col, scalar);
        }
        return gamma;
    };

    Array<double, 2> X_mat({n_prepad, total_cols});
    for (uint32_t idx = 0; idx < n_slot; idx++) {
        double x = -40.0 + 80.0 * static_cast<double>(idx) / static_cast<double>(n_slot - 1);
        X_mat.set(idx / total_cols, idx % total_cols, x);
    }

    ParUpperDiagonalPolyActRNGamma input_scale_layer(param, shape, head_shape, n_heads, init_level,
                                                     make_scalar_gamma(2.0 / input_scale));
    ParUpperDiagonalPoly poly1(param, shape, head_shape, n_heads, init_level - 1, make_coeff_matrix(p1), order);
    ParUpperDiagonalPoly poly2(param, shape, head_shape, n_heads, init_level - 1 - stockmeyer_level_cost,
                               make_coeff_matrix(p2), order);

    Array<double, 2> X_scaled_plain = input_scale_layer.run_plaintext(X_mat);
    Array<double, 2> f1_plain = poly1.run_plaintext(X_scaled_plain);
    Array<double, 2> expected = poly2.run_plaintext(f1_plain);
    Array<double, 2> ground_truth({n_prepad, total_cols});
    for (uint32_t i = 0; i < n_prepad; i++) {
        for (uint32_t j = 0; j < total_cols; j++) {
            ground_truth.set(i, j, tanh(X_mat.get(i, j)));
        }
    }

    FeatureMatEncrypted X_enc(&context, init_level);
    X_enc.par_diagonal_pack(X_mat, n_heads, head_shape, false, false, default_scale);

    input_scale_layer.prepare_weight();
    FeatureMatEncrypted X_scaled = input_scale_layer.run(context, X_enc);
    REQUIRE(X_scaled.level == init_level - 1);

    poly1.prepare_weight_stockmeyer();
    FeatureMatEncrypted f1_enc = poly1.run_stockmeyer(context, X_scaled);
    REQUIRE(f1_enc.level == X_scaled.level - stockmeyer_level_cost);

    poly2.prepare_weight_stockmeyer();
    FeatureMatEncrypted tanh_enc = poly2.run_stockmeyer(context, f1_enc);
    REQUIRE(tanh_enc.level == f1_enc.level - stockmeyer_level_cost);
    CAPTURE(tanh_enc.data[0].scale(), default_scale);
    REQUIRE(fabs(tanh_enc.data[0].scale() - default_scale) <= default_scale * 1.0e-12);

    Array<double, 2> output_mg = tanh_enc.par_diagonal_unpack(n_heads, head_shape, false, false);
    Array<double, 2> plain_output = expected.copy();
    Array<double, 2> ground_truth_output = ground_truth.copy();
    auto output_mg_1d = output_mg.to_array_1d();
    auto plain_output_1d = plain_output.to_array_1d();
    auto ground_truth_1d = ground_truth_output.to_array_1d();
    print_double_message(output_mg_1d.data(), "output_mg", 10);
    print_double_message(plain_output_1d.data(), "plain_output", 10);
    print_double_message(ground_truth_1d.data(), "ground_truth", 10);
    print_double_message(output_mg_1d.data() + output_mg_1d.size() - 10, "output_mg_last", 10);
    print_double_message(plain_output_1d.data() + plain_output_1d.size() - 10, "plain_output_last", 10);
    print_double_message(ground_truth_1d.data() + ground_truth_1d.size() - 10, "ground_truth_last", 10);

    auto compare_result = compare(plain_output, output_mg);
    auto ground_truth_compare_result = compare(ground_truth_output, output_mg);
    std::cout << "tanh max_error=" << compare_result.max_error << " max_abs=" << compare_result.max_abs
              << " rmse=" << compare_result.rmse << " rms=" << compare_result.rms << std::endl;
    std::cout << "tanh ground_truth max_error=" << ground_truth_compare_result.max_error
              << " max_abs=" << ground_truth_compare_result.max_abs << " rmse=" << ground_truth_compare_result.rmse
              << " rms=" << ground_truth_compare_result.rms << std::endl;
    CAPTURE(compare_result.max_error, compare_result.max_abs, compare_result.rmse, compare_result.rms);
    CAPTURE(ground_truth_compare_result.max_error, ground_truth_compare_result.max_abs,
            ground_truth_compare_result.rmse, ground_truth_compare_result.rms);
    REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
    REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
}

TEST_CASE("par_upper_diagonal_poly_stockmeyer_horner_polygelu_cpp",
          "[fhe_layers][par_upper_diagonal_poly][stockmeyer][horner]") {
    const uint32_t N = 1 << 15;
    CkksParameter param = make_long_ci_parameter();

    const uint32_t n_slot = param.max_slots();
    const uint32_t n_prepad = 128;
    const uint32_t n_heads = 2;
    const uint32_t m_prepad = 128;
    const uint32_t total_cols = 256;
    const uint32_t order = 15;
    const int horner_baby_steps = 8;
    const int horner_level_cost = PolyBase::compute_stockmeyer_horner_level_cost(order, horner_baby_steps);
    const int init_level = 3 * horner_level_cost + 5;
    CkksContext context = CkksContext::create_random_context(param);

    const double scale = 64.0;
    const double f2_input_scale = 0.5;
    const double f3_input_scale = 0.5;
    const double default_scale = param.default_scale();
    Duo shape = {n_prepad, total_cols};
    Duo head_shape = {n_prepad, m_prepad};

    REQUIRE(n_slot == n_prepad * total_cols);
    REQUIRE(horner_baby_steps == 8);
    REQUIRE(horner_level_cost == 4);

    const vector<double> p1_desc = {
        -0.029080343483084341, 2.3066773634990557e-17,  0.45471038861073326, -2.8078925777913385e-16,
        -2.8915411182152724,   1.2572521416511905e-15,  9.6019913623544397,  -2.3997776216928771e-15,
        -17.785134844150317,   1.4296496501542554e-15,  18.198630605563167,  5.618746855114301e-16,
        -9.6389986430588284,   -2.7157761908506778e-16, 2.6150871121857149,  -2.1483406699330675e-17,
    };
    const vector<double> p2_desc = {
        -22.772929128163085, -9.9238473151061153e-13, 133.52575642060197, 5.1885599061549671e-12,
        -322.28293092520312, -1.057124065245549e-11,  412.46567940639022, 1.0572474934837157e-11,
        -300.32201839910158, -5.3476054574106177e-12, 124.10645061245243, 1.2506387010969551e-12,
        -27.696116404655168, -1.0228470304160076e-13, 3.4795291737254996, 9.2762077723673713e-16,
    };
    const vector<double> p3_desc = {
        -3.2153402192444749,  -1.9791353470822114e-13, 17.416597794210833,  5.9631920964000009e-13,
        -36.618586273953056,  -6.1161725269934055e-13, 38.03096482489083,   2.6665573637468773e-13,
        -20.470163356627168,  -1.136498515409159e-13,  5.5037834886270582,  8.0623177269952888e-14,
        -0.86849898556709171, -1.9246732444228731e-14, 0.72131651367427774, -1.0400333128983701e-15,
    };

    auto to_increasing = [](const vector<double>& desc) { return vector<double>(desc.rbegin(), desc.rend()); };
    const vector<double> p1 = to_increasing(p1_desc);
    const vector<double> p2 = to_increasing(p2_desc);
    const vector<double> p3 = to_increasing(p3_desc);

    auto eval_poly = [](double x, const vector<double>& coeffs) {
        double result = 0.0;
        for (auto it = coeffs.rbegin(); it != coeffs.rend(); ++it) {
            result = result * x + *it;
        }
        return result;
    };

    auto make_coeff_matrix = [&](const vector<double>& coeffs) {
        Array<double, 2> result({static_cast<uint64_t>(coeffs.size()), static_cast<uint64_t>(total_cols)});
        for (uint32_t k = 0; k < coeffs.size(); k++) {
            for (uint32_t j = 0; j < total_cols; j++) {
                result.set(k, j, coeffs[k]);
            }
        }
        return result;
    };

    Array<double, 2> X_mat({n_prepad, total_cols});
    for (uint32_t idx = 0; idx < n_slot; idx++) {
        double x = -128.0 + 256.0 * static_cast<double>(idx) / static_cast<double>(n_slot - 1);
        X_mat.set(idx / total_cols, idx % total_cols, x);
    }

    Array<double, 2> expected({n_prepad, total_cols});
    for (uint32_t i = 0; i < n_prepad; i++) {
        for (uint32_t j = 0; j < total_cols; j++) {
            double x = X_mat.get(i, j);
            double f1 = eval_poly(x / scale, p1);
            double half_tanh = eval_poly(f1 / f2_input_scale, p2);
            double refined = eval_poly(half_tanh / f3_input_scale, p3);
            expected.set(i, j, x * (0.5 + refined));
        }
    }

    FeatureMatEncrypted X_enc(&context, init_level);
    X_enc.par_diagonal_pack(X_mat, n_heads, head_shape, false, false, default_scale);

    auto make_scalar_gamma = [&](double scalar) {
        Array<double, 1> gamma({total_cols});
        for (uint32_t col = 0; col < total_cols; col++) {
            gamma.set(col, scalar);
        }
        return gamma;
    };

    auto multiply_by_scalar = [&](const FeatureMatEncrypted& input, double scalar) {
        ParUpperDiagonalPolyActRNGamma scalar_layer(param, shape, head_shape, n_heads, input.level,
                                                    make_scalar_gamma(scalar));
        scalar_layer.prepare_weight();
        return scalar_layer.run(context, input);
    };

    FeatureMatEncrypted X_scaled = multiply_by_scalar(X_enc, 1.0 / scale);
    REQUIRE(X_scaled.level == init_level - 1);

    ParUpperDiagonalPoly poly1(param, shape, head_shape, n_heads, X_scaled.level, make_coeff_matrix(p1), order);
    poly1.prepare_weight_stockmeyer_horner(horner_baby_steps);
    FeatureMatEncrypted f1_enc = poly1.run_stockmeyer_horner(context, X_scaled, horner_baby_steps);
    REQUIRE(f1_enc.level == X_scaled.level - horner_level_cost);

    FeatureMatEncrypted f1_scaled_for_poly2 = multiply_by_scalar(f1_enc, 1.0 / f2_input_scale);
    REQUIRE(f1_scaled_for_poly2.level == f1_enc.level - 1);

    ParUpperDiagonalPoly poly2(param, shape, head_shape, n_heads, f1_scaled_for_poly2.level, make_coeff_matrix(p2),
                               order);
    poly2.prepare_weight_stockmeyer_horner(horner_baby_steps);
    FeatureMatEncrypted half_tanh_enc = poly2.run_stockmeyer_horner(context, f1_scaled_for_poly2, horner_baby_steps);
    REQUIRE(half_tanh_enc.level == f1_scaled_for_poly2.level - horner_level_cost);

    FeatureMatEncrypted half_tanh_scaled_for_poly3 = multiply_by_scalar(half_tanh_enc, 1.0 / f3_input_scale);
    REQUIRE(half_tanh_scaled_for_poly3.level == half_tanh_enc.level - 1);

    ParUpperDiagonalPoly poly3(param, shape, head_shape, n_heads, half_tanh_scaled_for_poly3.level,
                               make_coeff_matrix(p3), order);
    poly3.prepare_weight_stockmeyer_horner(horner_baby_steps);
    FeatureMatEncrypted half_tanh_refined =
        poly3.run_stockmeyer_horner(context, half_tanh_scaled_for_poly3, horner_baby_steps);
    REQUIRE(half_tanh_refined.level == half_tanh_scaled_for_poly3.level - horner_level_cost);

    FeatureMatEncrypted X_for_gelu = X_enc.drop_level(X_enc.level - half_tanh_refined.level);
    REQUIRE(X_for_gelu.level == half_tanh_refined.level);
    ParUpperDiagonalPolyMultCt gelu_layer(param, shape, head_shape, n_heads, half_tanh_refined.level);
    gelu_layer.prepare_weight();
    FeatureMatEncrypted gelu_enc = gelu_layer.run(context, half_tanh_refined, X_for_gelu);
    REQUIRE(gelu_enc.level == init_level - (3 * horner_level_cost + 5));
    CAPTURE(gelu_enc.data[0].scale(), default_scale);
    REQUIRE(fabs(gelu_enc.data[0].scale() - default_scale) <= default_scale * 1.0e-12);

    Array<double, 2> output_mg = gelu_enc.par_diagonal_unpack(n_heads, head_shape, false, false);
    auto compare_result = compare(expected, output_mg);
    std::cout << "horner max_error=" << compare_result.max_error << " max_abs=" << compare_result.max_abs
              << " rmse=" << compare_result.rmse << " rms=" << compare_result.rms << std::endl;
    CAPTURE(compare_result.max_error, compare_result.max_abs, compare_result.rmse, compare_result.rms);
    REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
    REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
}

TEST_CASE("par_upper_diagonal_polysoftmax_cpp", "[fhe_layers][par_upper_diagonal_softmax]") {
    CkksParameter param = CkksParameter::create_parameter(15, RingType::ConjugateInvariant);
    const double default_scale = param.default_scale();
    const uint32_t plaintext_level = static_cast<uint32_t>(param.max_level());
    const uint32_t n_prepad = 96;
    const uint32_t n_heads = 2;
    const uint32_t total_cols = n_prepad * n_heads;
    const uint32_t max_inverse_iterations = 15;
    Duo shape = {n_prepad, total_cols};
    Duo head_shape = {n_prepad, n_prepad};

    struct SoftmaxProfile {
        string name;
        double range_min;
        double range_max;
        double exp_divisor;
        double initial_denominator_scale;
        double first_refinement_denominator_scale;
        double later_refinement_denominator_scale;
        uint32_t delta_2;
        vector<double> coeffs;
    };

    const vector<SoftmaxProfile> profiles = {
        {"thor-small",
         -27.2493,
         21.72692,
         32.0,
         16.0,
         2.0,
         1.0,
         2,
         {0.0006522770224130905, 0.005218196900295354, 0.020873931555133732, 0.05566510463488879, 0.11128698173597897,
          0.1780344447042791, 0.2379982262906916, 0.27220647178765545, 0.26787311936472025, 0.23721029007596767,
          0.20618261949210986, 0.15202099984697098, 0.0670090353368128, 0.03881607331549499, 0.05948672763856172,
          0.032855468333339584}},
        {"thor-wide",
         -70.0,
         70.0,
         64.0,
         16.0,
         2.0,
         1.0,
         4,
         {8.615994668877663e-05, 0.0006877176070101981, 0.0027930565779849003, 0.00749909544008294,
          0.014216636671807894, 0.022268203231858744, 0.03517495704921328, 0.04217318306400685, 0.02336452059478656,
          0.016604320800445653, 0.04817928878801878, 0.0397324053296174, -0.009262268572236316, -0.008386712802267769,
          0.014226972463907047, 0.008201736399899691}},
    };

    auto make_coeff_matrix = [&](const vector<double>& coeffs) {
        Array<double, 2> result({static_cast<uint64_t>(coeffs.size()), static_cast<uint64_t>(total_cols)});
        for (uint32_t k = 0; k < coeffs.size(); k++) {
            for (uint32_t j = 0; j < total_cols; j++) {
                result.set(k, j, coeffs[k]);
            }
        }
        return result;
    };

    auto make_scalar_gamma = [&](double scalar) {
        Array<double, 1> gamma({total_cols});
        for (uint32_t col = 0; col < total_cols; col++) {
            gamma.set(col, scalar);
        }
        return gamma;
    };

    auto run_ground_truth = [&](const Array<double, 2>& input) {
        Array<double, 2> result({n_prepad, total_cols});
        for (uint32_t row = 0; row < n_prepad; row++) {
            for (uint32_t h = 0; h < n_heads; h++) {
                uint32_t base = h * n_prepad;
                double row_max = input.get(row, base);
                for (uint32_t col = 0; col < n_prepad; col++) {
                    row_max = std::max(row_max, input.get(row, base + col));
                }
                double denominator = 0.0;
                for (uint32_t col = 0; col < n_prepad; col++) {
                    denominator += exp(input.get(row, base + col) - row_max);
                }
                for (uint32_t col = 0; col < n_prepad; col++) {
                    result.set(row, base + col, exp(input.get(row, base + col) - row_max) / denominator);
                }
            }
        }
        return result;
    };

    auto run_normalize_plain = [&](const Array<double, 2>& values, double denominator_scale) {
        ParUpperDiagonalHeadColSum sum_layer(param, shape, head_shape, n_heads, plaintext_level);
        Array<double, 2> denominator = sum_layer.run_plaintext(values);
        if (denominator_scale != 1.0) {
            ParUpperDiagonalPolyActRNGamma denom_scale_layer(param, shape, head_shape, n_heads, plaintext_level,
                                                             make_scalar_gamma(denominator_scale));
            denominator = denom_scale_layer.run_plaintext(denominator);
        }

        ParUpperDiagonalInverseInit inv_init(param, shape, head_shape, n_heads, plaintext_level);
        Array<double, 2> inv_denominator = inv_init.run_plaintext(denominator);

        for (uint32_t iter = 1; iter < max_inverse_iterations; iter++) {
            ParUpperDiagonalInverseIter inv_iter(param, shape, head_shape, n_heads, plaintext_level);
            inv_denominator = inv_iter.run_plaintext(inv_denominator, denominator);
        }

        if (denominator_scale != 1.0) {
            ParUpperDiagonalPolyActRNGamma inv_scale_layer(param, shape, head_shape, n_heads, plaintext_level,
                                                           make_scalar_gamma(denominator_scale));
            inv_denominator = inv_scale_layer.run_plaintext(inv_denominator);
        }

        ParUpperDiagonalGELU mult_layer(param, shape, head_shape, n_heads, plaintext_level);
        return mult_layer.run_plaintext(values, inv_denominator);
    };

    auto run_expected = [&](const Array<double, 2>& input, const SoftmaxProfile& profile) {
        double mid = (profile.range_min + profile.range_max) / 2.0;
        ParUpperDiagonalAddPt add_mid(param, shape, head_shape, n_heads, plaintext_level, -mid);
        Array<double, 2> centered = add_mid.run_plaintext(input);

        ParUpperDiagonalPolyActRNGamma gamma_layer(param, shape, head_shape, n_heads, plaintext_level,
                                                   make_scalar_gamma(1.0 / profile.exp_divisor));
        Array<double, 2> exp_arg = gamma_layer.run_plaintext(centered);

        uint32_t order = static_cast<uint32_t>(profile.coeffs.size() - 1);
        ParUpperDiagonalPoly exp_poly(param, shape, head_shape, n_heads, plaintext_level,
                                      make_coeff_matrix(profile.coeffs), order);
        Array<double, 2> exp_poly_out = exp_poly.run_plaintext(exp_arg);

        ParUpperDiagonalMultipleSquare delta1_square(param, shape, head_shape, n_heads, plaintext_level);
        Array<double, 2> values = delta1_square.run_plaintext(exp_poly_out);

        Array<double, 2> probabilities = run_normalize_plain(values, profile.initial_denominator_scale);
        uint32_t refinement_steps = 0;
        for (uint32_t delta = profile.delta_2; delta > 1; delta >>= 1) {
            refinement_steps++;
        }
        for (uint32_t step = 0; step < refinement_steps; step++) {
            ParUpperDiagonalGELU square_layer(param, shape, head_shape, n_heads, plaintext_level);
            Array<double, 2> squared = square_layer.run_plaintext(probabilities, probabilities);
            double denominator_scale =
                (step == 0) ? profile.first_refinement_denominator_scale : profile.later_refinement_denominator_scale;
            probabilities = run_normalize_plain(squared, denominator_scale);
        }
        return probabilities;
    };

    auto refresh_to_min_level = [&](FeatureMatEncrypted&& x, uint32_t min_level) {
        if (x.level < min_level) {
            if (x.level > 0) {
                x = x.drop_level(x.level);
            }
            x = x.refresh_ciphertext();
        }
        REQUIRE(x.level >= min_level);
        return std::move(x);
    };

    auto align_to_level = [&](FeatureMatEncrypted&& x, uint32_t target_level) {
        if (x.level < target_level) {
            if (x.level > 0) {
                x = x.drop_level(x.level);
            }
            x = x.refresh_ciphertext();
        }
        REQUIRE(x.level >= target_level);
        if (x.level > target_level) {
            x = x.drop_level(static_cast<int>(x.level - target_level));
        }
        REQUIRE(x.level == target_level);
        return std::move(x);
    };

    auto multiply_scalar = [&](CkksContext& context, FeatureMatEncrypted&& x, double scalar) {
        REQUIRE(x.level >= 1);
        ParUpperDiagonalPolyActRNGamma gamma_layer(param, shape, head_shape, n_heads, x.level,
                                                   make_scalar_gamma(scalar));
        gamma_layer.prepare_weight();
        return gamma_layer.run(context, x);
    };

    auto refresh_inverse_to_min_level = [&](CkksContext& context, FeatureMatEncrypted&& x, uint32_t min_level) {
        if (x.level < min_level) {
            constexpr double inverse_refresh_scale = 8.0;
            x = multiply_scalar(context, std::move(x), 1.0 / inverse_refresh_scale);
            if (x.level > 0) {
                x = x.drop_level(x.level);
            }
            x = x.refresh_ciphertext();
            x = multiply_scalar(context, std::move(x), inverse_refresh_scale);
        }
        REQUIRE(x.level >= min_level);
        return std::move(x);
    };

    auto run_normalize_encrypt = [&](CkksContext& context, const FeatureMatEncrypted& values_in,
                                     double denominator_scale) {
        FeatureMatEncrypted values = refresh_to_min_level(values_in.drop_level(0), 1);
        ParUpperDiagonalHeadColSum sum_layer(param, shape, head_shape, n_heads, values.level);
        sum_layer.prepare_weight();
        FeatureMatEncrypted denominator = sum_layer.run(context, values);
        if (denominator_scale != 1.0) {
            // THOR scales small inverse denominators before bootstrap; direct low-level bootstrap
            // of tiny sums is noisy.
            if (denominator.level < 2) {
                constexpr double denominator_refresh_scale = 16.0;
                denominator = multiply_scalar(context, std::move(denominator), denominator_refresh_scale);
                if (denominator.level > 0) {
                    denominator = denominator.drop_level(denominator.level);
                }
                denominator = denominator.refresh_ciphertext();
                denominator = multiply_scalar(context, std::move(denominator), 1.0 / denominator_refresh_scale);
            } else {
                denominator = refresh_to_min_level(std::move(denominator), 2);
            }
            denominator = multiply_scalar(context, std::move(denominator), denominator_scale);
        }

        ParUpperDiagonalInverseInit inv_init(param, shape, head_shape, n_heads, denominator.level);
        inv_init.prepare_weight();
        FeatureMatEncrypted inv_denominator = inv_init.run(context, denominator);
        FeatureMatEncrypted denominator_work = denominator.drop_level(0);

        for (uint32_t iter = 1; iter < max_inverse_iterations; iter++) {
            inv_denominator = refresh_inverse_to_min_level(context, std::move(inv_denominator), 4);
            denominator_work = align_to_level(std::move(denominator_work), inv_denominator.level);
            ParUpperDiagonalInverseIter inv_iter(param, shape, head_shape, n_heads, inv_denominator.level);
            inv_iter.prepare_weight();
            inv_denominator = inv_iter.run(context, inv_denominator, denominator_work);
        }

        if (denominator_scale != 1.0) {
            inv_denominator = refresh_inverse_to_min_level(context, std::move(inv_denominator), 3);
            inv_denominator = multiply_scalar(context, std::move(inv_denominator), denominator_scale);
        } else {
            inv_denominator = refresh_inverse_to_min_level(context, std::move(inv_denominator), 2);
        }
        FeatureMatEncrypted values_dropped = align_to_level(std::move(values), inv_denominator.level);
        ParUpperDiagonalGELU mult_layer(param, shape, head_shape, n_heads, inv_denominator.level);
        mult_layer.prepare_weight();
        return mult_layer.run(context, values_dropped, inv_denominator);
    };

    auto run_encrypted = [&](CkksContext& context, const Array<double, 2>& input, const SoftmaxProfile& profile,
                             int init_level) {
        FeatureMatEncrypted input_enc(&context, init_level);
        input_enc.par_diagonal_pack(input, n_heads, head_shape, false, false, default_scale);

        double mid = (profile.range_min + profile.range_max) / 2.0;
        ParUpperDiagonalAddPt add_mid(param, shape, head_shape, n_heads, input_enc.level, -mid);
        add_mid.prepare_weight();
        FeatureMatEncrypted centered = add_mid.run(context, input_enc);

        ParUpperDiagonalPolyActRNGamma gamma_layer(param, shape, head_shape, n_heads, centered.level,
                                                   make_scalar_gamma(1.0 / profile.exp_divisor));
        gamma_layer.prepare_weight();
        FeatureMatEncrypted exp_arg = gamma_layer.run(context, centered);

        uint32_t order = static_cast<uint32_t>(profile.coeffs.size() - 1);
        exp_arg =
            refresh_to_min_level(std::move(exp_arg), PolyBase::compute_stockmeyer_level_cost(static_cast<int>(order)));
        ParUpperDiagonalPoly exp_poly(param, shape, head_shape, n_heads, exp_arg.level,
                                      make_coeff_matrix(profile.coeffs), order);
        exp_poly.prepare_weight_stockmeyer();
        FeatureMatEncrypted exp_poly_out = exp_poly.run_stockmeyer(context, exp_arg);

        exp_poly_out = refresh_to_min_level(std::move(exp_poly_out), 2);
        ParUpperDiagonalMultipleSquare delta1_square(param, shape, head_shape, n_heads, exp_poly_out.level);
        delta1_square.prepare_weight();
        FeatureMatEncrypted values = delta1_square.run(context, exp_poly_out);

        FeatureMatEncrypted probabilities = run_normalize_encrypt(context, values, profile.initial_denominator_scale);
        uint32_t refinement_step = 0;
        for (uint32_t delta = profile.delta_2; delta > 1; delta >>= 1) {
            probabilities = refresh_to_min_level(std::move(probabilities), 3);
            ParUpperDiagonalGELU square_layer(param, shape, head_shape, n_heads, probabilities.level);
            square_layer.prepare_weight();
            FeatureMatEncrypted squared = square_layer.run(context, probabilities, probabilities);
            double denominator_scale = (refinement_step == 0) ? profile.first_refinement_denominator_scale :
                                                                profile.later_refinement_denominator_scale;
            probabilities = run_normalize_encrypt(context, squared, denominator_scale);
            refinement_step++;
        }
        return probabilities;
    };

    for (const auto& profile : profiles) {
        DYNAMIC_SECTION(profile.name) {
            uint32_t refinement_steps = 0;
            for (uint32_t delta = profile.delta_2; delta > 1; delta >>= 1) {
                refinement_steps++;
            }
            (void)refinement_steps;
            int init_level = 9;
            REQUIRE(init_level <= param.max_level());
            CkksContext context = CkksContext::create_random_context(param, EncryptorType::PUBLIC_KEY, true);
            context.gen_rotation_keys();

            Array<double, 2> input({n_prepad, total_cols});
            double width = profile.range_max - profile.range_min;
            for (uint32_t row = 0; row < n_prepad; row++) {
                for (uint32_t h = 0; h < n_heads; h++) {
                    for (uint32_t col = 0; col < n_prepad; col++) {
                        uint32_t mixed = (row * 17 + h * 23 + col * 7) % 101;
                        double u = static_cast<double>(mixed) / 100.0;
                        double value = profile.range_min + width * (0.1 + 0.8 * u);
                        input.set(row, h * n_prepad + col, value);
                    }
                }
            }

            Array<double, 2> expected = run_expected(input, profile);
            Array<double, 2> ground_truth = run_ground_truth(input);
            FeatureMatEncrypted encrypted = run_encrypted(context, input, profile, init_level);
            CAPTURE(encrypted.data[0].scale(), default_scale);
            // The final scale accumulates q[level]/default_scale ratios over many
            // rescales; the CI default chain primes are not exactly 2^log_default_scale,
            // so a tight 1e-10 bound is not meaningful. 2e-3 still detects real
            // scale-planning errors while tolerating modulus-ratio drift.
            REQUIRE(fabs(encrypted.data[0].scale() - default_scale) <= default_scale * 2.0e-3);

            Array<double, 2> output_mg = encrypted.par_diagonal_unpack(n_heads, head_shape, false, false);
            Array<double, 2> plain_output = expected.copy();
            Array<double, 2> ground_truth_output = ground_truth.copy();
            auto output_mg_1d = output_mg.to_array_1d();
            auto plain_output_1d = plain_output.to_array_1d();
            auto ground_truth_1d = ground_truth_output.to_array_1d();
            print_double_message(output_mg_1d.data(), "output_mg", 20);
            print_double_message(plain_output_1d.data(), "plain_output", 20);
            print_double_message(ground_truth_1d.data(), "ground_truth", 20);
            print_double_message(output_mg_1d.data() + output_mg_1d.size() - 20, "output_mg_last", 20);
            print_double_message(plain_output_1d.data() + plain_output_1d.size() - 20, "plain_output_last", 20);
            print_double_message(ground_truth_1d.data() + ground_truth_1d.size() - 20, "ground_truth_last", 20);

            auto compare_result = compare(plain_output, output_mg);
            auto ground_truth_compare_result = compare(ground_truth_output, output_mg);
            cout << profile.name << " softmax max_error=" << compare_result.max_error
                 << " max_abs=" << compare_result.max_abs << " rmse=" << compare_result.rmse
                 << " rms=" << compare_result.rms << endl;
            cout << profile.name << " softmax ground_truth max_error=" << ground_truth_compare_result.max_error
                 << " max_abs=" << ground_truth_compare_result.max_abs << " rmse=" << ground_truth_compare_result.rmse
                 << " rms=" << ground_truth_compare_result.rms << endl;
            CAPTURE(profile.name, compare_result.max_error, compare_result.max_abs, compare_result.rmse,
                    compare_result.rms);
            CAPTURE(profile.name, ground_truth_compare_result.max_error, ground_truth_compare_result.max_abs,
                    ground_truth_compare_result.rmse, ground_truth_compare_result.rms);
            REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
            REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
            REQUIRE(ground_truth_compare_result.max_error < 5.0e-2 * ground_truth_compare_result.max_abs);
            REQUIRE(ground_truth_compare_result.rmse < 1.0e-2 * ground_truth_compare_result.rms);
        }
    }
}

TEST_CASE("par_upper_diagonal_polylayernorm_ln3_cpp", "[fhe_layers][par_upper_diagonal_layernorm][polylayernorm]") {
    CkksParameter param = CkksParameter::create_parameter(15, RingType::ConjugateInvariant);
    const double default_scale = param.default_scale();
    const uint32_t plaintext_level = static_cast<uint32_t>(param.max_level());
    const uint32_t n_prepad = 65;
    const uint32_t n_heads = 3;
    const uint32_t m_prepad = 64;
    const uint32_t total_cols = n_heads * m_prepad;
    const uint32_t max_inverse_sqrt_iterations = 7;
    const int init_level = 9;
    REQUIRE(init_level <= param.max_level());

    Duo shape = {n_prepad, total_cols};
    Duo head_shape = {n_prepad, m_prepad};

    const double eps = 1.0e-5;
    const double min_var = 0.75;
    const double max_var = 2500.0;
    const double w_buffer = 1.05;
    const double input_scale = 1.0;
    const double max_denominator = (max_var * w_buffer + eps) * input_scale * input_scale;
    const double epsilon = (min_var + eps) / max_denominator;
    const double inv_var = 1.0 / max_denominator;
    const double inv_std = sqrt(inv_var);
    const double c0 = 6.19067182;
    const double c1 = -16.15885111;
    const double c2 = 11.52830778;

    auto make_gamma = [&]() {
        Array<double, 1> gamma({total_cols});
        for (uint32_t col = 0; col < total_cols; col++) {
            gamma.set(col, 0.8 + 0.002 * static_cast<double>(col % 17) - 0.001 * static_cast<double>(col / m_prepad));
        }
        return gamma;
    };

    auto make_beta = [&]() {
        Array<double, 1> beta({total_cols});
        for (uint32_t col = 0; col < total_cols; col++) {
            beta.set(col, 0.03 * sin(static_cast<double>(col + 1) * 0.1));
        }
        return beta;
    };

    auto make_input = [&]() {
        Array<double, 2> input({n_prepad, total_cols});
        for (uint32_t row = 0; row < n_prepad; row++) {
            double amplitude = 54.0 + 1.4 * static_cast<double>(row % 7);
            for (uint32_t col = 0; col < total_cols; col++) {
                double centered_col = static_cast<double>(col % m_prepad) - (static_cast<double>(m_prepad) - 1.0) / 2.0;
                double wave = sin(static_cast<double>((row + 1) * (col + 3)) * 0.03125) +
                              0.35 * cos(static_cast<double>((row + 5) * (col + 11)) * 0.017);
                double trend = 0.015 * centered_col;
                double head_offset = 0.2 * (static_cast<double>(col / m_prepad) - 1.0);
                input.set(row, col, amplitude * wave + trend + head_offset);
            }
        }
        return input;
    };

    auto run_expected = [&](const Array<double, 2>& input, const Array<double, 1>& gamma,
                            const Array<double, 1>& beta) {
        ParUpperDiagonalLNStats stats(param, shape, head_shape, n_heads, plaintext_level, eps, inv_var);
        Array<double, 2> a = stats.run_plaintext(input);

        ParUpperDiagonalLNXCentered xcentered(param, shape, head_shape, n_heads, plaintext_level);
        Array<double, 2> centered = xcentered.run_plaintext(input);

        ParUpperDiagonalLNMinimaxInit minimax(param, shape, head_shape, n_heads, plaintext_level, c0, c1, c2);
        Array<double, 2> y = minimax.run_plaintext(a);

        for (uint32_t iter = 0; iter < max_inverse_sqrt_iterations; iter++) {
            ParUpperDiagonalLNGoldschmidt gold(param, shape, head_shape, n_heads, plaintext_level);
            y = gold.run_plaintext(y, a);
        }

        ParUpperDiagonalLNAffine affine(param, shape, head_shape, n_heads, plaintext_level, inv_std, gamma.copy(),
                                        beta.copy());
        return affine.run_plaintext(centered, y);
    };

    auto run_ground_truth = [&](const Array<double, 2>& input, const Array<double, 1>& gamma,
                                const Array<double, 1>& beta) {
        Array<double, 2> result({n_prepad, total_cols});
        double min_observed_var = numeric_limits<double>::max();
        double max_observed_var = 0.0;
        for (uint32_t row = 0; row < n_prepad; row++) {
            double sum = 0.0;
            for (uint32_t col = 0; col < total_cols; col++) {
                sum += input.get(row, col);
            }
            double mean = sum / static_cast<double>(total_cols);

            double sum_sq = 0.0;
            for (uint32_t col = 0; col < total_cols; col++) {
                double centered = input.get(row, col) - mean;
                sum_sq += centered * centered;
            }
            double variance = sum_sq / static_cast<double>(total_cols);
            min_observed_var = std::min(min_observed_var, variance);
            max_observed_var = std::max(max_observed_var, variance);
            double denominator = sqrt(variance + eps);

            for (uint32_t col = 0; col < total_cols; col++) {
                double normalized = (input.get(row, col) - mean) / denominator;
                result.set(row, col, normalized * gamma.get(col) + beta.get(col));
            }
        }
        CAPTURE(epsilon, min_observed_var, max_observed_var);
        REQUIRE(min_observed_var >= min_var);
        REQUIRE(max_observed_var <= max_var);
        return result;
    };

    auto cts_level = [](const vector<CkksCiphertext>& cts) {
        REQUIRE_FALSE(cts.empty());
        int level = cts[0].level();
        for (const auto& ct : cts) {
            REQUIRE(ct.level() == level);
        }
        return static_cast<uint32_t>(level);
    };

    auto require_default_scale = [&](const vector<CkksCiphertext>& cts, const string& label) {
        REQUIRE_FALSE(cts.empty());
        for (const auto& ct : cts) {
            INFO(label);
            CAPTURE(ct.scale(), default_scale);
            REQUIRE(fabs(ct.scale() - default_scale) <= default_scale * 1.0e-10);
        }
    };

    auto make_feature = [&](CkksContext& context, vector<CkksCiphertext>&& cts) {
        int level = cts.empty() ? 0 : cts[0].level();
        FeatureMatEncrypted feature(&context, level);
        feature.shape = shape;
        feature.head_shape = head_shape;
        feature.matmul_block_size = next_pow2_u32(m_prepad);
        feature.data = std::move(cts);
        return feature;
    };

    auto refresh_cts_to_min_level = [&](CkksContext& context, vector<CkksCiphertext>&& cts, uint32_t min_level) {
        FeatureMatEncrypted feature = make_feature(context, std::move(cts));
        if (feature.level < static_cast<int>(min_level)) {
            if (feature.level > 0) {
                feature = feature.drop_level(feature.level);
            }
            feature = feature.refresh_ciphertext();
        }
        REQUIRE(feature.level >= static_cast<int>(min_level));
        return std::move(feature.data);
    };

    auto align_cts_to_level = [&](CkksContext& context, vector<CkksCiphertext>&& cts, uint32_t target_level) {
        FeatureMatEncrypted feature = make_feature(context, std::move(cts));
        if (feature.level < static_cast<int>(target_level)) {
            if (feature.level > 0) {
                feature = feature.drop_level(feature.level);
            }
            feature = feature.refresh_ciphertext();
        }
        REQUIRE(feature.level >= static_cast<int>(target_level));
        if (feature.level > static_cast<int>(target_level)) {
            feature = feature.drop_level(feature.level - static_cast<int>(target_level));
        }
        REQUIRE(feature.level == static_cast<int>(target_level));
        return std::move(feature.data);
    };

    auto run_encrypted = [&](CkksContext& context, const Array<double, 2>& input, const Array<double, 1>& gamma,
                             const Array<double, 1>& beta) {
        FeatureMatEncrypted input_enc(&context, init_level);
        input_enc.par_diagonal_pack(input, n_heads, head_shape, false, false, default_scale);

        ParUpperDiagonalLNStats stats(param, shape, head_shape, n_heads, input_enc.level, eps, inv_var);
        stats.prepare_weight();
        vector<CkksCiphertext> a_cts = stats.run(context, input_enc);
        require_default_scale(a_cts, "ParUpperDiagonalLNStats");

        ParUpperDiagonalLNXCentered xcentered(param, shape, head_shape, n_heads, input_enc.level);
        xcentered.prepare_weight();
        vector<CkksCiphertext> x_centered_cts = xcentered.run(context, input_enc);
        require_default_scale(x_centered_cts, "ParUpperDiagonalLNXCentered");

        uint32_t a_level = cts_level(a_cts);
        ParUpperDiagonalLNMinimaxInit minimax(param, shape, head_shape, n_heads, a_level, c0, c1, c2);
        minimax.prepare_weight();
        vector<CkksCiphertext> y_cts = minimax.run(context, a_cts);

        for (uint32_t iter = 0; iter < max_inverse_sqrt_iterations; iter++) {
            y_cts = refresh_cts_to_min_level(context, std::move(y_cts), 3);
            uint32_t y_level = cts_level(y_cts);
            a_cts = refresh_cts_to_min_level(context, std::move(a_cts), y_level);
            ParUpperDiagonalLNGoldschmidt gold(param, shape, head_shape, n_heads, y_level);
            gold.prepare_weight();
            y_cts = gold.run(context, y_cts, a_cts);
        }

        const uint32_t affine_y_level = 3;
        y_cts = align_cts_to_level(context, std::move(y_cts), affine_y_level);
        x_centered_cts = refresh_cts_to_min_level(context, std::move(x_centered_cts), affine_y_level - 1);

        ParUpperDiagonalLNAffine affine(param, shape, head_shape, n_heads, affine_y_level, inv_std, gamma.copy(),
                                        beta.copy());
        affine.prepare_weight();
        return affine.run(context, x_centered_cts, y_cts);
    };

    CkksContext context = CkksContext::create_random_context(param, EncryptorType::PUBLIC_KEY, true);
    context.gen_rotation_keys();

    Array<double, 2> input = make_input();
    Array<double, 1> gamma = make_gamma();
    Array<double, 1> beta = make_beta();
    Array<double, 2> expected = run_expected(input, gamma, beta);
    Array<double, 2> ground_truth = run_ground_truth(input, gamma, beta);

    FeatureMatEncrypted encrypted = run_encrypted(context, input, gamma, beta);
    CAPTURE(encrypted.data[0].scale(), default_scale, encrypted.level);
    REQUIRE(fabs(encrypted.data[0].scale() - default_scale) <= default_scale * 1.0e-10);

    Array<double, 2> output_mg = encrypted.par_diagonal_unpack(n_heads, head_shape, false, false);
    Array<double, 2> plain_output = expected.copy();
    Array<double, 2> ground_truth_output = ground_truth.copy();
    auto output_mg_1d = output_mg.to_array_1d();
    auto plain_output_1d = plain_output.to_array_1d();
    auto ground_truth_1d = ground_truth_output.to_array_1d();
    print_double_message(output_mg_1d.data(), "output_mg", 20);
    print_double_message(plain_output_1d.data(), "plain_output", 20);
    print_double_message(ground_truth_1d.data(), "ground_truth", 20);
    print_double_message(output_mg_1d.data() + output_mg_1d.size() - 20, "output_mg_last", 20);
    print_double_message(plain_output_1d.data() + plain_output_1d.size() - 20, "plain_output_last", 20);
    print_double_message(ground_truth_1d.data() + ground_truth_1d.size() - 20, "ground_truth_last", 20);

    auto compare_result = compare(plain_output, output_mg);
    auto ground_truth_compare_result = compare(ground_truth_output, output_mg);
    cout << "ln3 use_aSOR=false max_error=" << compare_result.max_error << " max_abs=" << compare_result.max_abs
         << " rmse=" << compare_result.rmse << " rms=" << compare_result.rms << endl;
    cout << "ln3 use_aSOR=false ground_truth max_error=" << ground_truth_compare_result.max_error
         << " max_abs=" << ground_truth_compare_result.max_abs << " rmse=" << ground_truth_compare_result.rmse
         << " rms=" << ground_truth_compare_result.rms << endl;

    CAPTURE(compare_result.max_error, compare_result.max_abs, compare_result.rmse, compare_result.rms);
    CAPTURE(ground_truth_compare_result.max_error, ground_truth_compare_result.max_abs,
            ground_truth_compare_result.rmse, ground_truth_compare_result.rms);
    REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
    REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
    REQUIRE(ground_truth_compare_result.max_error < 5.0e-2 * ground_truth_compare_result.max_abs);
    REQUIRE(ground_truth_compare_result.rmse < 1.0e-2 * ground_truth_compare_result.rms);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "sq", "", HeteroProcessors) {
    int init_level = 2;
    vector<Duo> input_shapes = {{16, 16}, {32, 32}, {64, 64}};

    FOR_EACH_SECTION(const Duo& input_shape, input_shapes, "input_shape=" + str(input_shape)) {
        Array<double, 3> input_array = gen_random_array<3>({1, input_shape[0], input_shape[1]}, 1.0);

        Feature2DEncrypted input_feature(&this->context, init_level);
        input_feature.pack_multiple_channel(input_array, this->param.default_scale());

        Feature2DEncrypted output_feature(&this->context, init_level);

        for (int i = 0; i < 1; i++) {
            output_feature.data.emplace_back(this->context.parameter(), init_level - 1);
            output_feature.data.back().set_scale(this->param.default_scale());
        }

        fs::path project_path = base_path /
                                ("CKKS_square_" + to_string(input_shape[0]) + "_" + to_string(input_shape[1])) /
                                ("level_" + to_string(init_level)) / "server";
        cout << "project_path=" << project_path << endl;
        auto arg_names = read_arg_names(project_path);
        vector<CxxVectorArgument> cxx_args = {
            {arg_names[0], &input_feature.data},
            {arg_names[1], &output_feature.data},
        };
        this->run(project_path, cxx_args);

        output_feature.skip = {1, 1};
        output_feature.n_channel = 1;
        output_feature.n_channel_per_ct = this->n_slot / prod(input_shape);
        output_feature.shape = input_shape;
        auto output_mg = output_feature.unpack_multiple_channel();

        SquareLayer square_layer(this->param);
        auto plain_output = square_layer.run_plaintext(input_array);

        // print_double_message(output_mg.to_array_1d().data(), "output_mg", 10);
        // print_double_message(plain_output.to_array_1d().data(), "plain_output", 10);

        auto compare_result = compare(plain_output, output_mg);
        REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
        REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "conv2d_packed", "", HeteroProcessors) {
    Duo skip = {1, 1};
    int init_level = 2;

    auto run_conv2d_packed_test = [&](uint32_t n_in_channel, uint32_t n_out_channel, Duo stride, Duo input_shape,
                                      Duo kernel_shape) {
        uint32_t n_channel_per_ct = div_ceil(this->n_slot, prod(input_shape));

        Array<double, 4> conv0_weight =
            gen_random_array<4>({n_out_channel, n_in_channel, kernel_shape[0], kernel_shape[1]}, 0.1);
        Array<double, 1> conv0_bias = gen_random_array<1>({n_out_channel}, 0.1);
        Array<double, 3> input_array = gen_random_array<3>({n_in_channel, input_shape[0], input_shape[1]}, 1.0);

        Feature2DEncrypted input_feature(&this->context, init_level);
        input_feature.pack_multiple_channel(input_array, this->param.default_scale());

        Conv2DPackedLayer conv0_layer(this->context.parameter(), input_shape, move(conv0_weight), move(conv0_bias),
                                      stride, skip, n_channel_per_ct, init_level);
        conv0_layer.prepare_weight();

        Feature2DEncrypted output_feature(&this->context, init_level - 1);
        output_feature.shape = input_shape / stride;
        output_feature.skip = skip * stride;
        output_feature.n_channel = n_out_channel;
        output_feature.n_channel_per_ct = n_channel_per_ct;
        for (int i = 0; i < div_ceil(n_out_channel, n_channel_per_ct); i++) {
            output_feature.data.emplace_back(this->context.parameter(), init_level - 1);
            output_feature.data.back().set_scale(this->param.default_scale());
        }

        fs::path project_path = base_path / "CKKS_conv2d" /
                                ("stride_" + to_string(stride[0]) + "_" + to_string(stride[1])) /
                                ("kernel_shape_" + to_string(kernel_shape[0]) + "_" + to_string(kernel_shape[1])) /
                                ("cin_" + to_string(n_in_channel) + "_cout_" + to_string(n_out_channel)) /
                                ("input_shape_" + to_string(input_shape[0]) + "_" + to_string(input_shape[1])) /
                                ("level_" + to_string(init_level)) / "server";

        auto arg_names = read_arg_names(project_path);
        vector<CxxVectorArgument> cxx_args;
        for (const auto& name : arg_names) {
            if (name.rfind("input", 0) == 0)
                cxx_args.push_back({name, &input_feature.data});
            else if (name.rfind("convw_", 0) == 0)
                cxx_args.push_back({name, &conv0_layer.weight_pt_});
            else if (name.rfind("convb_", 0) == 0)
                cxx_args.push_back({name, &conv0_layer.bias_pt_});
            else if (name.rfind("output", 0) == 0)
                cxx_args.push_back({name, &output_feature.data});
        }

        this->run(project_path, cxx_args);

        auto output_mg = output_feature.unpack_multiple_channel();
        auto plain_output = conv0_layer.run_plaintext(input_array);

        print_double_message(output_mg.to_array_1d().data(), "output_mg", 10);
        print_double_message(plain_output.to_array_1d().data(), "plain_output", 10);

        auto compare_result = compare(plain_output, output_mg);
        REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
        REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
    };

    SECTION("stride=(1,1)") {
        Duo stride = {1, 1};

        SECTION("single_channel") {
            uint32_t n_in_channel = 1;
            uint32_t n_out_channel = 1;
            vector<Duo> input_shapes = {/*{4, 4}, */ {8, 8}, {16, 16}, {32, 32}, {64, 64}};
            vector<Duo> kernel_shapes = {{1, 1}, {3, 3}, {5, 5}};

            FOR_EACH_SECTION(const Duo& input_shape, input_shapes, "input_shape=" + str(input_shape)) {
                FOR_EACH_SECTION(const Duo& kernel_shape, kernel_shapes, "kernel_shape=" + str(kernel_shape)) {
                    run_conv2d_packed_test(n_in_channel, n_out_channel, stride, input_shape, kernel_shape);
                }
            }
        }

        SECTION("multi_channel") {
            Duo input_shape = {32, 32};
            Duo kernel_shape = {3, 3};
            vector<uint32_t> nc_ins = {1, 3, 4, 16, 17};
            vector<uint32_t> nc_outs = {1, 3, 4, 32, 33};

            FOR_EACH_SECTION(uint32_t n_in_channel, nc_ins, "n_in_channel=" + to_string(n_in_channel)) {
                FOR_EACH_SECTION(uint32_t n_out_channel, nc_outs, "n_out_channel=" + to_string(n_out_channel)) {
                    run_conv2d_packed_test(n_in_channel, n_out_channel, stride, input_shape, kernel_shape);
                }
            }
        }
    }

    SECTION("stride=(2,2)") {
        Duo stride = {2, 2};

        SECTION("single_channel") {
            uint32_t n_in_channel = 1;
            uint32_t n_out_channel = 1;
            vector<Duo> input_shapes = {{32, 32}, {64, 64}};
            vector<Duo> kernel_shapes = {{1, 1}, {3, 3}, {5, 5}};

            FOR_EACH_SECTION(const Duo& input_shape, input_shapes, "input_shape=" + str(input_shape)) {
                FOR_EACH_SECTION(const Duo& kernel_shape, kernel_shapes, "kernel_shape=" + str(kernel_shape)) {
                    run_conv2d_packed_test(n_in_channel, n_out_channel, stride, input_shape, kernel_shape);
                }
            }
        }

        SECTION("multi_channel") {
            Duo input_shape = {32, 32};
            Duo kernel_shape = {3, 3};
            vector<uint32_t> nc_ins = {1, 3, 4, 16, 17};
            vector<uint32_t> nc_outs = {1, 3, 4, 32, 33};

            FOR_EACH_SECTION(uint32_t n_in_channel, nc_ins, "n_in_channel=" + to_string(n_in_channel)) {
                FOR_EACH_SECTION(uint32_t n_out_channel, nc_outs, "n_out_channel=" + to_string(n_out_channel)) {
                    run_conv2d_packed_test(n_in_channel, n_out_channel, stride, input_shape, kernel_shape);
                }
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "conv2d_depthwise", "", HeteroProcessors) {
    Duo skip = {1, 1};
    int init_level = 5;

    vector<uint32_t> channels = {4, 8, 32};
    vector<Duo> strides = {{1, 1}, {2, 2}};

    FOR_EACH_SECTION(const Duo& stride, strides, "stride=" + str(stride)) {
        vector<Duo> input_shapes = {{16, 16}, {32, 32}};
        vector<Duo> kernel_shapes = {{1, 1}, {3, 3}, {5, 5}};

        FOR_EACH_SECTION(uint32_t n_channel, channels, "ch=" + to_string(n_channel)) {
            FOR_EACH_SECTION(const Duo& input_shape, input_shapes, "input=" + str(input_shape)) {
                uint32_t n_channel_per_ct = div_ceil(this->n_slot, prod(input_shape));
                FOR_EACH_SECTION(const Duo& kernel_shape, kernel_shapes, "kernel=" + str(kernel_shape)) {
                    Array<double, 4> conv0_weight =
                        gen_random_array<4>({n_channel, n_channel, kernel_shape[0], kernel_shape[1]}, 0.1);
                    Array<double, 1> conv0_bias = gen_random_array<1>({n_channel}, 0);
                    Array<double, 3> input = gen_random_array<3>({n_channel, input_shape[0], input_shape[1]}, 1);

                    Conv2DPackedDepthwiseLayer conv(this->context.parameter(), input_shape, move(conv0_weight),
                                                    move(conv0_bias), stride, skip, n_channel_per_ct, init_level);
                    conv.prepare_weight();

                    Feature2DEncrypted f2d(&this->context, init_level);
                    f2d.pack_multiple_channel(input, this->param.default_scale());

                    Feature2DEncrypted output_feature(&this->context, init_level - 1);
                    output_feature.shape = input_shape / stride;
                    output_feature.skip = skip * stride;
                    output_feature.n_channel = n_channel;
                    output_feature.n_channel_per_ct = n_channel_per_ct;
                    for (int i = 0; i < div_ceil(n_channel, n_channel_per_ct); i++) {
                        output_feature.data.emplace_back(this->context.parameter(), init_level - 1);
                        output_feature.data.back().set_scale(this->param.default_scale());
                    }

                    fs::path project_path =
                        base_path / "CKKS_dw_conv2d" / ("stride_" + to_string(stride[0]) + "_" + to_string(stride[1])) /
                        ("kernel_shape_" + to_string(kernel_shape[0]) + "_" + to_string(kernel_shape[1])) /
                        ("cin_" + to_string(n_channel) + "_cout_" + to_string(n_channel)) /
                        ("input_shape_" + to_string(input_shape[0]) + "_" + to_string(input_shape[1])) /
                        ("level_" + to_string(init_level)) / "server";

                    auto arg_names = read_arg_names(project_path);
                    vector<CxxVectorArgument> cxx_args;
                    for (const auto& name : arg_names) {
                        if (name.rfind("input", 0) == 0)
                            cxx_args.push_back({name, &f2d.data});
                        else if (name.rfind("convw_", 0) == 0)
                            cxx_args.push_back({name, &conv.weight_pt_});
                        else if (name.rfind("convb_", 0) == 0)
                            cxx_args.push_back({name, &conv.bias_pt_});
                        else if (name.rfind("output", 0) == 0)
                            cxx_args.push_back({name, &output_feature.data});
                    }

                    this->run(project_path, cxx_args);

                    Array<double, 3> output_mg = output_feature.unpack_multiple_channel();
                    Array<double, 3> plain_output = conv.run_plaintext(input);

                    print_double_message(output_mg.to_array_1d().data(), "output_mg", 10);
                    print_double_message(plain_output.to_array_1d().data(), "plain_output", 10);

                    auto compare_result = compare(plain_output, output_mg);
                    REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
                    REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
                }
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "mux_conv2d_packed", "", HeteroProcessors) {
    Duo skip = {1, 1};
    int init_level = 5;

    auto run_mux_conv2d_test = [&](uint32_t n_in_channel, uint32_t n_out_channel, Duo stride, Duo input_shape,
                                   Duo kernel_shape) {
        uint32_t n_channel_per_ct = div_ceil(this->n_slot, prod(input_shape));
        int output_level = (stride[0] == 1) ? init_level - 1 : init_level - 2;

        Array<double, 4> conv0_weight =
            gen_random_array<4>({n_out_channel, n_in_channel, kernel_shape[0], kernel_shape[1]}, 0.1);
        Array<double, 1> conv0_bias = gen_random_array<1>({n_out_channel}, 0.1);
        Array<double, 3> input_array = gen_random_array<3>({n_in_channel, input_shape[0], input_shape[1]}, 1.0);

        MultiplexedConv2DPackedLayer conv_layer(this->context.parameter(), input_shape, move(conv0_weight),
                                                move(conv0_bias), stride, skip, n_channel_per_ct, init_level, 1.0);
        conv_layer.prepare_weight_for_post_skip_rotation();

        Feature2DEncrypted input_feature(&this->context, init_level, skip);
        input_feature.pack_multiplexed(input_array, this->context.parameter().default_scale());

        Feature2DEncrypted output_feature(&this->context, output_level);
        output_feature.shape = input_shape / stride;
        output_feature.skip = skip * stride;
        output_feature.n_channel = n_out_channel;
        output_feature.n_channel_per_ct = (n_channel_per_ct * prod(stride));
        for (int i = 0; i < div_ceil(n_out_channel, (n_channel_per_ct * prod(stride))); i++) {
            output_feature.data.emplace_back(this->context.parameter(), output_level);
            output_feature.data.back().set_scale(this->param.default_scale());
        }

        fs::path project_path = base_path / "CKKS_multiplexed_conv2d" /
                                ("stride_" + to_string(stride[0]) + "_" + to_string(stride[1])) /
                                ("kernel_shape_" + to_string(kernel_shape[0]) + "_" + to_string(kernel_shape[1])) /
                                ("cin_" + to_string(n_in_channel) + "_cout_" + to_string(n_out_channel)) /
                                ("input_shape_" + to_string(input_shape[0]) + "_" + to_string(input_shape[1])) /
                                ("level_" + to_string(init_level)) / "server";

        auto arg_names = read_arg_names(project_path);
        vector<CxxVectorArgument> cxx_args;
        for (const auto& name : arg_names) {
            if (name.rfind("input", 0) == 0)
                cxx_args.push_back({name, &input_feature.data});
            else if (name.rfind("convm_", 0) == 0)
                cxx_args.push_back({name, &conv_layer.mask_pt});
            else if (name.rfind("convw_", 0) == 0)
                cxx_args.push_back({name, &conv_layer.weight_pt});
            else if (name.rfind("convb_", 0) == 0)
                cxx_args.push_back({name, &conv_layer.bias_pt});
            else if (name.rfind("output", 0) == 0)
                cxx_args.push_back({name, &output_feature.data});
        }

        this->run(project_path, cxx_args);

        auto y_mg = output_feature.unpack_multiplexed();
        auto y_expected = conv_layer.run_plaintext(input_array);

        auto compare_result = compare(y_expected, y_mg);
        REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
        REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
    };

    SECTION("varied_stride") {
        vector<Duo> strides = {{1, 1}, {2, 2}};
        vector<uint32_t> nc_ins = {4, 8, 32};
        vector<uint32_t> nc_outs = {4, 8, 32};

        FOR_EACH_SECTION(const Duo& stride, strides, "stride=" + str(stride)) {
            FOR_EACH_SECTION(uint32_t n_in_channel, nc_ins, "n_in_channel=" + to_string(n_in_channel)) {
                FOR_EACH_SECTION_IF(uint32_t n_out_channel, nc_outs, n_in_channel == n_out_channel,
                                    "n_out_channel=" + to_string(n_out_channel)) {
                    Duo input_shape = {32, 32};
                    Duo kernel_shape = {3, 3};
                    run_mux_conv2d_test(n_in_channel, n_out_channel, stride, input_shape, kernel_shape);
                }
            }
        }
    }

    SECTION("varied_input_shape") {
        uint32_t n_in_channel = 32;
        uint32_t n_out_channel = 32;
        Duo stride = {1, 1};
        Duo kernel_shape = {3, 3};
        // input_shape=2 removed: 2x2 input is too small for multiplexed packing
        vector<Duo> input_shapes = {{4, 4}, {8, 8}, {16, 16}, {32, 32}, {64, 64}};

        FOR_EACH_SECTION(const Duo& input_shape, input_shapes, "input_shape=" + str(input_shape)) {
            run_mux_conv2d_test(n_in_channel, n_out_channel, stride, input_shape, kernel_shape);
        }
    }

    SECTION("varied_kernel_shape") {
        uint32_t n_in_channel = 32;
        uint32_t n_out_channel = 32;
        Duo input_shape = {32, 32};
        Duo stride = {1, 1};
        vector<Duo> kernel_shapes = {{1, 1}, {3, 3}, {5, 5}};

        FOR_EACH_SECTION(const Duo& kernel_shape, kernel_shapes, "kernel_shape=" + str(kernel_shape)) {
            run_mux_conv2d_test(n_in_channel, n_out_channel, stride, input_shape, kernel_shape);
        }
    }

    SECTION("varied_channels") {
        Duo input_shape = {32, 32};
        Duo kernel_shape = {3, 3};
        Duo stride = {1, 1};
        vector<uint32_t> nc_ins = {1, 3, 4, 16, 17};
        vector<uint32_t> nc_outs = {1, 3, 4, 32, 33};

        FOR_EACH_SECTION(uint32_t n_in_channel, nc_ins, "n_in_channel=" + to_string(n_in_channel)) {
            FOR_EACH_SECTION(uint32_t n_out_channel, nc_outs, "n_out_channel=" + to_string(n_out_channel)) {
                run_mux_conv2d_test(n_in_channel, n_out_channel, stride, input_shape, kernel_shape);
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "mux_dw_s2_64x64_k3", "", HeteroProcessors) {
    Duo input_shape = {64, 64};
    Duo kernel_shape = {3, 3};
    Duo stride = {2, 2};
    Duo skip = {1, 1};
    uint32_t n_channel_per_ct = div_ceil(this->n_slot, prod(input_shape));
    int init_level = 5;

    vector<uint32_t> nc_ins = {4, 8, 32};
    vector<uint32_t> nc_outs = {4, 8, 32};

    FOR_EACH_SECTION(uint32_t n_in_channel, nc_ins, "n_in_channel=" + to_string(n_in_channel)) {
        FOR_EACH_SECTION_IF(uint32_t n_out_channel, nc_outs, n_in_channel == n_out_channel,
                            "n_out_channel=" + to_string(n_out_channel)) {
            Array<double, 4> conv0_weight =
                gen_random_array<4>({n_out_channel, n_in_channel, kernel_shape[0], kernel_shape[1]}, 0.1);
            Array<double, 1> conv0_bias = gen_random_array<1>({n_out_channel}, 0.1);
            Array<double, 3> input_array = gen_random_array<3>({n_in_channel, input_shape[0], input_shape[1]}, 1.0);

            MultiplexedConv2DPackedLayerDepthwise dw_conv_layer(this->context.parameter(), input_shape,
                                                                move(conv0_weight), move(conv0_bias), stride, skip,
                                                                n_channel_per_ct, init_level, 1.0);
            dw_conv_layer.prepare_weight();

            Feature2DEncrypted input_feature(&this->context, init_level, skip);
            input_feature.pack_multiplexed(input_array, this->context.parameter().default_scale());

            Feature2DEncrypted output_feature(&this->context, init_level - 2);
            output_feature.shape = input_shape / stride;
            output_feature.skip = skip * stride;
            output_feature.n_channel = n_out_channel;
            output_feature.n_channel_per_ct = (n_channel_per_ct * prod(stride));
            for (int i = 0; i < div_ceil(n_out_channel, (n_channel_per_ct * prod(stride))); i++) {
                output_feature.data.emplace_back(this->context.parameter(), init_level - 2);
                output_feature.data.back().set_scale(this->param.default_scale());
            }

            fs::path project_path = base_path / "CKKS_multiplexed_dw_conv2d" /
                                    ("stride_" + to_string(stride[0]) + "_" + to_string(stride[1])) /
                                    "kernel_shape_3_3" /
                                    ("cin_" + to_string(n_in_channel) + "_cout_" + to_string(n_out_channel)) /
                                    "input_shape_64_64" / ("level_" + to_string(init_level)) / "server";

            auto arg_names = read_arg_names(project_path);
            vector<CxxVectorArgument> cxx_args;
            for (const auto& name : arg_names) {
                if (name.rfind("input", 0) == 0)
                    cxx_args.push_back({name, &input_feature.data});
                else if (name.rfind("convm_", 0) == 0)
                    cxx_args.push_back({name, &dw_conv_layer.mask_pt});
                else if (name.rfind("convw_", 0) == 0)
                    cxx_args.push_back({name, &dw_conv_layer.weight_pt});
                else if (name.rfind("convb_", 0) == 0)
                    cxx_args.push_back({name, &dw_conv_layer.bias_pt});
                else if (name.rfind("output", 0) == 0)
                    cxx_args.push_back({name, &output_feature.data});
            }

            this->run(project_path, cxx_args);

            auto y_mg = output_feature.unpack_multiplexed();
            auto y_expected = dw_conv_layer.run_plaintext(input_array);

            auto compare_result = compare(y_expected, y_mg);
            REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
            REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
        }
    }
}
TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "inv_mux_conv", "", HeteroProcessors) {
    Duo skip = {1, 1};
    int init_level = 1;

    vector<Duo> kernel_shapes = {{1, 1}, {3, 3}, {5, 5}};
    vector<Duo> strides_list = {{1, 1}, {2, 2}};
    vector<Duo> block_shapes = {{64, 64}, {64, 128}};
    vector<uint32_t> multipliers = {2, 4};
    vector<uint32_t> nc_ins = {2, 3, 5};
    vector<uint32_t> nc_outs = {3, 4, 15};

    for (auto& kernel_shape : kernel_shapes) {
        for (auto& stride : strides_list) {
            for (auto& block_shape : block_shapes) {
                for (auto mult : multipliers) {
                    Duo input_shape = block_shape * mult;
                    Duo stride_next = input_shape / (block_shape * stride);

                    Array<int, 1> padding({2});
                    padding.set(0, -1);
                    padding.set(1, -1);

                    string config_name = "ks_" + to_string(kernel_shape[0]) + "x" + to_string(kernel_shape[1]) +
                                         "_st_" + to_string(stride[0]) + "x" + to_string(stride[1]) + "_bs_" +
                                         to_string(block_shape[0]) + "x" + to_string(block_shape[1]) + "_is_" +
                                         to_string(input_shape[0]) + "x" + to_string(input_shape[1]);

                    SECTION(config_name) {
                        for (size_t idx = 0; idx < nc_ins.size(); idx++) {
                            uint32_t n_in_channel = nc_ins[idx];
                            uint32_t n_out_channel = nc_outs[idx];
                            SECTION("cin=" + to_string(n_in_channel) + "_cout=" + to_string(n_out_channel)) {
                                Array<double, 4> conv0_weight = gen_random_array<4>(
                                    {n_out_channel, n_in_channel, kernel_shape[0], kernel_shape[1]}, 0.1);
                                Array<double, 1> conv0_bias = gen_random_array<1>({n_out_channel}, 0.1);
                                Array<double, 3> input_array =
                                    gen_random_array<3>({n_in_channel, input_shape[0], input_shape[1]}, 1.0);

                                InverseMultiplexedConv2DLayer conv_layer(this->context.parameter(), input_shape,
                                                                         move(conv0_weight), move(conv0_bias), padding,
                                                                         stride, block_shape, init_level, 1.0);
                                conv_layer.prepare_weight();

                                Feature2DEncrypted input_feature(&this->context, init_level, skip);
                                Duo total_stride = stride * stride_next;
                                input_feature.pack_interleaved(input_array, block_shape, total_stride,
                                                               this->context.parameter().default_scale());

                                Duo output_shape = input_shape / stride;
                                uint32_t output_total = prod(output_shape);
                                uint32_t output_n_channel_per_ct;
                                if (2 * output_total < this->N) {
                                    output_n_channel_per_ct = this->N / (2 * output_total);
                                } else {
                                    output_n_channel_per_ct = 1;
                                }

                                Feature2DEncrypted output_feature(&this->context, init_level - 1);
                                output_feature.shape = output_shape;
                                output_feature.skip = {1, 1};
                                output_feature.n_channel = n_out_channel;
                                output_feature.n_channel_per_ct = output_n_channel_per_ct;
                                int n_out_cts = div_ceil(n_out_channel, output_n_channel_per_ct) * prod(stride_next);
                                for (int i = 0; i < n_out_cts; i++) {
                                    output_feature.data.emplace_back(this->context.parameter(), init_level - 1);
                                    output_feature.data.back().set_scale(this->param.default_scale());
                                }

                                vector<CxxVectorArgument> cxx_args;
                                fs::path project_path =
                                    base_path / "CKKS_inverse_multiplexed_conv2d" /
                                    ("stride_" + to_string(stride[0]) + "_" + to_string(stride[1])) /
                                    ("kernel_shape_" + to_string(kernel_shape[0]) + "_" + to_string(kernel_shape[1])) /
                                    ("cin_" + to_string(n_in_channel) + "_cout_" + to_string(n_out_channel)) /
                                    ("input_shape_" + to_string(input_shape[0]) + "_" + to_string(input_shape[1])) /
                                    ("level_" + to_string(init_level)) / "server";

                                auto arg_names = read_arg_names(project_path);
                                for (const auto& name : arg_names) {
                                    if (name.rfind("input", 0) == 0)
                                        cxx_args.push_back({name, &input_feature.data});
                                    else if (name.rfind("convw_", 0) == 0)
                                        cxx_args.push_back({name, &conv_layer.weight_pt});
                                    else if (name.rfind("convb_", 0) == 0)
                                        cxx_args.push_back({name, &conv_layer.bias_pt});
                                    else if (name.rfind("output", 0) == 0)
                                        cxx_args.push_back({name, &output_feature.data});
                                }

                                this->run(project_path, cxx_args);

                                Array<double, 3> y_mg;
                                if (output_shape[0] > block_shape[0] || output_shape[1] > block_shape[1]) {
                                    y_mg = output_feature.unpack_interleaved(block_shape, stride_next);
                                } else {
                                    y_mg = output_feature.unpack_multiplexed();
                                }
                                auto y_expected = conv_layer.run_plaintext(input_array);

                                print_double_message(y_mg.to_array_1d().data(), "output_mg", 10);
                                print_double_message(y_expected.to_array_1d().data(), "plain_output", 10);

                                auto compare_result = compare(y_expected, y_mg);
                                REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
                                REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "inv_mux_conv_repack", "", HeteroProcessors) {
    // Test case: output_shape < block_shape, triggers repack to multiplexed format
    Duo stride = {4, 4};
    Duo skip = {1, 1};
    int init_level = 2;

    vector<Duo> kernel_shapes = {{1, 1}, {3, 3}, {5, 5}};
    vector<Duo> block_shapes = {{64, 64}, {64, 128}};
    vector<uint32_t> nc_ins = {2, 3, 3};
    vector<uint32_t> nc_outs = {3, 4, 15};

    for (auto& kernel_shape : kernel_shapes) {
        for (auto& block_shape : block_shapes) {
            Duo input_shape = block_shape * 2;
            Duo stride_next = {1, 1};

            Array<int, 1> padding({2});
            padding.set(0, -1);
            padding.set(1, -1);

            string config_name = "ks_" + to_string(kernel_shape[0]) + "x" + to_string(kernel_shape[1]) + "_bs_" +
                                 to_string(block_shape[0]) + "x" + to_string(block_shape[1]) + "_is_" +
                                 to_string(input_shape[0]) + "x" + to_string(input_shape[1]);

            SECTION(config_name) {
                for (size_t idx = 0; idx < nc_ins.size(); idx++) {
                    uint32_t n_in_channel = nc_ins[idx];
                    uint32_t n_out_channel = nc_outs[idx];
                    SECTION("cin=" + to_string(n_in_channel) + "_cout=" + to_string(n_out_channel)) {
                        Array<double, 4> conv0_weight =
                            gen_random_array<4>({n_out_channel, n_in_channel, kernel_shape[0], kernel_shape[1]}, 0.1);
                        Array<double, 1> conv0_bias = gen_random_array<1>({n_out_channel}, 0.1);
                        Array<double, 3> input_array =
                            gen_random_array<3>({n_in_channel, input_shape[0], input_shape[1]}, 1.0);

                        InverseMultiplexedConv2DLayer conv_layer(this->context.parameter(), input_shape,
                                                                 move(conv0_weight), move(conv0_bias), padding, stride,
                                                                 block_shape, init_level, 1.0);
                        conv_layer.prepare_weight();

                        Feature2DEncrypted input_feature(&this->context, init_level, skip);
                        Duo effective_stride = input_shape / block_shape;
                        Duo total_stride = effective_stride;
                        input_feature.pack_interleaved(input_array, block_shape, total_stride,
                                                       this->context.parameter().default_scale());

                        Duo output_shape = input_shape / stride;
                        Duo out_skip = block_shape / output_shape;
                        uint32_t output_n_channel_per_ct = this->N / (2 * prod(output_shape));

                        Feature2DEncrypted output_feature(&this->context, init_level - 2);
                        output_feature.shape = output_shape;
                        output_feature.skip = out_skip;
                        output_feature.n_channel = n_out_channel;
                        output_feature.n_channel_per_ct = output_n_channel_per_ct;
                        int n_out_cts = div_ceil(n_out_channel, output_n_channel_per_ct);
                        for (int i = 0; i < n_out_cts; i++) {
                            output_feature.data.emplace_back(this->context.parameter(), init_level - 2);
                            output_feature.data.back().set_scale(this->param.default_scale());
                        }

                        vector<CxxVectorArgument> cxx_args;
                        fs::path project_path =
                            base_path / "CKKS_inverse_multiplexed_conv2d" /
                            ("stride_" + to_string(stride[0]) + "_" + to_string(stride[1])) /
                            ("kernel_shape_" + to_string(kernel_shape[0]) + "_" + to_string(kernel_shape[1])) /
                            ("cin_" + to_string(n_in_channel) + "_cout_" + to_string(n_out_channel)) /
                            ("input_shape_" + to_string(input_shape[0]) + "_" + to_string(input_shape[1])) /
                            ("level_" + to_string(init_level)) / "server";

                        // Generate repack mask for CxxVectorArgument
                        vector<CkksPlaintext> repack_mask_vec;
                        repack_mask_vec.push_back(conv_layer.generate_repack_mask_pt(this->context));

                        auto arg_names = read_arg_names(project_path);
                        for (const auto& name : arg_names) {
                            if (name.rfind("input", 0) == 0)
                                cxx_args.push_back({name, &input_feature.data});
                            else if (name.rfind("convw_", 0) == 0)
                                cxx_args.push_back({name, &conv_layer.weight_pt});
                            else if (name.rfind("convb_", 0) == 0)
                                cxx_args.push_back({name, &conv_layer.bias_pt});
                            else if (name.rfind("repack_mask_", 0) == 0)
                                cxx_args.push_back({name, &repack_mask_vec});
                            else if (name.rfind("output", 0) == 0)
                                cxx_args.push_back({name, &output_feature.data});
                        }

                        this->run(project_path, cxx_args);

                        auto y_mg = output_feature.unpack_multiplexed();
                        auto y_expected = conv_layer.run_plaintext(input_array);

                        print_double_message(y_mg.to_array_1d().data(), "output_mg", 10);
                        print_double_message(y_expected.to_array_1d().data(), "plain_output", 10);

                        auto compare_result = compare(y_expected, y_mg);
                        REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
                        REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
                    }
                }
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "inv_mux_dw_conv", "", HeteroProcessors) {
    Duo skip = {1, 1};
    int init_level = 1;

    vector<Duo> kernel_shapes = {{3, 3}, {5, 5}};
    vector<Duo> strides_list = {{1, 1}, {2, 2}};
    vector<Duo> block_shapes = {{64, 64}};
    vector<uint32_t> multipliers = {2, 4};
    vector<uint32_t> n_channels = {2, 3, 5, 8};

    for (auto& kernel_shape : kernel_shapes) {
        for (auto& stride : strides_list) {
            for (auto& block_shape : block_shapes) {
                for (auto mult : multipliers) {
                    Duo input_shape = block_shape * mult;
                    Duo stride_next = input_shape / (block_shape * stride);

                    Array<int, 1> padding({2});
                    padding.set(0, -1);
                    padding.set(1, -1);

                    string config_name = "ks_" + to_string(kernel_shape[0]) + "x" + to_string(kernel_shape[1]) +
                                         "_st_" + to_string(stride[0]) + "x" + to_string(stride[1]) + "_bs_" +
                                         to_string(block_shape[0]) + "x" + to_string(block_shape[1]) + "_is_" +
                                         to_string(input_shape[0]) + "x" + to_string(input_shape[1]);

                    SECTION(config_name) {
                        FOR_EACH_SECTION(uint32_t n_channel, n_channels, "ch=" + to_string(n_channel)) {
                            // Depthwise: weight shape is [n_channel, 1, kH, kW]
                            Array<double, 4> conv0_weight =
                                gen_random_array<4>({n_channel, 1, kernel_shape[0], kernel_shape[1]}, 0.1);
                            Array<double, 1> conv0_bias = gen_random_array<1>({n_channel}, 0.1);
                            // Depthwise: input channels == output channels
                            Array<double, 3> input_array =
                                gen_random_array<3>({n_channel, input_shape[0], input_shape[1]}, 1.0);

                            InverseMultiplexedConv2DLayerDepthwise conv_layer(
                                this->context.parameter(), input_shape, move(conv0_weight), move(conv0_bias),
                                padding, stride, block_shape, init_level, 1.0);
                            conv_layer.prepare_weight();

                            // Pack input using interleaved packing
                            Feature2DEncrypted input_feature(&this->context, init_level, skip);
                            Duo total_stride = stride * stride_next;
                            input_feature.pack_interleaved(input_array, block_shape, total_stride,
                                                           this->context.parameter().default_scale());

                            Duo output_shape = input_shape / stride;
                            uint32_t output_total = prod(output_shape);
                            uint32_t output_n_channel_per_ct;
                            if (2 * output_total < this->N) {
                                output_n_channel_per_ct = this->N / (2 * output_total);
                            } else {
                                output_n_channel_per_ct = 1;
                            }

                            Feature2DEncrypted output_feature(&this->context, init_level - 1);
                            output_feature.shape = output_shape;
                            output_feature.skip = {1, 1};
                            output_feature.n_channel = n_channel;
                            output_feature.n_channel_per_ct = output_n_channel_per_ct;
                            int n_out_cts = div_ceil(n_channel, output_n_channel_per_ct) * prod(stride_next);
                            for (int i = 0; i < n_out_cts; i++) {
                                output_feature.data.emplace_back(this->context.parameter(), init_level - 1);
                                output_feature.data.back().set_scale(this->param.default_scale());
                            }

                            vector<CxxVectorArgument> cxx_args;
                            fs::path project_path =
                                base_path / "CKKS_inverse_multiplexed_dw_conv2d" /
                                ("stride_" + to_string(stride[0]) + "_" + to_string(stride[1])) /
                                ("kernel_shape_" + to_string(kernel_shape[0]) + "_" + to_string(kernel_shape[1])) /
                                ("cin_" + to_string(n_channel) + "_cout_" + to_string(n_channel)) /
                                ("input_shape_" + to_string(input_shape[0]) + "_" + to_string(input_shape[1])) /
                                ("level_" + to_string(init_level)) / "server";

                            auto arg_names = read_arg_names(project_path);
                            for (const auto& name : arg_names) {
                                if (name.rfind("input", 0) == 0)
                                    cxx_args.push_back({name, &input_feature.data});
                                else if (name.rfind("convw_", 0) == 0)
                                    cxx_args.push_back({name, &conv_layer.weight_pt});
                                else if (name.rfind("convb_", 0) == 0)
                                    cxx_args.push_back({name, &conv_layer.bias_pt});
                                else if (name.rfind("output", 0) == 0)
                                    cxx_args.push_back({name, &output_feature.data});
                            }

                            this->run(project_path, cxx_args);

                            Array<double, 3> y_mg;
                            if (output_shape[0] > block_shape[0] || output_shape[1] > block_shape[1]) {
                                y_mg = output_feature.unpack_interleaved(block_shape, stride_next);
                            } else {
                                y_mg = output_feature.unpack_multiplexed();
                            }
                            auto y_expected = conv_layer.run_plaintext(input_array);

                            print_double_message(y_mg.to_array_1d().data(), "output_mg", 10);
                            print_double_message(y_expected.to_array_1d().data(), "plain_output", 10);

                            auto compare_result = compare(y_expected, y_mg);
                            REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
                            REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
                        }
                    }
                }
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "fc_skip_feature0d", "", HeteroProcessors) {
    // Matches Python test_fc_pack_skip: n_in=4096, n_out=10, level=2, skips=[2,4,8,16]
    uint32_t n_in_channel = 4096;
    uint32_t n_out_channel = 10;
    int init_level = 2;

    auto input_1d = gen_random_array<1>({n_in_channel}, 0.1);
    auto weight = gen_random_array<2>({n_out_channel, n_in_channel}, 0.5);
    auto bias = gen_random_array<1>({n_out_channel}, 0.1);

    vector<uint32_t> skip_shapes = {2, 4, 8, 16};
    FOR_EACH_SECTION(uint32_t s, skip_shapes, "skip=" + to_string(s * s)) {
        uint32_t skip_0d = s * s;
        uint32_t n_channel_per_ct = this->n_slot / skip_0d;
        uint32_t n_packed_in = div_ceil(n_in_channel, n_channel_per_ct);
        uint32_t n_packed_out = div_ceil(n_out_channel, n_channel_per_ct);

        Feature0DEncrypted input_feature(&this->context, init_level);
        input_feature.pack(input_1d, this->param.default_scale(), skip_0d);

        DensePackedLayer dense(this->context.parameter(), move(weight), move(bias), n_channel_per_ct, init_level,
                               0);
        dense.prepare_weight_0d_skip(skip_0d);

        Feature0DEncrypted output_feature(&this->context, init_level - 1);
        output_feature.n_channel = n_out_channel;
        output_feature.n_channel_per_ct = n_channel_per_ct;
        output_feature.skip = skip_0d;
        for (uint32_t i = 0; i < n_packed_out; i++) {
            output_feature.data.emplace_back(this->context.parameter(), init_level - 1);
            output_feature.data.back().set_scale(this->param.default_scale());
        }

        fs::path project_path = base_path /
                                ("CKKS_fc_prepare_weight1_1D_pack_skip_" + to_string(s) + "_" + to_string(s)) /
                                ("level_" + to_string(init_level)) / "server";

        auto arg_names = read_arg_names(project_path);
        vector<CxxVectorArgument> cxx_args;
        for (const auto& name : arg_names) {
            if (name == "input_node")
                cxx_args.push_back({name, &input_feature.data});
            else if (name == "weight_pt")
                cxx_args.push_back({name, &dense.weight_pt});
            else if (name == "bias_pt")
                cxx_args.push_back({name, &dense.bias_pt});
            else if (name == "output_ct")
                cxx_args.push_back({name, &output_feature.data});
        }

        this->run(project_path, cxx_args);

        Array<double, 1> output_mg = output_feature.unpack();
        Array<double, 1> plain_output = dense.run_plaintext(input_1d);

        print_double_message(output_mg.to_array_1d().data(), "output_mg", 10);
        print_double_message(plain_output.to_array_1d().data(), "plain_output", 10);

        auto compare_result = compare(plain_output, output_mg);
        REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
        REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "fc_multiplexed_feature2d", "", HeteroProcessors) {
    int init_level = 3;
    uint32_t n_in_channel = 64;
    uint32_t n_out_channel = 10;

    auto weight = gen_random_array<2>({n_out_channel, n_in_channel}, 0.5);
    auto bias = gen_random_array<1>({n_out_channel}, 0.1);

    struct TestConfig {
        Duo shape;
        Duo skip;
        Duo invalid_fill;
    };
    vector<TestConfig> configs = {
        {{1, 1}, {2, 2}, {1, 1}},   {{1, 1}, {4, 4}, {1, 1}},   {{1, 1}, {8, 8}, {1, 1}},
        {{1, 1}, {32, 32}, {8, 8}}, {{1, 1}, {16, 16}, {4, 4}}, {{2, 2}, {4, 4}, {4, 4}},
    };

    FOR_EACH_SECTION(auto& cfg, configs,
                     "shape=" + str(cfg.shape) + " skip=" + str(cfg.skip) + " inv=" + str(cfg.invalid_fill)) {
        auto input_3d = gen_random_array<3>({n_in_channel, cfg.shape[0], cfg.shape[1]}, 0.1);
        auto input_1d = Array<double, 1>::from_array_1d(input_3d.to_array_1d());

        Feature2DEncrypted input_feature(&this->context, init_level, cfg.skip, cfg.invalid_fill);
        input_feature.pack_multiplexed(input_3d, this->param.default_scale());

        ReshapeLayer reshape(this->param);
        Feature0DEncrypted input_0d = reshape.call(this->context, input_feature);

        uint32_t block_size = prod(cfg.shape * cfg.skip);
        uint32_t n_blocks_per_ct = div_ceil((uint32_t)this->n_slot, block_size);

        DensePackedLayer dense(this->context.parameter(), move(weight), move(bias), n_blocks_per_ct, init_level, 0);
        dense.prepare_weight_for_2d_multiplexed(cfg.shape, cfg.skip, cfg.invalid_fill);

        uint32_t n_packed_out = div_ceil(n_out_channel, n_blocks_per_ct);
        Feature0DEncrypted output_feature(&this->context, init_level - 1);
        output_feature.n_channel = n_out_channel;
        output_feature.n_channel_per_ct = n_blocks_per_ct;
        output_feature.skip = input_0d.skip;
        for (uint32_t i = 0; i < n_packed_out; i++) {
            output_feature.data.emplace_back(this->context.parameter(), init_level - 1);
            output_feature.data.back().set_scale(this->param.default_scale());
        }

        fs::path project_path =
            base_path /
            ("CKKS_fc_multiplexed"
             "_shape" +
             to_string(cfg.shape[0]) + "x" + to_string(cfg.shape[1]) + "_skip" + to_string(cfg.skip[0]) + "x" +
             to_string(cfg.skip[1]) + "_inv" + to_string(cfg.invalid_fill[0]) + "x" + to_string(cfg.invalid_fill[1])) /
            ("level_" + to_string(init_level)) / "server";

        auto arg_names = read_arg_names(project_path);
        vector<CxxVectorArgument> cxx_args;
        for (const auto& name : arg_names) {
            if (name == "input_node")
                cxx_args.push_back({name, &input_0d.data});
            else if (name == "weight_pt")
                cxx_args.push_back({name, &dense.weight_pt});
            else if (name == "bias_pt")
                cxx_args.push_back({name, &dense.bias_pt});
            else if (name == "output_ct")
                cxx_args.push_back({name, &output_feature.data});
        }

        this->run(project_path, cxx_args);

        Array<double, 1> output_mg = output_feature.unpack();
        Array<double, 1> plain_output = dense.run_plaintext(input_1d);

        print_double_message(output_mg.to_array_1d().data(), "output_mg", 10);
        print_double_message(plain_output.to_array_1d().data(), "plain_output", 10);

        auto compare_result = compare(plain_output, output_mg);
        REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
        REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "fc_fc_feature0d", "", HeteroProcessors) {
    int init_level = 2;
    // fc0: multiplexed, dense_shape=[4,4], skip=(1,1), 1024->1024
    // fc1: skip_0d, skip1=[4,4], 1024->128
    uint32_t input_channel = 1024;
    uint32_t output_channel = 1024;
    Duo dense_shape = {4, 4};
    Duo skip0 = {1, 1};

    Array<double, 2> weight0 = gen_random_array<2>({output_channel, input_channel}, 1);
    Array<double, 1> bias0 = gen_random_array<1>({output_channel}, 1);

    // fc1: skip_0d — 1024-in -> 128-out, skip = [4,4] (dense_shape * skip0)
    uint32_t output_channel1 = 128;
    Duo skip1 = dense_shape * skip0;

    Array<double, 2> weight1 = gen_random_array<2>({output_channel1, output_channel}, 1);
    Array<double, 1> bias1 = gen_random_array<1>({output_channel1}, 1);

    // Input: direct 0D pack — ceil(1024/8192)=1 CT, skip=1 (matches Python)
    Array<double, 1> input_1d = gen_random_array<1>({input_channel}, 0.1);
    Feature0DEncrypted input_feature(&this->context, init_level);
    input_feature.pack(input_1d, this->param.default_scale(), /*skip=*/1);

    // fc0: multiplexed — block=[4*1]*[4*1]=16 slots, n_num_pre_ct=ceil(8192/16)=512
    uint32_t block_size0 = prod(dense_shape * skip0);
    uint32_t n_num_pre_ct0 = div_ceil((uint32_t)this->n_slot, block_size0);
    DensePackedLayer dense0(this->context.parameter(), move(weight0), move(bias0), n_num_pre_ct0, init_level, 0);
    dense0.prepare_weight_for_2d_multiplexed(dense_shape, skip0);

    uint32_t skip_0d1 = prod(skip1);
    uint32_t n_channel_per_ct1 = this->n_slot / skip_0d1;
    DensePackedLayer dense1(this->context.parameter(), move(weight1), move(bias1), n_channel_per_ct1,
                            init_level - 1, 0);
    dense1.prepare_weight_0d_skip(skip_0d1);

    // Output placeholder
    uint32_t n_out1 = div_ceil(output_channel1, n_channel_per_ct1);
    Feature0DEncrypted output_feature(&this->context, init_level - 2);
    output_feature.n_channel = output_channel1;
    output_feature.n_channel_per_ct = n_channel_per_ct1;
    output_feature.skip = skip_0d1;
    for (uint32_t i = 0; i < n_out1; i++) {
        output_feature.data.emplace_back(this->context.parameter(), init_level - 2);
        output_feature.data.back().set_scale(this->param.default_scale());
    }

    // mega_ag execution
    fs::path project_path = base_path /
                            ("CKKS_fc_fc_" + to_string(input_channel) + "_" + to_string(output_channel) + "_" +
                             to_string(output_channel1)) /
                            ("level_" + to_string(init_level)) / "server";

    auto arg_names = read_arg_names(project_path);
    vector<CxxVectorArgument> cxx_args;
    for (const auto& name : arg_names) {
        if (name == "input_node")
            cxx_args.push_back({name, &input_feature.data});
        else if (name == "weight_pt0")
            cxx_args.push_back({name, &dense0.weight_pt});
        else if (name == "bias_pt0")
            cxx_args.push_back({name, &dense0.bias_pt});
        else if (name == "weight_pt1")
            cxx_args.push_back({name, &dense1.weight_pt});
        else if (name == "bias_pt1")
            cxx_args.push_back({name, &dense1.bias_pt});
        else if (name == "output_ct")
            cxx_args.push_back({name, &output_feature.data});
    }

    this->run(project_path, cxx_args);

    Array<double, 1> output_mg = output_feature.unpack();

    Array<double, 1> output_plain_0 = dense0.run_plaintext(input_1d);
    Array<double, 1> output_plain_1 = dense1.run_plaintext(output_plain_0);

    print_double_message(output_mg.to_array_1d().data(), "output_mg", 10);
    print_double_message(output_plain_1.to_array_1d().data(), "plain_output", 10);
    ArrayComparison result = compare(output_plain_1, output_mg);
    REQUIRE(result.max_error < 5.0e-2 * result.max_abs);
    REQUIRE(result.rmse < 1.0e-2 * result.rms);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "poly_bsgs_feature2d", "", HeteroProcessors) {
    Duo input_shape = {32, 32};
    uint32_t n_channel = 32;
    Duo skip = {1, 1};
    uint32_t n_channel_per_ct = div_ceil(this->n_slot, prod(input_shape));
    int init_level = 8;
    vector<int> orders = {2, 4, 6, 8, 10, 12, 16, 32, 64};

    FOR_EACH_SECTION(uint32_t order, orders, "order=" + to_string(order)) {
        auto input_array = gen_random_array<3>({n_channel, input_shape[0], input_shape[1]}, 1.0);
        auto weight = gen_random_array<2>({order + 1, n_channel}, 1.0);

        Feature2DEncrypted input_feature(&this->context, init_level, skip);
        input_feature.pack_multiplexed(input_array, this->context.parameter().default_scale());

        PolyRelu2D polyx(this->context.parameter(), {input_shape[0], input_shape[1]}, order, move(weight), skip,
                         n_channel_per_ct, init_level);
        polyx.prepare_weight_bsgs();

        int output_level = init_level - PolyRelu2D::compute_bsgs_level_cost(order);
        Feature2DEncrypted output_feature(&this->context, output_level);
        output_feature.skip = skip;
        output_feature.shape = input_shape;
        output_feature.n_channel = n_channel;
        output_feature.n_channel_per_ct = input_feature.n_channel_per_ct;
        for (int i = 0; i < div_ceil(n_channel, n_channel_per_ct); i++) {
            output_feature.data.emplace_back(this->context.parameter(), output_level);
            output_feature.data.back().set_scale(this->param.default_scale());
        }

        fs::path project_path = base_path /
                                ("CKKS_poly_relu_bsgs_" + to_string(n_channel) + "_channel_order_" + to_string(order)) /
                                ("level_" + to_string(init_level));

        auto arg_names = read_arg_names(project_path);
        vector<CxxVectorArgument> cxx_args;
        int idx = 0;
        cxx_args.push_back({arg_names[idx++], &input_feature.data});
        for (int i = 0; i <= order; i++) {
            cxx_args.push_back({arg_names[idx++], &polyx.weight_pt[i]});
        }
        cxx_args.push_back({arg_names[idx++], &output_feature.data});

        this->run(project_path, cxx_args);

        auto output_mg = output_feature.unpack_multiplexed();
        auto output_mg_expected = polyx.run_plaintext_for_non_absorb_case(input_array);

        INFO("order=" << order);
        print_double_message(output_mg.to_array_1d().data(), "output_mg", 10);
        print_double_message(output_mg_expected.to_array_1d().data(), "output_mg_expected", 10);

        auto compare_result = compare(output_mg_expected, output_mg);
        REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
        REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "poly_bsgs_feature0d", "", HeteroProcessors) {
    uint32_t n_channel = 32;
    int init_level = 8;
    vector<int> orders = {2, 4, 6, 8};
    vector<uint32_t> skips = {1, 2, 128, 256};

    FOR_EACH_SECTION(uint32_t skip_val, skips, "skip=" + to_string(skip_val)) {
        uint32_t n_channel_per_ct = this->n_slot / skip_val;

        for (uint32_t order : orders) {
            int level_cost = PolyRelu2D::compute_bsgs_level_cost(order);
            if (init_level < level_cost)
                continue;

            SECTION("order=" + to_string(order)) {
                auto input_array = gen_random_array<1>({n_channel}, 1.0);
                auto weight = gen_random_array<2>({order + 1, n_channel}, 0.5);

                // Pack into Feature0DEncrypted
                Feature0DEncrypted input_feature(&this->context, init_level);
                input_feature.n_channel = n_channel;
                input_feature.pack(input_array, this->param.default_scale(), skip_val);

                // Create PolyRelu0D for Feature0D
                PolyRelu0D polyx(this->context.parameter(), move(weight), init_level, order, skip_val);
                polyx.prepare_weight_0d_skip();

                int output_level = init_level - level_cost;
                uint32_t n_packed_ct = div_ceil(n_channel, n_channel_per_ct);

                Feature0DEncrypted output_feature(&this->context, output_level);
                output_feature.skip = skip_val;
                output_feature.n_channel = n_channel;
                output_feature.n_channel_per_ct = n_channel_per_ct;
                for (uint32_t i = 0; i < n_packed_ct; i++) {
                    output_feature.data.emplace_back(this->context.parameter(), output_level);
                    output_feature.data.back().set_scale(this->param.default_scale());
                }

                fs::path project_path = base_path /
                                        ("CKKS_poly_relu_bsgs_feature0d_" + to_string(n_channel) + "_channel_order_" +
                                         to_string(order) + "_skip_" + to_string(skip_val)) /
                                        ("level_" + to_string(init_level));

                auto arg_names = read_arg_names(project_path);
                vector<CxxVectorArgument> cxx_args;
                int idx = 0;
                cxx_args.push_back({arg_names[idx++], &input_feature.data});
                for (int i = 0; i <= (int)order; i++) {
                    cxx_args.push_back({arg_names[idx++], &polyx.weight_pt[i]});
                }
                cxx_args.push_back({arg_names[idx++], &output_feature.data});

                this->run(project_path, cxx_args);

                auto output_mg = output_feature.unpack();
                auto output_mg_expected = polyx.run_plaintext(input_array);

                INFO("order=" << order << " skip=" << skip_val);
                print_double_message(output_mg.to_array_1d().data(), "output_mg", 10);
                print_double_message(output_mg_expected.to_array_1d().data(), "output_mg_expected", 10);

                auto compare_result = compare(output_mg_expected, output_mg);
                REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
                REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
            }
        }
    }
}

// ---- poly_relu 1D, skip-pack mode ----
// Matches Feature1DEncrypted::pack():
//   channel ch, position i → slot = ch * shape * skip + i * skip
//   n_channel_per_ct = N/2 / (shape * skip)
TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "poly_bsgs_feature1d_skip", "", HeteroProcessors) {
    uint32_t n_channel = 64;
    int init_level = 8;
    vector<uint32_t> shapes = {32, 64};
    vector<uint32_t> skips = {1, 2};
    vector<int> orders = {2, 4, 7};

    FOR_EACH_SECTION(uint32_t shape, shapes, "shape=" + to_string(shape)) {
        for (uint32_t skip_val : skips) {
            uint32_t n_channel_per_ct = this->n_slot / shape / skip_val;
            if (n_channel_per_ct == 0)
                continue;

            SECTION("skip=" + to_string(skip_val)) {
                for (int order : orders) {
                    int level_cost = PolyRelu1D::compute_bsgs_level_cost(order);
                    if (init_level < level_cost)
                        continue;

                    SECTION("order=" + to_string(order)) {
                        auto input_array = gen_random_array<2>({n_channel, shape}, 1.0);
                        auto weight = gen_random_array<2>({(uint64_t)order + 1, n_channel}, 0.5);

                        Feature1DEncrypted input_feature(&this->context, init_level, skip_val);
                        input_feature.pack(input_array, this->param.default_scale());

                        PolyRelu1D polyx(this->context.parameter(), move(weight), init_level, order, skip_val,
                                         shape);
                        polyx.prepare_weight_bsgs();

                        int output_level = init_level - level_cost;
                        uint32_t n_packed_ct = div_ceil(n_channel, n_channel_per_ct);

                        Feature1DEncrypted output_feature(&this->context, output_level, skip_val);
                        output_feature.shape = shape;
                        output_feature.skip = skip_val;
                        output_feature.n_channel = n_channel;
                        output_feature.n_channel_per_ct = n_channel_per_ct;
                        for (uint32_t i = 0; i < n_packed_ct; i++) {
                            output_feature.data.emplace_back(this->context.parameter(), output_level);
                            output_feature.data.back().set_scale(this->param.default_scale());
                        }

                        fs::path project_path =
                            base_path /
                            ("CKKS_poly_relu_bsgs_feature1d_skip_" + to_string(n_channel) + "_channel" + "_shape" +
                             to_string(shape) + "_skip" + to_string(skip_val) + "_order" + to_string(order)) /
                            ("level_" + to_string(init_level));

                        auto arg_names = read_arg_names(project_path);
                        vector<CxxVectorArgument> cxx_args;
                        int idx = 0;
                        cxx_args.push_back({arg_names[idx++], &input_feature.data});
                        for (int i = 0; i <= order; i++) {
                            cxx_args.push_back({arg_names[idx++], &polyx.weight_pt[i]});
                        }
                        cxx_args.push_back({arg_names[idx++], &output_feature.data});

                        this->run(project_path, cxx_args);

                        auto output_mg = output_feature.unpack();
                        auto output_mg_expected = polyx.run_plaintext(input_array);

                        INFO("shape=" << shape << " skip=" << skip_val << " order=" << order);
                        print_double_message(output_mg.to_array_1d().data(), "output_mg", 10);
                        print_double_message(output_mg_expected.to_array_1d().data(), "output_mg_expected", 10);

                        auto compare_result = compare(output_mg_expected, output_mg);
                        REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
                        REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
                    }
                }
            }
        }
    }
}

// ---- poly_relu 1D, multiplexed/interleaved-pack mode ----
// Matches Feature1DEncrypted::pack_multiplexed():
//   channel j (CT-local), position i → slot = (j/skip)*shape*skip + i*skip + (j%skip)
//   n_channel_per_ct = N/2 / shape   (skip channels share each shape*skip block)
TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "poly_bsgs_feature1d_mux", "", HeteroProcessors) {
    uint32_t n_channel = 64;
    int init_level = 8;
    vector<uint32_t> shapes = {32, 64};
    vector<uint32_t> skips = {1, 2};
    vector<int> orders = {2, 4, 7};

    FOR_EACH_SECTION(uint32_t shape, shapes, "shape=" + to_string(shape)) {
        for (uint32_t skip_val : skips) {
            // multiplexed: n_channel_per_ct = N/2 / shape (no skip in denominator)
            uint32_t n_channel_per_ct = this->n_slot / shape;

            SECTION("skip=" + to_string(skip_val)) {
                for (int order : orders) {
                    int level_cost = PolyRelu1D::compute_bsgs_level_cost(order);
                    if (init_level < level_cost)
                        continue;

                    SECTION("order=" + to_string(order)) {
                        auto input_array = gen_random_array<2>({n_channel, shape}, 1.0);
                        auto weight = gen_random_array<2>({(uint64_t)order + 1, n_channel}, 0.5);

                        Feature1DEncrypted input_feature(&this->context, init_level, skip_val);
                        input_feature.pack_multiplexed(input_array, this->param.default_scale());

                        PolyRelu1D polyx(this->context.parameter(), move(weight), init_level, order, skip_val,
                                         shape);
                        polyx.prepare_weight_bsgs_mux();

                        int output_level = init_level - level_cost;
                        uint32_t n_packed_ct = div_ceil(n_channel, n_channel_per_ct);

                        Feature1DEncrypted output_feature(&this->context, output_level, skip_val);
                        output_feature.shape = shape;
                        output_feature.skip = skip_val;
                        output_feature.n_channel = n_channel;
                        output_feature.n_channel_per_ct = n_channel_per_ct;
                        for (uint32_t i = 0; i < n_packed_ct; i++) {
                            output_feature.data.emplace_back(this->context.parameter(), output_level);
                            output_feature.data.back().set_scale(this->param.default_scale());
                        }

                        fs::path project_path =
                            base_path /
                            ("CKKS_poly_relu_bsgs_feature1d_mux_" + to_string(n_channel) + "_channel" + "_shape" +
                             to_string(shape) + "_skip" + to_string(skip_val) + "_order" + to_string(order)) /
                            ("level_" + to_string(init_level));

                        auto arg_names = read_arg_names(project_path);
                        vector<CxxVectorArgument> cxx_args;
                        int idx = 0;
                        cxx_args.push_back({arg_names[idx++], &input_feature.data});
                        for (int i = 0; i <= order; i++) {
                            cxx_args.push_back({arg_names[idx++], &polyx.weight_pt[i]});
                        }
                        cxx_args.push_back({arg_names[idx++], &output_feature.data});

                        this->run(project_path, cxx_args);

                        auto output_mg = output_feature.unpack_multiplexed();
                        auto output_mg_expected = polyx.run_plaintext(input_array);

                        INFO("shape=" << shape << " skip=" << skip_val << " order=" << order);
                        print_double_message(output_mg.to_array_1d().data(), "output_mg", 10);
                        print_double_message(output_mg_expected.to_array_1d().data(), "output_mg_expected", 10);

                        auto compare_result = compare(output_mg_expected, output_mg);
                        REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
                        REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
                    }
                }
            }
        }
    }
}

// ── BlockColMajor / ParBlockColMajor E2E tests ──────────────────────────────
//
// These tests exercise the mega_ag pipeline generated by the Python test_gen_layers.py.
// They use FheTask with a custom encode_pt executor to lazily generate plaintext
// diagonals during task execution.

static double e2e_max_rel_error(const Array<double, 2>& expected, const Array<double, 2>& actual) {
    double max_abs = 0, max_err = 0;
    for (uint32_t i = 0; i < expected.get_shape()[0]; i++)
        for (uint32_t j = 0; j < expected.get_shape()[1]; j++) {
            max_abs = std::max(max_abs, std::abs(expected.get(i, j)));
            max_err = std::max(max_err, std::abs(expected.get(i, j) - actual.get(i, j)));
        }
    return max_abs > 0 ? max_err / max_abs : max_err;
}

static ExecutorFunc make_block_col_major_encode_pt_executor() {
    return [](ExecutionContext& exec_ctx, std::unordered_map<NodeId, std::any>& local_data,
              const ComputeNode& self) -> void {
        CkksContext* ctx_ptr = exec_ctx.get_arithmetic_context<CkksContext>();
        if (!ctx_ptr)
            throw std::runtime_error("encode_pt: cannot get CkksContext");
        if (!self.custom_prop.has_value())
            throw std::runtime_error("encode_pt: missing custom_prop");

        const auto& attrs = self.custom_prop->attributes;
        const std::string op_class = attrs.at("op_class").get<std::string>();
        const std::string type = attrs.at("type").get<std::string>();
        auto attr_u32 = [&](const std::string& key, uint32_t fallback) -> uint32_t {
            return attrs.contains(key) ? attrs.at(key).get<uint32_t>() : fallback;
        };

        const NodeId& in_id = self.input_nodes[0]->id;
        auto cd_ptr = std::any_cast<std::shared_ptr<CustomData>>(local_data.at(in_id));
        void* layer_ptr = cd_ptr->get_typed_data<void>();

        CkksPlaintext pt = [&]() -> CkksPlaintext {
            if (op_class == "ParBlockColMajorCPMM") {
                auto* layer = static_cast<ParBlockColMajorCPMM*>(layer_ptr);
                if (type == "diag_pt")
                    return layer->generate_diag_pt(*ctx_ptr, attrs.value("mb", 0), attrs.value("g", 0),
                                                   attrs.value("bp", 0), attrs.value("k", 0));
                if (type == "mask_h0_pt")
                    return layer->generate_mask_h0_pt(*ctx_ptr);
                if (type == "bias_pt")
                    return layer->generate_bias_pt(*ctx_ptr, attrs.value("mb", 0), attrs.value("bi", 0),
                                                   attrs.value("g", 0));
                throw std::runtime_error("encode_pt: unknown type for ParBlockColMajorCPMM: " + type);
            }
            if (op_class == "ParBlockColMajorTranspose") {
                auto* layer = static_cast<ParBlockColMajorTranspose*>(layer_ptr);
                if (type == "transpose_diag_pt")
                    return layer->generate_transpose_diag_pt(
                        *ctx_ptr, (attrs.contains("k_idx") ? attrs.at("k_idx").get<uint32_t>() : attrs.value("i", 0u)));
                throw std::runtime_error("encode_pt: unknown type for ParBlockColMajorTranspose: " + type);
            }
            if (op_class == "ParBlockColMajorAddPt") {
                auto* layer = static_cast<ParBlockColMajorAddPt*>(layer_ptr);
                return layer->generate_pt(*ctx_ptr, attrs.value("bi", 0), attrs.value("bj", 0), attrs.value("g", 0));
            }
            if (op_class == "ParBlockColMajorCCMM") {
                auto* layer = static_cast<ParBlockColMajorCCMM*>(layer_ptr);
                if (type == "sigma_pt")
                    return layer->generate_sigma_pt(*ctx_ptr, attrs.value("k", 0));
                if (type == "tau_pt")
                    return layer->generate_tau_pt(*ctx_ptr, attrs.value("offset_idx", 0));
                if (type == "psi_k0_pt")
                    return layer->generate_psi_k0_pt(*ctx_ptr);
                if (type == "psi_wk_pt")
                    return layer->generate_psi_wk_pt(*ctx_ptr, attrs.value("i", 1));
                if (type == "psi_wkd_pt")
                    return layer->generate_psi_wkd_pt(*ctx_ptr, attrs.value("i", 1));
                throw std::runtime_error("encode_pt: unknown type for ParBlockColMajorCCMM: " + type);
            }
            if (op_class == "ParBlockColMajorPolyActRNGamma") {
                auto* layer = static_cast<ParBlockColMajorPolyActRNGamma*>(layer_ptr);
                return layer->generate_gamma_pt(*ctx_ptr, attrs.value("mb", 0), attrs.value("bi", 0),
                                                attrs.value("bj", 0), attrs.value("g", 0));
            }
            if (op_class == "ParBlockColMajorPolyActRNPoly") {
                auto* layer = static_cast<ParBlockColMajorPolyActRNPoly*>(layer_ptr);
                return layer->generate_coeff_pt(*ctx_ptr, attrs.value("coeff_idx", 0), attrs.value("mb", 0),
                                                attrs.value("bi", 0), attrs.value("bj", 0), attrs.value("g", 0));
            }
            if (op_class == "ParBlockColMajorLNStats") {
                auto* layer = static_cast<ParBlockColMajorLNStats*>(layer_ptr);
                return layer->generate_pt(*ctx_ptr, attrs.value("pt_idx", 0), attrs.value("bi", 0),
                                          attrs.value("bj", 0), attrs.value("g", 0));
            }
            if (op_class == "ParBlockColMajorLNXCentered") {
                auto* layer = static_cast<ParBlockColMajorLNXCentered*>(layer_ptr);
                return layer->generate_pt(*ctx_ptr, attrs.value("pt_idx", 0), attrs.value("bi", 0),
                                          attrs.value("bj", 0), attrs.value("g", 0));
            }
            if (op_class == "ParBlockColMajorLNMinimaxInit") {
                auto* layer = static_cast<ParBlockColMajorLNMinimaxInit*>(layer_ptr);
                return layer->generate_pt(*ctx_ptr, attrs.value("pt_idx", 0), attrs.value("bi", 0),
                                          attrs.value("bj", 0), attrs.value("g", 0));
            }
            if (op_class == "ParBlockColMajorLNGoldschmidt") {
                auto* layer = static_cast<ParBlockColMajorLNGoldschmidt*>(layer_ptr);
                return layer->generate_pt(*ctx_ptr, attrs.value("pt_idx", 0), attrs.value("bi", 0),
                                          attrs.value("bj", 0), attrs.value("g", 0));
            }
            if (op_class == "ParBlockColMajorLNAffine") {
                auto* layer = static_cast<ParBlockColMajorLNAffine*>(layer_ptr);
                return layer->generate_pt(*ctx_ptr, attrs.value("pt_idx", 0), attrs.value("bi", 0),
                                          attrs.value("bj", 0), attrs.value("g", 0));
            }
            if (op_class == "ParUpperDiagonalPoly") {
                auto* layer = static_cast<ParUpperDiagonalPoly*>(layer_ptr);
                if (type == "stockmeyer_coeff_pt")
                    return layer->generate_weight_pt_for_stockmeyer(*ctx_ptr, attrs.value("coeff_idx", 0),
                                                                    attrs.value("ct_idx", 0));
                throw std::runtime_error("encode_pt: unknown type for ParUpperDiagonalPoly: " + type);
            }
            if (op_class == "ParUpperDiagonalPolyMultCt") {
                auto* layer = static_cast<ParUpperDiagonalPolyMultCt*>(layer_ptr);
                uint32_t mb = attr_u32("mb", 0);
                uint32_t ct_local = attr_u32("ct_local", attr_u32("ct_idx", 0));
                uint32_t g = attr_u32("g", 0);
                if (type == "one_pt")
                    return layer->generate_one_pt(*ctx_ptr, mb, ct_local, g);
                if (type == "half_pt")
                    return layer->generate_half_pt(*ctx_ptr, mb, ct_local, g);
                throw std::runtime_error("encode_pt: unknown type for ParUpperDiagonalPolyMultCt: " + type);
            }
            if (op_class == "ParUpperDiagonalAddPt") {
                auto* layer = static_cast<ParUpperDiagonalAddPt*>(layer_ptr);
                if (type == "pt")
                    return layer->generate_pt(*ctx_ptr, attr_u32("ct_idx", 0));
                throw std::runtime_error("encode_pt: unknown type for ParUpperDiagonalAddPt: " + type);
            }
            if (op_class == "ParUpperDiagonalMultipleSquare") {
                auto* layer = static_cast<ParUpperDiagonalMultipleSquare*>(layer_ptr);
                if (type == "mask_pt")
                    return layer->generate_mask_pt(*ctx_ptr, attr_u32("ct_idx", 0));
                throw std::runtime_error("encode_pt: unknown type for ParUpperDiagonalMultipleSquare: " + type);
            }
            if (op_class == "ParUpperDiagonalHeadColSum") {
                auto* layer = static_cast<ParUpperDiagonalHeadColSum*>(layer_ptr);
                if (type == "mask_pt")
                    return layer->generate_mask_pt(*ctx_ptr, attr_u32("ct_idx", 0));
                throw std::runtime_error("encode_pt: unknown type for ParUpperDiagonalHeadColSum: " + type);
            }
            if (op_class == "ParUpperDiagonalInverseInit") {
                auto* layer = static_cast<ParUpperDiagonalInverseInit*>(layer_ptr);
                if (type == "two_pt")
                    return layer->generate_two_pt(*ctx_ptr, attr_u32("ct_idx", 0));
                throw std::runtime_error("encode_pt: unknown type for ParUpperDiagonalInverseInit: " + type);
            }
            if (op_class == "ParUpperDiagonalInverseIter") {
                auto* layer = static_cast<ParUpperDiagonalInverseIter*>(layer_ptr);
                uint32_t ct_idx = attr_u32("ct_idx", 0);
                if (type == "one_pt")
                    return layer->generate_one_pt(*ctx_ptr, ct_idx);
                if (type == "two_pt")
                    return layer->generate_two_pt(*ctx_ptr, ct_idx);
                throw std::runtime_error("encode_pt: unknown type for ParUpperDiagonalInverseIter: " + type);
            }
            if (op_class == "ParUpperDiagonalGELU") {
                auto* layer = static_cast<ParUpperDiagonalGELU*>(layer_ptr);
                if (type == "mask_pt")
                    return layer->generate_mask_pt(*ctx_ptr, attr_u32("ct_idx", 0));
                throw std::runtime_error("encode_pt: unknown type for ParUpperDiagonalGELU: " + type);
            }
            throw std::runtime_error("encode_pt: unknown op_class: " + op_class);
        }();

        local_data[self.output_nodes[0]->id] = std::make_shared<CkksPlaintext>(std::move(pt));
    };
}

template <typename ProcessorT>
static void run_generated_encode_pt_task(FheContext* context,
                                         const fs::path& server_dir,
                                         const vector<CxxVectorArgument>& cxx_args) {
    std::unordered_map<std::string, ExecutorFunc> executors;
    executors["encode_pt"] = make_block_col_major_encode_pt_executor();

    if constexpr (is_same_v<ProcessorT, ProcessorCpu>) {
        FheTaskCpu task(server_dir.string());
        task.bind_custom_executors(executors);
        task.run(context, cxx_args);
#ifdef INFERENCE_SDK_ENABLE_GPU
    } else if constexpr (is_same_v<ProcessorT, ProcessorGpu>) {
        FheTaskGpu task(server_dir.string());
        task.bind_custom_executors(executors);
        task.run(context, cxx_args);
#endif
    }
}

static void require_upper_diagonal_close(CkksContext& context,
                                         int output_level,
                                         const vector<CkksCiphertext>& out_cts,
                                         const Array<double, 2>& expected,
                                         Duo shape,
                                         Duo head_shape,
                                         uint32_t n_heads,
                                         const string& label) {
    Array<double, 2> actual =
        unpack_par_diagonal_output(context, output_level, out_cts, shape, head_shape, n_heads, false, false);
    print_double_message(actual.to_array_1d().data(), (label + "_actual").c_str(), 10);
    print_double_message(expected.to_array_1d().data(), (label + "_expected").c_str(), 10);
    auto comparison = compare(expected, actual);
    CAPTURE(label, comparison.max_error, comparison.max_abs, comparison.rmse, comparison.rms);
    REQUIRE(comparison.max_error < 5.0e-2 * comparison.max_abs);
    REQUIRE(comparison.rmse < 1.0e-2 * comparison.rms);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "par_upper_diagonal_poly_stockmeyer_hetero",
                               "[fhe_layers][par_upper_diagonal_poly][hetero]",
                               UpperDiagonalGraphProcessors) {
    const uint32_t n_prepad = 64;
    const uint32_t n_heads = 2;
    const uint32_t m_prepad = 64;
    const uint32_t total_cols = n_heads * m_prepad;
    const uint32_t order = 7;
    const int init_level = 4;
    Duo shape = {n_prepad, total_cols};
    Duo head_shape = {n_prepad, m_prepad};
    auto& small_resources = UpperDiagonalHeteroResources::get();
    auto& param = small_resources.param;
    auto& context = small_resources.context;

    fs::path server_dir = base_path / "CKKS_par_upper_diagonal_poly" / "stockmeyer" / ("order_" + to_string(order)) /
                          ("level_" + to_string(init_level)) / "server";
    if (!fs::exists(server_dir / "mega_ag.json"))
        return;

    auto X = gen_random_array<2>({n_prepad, total_cols}, 0.35);
    auto coeffs = gen_random_array<2>({static_cast<uint64_t>(order) + 1, static_cast<uint64_t>(total_cols)}, 0.08);

    auto layer =
        std::make_shared<ParUpperDiagonalPoly>(param, shape, head_shape, n_heads, init_level, coeffs.copy(), order);
    layer->prepare_weight_stockmeyer_lazy();
    Array<double, 2> expected = layer->run_plaintext(X);

    FeatureMatEncrypted X_enc(&context, init_level);
    X_enc.par_diagonal_pack(X, n_heads, head_shape, false, param.default_scale());

    vector<CkksCiphertext> in_cts, out_cts;
    vector<CustomData> layer_data;
    for (auto& ct : X_enc.data)
        in_cts.push_back(ct.copy());
    layer_data.emplace_back(static_cast<void*>(layer.get()));

    const int output_level = init_level - PolyBase::compute_stockmeyer_level_cost(static_cast<int>(order));
    uint32_t n_out = par_diagonal_total_cts(param, shape, head_shape, n_heads, false);
    for (uint32_t i = 0; i < n_out; i++)
        out_cts.emplace_back(context.parameter(), output_level);
        out_cts.back().set_scale(param.default_scale());

    vector<CxxVectorArgument> cxx_args = {
        {"input", &in_cts},
        {"_upper_poly_layer", &layer_data},
        {"output", &out_cts},
    };
    run_generated_encode_pt_task<TestType>(&context, server_dir, cxx_args);
    require_upper_diagonal_close(context, output_level, out_cts, expected, shape, head_shape, n_heads,
                                 "upper_poly_stockmeyer");
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "par_upper_diagonal_poly_mult_ct_hetero",
                               "[fhe_layers][par_upper_diagonal_poly_mult_ct][hetero]",
                               UpperDiagonalGraphProcessors) {
    const uint32_t n_prepad = 64;
    const uint32_t n_heads = 2;
    const uint32_t m_prepad = 64;
    const uint32_t total_cols = n_heads * m_prepad;
    const int init_level = 2;
    Duo shape = {n_prepad, total_cols};
    Duo head_shape = {n_prepad, m_prepad};
    auto& small_resources = UpperDiagonalHeteroResources::get();
    auto& param = small_resources.param;
    auto& context = small_resources.context;

    fs::path server_dir =
        base_path / "CKKS_par_upper_diagonal_poly_mult_ct" / ("level_" + to_string(init_level)) / "server";
    if (!fs::exists(server_dir / "mega_ag.json"))
        return;

    auto half_tanh = gen_random_array<2>({n_prepad, total_cols}, 0.25);
    auto X = gen_random_array<2>({n_prepad, total_cols}, 0.35);

    auto layer = std::make_shared<ParUpperDiagonalPolyMultCt>(param, shape, head_shape, n_heads, init_level);
    Array<double, 2> expected = layer->run_plaintext(half_tanh, X);

    FeatureMatEncrypted half_tanh_enc(&context, init_level);
    FeatureMatEncrypted X_enc(&context, init_level);
    half_tanh_enc.par_diagonal_pack(half_tanh, n_heads, head_shape, false, param.default_scale());
    X_enc.par_diagonal_pack(X, n_heads, head_shape, false, param.default_scale());

    vector<CkksCiphertext> half_tanh_cts, x_cts, out_cts;
    vector<CustomData> layer_data;
    for (auto& ct : half_tanh_enc.data)
        half_tanh_cts.push_back(ct.copy());
    for (auto& ct : X_enc.data)
        x_cts.push_back(ct.copy());
    layer_data.emplace_back(static_cast<void*>(layer.get()));

    const int output_level = init_level - 2;
    uint32_t n_out = par_diagonal_total_cts(param, shape, head_shape, n_heads, false);
    for (uint32_t i = 0; i < n_out; i++)
        out_cts.emplace_back(context.parameter(), output_level);
        out_cts.back().set_scale(param.default_scale());

    vector<CxxVectorArgument> cxx_args = {
        {"half_tanh_input", &half_tanh_cts},
        {"x_input", &x_cts},
        {"_poly_mult_ct_layer", &layer_data},
        {"output", &out_cts},
    };
    run_generated_encode_pt_task<TestType>(&context, server_dir, cxx_args);
    require_upper_diagonal_close(context, output_level, out_cts, expected, shape, head_shape, n_heads,
                                 "upper_poly_mult_ct");
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "par_upper_diagonal_add_pt_hetero",
                               "[fhe_layers][par_upper_diagonal_softmax][add_pt][hetero]",
                               UpperDiagonalGraphProcessors) {
    const uint32_t n_prepad = 64, n_heads = 2, m_prepad = 64, total_cols = n_heads * m_prepad;
    const int init_level = 1;
    const double value = -0.125;
    Duo shape = {n_prepad, total_cols};
    Duo head_shape = {n_prepad, m_prepad};
    auto& small_resources = UpperDiagonalHeteroResources::get();
    auto& param = small_resources.param;
    auto& context = small_resources.context;
    fs::path server_dir =
        base_path / "CKKS_par_upper_diagonal_softmax" / "add_pt" / ("level_" + to_string(init_level)) / "server";
    if (!fs::exists(server_dir / "mega_ag.json"))
        return;

    auto X = gen_random_array<2>({n_prepad, total_cols}, 0.5);
    auto layer = std::make_shared<ParUpperDiagonalAddPt>(param, shape, head_shape, n_heads, init_level, value);
    Array<double, 2> expected = layer->run_plaintext(X);

    FeatureMatEncrypted X_enc(&context, init_level);
    X_enc.par_diagonal_pack(X, n_heads, head_shape, false, param.default_scale());
    vector<CkksCiphertext> in_cts, out_cts;
    vector<CustomData> layer_data;
    for (auto& ct : X_enc.data)
        in_cts.push_back(ct.copy());
    layer_data.emplace_back(static_cast<void*>(layer.get()));
    uint32_t n_out = par_diagonal_total_cts(param, shape, head_shape, n_heads, false);
    for (uint32_t i = 0; i < n_out; i++)
        out_cts.emplace_back(context.parameter(), init_level);
        out_cts.back().set_scale(param.default_scale());

    vector<CxxVectorArgument> cxx_args = {
        {"input", &in_cts},
        {"_upper_add_pt_layer", &layer_data},
        {"output", &out_cts},
    };
    run_generated_encode_pt_task<TestType>(&context, server_dir, cxx_args);
    require_upper_diagonal_close(context, init_level, out_cts, expected, shape, head_shape, n_heads, "upper_add_pt");
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "par_upper_diagonal_multiple_square_hetero",
                               "[fhe_layers][par_upper_diagonal_softmax][multiple_square][hetero]",
                               UpperDiagonalGraphProcessors) {
    const uint32_t n_prepad = 64, n_heads = 2, m_prepad = 64, total_cols = n_heads * m_prepad;
    const int init_level = 2;
    Duo shape = {n_prepad, total_cols};
    Duo head_shape = {n_prepad, m_prepad};
    auto& small_resources = UpperDiagonalHeteroResources::get();
    auto& param = small_resources.param;
    auto& context = small_resources.context;
    fs::path server_dir = base_path / "CKKS_par_upper_diagonal_softmax" / "multiple_square" /
                          ("level_" + to_string(init_level)) / "server";
    if (!fs::exists(server_dir / "mega_ag.json"))
        return;

    auto X = gen_random_array<2>({n_prepad, total_cols}, 0.35);
    auto layer = std::make_shared<ParUpperDiagonalMultipleSquare>(param, shape, head_shape, n_heads, init_level);
    Array<double, 2> expected = layer->run_plaintext(X);

    FeatureMatEncrypted X_enc(&context, init_level);
    X_enc.par_diagonal_pack(X, n_heads, head_shape, false, param.default_scale());
    vector<CkksCiphertext> in_cts, out_cts;
    vector<CustomData> layer_data;
    for (auto& ct : X_enc.data)
        in_cts.push_back(ct.copy());
    layer_data.emplace_back(static_cast<void*>(layer.get()));
    const int output_level = init_level - 2;
    uint32_t n_out = par_diagonal_total_cts(param, shape, head_shape, n_heads, false);
    for (uint32_t i = 0; i < n_out; i++)
        out_cts.emplace_back(context.parameter(), output_level);
        out_cts.back().set_scale(param.default_scale());

    vector<CxxVectorArgument> cxx_args = {
        {"input", &in_cts},
        {"_upper_multiple_square_layer", &layer_data},
        {"output", &out_cts},
    };
    run_generated_encode_pt_task<TestType>(&context, server_dir, cxx_args);
    require_upper_diagonal_close(context, output_level, out_cts, expected, shape, head_shape, n_heads,
                                 "upper_multiple_square");
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "par_upper_diagonal_head_col_sum_hetero",
                               "[fhe_layers][par_upper_diagonal_softmax][head_col_sum][hetero]",
                               UpperDiagonalGraphProcessors) {
    const uint32_t n_prepad = 64, n_heads = 2, m_prepad = 64, total_cols = n_heads * m_prepad;
    const int init_level = 1;
    Duo shape = {n_prepad, total_cols};
    Duo head_shape = {n_prepad, m_prepad};
    auto& small_resources = UpperDiagonalHeteroResources::get();
    auto& param = small_resources.param;
    auto& context = small_resources.context;
    fs::path server_dir =
        base_path / "CKKS_par_upper_diagonal_softmax" / "head_col_sum" / ("level_" + to_string(init_level)) / "server";
    if (!fs::exists(server_dir / "mega_ag.json"))
        return;

    auto X = gen_random_array<2>({n_prepad, total_cols}, 0.2);
    auto layer = std::make_shared<ParUpperDiagonalHeadColSum>(param, shape, head_shape, n_heads, init_level);
    Array<double, 2> expected = layer->run_plaintext(X);

    FeatureMatEncrypted X_enc(&context, init_level);
    X_enc.par_diagonal_pack(X, n_heads, head_shape, false, param.default_scale());
    vector<CkksCiphertext> in_cts, out_cts;
    vector<CustomData> layer_data;
    for (auto& ct : X_enc.data)
        in_cts.push_back(ct.copy());
    layer_data.emplace_back(static_cast<void*>(layer.get()));
    const int output_level = init_level - 1;
    uint32_t n_out = par_diagonal_total_cts(param, shape, head_shape, n_heads, false);
    for (uint32_t i = 0; i < n_out; i++)
        out_cts.emplace_back(context.parameter(), output_level);
        out_cts.back().set_scale(param.default_scale());

    vector<CxxVectorArgument> cxx_args = {
        {"input", &in_cts},
        {"_upper_head_col_sum_layer", &layer_data},
        {"output", &out_cts},
    };
    run_generated_encode_pt_task<TestType>(&context, server_dir, cxx_args);
    require_upper_diagonal_close(context, output_level, out_cts, expected, shape, head_shape, n_heads,
                                 "upper_head_col_sum");
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "par_upper_diagonal_inverse_init_hetero",
                               "[fhe_layers][par_upper_diagonal_softmax][inverse_init][hetero]",
                               UpperDiagonalGraphProcessors) {
    const uint32_t n_prepad = 64, n_heads = 2, m_prepad = 64, total_cols = n_heads * m_prepad;
    const int init_level = 1;
    Duo shape = {n_prepad, total_cols};
    Duo head_shape = {n_prepad, m_prepad};
    auto& small_resources = UpperDiagonalHeteroResources::get();
    auto& param = small_resources.param;
    auto& context = small_resources.context;
    fs::path server_dir =
        base_path / "CKKS_par_upper_diagonal_softmax" / "inverse_init" / ("level_" + to_string(init_level)) / "server";
    if (!fs::exists(server_dir / "mega_ag.json"))
        return;

    auto B = gen_random_array<2>({n_prepad, total_cols}, 0.35);
    auto layer = std::make_shared<ParUpperDiagonalInverseInit>(param, shape, head_shape, n_heads, init_level);
    Array<double, 2> expected = layer->run_plaintext(B);

    FeatureMatEncrypted B_enc(&context, init_level);
    B_enc.par_diagonal_pack(B, n_heads, head_shape, false, param.default_scale());
    vector<CkksCiphertext> in_cts, out_cts;
    vector<CustomData> layer_data;
    for (auto& ct : B_enc.data)
        in_cts.push_back(ct.copy());
    layer_data.emplace_back(static_cast<void*>(layer.get()));
    uint32_t n_out = par_diagonal_total_cts(param, shape, head_shape, n_heads, false);
    for (uint32_t i = 0; i < n_out; i++)
        out_cts.emplace_back(context.parameter(), init_level);
        out_cts.back().set_scale(param.default_scale());

    vector<CxxVectorArgument> cxx_args = {
        {"input", &in_cts},
        {"_upper_inverse_init_layer", &layer_data},
        {"output", &out_cts},
    };
    run_generated_encode_pt_task<TestType>(&context, server_dir, cxx_args);
    require_upper_diagonal_close(context, init_level, out_cts, expected, shape, head_shape, n_heads,
                                 "upper_inverse_init");
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "par_upper_diagonal_inverse_iter_hetero",
                               "[fhe_layers][par_upper_diagonal_softmax][inverse_iter][hetero]",
                               UpperDiagonalGraphProcessors) {
    const uint32_t n_prepad = 64, n_heads = 2, m_prepad = 64, total_cols = n_heads * m_prepad;
    const int init_level = 2;
    Duo shape = {n_prepad, total_cols};
    Duo head_shape = {n_prepad, m_prepad};
    auto& small_resources = UpperDiagonalHeteroResources::get();
    auto& param = small_resources.param;
    auto& context = small_resources.context;
    fs::path server_dir =
        base_path / "CKKS_par_upper_diagonal_softmax" / "inverse_iter" / ("level_" + to_string(init_level)) / "server";
    if (!fs::exists(server_dir / "mega_ag.json"))
        return;

    auto A = gen_random_array<2>({n_prepad, total_cols}, 0.25);
    auto B = gen_random_array<2>({n_prepad, total_cols}, 0.2);
    auto layer = std::make_shared<ParUpperDiagonalInverseIter>(param, shape, head_shape, n_heads, init_level);
    Array<double, 2> expected = layer->run_plaintext(A, B);

    FeatureMatEncrypted A_enc(&context, init_level);
    FeatureMatEncrypted B_enc(&context, init_level);
    A_enc.par_diagonal_pack(A, n_heads, head_shape, false, param.default_scale());
    B_enc.par_diagonal_pack(B, n_heads, head_shape, false, param.default_scale());
    vector<CkksCiphertext> a_cts, b_cts, out_cts;
    vector<CustomData> layer_data;
    for (auto& ct : A_enc.data)
        a_cts.push_back(ct.copy());
    for (auto& ct : B_enc.data)
        b_cts.push_back(ct.copy());
    layer_data.emplace_back(static_cast<void*>(layer.get()));
    const int output_level = init_level - 2;
    uint32_t n_out = par_diagonal_total_cts(param, shape, head_shape, n_heads, false);
    for (uint32_t i = 0; i < n_out; i++)
        out_cts.emplace_back(context.parameter(), output_level);
        out_cts.back().set_scale(param.default_scale());

    vector<CxxVectorArgument> cxx_args = {
        {"a_input", &a_cts},
        {"b_input", &b_cts},
        {"_upper_inverse_iter_layer", &layer_data},
        {"output", &out_cts},
    };
    run_generated_encode_pt_task<TestType>(&context, server_dir, cxx_args);
    require_upper_diagonal_close(context, output_level, out_cts, expected, shape, head_shape, n_heads,
                                 "upper_inverse_iter");
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "par_upper_diagonal_gelu_hetero",
                               "[fhe_layers][par_upper_diagonal_softmax][gelu][hetero]",
                               UpperDiagonalGraphProcessors) {
    const uint32_t n_prepad = 64, n_heads = 2, m_prepad = 64, total_cols = n_heads * m_prepad;
    const int init_level = 2;
    Duo shape = {n_prepad, total_cols};
    Duo head_shape = {n_prepad, m_prepad};
    auto& small_resources = UpperDiagonalHeteroResources::get();
    auto& param = small_resources.param;
    auto& context = small_resources.context;
    fs::path server_dir =
        base_path / "CKKS_par_upper_diagonal_softmax" / "gelu" / ("level_" + to_string(init_level)) / "server";
    if (!fs::exists(server_dir / "mega_ag.json"))
        return;

    auto A = gen_random_array<2>({n_prepad, total_cols}, 0.3);
    auto B = gen_random_array<2>({n_prepad, total_cols}, 0.25);
    auto layer = std::make_shared<ParUpperDiagonalGELU>(param, shape, head_shape, n_heads, init_level);
    Array<double, 2> expected = layer->run_plaintext(A, B);

    FeatureMatEncrypted A_enc(&context, init_level);
    FeatureMatEncrypted B_enc(&context, init_level);
    A_enc.par_diagonal_pack(A, n_heads, head_shape, false, param.default_scale());
    B_enc.par_diagonal_pack(B, n_heads, head_shape, false, param.default_scale());
    vector<CkksCiphertext> a_cts, b_cts, out_cts;
    vector<CustomData> layer_data;
    for (auto& ct : A_enc.data)
        a_cts.push_back(ct.copy());
    for (auto& ct : B_enc.data)
        b_cts.push_back(ct.copy());
    layer_data.emplace_back(static_cast<void*>(layer.get()));
    const int output_level = init_level - 2;
    uint32_t n_out = par_diagonal_total_cts(param, shape, head_shape, n_heads, false);
    for (uint32_t i = 0; i < n_out; i++)
        out_cts.emplace_back(context.parameter(), output_level);
        out_cts.back().set_scale(param.default_scale());

    vector<CxxVectorArgument> cxx_args = {
        {"a_input", &a_cts},
        {"b_input", &b_cts},
        {"_upper_gelu_layer", &layer_data},
        {"output", &out_cts},
    };
    run_generated_encode_pt_task<TestType>(&context, server_dir, cxx_args);
    require_upper_diagonal_close(context, output_level, out_cts, expected, shape, head_shape, n_heads, "upper_gelu");
}

template <typename T>
static void run_block_col_major_e2e_test(HeteroFixture<T>& fixture,
                                         const fs::path& server_dir,
                                         const vector<CxxVectorArgument>& cxx_args,
                                         const Array<double, 2>& ref_output,
                                         Duo out_shape,
                                         uint32_t out_d,
                                         uint32_t out_n_heads,
                                         bool is_par,
                                         vector<CkksCiphertext>& out_cts) {
    auto& res = SharedHeteroResources::get();

    std::unordered_map<std::string, ExecutorFunc> executors;
    executors["encode_pt"] = make_block_col_major_encode_pt_executor();

    auto t0 = std::chrono::high_resolution_clock::now();
    if constexpr (is_same_v<T, ProcessorCpu>) {
        FheTaskCpu task(server_dir.string());
        task.bind_custom_executors(executors);
        task.run(&res.context, cxx_args);
#ifdef INFERENCE_SDK_ENABLE_GPU
    } else if constexpr (is_same_v<T, ProcessorGpu>) {
        FheTaskGpu task(server_dir.string());
        task.bind_custom_executors(executors);
        task.run(&res.context, cxx_args);
#endif
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "\n[block_col_major_e2e] elapsed=" << std::fixed << std::setprecision(2) << ms << " ms\n";

    FeatureMatEncrypted out_enc(&res.context, 0);
    out_enc.data.clear();
    for (auto& ct : out_cts)
        out_enc.data.push_back(std::move(ct));
    out_enc.head_shape = out_shape;
    out_enc.shape = {out_shape[0], static_cast<uint32_t>(ref_output.get_shape()[1])};
    out_enc.matmul_block_size = out_d;
    Array<double, 2> actual({1, 1});
    if (is_par) {
        actual = out_enc.par_block_col_major_unpack(out_shape[0], out_shape[1], out_d, out_n_heads);
    } else {
        actual = out_enc.block_col_major_unpack(out_shape[0], out_shape[1], out_d);
    }

    double rel_err = e2e_max_rel_error(ref_output, actual);
    std::cout << "  max_rel_error = " << std::scientific << rel_err << "\n";
    REQUIRE(rel_err < 0.05);
}
TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "single_par_cpmm_square", "[block_col_major_e2e]", HeteroProcessors) {
    auto& res = SharedHeteroResources::get();

    struct Cfg {
        uint32_t m, n_heads, head_dim;
        int level;
    };
    vector<Cfg> configs = {
        {53, 3, 16, 2},
        {83, 3, 64, 2},
    };

    FOR_EACH_SECTION(auto& cfg, configs,
                     "m=" + to_string(cfg.m) + "_heads=" + to_string(cfg.n_heads) + "_dim=" + to_string(cfg.head_dim)) {
        uint32_t d = cfg.head_dim;
        uint32_t total_dim = cfg.n_heads * cfg.head_dim;

        fs::path server_dir =
            base_path / "CKKS_par_cpmm_square" /
            ("m_" + to_string(cfg.m) + "_heads_" + to_string(cfg.n_heads) + "_dim_" + to_string(cfg.head_dim)) /
            ("level_" + to_string(cfg.level)) / "server";
        if (!fs::exists(server_dir / "mega_ag.json"))
            return;

        auto W = gen_random_array<2>({total_dim, total_dim}, 0.1);
        auto A = gen_random_array<2>({cfg.m, total_dim}, 0.5);

        auto layer_ptr =
            std::make_shared<ParBlockColMajorCPMM>(res.param, Duo{cfg.m, cfg.head_dim}, W, d, cfg.n_heads, cfg.level);
        layer_ptr->precompute_diagonals();
        auto ref_output = layer_ptr->run_plaintext(A);

        FeatureMatEncrypted A_enc(&res.context, cfg.level);
        A_enc.par_block_col_major_pack(A, d, cfg.n_heads, d, res.param.default_scale());

        static vector<CkksCiphertext> in_cts, out_cts;
        static vector<CustomData> layer_data;
        in_cts.clear();
        out_cts.clear();
        layer_data.clear();

        for (auto& ct : A_enc.data)
            in_cts.push_back(ct.copy());
        layer_data.emplace_back(static_cast<void*>(layer_ptr.get()));

        uint32_t num_block_rows_A = div_ceil(cfg.m, d);
        uint32_t n_h_padded = 1;
        while (n_h_padded < cfg.n_heads)
            n_h_padded <<= 1;
        uint32_t n_slot = res.param.max_slots();
        if (n_slot < n_h_padded * d * d && n_slot / (d * d) == 1) {
            n_h_padded = cfg.n_heads;
        }
        uint32_t G = (n_slot >= n_h_padded * d * d) ? 1 : n_h_padded / (n_slot / (d * d));
        uint32_t n_out = num_block_rows_A * G;

        for (uint32_t i = 0; i < n_out; i++)
            out_cts.emplace_back(res.context.parameter(), cfg.level - 2);
            out_cts.back().set_scale(res.param.default_scale());

        vector<CxxVectorArgument> cxx_args = {
            {"input", &in_cts},
            {"_cpmm_layer", &layer_data},
            {"output", &out_cts},
        };
        run_block_col_major_e2e_test(*this, server_dir, cxx_args, ref_output, {cfg.m, cfg.head_dim}, d, cfg.n_heads,
                                     true, out_cts);
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "single_par_cpmm_square_with_bias",
                               "[block_col_major_e2e]",
                               HeteroProcessors) {
    auto& res = SharedHeteroResources::get();

    struct Cfg {
        uint32_t m, n_heads, head_dim, W_cols;
        int level;
    };
    vector<Cfg> configs = {
        {53, 3, 16, 35, 2},  // W_cols=35 < total_dim=48, non-square but K_row=K_col=1
    };

    FOR_EACH_SECTION(auto& cfg, configs,
                     "m=" + to_string(cfg.m) + "_heads=" + to_string(cfg.n_heads) + "_dim=" + to_string(cfg.head_dim) +
                         "_wcols=" + to_string(cfg.W_cols)) {
        uint32_t d = cfg.head_dim;
        uint32_t total_dim = cfg.n_heads * cfg.head_dim;

        fs::path server_dir = base_path / "CKKS_par_cpmm_square_with_bias" /
                              ("m_" + to_string(cfg.m) + "_heads_" + to_string(cfg.n_heads) + "_dim_" +
                               to_string(cfg.head_dim) + "_wcols_" + to_string(cfg.W_cols)) /
                              ("level_" + to_string(cfg.level)) / "server";
        if (!fs::exists(server_dir / "mega_ag.json"))
            return;

        auto W = gen_random_array<2>({total_dim, cfg.W_cols}, 0.1);
        auto A = gen_random_array<2>({cfg.m, total_dim}, 0.5);
        auto bias = gen_random_array<1>({cfg.W_cols}, 0.3);

        auto layer_ptr = std::make_shared<ParBlockColMajorCPMM>(res.param, Duo{cfg.m, cfg.head_dim}, W, d, cfg.n_heads,
                                                                cfg.level, std::move(bias));
        layer_ptr->precompute_diagonals();
        auto ref_output = layer_ptr->run_plaintext(A);

        FeatureMatEncrypted A_enc(&res.context, cfg.level);
        A_enc.par_block_col_major_pack(A, d, cfg.n_heads, d, res.param.default_scale());

        static vector<CkksCiphertext> in_cts, out_cts;
        static vector<CustomData> layer_data;
        in_cts.clear();
        out_cts.clear();
        layer_data.clear();

        for (auto& ct : A_enc.data)
            in_cts.push_back(ct.copy());
        layer_data.emplace_back(static_cast<void*>(layer_ptr.get()));

        uint32_t num_block_rows_A = div_ceil(cfg.m, d);
        uint32_t n_h_padded = 1;
        while (n_h_padded < cfg.n_heads)
            n_h_padded <<= 1;
        uint32_t n_slot = res.param.max_slots();
        if (n_slot < n_h_padded * d * d && n_slot / (d * d) == 1) {
            n_h_padded = cfg.n_heads;
        }
        uint32_t G = (n_slot >= n_h_padded * d * d) ? 1 : n_h_padded / (n_slot / (d * d));
        uint32_t n_out = num_block_rows_A * G;

        for (uint32_t i = 0; i < n_out; i++)
            out_cts.emplace_back(res.context.parameter(), cfg.level - 2);
            out_cts.back().set_scale(res.param.default_scale());

        vector<CxxVectorArgument> cxx_args = {
            {"input", &in_cts},
            {"_cpmm_layer", &layer_data},
            {"output", &out_cts},
        };
        run_block_col_major_e2e_test(*this, server_dir, cxx_args, ref_output, {cfg.m, cfg.head_dim}, d, cfg.n_heads,
                                     true, out_cts);
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "single_par_cpmm_expand_with_bias",
                               "[block_col_major_e2e]",
                               HeteroProcessors) {
    auto& res = SharedHeteroResources::get();

    // m=53, n_heads=3, head_dim=16, total_dim=48, W_cols=100 (non-divisible, K_col=ceil(100/48)=3)
    const uint32_t m = 53, n_heads = 3, head_dim = 16, d = head_dim;
    const uint32_t total_dim = n_heads * head_dim;
    const uint32_t W_cols = 100;
    const int level = 2;
    const uint32_t K_col = div_ceil(W_cols, total_dim);  // 3

    fs::path server_dir = base_path / "CKKS_par_cpmm_expand_with_bias" /
                          ("m_" + to_string(m) + "_heads_" + to_string(n_heads) + "_dim_" + to_string(head_dim) +
                           "_wcols_" + to_string(W_cols)) /
                          ("level_" + to_string(level)) / "server";
    if (!fs::exists(server_dir / "mega_ag.json"))
        return;

    auto W = gen_random_array<2>({total_dim, W_cols}, 0.1);
    auto A = gen_random_array<2>({m, total_dim}, 0.5);
    auto bias = gen_random_array<1>({W_cols}, 0.3);

    auto layer_ptr =
        std::make_shared<ParBlockColMajorCPMM>(res.param, Duo{m, head_dim}, W, d, n_heads, level, std::move(bias));
    layer_ptr->precompute_diagonals();
    auto ref_output = layer_ptr->run_plaintext(A);

    FeatureMatEncrypted A_enc(&res.context, level);
    A_enc.par_block_col_major_pack(A, d, n_heads, head_dim, res.param.default_scale());

    static vector<CkksCiphertext> in_cts, out_cts;
    static vector<CustomData> layer_data;
    in_cts.clear();
    out_cts.clear();
    layer_data.clear();

    for (auto& ct : A_enc.data)
        in_cts.push_back(ct.copy());
    layer_data.emplace_back(static_cast<void*>(layer_ptr.get()));

    uint32_t num_block_rows_A = div_ceil(m, d);
    uint32_t n_h_padded = 1;
    while (n_h_padded < n_heads)
        n_h_padded <<= 1;
    uint32_t n_slot = res.param.max_slots();
    if (n_slot < n_h_padded * d * d && n_slot / (d * d) == 1) {
        n_h_padded = n_heads;
    }
    uint32_t G = (n_slot >= n_h_padded * d * d) ? 1 : n_h_padded / (n_slot / (d * d));
    uint32_t n_out = K_col * num_block_rows_A * G;

    for (uint32_t i = 0; i < n_out; i++)
        out_cts.emplace_back(res.context.parameter(), level - 2);
        out_cts.back().set_scale(res.param.default_scale());

    vector<CxxVectorArgument> cxx_args = {
        {"input", &in_cts},
        {"_cpmm_layer", &layer_data},
        {"output", &out_cts},
    };
    run_block_col_major_e2e_test(*this, server_dir, cxx_args, ref_output, {m, head_dim}, d, n_heads, true, out_cts);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "single_par_cpmm_reduce_with_bias",
                               "[block_col_major_e2e]",
                               HeteroProcessors) {
    auto& res = SharedHeteroResources::get();

    // m=53, n_heads=3, head_dim=16, total_dim=48, W_rows=100 (non-divisible, K_row=ceil(100/48)=3)
    const uint32_t m = 53, n_heads = 3, head_dim = 16, d = head_dim;
    const uint32_t total_dim = n_heads * head_dim;
    const uint32_t W_rows = 100;
    const int level = 2;
    const uint32_t K_row = div_ceil(W_rows, total_dim);  // 3

    fs::path server_dir = base_path / "CKKS_par_cpmm_reduce_with_bias" /
                          ("m_" + to_string(m) + "_heads_" + to_string(n_heads) + "_dim_" + to_string(head_dim) +
                           "_wrows_" + to_string(W_rows)) /
                          ("level_" + to_string(level)) / "server";
    if (!fs::exists(server_dir / "mega_ag.json"))
        return;

    auto W = gen_random_array<2>({W_rows, total_dim}, 0.1);
    // REDUCE input has K_row megablocks: m × (K_row * total_dim) padded columns
    auto A = gen_random_array<2>({m, W_rows}, 0.5);
    auto bias = gen_random_array<1>({total_dim}, 0.3);

    auto layer_ptr =
        std::make_shared<ParBlockColMajorCPMM>(res.param, Duo{m, head_dim}, W, d, n_heads, level, std::move(bias));
    layer_ptr->precompute_diagonals();
    auto ref_output = layer_ptr->run_plaintext(A);

    FeatureMatEncrypted A_enc(&res.context, level);
    A_enc.par_block_col_major_pack(A, d, n_heads, head_dim, res.param.default_scale());

    static vector<CkksCiphertext> in_cts, out_cts;
    static vector<CustomData> layer_data;
    in_cts.clear();
    out_cts.clear();
    layer_data.clear();

    for (auto& ct : A_enc.data)
        in_cts.push_back(ct.copy());
    layer_data.emplace_back(static_cast<void*>(layer_ptr.get()));

    uint32_t num_block_rows_A = div_ceil(m, d);
    uint32_t n_h_padded = 1;
    while (n_h_padded < n_heads)
        n_h_padded <<= 1;
    uint32_t n_slot = res.param.max_slots();
    if (n_slot < n_h_padded * d * d && n_slot / (d * d) == 1) {
        n_h_padded = n_heads;
    }
    uint32_t G = (n_slot >= n_h_padded * d * d) ? 1 : n_h_padded / (n_slot / (d * d));
    uint32_t n_out = num_block_rows_A * G;  // REDUCE: single output megablock

    for (uint32_t i = 0; i < n_out; i++)
        out_cts.emplace_back(res.context.parameter(), level - 2);
        out_cts.back().set_scale(res.param.default_scale());

    vector<CxxVectorArgument> cxx_args = {
        {"input", &in_cts},
        {"_cpmm_layer", &layer_data},
        {"output", &out_cts},
    };
    run_block_col_major_e2e_test(*this, server_dir, cxx_args, ref_output, {m, head_dim}, d, n_heads, true, out_cts);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "single_par_attention", "[block_col_major_e2e]", HeteroProcessors) {
    auto& res = SharedHeteroResources::get();
    const uint32_t seq_len = 53, n_heads = 3, head_dim = 16, d = head_dim;
    const int init_level = 7;

    fs::path server_dir = base_path / ("CKKS_par_attention_seq53_heads3_dim16") / "level_7" / "server";
    if (!fs::exists(server_dir / "mega_ag.json"))
        return;

    uint32_t total_dim = n_heads * head_dim;
    auto Q_mat = gen_random_array<2>({seq_len, total_dim}, 1.0);
    auto K_mat = gen_random_array<2>({seq_len, total_dim}, 0.1);
    auto V_mat = gen_random_array<2>({seq_len, total_dim}, 1.0);

    // Reference: per-head Q * K^T * V
    Array<double, 2> expected({(uint64_t)seq_len, (uint64_t)total_dim});
    for (uint32_t h = 0; h < n_heads; h++)
        for (uint32_t i = 0; i < seq_len; i++)
            for (uint32_t j = 0; j < head_dim; j++) {
                double sum = 0.0;
                for (uint32_t k1 = 0; k1 < seq_len; k1++) {
                    double attn = 0.0;
                    for (uint32_t k2 = 0; k2 < head_dim; k2++)
                        attn += Q_mat.get(i, h * head_dim + k2) * K_mat.get(k1, h * head_dim + k2);
                    sum += attn * V_mat.get(k1, h * head_dim + j);
                }
                expected.set(i, h * head_dim + j, sum);
            }

    // Create layers
    auto kt_transpose =
        std::make_shared<ParBlockColMajorTranspose>(res.param, Duo{seq_len, head_dim}, d, n_heads, init_level);

    auto ccmm_qkt = std::make_shared<ParBlockColMajorCCMM>(res.param, Duo{seq_len, head_dim}, Duo{head_dim, seq_len}, d,
                                                           n_heads, init_level - 1);
    ccmm_qkt->precompute_diagonals();

    auto ccmm_attnv = std::make_shared<ParBlockColMajorCCMM>(res.param, Duo{seq_len, seq_len}, Duo{seq_len, head_dim},
                                                             d, n_heads, init_level - 4);
    ccmm_attnv->precompute_diagonals();

    // Pack Q, K, V
    FeatureMatEncrypted Q_enc(&res.context, init_level);
    Q_enc.par_block_col_major_pack(Q_mat, d, n_heads, d, res.param.default_scale());
    FeatureMatEncrypted K_enc(&res.context, init_level);
    K_enc.par_block_col_major_pack(K_mat, d, n_heads, d, res.param.default_scale());
    FeatureMatEncrypted V_enc(&res.context, init_level);
    V_enc.par_block_col_major_pack(V_mat, d, n_heads, d, res.param.default_scale());

    static vector<CkksCiphertext> Q_cts, K_cts, V_cts, out_cts;
    static vector<CustomData> kt_data, qkt_data, attnv_data;
    Q_cts.clear();
    K_cts.clear();
    V_cts.clear();
    out_cts.clear();
    kt_data.clear();
    qkt_data.clear();
    attnv_data.clear();

    for (auto& ct : Q_enc.data)
        Q_cts.push_back(ct.copy());
    for (auto& ct : K_enc.data)
        K_cts.push_back(ct.copy());
    for (auto& ct : V_enc.data)
        V_cts.push_back(ct.copy());

    kt_data.emplace_back(static_cast<void*>(kt_transpose.get()));
    qkt_data.emplace_back(static_cast<void*>(ccmm_qkt.get()));
    attnv_data.emplace_back(static_cast<void*>(ccmm_attnv.get()));

    // Output: seq_len × head_dim per head, level = init_level - 7 = 0
    uint32_t n_out = div_ceil(seq_len, d) * div_ceil(head_dim, d);
    for (uint32_t i = 0; i < n_out; i++)
        out_cts.emplace_back(res.context.parameter(), init_level - 7);
        out_cts.back().set_scale(res.param.default_scale());

    vector<CxxVectorArgument> cxx_args = {
        {"Q", &Q_cts},
        {"K", &K_cts},
        {"V", &V_cts},
        {"_kt_transpose", &kt_data},
        {"_qkt_ccmm", &qkt_data},
        {"_attnv_ccmm", &attnv_data},
        {"output", &out_cts},
    };
    run_block_col_major_e2e_test(*this, server_dir, cxx_args, expected, {seq_len, head_dim}, d, n_heads, true, out_cts);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "single_par_full_attention", "[block_col_major_e2e]", HeteroProcessors) {
    auto& res = SharedHeteroResources::get();
    const uint32_t seq_len = 197, n_heads = 3, head_dim = 64, d = head_dim;
    const uint32_t total_dim = n_heads * head_dim;
    const int init_level = 9;

    fs::path server_dir = base_path / "CKKS_par_full_attention_seq197_heads3_dim64" / "level_9" / "server";
    if (!fs::exists(server_dir / "mega_ag.json"))
        return;

    // Random input and weight matrices
    auto X_mat = gen_random_array<2>({seq_len, total_dim}, 0.5);
    auto W_Q = gen_random_array<2>({total_dim, total_dim}, 0.1);
    auto W_K = gen_random_array<2>({total_dim, total_dim}, 0.1);
    auto W_V = gen_random_array<2>({total_dim, total_dim}, 0.1);

    // Create 7 layer objects (3 CPMM + 1 Transpose + 2 CCMM)
    auto cpmm_q =
        std::make_shared<ParBlockColMajorCPMM>(res.param, Duo{seq_len, head_dim}, W_Q, d, n_heads, init_level);
    cpmm_q->precompute_diagonals();
    auto cpmm_k =
        std::make_shared<ParBlockColMajorCPMM>(res.param, Duo{seq_len, head_dim}, W_K, d, n_heads, init_level);
    cpmm_k->precompute_diagonals();
    auto cpmm_v =
        std::make_shared<ParBlockColMajorCPMM>(res.param, Duo{seq_len, head_dim}, W_V, d, n_heads, init_level);
    cpmm_v->precompute_diagonals();

    // Reference: Q=X*W_Q, K=X*W_K, V=X*W_V, then per-head Q*K^T*V
    auto Q_mat = cpmm_q->run_plaintext(X_mat);
    auto K_mat = cpmm_k->run_plaintext(X_mat);
    auto V_mat = cpmm_v->run_plaintext(X_mat);

    Array<double, 2> expected({(uint64_t)seq_len, (uint64_t)total_dim});
    for (uint32_t h = 0; h < n_heads; h++)
        for (uint32_t i = 0; i < seq_len; i++)
            for (uint32_t j = 0; j < head_dim; j++) {
                double sum = 0.0;
                for (uint32_t k1 = 0; k1 < seq_len; k1++) {
                    double attn = 0.0;
                    for (uint32_t k2 = 0; k2 < head_dim; k2++)
                        attn += Q_mat.get(i, h * head_dim + k2) * K_mat.get(k1, h * head_dim + k2);
                    sum += attn * V_mat.get(k1, h * head_dim + j);
                }
                expected.set(i, h * head_dim + j, sum);
            }

    auto kt_transpose = std::make_shared<ParBlockColMajorTranspose>(res.param, Duo{seq_len, head_dim}, d, n_heads,
                                                                    init_level - 2);  // K at level 7

    auto ccmm_qkt =
        std::make_shared<ParBlockColMajorCCMM>(res.param, Duo{seq_len, head_dim}, Duo{head_dim, seq_len}, d, n_heads,
                                               init_level - 3);  // Q',K^T at level 6
    ccmm_qkt->precompute_diagonals();

    auto ccmm_attnv =
        std::make_shared<ParBlockColMajorCCMM>(res.param, Duo{seq_len, seq_len}, Duo{seq_len, head_dim}, d, n_heads,
                                               init_level - 6);  // attn,V' at level 3
    ccmm_attnv->precompute_diagonals();

    // Pack input X
    FeatureMatEncrypted X_enc(&res.context, init_level);
    X_enc.par_block_col_major_pack(X_mat, d, n_heads, d, res.param.default_scale());

    static vector<CkksCiphertext> X_cts, out_cts;
    static vector<CustomData> q_data, k_data, v_data, kt_data, qkt_data, attnv_data;
    X_cts.clear();
    out_cts.clear();
    q_data.clear();
    k_data.clear();
    v_data.clear();
    kt_data.clear();
    qkt_data.clear();
    attnv_data.clear();

    for (auto& ct : X_enc.data)
        X_cts.push_back(ct.copy());

    q_data.emplace_back(static_cast<void*>(cpmm_q.get()));
    k_data.emplace_back(static_cast<void*>(cpmm_k.get()));
    v_data.emplace_back(static_cast<void*>(cpmm_v.get()));
    kt_data.emplace_back(static_cast<void*>(kt_transpose.get()));
    qkt_data.emplace_back(static_cast<void*>(ccmm_qkt.get()));
    attnv_data.emplace_back(static_cast<void*>(ccmm_attnv.get()));

    // Output: seq_len × head_dim per head, level = 0
    uint32_t num_block_rows_out = div_ceil(seq_len, d);
    uint32_t n_h_padded = 1;
    while (n_h_padded < n_heads)
        n_h_padded <<= 1;
    uint32_t n_slot = res.param.max_slots();
    if (n_slot < n_h_padded * d * d && n_slot / (d * d) == 1) {
        n_h_padded = n_heads;
    }
    uint32_t G = (n_slot >= n_h_padded * d * d) ? 1 : n_h_padded / (n_slot / (d * d));
    uint32_t n_out = num_block_rows_out * G;
    for (uint32_t i = 0; i < n_out; i++)
        out_cts.emplace_back(res.context.parameter(), 0);
        out_cts.back().set_scale(res.param.default_scale());

    vector<CxxVectorArgument> cxx_args = {
        {"X", &X_cts},
        {"_cpmm_q", &q_data},
        {"_cpmm_k", &k_data},
        {"_cpmm_v", &v_data},
        {"_kt_transpose", &kt_data},
        {"_qkt_ccmm", &qkt_data},
        {"_attnv_ccmm", &attnv_data},
        {"output", &out_cts},
    };
    run_block_col_major_e2e_test(*this, server_dir, cxx_args, expected, {seq_len, head_dim}, d, n_heads, true, out_cts);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "single_par_cpmm_expand_reduce",
                               "[block_col_major_e2e]",
                               HeteroProcessors) {
    const uint32_t m = 53, n_heads = 3, head_dim = 16, K_factor = 4, d = head_dim;
    const uint32_t total_dim = n_heads * head_dim;
    const uint32_t expanded_dim = K_factor * total_dim;
    const int init_level = 4;

    fs::path server_dir = base_path / "CKKS_par_cpmm_expand_reduce_m53_heads3_dim16_K4" / "level_4" / "server";
    if (!fs::exists(server_dir / "mega_ag.json"))
        return;

    auto A_mat = gen_random_array<2>({m, total_dim}, 0.5);
    auto W1_mat = gen_random_array<2>({total_dim, expanded_dim}, 0.1);
    auto W2_mat = gen_random_array<2>({expanded_dim, total_dim}, 0.1);

    // Reference: (A @ W1) @ W2
    Array<double, 2> expected({(uint64_t)m, (uint64_t)total_dim});
    std::vector<double> mid(m * expanded_dim, 0.0);
    for (uint32_t i = 0; i < m; i++)
        for (uint32_t j = 0; j < expanded_dim; j++) {
            double s = 0;
            for (uint32_t k = 0; k < total_dim; k++)
                s += A_mat.get(i, k) * W1_mat.get(k, j);
            mid[i * expanded_dim + j] = s;
        }
    for (uint32_t i = 0; i < m; i++)
        for (uint32_t j = 0; j < total_dim; j++) {
            double s = 0;
            for (uint32_t k = 0; k < expanded_dim; k++)
                s += mid[i * expanded_dim + k] * W2_mat.get(k, j);
            expected.set(i, j, s);
        }

    // Create layers
    auto expand_ptr =
        std::make_shared<ParBlockColMajorCPMM>(this->param, Duo{m, head_dim}, W1_mat, d, n_heads, init_level);
    expand_ptr->precompute_diagonals();

    int mid_level = init_level - 2;
    auto reduce_ptr =
        std::make_shared<ParBlockColMajorCPMM>(this->param, Duo{m, head_dim}, W2_mat, d, n_heads, mid_level);
    reduce_ptr->precompute_diagonals();

    // Pack input
    FeatureMatEncrypted A_enc(&this->context, init_level);
    A_enc.par_block_col_major_pack(A_mat, d, n_heads, d, this->param.default_scale());

    static vector<CkksCiphertext> in_cts, out_cts;
    static vector<CustomData> expand_data, reduce_data;
    in_cts.clear();
    out_cts.clear();
    expand_data.clear();
    reduce_data.clear();

    for (auto& ct : A_enc.data)
        in_cts.push_back(ct.copy());
    expand_data.emplace_back(static_cast<void*>(expand_ptr.get()));
    reduce_data.emplace_back(static_cast<void*>(reduce_ptr.get()));

    // Output: same shape as input, level = init_level - 4
    uint32_t num_block_rows_A = div_ceil(m, d);
    uint32_t n_h_padded = 1;
    while (n_h_padded < n_heads)
        n_h_padded <<= 1;
    uint32_t n_slot = this->param.max_slots();
    if (n_slot < n_h_padded * d * d && n_slot / (d * d) == 1) {
        n_h_padded = n_heads;
    }
    uint32_t G = (n_slot >= n_h_padded * d * d) ? 1 : n_h_padded / (n_slot / (d * d));
    uint32_t n_out = num_block_rows_A * G;

    for (uint32_t i = 0; i < n_out; i++)
        out_cts.emplace_back(this->context.parameter(), init_level - 4);
        out_cts.back().set_scale(this->param.default_scale());

    vector<CxxVectorArgument> cxx_args = {
        {"input", &in_cts},
        {"_expand_cpmm", &expand_data},
        {"_reduce_cpmm", &reduce_data},
        {"output", &out_cts},
    };
    run_block_col_major_e2e_test(*this, server_dir, cxx_args, expected, {m, head_dim}, d, n_heads, true, out_cts);
}

static Array<double, 2> make_uniform_coeff(const vector<double>& c, uint32_t n_channel) {
    Array<double, 2> coeff({(uint64_t)c.size(), (uint64_t)n_channel});
    for (int i = 0; i < (int)c.size(); i++) {
        for (uint32_t ch = 0; ch < n_channel; ch++) {
            coeff.set(i, ch, c[i]);
        }
    }
    return coeff;
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "poly_relu_bsgs_feature2d", "", HeteroProcessors) {
    Duo input_shape = {32, 32};
    uint32_t n_channel = 32;
    Duo skip = {1, 1};
    uint32_t n_channel_per_ct = div_ceil(this->n_slot, prod(input_shape));
    int order0 = 7;
    int order1 = 7;
    int level_cost0 = PolyRelu2D::compute_bsgs_level_cost(order0);
    int level_cost1 = PolyRelu2D::compute_bsgs_level_cost(order1);
    int init_level = level_cost0 + level_cost1 + 1;  // +1 for sign(x)*x multiplication

    auto input_array = gen_random_array<3>({n_channel, input_shape[0], input_shape[1]}, 1.0);
    // auto weight0 = gen_random_array<2>({order0 + 1, (int)n_channel}, 1.0);
    // auto weight1 = gen_random_array<2>({order1 + 1, (int)n_channel}, 1.0);
    auto weight0 = make_uniform_coeff(
        {0.0, 7.30445164958251, 0.0, -3.46825871108659e1, 0.0, 5.98596518298826e1, 0.0, -3.18755225906466e1},
        n_channel);
    auto weight1 = make_uniform_coeff(
        {0.0, 2.40085652217597, 0.0, -2.63125454261783, 0.0, 1.54912674773593, 0.0, -3.31172956504304e-1}, n_channel);

    Feature2DEncrypted input_feature(&this->context, init_level, skip);
    input_feature.pack_multiplexed(input_array, this->context.parameter().default_scale());

    // Layer 0: p0(x)
    PolyRelu2D poly0(this->context.parameter(), input_shape, order0, move(weight0), skip, n_channel_per_ct,
                     init_level);
    poly0.prepare_weight_bsgs();

    // Layer 1: sign(x) ≈ p1(p0(x))
    PolyRelu2D poly1(this->context.parameter(), input_shape, order1, move(weight1), skip, n_channel_per_ct,
                     init_level - level_cost0);
    poly1.prepare_weight_bsgs();

    // Output: after sign*x mult + rescale
    int output_level = init_level - level_cost0 - level_cost1 - 1;
    Feature2DEncrypted output_feature(&this->context, output_level);
    output_feature.skip = skip;
    output_feature.shape = input_shape;
    output_feature.n_channel = n_channel;
    output_feature.n_channel_per_ct = input_feature.n_channel_per_ct;
    for (int i = 0; i < div_ceil(n_channel, n_channel_per_ct); i++) {
        output_feature.data.emplace_back(this->context.parameter(), output_level);
        output_feature.data.back().set_scale(this->param.default_scale());
    }

    fs::path project_path = base_path /
                            ("CKKS_poly_relu_bsgs_" + to_string(n_channel) + "_channel_order_" + to_string(order0) +
                             "_" + to_string(order1)) /
                            ("level_" + to_string(init_level));

    auto arg_names = read_arg_names(project_path);
    // Build cxx_args matching Python naming: poly0_weight_pt{i}, poly1_weight_pt{i}
    vector<CxxVectorArgument> cxx_args;
    int idx = 0;
    cxx_args.push_back({arg_names[idx++], &input_feature.data});
    for (int i = 0; i <= order0; i++)
        cxx_args.push_back({arg_names[idx++], &poly0.weight_pt[i]});
    for (int i = 0; i <= order1; i++)
        cxx_args.push_back({arg_names[idx++], &poly1.weight_pt[i]});
    cxx_args.push_back({arg_names[idx++], &output_feature.data});

    this->run(project_path, cxx_args);

    auto output_mg = output_feature.unpack_multiplexed();

    // Plaintext reference: result = x + sign(x) * x
    auto p0_plain = poly0.run_plaintext_for_non_absorb_case(input_array);
    auto sign_plain = poly1.run_plaintext_for_non_absorb_case(p0_plain);
    Array<double, 3> expected({n_channel, input_shape[0], input_shape[1]});
    for (uint64_t i = 0; i < input_array.get_size(); i++) {
        expected.set(i, input_array.get(i) + sign_plain.get(i) * input_array.get(i));
    }

    // relu(x) = (x + sign(x) * x) / 2
    Array<double, 3> relu_ct({n_channel, input_shape[0], input_shape[1]});
    Array<double, 3> relu_expected({n_channel, input_shape[0], input_shape[1]});
    Array<double, 3> relu_true({n_channel, input_shape[0], input_shape[1]});
    for (uint64_t i = 0; i < input_array.get_size(); i++) {
        relu_ct.set(i, output_mg.get(i) / 2.0);
        relu_expected.set(i, expected.get(i) / 2.0);
        relu_true.set(i, std::max(0.0, input_array.get(i)));
    }

    print_double_message(input_array.to_array_1d().data(), "input_array", 10);
    print_double_message(relu_true.to_array_1d().data(), "relu_true", 10);
    print_double_message(relu_expected.to_array_1d().data(), "relu_plain", 10);
    print_double_message(relu_ct.to_array_1d().data(), "relu_ct", 10);

    auto compare_result = compare(expected, output_mg);
    REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
    REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "poly_relu_stockmeyer_feature2d", "", HeteroProcessors) {
    Duo input_shape = {32, 32};
    uint32_t n_channel = 32;
    Duo skip = {1, 1};
    uint32_t n_channel_per_ct = div_ceil(this->n_slot, prod(input_shape));
    int order0 = 7;
    int order1 = 7;
    int level_cost0 = Poly2D::compute_stockmeyer_level_cost(order0);
    int level_cost1 = Poly2D::compute_stockmeyer_level_cost(order1);
    int init_level = level_cost0 + level_cost1 + 1;  // +1 for sign(x)*x multiplication

    auto input_array = gen_random_array<3>({n_channel, input_shape[0], input_shape[1]}, 1.0);
    // auto weight0 = gen_random_array<2>({order0 + 1, (int)n_channel}, 1.0);
    // auto weight1 = gen_random_array<2>({order1 + 1, (int)n_channel}, 1.0);
    auto weight0 = make_uniform_coeff(
        {0.0, 7.30445164958251, 0.0, -3.46825871108659e1, 0.0, 5.98596518298826e1, 0.0, -3.18755225906466e1},
        n_channel);
    auto weight1 = make_uniform_coeff(
        {0.0, 2.40085652217597, 0.0, -2.63125454261783, 0.0, 1.54912674773593, 0.0, -3.31172956504304e-1}, n_channel);

    Feature2DEncrypted input_feature(&this->context, init_level, skip);
    input_feature.pack_multiplexed(input_array, this->context.parameter().default_scale());

    // Layer 0: p0(x)
    Poly2D poly0(this->context.parameter(), input_shape, order0, move(weight0), skip, n_channel_per_ct, init_level);
    poly0.prepare_weight_stockmeyer();

    // Layer 1: sign(x) ≈ p1(p0(x))
    Poly2D poly1(this->context.parameter(), input_shape, order1, move(weight1), skip, n_channel_per_ct,
                 init_level - level_cost0);
    poly1.prepare_weight_stockmeyer();

    // Output: after sign*x mult + rescale
    int output_level = init_level - level_cost0 - level_cost1 - 1;
    Feature2DEncrypted output_feature(&this->context, output_level);
    output_feature.skip = skip;
    output_feature.shape = input_shape;
    output_feature.n_channel = n_channel;
    output_feature.n_channel_per_ct = input_feature.n_channel_per_ct;
    for (int i = 0; i < div_ceil(n_channel, n_channel_per_ct); i++) {
        output_feature.data.push_back(CkksCiphertext(this->param, output_level));
    }

    fs::path project_path = base_path /
                            ("CKKS_poly_relu_stockmeyer_" + to_string(n_channel) + "_channel_order_" +
                             to_string(order0) + "_" + to_string(order1)) /
                            ("level_" + to_string(init_level));

    auto arg_names = read_arg_names(project_path);
    // Build cxx_args matching Python naming: poly0_weight_pt{i}, poly1_weight_pt{i}
    vector<CxxVectorArgument> cxx_args;
    int idx = 0;
    cxx_args.push_back({arg_names[idx++], &input_feature.data});
    for (int i = 0; i <= order0; i++)
        cxx_args.push_back({arg_names[idx++], &poly0.weight_pt[i]});
    for (int i = 0; i <= order1; i++)
        cxx_args.push_back({arg_names[idx++], &poly1.weight_pt[i]});
    cxx_args.push_back({arg_names[idx++], &output_feature.data});

    this->run(project_path, cxx_args);

    auto output_mg = output_feature.unpack_multiplexed();

    // Plaintext reference: result = x + sign(x) * x
    auto p0_plain = poly0.run_plaintext_for_non_absorb_case(input_array);
    auto sign_plain = poly1.run_plaintext_for_non_absorb_case(p0_plain);
    Array<double, 3> expected({n_channel, input_shape[0], input_shape[1]});
    for (uint64_t i = 0; i < input_array.get_size(); i++) {
        expected.set(i, input_array.get(i) + sign_plain.get(i) * input_array.get(i));
    }

    // relu(x) = (x + sign(x) * x) / 2
    Array<double, 3> relu_ct({n_channel, input_shape[0], input_shape[1]});
    Array<double, 3> relu_expected({n_channel, input_shape[0], input_shape[1]});
    Array<double, 3> relu_true({n_channel, input_shape[0], input_shape[1]});
    for (uint64_t i = 0; i < input_array.get_size(); i++) {
        relu_ct.set(i, output_mg.get(i) / 2.0);
        relu_expected.set(i, expected.get(i) / 2.0);
        relu_true.set(i, std::max(0.0, input_array.get(i)));
    }

    print_double_message(input_array.to_array_1d().data(), "input_array", 10);
    print_double_message(relu_true.to_array_1d().data(), "relu_true", 10);
    print_double_message(relu_expected.to_array_1d().data(), "relu_plain", 10);
    print_double_message(relu_ct.to_array_1d().data(), "relu_ct", 10);

    auto compare_result = compare(expected, output_mg);
    REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
    REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
}


TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "poly_maxpool2x2_param_check", "", HeteroProcessors) {
    const int N = 32768;

    CkksParameter local_param =
        CkksParameter::create_parameter(static_cast<int>(std::log2(N)), RingType::Standard);

    // Check PN15QP880 parameter compatibility
    REQUIRE(local_param.n() == 32768);
    REQUIRE(local_param.max_level() == 17);
    REQUIRE(local_param.p().size() == 3);

    // Compare representative Q and P moduli
    REQUIRE(local_param.q()[0] == 0x4000000120001ULL);
    REQUIRE(local_param.q()[17] == 0x10000960001ULL);
    REQUIRE(local_param.p()[0] == 0x40000001B0001ULL);
    REQUIRE(local_param.p()[2] == 0x4000000270001ULL);

    // PN15QP880 default scale = 2^40
    REQUIRE(
        local_param.default_scale() ==
        std::ldexp(1.0, 40)
    );
}



TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "poly_maxpool2x2_stockmeyer_feature2d", "", HeteroProcessors) {
    // =========================================================
    // 1. Test configuration
    // =========================================================
    const int N = 32768;
    const int init_level = 17;
    const int order = 7;
    const int n_channel = 32;

    const Duo input_shape = {32, 32};
    const Duo output_shape = {16, 16};
    const Duo skip = {1, 1};

    // Independent approximation bounds for two rounds.
    // Round 1: horizontal Max
    // Round 2: vertical Max
    const double B_r1 = 1.0;
    const double B_r2 = 1.0;

    // Input amplitude for range testing
    const double input_amplitude = 10.0;

    REQUIRE(std::isfinite(B_r1));
    REQUIRE(std::isfinite(B_r2));
    REQUIRE(B_r1 > 0.0);
    REQUIRE(B_r2 > 0.0);

    // =========================================================
    // 2. Independent CKKS context
    // =========================================================
    CkksParameter local_param =
        CkksParameter::create_parameter(static_cast<int>(std::log2(N)), RingType::Standard);

    REQUIRE(local_param.max_level() >= init_level);

    CkksContext local_context =
        CkksContext::create_random_context(local_param);

    local_context.gen_rotation_keys();

    const double scale = local_param.default_scale();
    const int n_slot = N / 2;

    const int n_channel_per_ct =
        n_slot / (input_shape[0] * input_shape[1]);

    const int n_ct =
        div_ceil(n_channel, n_channel_per_ct);

    // =========================================================
    // 3. Original alpha=7 polynomial coefficients
    // =========================================================
    const vector<double> coeff0 = {
        0.0,
        7.30445164958251,
        0.0,
        -34.6825871108659,
        0.0,
        59.8596518298826,
        0.0,
        -31.8755225906466
    };

    const vector<double> coeff1 = {
        0.0,
        2.40085652217597,
        0.0,
        -2.63125454261783,
        0.0,
        1.54912674773593,
        0.0,
        -0.331172956504304
    };

    // =========================================================
    // 4. Apply B to first polynomial P0
    //
    // P0(x / (2B)) = sum c[k] / (2B)^k * x^k
    //
    // This changes coefficients without adding CKKS levels.
    // =========================================================
    auto make_scaled_coeff0 = [&](double B) {
        vector<double> scaled = coeff0;
        const double s = 2.0 * B;

        for (size_t k = 0; k < scaled.size(); ++k) {
            scaled[k] /= std::pow(
                s, static_cast<int>(k)
            );
        }

        return scaled;
    };

    const vector<double> coeff0_r1 =
        make_scaled_coeff0(B_r1);

    const vector<double> coeff0_r2 =
        make_scaled_coeff0(B_r2);

    auto weight0_r1 =
        make_uniform_coeff(coeff0_r1, n_channel);

    auto weight0_r2 =
        make_uniform_coeff(coeff0_r2, n_channel);

    auto weight1 =
        make_uniform_coeff(coeff1, n_channel);

    // =========================================================
    // 5. Prepare polynomial layers
    // =========================================================

    // Round 1: horizontal Max
    // Level: 17 -> 14 -> 11
    Poly2D poly0_r1(
        local_param,
        input_shape,
        order,
        weight0_r1.copy(),
        skip,
        n_channel_per_ct,
        17
    );

    Poly2D poly1_r1(
        local_param,
        input_shape,
        order,
        weight1.copy(),
        skip,
        n_channel_per_ct,
        14
    );

    // Round 2: vertical Max
    // Level: 9 -> 6 -> 3
    Poly2D poly0_r2(
        local_param,
        input_shape,
        order,
        weight0_r2.copy(),
        skip,
        n_channel_per_ct,
        9
    );

    Poly2D poly1_r2(
        local_param,
        input_shape,
        order,
        weight1.copy(),
        skip,
        n_channel_per_ct,
        6
    );

    poly0_r1.prepare_weight_stockmeyer();
    poly1_r1.prepare_weight_stockmeyer();
    poly0_r2.prepare_weight_stockmeyer();
    poly1_r2.prepare_weight_stockmeyer();

    // =========================================================
    // 6. Encode 0.5 plaintext weights
    // =========================================================
    auto encode_half = [&](int level) {
        vector<CkksPlaintext> result;
        result.reserve(n_ct);

        vector<double> values(n_slot, 0.5);

        for (int i = 0; i < n_ct; ++i) {
            result.push_back(
                fhe_api::encode_ringt(local_context,
                    values,
                    local_param.q()[level]
                )
            );
        }

        return result;
    };

    auto half_pt_r1 = encode_half(10);
    auto half_pt_r2 = encode_half(2);

    // =========================================================
    // 7. Prepare MaxPooling selection masks
    // =========================================================
    Avgpool2DLayer mask_layer(
        input_shape,
        {2, 2}
    );

    mask_layer.prepare_weight_lazy(
        local_param,
        n_channel_per_ct,
        n_channel,
        1,
        skip,
        input_shape
    );

    const int out_channels_per_ct =
        n_channel_per_ct * 4;

    const int n_select_pt =
        std::min(n_channel, out_channels_per_ct);

    vector<CkksPlaintext> select_pt;
    select_pt.reserve(n_select_pt);

    for (int i = 0; i < n_select_pt; ++i) {
        auto mask = mask_layer.select_tensor(i);

        // AvgPooling mask uses 0.25.
        // MaxPooling needs a 0/1 selection mask.
        for (auto& v : mask) {
            v *= 4.0;
        }

        select_pt.push_back(
            fhe_api::encode_ringt(local_context,
                mask,
                local_param.q()[1]
            )
        );
    }

    // =========================================================
    // 8. Generate and encrypt input (normalized experiment)
    // =========================================================

    // 原始输入范围 [-10, 10]
    const double normalization_factor = 10.0;

    Array<double, 3> input_array({
        static_cast<uint64_t>(n_channel),
        32,
        32
    });

    // 固定随机种子
    std::mt19937 rng(20261022);
    std::uniform_real_distribution<double> dis(
        -input_amplitude,
        input_amplitude
    );

    // 生成原始输入
    for (uint64_t i = 0; i < input_array.get_size(); ++i) {
        input_array.set(i, dis(rng));
    }

    // 创建归一化副本，不修改原始 input_array
    // 创建独立的归一化输入数组
    Array<double, 3> normalized_input({
        static_cast<uint64_t>(n_channel),
        32,
        32
    });

    // 逐元素归一化
    for (uint64_t i = 0; i < input_array.get_size(); ++i) {
        normalized_input.set(
            i,
            input_array.get(i) / normalization_factor
        );
    }

    // 使用归一化数据进行 CKKS 编码和加密
    Feature2DEncrypted input_feature(
        &local_context,
        init_level,
        skip
    );

    input_feature.pack_multiplexed(
        normalized_input,
        scale
    );

    REQUIRE(input_feature.data.size() == n_ct);

    // =========================================================
    // 9. Prepare output ciphertext
    // =========================================================
    Feature2DEncrypted output_feature(
        &local_context,
        0
    );

    output_feature.shape = output_shape;
    output_feature.skip = {2, 2};
    output_feature.n_channel = n_channel;

    output_feature.n_channel_per_ct =
        n_channel_per_ct * 4;

    output_feature.data.push_back(
        CkksCiphertext(local_param, 0)
    );

    // =========================================================
    // 10. Load generated Python FHE task
    // =========================================================
    fs::path project_path =
        base_path
        / "CKKS_poly_maxpool2x2_repack_stockmeyer"
        / "level_17";

    auto arg_names = read_arg_names(project_path);

    vector<CxxVectorArgument> cxx_args;
    int idx = 0;

    cxx_args.push_back({
        arg_names.at(idx++),
        &input_feature.data
    });

    auto append_poly_weights = [&](Poly2D& poly) {
        for (int i = 0; i <= order; ++i) {
            cxx_args.push_back({
                arg_names.at(idx++),
                &poly.weight_pt[i]
            });
        }
    };

    append_poly_weights(poly0_r1);
    append_poly_weights(poly1_r1);
    append_poly_weights(poly0_r2);
    append_poly_weights(poly1_r2);

    cxx_args.push_back({
        arg_names.at(idx++),
        &half_pt_r1
    });

    cxx_args.push_back({
        arg_names.at(idx++),
        &half_pt_r2
    });

    cxx_args.push_back({
        arg_names.at(idx++),
        &select_pt
    });

    cxx_args.push_back({
        arg_names.at(idx++),
        &output_feature.data
    });

    REQUIRE(idx == static_cast<int>(arg_names.size()));

    // =========================================================
    // 11. Execute FHE task
    // =========================================================
    if constexpr (is_same_v<TestType, ProcessorCpu>) {
        FheTaskCpu task(project_path.string());
        task.run(&local_context, cxx_args);

#ifdef INFERENCE_SDK_ENABLE_GPU
    } else if constexpr (
        is_same_v<TestType, ProcessorGpu>
    ) {
        FheTaskGpu task(project_path.string());
        task.run(&local_context, cxx_args);
#endif
    }

    // =========================================================
    // 12. Decrypt and unpack output
    // =========================================================
    // auto output_mg =
    //     output_feature.unpack_multiplexed();

    auto output_mg =
    output_feature.unpack_multiplexed();

    // 恢复到原始输入尺度
    for (uint64_t i = 0; i < output_mg.get_size(); ++i) {
        output_mg.set(
            i,
            output_mg.get(i) * normalization_factor
        );
    }

    // =========================================================
    // 13. Plaintext polynomial reference
    // =========================================================
    auto eval_poly = [](
        const vector<double>& coeff,
        double x
    ) {
        double y = 0.0;

        for (int i = static_cast<int>(coeff.size()) - 1;
             i >= 0;
             --i) {
            y = y * x + coeff[i];
        }

        return y;
    };

    auto approx_max = [&](double a, double b, double B) {
        const double d = a - b;

        // Normalize Sign polynomial input
        const double normalized_d =
            d / (2.0 * B);

        const double sign_approx =
            eval_poly(
                coeff1,
                eval_poly(coeff0, normalized_d)
            );

        return 0.5 * (
            a + b + d * sign_approx
        );
    };

    Array<double, 3> expected({
        static_cast<uint64_t>(n_channel),
        16,
        16
    });

    Array<double, 3> true_max({
        static_cast<uint64_t>(n_channel),
        16,
        16
    });

    // 明文参考使用原始输入尺度对应的 B
    const double reference_B_r1 = B_r1 * normalization_factor;
    const double reference_B_r2 = B_r2 * normalization_factor;

    for (int ch = 0; ch < n_channel; ++ch) {
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 16; ++j) {

                const double a =
                    input_array.get(ch, 2*i, 2*j);

                const double b =
                    input_array.get(ch, 2*i, 2*j+1);

                const double c =
                    input_array.get(ch, 2*i+1, 2*j);

                const double d =
                    input_array.get(ch, 2*i+1, 2*j+1);

                // // Round 1
                // const double m1 =
                //     approx_max(a, b, B_r1);

                // const double m2 =
                //     approx_max(c, d, B_r1);

                // // Round 2
                // const double y_poly =
                //     approx_max(m1, m2, B_r2);


                // Round 1
                const double m1 = approx_max(a, b, reference_B_r1);
                const double m2 = approx_max(c, d, reference_B_r1);

                // Round 2
                const double y_poly = approx_max(m1, m2, reference_B_r2);

                expected.set(
                    ch, i, j, y_poly
                );

                const double y_true =
                    std::max(
                        std::max(a, b),
                        std::max(c, d)
                    );

                true_max.set(
                    ch, i, j, y_true
                );
            }
        }
    }

    // =========================================================
    // 14. Check for non-finite values
    // =========================================================
    bool poly_finite = true;
    bool fhe_finite = true;

    for (uint64_t i = 0; i < expected.get_size(); ++i) {
        if (!std::isfinite(expected.get(i))) {
            poly_finite = false;
        }
        if (!std::isfinite(output_mg.get(i))) {
            fhe_finite = false;
        }
    }

    std::cout
        << "========================================\n"
        << "Poly MaxPooling test configuration\n"
        << "========================================\n"
        << "alpha = 7\n"
        << "B_r1 = " << B_r1 << "\n"
        << "B_r2 = " << B_r2 << "\n"
        << "input_amplitude = "
        << input_amplitude << "\n"
        << "polynomial_finite = "
        << poly_finite << "\n"
        << "fhe_finite = "
        << fhe_finite << "\n";

    // Do not compute RMSE against inf/NaN references.
    REQUIRE(poly_finite);
    REQUIRE(fhe_finite);

    // =========================================================
    // 15. Compare numerical errors
    // =========================================================
    auto fhe_error =
        compare(expected, output_mg);

    auto approx_error =
        compare(true_max, expected);

    // Total error: exact MaxPooling vs FHE output
    auto total_error =
        compare(true_max, output_mg);

    std::cout
        << "FHE vs polynomial reference:"
        << " max_error=" << fhe_error.max_error
        << " rmse=" << fhe_error.rmse
        << std::endl;

    std::cout
        << "Polynomial vs exact MaxPooling:"
        << " max_error=" << approx_error.max_error
        << " rmse=" << approx_error.rmse
        << std::endl;

    std::cout
        << "FHE vs exact MaxPooling:"
        << " max_error=" << total_error.max_error
        << " rmse=" << total_error.rmse
        << std::endl;

    REQUIRE(
        fhe_error.max_error <
        5.0e-2 * fhe_error.max_abs
    );

    REQUIRE(
        fhe_error.rmse <
        1.0e-2 * fhe_error.rms
    );
}


TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "poly_maxpool2x2_stockmeyer_fused17_feature2d", "[fhe_layers][poly_maxpool][fused17]", HeteroProcessors) {
    // =========================================================
    // 1. Test configuration
    // =========================================================
    const int N = 32768;
    const int init_level = 17;
    const int order = 7;
    const int n_channel = 32;

    const Duo input_shape = {32, 32};
    const Duo output_shape = {16, 16};
    const Duo skip = {1, 1};

    // Default PolyMaxPoolConfig
    const double upper_bound = 1.0;
    const double k1 = 1.0;
    const double k2 = 1.05;

    const double B_r1 = upper_bound * k1;
    const double B_r2 = upper_bound * k2;

    // Plaintext calibration
    const double running_max = 10.0;
    const double input_amplitude = 10.0;

    const double normalization_factor =
        running_max / upper_bound;

    // Reference bounds for ORIGINAL input scale
    const double reference_B_r1 =
        normalization_factor * B_r1;

    const double reference_B_r2 =
        normalization_factor * B_r2;

    REQUIRE(std::isfinite(normalization_factor));
    REQUIRE(normalization_factor > 0.0);
    REQUIRE(B_r1 > 0.0);
    REQUIRE(B_r2 > 0.0);

    // =========================================================
    // 2. Independent CKKS context
    // =========================================================
    CkksParameter local_param =
        CkksParameter::create_parameter(static_cast<int>(std::log2(N)), RingType::Standard);

    REQUIRE(local_param.max_level() >= init_level);

    CkksContext local_context =
        CkksContext::create_random_context(local_param);

    local_context.gen_rotation_keys();

    const double scale = local_param.default_scale();
    const int n_slot = N / 2;

    const int n_channel_per_ct =
        n_slot / (input_shape[0] * input_shape[1]);

    const int n_ct =
        div_ceil(n_channel, n_channel_per_ct);

    const int out_channels_per_ct =
        n_channel_per_ct * 4;

    const int n_out_ct =
        div_ceil(n_channel, out_channels_per_ct);

    const int n_select_pt =
        std::min(n_channel, out_channels_per_ct);

    REQUIRE(n_ct == 2);
    REQUIRE(n_out_ct == 1);

    // =========================================================
    // 3. Original alpha=7 polynomial coefficients
    // =========================================================
    const vector<double> coeff0 = {
        0.0,
        7.30445164958251,
        0.0,
        -34.6825871108659,
        0.0,
        59.8596518298826,
        0.0,
        -31.8755225906466
    };

    const vector<double> coeff1 = {
        0.0,
        2.40085652217597,
        0.0,
        -2.63125454261783,
        0.0,
        1.54912674773593,
        0.0,
        -0.331172956504304
    };

    // =========================================================
    // 4. Scale P0 polynomial coefficients
    //
    // P0(d/(2B)) = sum c[k]/(2B)^k * d^k
    //
    // Here d is already normalized.
    // =========================================================
    auto make_scaled_coeff0 = [&](double B) {
        vector<double> scaled = coeff0;

        const double divisor = 2.0 * B;

        for (size_t k = 0; k < scaled.size(); ++k) {
            scaled[k] /= std::pow(
                divisor,
                static_cast<int>(k)
            );
        }

        return scaled;
    };

    const vector<double> coeff0_r1 =
        make_scaled_coeff0(B_r1);

    const vector<double> coeff0_r2 =
        make_scaled_coeff0(B_r2);

    auto weight0_r1 =
        make_uniform_coeff(coeff0_r1, n_channel);

    auto weight0_r2 =
        make_uniform_coeff(coeff0_r2, n_channel);

    auto weight1 =
        make_uniform_coeff(coeff1, n_channel);

    // =========================================================
    // 5. Prepare Stockmeyer layers
    //
    // Round 1:
    //     P0: 17 -> 14
    //     P1: 14 -> 11
    //     d*sign: 11 -> 10
    //     *0.5: 10 -> 9
    //
    // Round 2:
    //     P0: 9 -> 6
    //     P1: 6 -> 3
    //     d*sign: 3 -> 2
    //     *0.5: 2 -> 1
    //
    // Repack + scale restoration:
    //     1 -> 0
    // =========================================================

    Poly2D poly0_r1(
        local_param,
        input_shape,
        order,
        weight0_r1.copy(),
        skip,
        n_channel_per_ct,
        17
    );

    Poly2D poly1_r1(
        local_param,
        input_shape,
        order,
        weight1.copy(),
        skip,
        n_channel_per_ct,
        14
    );

    Poly2D poly0_r2(
        local_param,
        input_shape,
        order,
        weight0_r2.copy(),
        skip,
        n_channel_per_ct,
        9
    );

    Poly2D poly1_r2(
        local_param,
        input_shape,
        order,
        weight1.copy(),
        skip,
        n_channel_per_ct,
        6
    );

    poly0_r1.prepare_weight_stockmeyer();
    poly1_r1.prepare_weight_stockmeyer();
    poly0_r2.prepare_weight_stockmeyer();
    poly1_r2.prepare_weight_stockmeyer();

    // =========================================================
    // 6. Encode 0.5 plaintext constants
    // =========================================================
    auto encode_half = [&](int level) {
        vector<CkksPlaintext> result;
        result.reserve(n_ct);

        vector<double> values(n_slot, 0.5);

        for (int i = 0; i < n_ct; ++i) {
            result.push_back(
                fhe_api::encode_ringt(local_context,
                    values,
                    local_param.q()[level]
                )
            );
        }

        return result;
    };

    auto half_pt_r1 = encode_half(10);
    auto half_pt_r2 = encode_half(2);

    // =========================================================
    // 7. Prepare MaxPooling selection masks
    //
    // Important optimization:
    //
    // OLD:
    //   mask = 0 or 1
    //   Repack(v, mask)
    //   output = Repack(v, mask) * 10
    //
    // NEW:
    //   fused_mask = mask * 10
    //   output = Repack(v, fused_mask)
    //
    // Both repacking and output restoration are
    // completed in a single mult + rescale.
    // =========================================================
    Avgpool2DLayer mask_layer(
        input_shape,
        {2, 2}
    );

    mask_layer.prepare_weight_lazy(
        local_param,
        n_channel_per_ct,
        n_channel,
        1,
        skip,
        input_shape
    );

    vector<CkksPlaintext> select_pt;
    select_pt.reserve(n_select_pt);

    for (int i = 0; i < n_select_pt; ++i) {
        auto mask = mask_layer.select_tensor(i);

        // AvgPooling selection mask: 0.25
        // MaxPooling selection mask: 1.0
        // Fused output scale: 10.0
        for (auto& v : mask) {
            v *= 4.0 * normalization_factor;
        }

        select_pt.push_back(
            fhe_api::encode_ringt(local_context,
                mask,
                local_param.q()[1]
            )
        );
    }

    // =========================================================
    // 8. Generate original input
    // =========================================================
    Array<double, 3> input_array({
        static_cast<uint64_t>(n_channel),
        32,
        32
    });

    // Same seed as previous experiments
    std::mt19937 rng(20261022);

    std::uniform_real_distribution<double> dis(
        -input_amplitude,
        input_amplitude
    );

    for (uint64_t i = 0; i < input_array.get_size(); ++i) {
        input_array.set(i, dis(rng));
    }

    // =========================================================
    // 9. Simulate normalization absorbed by
    //    the preceding Conv/ReLU layer
    //
    // This test does NOT yet implement Conv -> MaxPool.
    // It verifies the MaxPool core given a normalized
    // encrypted input.
    // =========================================================
    Array<double, 3> normalized_input({
        static_cast<uint64_t>(n_channel),
        32,
        32
    });

    for (uint64_t i = 0; i < input_array.get_size(); ++i) {
        normalized_input.set(
            i,
            input_array.get(i) / normalization_factor
        );
    }

    // =========================================================
    // 10. Encrypt normalized input at Level 17
    // =========================================================
    Feature2DEncrypted input_feature(
        &local_context,
        init_level,
        skip
    );

    input_feature.pack_multiplexed(
        normalized_input,
        scale
    );

    REQUIRE(input_feature.data.size() == n_ct);

    // =========================================================
    // 11. Prepare output ciphertext at Level 0
    // =========================================================
    Feature2DEncrypted output_feature(
        &local_context,
        0
    );

    output_feature.shape = output_shape;
    output_feature.skip = {2, 2};
    output_feature.n_channel = n_channel;
    output_feature.n_channel_per_ct =
        out_channels_per_ct;

    for (int i = 0; i < n_out_ct; ++i) {
        output_feature.data.push_back(
            CkksCiphertext(local_param, 0)
        );
    }

    // =========================================================
    // 12. Load existing Level-17 DAG
    //
    // DAG structure is unchanged.
    // Only plaintext mask values differ.
    // =========================================================
    fs::path project_path =
        base_path
        / "CKKS_poly_maxpool2x2_repack_stockmeyer"
        / "level_17";

    REQUIRE(fs::exists(project_path / "mega_ag.json"));

    auto arg_names = read_arg_names(project_path);

    vector<CxxVectorArgument> cxx_args;
    int idx = 0;

    // Explicitly check argument names to prevent
    // silent misbinding when the DAG changes.
    auto append_arg = [&](const std::string& expected,
                          auto* buffer) {
        const auto& name = arg_names.at(idx++);
        REQUIRE(name == expected);
        cxx_args.push_back({name, buffer});
    };

    append_arg(
        "input_node",
        &input_feature.data
    );

    auto append_poly_weights = [&](
        const std::string& prefix,
        Poly2D& poly
    ) {
        for (int i = 0; i <= order; ++i) {
            append_arg(
                prefix + "_weight_pt" + std::to_string(i),
                &poly.weight_pt.at(i)
            );
        }
    };

    append_poly_weights("poly0_r1", poly0_r1);
    append_poly_weights("poly1_r1", poly1_r1);
    append_poly_weights("poly0_r2", poly0_r2);
    append_poly_weights("poly1_r2", poly1_r2);

    append_arg(
        "half_weight_pt_r1",
        &half_pt_r1
    );

    append_arg(
        "half_weight_pt_r2",
        &half_pt_r2
    );

    append_arg(
        "maxpool_select_pt",
        &select_pt
    );

    append_arg(
        "output_ct",
        &output_feature.data
    );

    REQUIRE(idx == static_cast<int>(arg_names.size()));

    // =========================================================
    // 13. Execute FHE task
    // =========================================================
    std::cout
        << "\n=== Fused17 MaxPooling configuration ===\n"
        << "N=" << N
        << ", init_level=" << init_level
        << ", channels=" << n_channel
        << ", input_ct=" << n_ct
        << ", output_ct=" << n_out_ct
        << ", B1=" << B_r1
        << ", B2=" << B_r2
        << ", normalization_factor="
        << normalization_factor
        << std::endl;

    if constexpr (is_same_v<TestType, ProcessorCpu>) {
        FheTaskCpu task(project_path.string());
        task.run(&local_context, cxx_args);

#ifdef INFERENCE_SDK_ENABLE_GPU
    } else if constexpr (
        is_same_v<TestType, ProcessorGpu>
    ) {
        FheTaskGpu task(project_path.string());
        task.run(&local_context, cxx_args);
#endif
    }

    // =========================================================
    // 14. Decrypt and unpack
    //
    // Do NOT multiply by normalization_factor here.
    // The output has already been restored by
    // the fused Repack plaintext mask.
    // =========================================================
    auto output_mg =
        output_feature.unpack_multiplexed();

    // =========================================================
    // 15. Plaintext polynomial reference
    //
    // Use original-scale input and original-scale B.
    // This is mathematically equivalent to:
    //
    // z = x / 10
    // v = PolyMaxPool(z, B1=1.0, B2=1.05)
    // y = 10 * v
    // =========================================================
    auto eval_poly = [](
        const vector<double>& coeff,
        double x
    ) {
        double result = 0.0;

        for (int i = static_cast<int>(coeff.size()) - 1;
             i >= 0; --i) {
            result = result * x + coeff[i];
        }

        return result;
    };

    auto approx_max = [&](
        double a,
        double b,
        double B
    ) {
        const double diff = a - b;

        const double normalized_diff =
            diff / (2.0 * B);

        const double p0 =
            eval_poly(coeff0, normalized_diff);

        const double sign_approx =
            eval_poly(coeff1, p0);

        return 0.5 * (
            a + b + diff * sign_approx
        );
    };

    Array<double, 3> expected({
        static_cast<uint64_t>(n_channel),
        16,
        16
    });

    Array<double, 3> exact_max({
        static_cast<uint64_t>(n_channel),
        16,
        16
    });

    for (int ch = 0; ch < n_channel; ++ch) {
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 16; ++j) {

                const double a =
                    input_array.get(
                        ch, 2 * i, 2 * j
                    );

                const double b =
                    input_array.get(
                        ch, 2 * i, 2 * j + 1
                    );

                const double c =
                    input_array.get(
                        ch, 2 * i + 1, 2 * j
                    );

                const double d =
                    input_array.get(
                        ch, 2 * i + 1, 2 * j + 1
                    );

                // Round 1: horizontal binary Max
                const double m1 =
                    approx_max(
                        a, b, reference_B_r1
                    );

                const double m2 =
                    approx_max(
                        c, d, reference_B_r1
                    );

                // Round 2: vertical binary Max
                const double approx_result =
                    approx_max(
                        m1, m2, reference_B_r2
                    );

                const double true_result =
                    std::max(
                        std::max(a, b),
                        std::max(c, d)
                    );

                expected.set(
                    ch, i, j, approx_result
                );

                exact_max.set(
                    ch, i, j, true_result
                );
            }
        }
    }

    // =========================================================
    // 16. Error comparison
    // =========================================================
    auto fhe_vs_poly = compare(
        expected,
        output_mg
    );

    auto poly_vs_exact = compare(
        exact_max,
        expected
    );

    auto fhe_vs_exact = compare(
        exact_max,
        output_mg
    );

    std::cout << std::setprecision(12);

    std::cout
        << "\n========================================\n"
        << "Fused17 MaxPooling Test Results\n"
        << "========================================\n"

        << "\nFHE vs Plaintext Polynomial\n"
        << "max_error = "
        << fhe_vs_poly.max_error
        << "\nrmse      = "
        << fhe_vs_poly.rmse

        << "\n\nPolynomial vs Exact Max\n"
        << "max_error = "
        << poly_vs_exact.max_error
        << "\nrmse      = "
        << poly_vs_exact.rmse

        << "\n\nFHE vs Exact Max\n"
        << "max_error = "
        << fhe_vs_exact.max_error
        << "\nrmse      = "
        << fhe_vs_exact.rmse

        << "\n========================================\n"
        << std::endl;

    // =========================================================
    // 17. Regression assertions
    // =========================================================
    REQUIRE(std::isfinite(fhe_vs_poly.max_error));
    REQUIRE(std::isfinite(fhe_vs_poly.rmse));

    REQUIRE(std::isfinite(poly_vs_exact.max_error));
    REQUIRE(std::isfinite(poly_vs_exact.rmse));

    REQUIRE(std::isfinite(fhe_vs_exact.max_error));
    REQUIRE(std::isfinite(fhe_vs_exact.rmse));

    // Preliminary CKKS regression thresholds
    REQUIRE(fhe_vs_poly.max_error < 5e-4);
    REQUIRE(fhe_vs_poly.rmse < 1e-4);
}




TEST_CASE("poly_maxpool2x2_range_norm_stockmeyer_feature2d", "[fhe_layers][poly_maxpool][range_norm]") {
    // =====================================================
    // 1. Configuration
    // =====================================================
    const int N = 65536;
    const int init_level = 19;
    const int order = 7;
    const int n_channel = 32;

    const Duo input_shape = {32, 32};
    const Duo output_shape = {16, 16};
    const Duo skip = {1, 1};

    // Default PolyMaxPoolConfig values
    const double upper_bound = 1.0;
    const double k1 = 1.0;
    const double k2 = 1.05;

    const double B_r1 = k1 * upper_bound;
    const double B_r2 = k2 * upper_bound;

    // Fixed plaintext calibration for this test
    const double running_max = 10.0;
    const double input_amplitude = 10.0;

    const double normalization_factor =
        running_max / upper_bound;

    const double input_multiplier =
        1.0 / normalization_factor;

    const double reference_B_r1 =
        normalization_factor * B_r1;

    const double reference_B_r2 =
        normalization_factor * B_r2;

    REQUIRE(B_r1 > 0.0);
    REQUIRE(B_r2 > 0.0);
    REQUIRE(normalization_factor > 0.0);

    // =====================================================
    // 2. Independent CKKS context
    // =====================================================
    // CkksParameter local_param =
    //     CkksParameter::create_parameter(static_cast<int>(std::log2(N)), RingType::Standard);
    // Load the same CKKS parameters used by the compiler
    const fs::path param_path =
        "/tmp/poly_maxpool2x2_task/task/server/ckks_parameter.json";

    std::ifstream param_file(param_path);
    REQUIRE(param_file.is_open());

    nlohmann::json param_json;
    param_file >> param_json;

    const auto& cfg = param_json.at("param0");

    REQUIRE(
        cfg.at("param_name").get<std::string>() == "PN16QP1761"
    );

    const auto q =
        cfg.at("q").get<std::vector<uint64_t>>();

    const auto p =
        cfg.at("p").get<std::vector<uint64_t>>();

    CkksParameter local_param =
        CkksParameter::create_custom_parameter(
            static_cast<int>(std::log2(N)), cfg.at("log_default_scale").get<int>(), q, p, RingType::Standard
        );

    REQUIRE(local_param.n() == N);
    REQUIRE(local_param.max_level() >= init_level);

    std::cout << "CKKS Max Level: "
            << local_param.max_level()
            << std::endl;

    std::cout << "CKKS Default Scale: "
            << local_param.default_scale()
            << std::endl;

    REQUIRE(local_param.n() == N);
    REQUIRE(local_param.max_level() >= init_level);

    CkksContext local_context =
        CkksContext::create_random_context(local_param);

    local_context.gen_rotation_keys();

    const double scale = local_param.default_scale();
    const int n_slot = N / 2;

    const int n_channel_per_ct =
        n_slot / (input_shape[0] * input_shape[1]);

    const int n_ct =
        div_ceil(n_channel, n_channel_per_ct);

    const int out_channels_per_ct =
        n_channel_per_ct * 4;

    const int n_out_ct =
        div_ceil(n_channel, out_channels_per_ct);

    const int n_select_pt =
        std::min(n_channel, out_channels_per_ct);

    REQUIRE(n_ct == 1);
    REQUIRE(n_out_ct == 1);

    // =====================================================
    // 3. Original alpha=7 polynomial coefficients
    // =====================================================
    const vector<double> coeff0 = {
        0.0,
        7.30445164958251,
        0.0,
        -34.6825871108659,
        0.0,
        59.8596518298826,
        0.0,
        -31.8755225906466
    };

    const vector<double> coeff1 = {
        0.0,
        2.40085652217597,
        0.0,
        -2.63125454261783,
        0.0,
        1.54912674773593,
        0.0,
        -0.331172956504304
    };

    // P0(d / (2B)): scale polynomial coefficients.
    // The encrypted d has ALREADY been range-normalized.
    auto make_scaled_coeff0 = [&](double B) {
        vector<double> result = coeff0;

        for (size_t i = 0; i < result.size(); ++i) {
            result[i] /= std::pow(
                2.0 * B,
                static_cast<int>(i)
            );
        }

        return result;
    };

    auto weight0_r1 = make_uniform_coeff(
        make_scaled_coeff0(B_r1), n_channel
    );

    auto weight0_r2 = make_uniform_coeff(
        make_scaled_coeff0(B_r2), n_channel
    );

    auto weight1 = make_uniform_coeff(
        coeff1, n_channel
    );

    // =====================================================
    // 4. Prepare Stockmeyer polynomial weights
    //
    // Input normalization: 19 -> 18
    // Round 1 P0:         18 -> 15
    // Round 1 P1:         15 -> 12
    // Round 1 d*P:        12 -> 11
    // Round 1 *0.5:       11 -> 10
    //
    // Round 2 P0:         10 -> 7
    // Round 2 P1:          7 -> 4
    // Round 2 d*P:         4 -> 3
    // Round 2 *0.5:        3 -> 2
    //
    // Repack:              2 -> 1
    // Output restoration:  1 -> 0
    // =====================================================
    Poly2D poly0_r1(
        local_param, input_shape, order,
        weight0_r1.copy(), skip,
        n_channel_per_ct, 18
    );

    Poly2D poly1_r1(
        local_param, input_shape, order,
        weight1.copy(), skip,
        n_channel_per_ct, 15
    );

    Poly2D poly0_r2(
        local_param, input_shape, order,
        weight0_r2.copy(), skip,
        n_channel_per_ct, 10
    );

    Poly2D poly1_r2(
        local_param, input_shape, order,
        weight1.copy(), skip,
        n_channel_per_ct, 7
    );

    poly0_r1.prepare_weight_stockmeyer();
    poly1_r1.prepare_weight_stockmeyer();
    poly0_r2.prepare_weight_stockmeyer();
    poly1_r2.prepare_weight_stockmeyer();

    // =====================================================
    // 5. Encode plaintext constants
    // =====================================================
    auto encode_constant = [&](double value,
                               int level,
                               int count) {
        vector<CkksPlaintext> result;
        result.reserve(count);

        vector<double> values(n_slot, value);

        for (int i = 0; i < count; ++i) {
            result.push_back(
                fhe_api::encode_ringt(local_context,
                    values,
                    local_param.q()[level]
                )
            );
        }

        return result;
    };

    // Actual encrypted input normalization
    auto input_norm_pt = encode_constant(
        input_multiplier, 19, n_ct
    );

    // Each binary Max ends with a multiplication by 0.5
    auto half_pt_r1 = encode_constant(
        0.5, 11, n_ct
    );

    auto half_pt_r2 = encode_constant(
        0.5, 3, n_ct
    );

    // Actual encrypted output scale restoration
    auto output_restore_pt = encode_constant(
        normalization_factor, 1, n_out_ct
    );

    // =====================================================
    // 6. Prepare 0/1 selection masks for MaxPooling
    // =====================================================
    Avgpool2DLayer mask_layer(
        input_shape, {2, 2}
    );

    mask_layer.prepare_weight_lazy(
        local_param,
        n_channel_per_ct,
        n_channel,
        2,
        skip,
        input_shape
    );

    vector<CkksPlaintext> select_pt;
    select_pt.reserve(n_select_pt);

    for (int i = 0; i < n_select_pt; ++i) {
        auto mask = mask_layer.select_tensor(i);

        // AvgPooling mask contains 0.25.
        // MaxPooling selection must contain 0 or 1.
        for (auto& v : mask) {
            v *= 4.0;
        }

        select_pt.push_back(
            fhe_api::encode_ringt(local_context,
                mask,
                local_param.q()[2]
            )
        );
    }

    // =====================================================
    // 7. Generate ORIGINAL input and encrypt directly
    // =====================================================
    Array<double, 3> input_array({
        static_cast<uint64_t>(n_channel),
        32,
        32
    });

    std::mt19937 rng(20261022);

    std::uniform_real_distribution<double> dis(
        -input_amplitude,
        input_amplitude
    );

    for (uint64_t i = 0; i < input_array.get_size(); ++i) {
        input_array.set(i, dis(rng));
    }

    Feature2DEncrypted input_feature(
        &local_context,
        init_level,
        skip
    );

    // Important: encrypt raw x, NOT x / 10.
    input_feature.pack_multiplexed(
        input_array,
        scale
    );

    REQUIRE(input_feature.data.size() == n_ct);

    // =====================================================
    // 8. Prepare output ciphertext
    // =====================================================
    Feature2DEncrypted output_feature(
        &local_context, 0
    );

    output_feature.shape = output_shape;
    output_feature.skip = {2, 2};
    output_feature.n_channel = n_channel;
    output_feature.n_channel_per_ct =
        out_channels_per_ct;

    for (int i = 0; i < n_out_ct; ++i) {
        output_feature.data.push_back(
            CkksCiphertext(local_param, 0)
        );
    }

    // =====================================================
    // 9. Load generated 19-Level FHE DAG
    // =====================================================
    fs::path project_path =
        base_path
        / "CKKS_poly_maxpool2x2_range_norm_stockmeyer"
        / "level_19";

    REQUIRE(fs::exists(project_path / "mega_ag.json"));

    auto arg_names = read_arg_names(project_path);

    vector<CxxVectorArgument> cxx_args;

    // Bind polynomial weight by its explicit argument name,
    // rather than relying on the order of arg_names.
    auto bind_poly = [&](const string& name,
                         const string& prefix,
                         Poly2D& poly) -> bool {
        if (name.rfind(prefix, 0) != 0) {
            return false;
        }

        int coeff_idx = std::stoi(
            name.substr(prefix.size())
        );

        REQUIRE(coeff_idx >= 0);
        REQUIRE(coeff_idx <= order);

        cxx_args.push_back({
            name, &poly.weight_pt.at(coeff_idx)
        });

        return true;
    };

    for (const auto& name : arg_names) {
        if (name == "input_node") {
            cxx_args.push_back({
                name, &input_feature.data
            });
        }
        else if (name == "input_norm_pt") {
            cxx_args.push_back({
                name, &input_norm_pt
            });
        }
        else if (name == "half_weight_pt_r1") {
            cxx_args.push_back({
                name, &half_pt_r1
            });
        }
        else if (name == "half_weight_pt_r2") {
            cxx_args.push_back({
                name, &half_pt_r2
            });
        }
        else if (name == "maxpool_select_pt") {
            cxx_args.push_back({
                name, &select_pt
            });
        }
        else if (name == "output_restore_pt") {
            cxx_args.push_back({
                name, &output_restore_pt
            });
        }
        else if (name == "output_ct") {
            cxx_args.push_back({
                name, &output_feature.data
            });
        }
        else if (
            bind_poly(
                name, "poly0_r1_weight_pt", poly0_r1
            )
        ) {}
        else if (
            bind_poly(
                name, "poly1_r1_weight_pt", poly1_r1
            )
        ) {}
        else if (
            bind_poly(
                name, "poly0_r2_weight_pt", poly0_r2
            )
        ) {}
        else if (
            bind_poly(
                name, "poly1_r2_weight_pt", poly1_r2
            )
        ) {}
        else {
            FAIL("Unknown DAG argument: " << name);
        }
    }

    REQUIRE(cxx_args.size() == arg_names.size());

    // =====================================================
    // 10. Execute encrypted computation on CPU
    // =====================================================
    std::cout
        << "N=" << N
        << ", level=" << init_level
        << ", input_multiplier=" << input_multiplier
        << ", output_multiplier=" << normalization_factor
        << ", B1=" << B_r1
        << ", B2=" << B_r2
        << std::endl;

    FheTaskCpu task(project_path.string());
    task.run(&local_context, cxx_args);

    // =====================================================
    // 11. Decrypt output
    //
    // NO post-decryption multiplication by 10.
    // Output restoration is already inside the FHE DAG.
    // =====================================================
    auto output_mg =
        output_feature.unpack_multiplexed();

    // =====================================================
    // 12. Plaintext polynomial reference
    // =====================================================
    auto eval_poly = [](
        const vector<double>& coeff, double x
    ) {
        double value = 0.0;

        for (int i = static_cast<int>(coeff.size()) - 1;
             i >= 0; --i) {
            value = value * x + coeff[i];
        }

        return value;
    };

    auto approx_max = [&](double a, double b, double B) {
        const double d = a - b;
        const double t = d / (2.0 * B);

        const double p0 = eval_poly(coeff0, t);
        const double sign_approx =
            eval_poly(coeff1, p0);

        return 0.5 * (a + b + d * sign_approx);
    };

    Array<double, 3> expected({
        static_cast<uint64_t>(n_channel),
        16,
        16
    });

    Array<double, 3> exact_max({
        static_cast<uint64_t>(n_channel),
        16,
        16
    });

    for (int ch = 0; ch < n_channel; ++ch) {
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 16; ++j) {
                const double a =
                    input_array.get(ch, 2 * i, 2 * j);
                const double b =
                    input_array.get(ch, 2 * i, 2 * j + 1);
                const double c =
                    input_array.get(ch, 2 * i + 1, 2 * j);
                const double d =
                    input_array.get(ch, 2 * i + 1, 2 * j + 1);

                const double m1 =
                    approx_max(a, b, reference_B_r1);

                const double m2 =
                    approx_max(c, d, reference_B_r1);

                const double approx_result =
                    approx_max(m1, m2, reference_B_r2);

                const double true_result =
                    std::max(
                        std::max(a, b),
                        std::max(c, d)
                    );

                expected.set(ch, i, j, approx_result);
                exact_max.set(ch, i, j, true_result);
            }
        }
    }

    // =====================================================
    // 13. Compare three kinds of errors
    // =====================================================
    auto fhe_vs_poly = compare(
        expected, output_mg
    );

    auto poly_vs_exact = compare(
        exact_max, expected
    );

    auto fhe_vs_exact = compare(
        exact_max, output_mg
    );

    std::cout << std::setprecision(12);

    std::cout
        << "\n=== FHE vs Plaintext Polynomial ===\n"
        << "max_error=" << fhe_vs_poly.max_error
        << ", rmse=" << fhe_vs_poly.rmse
        << "\n=== Polynomial vs Exact Max ===\n"
        << "max_error=" << poly_vs_exact.max_error
        << ", rmse=" << poly_vs_exact.rmse
        << "\n=== FHE vs Exact Max ===\n"
        << "max_error=" << fhe_vs_exact.max_error
        << ", rmse=" << fhe_vs_exact.rmse
        << std::endl;

    REQUIRE(std::isfinite(fhe_vs_poly.max_error));
    REQUIRE(std::isfinite(fhe_vs_poly.rmse));
    REQUIRE(std::isfinite(fhe_vs_exact.max_error));
    REQUIRE(std::isfinite(fhe_vs_exact.rmse));
    // V1 regression thresholds for CKKS numerical error
    REQUIRE(fhe_vs_poly.max_error < 1e-4);
    REQUIRE(fhe_vs_poly.rmse < 1e-5);
}



TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "conv1d", "", HeteroProcessors) {
    uint32_t n_in_channel = 4;
    uint32_t n_out_channel = 4;
    int init_level = 5;

    vector<uint32_t> input_shapes = {32, 64, 512};
    vector<uint32_t> kernel_shapes = {1, 4, 3, 5};
    vector<uint32_t> skips = {2, 4};
    vector<uint32_t> strides = {1, 2};

    FOR_EACH_SECTION(uint32_t input_shape, input_shapes, "input_shape=" + to_string(input_shape)) {
        FOR_EACH_SECTION(uint32_t kernel_shape, kernel_shapes, "kernel_shape=" + to_string(kernel_shape)) {
            FOR_EACH_SECTION(uint32_t skip, skips, "skip=" + to_string(skip)) {
                uint32_t n_channel_per_ct = div_ceil(this->N / 2, input_shape * skip);
                FOR_EACH_SECTION(uint32_t stride, strides, "stride=" + to_string(stride)) {
                    Array<double, 3> conv0_weight =
                        gen_random_array<3>({n_out_channel, n_in_channel, kernel_shape}, 1.0);
                    Array<double, 1> conv0_bias = gen_random_array<1>({n_out_channel}, 1.0);
                    Array<double, 2> input_array = gen_random_array<2>({n_in_channel, input_shape}, 1.0);

                    Feature1DEncrypted input_feature(&this->context, init_level, skip);
                    input_feature.pack(input_array);
                    Conv1DPackedLayer conv0_layer(this->context.parameter(), input_shape, move(conv0_weight),
                                                  move(conv0_bias), stride, skip, n_channel_per_ct, init_level);
                    conv0_layer.prepare_weight();

                    Feature1DEncrypted output_feature(&this->context, init_level - 1, skip * stride);
                    output_feature.shape = input_shape / stride;
                    output_feature.skip = skip * stride;
                    output_feature.n_channel = n_out_channel;
                    output_feature.n_channel_per_ct = n_channel_per_ct;
                    for (int i = 0; i < div_ceil(n_out_channel, n_channel_per_ct); i++) {
                        output_feature.data.emplace_back(this->context.parameter(), init_level - 1);
                        output_feature.data.back().set_scale(this->param.default_scale());
                    }

                    fs::path project_path =
                        base_path /
                        ("conv1d_input_shape_" + to_string(input_shape) + "_kernel_shape_" + to_string(kernel_shape) +
                         "_skip_" + to_string(skip) + "_stride_" + to_string(stride)) /
                        ("level_" + to_string(init_level)) / "server";

                    auto arg_names = read_arg_names(project_path);
                    vector<CxxVectorArgument> cxx_args;
                    cxx_args.push_back({arg_names[0], &input_feature.data});
                    cxx_args.push_back({arg_names[1], &conv0_layer.weight_pt});
                    cxx_args.push_back({arg_names[2], &conv0_layer.bias_pt});
                    cxx_args.push_back({arg_names[3], &output_feature.data});

                    this->run(project_path, cxx_args);

                    Array<double, 2> output_mg = output_feature.unpack();
                    Array<double, 2> plain_output = conv0_layer.run_plaintext(input_array);

                    print_double_message(output_mg.to_array_1d().data(), "output_mg", 10);
                    print_double_message(plain_output.to_array_1d().data(), "plain_output", 10);

                    auto compare_result = compare(plain_output, output_mg);
                    REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
                    REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
                }
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "multiplexed_conv1d", "", HeteroProcessors) {
    uint32_t n_in_channel = 16;
    uint32_t n_out_channel = 32;
    int init_level = 5;

    vector<uint32_t> input_shapes = {32, 64, 512};
    vector<uint32_t> kernel_shapes = {1, 3, 4, 5};
    vector<uint32_t> skips = {2, 4};
    vector<uint32_t> strides = {1, 2};

    FOR_EACH_SECTION(uint32_t input_shape, input_shapes, "input_shape=" + to_string(input_shape)) {
        FOR_EACH_SECTION_IF(uint32_t kernel_shape, kernel_shapes, kernel_shape <= input_shape,
                            "kernel_shape=" + to_string(kernel_shape)) {
            FOR_EACH_SECTION(uint32_t skip, skips, "skip=" + to_string(skip)) {
                uint32_t n_channel_per_ct = div_ceil(this->N / 2, input_shape);
                FOR_EACH_SECTION(uint32_t stride, strides, "stride=" + to_string(stride)) {
                    Array<double, 3> conv0_weight =
                        gen_random_array<3>({n_out_channel, n_in_channel, kernel_shape}, 1.0);
                    Array<double, 1> conv0_bias = gen_random_array<1>({n_out_channel}, 1.0);
                    Array<double, 2> input_array = gen_random_array<2>({n_in_channel, input_shape}, 1.0);

                    Feature1DEncrypted input_feature(&this->context, init_level, skip);
                    input_feature.pack_multiplexed(input_array, this->param.default_scale());
                    MultiplexedConv1DPackedLayer conv0_layer(this->context.parameter(), input_shape,
                                                             move(conv0_weight), move(conv0_bias), stride, skip,
                                                             n_channel_per_ct, init_level);
                    conv0_layer.prepare_weight();

                    bool needs_rearrange = (skip > 1 || stride > 1);
                    int output_level = needs_rearrange ? init_level - 2 : init_level - 1;
                    uint32_t n_block_per_ct = div_ceil(n_channel_per_ct, skip);
                    uint32_t n_output_cts = needs_rearrange ? div_ceil(n_out_channel, n_channel_per_ct) :
                                                              div_ceil(n_out_channel, n_block_per_ct);

                    Feature1DEncrypted output_feature(&this->context, output_level, skip * stride);
                    output_feature.shape = input_shape / stride;
                    output_feature.skip = skip * stride;
                    output_feature.n_channel = n_out_channel;
                    output_feature.n_channel_per_ct = n_channel_per_ct;
                    for (uint32_t i = 0; i < n_output_cts; i++) {
                        output_feature.data.emplace_back(this->context.parameter(), output_level);
                        output_feature.data.back().set_scale(this->param.default_scale());
                    }

                    uint32_t n_select_pt = min(n_block_per_ct, n_out_channel);
                    vector<CkksPlaintext> select_pt_subset;
                    for (int i = 0; i < n_select_pt; i++) {
                        select_pt_subset.push_back(move(conv0_layer.block_select_pt[i]));
                    }

                    fs::path project_path =
                        base_path /
                        ("multiplexed_conv1d_input_shape_" + to_string(input_shape) + "_kernel_shape_" +
                         to_string(kernel_shape) + "_skip_" + to_string(skip) + "_stride_" + to_string(stride)) /
                        ("level_" + to_string(init_level)) / "server";

                    auto arg_names = read_arg_names(project_path);
                    vector<CxxVectorArgument> cxx_args;
                    int idx = 0;
                    cxx_args.push_back({arg_names[idx++], &input_feature.data});
                    cxx_args.push_back({arg_names[idx++], &conv0_layer.weight_pt});
                    cxx_args.push_back({arg_names[idx++], &conv0_layer.bias_pt});
                    if (needs_rearrange) {
                        cxx_args.push_back({arg_names[idx++], &select_pt_subset});
                    }
                    cxx_args.push_back({arg_names[idx++], &output_feature.data});

                    this->run(project_path, cxx_args);

                    Array<double, 2> output_mg = output_feature.unpack_multiplexed();
                    Array<double, 2> plain_output = conv0_layer.run_plaintext(input_array);

                    print_double_message(output_mg.to_array_1d().data(), "output_mg", 10);
                    print_double_message(plain_output.to_array_1d().data(), "plain_output", 10);

                    auto compare_result = compare(plain_output, output_mg);
                    REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
                    REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
                }
            }
        }
    }
}
TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "add_layer", "", HeteroProcessors) {
    int init_level = 2;
    Duo skip = {1, 1};

    auto run_add_test = [&](uint32_t n_channel, uint32_t s) {
        Duo input_shape = {s, s};
        uint32_t n_channel_per_ct = div_ceil(this->n_slot, prod(input_shape));
        uint32_t n_ct = div_ceil(n_channel, n_channel_per_ct);

        Array<double, 3> input_x0 = gen_random_array<3>({n_channel, input_shape[0], input_shape[1]}, 1.0);
        Array<double, 3> input_x1 = gen_random_array<3>({n_channel, input_shape[0], input_shape[1]}, 1.0);

        Feature2DEncrypted x0_enc(&this->context, init_level, skip);
        x0_enc.pack_multiplexed(input_x0, this->param.default_scale());

        Feature2DEncrypted x1_enc(&this->context, init_level, skip);
        x1_enc.pack_multiplexed(input_x1, this->param.default_scale());

        // Pre-allocate output (add doesn't consume levels)
        Feature2DEncrypted output_feature(&this->context, init_level);
        for (uint32_t i = 0; i < n_ct; i++) {
            output_feature.data.emplace_back(this->context.parameter(), init_level);
            output_feature.data.back().set_scale(this->param.default_scale());
        }

        fs::path project_path =
            base_path / ("CKKS_add_layer/ch_" + to_string(n_channel) + "_shape_" + to_string(s) + "_" + to_string(s)) /
            ("level_" + to_string(init_level)) / "server";
        cout << "project_path=" << project_path << endl;
        auto arg_names = read_arg_names(project_path);
        // Python arg order: input_node1, input_node2, output_ct
        vector<CxxVectorArgument> cxx_args;
        for (const auto& name : arg_names) {
            if (name == "input_node1")
                cxx_args.push_back({name, &x0_enc.data});
            else if (name == "input_node2")
                cxx_args.push_back({name, &x1_enc.data});
            else if (name == "output_ct")
                cxx_args.push_back({name, &output_feature.data});
        }
        this->run(project_path, cxx_args);

        // Set output metadata
        output_feature.skip = skip;
        output_feature.n_channel = n_channel;
        output_feature.n_channel_per_ct = n_channel_per_ct;
        output_feature.shape = input_shape;
        auto result_mg = output_feature.unpack_multiplexed();

        AddLayer add_layer(this->param);
        auto result_expected = add_layer.run_plaintext(input_x0, input_x1);

        print_double_message(result_mg.to_array_1d().data(), "output_mg", 10);
        print_double_message(result_expected.to_array_1d().data(), "plain_output", 10);

        auto compare_result = compare(result_expected, result_mg);
        REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
        REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
    };

    SECTION("n_channel=4, shape=16x16") {
        run_add_test(4, 16);
    }
    SECTION("n_channel=4, shape=32x32") {
        run_add_test(4, 32);
    }
    SECTION("n_channel=32, shape=16x16") {
        run_add_test(32, 16);
    }
    SECTION("n_channel=32, shape=32x32") {
        run_add_test(32, 32);
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "par_block_col_major_add_generated",
                               "[block_col_major_e2e]",
                               HeteroProcessors) {
    auto& res = SharedHeteroResources::get();
    fs::path server_dir = base_path / "CKKS_par_block_col_major_add" / "par_block_col_major" / "level_2" / "server";
    if (!fs::exists(server_dir / "mega_ag.json"))
        return;

    const int init_level = 2;
    const uint32_t m = 8, n_heads = 2, head_dim = 4, d = 4;
    const uint32_t total_dim = n_heads * head_dim;
    auto x0 = gen_random_array<2>({m, total_dim}, 1.0);
    auto x1 = gen_random_array<2>({m, total_dim}, 1.0);

    FeatureMatEncrypted x0_enc(&res.context, init_level);
    x0_enc.par_block_col_major_pack(x0, d, n_heads, d, res.param.default_scale());
    FeatureMatEncrypted x1_enc(&res.context, init_level);
    x1_enc.par_block_col_major_pack(x1, d, n_heads, d, res.param.default_scale());

    ParBlockColMajorAdd add_layer(res.param);
    auto expected = add_layer.run_plaintext(x0, x1);

    vector<CkksCiphertext> x0_cts, x1_cts, out_cts;
    for (auto& ct : x0_enc.data)
        x0_cts.push_back(ct.copy());
    for (auto& ct : x1_enc.data)
        x1_cts.push_back(ct.copy());
    for (size_t i = 0; i < x0_enc.data.size(); i++)
        out_cts.emplace_back(res.context.parameter(), init_level);
        out_cts.back().set_scale(res.param.default_scale());

    vector<CxxVectorArgument> cxx_args = {
        {"input0", &x0_cts},
        {"input1", &x1_cts},
        {"output", &out_cts},
    };
    run_block_col_major_e2e_test(*this, server_dir, cxx_args, expected, {m, head_dim}, d, n_heads, true, out_cts);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "par_block_col_major_add_pt_generated",
                               "[block_col_major_e2e]",
                               HeteroProcessors) {
    auto& res = SharedHeteroResources::get();
    fs::path server_dir = base_path / "CKKS_par_block_col_major_add_pt" / "par_block_col_major" / "level_2" / "server";
    if (!fs::exists(server_dir / "mega_ag.json"))
        return;

    const int init_level = 2;
    const uint32_t m = 8, n_heads = 2, head_dim = 4, d = 4;
    const uint32_t total_dim = n_heads * head_dim;
    auto A = gen_random_array<2>({m, total_dim}, 1.0);
    auto B = gen_random_array<2>({m, total_dim}, 0.5);

    auto layer_ptr =
        std::make_shared<ParBlockColMajorAddPt>(res.param, Duo{m, total_dim}, d, n_heads, init_level, std::move(B));
    layer_ptr->precompute_pts();
    auto expected = layer_ptr->run_plaintext(A);

    FeatureMatEncrypted A_enc(&res.context, init_level);
    A_enc.par_block_col_major_pack(A, d, n_heads, d, res.param.default_scale());

    vector<CkksCiphertext> in_cts, out_cts;
    vector<CustomData> layer_data;
    for (auto& ct : A_enc.data)
        in_cts.push_back(ct.copy());
    layer_data.emplace_back(static_cast<void*>(layer_ptr.get()));
    for (size_t i = 0; i < A_enc.data.size(); i++)
        out_cts.emplace_back(res.context.parameter(), init_level);
        out_cts.back().set_scale(res.param.default_scale());

    vector<CxxVectorArgument> cxx_args = {
        {"input0", &in_cts},
        {"add_pt_0", &layer_data},
        {"output", &out_cts},
    };
    run_block_col_major_e2e_test(*this, server_dir, cxx_args, expected, {m, head_dim}, d, n_heads, true, out_cts);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "avgpool2d_layer", "", HeteroProcessors) {
    int init_level = 3;
    Duo skip = {1, 1};

    auto run_avgpool_test = [&](uint32_t n_channel, uint32_t s, const Duo& stride) {
        Duo input_shape = {s, s};
        uint32_t n_channel_per_ct = div_ceil(this->n_slot, prod(input_shape));
        uint32_t n_ct = div_ceil(n_channel, n_channel_per_ct);

        Array<double, 3> input_array = gen_random_array<3>({n_channel, input_shape[0], input_shape[1]}, 1.0);

        Feature2DEncrypted input_feature(&this->context, init_level, skip);
        input_feature.pack_multiplexed(input_array, this->param.default_scale());

        // Prepare select_tensor_pt via Avgpool2DLayer
        Avgpool2DLayer avgpool(input_shape, stride);
        avgpool.prepare_weight(this->param, n_channel_per_ct, n_channel, init_level, skip, input_shape);

        // Pre-allocate output (rescale consumes one level)
        uint32_t out_channels_per_ct = n_channel_per_ct * prod(stride);
        uint32_t n_packed_out_channel = div_ceil(n_channel, out_channels_per_ct);
        Feature2DEncrypted output_feature(&this->context, init_level - 1);
        for (uint32_t i = 0; i < n_packed_out_channel; i++) {
            output_feature.data.emplace_back(this->context.parameter(), init_level - 1);
            output_feature.data.back().set_scale(this->param.default_scale());
        }

        fs::path project_path = base_path /
                                ("CKKS_avgpool2d/stride_" + to_string(stride[0]) + "_" + to_string(stride[1]) + "/ch_" +
                                 to_string(n_channel) + "_shape_" + to_string(s) + "_" + to_string(s)) /
                                ("level_" + to_string(init_level)) / "server";
        cout << "project_path=" << project_path << endl;
        auto arg_names = read_arg_names(project_path);
        // Python arg order: input_node, select_tensor_pt, output_ct
        vector<CxxVectorArgument> cxx_args;
        for (const auto& name : arg_names) {
            if (name == "input_node")
                cxx_args.push_back({name, &input_feature.data});
            else if (name == "select_tensor_pt")
                cxx_args.push_back({name, &avgpool.select_tensor_pt});
            else if (name == "output_ct")
                cxx_args.push_back({name, &output_feature.data});
        }
        this->run(project_path, cxx_args);

        // Set output metadata: shape = input_shape/stride, skip = input_skip*stride
        output_feature.skip = skip * stride;
        output_feature.n_channel = n_channel;
        output_feature.n_channel_per_ct = n_channel_per_ct * prod(stride);
        output_feature.shape = input_shape / stride;
        auto result_mg = output_feature.unpack_multiplexed();

        auto result_expected = avgpool.run_plaintext_multiplexed(input_array);

        print_double_message(result_mg.to_array_1d().data(), "output_mg", 10);
        print_double_message(result_expected.to_array_1d().data(), "plain_output", 10);

        auto compare_result = compare(result_expected, result_mg);
        REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
        REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
    };

    vector<uint32_t> shapes = {8, 16, 32, 64};
    vector<uint32_t> channels = {4, 10, 15, 32, 37};
    vector<Duo> strides = {{2, 2}, {4, 4}, {8, 8}};

    FOR_EACH_SECTION(const auto& stride, strides, "stride=" + to_string(stride[0])) {
        FOR_EACH_SECTION(uint32_t n_channel, channels, "n_channel=" + to_string(n_channel)) {
            FOR_EACH_SECTION_IF(uint32_t s, shapes, s >= stride[0], "shape=" + to_string(s) + "x" + to_string(s)) {
                run_avgpool_test(n_channel, s, stride);
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "adaptive_avgpool2d_layer", "", HeteroProcessors) {
    int init_level = 3;
    Duo skip = {1, 1};

    auto run_adaptive_avgpool_test = [&](uint32_t n_channel, uint32_t s, const Duo& stride) {
        Duo input_shape = {s, s};
        uint32_t n_channel_per_ct = div_ceil(this->n_slot, prod(input_shape));
        uint32_t n_ct = div_ceil(n_channel, n_channel_per_ct);

        Array<double, 3> input_array = gen_random_array<3>({n_channel, input_shape[0], input_shape[1]}, 1.0);

        Feature2DEncrypted input_feature(&this->context, init_level, skip);
        input_feature.pack_multiplexed(input_array, this->param.default_scale());

        // No prepare_weight needed for adaptive avgpool
        Avgpool2DLayer avgpool(input_shape, stride);

        // Normalization consumes one level, with the same number of CTs.
        Feature2DEncrypted output_feature(&this->context, init_level - 1);
        for (uint32_t i = 0; i < n_ct; i++) {
            output_feature.data.emplace_back(this->context.parameter(), init_level - 1);
            output_feature.data.back().set_scale(this->param.default_scale());
        }

        fs::path project_path = base_path /
                                ("CKKS_adaptive_avgpool2d/stride_" + to_string(stride[0]) + "_" + to_string(stride[1]) +
                                 "/ch_" + to_string(n_channel) + "_shape_" + to_string(s) + "_" + to_string(s)) /
                                ("level_" + to_string(init_level)) / "server";
        cout << "project_path=" << project_path << endl;
        auto arg_names = read_arg_names(project_path);
        // Python arg order: input_node, output_ct (NO select_tensor_pt)
        vector<CxxVectorArgument> cxx_args;
        for (const auto& name : arg_names) {
            if (name == "input_node")
                cxx_args.push_back({name, &input_feature.data});
            else if (name == "output_ct")
                cxx_args.push_back({name, &output_feature.data});
        }
        this->run(project_path, cxx_args);

        // Set output metadata: invalid_fill = stride (key property of adaptive avgpool)
        output_feature.skip = skip * stride;
        output_feature.invalid_fill = stride;
        output_feature.n_channel = n_channel;
        output_feature.n_channel_per_ct = n_channel_per_ct;
        output_feature.shape = input_shape / stride;
        auto result_mg = output_feature.unpack_multiplexed();

        auto result_expected = avgpool.run_plaintext(input_array);

        print_double_message(result_mg.to_array_1d().data(), "output_mg", 10);
        print_double_message(result_expected.to_array_1d().data(), "plain_output", 10);

        auto compare_result = compare(result_expected, result_mg);
        REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
        REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
    };

    vector<uint32_t> channels = {4, 10, 15, 32, 37};
    vector<uint32_t> shapes = {8, 16, 32, 64};
    vector<Duo> strides = {{2, 2}, {4, 4}, {8, 8}};

    FOR_EACH_SECTION(const auto& stride, strides, "stride=" + to_string(stride[0])) {
        FOR_EACH_SECTION(uint32_t n_channel, channels, "n_channel=" + to_string(n_channel)) {
            FOR_EACH_SECTION_IF(uint32_t s, shapes, s >= stride[0], "shape=" + to_string(s) + "x" + to_string(s)) {
                run_adaptive_avgpool_test(n_channel, s, stride);
            }
        }
    }
}

TEST_CASE("rectangular_polynomial_channel_layout", "[spatial]") {
    auto param = CkksParameter::create_parameter(13, RingType::ConjugateInvariant);
    auto ctx = CkksContext::create_random_context(param);
    const Duo shape{4, 8};
    const int level = 5, channels = 11, order = 4;
    for (const Duo skip : {Duo{1, 2}, Duo{2, 4}}) {
        auto values = gen_random_array<3>({channels, shape[0], shape[1]}, 0.25);
        auto coefficients = gen_random_array<2>({order + 1, channels}, 0.1);
        Feature2DEncrypted input(&ctx, level, skip);
        input.pack_multiplexed(values, param.default_scale());
        PolyRelu2D layer(param, shape, order, move(coefficients), skip, input.n_channel_per_ct, level);
        auto expected = layer.run_plaintext_for_non_absorb_case(values);
        for (bool lazy : {false, true}) {
            if (lazy)
                layer.prepare_weight_bsgs_lazy();
            else
                layer.prepare_weight_bsgs();
            auto actual = layer.run_bsgs(ctx, input).unpack_multiplexed();
            INFO("skip=" << skip[0] << "," << skip[1] << " lazy=" << lazy);
            REQUIRE(compare(expected, actual).max_error < 1e-5);
        }
    }
}

TEST_CASE("rectangular_adaptive_avgpool", "[spatial]") {
    auto param = CkksParameter::create_parameter(13, RingType::ConjugateInvariant);
    auto ctx = CkksContext::create_random_context(param);
    ctx.gen_rotation_keys();
    const Duo skip{2, 4}, stride{2, 4};
    for (const Duo shape : {Duo{4, 8}, Duo{8, 4}}) {
        auto values = gen_random_array<3>({8, shape[0], shape[1]}, 1.0);
        Feature2DEncrypted input(&ctx, 1, skip);
        input.pack_multiplexed(values, param.default_scale());
        Avgpool2DLayer layer(shape, stride);
        const auto pt = Avgpool2DLayer::normalization_pt(ctx, prod(stride), input.level);
        REQUIRE(pt.scale() == Approx(param.q()[input.level]));
        auto encrypted_output = layer.run_adaptive_avgpool(ctx, input);
        REQUIRE(encrypted_output.level == input.level - 1);
        REQUIRE(encrypted_output.data[0].scale() == Approx(input.data[0].scale()).epsilon(1e-10));
        auto output = encrypted_output.unpack_multiplexed();
        layer.prepare_weight(param, input.n_channel_per_ct, input.n_channel, input.level, skip, shape);
        auto multiplexed = layer.run_multiplexed_avgpool(ctx, input).unpack_multiplexed();
        for (uint32_t channel = 0; channel < 8; ++channel) {
            for (uint32_t h = 0; h < shape[0] / stride[0]; ++h) {
                for (uint32_t w = 0; w < shape[1] / stride[1]; ++w) {
                    double expected = 0;
                    for (uint32_t dh = 0; dh < stride[0]; ++dh)
                        for (uint32_t dw = 0; dw < stride[1]; ++dw)
                            expected += values.get(channel, h * stride[0] + dh, w * stride[1] + dw);
                    expected /= prod(stride);
                    REQUIRE(std::abs(output.get(channel, h, w) - expected) < 1e-4);
                    REQUIRE(std::abs(multiplexed.get(channel, h, w) - expected) < 1e-4);
                }
            }
        }
    }
}

TEST_CASE("adaptive_avgpool1d_normalization", "[spatial]") {
    auto param = CkksParameter::create_parameter(13, RingType::ConjugateInvariant);
    auto ctx = CkksContext::create_random_context(param);
    ctx.gen_rotation_keys();
    const uint32_t shape = 16, stride = 4, channels = 8;
    auto values = gen_random_array<2>({channels, shape}, 2.0);
    Feature1DEncrypted input(&ctx, 2);
    input.pack_multiplexed(values, param.default_scale());
    Avgpool1DLayer layer(shape, stride);
    const auto pt = Avgpool1DLayer::normalization_pt(ctx, stride, input.level);
    REQUIRE(pt.scale() == Approx(param.q()[input.level]));
    auto result = layer.run_adaptive_avgpool(ctx, input);
    REQUIRE(result.level == input.level - 1);
    REQUIRE(result.data[0].scale() == Approx(input.data[0].scale()).epsilon(1e-10));
    auto output = result.unpack_multiplexed();
    auto expected = layer.run_plaintext(values);
    REQUIRE(compare(expected, output).max_error < 1e-4);
    layer.prepare_weight(param, input.n_channel_per_ct, channels, input.level, input.skip, shape);
    auto multiplexed = layer.run_multiplexed_avgpool(ctx, input).unpack_multiplexed();
    REQUIRE(compare(expected, multiplexed).max_error < 1e-4);
}

TEST_CASE("split_avgpool_normalization", "[spatial]") {
    auto param = CkksParameter::create_parameter(13, RingType::ConjugateInvariant);
    auto ctx = CkksContext::create_random_context(param);
    ctx.gen_rotation_keys();
    const Duo block{32, 256}, expansion{2, 2}, shape = block * expansion;
    auto values = gen_random_array<3>({2, shape[0], shape[1]}, 1.0);
    Feature2DEncrypted input(&ctx, 2);
    input.pack_interleaved(values, block, expansion, param.default_scale());
    for (const Duo stride : {Duo{2, 2}, Duo{4, 4}}) {
        Avgpool2DLayer layer(shape, stride);
        if (stride[0] > expansion[0])
            layer.prepare_weight_repack(param, input.n_channel, input.level, stride / expansion, block);
        auto result = layer.run_split_avgpool(ctx, input, expansion);
        REQUIRE(result.level == input.level - 1);
        REQUIRE(result.data[0].scale() == Approx(input.data[0].scale()).epsilon(1e-10));
        auto output = layer.need_repack ? result.unpack_multiplexed()
                                       : result.unpack_interleaved(block, expansion / stride);
        REQUIRE(compare(layer.run_plaintext(values), output).max_error < 1e-4);
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "interleaved_avgpool2d_layer", "", HeteroProcessors) {
    int init_level = 3;

    auto run_interleaved_avgpool_test = [&](uint32_t n_channel, const Duo& stride, const Duo& block_shape,
                                            uint32_t mult) {
        Duo input_shape = block_shape * mult;
        Duo block_expansion = input_shape / block_shape;

        Array<double, 3> input_array = gen_random_array<3>({n_channel, input_shape[0], input_shape[1]}, 1.0);

        Feature2DEncrypted input_feature(&this->context, init_level);
        Duo total_stride = block_expansion;
        input_feature.pack_interleaved(input_array, block_shape, total_stride, this->param.default_scale());

        Avgpool2DLayer avgpool(block_shape, stride);

        // Normalization consumes one level.
        uint32_t out_size = input_feature.data.size() / prod(stride);
        Feature2DEncrypted output_feature(&this->context, init_level - 1);
        for (uint32_t i = 0; i < out_size; i++) {
            output_feature.data.emplace_back(this->context.parameter(), init_level - 1);
            output_feature.data.back().set_scale(this->param.default_scale());
        }

        fs::path project_path =
            base_path /
            ("CKKS_interleaved_avgpool2d/stride_" + to_string(stride[0]) + "_" + to_string(stride[1]) + "/ch_" +
             to_string(n_channel) + "/block_shape_" + to_string(block_shape[0]) + "_" + to_string(block_shape[1]) +
             "/input_shape_" + to_string(input_shape[0]) + "_" + to_string(input_shape[1])) /
            ("level_" + to_string(init_level)) / "server";
        cout << "project_path=" << project_path << endl;

        auto arg_names = read_arg_names(project_path);
        vector<CxxVectorArgument> cxx_args;
        for (const auto& name : arg_names) {
            if (name == "input_node")
                cxx_args.push_back({name, &input_feature.data});
            else if (name == "output_ct")
                cxx_args.push_back({name, &output_feature.data});
        }
        this->run(project_path, cxx_args);

        // Set output metadata
        Duo out_expansion = block_expansion / stride;
        output_feature.n_channel = n_channel;
        output_feature.n_channel_per_ct = 1;
        output_feature.skip = {1, 1};
        output_feature.shape = input_shape / stride;
        auto result_mg = output_feature.unpack_interleaved(block_shape, out_expansion);

        auto result_expected = avgpool.run_plaintext(input_array);

        print_double_message(result_mg.to_array_1d().data(), "output_mg", 10);
        print_double_message(result_expected.to_array_1d().data(), "plain_output", 10);

        auto compare_result = compare(result_expected, result_mg);
        REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
        REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
    };

    vector<uint32_t> channels = {2, 4, 8};
    vector<Duo> strides = {{2, 2}, {4, 4}};
    Duo block_shape = {64, 64};
    vector<uint32_t> multipliers = {2, 4};

    FOR_EACH_SECTION(const auto& stride, strides, "stride=" + to_string(stride[0])) {
        FOR_EACH_SECTION(uint32_t n_channel, channels, "ch=" + to_string(n_channel)) {
            FOR_EACH_SECTION_IF(uint32_t mult, multipliers, mult >= stride[0], "mult=" + to_string(mult)) {
                run_interleaved_avgpool_test(n_channel, stride, block_shape, mult);
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "adaptive_avgpool1d_layer", "", HeteroProcessors) {
    int init_level = 3;
    uint32_t skip = 1;

    auto run_adaptive_avgpool1d_test = [&](uint32_t n_channel, uint32_t s, uint32_t stride) {
        uint32_t n_channel_per_ct = div_ceil(this->n_slot, s);
        uint32_t n_ct = div_ceil(n_channel, n_channel_per_ct);

        Array<double, 2> input_array = gen_random_array<2>({n_channel, s}, 1.0);

        Feature1DEncrypted input_feature(&this->context, init_level, skip);
        input_feature.pack_multiplexed(input_array, this->param.default_scale());

        Avgpool1DLayer avgpool(s, stride);

        Feature1DEncrypted output_feature(&this->context, init_level - 1, skip * stride, stride);
        for (uint32_t i = 0; i < n_ct; i++) {
            output_feature.data.emplace_back(this->context.parameter(), init_level - 1);
            output_feature.data.back().set_scale(this->param.default_scale());
        }

        fs::path project_path = base_path / ("CKKS_adaptive_avgpool1d/stride_" + to_string(stride)) /
                                ("ch_" + to_string(n_channel) + "_shape_" + to_string(s)) /
                                ("level_" + to_string(init_level)) / "server";
        cout << "project_path=" << project_path << endl;
        auto arg_names = read_arg_names(project_path);
        vector<CxxVectorArgument> cxx_args;
        for (const auto& name : arg_names) {
            if (name == "input_node")
                cxx_args.push_back({name, &input_feature.data});
            else if (name == "output_ct")
                cxx_args.push_back({name, &output_feature.data});
        }
        this->run(project_path, cxx_args);

        output_feature.skip = skip * stride;
        output_feature.invalid_fill = stride;
        output_feature.n_channel = n_channel;
        output_feature.n_channel_per_ct = n_channel_per_ct;
        output_feature.shape = s / stride;
        auto result_mg = output_feature.unpack_multiplexed();

        auto result_expected = avgpool.run_plaintext(input_array);

        print_double_message(result_mg.to_array_1d().data(), "output_mg", 10);
        print_double_message(result_expected.to_array_1d().data(), "plain_output", 10);

        auto compare_result = compare(result_expected, result_mg);
        REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
        REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
    };

    vector<uint32_t> channels = {4, 10, 15, 32};
    vector<uint32_t> shapes = {8, 16, 32, 64};
    vector<uint32_t> strides = {2, 4, 8};

    FOR_EACH_SECTION(uint32_t stride, strides, "stride=" + to_string(stride)) {
        FOR_EACH_SECTION(uint32_t n_channel, channels, "n_channel=" + to_string(n_channel)) {
            FOR_EACH_SECTION_IF(uint32_t s, shapes, s >= stride, "shape=" + to_string(s)) {
                run_adaptive_avgpool1d_test(n_channel, s, stride);
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "multiplexed_avgpool1d_layer", "", HeteroProcessors) {
    int init_level = 3;
    uint32_t skip = 1;

    auto run_multiplexed_avgpool1d_test = [&](uint32_t n_channel, uint32_t s, uint32_t stride) {
        uint32_t n_channel_per_ct = div_ceil(this->n_slot, s);

        Array<double, 2> input_array = gen_random_array<2>({n_channel, s}, 1.0);

        Feature1DEncrypted input_feature(&this->context, init_level, skip);
        input_feature.pack_multiplexed(input_array, this->param.default_scale());

        Avgpool1DLayer avgpool(s, stride);
        avgpool.prepare_weight(this->param, n_channel_per_ct, n_channel, init_level, skip, s);

        uint32_t out_channels_per_ct = n_channel_per_ct * stride;
        uint32_t n_packed_out_channel = div_ceil(n_channel, out_channels_per_ct);
        Feature1DEncrypted output_feature(&this->context, init_level - 1, skip * stride);
        for (uint32_t i = 0; i < n_packed_out_channel; i++) {
            output_feature.data.emplace_back(this->context.parameter(), init_level - 1);
            output_feature.data.back().set_scale(this->param.default_scale());
        }

        fs::path project_path = base_path / ("CKKS_avgpool1d/stride_" + to_string(stride)) /
                                ("ch_" + to_string(n_channel) + "_shape_" + to_string(s)) /
                                ("level_" + to_string(init_level)) / "server";
        cout << "project_path=" << project_path << endl;
        auto arg_names = read_arg_names(project_path);
        vector<CxxVectorArgument> cxx_args;
        for (const auto& name : arg_names) {
            if (name == "input_node")
                cxx_args.push_back({name, &input_feature.data});
            else if (name == "select_tensor_pt")
                cxx_args.push_back({name, &avgpool.select_tensor_pt});
            else if (name == "output_ct")
                cxx_args.push_back({name, &output_feature.data});
        }
        this->run(project_path, cxx_args);

        output_feature.skip = skip * stride;
        output_feature.n_channel = n_channel;
        output_feature.n_channel_per_ct = n_channel_per_ct * stride;
        output_feature.shape = s / stride;
        auto result_mg = output_feature.unpack_multiplexed();

        auto result_expected = avgpool.run_plaintext_multiplexed(input_array);

        print_double_message(result_mg.to_array_1d().data(), "output_mg", 10);
        print_double_message(result_expected.to_array_1d().data(), "plain_output", 10);

        auto compare_result = compare(result_expected, result_mg);
        REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
        REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
    };

    vector<uint32_t> shapes = {8, 16, 32, 64};
    vector<uint32_t> channels = {4, 10, 15, 32};
    vector<uint32_t> strides = {2, 4, 8};

    FOR_EACH_SECTION(uint32_t stride, strides, "stride=" + to_string(stride)) {
        FOR_EACH_SECTION(uint32_t n_channel, channels, "n_channel=" + to_string(n_channel)) {
            FOR_EACH_SECTION_IF(uint32_t s, shapes, s >= stride, "shape=" + to_string(s)) {
                run_multiplexed_avgpool1d_test(n_channel, s, stride);
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "concat_layer", "", HeteroProcessors) {
    int init_level = 2;
    Duo skip = {1, 1};

    auto run_concat_test = [&](uint32_t n_channel_1, uint32_t n_channel_2, uint32_t s) {
        Duo input_shape = {s, s};

        Array<double, 3> input_x1 = gen_random_array<3>({n_channel_1, input_shape[0], input_shape[1]}, 1.0);
        Array<double, 3> input_x2 = gen_random_array<3>({n_channel_2, input_shape[0], input_shape[1]}, 1.0);

        Feature2DEncrypted x1_enc(&this->context, init_level, skip);
        x1_enc.pack_multiplexed(input_x1, this->param.default_scale());

        Feature2DEncrypted x2_enc(&this->context, init_level, skip);
        x2_enc.pack_multiplexed(input_x2, this->param.default_scale());

        ConcatLayer concat;
        Feature2DEncrypted result_enc = concat.run(this->context, x1_enc, x2_enc);

        auto result_mg = result_enc.unpack_multiplexed();
        auto result_expected = concat.concatenate_channels(input_x1, input_x2);

        print_double_message(result_mg.to_array_1d().data(), "output_mg", 10);
        print_double_message(result_expected.to_array_1d().data(), "plain_output", 10);

        auto compare_result = compare(result_expected, result_mg);
        REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
        REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
    };

    SECTION("n_ch1=8, n_ch2=8, shape=32x32") {
        run_concat_test(8, 8, 32);
    }
    SECTION("n_ch1=8, n_ch2=16, shape=32x32") {
        run_concat_test(8, 16, 32);
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "mult_scalar_layer", "", HeteroProcessors) {
    int init_level = 3;
    Duo skip = {1, 1};

    auto run_mult_scalar_test = [&](uint32_t n_channel, uint32_t s) {
        Duo input_shape = {s, s};
        uint32_t n_channel_per_ct = div_ceil(this->n_slot, prod(input_shape));
        uint32_t n_ct = div_ceil(n_channel, n_channel_per_ct);

        Array<double, 3> input_array = gen_random_array<3>({n_channel, input_shape[0], input_shape[1]}, 1.0);
        Array<double, 1> weight = gen_random_array<1>({n_channel}, 1.0);

        Feature2DEncrypted input_feature(&this->context, init_level, skip);
        input_feature.pack_multiplexed(input_array, this->param.default_scale());

        // Prepare weight plaintexts
        MultScalarLayer mult_layer(this->param, input_shape, move(weight), skip, n_channel_per_ct, init_level);
        mult_layer.prepare_weight();

        // Pre-allocate output (mult_scalar consumes one level due to rescale)
        Feature2DEncrypted output_feature(&this->context, init_level - 1);
        for (uint32_t i = 0; i < n_ct; i++) {
            output_feature.data.emplace_back(this->context.parameter(), init_level - 1);
            output_feature.data.back().set_scale(this->param.default_scale());
        }

        fs::path project_path =
            base_path /
            ("CKKS_mult_scalar/ch_" + to_string(n_channel) + "_shape_" + to_string(s) + "_" + to_string(s)) /
            ("level_" + to_string(init_level)) / "server";
        cout << "project_path=" << project_path << endl;
        auto arg_names = read_arg_names(project_path);
        // Python arg order: input_node, weight_pt, output_ct
        vector<CxxVectorArgument> cxx_args;
        for (const auto& name : arg_names) {
            if (name == "input_node")
                cxx_args.push_back({name, &input_feature.data});
            else if (name == "weight_pt")
                cxx_args.push_back({name, &mult_layer.weight_pt});
            else if (name == "output_ct")
                cxx_args.push_back({name, &output_feature.data});
        }
        this->run(project_path, cxx_args);

        // Set output metadata
        output_feature.skip = skip;
        output_feature.n_channel = n_channel;
        output_feature.n_channel_per_ct = n_channel_per_ct;
        output_feature.shape = input_shape;
        auto result_mg = output_feature.unpack_multiplexed();

        auto result_expected = mult_layer.run_plaintext(input_array);

        print_double_message(result_mg.to_array_1d().data(), "output_mg", 10);
        print_double_message(result_expected.to_array_1d().data(), "plain_output", 10);

        auto compare_result = compare(result_expected, result_mg);
        REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
        REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
    };

    SECTION("n_channel=32, shape=32x32") {
        run_mult_scalar_test(32, 32);
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "mux_dw_conv1d", "", HeteroProcessors) {
    int init_level = 5;

    vector<uint32_t> channels = {4, 16, 32};
    vector<uint32_t> input_shapes = {32, 64, 512};
    vector<uint32_t> kernel_shapes = {1, 3, 5};
    vector<uint32_t> skips = {2, 4};
    vector<uint32_t> strides = {1, 2};

    FOR_EACH_SECTION(uint32_t n_channel, channels, "ch=" + to_string(n_channel)) {
        FOR_EACH_SECTION(uint32_t input_shape, input_shapes, "input_shape=" + to_string(input_shape)) {
            FOR_EACH_SECTION_IF(uint32_t kernel_shape, kernel_shapes, kernel_shape <= input_shape,
                                "kernel_shape=" + to_string(kernel_shape)) {
                FOR_EACH_SECTION(uint32_t skip, skips, "skip=" + to_string(skip)) {
                    uint32_t n_channel_per_ct = div_ceil(this->N / 2, input_shape);
                    FOR_EACH_SECTION(uint32_t stride, strides, "stride=" + to_string(stride)) {
                        // weight: [n_channel, 1, kernel_shape]
                        Array<double, 3> weight = gen_random_array<3>({n_channel, 1, kernel_shape}, 1.0);
                        Array<double, 1> bias = gen_random_array<1>({n_channel}, 1.0);
                        Array<double, 2> input_array = gen_random_array<2>({n_channel, input_shape}, 1.0);

                        Feature1DEncrypted input_feature(&this->context, init_level, skip);
                        input_feature.pack_multiplexed(input_array, this->param.default_scale());

                        MultiplexedDWConv1DPackedLayer conv_layer(this->context.parameter(), input_shape,
                                                                  move(weight), move(bias), stride, skip,
                                                                  n_channel_per_ct, init_level);
                        conv_layer.prepare_weight();

                        bool needs_rearrange = (skip > 1 || stride > 1);
                        int output_level = needs_rearrange ? init_level - 2 : init_level - 1;
                        uint32_t n_block_per_ct = div_ceil(n_channel_per_ct, skip);
                        uint32_t n_packed_ct = div_ceil(n_channel, n_channel_per_ct);
                        uint32_t n_output_cts = needs_rearrange ? div_ceil(n_channel, n_channel_per_ct) : n_packed_ct;

                        Feature1DEncrypted output_feature(&this->context, output_level, skip * stride);
                        output_feature.shape = input_shape / stride;
                        output_feature.skip = skip * stride;
                        output_feature.n_channel = n_channel;
                        output_feature.n_channel_per_ct = n_channel_per_ct;
                        for (uint32_t i = 0; i < n_output_cts; i++) {
                            output_feature.data.emplace_back(this->context.parameter(), output_level);
                            output_feature.data.back().set_scale(this->param.default_scale());
                        }

                        uint32_t n_select_pt = min(n_channel_per_ct, n_channel);
                        vector<CkksPlaintext> select_pt_subset;
                        for (uint32_t i = 0; i < n_select_pt; i++) {
                            select_pt_subset.push_back(move(conv_layer.block_select_pt[i]));
                        }

                        fs::path project_path = base_path /
                                                ("mux_dw_conv1d_ch_" + to_string(n_channel) + "_input_" +
                                                 to_string(input_shape) + "_kernel_" + to_string(kernel_shape) +
                                                 "_skip_" + to_string(skip) + "_stride_" + to_string(stride)) /
                                                ("level_" + to_string(init_level)) / "server";

                        auto arg_names = read_arg_names(project_path);
                        vector<CxxVectorArgument> cxx_args;
                        int idx = 0;
                        cxx_args.push_back({arg_names[idx++], &input_feature.data});
                        cxx_args.push_back({arg_names[idx++], &conv_layer.weight_pt});
                        cxx_args.push_back({arg_names[idx++], &conv_layer.bias_pt});
                        if (needs_rearrange) {
                            cxx_args.push_back({arg_names[idx++], &select_pt_subset});
                        }
                        cxx_args.push_back({arg_names[idx++], &output_feature.data});

                        this->run(project_path, cxx_args);

                        Array<double, 2> output_mg = output_feature.unpack_multiplexed();
                        Array<double, 2> plain_output = conv_layer.run_plaintext(input_array);

                        print_double_message(output_mg.to_array_1d().data(), "output_mg", 10);
                        print_double_message(plain_output.to_array_1d().data(), "plain_output", 10);

                        auto compare_result = compare(plain_output, output_mg);
                        REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
                        REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
                    }
                }
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "fc_1d_multiplexed", "", HeteroProcessors) {
    // Dense 1D multiplexed: Feature1DEncrypted (n_channel × shape) → Feature0DEncrypted (n_out scalars)
    // block_stride = skip  (skip already contains invalid_fill)
    // block_size   = shape * skip
    // n_block_per_ct = N/2 / block_size
    // valid_sub      = skip / invalid_fill
    // n_valid_per_ct = n_block_per_ct * valid_sub  (channels with data per CT)
    // weight shape: (n_out, n_channel * shape)  — channel-major flatten
    int init_level = 2;

    struct Cfg {
        uint32_t shape;
        uint32_t skip;
        uint32_t invalid_fill;
        uint32_t n_channel;  // channels in input Feature1DEncrypted
        uint32_t n_out;
    };
    vector<Cfg> configs = {
        {32, 2, 1, 16, 8},  {32, 2, 1, 64, 32}, {32, 4, 1, 32, 16}, {32, 4, 2, 32, 16},
        {32, 4, 4, 32, 16}, {64, 2, 1, 16, 8},  {64, 4, 2, 64, 32}, {64, 8, 4, 64, 32},
    };

    FOR_EACH_SECTION(auto& cfg, configs,
                     "shape=" + to_string(cfg.shape) + " skip=" + to_string(cfg.skip) +
                         " inv=" + to_string(cfg.invalid_fill) + " ch=" + to_string(cfg.n_channel) +
                         " out=" + to_string(cfg.n_out)) {
        // Layout (corrected: block_stride = skip, not skip*invalid_fill)
        uint32_t block_size = cfg.shape * cfg.skip;
        uint32_t n_block_per_ct = this->n_slot / block_size;
        uint32_t valid_sub = cfg.skip / cfg.invalid_fill;
        uint32_t n_valid_per_ct = n_block_per_ct * valid_sub;
        uint32_t n_input_ct = div_ceil(cfg.n_channel, n_valid_per_ct);
        uint32_t n_in_feature = cfg.n_channel * cfg.shape;  // weight second dim
        uint32_t n_packed_out = div_ceil(cfg.n_out, n_block_per_ct);

        // Random data
        auto input_2d = gen_random_array<2>({cfg.n_channel, cfg.shape}, 1.0);
        auto weight = gen_random_array<2>({cfg.n_out, n_in_feature}, 0.5);
        auto bias = gen_random_array<1>({cfg.n_out}, 0.1);

        // Flatten input for plaintext reference (channel-major: matches in_flat = in_ch*shape + data_idx)
        auto input_1d = Array<double, 1>::from_array_1d(input_2d.to_array_1d());

        // Pack input into Feature1DEncrypted
        Feature1DEncrypted input_feature(&this->context, init_level, cfg.skip, cfg.invalid_fill);
        input_feature.pack_multiplexed(input_2d, this->param.default_scale());

        // Wrap as Feature0DEncrypted (run_1d_multiplexed only uses .data)
        Feature0DEncrypted input_0d(&this->context, init_level);
        input_0d.data = move(input_feature.data);
        input_0d.n_channel = cfg.n_channel;
        input_0d.n_channel_per_ct = n_valid_per_ct;
        input_0d.skip = 1;

        // Layer
        DensePackedLayer dense(this->context.parameter(), move(weight), move(bias), n_block_per_ct, init_level, 0);
        dense.prepare_weight_for_1d_multiplexed(cfg.shape, cfg.skip, cfg.invalid_fill);

        Feature0DEncrypted output(&this->context, init_level - 1);
        output.skip = block_size;
        output.n_channel = cfg.n_out;
        output.n_channel_per_ct = n_block_per_ct;
        for (uint32_t i = 0; i < n_packed_out; i++) {
            output.data.emplace_back(this->context.parameter(), init_level - 1);
            output.data.back().set_scale(this->param.default_scale());
        }

        fs::path project_path =
            base_path /
            ("CKKS_fc_1d_multiplexed_shape" + to_string(cfg.shape) + "_skip" + to_string(cfg.skip) + "_inv" +
             to_string(cfg.invalid_fill) + "_cin" + to_string(cfg.n_channel) + "_cout" + to_string(cfg.n_out)) /
            ("level_" + to_string(init_level)) / "server";
        cout << "project_path=" << project_path << endl;
        auto arg_names = read_arg_names(project_path);
        vector<CxxVectorArgument> cxx_args;
        for (const auto& name : arg_names) {
            if (name == "input_node")
                cxx_args.push_back({name, &input_0d.data});
            else if (name == "weight_pt")
                cxx_args.push_back({name, &dense.weight_pt});
            else if (name == "bias_pt")
                cxx_args.push_back({name, &dense.bias_pt});
            else if (name == "output_ct")
                cxx_args.push_back({name, &output.data});
        }
        this->run(project_path, cxx_args);

        // Unpack and compare
        Array<double, 1> output_mg = output.unpack();
        Array<double, 1> plain_output = dense.run_plaintext(input_1d);

        print_double_message(output_mg.to_array_1d().data(), "output_mg", 10);
        print_double_message(plain_output.to_array_1d().data(), "plain_output", 10);

        auto cmp = compare(plain_output, output_mg);
        REQUIRE(cmp.max_error < 5.0e-2 * cmp.max_abs);
        REQUIRE(cmp.rmse < 1.0e-2 * cmp.rms);
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "upsample_layer", "", HeteroProcessors) {
    int init_level = 2;
    Duo upsample_factor = {2, 2};
    Duo stride = {2, 2};
    uint32_t n_channel = 4;

    auto run_upsample_test = [&](uint32_t s) {
        Duo input_shape = {s, s};
        int n_channel_per_ct = 1;

        Array<double, 3> input_array = gen_random_array<3>({n_channel, input_shape[0], input_shape[1]}, 1.0);

        Feature2DEncrypted input_feature(&this->context, init_level);
        Duo block_shape = input_shape / stride;
        input_feature.pack_interleaved(input_array, block_shape, stride, this->param.default_scale());

        UpsampleLayer upsample(this->param, stride, upsample_factor, init_level, n_channel, n_channel_per_ct);
        upsample.prepare_data();
        Feature2DEncrypted result_enc = upsample.run(this->context, input_feature);

        Duo out_stride = stride * upsample_factor;
        auto result_mg = result_enc.unpack_interleaved(block_shape, out_stride);
        auto result_expected = upsample.upsample_with_zero(input_array);

        print_double_message(result_mg.to_array_1d().data(), "output_mg", 10);
        print_double_message(result_expected.to_array_1d().data(), "plain_output", 10);

        auto compare_result = compare(result_expected, result_mg);
        REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
        REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
    };

    SECTION("shape=16x16") {
        run_upsample_test(16);
    }
    SECTION("shape=32x32") {
        run_upsample_test(32);
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "upsample_nearest_layer", "", HeteroProcessors) {
    int init_level = 2;
    Duo upsample_factor = {2, 2};
    // skip must be >= upsample_factor, because output skip = input skip / upsample_factor
    Duo skip = {2, 2};

    auto run_upsample_nearest_test = [&](uint32_t n_channel, uint32_t s) {
        Duo input_shape = {s, s};
        uint32_t n_channel_per_ct = div_ceil(this->n_slot, prod(input_shape));

        Array<double, 3> input_array = gen_random_array<3>({n_channel, input_shape[0], input_shape[1]}, 1.0);

        Feature2DEncrypted input_feature(&this->context, init_level, skip);
        input_feature.pack_multiplexed(input_array, this->param.default_scale());

        UpsampleNearestLayer upsample_nearest(this->param, input_shape, skip, upsample_factor, n_channel_per_ct,
                                              init_level);
        upsample_nearest.prepare_weight_lazy();
        Feature2DEncrypted result_enc = upsample_nearest.run(this->context, input_feature);

        auto result_mg = result_enc.unpack_multiplexed();
        auto result_expected = upsample_nearest.run_plaintext(input_array);

        print_double_message(result_mg.to_array_1d().data(), "output_mg", 10);
        print_double_message(result_expected.to_array_1d().data(), "plain_output", 10);

        auto compare_result = compare(result_expected, result_mg);
        REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
        REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
    };

    SECTION("n_channel=4, shape=8x8") {
        run_upsample_nearest_test(4, 8);
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "block_col_major_layernorm", "", HeteroProcessors) {
    // LayerNorm needs higher levels; use local context with N=32768
    const int N = 32768;
    CkksParameter param = CkksParameter::create_parameter(static_cast<int>(std::log2(N)), RingType::ConjugateInvariant);
    CkksContext context = CkksContext::create_random_context(param);
    context.gen_rotation_keys();
    double default_scale = param.default_scale();

    auto run_ln_test = [&](uint32_t d, uint32_t m, uint32_t n, uint32_t num_iters, int init_level) {
        double eps = 1e-5;
        double var_std_bound = 4.0;
        double inv_var = 1.0 / (var_std_bound * var_std_bound);
        double inv_std = 1.0 / var_std_bound;
        double c0 = 6.19067182, c1 = -16.15885111, c2 = 11.52830778;

        // Generate random data
        Array<double, 2> X_mat = gen_random_array<2>({m, n}, 1.0);
        Array<double, 1> gamma = gen_random_array<1>({n}, 0.5);
        Array<double, 1> beta = gen_random_array<1>({n}, 0.1);

        // Encrypt
        FeatureMatEncrypted X_enc(&context, init_level);
        X_enc.shape = {m, n};
        X_enc.matmul_block_size = d;
        X_enc.block_col_major_pack(X_mat, d, default_scale);

        // Phase 1a: Stats (a_cts only)
        BlockColMajorLNStats stats(param, {m, n}, d, init_level, eps, inv_var);
        stats.prepare_weight();
        auto a_cts = stats.run(context, X_enc);

        // Phase 1b: XCentered
        BlockColMajorLNXCentered xc(param, {m, n}, d, init_level);
        xc.prepare_weight();
        auto x_centered = xc.run(context, X_enc);

        // Phase 2: Minimax Init, 2 levels consumpsion
        BlockColMajorLNMinimaxInit minimax(param, d, init_level - 3, c0, c1, c2);
        minimax.prepare_weight();
        auto y_cts = minimax.run(context, a_cts);

        // Phase 3: Goldschmidt (3 levels per iteration)
        for (uint32_t k = 0; k < num_iters; k++) {
            uint32_t y_level = init_level - 5 - 3 * k;
            BlockColMajorLNGoldschmidt gold(param, d, y_level);
            gold.prepare_weight();
            y_cts = gold.run(context, y_cts, a_cts);
        }

        // Phase 4: Affine
        uint32_t y_final_level = init_level - 5 - 3 * num_iters;
        BlockColMajorLNAffine affine(param, {m, n}, d, y_final_level, inv_std, gamma.copy(), beta.copy());
        affine.prepare_weight();
        FeatureMatEncrypted result_enc = affine.run(context, x_centered, y_cts);

        // Unpack
        auto result = result_enc.block_col_major_unpack(m, n, d);

        // Compute expected (same algorithm: biased var E[x^2]-E[x]^2, minimax, goldschmidt)
        Array<double, 2> expected({(uint64_t)m, (uint64_t)n});
        for (uint32_t i = 0; i < m; i++) {
            double sum_x = 0.0, sum_x2 = 0.0;
            for (uint32_t j = 0; j < n; j++) {
                double v = X_mat.get(i, j);
                sum_x += v;
                sum_x2 += v * v;
            }
            double mean = sum_x / n;
            double var = sum_x2 / n - mean * mean;
            double a = (var + eps) * inv_var;

            double y = c0 + c1 * a + c2 * a * a;
            for (uint32_t k = 0; k < num_iters; k++) {
                y = 0.5 * y * (3.0 - a * y * y);
            }

            for (uint32_t j = 0; j < n; j++) {
                double x_norm = (X_mat.get(i, j) - mean) * inv_std * y;
                expected.set(i, j, x_norm * gamma.get(j) + beta.get(j));
            }
        }

        print_double_message(result.to_array_1d().data(), "output_mg", 10);
        print_double_message(expected.to_array_1d().data(), "output_expected", 10);

        auto compare_result = compare(expected, result);
        std::cout << "max_error=" << compare_result.max_error << " max_abs=" << compare_result.max_abs
                  << " rmse=" << compare_result.rmse << " rms=" << compare_result.rms << std::endl;
        REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
        REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
    };

    SECTION("d=64, m=197, n=192, iters=1") {
        run_ln_test(64, 197, 192, 1, 10);
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "par_block_col_major_layernorm",
                               "[block_col_major_e2e]",
                               HeteroProcessors) {
    fs::path server_dir =
        base_path / "CKKS_par_block_col_major_layernorm" / "seq_197_heads_3_dim_64" / "level_11" / "server";
    if (!fs::exists(server_dir / "mega_ag.json"))
        return;

    const int N = 32768;
    CkksParameter param = CkksParameter::create_parameter(static_cast<int>(std::log2(N)), RingType::ConjugateInvariant);
    CkksContext context = CkksContext::create_random_context(param);
    context.gen_rotation_keys();

    const uint32_t seq_len = 197, n_heads = 3, head_dim = 64, d = 64, num_iters = 1;
    const uint32_t total_dim = n_heads * head_dim;
    const int init_level = 11;
    const double eps = 1e-5;
    const double var_std_bound = 4.0;
    const double inv_var = 1.0 / (var_std_bound * var_std_bound);
    const double inv_std = 1.0 / var_std_bound;
    const double c0 = 6.19067182, c1 = -16.15885111, c2 = 11.52830778;

    auto X_mat = gen_random_array<2>({seq_len, total_dim}, 1.0);
    auto gamma = gen_random_array<1>({total_dim}, 0.5);
    auto beta = gen_random_array<1>({total_dim}, 0.1);

    Array<double, 2> expected({(uint64_t)seq_len, (uint64_t)total_dim});
    for (uint32_t i = 0; i < seq_len; i++) {
        double sum_x = 0.0, sum_x2 = 0.0;
        for (uint32_t j = 0; j < total_dim; j++) {
            double v = X_mat.get(i, j);
            sum_x += v;
            sum_x2 += v * v;
        }
        double mean = sum_x / total_dim;
        double var = sum_x2 / total_dim - mean * mean;
        double a = (var + eps) * inv_var;
        double y = c0 + c1 * a + c2 * a * a;
        for (uint32_t k = 0; k < num_iters; k++) {
            y = 0.5 * y * (3.0 - a * y * y);
        }
        for (uint32_t j = 0; j < total_dim; j++) {
            double x_norm = (X_mat.get(i, j) - mean) * inv_std * y;
            expected.set(i, j, x_norm * gamma.get(j) + beta.get(j));
        }
    }

    auto stats_layer =
        std::make_shared<ParBlockColMajorLNStats>(param, Duo{seq_len, total_dim}, d, n_heads, init_level, eps, inv_var);
    auto xcenter_layer =
        std::make_shared<ParBlockColMajorLNXCentered>(param, Duo{seq_len, total_dim}, d, n_heads, init_level);
    auto minimax_layer = std::make_shared<ParBlockColMajorLNMinimaxInit>(param, d, init_level - 4, c0, c1, c2);
    auto gold_layer = std::make_shared<ParBlockColMajorLNGoldschmidt>(param, d, init_level - 6);
    auto affine_layer = std::make_shared<ParBlockColMajorLNAffine>(param, Duo{seq_len, total_dim}, d, n_heads,
                                                                   init_level - 9, inv_std, gamma.copy(), beta.copy());

    FeatureMatEncrypted X_enc(&context, init_level);
    X_enc.shape = {seq_len, head_dim};
    X_enc.matmul_block_size = d;
    X_enc.par_block_col_major_pack(X_mat, d, n_heads, d, param.default_scale());

    vector<CkksCiphertext> in_cts, out_cts;
    vector<CustomData> stats_data, xcenter_data, minimax_data, gold_data, affine_data;
    for (auto& ct : X_enc.data)
        in_cts.push_back(ct.copy());
    stats_data.emplace_back(static_cast<void*>(stats_layer.get()));
    xcenter_data.emplace_back(static_cast<void*>(xcenter_layer.get()));
    minimax_data.emplace_back(static_cast<void*>(minimax_layer.get()));
    gold_data.emplace_back(static_cast<void*>(gold_layer.get()));
    affine_data.emplace_back(static_cast<void*>(affine_layer.get()));

    uint32_t num_block_rows = div_ceil(seq_len, d);
    uint32_t num_block_cols = div_ceil(head_dim, d);
    uint32_t n_h_padded = 1;
    while (n_h_padded < n_heads)
        n_h_padded <<= 1;
    uint32_t n_slot = param.max_slots();
    if (n_slot < n_h_padded * d * d && n_slot / (d * d) == 1) {
        n_h_padded = n_heads;
    }
    uint32_t G = (n_slot >= n_h_padded * d * d) ? 1 : n_h_padded / (n_slot / (d * d));
    uint32_t n_out = num_block_rows * num_block_cols * G;
    for (uint32_t i = 0; i < n_out; i++)
        out_cts.emplace_back(context.parameter(), 0);
        out_cts.back().set_scale(param.default_scale());

    vector<CxxVectorArgument> cxx_args = {
        {"input", &in_cts},
        {"_ln_stats_layer", &stats_data},
        {"_ln_xcenter_layer", &xcenter_data},
        {"_ln_minimax_layer", &minimax_data},
        {"_ln_gold_layer", &gold_data},
        {"_ln_affine_layer", &affine_data},
        {"output", &out_cts},
    };

    std::unordered_map<std::string, ExecutorFunc> executors;
    executors["encode_pt"] = make_block_col_major_encode_pt_executor();
    if constexpr (is_same_v<TestType, ProcessorCpu>) {
        FheTaskCpu task(server_dir.string());
        task.bind_custom_executors(executors);
        task.run(&context, cxx_args);
#ifdef INFERENCE_SDK_ENABLE_GPU
    } else if constexpr (is_same_v<TestType, ProcessorGpu>) {
        FheTaskGpu task(server_dir.string());
        task.bind_custom_executors(executors);
        task.run(&context, cxx_args);
#endif
    }

    FeatureMatEncrypted out_enc(&context, 0);
    for (auto& ct : out_cts)
        out_enc.data.push_back(std::move(ct));
    out_enc.head_shape = {seq_len, head_dim};
    out_enc.shape = {seq_len, total_dim};
    out_enc.matmul_block_size = d;
    auto result = out_enc.par_block_col_major_unpack(seq_len, head_dim, d, n_heads);

    print_double_message(result.to_array_1d().data(), "output_mg", 10);
    print_double_message(expected.to_array_1d().data(), "output_expected", 10);

    auto compare_result = compare(expected, result);
    std::cout << "max_error=" << compare_result.max_error << " max_abs=" << compare_result.max_abs
              << " rmse=" << compare_result.rmse << " rms=" << compare_result.rms << std::endl;
    REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
    REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "block_col_major_polyactrn", "", HeteroProcessors) {
    auto run_polyactrn_test = [&](uint32_t d, uint32_t m, uint32_t n, uint32_t degree, int init_level) {
        // Generate random data
        Array<double, 2> X_mat = gen_random_array<2>({m, n}, 1.0);
        Array<double, 1> gamma = gen_random_array<1>({n}, 0.5);
        Array<double, 2> coeffs = gen_random_array<2>({degree + 1, n}, 0.3);

        // Encrypt
        FeatureMatEncrypted X_enc(&this->context, init_level);
        X_enc.shape = {m, n};
        X_enc.matmul_block_size = d;
        X_enc.block_col_major_pack(X_mat, d, this->default_scale);

        // Phase 1: Gamma scaling
        BlockColMajorPolyActRNGamma gamma_layer(this->param, {m, n}, d, init_level, gamma.copy());
        gamma_layer.prepare_weight();
        auto gamma_result = gamma_layer.run(this->context, X_enc);

        // Phase 2: Polynomial
        uint32_t poly_level = init_level - 1;
        BlockColMajorPolyActRNPoly poly(this->param, {m, n}, d, poly_level, coeffs.copy(), degree);
        poly.prepare_weight();
        auto result_enc = poly.run(this->context, gamma_result);

        // Unpack
        auto result = result_enc.block_col_major_unpack(m, n, d);

        // Compute expected: poly(gamma * x)
        Array<double, 2> expected({(uint64_t)m, (uint64_t)n});
        for (uint32_t i = 0; i < m; i++) {
            for (uint32_t j = 0; j < n; j++) {
                double gx = gamma.get(j) * X_mat.get(i, j);
                double poly_val = 0.0;
                double gx_pow = 1.0;
                for (uint32_t k = 0; k <= degree; k++) {
                    poly_val += coeffs.get(k, j) * gx_pow;
                    gx_pow *= gx;
                }
                expected.set(i, j, poly_val);
            }
        }

        print_double_message(result.to_array_1d().data(), "output_mg", 10);
        print_double_message(expected.to_array_1d().data(), "output_expected", 10);

        auto compare_result = compare(expected, result);
        std::cout << "max_error=" << compare_result.max_error << " max_abs=" << compare_result.max_abs
                  << " rmse=" << compare_result.rmse << " rms=" << compare_result.rms << std::endl;
        REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
        REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
    };

    SECTION("degree=2, d=64, m=197, n=192") {
        run_polyactrn_test(64, 197, 192, 2, 3);
    }
    SECTION("degree=4, d=64, m=197, n=192") {
        run_polyactrn_test(64, 197, 192, 4, 4);
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "par_block_col_major_polyactrn",
                               "[block_col_major_e2e]",
                               HeteroProcessors) {
    auto& res = SharedHeteroResources::get();

    auto run_par_polyactrn_test = [&](uint32_t degree, int init_level) {
        fs::path server_dir = base_path / "CKKS_par_block_col_major_polyactrn" / ("degree_" + to_string(degree)) /
                              ("level_" + to_string(init_level)) / "server";
        if (!fs::exists(server_dir / "mega_ag.json"))
            return;

        const uint32_t d = 64, seq_len = 197, n_heads = 3, head_dim = 64;
        const uint32_t total_dim = n_heads * head_dim;

        Array<double, 2> X_mat = gen_random_array<2>({seq_len, total_dim}, 1.0);
        Array<double, 1> gamma = gen_random_array<1>({total_dim}, 0.5);
        Array<double, 2> coeffs = gen_random_array<2>({degree + 1, total_dim}, 0.3);
        // Compute expected: poly(gamma * x)
        Array<double, 2> expected({(uint64_t)seq_len, (uint64_t)total_dim});
        for (uint32_t i = 0; i < seq_len; i++) {
            for (uint32_t j = 0; j < total_dim; j++) {
                double gx = gamma.get(j) * X_mat.get(i, j);
                double poly_val = 0.0;
                double gx_pow = 1.0;
                for (uint32_t k = 0; k <= degree; k++) {
                    poly_val += coeffs.get(k, j) * gx_pow;
                    gx_pow *= gx;
                }
                expected.set(i, j, poly_val);
            }
        }

        auto gamma_layer = std::make_shared<ParBlockColMajorPolyActRNGamma>(res.param, Duo{seq_len, total_dim}, d,
                                                                            n_heads, 1, init_level, gamma.copy());
        auto poly_layer = std::make_shared<ParBlockColMajorPolyActRNPoly>(
            res.param, Duo{seq_len, total_dim}, d, n_heads, 1, init_level - 1, coeffs.copy(), degree);

        FeatureMatEncrypted X_enc(&res.context, init_level);
        X_enc.shape = {seq_len, head_dim};
        X_enc.matmul_block_size = d;
        X_enc.par_block_col_major_pack(X_mat, d, n_heads, d, res.param.default_scale());

        vector<CkksCiphertext> in_cts, out_cts;
        vector<CustomData> gamma_data, poly_data;
        for (auto& ct : X_enc.data)
            in_cts.push_back(ct.copy());
        gamma_data.emplace_back(static_cast<void*>(gamma_layer.get()));
        poly_data.emplace_back(static_cast<void*>(poly_layer.get()));

        uint32_t num_block_rows = div_ceil(seq_len, d);
        uint32_t num_block_cols = div_ceil(head_dim, d);
        uint32_t n_h_padded = 1;
        while (n_h_padded < n_heads)
            n_h_padded <<= 1;
        uint32_t n_slot = res.param.max_slots();
        if (n_slot < n_h_padded * d * d && n_slot / (d * d) == 1) {
            n_h_padded = n_heads;
        }
        uint32_t G = (n_slot >= n_h_padded * d * d) ? 1 : n_h_padded / (n_slot / (d * d));
        uint32_t n_out = num_block_rows * num_block_cols * G;
        int out_level = init_level - 1 - (degree == 4 ? 3 : 2);
        for (uint32_t i = 0; i < n_out; i++)
            out_cts.emplace_back(res.context.parameter(), out_level);
            out_cts.back().set_scale(res.param.default_scale());

        string gamma_layer_id = "_gamma_layer_deg" + to_string(degree);
        string poly_layer_id = "_poly_layer_deg" + to_string(degree);
        vector<CxxVectorArgument> cxx_args = {
            {"input", &in_cts},
            {gamma_layer_id, &gamma_data},
            {poly_layer_id, &poly_data},
            {"output", &out_cts},
        };
        run_block_col_major_e2e_test(*this, server_dir, cxx_args, expected, {seq_len, head_dim}, d, n_heads, true,
                                     out_cts);
    };

    SECTION("degree=2, d=64, seq=197, heads=3, head_dim=64") {
        run_par_polyactrn_test(2, 3);
    }
    SECTION("degree=4, d=64, seq=197, heads=3, head_dim=64") {
        run_par_polyactrn_test(4, 4);
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "par_cpmm_expand_polyactrn_reduce",
                               "[block_col_major_e2e]",
                               HeteroProcessors) {
    auto& res = SharedHeteroResources::get();

    auto run_test = [&](uint32_t d, uint32_t seq_len, uint32_t n_heads, uint32_t head_dim, uint32_t K, uint32_t degree,
                        int init_level) {
        fs::path server_dir = base_path /
                              ("CKKS_par_cpmm_expand_polyactrn_reduce_m" + to_string(seq_len) + "_heads" +
                               to_string(n_heads) + "_dim" + to_string(head_dim) + "_K" + to_string(K)) /
                              ("degree_" + to_string(degree)) / ("level_" + to_string(init_level)) / "server";
        if (!fs::exists(server_dir / "mega_ag.json"))
            return;

        uint32_t total_dim = n_heads * head_dim;
        uint32_t expanded_dim = K * total_dim;

        Array<double, 2> A_mat = gen_random_array<2>({seq_len, total_dim}, 0.5);
        Array<double, 2> W_expand = gen_random_array<2>({total_dim, expanded_dim}, 0.1);
        Array<double, 1> gamma = gen_random_array<1>({expanded_dim}, 0.3);
        Array<double, 2> coeffs = gen_random_array<2>({degree + 1, expanded_dim}, 0.2);
        Array<double, 2> W_reduce = gen_random_array<2>({expanded_dim, total_dim}, 0.1);

        std::vector<double> mid(seq_len * expanded_dim, 0.0);
        for (uint32_t i = 0; i < seq_len; i++)
            for (uint32_t j = 0; j < expanded_dim; j++) {
                double s = 0;
                for (uint32_t k = 0; k < total_dim; k++)
                    s += A_mat.get(i, k) * W_expand.get(k, j);
                mid[i * expanded_dim + j] = s;
            }

        std::vector<double> poly_out(seq_len * expanded_dim, 0.0);
        for (uint32_t i = 0; i < seq_len; i++)
            for (uint32_t j = 0; j < expanded_dim; j++) {
                double gx = gamma.get(j) * mid[i * expanded_dim + j];
                double val = 0.0, gx_pow = 1.0;
                for (uint32_t c = 0; c <= degree; c++) {
                    val += coeffs.get(c, j) * gx_pow;
                    gx_pow *= gx;
                }
                poly_out[i * expanded_dim + j] = val;
            }

        Array<double, 2> expected({(uint64_t)seq_len, (uint64_t)total_dim});
        for (uint32_t i = 0; i < seq_len; i++)
            for (uint32_t j = 0; j < total_dim; j++) {
                double s = 0;
                for (uint32_t k = 0; k < expanded_dim; k++)
                    s += poly_out[i * expanded_dim + k] * W_reduce.get(k, j);
                expected.set(i, j, s);
            }

        auto expand_ptr =
            std::make_shared<ParBlockColMajorCPMM>(res.param, Duo{seq_len, head_dim}, W_expand, d, n_heads, init_level);
        expand_ptr->precompute_diagonals();

        int gamma_level = init_level - 2;
        auto gamma_ptr = std::make_shared<ParBlockColMajorPolyActRNGamma>(res.param, Duo{seq_len, expanded_dim}, d,
                                                                          n_heads, K, gamma_level, gamma.copy());

        int poly_level = init_level - 3;
        auto poly_ptr = std::make_shared<ParBlockColMajorPolyActRNPoly>(res.param, Duo{seq_len, expanded_dim}, d,
                                                                        n_heads, K, poly_level, coeffs.copy(), degree);

        int reduce_level = (degree == 2) ? init_level - 5 : init_level - 6;
        auto reduce_ptr = std::make_shared<ParBlockColMajorCPMM>(res.param, Duo{seq_len, head_dim}, W_reduce, d,
                                                                 n_heads, reduce_level);
        reduce_ptr->precompute_diagonals();

        FeatureMatEncrypted A_enc(&res.context, init_level);
        A_enc.shape = {seq_len, head_dim};
        A_enc.matmul_block_size = d;
        A_enc.par_block_col_major_pack(A_mat, d, n_heads, d, res.param.default_scale());

        vector<CkksCiphertext> in_cts, out_cts;
        vector<CustomData> expand_data, gamma_data, poly_data, reduce_data;
        for (auto& ct : A_enc.data)
            in_cts.push_back(ct.copy());
        expand_data.emplace_back(static_cast<void*>(expand_ptr.get()));
        gamma_data.emplace_back(static_cast<void*>(gamma_ptr.get()));
        poly_data.emplace_back(static_cast<void*>(poly_ptr.get()));
        reduce_data.emplace_back(static_cast<void*>(reduce_ptr.get()));

        uint32_t num_block_rows_A = div_ceil(seq_len, d);
        uint32_t n_h_padded = 1;
        while (n_h_padded < n_heads)
            n_h_padded <<= 1;
        uint32_t n_slot = res.param.max_slots();
        if (n_slot < n_h_padded * d * d && n_slot / (d * d) == 1) {
            n_h_padded = n_heads;
        }
        uint32_t G = (n_slot >= n_h_padded * d * d) ? 1 : n_h_padded / (n_slot / (d * d));
        uint32_t n_out = num_block_rows_A * G;
        int out_level = reduce_level - 2;
        for (uint32_t i = 0; i < n_out; i++)
            out_cts.emplace_back(res.context.parameter(), out_level);
            out_cts.back().set_scale(res.param.default_scale());

        vector<CxxVectorArgument> cxx_args = {
            {"input", &in_cts},          {"_expand_cpmm", &expand_data}, {"_gamma_layer", &gamma_data},
            {"_poly_layer", &poly_data}, {"_reduce_cpmm", &reduce_data}, {"output", &out_cts},
        };
        run_block_col_major_e2e_test(*this, server_dir, cxx_args, expected, {seq_len, head_dim}, d, n_heads, true,
                                     out_cts);
    };

    SECTION("degree=2, K=2, d=16, seq=53, heads=3, head_dim=16") {
        run_test(16, 53, 3, 16, 2, 2, 7);
    }
    SECTION("degree=4, K=2, d=16, seq=53, heads=3, head_dim=16") {
        run_test(16, 53, 3, 16, 2, 4, 8);
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "single_par_block_col_major_cpmm",
                               "[block_col_major_e2e]",
                               HeteroProcessors) {
    auto& res = SharedHeteroResources::get();
    fs::path server_dir = base_path / "CKKS_par_block_col_major_cpmm" / "level_5" / "server";
    if (!fs::exists(server_dir / "mega_ag.json"))
        return;

    const uint32_t m = 8, n_per_head = 4, n_heads = 2, d = 4;
    const uint32_t n_total = n_heads * n_per_head;
    const int level = 5;

    auto W = gen_random_array<2>({n_total, n_total}, 0.1);
    auto A = gen_random_array<2>({m, n_total}, 1.0);

    auto layer_ptr = std::make_shared<ParBlockColMajorCPMM>(res.param, Duo{m, n_per_head}, W, d, n_heads, level);
    layer_ptr->precompute_diagonals();
    auto ref_output = layer_ptr->run_plaintext(A);

    FeatureMatEncrypted A_enc(&res.context, level);
    A_enc.par_block_col_major_pack(A, d, n_heads, d, res.param.default_scale());

    static vector<CkksCiphertext> in_cts, out_cts;
    static vector<CustomData> layer_data;
    in_cts.clear();
    out_cts.clear();
    layer_data.clear();

    for (auto& ct : A_enc.data)
        in_cts.push_back(ct.copy());
    layer_data.emplace_back(static_cast<void*>(layer_ptr.get()));

    uint32_t num_block_rows_A = div_ceil(m, d);
    uint32_t n_h_padded = 1;
    while (n_h_padded < n_heads)
        n_h_padded <<= 1;
    uint32_t n_slot = res.param.max_slots();
    if (n_slot < n_h_padded * d * d && n_slot / (d * d) == 1) {
        n_h_padded = n_heads;
    }
    uint32_t G = (n_slot >= n_h_padded * d * d) ? 1 : n_h_padded / (n_slot / (d * d));
    uint32_t n_out = num_block_rows_A * G;

    for (uint32_t i = 0; i < n_out; i++)
        out_cts.emplace_back(res.context.parameter(), level - 2);
        out_cts.back().set_scale(res.param.default_scale());

    vector<CxxVectorArgument> cxx_args = {
        {"input", &in_cts},
        {"_cpmm_layer", &layer_data},
        {"output", &out_cts},
    };
    run_block_col_major_e2e_test(*this, server_dir, cxx_args, ref_output, {m, n_per_head}, d, n_heads, true, out_cts);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "single_par_block_col_major_transpose",
                               "[block_col_major_e2e]",
                               HeteroProcessors) {
    auto& res = SharedHeteroResources::get();
    fs::path server_dir = base_path / "CKKS_par_block_col_major_transpose" / "level_5" / "server";
    if (!fs::exists(server_dir / "mega_ag.json"))
        return;

    const uint32_t M = 8, K = 8, d = 4, n_heads = 2;
    const int level = 5;

    auto A = gen_random_array<2>({M, n_heads * K}, 1.0);

    auto layer_ptr = std::make_shared<ParBlockColMajorTranspose>(res.param, Duo{M, K}, d, n_heads, level);
    auto ref_output = layer_ptr->run_plaintext(A);

    FeatureMatEncrypted A_enc(&res.context, level);
    A_enc.par_block_col_major_pack(A, d, n_heads, K, res.param.default_scale());

    static vector<CkksCiphertext> in_cts, out_cts;
    static vector<CustomData> layer_data;
    in_cts.clear();
    out_cts.clear();
    layer_data.clear();

    for (auto& ct : A_enc.data)
        in_cts.push_back(ct.copy());
    layer_data.emplace_back(static_cast<void*>(layer_ptr.get()));

    for (uint32_t i = 0; i < in_cts.size(); i++)
        out_cts.emplace_back(res.context.parameter(), level - 1);
        out_cts.back().set_scale(res.param.default_scale());

    vector<CxxVectorArgument> cxx_args = {
        {"input", &in_cts},
        {"_transpose_layer", &layer_data},
        {"output", &out_cts},
    };
    run_block_col_major_e2e_test(*this, server_dir, cxx_args, ref_output, {K, M}, d, n_heads, true, out_cts);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "single_par_block_col_major_ccmm",
                               "[block_col_major_e2e]",
                               HeteroProcessors) {
    auto& res = SharedHeteroResources::get();
    fs::path server_dir = base_path / "CKKS_par_block_col_major_ccmm" / "level_7" / "server";
    if (!fs::exists(server_dir / "mega_ag.json"))
        return;

    const uint32_t M = 8, N_dim = 8, P = 8, d = 4, n_heads = 2;
    const int level = 7;

    auto A = gen_random_array<2>({M, n_heads * N_dim}, 1.0);
    auto B = gen_random_array<2>({N_dim, n_heads * P}, 0.1);

    auto layer_ptr = std::make_shared<ParBlockColMajorCCMM>(res.param, Duo{M, N_dim}, Duo{N_dim, P}, d, n_heads, level);
    layer_ptr->precompute_diagonals();
    auto ref_output = layer_ptr->run_plaintext(A, B);

    FeatureMatEncrypted A_enc(&res.context, level);
    A_enc.par_block_col_major_pack(A, d, n_heads, N_dim, res.param.default_scale());
    FeatureMatEncrypted B_enc(&res.context, level);
    B_enc.par_block_col_major_pack(B, d, n_heads, P, res.param.default_scale());

    static vector<CkksCiphertext> A_cts, B_cts, out_cts;
    static vector<CustomData> layer_data;
    A_cts.clear();
    B_cts.clear();
    out_cts.clear();
    layer_data.clear();

    for (auto& ct : A_enc.data)
        A_cts.push_back(ct.copy());
    for (auto& ct : B_enc.data)
        B_cts.push_back(ct.copy());
    layer_data.emplace_back(static_cast<void*>(layer_ptr.get()));

    uint32_t n_out = div_ceil(M, d) * div_ceil(P, d);
    uint32_t n_h_padded = 1;
    while (n_h_padded < n_heads)
        n_h_padded <<= 1;
    uint32_t n_slot = res.param.max_slots();
    if (n_slot < n_h_padded * d * d && n_slot / (d * d) == 1) {
        n_h_padded = n_heads;
    }
    uint32_t G = (n_slot >= n_h_padded * d * d) ? 1 : n_h_padded / (n_slot / (d * d));
    n_out *= G;

    for (uint32_t i = 0; i < n_out; i++)
        out_cts.emplace_back(res.context.parameter(), level - 3);
        out_cts.back().set_scale(res.param.default_scale());

    vector<CxxVectorArgument> cxx_args = {
        {"A_input", &A_cts},
        {"B_input", &B_cts},
        {"_ccmm_layer", &layer_data},
        {"output", &out_cts},
    };
    run_block_col_major_e2e_test(*this, server_dir, cxx_args, ref_output, {M, P}, d, n_heads, true, out_cts);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture,
                               "single_par_block_col_major_polyactrn_poly",
                               "[block_col_major_e2e]",
                               HeteroProcessors) {
    auto& res = SharedHeteroResources::get();

    auto run_generated_polyactrn_poly = [&](uint32_t degree, int level) {
        fs::path server_dir = base_path / "CKKS_par_block_col_major_polyactrn_poly" / ("degree_" + to_string(degree)) /
                              ("level_" + to_string(level)) / "server";
        if (!fs::exists(server_dir / "mega_ag.json"))
            return;

        const uint32_t seq_len = 8, n_heads = 2, head_dim = 4, d = 4;
        const uint32_t total_dim = n_heads * head_dim;

        auto X = gen_random_array<2>({seq_len, total_dim}, 0.5);
        auto coeffs = gen_random_array<2>({degree + 1, total_dim}, 0.2);

        Array<double, 2> ref_output({(uint64_t)seq_len, (uint64_t)total_dim});
        for (uint32_t i = 0; i < seq_len; i++) {
            for (uint32_t j = 0; j < total_dim; j++) {
                double x_pow = 1.0;
                double y = 0.0;
                for (uint32_t k = 0; k <= degree; k++) {
                    y += coeffs.get(k, j) * x_pow;
                    x_pow *= X.get(i, j);
                }
                ref_output.set(i, j, y);
            }
        }

        auto layer_ptr = std::make_shared<ParBlockColMajorPolyActRNPoly>(res.param, Duo{seq_len, total_dim}, d, n_heads,
                                                                         1, level, coeffs.copy(), degree);

        FeatureMatEncrypted X_enc(&res.context, level);
        X_enc.shape = {seq_len, head_dim};
        X_enc.matmul_block_size = d;
        X_enc.par_block_col_major_pack(X, d, n_heads, d, res.param.default_scale());

        static vector<CkksCiphertext> in_cts, out_cts;
        static vector<CustomData> layer_data;
        in_cts.clear();
        out_cts.clear();
        layer_data.clear();

        for (auto& ct : X_enc.data)
            in_cts.push_back(ct.copy());
        layer_data.emplace_back(static_cast<void*>(layer_ptr.get()));

        uint32_t num_block_rows = div_ceil(seq_len, d);
        uint32_t num_block_cols = div_ceil(head_dim, d);
        uint32_t n_h_padded = 1;
        while (n_h_padded < n_heads)
            n_h_padded <<= 1;
        uint32_t n_slot = res.param.max_slots();
        if (n_slot < n_h_padded * d * d && n_slot / (d * d) == 1) {
            n_h_padded = n_heads;
        }
        uint32_t G = (n_slot >= n_h_padded * d * d) ? 1 : n_h_padded / (n_slot / (d * d));
        uint32_t n_out = num_block_rows * num_block_cols * G;
        int out_level = level - (degree == 4 ? 3 : 2);

        for (uint32_t i = 0; i < n_out; i++)
            out_cts.emplace_back(res.context.parameter(), out_level);
            out_cts.back().set_scale(res.param.default_scale());

        string layer_id = "_poly_layer_deg" + to_string(degree);
        vector<CxxVectorArgument> cxx_args = {
            {"input", &in_cts},
            {layer_id, &layer_data},
            {"output", &out_cts},
        };
        run_block_col_major_e2e_test(*this, server_dir, cxx_args, ref_output, {seq_len, head_dim}, d, n_heads, true,
                                     out_cts);
    };

    SECTION("degree=2, level=3") {
        run_generated_polyactrn_poly(2, 3);
    }
    SECTION("degree=4, level=4") {
        run_generated_polyactrn_poly(4, 4);
    }
}

TEST_CASE("batch_dense_packed_layer/correctness", "[fhe_layers][batch_dense_packed]") {
    constexpr uint32_t N = 16384;
    constexpr uint32_t batch = 9;
    constexpr uint32_t input_dim = 7;
    constexpr uint32_t output_dim = 6;
    constexpr uint32_t block_size = 4;
    constexpr int level = 5;

    auto param = CkksParameter::create_parameter(static_cast<int>(std::log2(N)), RingType::ConjugateInvariant);
    auto context = CkksContext::create_random_context(param);
    context.gen_rotation_keys();

    Array<double, 2> input({batch, input_dim});
    Array<double, 2> weight({input_dim, output_dim});
    Array<double, 1> bias({output_dim});
    for (uint32_t i = 0; i < batch; i++)
        for (uint32_t k = 0; k < input_dim; k++)
            input.set(i, k, 0.01 * static_cast<double>(1 + i * input_dim + k));
    for (uint32_t k = 0; k < input_dim; k++)
        for (uint32_t h = 0; h < output_dim; h++)
            weight.set(k, h, 0.02 * static_cast<double>(1 + k * output_dim + h));
    for (uint32_t h = 0; h < output_dim; h++)
        bias.set(h, -0.01 * static_cast<double>(h + 1));

    Feature0DEncrypted input_enc(&context, level);
    input_enc.batch_pack(input, block_size, param.default_scale());

    BatchDensePackedLayer layer(param, Duo{batch, input_dim}, Duo{input_dim, output_dim}, weight, block_size, level,
                                bias.copy());
    layer.precompute_diagonals();
    auto output_enc = layer.run(context, input_enc);
    auto actual = output_enc.batch_unpack(batch, output_dim, block_size);
    auto expected = layer.run_plaintext(input);

    double max_error = 0.0;
    for (uint32_t i = 0; i < batch; i++) {
        for (uint32_t h = 0; h < output_dim; h++)
            max_error = std::max(max_error, std::abs(actual.get(i, h) - expected.get(i, h)));
    }

    CAPTURE(max_error);
    REQUIRE(max_error < 1.0e-2);
    REQUIRE(output_enc.data.size() == 2);
    REQUIRE(input_enc.data.size() == 2);
}

TEST_CASE("batch_dense_packed_layer/multiple_groups_and_padding", "[fhe_layers][batch_dense_packed]") {
    constexpr uint32_t N = 16384;
    constexpr uint32_t batch = 2049;
    constexpr uint32_t input_dim = 1024;
    constexpr uint32_t output_dim = 64;
    constexpr uint32_t block_size = 32;
    constexpr int level = 5;

    auto param = CkksParameter::create_parameter(static_cast<int>(std::log2(N)), RingType::ConjugateInvariant);
    auto context = CkksContext::create_random_context(param);
    context.gen_rotation_keys();

    Array<double, 2> input({batch, input_dim});
    Array<double, 2> weight({input_dim, output_dim});
    Array<double, 1> bias({output_dim});
    for (uint32_t i = 0; i < batch; i++)
        for (uint32_t k = 0; k < input_dim; k++)
            input.set(i, k, 0.001 * static_cast<double>(1 + (i + 2 * k) % 17));
    for (uint32_t k = 0; k < input_dim; k++)
        for (uint32_t h = 0; h < output_dim; h++)
            weight.set(k, h, 0.002 * static_cast<double>(1 + (3 * k + h) % 13));
    for (uint32_t h = 0; h < output_dim; h++)
        bias.set(h, 0.001 * static_cast<double>(h + 1));

    Feature0DEncrypted input_enc(&context, level);
    input_enc.batch_pack(input, block_size, param.default_scale());

    BatchDensePackedLayer layer(param, Duo{batch, input_dim}, Duo{input_dim, output_dim}, weight, block_size, level,
                                bias.copy());
    layer.precompute_diagonals();
    auto output_enc = layer.run(context, input_enc);
    auto actual = output_enc.batch_unpack(batch, output_dim, block_size);
    auto expected = layer.run_plaintext(input);

    double max_error = 0.0;
    for (uint32_t i = 0; i < batch; i++) {
        for (uint32_t h = 0; h < output_dim; h++)
            max_error = std::max(max_error, std::abs(actual.get(i, h) - expected.get(i, h)));
    }

    CAPTURE(max_error);
    REQUIRE(max_error < 1.0e-2);
    // N=16384 (CI), d*d=1024, so one ciphertext has 16 chunks. B=2049 needs
    // ceil(ceil(2049/32)/16)=5 batch ciphertext groups.
    REQUIRE(input_enc.data.size() == 160);
    REQUIRE(output_enc.data.size() == 10);
}
// Append this TEST_CASE to inference/unittests/test_fhe_layers_hetero.cpp.
// Real CKKS, direct CkksContext API, no DAG or output repacking yet.
TEST_CASE("poly_maxpool3x3_deferred_half_real_ckks", "[fhe_layers][poly_maxpool3x3_real]") {
    constexpr int N = 65536;
    constexpr int L = 33;
    constexpr int H = 32;
    constexpr int W = 32;
    constexpr int stride = 2;
    constexpr int order = 7;
    constexpr double running_max = 10.0;
    constexpr double input_amplitude = 10.0;

    std::ifstream param_file("/tmp/poly_maxpool2x2_task/task/server/ckks_parameter.json");
    REQUIRE(param_file.is_open());
    nlohmann::json param_json;
    param_file >> param_json;
    const auto& cfg = param_json.at("param0");
    REQUIRE(cfg.at("param_name").get<std::string>() == "PN16QP1761");

    CkksParameter param = CkksParameter::create_custom_parameter(
        static_cast<int>(std::log2(N)), cfg.at("log_default_scale").get<int>(),
        cfg.at("q").get<std::vector<uint64_t>>(),
        cfg.at("p").get<std::vector<uint64_t>>());
    REQUIRE(param.max_level() >= L);
    CkksContext ctx = CkksContext::create_random_context(param);
    ctx.gen_rotation_keys();
    const double default_scale = param.default_scale();
    const int slots = N / 2;
    const int channels_per_ct = slots / (H * W);

    Array<double, 3> input({1, H, W});
    std::mt19937 rng(20261024);
    std::uniform_real_distribution<double> dist(-input_amplitude, input_amplitude);
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            input.set(0, r, c, dist(rng));

    Feature2DEncrypted encrypted(&ctx, L, Duo{1, 1});
    encrypted.pack_multiplexed(input, default_scale);
    REQUIRE(encrypted.data.size() == 1);
    REQUIRE(encrypted.data[0].level() == L);

    // Encrypted neighborhood taps: fused normalization and boundary replacement.
    // Valid output anchors are at (2*out_row, 2*out_col).
    std::vector<CkksCiphertext> current;
    current.reserve(9);
    for (int kr = 0; kr < 3; ++kr) {
        for (int kc = 0; kc < 3; ++kc) {
            const int dr = kr - 1;
            const int dc = kc - 1;
            const int rotation = dr * W + dc;
            std::vector<double> valid(slots, 0.0), invalid(slots, 0.0);
            bool any_valid = false, any_invalid = false;
            for (int r = 0; r < H / stride; ++r) {
                for (int c = 0; c < W / stride; ++c) {
                    const int ar = r * stride;
                    const int ac = c * stride;
                    const int pos = ar * W + ac;
                    const bool inside = ar + dr >= 0 && ar + dr < H &&
                                        ac + dc >= 0 && ac + dc < W;
                    if (inside) {
                        valid[pos] = 1.0 / running_max;
                        any_valid = true;
                    } else {
                        invalid[pos] = 1.0 / running_max;
                        any_invalid = true;
                    }
                }
            }

            auto multiply_mask = [&](const CkksCiphertext& source,
                                     const std::vector<double>& values) {
                auto pt = fhe_api::encode_ringt(ctx, values, param.q()[L]);
                const auto& mul_pt = pt;
                return ctx.mult(source, mul_pt);
            };

            CkksCiphertext accumulated;
            if (any_valid) {
                auto rotated = rotation == 0 ? encrypted.data[0].copy()
                                             : ctx.rotate(encrypted.data[0], rotation);
                accumulated = multiply_mask(rotated, valid);
            }
            if (any_invalid) {
                auto boundary = multiply_mask(encrypted.data[0], invalid);
                accumulated = any_valid ? ctx.add(accumulated, boundary)
                                        : std::move(boundary);
            }
            REQUIRE((any_valid || any_invalid));
            current.push_back(ctx.rescale(accumulated));
            REQUIRE(current.back().level() == L - 1);
        }
    }
    std::cout << "3x3 CKKS taps: " << current.size() << ", level "
              << current.front().level() << std::endl;

    const std::vector<double> coeff0 = {
        0.0, 7.30445164958251, 0.0, -34.6825871108659,
        0.0, 59.8596518298826, 0.0, -31.8755225906466};
    const std::vector<double> coeff1 = {
        0.0, 2.40085652217597, 0.0, -2.63125454261783,
        0.0, 1.54912674773593, 0.0, -0.331172956504304};

    auto make_weights = [](const std::vector<double>& coeff) {
        Array<double, 2> w({order + 1, 1});

        for (int i = 0; i <= order; ++i)
            w.set(i, 0, coeff.at(i));
        return w;
    };

    for (int round = 0; round < 4; ++round) {
        const int level_before = current.front().level();
        const double B = (round == 0 ? 1.0 : 1.05) * std::pow(2.0, round);
        std::vector<double> scaled0 = coeff0;
        for (int i = 0; i <= order; ++i)
            scaled0[i] /= std::pow(2.0 * B, i);

        Poly2D poly0(param, Duo{H, W}, order, make_weights(scaled0),
                     Duo{1, 1}, channels_per_ct, level_before);
        Poly2D poly1(param, Duo{H, W}, order, make_weights(coeff1),
                     Duo{1, 1}, channels_per_ct, level_before - 3);
        poly0.prepare_weight_stockmeyer();
        poly1.prepare_weight_stockmeyer();

        std::vector<CkksCiphertext> next;
        next.reserve((current.size() + 1) / 2);
        for (size_t i = 0; i + 1 < current.size(); i += 2) {
            auto difference = ctx.sub(current[i], current[i + 1]);
            auto sum = ctx.add(current[i], current[i + 1]);

            std::vector<CkksCiphertext> poly_input;
            poly_input.push_back(difference.copy());
            auto p0_out = poly0.run_core_stockmeyer(ctx, poly_input);
            REQUIRE(p0_out.at(0).level() == level_before - 3);
            auto p1_out = poly1.run_core_stockmeyer(ctx, p0_out);
            REQUIRE(p1_out.at(0).level() == level_before - 6);

            while (difference.level() > p1_out[0].level())
                difference = ctx.drop_level(difference);
            auto product = ctx.rescale(
                ctx.relinearize(ctx.mult(difference, p1_out[0])));
            REQUIRE(product.level() == level_before - 7);
            while (sum.level() > product.level())
                sum = ctx.drop_level(sum);
            next.push_back(ctx.add(sum, product));  // No per-round *0.5
        }
        if (current.size() % 2 != 0) {
            auto carry = current.back().copy();
            const int target_level = next.front().level();
            while (carry.level() > target_level)
                carry = ctx.drop_level(carry);
            REQUIRE(carry.level() == target_level);
            next.push_back(ctx.add(carry, carry));  // Deferred-half carry
        }
        current = std::move(next);
        REQUIRE(current.front().level() == level_before - 7);
        std::cout << "3x3 CKKS round " << round + 1
                  << ": candidates=" << current.size()
                  << ", level=" << current.front().level() << std::endl;
    }
    REQUIRE(current.size() == 1);
    REQUIRE(current[0].level() == 4);

    // Restore original input scale and divide by 2^4 in encrypted space.
    // No output repacking yet: valid results remain at (2*r, 2*c) slots.
    std::vector<double> restore(slots, 0.0);
    for (int r = 0; r < H / stride; ++r)
        for (int c = 0; c < W / stride; ++c)
            restore[(stride * r) * W + stride * c] = running_max / 16.0;
    const int restore_level = current[0].level();
    auto restore_pt = fhe_api::encode_ringt(ctx, restore, param.q()[restore_level]);
    const auto& restore_mul_pt = restore_pt;
    auto restored = ctx.rescale(
        ctx.mult(current[0], restore_mul_pt));
    REQUIRE(restored.level() == 3);

    // Genuine decrypt/decode, compare directly with exact 3x3 MaxPool.
    auto decoded_pt = ctx.decrypt(restored);
    std::vector<double> decrypted;
    ctx.decode(decoded_pt, decrypted);
    // PLAINTEXT_POLY_REFERENCE_V1: exact same input and polynomial as CKKS.
    auto eval_poly_ref = [](const std::vector<double>& coeff, double x) {
        double result = 0.0;
        for (int i = static_cast<int>(coeff.size()) - 1; i >= 0; --i)
            result = result * x + coeff[i];
        return result;
    };

    auto plaintext_poly_at = [&](int ar, int ac) {
        std::vector<double> candidates;
        candidates.reserve(9);
        const double anchor = input.get(0, ar, ac) / running_max;
        for (int kr = 0; kr < 3; ++kr) {
            for (int kc = 0; kc < 3; ++kc) {
                const int r = ar + kr - 1;
                const int c = ac + kc - 1;
                const bool inside = r >= 0 && r < H && c >= 0 && c < W;
                candidates.push_back(
                    inside ? input.get(0, r, c) / running_max : anchor);
            }
        }
        for (int round = 0; round < 4; ++round) {
            const double B = (round == 0 ? 1.0 : 1.05) * std::pow(2.0, round);
            std::vector<double> next;
            next.reserve((candidates.size() + 1) / 2);
            for (size_t i = 0; i + 1 < candidates.size(); i += 2) {
                const double a = candidates[i];
                const double b = candidates[i + 1];
                const double d = a - b;
                const double p0 = eval_poly_ref(coeff0, d / (2.0 * B));
                const double p1 = eval_poly_ref(coeff1, p0);
                next.push_back(a + b + d * p1); // Deferred-half
            }
            if (candidates.size() % 2 != 0)
                next.push_back(2.0 * candidates.back());
            candidates = std::move(next);
        }
        if (candidates.size() != 1)
            throw std::runtime_error("Plaintext reduction did not finish");
        return candidates[0] * running_max / 16.0;
    };

    double poly_vs_exact_mae = 0.0, poly_vs_exact_max = 0.0;
    double ckks_vs_poly_mae = 0.0, ckks_vs_poly_max = 0.0;
    double mae = 0.0;
    double max_error = 0.0;
    for (int r = 0; r < H / stride; ++r) {
        for (int c = 0; c < W / stride; ++c) {
            const int ar = r * stride;
            const int ac = c * stride;
            double exact = -std::numeric_limits<double>::infinity();
            for (int dr = -1; dr <= 1; ++dr)
                for (int dc = -1; dc <= 1; ++dc)
                    if (ar + dr >= 0 && ar + dr < H &&
                        ac + dc >= 0 && ac + dc < W)
                        exact = std::max(exact, input.get(0, ar + dr, ac + dc));

            const double actual = decrypted[ar * W + ac];
            REQUIRE(std::isfinite(actual));
            const double poly = plaintext_poly_at(ar, ac);
            REQUIRE(std::isfinite(poly));
            const double poly_err = std::abs(poly - exact);
            const double ckks_err = std::abs(actual - poly);
            poly_vs_exact_mae += poly_err;
            poly_vs_exact_max = std::max(poly_vs_exact_max, poly_err);
            ckks_vs_poly_mae += ckks_err;
            ckks_vs_poly_max = std::max(ckks_vs_poly_max, ckks_err);

            const double err = std::abs(actual - exact);
            mae += err;
            max_error = std::max(max_error, err);
        }
    }
    const double output_count = (H / stride) * (W / stride);
    poly_vs_exact_mae /= output_count;
    ckks_vs_poly_mae /= output_count;
    mae /= output_count;
    std::cout << "ERROR BREAKDOWN (same input, original units):\n"
              << "  Poly vs Exact : MAE=" << poly_vs_exact_mae
              << ", max_error=" << poly_vs_exact_max << "\n"
              << "  CKKS vs Poly  : MAE=" << ckks_vs_poly_mae
              << ", max_error=" << ckks_vs_poly_max << "\n"
              << "  CKKS vs Exact : MAE=" << mae
              << ", max_error=" << max_error << std::endl;
    std::cout << "REAL CKKS 3x3 MaxPool: output_level=" << restored.level()
              << ", MAE=" << mae << ", max_error=" << max_error
              << " (original input units)" << std::endl;
    REQUIRE(std::isfinite(mae));
    REQUIRE(std::isfinite(max_error));
}