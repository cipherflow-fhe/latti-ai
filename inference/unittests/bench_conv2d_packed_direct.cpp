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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "data_structs/feature.h"
#include "fhe_layers/conv2d_packed_layer.h"

using namespace std;
using namespace lattisense;

namespace {

constexpr uint32_t kInitLevel = 2;

struct BenchConfig {
    uint32_t in_channels = 4;
    uint32_t out_channels = 4;
    uint32_t height = 8;
    uint32_t width = 8;
    uint32_t kernel = 3;
};

double input_value(uint32_t idx) {
    return (static_cast<double>(idx % 17) - 8.0) / 16.0;
}

double weight_value(const BenchConfig& cfg, uint32_t oc, uint32_t ic, uint32_t kh, uint32_t kw) {
    const uint32_t idx = (((oc * cfg.in_channels + ic) * cfg.kernel + kh) * cfg.kernel + kw);
    return (static_cast<double>(idx % 29) - 14.0) / 128.0;
}

double bias_value(uint32_t oc) {
    return (static_cast<double>(oc) - 1.5) / 64.0;
}

Array<double, 3> make_input(const BenchConfig& cfg) {
    Array<double, 3> input({cfg.in_channels, cfg.height, cfg.width});
    uint32_t idx = 0;
    for (uint32_t c = 0; c < cfg.in_channels; ++c) {
        for (uint32_t h = 0; h < cfg.height; ++h) {
            for (uint32_t w = 0; w < cfg.width; ++w) {
                input.set(c, h, w, input_value(idx++));
            }
        }
    }
    return input;
}

Array<double, 4> make_weight(const BenchConfig& cfg) {
    Array<double, 4> weight({cfg.out_channels, cfg.in_channels, cfg.kernel, cfg.kernel});
    for (uint32_t oc = 0; oc < cfg.out_channels; ++oc) {
        for (uint32_t ic = 0; ic < cfg.in_channels; ++ic) {
            for (uint32_t kh = 0; kh < cfg.kernel; ++kh) {
                for (uint32_t kw = 0; kw < cfg.kernel; ++kw) {
                    weight.set(oc, ic, kh, kw, weight_value(cfg, oc, ic, kh, kw));
                }
            }
        }
    }
    return weight;
}

Array<double, 1> make_bias(const BenchConfig& cfg) {
    Array<double, 1> bias({cfg.out_channels});
    for (uint32_t oc = 0; oc < cfg.out_channels; ++oc) {
        bias.set(oc, bias_value(oc));
    }
    return bias;
}

Feature2DEncrypted pack_manual(CkksContext& context, const Array<double, 3>& input, const BenchConfig& cfg, uint32_t level) {
    const uint32_t n_slot = context.get_parameter().get_n() / 2;
    vector<double> slots(n_slot, 0.0);

    for (uint32_t c = 0; c < cfg.in_channels; ++c) {
        for (uint32_t h = 0; h < cfg.height; ++h) {
            for (uint32_t w = 0; w < cfg.width; ++w) {
                slots[c * cfg.height * cfg.width + h * cfg.width + w] = input.get(c, h, w);
            }
        }
    }

    Feature2DEncrypted feature(&context, level);
    feature.packing_type = PackType::MultipleChannelPacking;
    feature.shape = {cfg.height, cfg.width};
    feature.skip = {1, 1};
    feature.n_channel = cfg.in_channels;
    feature.n_channel_per_ct = cfg.in_channels;
    feature.data.push_back(context.encrypt_symmetric(context.encode(slots, level, context.get_parameter().get_default_scale())));
    return feature;
}

Feature2DEncrypted pack_default(CkksContext& context, const Array<double, 3>& input, uint32_t level) {
    Feature2DEncrypted feature(&context, level);
    feature.pack_multiple_channel(input, false, context.get_parameter().get_default_scale());
    return feature;
}

double median(vector<double> values) {
    sort(values.begin(), values.end());
    return values[values.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    uint32_t N = 16384;
    int iterations = 3;
    string pack_mode = "manual4";
    string preset = "4x8";

    if (argc > 1) {
        N = static_cast<uint32_t>(stoul(argv[1]));
    }
    if (argc > 2) {
        iterations = stoi(argv[2]);
    }
    if (argc > 3) {
        pack_mode = argv[3];
    }
    if (argc > 4) {
        preset = argv[4];
    }
    if (iterations <= 0) {
        cerr << "iterations must be positive\n";
        return 1;
    }
    if (pack_mode != "manual4" && pack_mode != "manual" && pack_mode != "default") {
        cerr << "pack_mode must be manual4, manual, or default\n";
        return 1;
    }
    BenchConfig cfg;
    if (preset == "cifar3x32") {
        cfg = BenchConfig{3, 3, 32, 32, 3};
    } else if (preset != "4x8") {
        cerr << "preset must be 4x8 or cifar3x32\n";
        return 1;
    }

    const Duo input_shape = {cfg.height, cfg.width};
    const Duo stride = {1, 1};
    const Duo skip = {1, 1};
    auto input = make_input(cfg);

    cout << fixed << setprecision(6);
    cout << "N=" << N << "\n";
    cout << "shape=[1," << cfg.in_channels << "," << cfg.height << "," << cfg.width << "], weight=["
         << cfg.out_channels << "," << cfg.in_channels << "," << cfg.kernel << "," << cfg.kernel
         << "], stride=1, padding=same\n";
    cout << "pack_mode=" << pack_mode << "\n";
    cout << "preset=" << preset << "\n";
    cout << "th_nums=" << th_nums << "\n";

    auto t_context0 = chrono::steady_clock::now();
    CkksParameter param = CkksParameter::create_parameter(N);
    CkksContext context = CkksContext::create_random_context(param);
    context.gen_rotation_keys();
    auto t_context1 = chrono::steady_clock::now();

    auto t_pack0 = chrono::steady_clock::now();
    Feature2DEncrypted input_feature =
        (pack_mode == "default") ? pack_default(context, input, kInitLevel) : pack_manual(context, input, cfg, kInitLevel);
    auto t_pack1 = chrono::steady_clock::now();

    const uint32_t n_channel_per_ct =
        (pack_mode == "default") ? (param.get_n() / 2 / (cfg.height * cfg.width)) : cfg.in_channels;

    Conv2DPackedLayer conv_layer(param, input_shape, make_weight(cfg), make_bias(cfg), stride, skip, n_channel_per_ct,
                                 kInitLevel);

    auto t_prepare0 = chrono::steady_clock::now();
    conv_layer.prepare_weight();
    auto t_prepare1 = chrono::steady_clock::now();

    vector<double> run_ms;
    run_ms.reserve(iterations);
    size_t output_ct_count = 0;
    uint32_t output_level = 0;

    for (int i = 0; i < iterations; ++i) {
        auto t_run0 = chrono::steady_clock::now();
        Feature2DEncrypted output = conv_layer.run(context, input_feature);
        auto t_run1 = chrono::steady_clock::now();

        output_ct_count = output.data.size();
        output_level = output.level;
        run_ms.push_back(chrono::duration<double, milli>(t_run1 - t_run0).count());
        cout << "run_ms[" << i << "]=" << run_ms.back() << "\n";
    }

    const double avg_ms = accumulate(run_ms.begin(), run_ms.end(), 0.0) / run_ms.size();
    cout << "context_keygen_ms=" << chrono::duration<double, milli>(t_context1 - t_context0).count() << "\n";
    cout << "pack_encrypt_ms=" << chrono::duration<double, milli>(t_pack1 - t_pack0).count() << "\n";
    cout << "prepare_weight_ms=" << chrono::duration<double, milli>(t_prepare1 - t_prepare0).count() << "\n";
    cout << "run_avg_ms=" << avg_ms << "\n";
    cout << "run_median_ms=" << median(run_ms) << "\n";
    cout << "input_ct_count=" << input_feature.data.size() << "\n";
    cout << "output_ct_count=" << output_ct_count << "\n";
    cout << "n_channel_per_ct=" << n_channel_per_ct << "\n";
    cout << "output_level=" << output_level << "\n";
    return 0;
}
