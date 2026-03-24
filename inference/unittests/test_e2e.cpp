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

#include <chrono>
#include <filesystem>
#include <fstream>
#include <vector>

#include "data_structs/feature.h"
#include "fhe_layers/conv2d_packed_layer.h"
#include "fhe_layers/multiplexed_conv2d_pack_layer.h"
#include "ut_util.h"
#include <cxx_sdk_v2/cxx_fhe_task.h>
#include <lattisense/lib/nlohmann/json.hpp>

using namespace cxx_sdk_v2;
using namespace std;
namespace fs = std::filesystem;

fs::path e2e_base_path = "../hetero_e2e";

struct TestManifest {
    string test_name;
    int N;
    string style;
    vector<string> input_features;
    vector<string> output_features;
    nlohmann::json features;
    nlohmann::json layers;
    double max_error_ratio;
    double rmse_ratio;
};

static TestManifest read_manifest(const fs::path& manifest_path) {
    ifstream f(manifest_path);
    auto j = nlohmann::json::parse(f);
    TestManifest m;
    m.test_name = j["test_name"];
    m.N = j["N"];
    m.style = j["style"];
    for (auto& v : j["input_features"])
        m.input_features.push_back(v.get<string>());
    for (auto& v : j["output_features"])
        m.output_features.push_back(v.get<string>());
    m.features = j["features"];
    m.layers = j["layers"];
    m.max_error_ratio = j["error_threshold"]["max_error_ratio"];
    m.rmse_ratio = j["error_threshold"]["rmse_ratio"];
    return m;
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

static vector<fs::path> discover_tests(const fs::path& base) {
    vector<fs::path> result;
    if (!fs::exists(base))
        return result;
    for (auto& entry : fs::directory_iterator(base)) {
        if (!entry.is_directory())
            continue;
        auto manifest = entry.path() / "task" / "server" / "test_manifest.json";
        if (fs::exists(manifest)) {
            result.push_back(entry.path());
        }
    }
    sort(result.begin(), result.end());
    return result;
}

TEST_CASE("e2e_conv2d", "[e2e]") {
    auto test_dirs = discover_tests(e2e_base_path);
    REQUIRE_FALSE(test_dirs.empty());

    for (const auto& test_dir : test_dirs) {
        fs::path server_path = test_dir / "task" / "server";
        auto manifest = read_manifest(server_path / "test_manifest.json");

        // Only handle conv2d tests in this test case
        bool has_conv = false;
        for (auto& [lid, linfo] : manifest.layers.items()) {
            string ltype = linfo["type"];
            if (ltype.find("conv") != string::npos) {
                has_conv = true;
                break;
            }
        }
        if (!has_conv)
            continue;

        SECTION(manifest.test_name) {
            int N = manifest.N;
            int n_slot = N / 2;
            auto param = CkksParameter::create_parameter(N);
            auto context = CkksContext::create_random_context(param);
            context.gen_rotation_keys();

            // Read feature info from manifest
            string input_id = manifest.input_features[0];
            string output_id = manifest.output_features[0];
            auto& f_in = manifest.features[input_id];
            auto& f_out = manifest.features[output_id];

            int n_in_channel = f_in["channel"];
            int n_out_channel = f_out["channel"];
            int init_level = f_in["level"];

            // Find the conv layer info
            nlohmann::json conv_layer;
            for (auto& [lid, linfo] : manifest.layers.items()) {
                string ltype = linfo["type"];
                if (ltype.find("conv") != string::npos) {
                    conv_layer = linfo;
                    break;
                }
            }

            Duo input_shape = {(uint32_t)f_in["shape"][0], (uint32_t)f_in["shape"][1]};
            Duo kernel_shape = {(uint32_t)conv_layer["kernel_shape"][0], (uint32_t)conv_layer["kernel_shape"][1]};
            Duo stride = {(uint32_t)conv_layer["stride"][0], (uint32_t)conv_layer["stride"][1]};
            Duo skip = {1, 1};
            if (f_in.contains("skip") && f_in["skip"].is_array()) {
                skip = {(uint32_t)f_in["skip"][0], (uint32_t)f_in["skip"][1]};
            }
            uint32_t n_channel_per_ct = f_in["pack_num"];
            int groups = conv_layer.value("groups", 1);

            // Generate random data
            Array<double, 4> weight = gen_random_array<4>(
                {(uint32_t)n_out_channel, (uint32_t)(groups == 1 ? n_in_channel : 1), kernel_shape[0], kernel_shape[1]},
                0.1);
            Array<double, 1> bias = gen_random_array<1>({(uint32_t)n_out_channel}, 0.1);
            Array<double, 3> input_array =
                gen_random_array<3>({(uint32_t)n_in_channel, input_shape[0], input_shape[1]}, 1.0);

            if (manifest.style == "multiplexed") {
                // Multiplexed conv path
                ParMultiplexedConv2DPackedLayer conv_layer_obj(param, input_shape, weight, bias, stride, skip,
                                                               n_channel_per_ct, init_level, 1.0);
                conv_layer_obj.prepare_weight_for_post_skip_rotation();

                Feature2DEncrypted input_feature(&context, init_level, skip);
                input_feature.pack_multiplexed(input_array, false, param.get_default_scale());

                int output_level = f_out["level"];
                uint32_t out_pack = f_out["pack_num"];
                Feature2DEncrypted output_feature(&context, output_level);
                output_feature.shape[0] = input_shape[0] / stride[0];
                output_feature.shape[1] = input_shape[1] / stride[1];
                output_feature.skip[0] = skip[0] * stride[0];
                output_feature.skip[1] = skip[1] * stride[1];
                output_feature.n_channel = n_out_channel;
                output_feature.n_channel_per_ct = out_pack;
                for (int i = 0; i < div_ceil(n_out_channel, out_pack); i++) {
                    output_feature.data.push_back(context.new_ciphertext(output_level, param.get_default_scale()));
                }

                auto arg_names = read_arg_names(server_path);
                vector<CxxVectorArgument> cxx_args;
                for (const auto& name : arg_names) {
                    if (name.rfind("input", 0) == 0)
                        cxx_args.push_back({name, &input_feature.data});
                    else if (name.rfind("convm_", 0) == 0)
                        cxx_args.push_back({name, &conv_layer_obj.mask_pt});
                    else if (name.rfind("convw_", 0) == 0)
                        cxx_args.push_back({name, &conv_layer_obj.weight_pt});
                    else if (name.rfind("convb_", 0) == 0)
                        cxx_args.push_back({name, &conv_layer_obj.bias_pt});
                    else if (name.rfind("output", 0) == 0)
                        cxx_args.push_back({name, &output_feature.data});
                }

                FheTaskCpu fhe_task(server_path.string());
                fhe_task.run(&context, cxx_args);

                auto y_mg = output_feature.unpack_multiplexed();
                auto y_expected = conv_layer_obj.run_plaintext(input_array);

                auto cmp = compare(y_expected, y_mg);
                REQUIRE(cmp.max_error < manifest.max_error_ratio * cmp.max_abs);
                REQUIRE(cmp.rmse < manifest.rmse_ratio * cmp.rms);
            } else {
                // Ordinary conv path
                Conv2DPackedLayer conv_layer_obj(param, input_shape, weight, bias, stride, skip, n_channel_per_ct,
                                                 init_level);
                conv_layer_obj.prepare_weight();

                Feature2DEncrypted input_feature(&context, init_level);
                input_feature.pack_multiple_channel(input_array, false, param.get_default_scale());

                int output_level = f_out["level"];
                Feature2DEncrypted output_feature(&context, output_level);
                output_feature.shape[0] = input_shape[0] / stride[0];
                output_feature.shape[1] = input_shape[1] / stride[1];
                output_feature.skip[0] = skip[0] * stride[0];
                output_feature.skip[1] = skip[1] * stride[1];
                output_feature.n_channel = n_out_channel;
                output_feature.n_channel_per_ct = n_channel_per_ct;
                for (int i = 0; i < div_ceil(n_out_channel, n_channel_per_ct); i++) {
                    output_feature.data.push_back(context.new_ciphertext(output_level, param.get_default_scale()));
                }

                auto arg_names = read_arg_names(server_path);
                vector<CxxVectorArgument> cxx_args;
                for (const auto& name : arg_names) {
                    if (name.rfind("input", 0) == 0)
                        cxx_args.push_back({name, &input_feature.data});
                    else if (name.rfind("convw_", 0) == 0)
                        cxx_args.push_back({name, &conv_layer_obj.weight_pt_});
                    else if (name.rfind("convb_", 0) == 0)
                        cxx_args.push_back({name, &conv_layer_obj.bias_pt_});
                    else if (name.rfind("output", 0) == 0)
                        cxx_args.push_back({name, &output_feature.data});
                }

                FheTaskCpu fhe_task(server_path.string());
                fhe_task.run(&context, cxx_args);

                auto output_mg = output_feature.unpack();
                auto plain_output = conv_layer_obj.run_plaintext(input_array);

                auto cmp = compare(plain_output, output_mg);
                REQUIRE(cmp.max_error < manifest.max_error_ratio * cmp.max_abs);
                REQUIRE(cmp.rmse < manifest.rmse_ratio * cmp.rms);
            }
        }
    }
}
