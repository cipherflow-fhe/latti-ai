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

#include "data_structs/feature.h"
#include "fhe_layers/conv2d_packed_layer.h"
#include "fhe_layers/poly_relu2d.h"
#include "fhe_layers/poly_relu_base.h"
#include "fhe_layers/multiplexed_conv2d_pack_layer.h"
#include "fhe_layers/multiplexed_conv2d_pack_layer_depthwise.h"
#include "fhe_layers/activation_layer.h"
#include "fhe_layers/conv2d_depthwise.h"
#include "fhe_layers/dense_packed_layer.h"
#include "fhe_layers/reshape_layer.h"
#include "fhe_layers/block_col_major_ccmm.h"
#include "fhe_layers/block_col_major_cpmm.h"
#include "fhe_layers/block_col_major_transpose.h"
#include "fhe_layers/par_block_col_major_transpose.h"
#include "fhe_layers/par_block_col_major_ccmm.h"
#include "fhe_layers/par_block_col_major_cpmm.h"
#include "fhe_layers/conv1d_packed_layer.h"
#include "fhe_layers/multiplexed_conv1d_pack_layer.h"
#include "fhe_layers/inverse_multiplexed_conv2d_layer.h"
#include "fhe_layers/inverse_multiplexed_conv2d_layer_depthwise.h"
#include "fhe_layers/add_layer.h"
#include "fhe_layers/avgpool2d_layer.h"
#include "fhe_layers/concat_layer.h"
#include "fhe_layers/mult_scaler.h"
#include "fhe_layers/upsample_layer.h"
#include "fhe_layers/upsample_nearest_layer.h"
#include "ut_util.h"
#include <cxx_sdk_v2/cxx_fhe_task.h>
#include <lattisense/lib/nlohmann/json.hpp>

using namespace std;
using namespace cxx_sdk_v2;
namespace fs = std::filesystem;

fs::path base_path = "../hetero";

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
    const int N = 16384;
    const int n_slot = N / 2;
    CkksParameter param;
    CkksContext context;

private:
    SharedHeteroResources()
        : param(CkksParameter::create_parameter(N)), context(CkksContext::create_random_context(param)) {
        context.gen_rotation_keys();
    }
    SharedHeteroResources(const SharedHeteroResources&) = delete;
    SharedHeteroResources& operator=(const SharedHeteroResources&) = delete;
};

template <typename T> class HeteroFixture {
public:
    HeteroFixture()
        : N{SharedHeteroResources::get().N}, n_slot{SharedHeteroResources::get().n_slot},
          param{SharedHeteroResources::get().param}, context{SharedHeteroResources::get().context}, level(3),
          min_level{0}, max_level{param.get_max_level()}, default_scale{param.get_default_scale()} {}

    ~HeteroFixture() {
        MetricsCollector::save_to_csv("hetero_performance_results.csv");
    }

    uint64_t run(const fs::path& project_path, const vector<CxxVectorArgument>& cxx_args) {
        auto start_time = std::chrono::high_resolution_clock::now();
        uint64_t result;
        std::string processor_type;

        if constexpr (is_same_v<T, ProcessorCpu>) {
            processor_type = "CPU";
            FheTaskCpu fhe_task(project_path.string());
            result = fhe_task.run(&context, cxx_args);
#ifdef INFERENCE_SDK_ENABLE_GPU
        } else if constexpr (is_same_v<T, ProcessorGpu>) {
            processor_type = "GPU";
            FheTaskGpu fhe_task(project_path.string());
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

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "sq", "", HeteroProcessors) {
    int init_level = 2;
    vector<uint32_t> input_shapes = {16, 32, 64};

    for (uint32_t s : input_shapes) {
        Duo input_shape = {s, s};
        SECTION("input_shape=" + str(input_shape)) {
            Array<double, 3> input_array = gen_random_array<3>({1, input_shape[0], input_shape[1]}, 1.0);

            Feature2DEncrypted input_feature(&this->context, init_level);
            input_feature.pack_multiple_channel(input_array, false, this->param.get_default_scale());

            Feature2DEncrypted output_feature(&this->context, init_level);

            for (int i = 0; i < 1; i++) {
                output_feature.data.push_back(
                    this->context.new_ciphertext(init_level - 1, this->param.get_default_scale()));
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
            output_feature.n_channel_per_ct = this->n_slot / (s * s);
            output_feature.shape = {s, s};
            auto output_mg = output_feature.unpack_multiple_channel();

            SquareLayer square_layer(this->param);
            auto plain_output = square_layer.run_plaintext(input_array);

            print_double_message(output_mg.to_array_1d().data(), "output_mg", 10);
            print_double_message(plain_output.to_array_1d().data(), "plain_output", 10);

            auto compare_result = compare(plain_output, output_mg);
            REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
            REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "conv2d_packed", "", HeteroProcessors) {
    Duo skip = {1, 1};
    int init_level = 2;

    auto run_conv2d_packed_test = [&](uint32_t n_in_channel, uint32_t n_out_channel, Duo stride, Duo input_shape,
                                      Duo kernel_shape) {
        uint32_t n_channel_per_ct = div_ceil(this->n_slot, (input_shape[0] * input_shape[1]));

        Array<double, 4> conv0_weight =
            gen_random_array<4>({n_out_channel, n_in_channel, kernel_shape[0], kernel_shape[1]}, 0.1);
        Array<double, 1> conv0_bias = gen_random_array<1>({n_out_channel}, 0.1);
        Array<double, 3> input_array = gen_random_array<3>({n_in_channel, input_shape[0], input_shape[1]}, 1.0);

        Feature2DEncrypted input_feature(&this->context, init_level);
        input_feature.pack_multiple_channel(input_array, false, this->param.get_default_scale());

        Conv2DPackedLayer conv0_layer(this->context.get_parameter(), input_shape, conv0_weight, conv0_bias, stride,
                                      skip, n_channel_per_ct, init_level);
        conv0_layer.prepare_weight();

        Feature2DEncrypted output_feature(&this->context, init_level - 1);
        output_feature.shape[0] = input_shape[0] / stride[0];
        output_feature.shape[1] = input_shape[1] / stride[1];
        output_feature.skip[0] = skip[0] * stride[0];
        output_feature.skip[1] = skip[1] * stride[1];
        output_feature.n_channel = n_out_channel;
        output_feature.n_channel_per_ct = n_channel_per_ct;
        for (int i = 0; i < div_ceil(n_out_channel, n_channel_per_ct); i++) {
            output_feature.data.push_back(
                this->context.new_ciphertext(init_level - 1, this->param.get_default_scale()));
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
            vector<uint32_t> input_shapes = {4, 8, 16, 32, 64};
            vector<uint32_t> kernel_shapes = {1, 3, 5};

            for (uint32_t s : input_shapes) {
                Duo input_shape = {s, s};
                SECTION("input_shape=" + str(input_shape)) {
                    for (uint32_t k : kernel_shapes) {
                        Duo kernel_shape = {k, k};
                        SECTION("kernel_shape=" + str(kernel_shape)) {
                            run_conv2d_packed_test(n_in_channel, n_out_channel, stride, input_shape, kernel_shape);
                        }
                    }
                }
            }
        }

        SECTION("multi_channel") {
            Duo input_shape = {32, 32};
            Duo kernel_shape = {3, 3};
            vector<uint32_t> nc_ins = {1, 3, 4, 16, 17};
            vector<uint32_t> nc_outs = {1, 3, 4, 32, 33};

            for (uint32_t n_in_channel : nc_ins) {
                SECTION("n_in_channel=" + to_string(n_in_channel)) {
                    for (uint32_t n_out_channel : nc_outs) {
                        SECTION("n_out_channel=" + to_string(n_out_channel)) {
                            run_conv2d_packed_test(n_in_channel, n_out_channel, stride, input_shape, kernel_shape);
                        }
                    }
                }
            }
        }
    }

    SECTION("stride=(2,2)") {
        Duo stride = {2, 2};

        SECTION("single_channel") {
            uint32_t n_in_channel = 1;
            uint32_t n_out_channel = 1;
            vector<uint32_t> input_shapes = {32, 64};
            vector<uint32_t> kernel_shapes = {1, 3, 5};

            for (uint32_t s : input_shapes) {
                Duo input_shape = {s, s};
                SECTION("input_shape=" + str(input_shape)) {
                    for (uint32_t k : kernel_shapes) {
                        Duo kernel_shape = {k, k};
                        SECTION("kernel_shape=" + str(kernel_shape)) {
                            run_conv2d_packed_test(n_in_channel, n_out_channel, stride, input_shape, kernel_shape);
                        }
                    }
                }
            }
        }

        SECTION("multi_channel") {
            Duo input_shape = {32, 32};
            Duo kernel_shape = {3, 3};
            vector<uint32_t> nc_ins = {1, 3, 4, 16, 17};
            vector<uint32_t> nc_outs = {1, 3, 4, 32, 33};

            for (uint32_t n_in_channel : nc_ins) {
                SECTION("n_in_channel=" + to_string(n_in_channel)) {
                    for (uint32_t n_out_channel : nc_outs) {
                        SECTION("n_out_channel=" + to_string(n_out_channel)) {
                            run_conv2d_packed_test(n_in_channel, n_out_channel, stride, input_shape, kernel_shape);
                        }
                    }
                }
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "conv2d_depthwise", "", HeteroProcessors) {
    Duo skip = {1, 1};
    int init_level = 5;

    vector<uint32_t> channels = {4, 8, 32};
    vector<uint32_t> input_sizes = {16, 32};
    vector<uint32_t> kernel_sizes = {1, 3, 5};
    vector<Duo> strides = {{1, 1}, {2, 2}};

    for (const Duo& stride : strides) {
        SECTION("stride=" + str(stride)) {
            for (uint32_t n_channel : channels) {
                SECTION("ch=" + to_string(n_channel)) {
                    for (uint32_t is : input_sizes) {
                        Duo input_shape = {is, is};
                        uint32_t n_channel_per_ct = div_ceil(this->n_slot, (input_shape[0] * input_shape[1]));
                        SECTION("input=" + str(input_shape)) {
                            for (uint32_t ks : kernel_sizes) {
                                Duo kernel_shape = {ks, ks};
                                SECTION("kernel=" + str(kernel_shape)) {
                                    Array<double, 4> conv0_weight = gen_random_array<4>(
                                        {n_channel, n_channel, kernel_shape[0], kernel_shape[1]}, 0.1);
                                    Array<double, 1> conv0_bias = gen_random_array<1>({n_channel}, 0);
                                    Array<double, 3> input =
                                        gen_random_array<3>({n_channel, input_shape[0], input_shape[1]}, 1);

                                    Conv2DPackedDepthwiseLayer conv(this->context.get_parameter(), input_shape,
                                                                    conv0_weight, conv0_bias, stride, skip,
                                                                    n_channel_per_ct, init_level);
                                    conv.prepare_weight();

                                    Feature2DEncrypted f2d(&this->context, init_level);
                                    f2d.pack_multiple_channel(input, false, this->param.get_default_scale());

                                    Feature2DEncrypted output_feature(&this->context, init_level - 1);
                                    output_feature.shape[0] = input_shape[0] / stride[0];
                                    output_feature.shape[1] = input_shape[1] / stride[1];
                                    output_feature.skip[0] = skip[0] * stride[0];
                                    output_feature.skip[1] = skip[1] * stride[1];
                                    output_feature.n_channel = n_channel;
                                    output_feature.n_channel_per_ct = n_channel_per_ct;
                                    for (int i = 0; i < div_ceil(n_channel, n_channel_per_ct); i++) {
                                        output_feature.data.push_back(this->context.new_ciphertext(
                                            init_level - 1, this->param.get_default_scale()));
                                    }

                                    fs::path project_path =
                                        base_path / "CKKS_dw_conv2d" /
                                        ("stride_" + to_string(stride[0]) + "_" + to_string(stride[1])) /
                                        ("kernel_shape_" + to_string(ks) + "_" + to_string(ks)) /
                                        ("cin_" + to_string(n_channel) + "_cout_" + to_string(n_channel)) /
                                        ("input_shape_" + to_string(is) + "_" + to_string(is)) /
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
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "mux_conv2d_packed", "", HeteroProcessors) {
    Duo skip = {1, 1};
    int init_level = 5;

    auto run_mux_conv2d_test = [&](uint32_t n_in_channel, uint32_t n_out_channel, Duo stride, Duo input_shape,
                                   Duo kernel_shape) {
        uint32_t n_channel_per_ct = div_ceil(this->n_slot, (input_shape[0] * input_shape[1]));
        int output_level = (stride[0] == 1) ? init_level - 1 : init_level - 2;

        Array<double, 4> conv0_weight =
            gen_random_array<4>({n_out_channel, n_in_channel, kernel_shape[0], kernel_shape[1]}, 0.1);
        Array<double, 1> conv0_bias = gen_random_array<1>({n_out_channel}, 0.1);
        Array<double, 3> input_array = gen_random_array<3>({n_in_channel, input_shape[0], input_shape[1]}, 1.0);

        ParMultiplexedConv2DPackedLayer conv_layer(this->context.get_parameter(), input_shape, conv0_weight, conv0_bias,
                                                   stride, skip, n_channel_per_ct, init_level, 1.0);
        conv_layer.prepare_weight_for_post_skip_rotation();

        Feature2DEncrypted input_feature(&this->context, init_level, skip);
        input_feature.pack_multiplexed(input_array, false, this->context.get_parameter().get_default_scale());

        Feature2DEncrypted output_feature(&this->context, output_level);
        output_feature.shape[0] = input_shape[0] / stride[0];
        output_feature.shape[1] = input_shape[1] / stride[1];
        output_feature.skip[0] = skip[0] * stride[0];
        output_feature.skip[1] = skip[1] * stride[1];
        output_feature.n_channel = n_out_channel;
        output_feature.n_channel_per_ct = (n_channel_per_ct * stride[0] * stride[1]);
        for (int i = 0; i < div_ceil(n_out_channel, (n_channel_per_ct * stride[0] * stride[1])); i++) {
            output_feature.data.push_back(this->context.new_ciphertext(output_level, this->param.get_default_scale()));
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

        for (const Duo& stride : strides) {
            SECTION("stride=" + str(stride)) {
                for (uint32_t n_in_channel : nc_ins) {
                    SECTION("n_in_channel=" + to_string(n_in_channel)) {
                        for (uint32_t n_out_channel : nc_outs) {
                            if (n_in_channel != n_out_channel)
                                continue;
                            SECTION("n_out_channel=" + to_string(n_out_channel)) {
                                Duo input_shape = {32, 32};
                                Duo kernel_shape = {3, 3};
                                run_mux_conv2d_test(n_in_channel, n_out_channel, stride, input_shape, kernel_shape);
                            }
                        }
                    }
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
        vector<uint32_t> input_shapes = {4, 8, 16, 32, 64};

        for (uint32_t s : input_shapes) {
            Duo input_shape = {s, s};
            SECTION("input_shape=" + str(input_shape)) {
                run_mux_conv2d_test(n_in_channel, n_out_channel, stride, input_shape, kernel_shape);
            }
        }
    }

    SECTION("varied_kernel_shape") {
        uint32_t n_in_channel = 32;
        uint32_t n_out_channel = 32;
        Duo input_shape = {32, 32};
        Duo stride = {1, 1};
        vector<uint32_t> kernel_shapes = {1, 3, 5};

        for (uint32_t k : kernel_shapes) {
            Duo kernel_shape = {k, k};
            SECTION("kernel_shape=" + str(kernel_shape)) {
                run_mux_conv2d_test(n_in_channel, n_out_channel, stride, input_shape, kernel_shape);
            }
        }
    }

    SECTION("varied_channels") {
        Duo input_shape = {32, 32};
        Duo kernel_shape = {3, 3};
        Duo stride = {1, 1};
        vector<uint32_t> nc_ins = {1, 3, 4, 16, 17};
        vector<uint32_t> nc_outs = {1, 3, 4, 32, 33};

        for (uint32_t n_in_channel : nc_ins) {
            SECTION("n_in_channel=" + to_string(n_in_channel)) {
                for (uint32_t n_out_channel : nc_outs) {
                    SECTION("n_out_channel=" + to_string(n_out_channel)) {
                        run_mux_conv2d_test(n_in_channel, n_out_channel, stride, input_shape, kernel_shape);
                    }
                }
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "mux_dw_s2_64x64_k3", "", HeteroProcessors) {
    Duo input_shape = {64, 64};
    Duo kernel_shape = {3, 3};
    Duo stride = {2, 2};
    Duo skip = {1, 1};
    uint32_t n_channel_per_ct = div_ceil(this->n_slot, (input_shape[0] * input_shape[1]));
    int init_level = 5;

    vector<uint32_t> nc_ins = {4, 8, 32};
    vector<uint32_t> nc_outs = {4, 8, 32};

    for (uint32_t n_in_channel : nc_ins) {
        SECTION("n_in_channel=" + to_string(n_in_channel)) {
            for (uint32_t n_out_channel : nc_outs) {
                if (n_in_channel != n_out_channel)
                    continue;
                SECTION("n_out_channel=" + to_string(n_out_channel)) {
                    Array<double, 4> conv0_weight =
                        gen_random_array<4>({n_out_channel, n_in_channel, kernel_shape[0], kernel_shape[1]}, 0.1);
                    Array<double, 1> conv0_bias = gen_random_array<1>({n_out_channel}, 0.1);
                    Array<double, 3> input_array =
                        gen_random_array<3>({n_in_channel, input_shape[0], input_shape[1]}, 1.0);

                    ParMultiplexedConv2DPackedLayerDepthwise dw_conv_layer(this->context.get_parameter(), input_shape,
                                                                           conv0_weight, conv0_bias, stride, skip,
                                                                           n_channel_per_ct, init_level, 1.0);
                    dw_conv_layer.prepare_weight();

                    Feature2DEncrypted input_feature(&this->context, init_level, skip);
                    input_feature.pack_multiplexed(input_array, false,
                                                   this->context.get_parameter().get_default_scale());

                    Feature2DEncrypted output_feature(&this->context, init_level - 2);
                    output_feature.shape[0] = input_shape[0] / stride[0];
                    output_feature.shape[1] = input_shape[1] / stride[1];
                    output_feature.skip[0] = skip[0] * stride[0];
                    output_feature.skip[1] = skip[1] * stride[1];
                    output_feature.n_channel = n_out_channel;
                    output_feature.n_channel_per_ct = (n_channel_per_ct * stride[0] * stride[1]);
                    for (int i = 0; i < div_ceil(n_out_channel, (n_channel_per_ct * stride[0] * stride[1])); i++) {
                        output_feature.data.push_back(
                            this->context.new_ciphertext(init_level - 2, this->param.get_default_scale()));
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
                    Duo input_shape = {block_shape[0] * mult, block_shape[1] * mult};
                    Duo stride_next = {input_shape[0] / (block_shape[0] * stride[0]),
                                       input_shape[1] / (block_shape[1] * stride[1])};

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

                                InverseMultiplexedConv2DLayer conv_layer(
                                    this->context.get_parameter(), input_shape, conv0_weight, conv0_bias, padding,
                                    stride, stride_next, skip, block_shape, init_level, 1.0);
                                conv_layer.prepare_weight();

                                Feature2DEncrypted input_feature(&this->context, init_level, skip);
                                Duo total_stride = {stride[0] * stride_next[0], stride[1] * stride_next[1]};
                                input_feature.pack_interleaved(input_array, block_shape, total_stride, false,
                                                               this->context.get_parameter().get_default_scale());

                                Duo output_shape = {input_shape[0] / stride[0], input_shape[1] / stride[1]};
                                uint32_t output_total = output_shape[0] * output_shape[1];
                                uint32_t output_n_channel_per_ct;
                                if (2 * output_total < this->N) {
                                    output_n_channel_per_ct = this->N / (2 * output_total);
                                } else {
                                    output_n_channel_per_ct = 1;
                                }

                                Feature2DEncrypted output_feature(&this->context, init_level - 1);
                                output_feature.shape[0] = output_shape[0];
                                output_feature.shape[1] = output_shape[1];
                                output_feature.skip[0] = 1;
                                output_feature.skip[1] = 1;
                                output_feature.n_channel = n_out_channel;
                                output_feature.n_channel_per_ct = output_n_channel_per_ct;
                                int n_out_cts =
                                    div_ceil(n_out_channel, output_n_channel_per_ct) * stride_next[0] * stride_next[1];
                                for (int i = 0; i < n_out_cts; i++) {
                                    output_feature.data.push_back(
                                        this->context.new_ciphertext(init_level - 1, this->param.get_default_scale()));
                                }

                                vector<CxxVectorArgument> cxx_args;
                                fs::path project_path =
                                    base_path / "CKKS_inverse_multiplexed_conv2d" /
                                    ("stride_" + to_string(stride[0]) + "_" + to_string(stride[1])) /
                                    ("kernel_shape_" + to_string(kernel_shape[0]) + "_" + to_string(kernel_shape[1])) /
                                    ("cin_" + to_string(n_in_channel) + "_cout_" + to_string(n_out_channel)) /
                                    ("input_shape_" + to_string(input_shape[0]) + "_" + to_string(input_shape[1])) /
                                    ("level_" + to_string(init_level)) / "server";

                                // Filter input CTs to only include those actually used by the layer
                                auto used_indices = conv_layer.get_used_input_indices();
                                vector<CkksCiphertext> used_input_data;
                                for (auto idx : used_indices) {
                                    used_input_data.push_back(std::move(input_feature.data[idx]));
                                }

                                auto arg_names = read_arg_names(project_path);
                                for (const auto& name : arg_names) {
                                    if (name.rfind("input", 0) == 0)
                                        cxx_args.push_back({name, &used_input_data});
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
            Duo input_shape = {block_shape[0] * 2, block_shape[1] * 2};
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

                        InverseMultiplexedConv2DLayer conv_layer(this->context.get_parameter(), input_shape,
                                                                 conv0_weight, conv0_bias, padding, stride, stride_next,
                                                                 skip, block_shape, init_level, 1.0);
                        conv_layer.prepare_weight();

                        Feature2DEncrypted input_feature(&this->context, init_level, skip);
                        Duo effective_stride = {input_shape[0] / block_shape[0], input_shape[1] / block_shape[1]};
                        Duo total_stride = {effective_stride[0] * 1, effective_stride[1] * 1};
                        input_feature.pack_interleaved(input_array, block_shape, total_stride, false,
                                                       this->context.get_parameter().get_default_scale());

                        uint32_t output_shape0 = input_shape[0] / stride[0];
                        uint32_t output_shape1 = input_shape[1] / stride[1];
                        uint32_t out_skip0 = block_shape[0] / output_shape0;
                        uint32_t out_skip1 = block_shape[1] / output_shape1;
                        uint32_t output_n_channel_per_ct = this->N / (2 * output_shape0 * output_shape1);

                        Feature2DEncrypted output_feature(&this->context, init_level - 2);
                        output_feature.shape[0] = output_shape0;
                        output_feature.shape[1] = output_shape1;
                        output_feature.skip[0] = out_skip0;
                        output_feature.skip[1] = out_skip1;
                        output_feature.n_channel = n_out_channel;
                        output_feature.n_channel_per_ct = output_n_channel_per_ct;
                        int n_out_cts = div_ceil(n_out_channel, output_n_channel_per_ct);
                        for (int i = 0; i < n_out_cts; i++) {
                            output_feature.data.push_back(
                                this->context.new_ciphertext(init_level - 2, this->param.get_default_scale()));
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
                        vector<CkksPlaintextRingt> repack_mask_vec;
                        repack_mask_vec.push_back(conv_layer.generate_repack_mask_pt(this->context));

                        // Filter input CTs to only include those actually used by the layer
                        auto used_indices = conv_layer.get_used_input_indices();
                        vector<CkksCiphertext> used_input_data;
                        for (auto idx : used_indices) {
                            used_input_data.push_back(std::move(input_feature.data[idx]));
                        }

                        auto arg_names = read_arg_names(project_path);
                        for (const auto& name : arg_names) {
                            if (name.rfind("input", 0) == 0)
                                cxx_args.push_back({name, &used_input_data});
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
                    Duo input_shape = {block_shape[0] * mult, block_shape[1] * mult};
                    Duo stride_next = {input_shape[0] / (block_shape[0] * stride[0]),
                                       input_shape[1] / (block_shape[1] * stride[1])};

                    Array<int, 1> padding({2});
                    padding.set(0, -1);
                    padding.set(1, -1);

                    string config_name = "ks_" + to_string(kernel_shape[0]) + "x" + to_string(kernel_shape[1]) +
                                         "_st_" + to_string(stride[0]) + "x" + to_string(stride[1]) + "_bs_" +
                                         to_string(block_shape[0]) + "x" + to_string(block_shape[1]) + "_is_" +
                                         to_string(input_shape[0]) + "x" + to_string(input_shape[1]);

                    SECTION(config_name) {
                        for (uint32_t n_channel : n_channels) {
                            SECTION("ch=" + to_string(n_channel)) {
                                // Depthwise: weight shape is [n_channel, 1, kH, kW]
                                Array<double, 4> conv0_weight =
                                    gen_random_array<4>({n_channel, 1, kernel_shape[0], kernel_shape[1]}, 0.1);
                                Array<double, 1> conv0_bias = gen_random_array<1>({n_channel}, 0.1);
                                // Depthwise: input channels == output channels
                                Array<double, 3> input_array =
                                    gen_random_array<3>({n_channel, input_shape[0], input_shape[1]}, 1.0);

                                InverseMultiplexedConv2DLayerDepthwise conv_layer(
                                    this->context.get_parameter(), input_shape, conv0_weight, conv0_bias, padding,
                                    stride, stride_next, skip, block_shape, init_level, 1.0);
                                conv_layer.prepare_weight();

                                // Pack input using interleaved packing
                                Feature2DEncrypted input_feature(&this->context, init_level, skip);
                                Duo total_stride = {stride[0] * stride_next[0], stride[1] * stride_next[1]};
                                input_feature.pack_interleaved(input_array, block_shape, total_stride, false,
                                                               this->context.get_parameter().get_default_scale());

                                Duo output_shape = {input_shape[0] / stride[0], input_shape[1] / stride[1]};
                                uint32_t output_total = output_shape[0] * output_shape[1];
                                uint32_t output_n_channel_per_ct;
                                if (2 * output_total < this->N) {
                                    output_n_channel_per_ct = this->N / (2 * output_total);
                                } else {
                                    output_n_channel_per_ct = 1;
                                }

                                Feature2DEncrypted output_feature(&this->context, init_level - 1);
                                output_feature.shape[0] = output_shape[0];
                                output_feature.shape[1] = output_shape[1];
                                output_feature.skip[0] = 1;
                                output_feature.skip[1] = 1;
                                output_feature.n_channel = n_channel;
                                output_feature.n_channel_per_ct = output_n_channel_per_ct;
                                int n_out_cts =
                                    div_ceil(n_channel, output_n_channel_per_ct) * stride_next[0] * stride_next[1];
                                for (int i = 0; i < n_out_cts; i++) {
                                    output_feature.data.push_back(
                                        this->context.new_ciphertext(init_level - 1, this->param.get_default_scale()));
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
    for (uint32_t s : skip_shapes) {
        uint32_t skip_0d = s * s;
        SECTION("skip=" + to_string(skip_0d)) {
            uint32_t n_channel_per_ct = this->n_slot / skip_0d;
            uint32_t n_packed_in = div_ceil(n_in_channel, n_channel_per_ct);
            uint32_t n_packed_out = div_ceil(n_out_channel, n_channel_per_ct);

            Feature0DEncrypted input_feature(&this->context, init_level);
            input_feature.pack(input_1d, false, this->param.get_default_scale(), skip_0d);

            DensePackedLayer dense(this->context.get_parameter(), weight, bias, n_channel_per_ct, init_level, 0);
            dense.prepare_weight_0d_skip(skip_0d);

            Feature0DEncrypted output_feature(&this->context, init_level - 1);
            output_feature.n_channel = n_out_channel;
            output_feature.n_channel_per_ct = n_channel_per_ct;
            output_feature.skip = skip_0d;
            for (uint32_t i = 0; i < n_packed_out; i++) {
                output_feature.data.push_back(
                    this->context.new_ciphertext(init_level - 1, this->param.get_default_scale()));
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
            Array<double, 1> plain_output = dense.plaintext_call(input_1d);

            print_double_message(output_mg.to_array_1d().data(), "output_mg", 10);
            print_double_message(plain_output.to_array_1d().data(), "plain_output", 10);

            auto compare_result = compare(plain_output, output_mg);
            REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
            REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
        }
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

    for (auto& cfg : configs) {
        SECTION("shape=" + str(cfg.shape) + " skip=" + str(cfg.skip) + " inv=" + str(cfg.invalid_fill)) {
            auto input_3d = gen_random_array<3>({n_in_channel, cfg.shape[0], cfg.shape[1]}, 0.1);
            auto input_1d = Array<double, 1>::from_array_1d(input_3d.to_array_1d());

            Feature2DEncrypted input_feature(&this->context, init_level, cfg.skip, cfg.invalid_fill);
            input_feature.pack_multiplexed(input_3d, false, this->param.get_default_scale());

            ReshapeLayer reshape(this->param);
            Feature0DEncrypted input_0d = reshape.call(this->context, input_feature);

            uint32_t block_size = (cfg.shape[0] * cfg.skip[0]) * (cfg.shape[1] * cfg.skip[1]);
            uint32_t n_blocks_per_ct = div_ceil((uint32_t)this->n_slot, block_size);

            DensePackedLayer dense(this->context.get_parameter(), weight, bias, n_blocks_per_ct, init_level, 0);
            dense.prepare_weight_for_2d_multiplexed(cfg.shape, cfg.skip, cfg.invalid_fill);

            uint32_t n_packed_out = div_ceil(n_out_channel, n_blocks_per_ct);
            Feature0DEncrypted output_feature(&this->context, init_level - 1);
            output_feature.n_channel = n_out_channel;
            output_feature.n_channel_per_ct = n_blocks_per_ct;
            output_feature.skip = input_0d.skip;
            for (uint32_t i = 0; i < n_packed_out; i++) {
                output_feature.data.push_back(
                    this->context.new_ciphertext(init_level - 1, this->param.get_default_scale()));
            }

            fs::path project_path = base_path /
                                    ("CKKS_fc_multiplexed"
                                     "_shape" +
                                     to_string(cfg.shape[0]) + "x" + to_string(cfg.shape[1]) + "_skip" +
                                     to_string(cfg.skip[0]) + "x" + to_string(cfg.skip[1]) + "_inv" +
                                     to_string(cfg.invalid_fill[0]) + "x" + to_string(cfg.invalid_fill[1])) /
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
            Array<double, 1> plain_output = dense.plaintext_call(input_1d);

            print_double_message(output_mg.to_array_1d().data(), "output_mg", 10);
            print_double_message(plain_output.to_array_1d().data(), "plain_output", 10);

            auto compare_result = compare(plain_output, output_mg);
            REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
            REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
        }
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
    Duo skip1 = {dense_shape[0] * skip0[0], dense_shape[1] * skip0[1]};

    Array<double, 2> weight1 = gen_random_array<2>({output_channel1, output_channel}, 1);
    Array<double, 1> bias1 = gen_random_array<1>({output_channel1}, 1);

    // Input: direct 0D pack — ceil(1024/8192)=1 CT, skip=1 (matches Python)
    Array<double, 1> input_1d = gen_random_array<1>({input_channel}, 0.1);
    Feature0DEncrypted input_feature(&this->context, init_level);
    input_feature.pack(input_1d, false, this->param.get_default_scale(), /*skip=*/1);

    // fc0: multiplexed — block=[4*1]*[4*1]=16 slots, n_num_pre_ct=ceil(8192/16)=512
    uint32_t block_size0 = (dense_shape[0] * skip0[0]) * (dense_shape[1] * skip0[1]);
    uint32_t n_num_pre_ct0 = div_ceil((uint32_t)this->n_slot, block_size0);
    DensePackedLayer dense0(this->context.get_parameter(), weight0, bias0, n_num_pre_ct0, init_level, 0);
    dense0.prepare_weight_for_2d_multiplexed(dense_shape, skip0);

    uint32_t skip_0d1 = skip1[0] * skip1[1];
    uint32_t n_channel_per_ct1 = this->n_slot / skip_0d1;
    DensePackedLayer dense1(this->context.get_parameter(), weight1, bias1, n_channel_per_ct1, init_level - 1, 0);
    dense1.prepare_weight_0d_skip(skip_0d1);

    // Output placeholder
    uint32_t n_out1 = div_ceil(output_channel1, n_channel_per_ct1);
    Feature0DEncrypted output_feature(&this->context, init_level - 2);
    output_feature.n_channel = output_channel1;
    output_feature.n_channel_per_ct = n_channel_per_ct1;
    output_feature.skip = skip_0d1;
    for (uint32_t i = 0; i < n_out1; i++) {
        output_feature.data.push_back(this->context.new_ciphertext(init_level - 2, this->param.get_default_scale()));
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

    Array<double, 1> output_plain_0 = dense0.plaintext_call(input_1d);
    Array<double, 1> output_plain_1 = dense1.plaintext_call(output_plain_0);

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
    uint32_t n_channel_per_ct = div_ceil(this->n_slot, (input_shape[0] * input_shape[1]));
    int init_level = 8;
    vector<int> orders = {2, 4, 6, 8, 10, 12, 16, 32, 64};

    for (uint32_t order : orders) {
        SECTION("order=" + to_string(order)) {
            auto input_array = gen_random_array<3>({n_channel, input_shape[0], input_shape[1]}, 1.0);
            auto weight = gen_random_array<2>({order + 1, n_channel}, 1.0);

            Feature2DEncrypted input_feature(&this->context, init_level, skip);
            input_feature.pack_multiplexed(input_array, false, this->context.get_parameter().get_default_scale());

            PolyRelu polyx(this->context.get_parameter(), {input_shape[0], input_shape[1]}, order, weight, skip,
                           n_channel_per_ct, init_level);
            polyx.prepare_weight_bsgs();

            int output_level = init_level - PolyRelu::compute_bsgs_level_cost(order);
            Feature2DEncrypted output_feature(&this->context, output_level);
            output_feature.skip = skip;
            output_feature.shape = input_shape;
            output_feature.n_channel = n_channel;
            output_feature.n_channel_per_ct = input_feature.n_channel_per_ct;
            for (int i = 0; i < div_ceil(n_channel, n_channel_per_ct); i++) {
                output_feature.data.push_back(
                    this->context.new_ciphertext(output_level, this->param.get_default_scale()));
            }

            fs::path project_path =
                base_path / ("CKKS_poly_relu_bsgs_" + to_string(n_channel) + "_channel_order_" + to_string(order)) /
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
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "poly_bsgs_feature0d", "", HeteroProcessors) {
    uint32_t n_channel = 32;
    int init_level = 8;
    vector<int> orders = {2, 4, 6, 8};
    vector<uint32_t> skips = {1, 2, 128, 256};

    for (uint32_t skip_val : skips) {
        SECTION("skip=" + to_string(skip_val)) {
            uint32_t n_channel_per_ct = this->n_slot / skip_val;

            for (uint32_t order : orders) {
                int level_cost = PolyRelu::compute_bsgs_level_cost(order);
                if (init_level < level_cost)
                    continue;

                SECTION("order=" + to_string(order)) {
                    auto input_array = gen_random_array<1>({n_channel}, 1.0);
                    auto weight = gen_random_array<2>({order + 1, n_channel}, 0.5);

                    // Pack into Feature0DEncrypted
                    Feature0DEncrypted input_feature(&this->context, init_level);
                    input_feature.n_channel = n_channel;
                    input_feature.pack(input_array, false, this->param.get_default_scale(), skip_val);

                    // Create PolyRelu0D for Feature0D
                    PolyRelu0D polyx(this->context.get_parameter(), weight, init_level, order, skip_val);
                    polyx.prepare_weight_0d_skip();

                    int output_level = init_level - level_cost;
                    uint32_t n_packed_ct = div_ceil(n_channel, n_channel_per_ct);

                    Feature0DEncrypted output_feature(&this->context, output_level);
                    output_feature.skip = skip_val;
                    output_feature.n_channel = n_channel;
                    output_feature.n_channel_per_ct = n_channel_per_ct;
                    for (uint32_t i = 0; i < n_packed_ct; i++) {
                        output_feature.data.push_back(
                            this->context.new_ciphertext(output_level, this->param.get_default_scale()));
                    }

                    fs::path project_path = base_path /
                                            ("CKKS_poly_relu_bsgs_feature0d_" + to_string(n_channel) +
                                             "_channel_order_" + to_string(order) + "_skip_" + to_string(skip_val)) /
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
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "block_ccmm_matmul", "", HeteroProcessors) {
    vector<uint32_t> ds = {16};
    vector<uint32_t> dims = {16, 20};
    vector<int> levels = {3};

    for (uint32_t d : ds) {
        SECTION("d=" + to_string(d)) {
            for (int init_level : levels) {
                SECTION("level=" + to_string(init_level)) {
                    for (uint32_t m : dims) {
                        SECTION("m=" + to_string(m)) {
                            for (uint32_t n : dims) {
                                SECTION("n=" + to_string(n)) {
                                    for (uint32_t p : dims) {
                                        SECTION("p=" + to_string(p)) {
                                            Array<double, 2> A_mat = gen_random_array<2>({m, n}, 1.0);
                                            Array<double, 2> B_mat = gen_random_array<2>({n, p}, 1.0);

                                            FeatureMatEncrypted A_enc(&this->context, init_level);
                                            A_enc.shape = {m, n};
                                            A_enc.matmul_block_size = d;
                                            A_enc.block_col_major_pack(
                                                A_mat, d, false, this->context.get_parameter().get_default_scale());

                                            FeatureMatEncrypted B_enc(&this->context, init_level);
                                            B_enc.shape = {n, p};
                                            B_enc.matmul_block_size = d;
                                            B_enc.block_col_major_pack(
                                                B_mat, d, false, this->context.get_parameter().get_default_scale());

                                            BlockColMajorCCMM ccmm(this->context.get_parameter(), A_enc.shape,
                                                                   B_enc.shape, A_enc.matmul_block_size,
                                                                   B_enc.matmul_block_size, A_enc.level, B_enc.level);
                                            ccmm.precompute_diagonals();
                                            FeatureMatEncrypted C_enc = ccmm.run(this->context, A_enc, B_enc);

                                            auto C_result = C_enc.block_col_major_unpack(C_enc.shape[0], C_enc.shape[1],
                                                                                         C_enc.matmul_block_size);

                                            Array<double, 2> C_expected({m, p});
                                            for (uint32_t i = 0; i < m; i++) {
                                                for (uint32_t j = 0; j < p; j++) {
                                                    double sum = 0.0;
                                                    for (uint32_t l = 0; l < n; l++) {
                                                        sum += A_mat.get(i, l) * B_mat.get(l, j);
                                                    }
                                                    C_expected.set(i, j, sum);
                                                }
                                            }

                                            print_double_message(C_result.to_array_1d().data(), "output_mg", 10);
                                            print_double_message(C_expected.to_array_1d().data(), "output_mg_expected",
                                                                 10);

                                            auto compare_result = compare(C_expected, C_result);
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
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "block_cpmm_matmul", "", HeteroProcessors) {
    vector<uint32_t> ds = {16};
    vector<uint32_t> dims = {16, 20};
    vector<int> levels = {1};

    for (uint32_t d : ds) {
        SECTION("d=" + to_string(d)) {
            for (int init_level : levels) {
                SECTION("level=" + to_string(init_level)) {
                    for (uint32_t m : dims) {
                        SECTION("m=" + to_string(m)) {
                            for (uint32_t n : dims) {
                                SECTION("n=" + to_string(n)) {
                                    for (uint32_t p : dims) {
                                        SECTION("p=" + to_string(p)) {
                                            Array<double, 2> A_mat = gen_random_array<2>({m, n}, 1.0);
                                            Array<double, 2> B_mat = gen_random_array<2>({n, p}, 1.0);

                                            FeatureMatEncrypted A_enc(&this->context, init_level);
                                            A_enc.shape = {m, n};
                                            A_enc.matmul_block_size = d;
                                            A_enc.block_col_major_pack(
                                                A_mat, d, false, this->context.get_parameter().get_default_scale());

                                            BlockColMajorCPMM cpmm(this->context.get_parameter(), A_enc.shape, {n, p},
                                                                   B_mat, d, A_enc.level);
                                            cpmm.precompute_diagonals();
                                            FeatureMatEncrypted C_enc = cpmm.run(this->context, A_enc);

                                            auto C_result = C_enc.block_col_major_unpack(C_enc.shape[0], C_enc.shape[1],
                                                                                         C_enc.matmul_block_size);

                                            Array<double, 2> C_expected({m, p});
                                            for (uint32_t i = 0; i < m; i++) {
                                                for (uint32_t j = 0; j < p; j++) {
                                                    double sum = 0.0;
                                                    for (uint32_t l = 0; l < n; l++) {
                                                        sum += A_mat.get(i, l) * B_mat.get(l, j);
                                                    }
                                                    C_expected.set(i, j, sum);
                                                }
                                            }

                                            print_double_message(C_result.to_array_1d().data(), "output_mg", 10);
                                            print_double_message(C_expected.to_array_1d().data(), "output_mg_expected",
                                                                 10);

                                            auto compare_result = compare(C_expected, C_result);
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
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "block_transpose", "", HeteroProcessors) {
    vector<uint32_t> ds = {16};
    vector<uint32_t> dims = {16, 20};
    vector<int> levels = {1};

    for (uint32_t d : ds) {
        SECTION("d=" + to_string(d)) {
            for (int init_level : levels) {
                SECTION("level=" + to_string(init_level)) {
                    for (uint32_t m : dims) {
                        SECTION("m=" + to_string(m)) {
                            for (uint32_t n : dims) {
                                SECTION("n=" + to_string(n)) {
                                    Array<double, 2> A_mat = gen_random_array<2>({m, n}, 1.0);

                                    FeatureMatEncrypted A_enc(&this->context, init_level);
                                    A_enc.shape = {m, n};
                                    A_enc.matmul_block_size = d;
                                    A_enc.block_col_major_pack(A_mat, d, false,
                                                               this->context.get_parameter().get_default_scale());

                                    BlockColMajorTranspose bt(this->context.get_parameter(), A_enc.shape,
                                                              A_enc.matmul_block_size, A_enc.level);
                                    bt.precompute_diagonals();
                                    FeatureMatEncrypted AT_enc = bt.run(this->context, A_enc);

                                    auto AT_result = AT_enc.block_col_major_unpack(AT_enc.shape[0], AT_enc.shape[1],
                                                                                   AT_enc.matmul_block_size);

                                    Array<double, 2> AT_expected({n, m});
                                    for (uint32_t i = 0; i < m; i++) {
                                        for (uint32_t j = 0; j < n; j++) {
                                            AT_expected.set(j, i, A_mat.get(i, j));
                                        }
                                    }

                                    print_double_message(AT_result.to_array_1d().data(), "output_mg", 10);
                                    print_double_message(AT_expected.to_array_1d().data(), "output_mg_expected", 10);

                                    auto compare_result = compare(AT_expected, AT_result);
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
    uint32_t n_channel_per_ct = div_ceil(this->n_slot, (input_shape[0] * input_shape[1]));
    int order0 = 7;
    int order1 = 7;
    int level_cost0 = PolyRelu::compute_bsgs_level_cost(order0);
    int level_cost1 = PolyRelu::compute_bsgs_level_cost(order1);
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
    input_feature.pack_multiplexed(input_array, false, this->context.get_parameter().get_default_scale());

    // Layer 0: p0(x)
    PolyRelu poly0(this->context.get_parameter(), input_shape, order0, weight0, skip, n_channel_per_ct, init_level);
    poly0.prepare_weight_bsgs();

    // Layer 1: sign(x) ≈ p1(p0(x))
    PolyRelu poly1(this->context.get_parameter(), input_shape, order1, weight1, skip, n_channel_per_ct,
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
        output_feature.data.push_back(this->context.new_ciphertext(output_level, this->param.get_default_scale()));
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

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "conv1d", "", HeteroProcessors) {
    uint32_t n_in_channel = 4;
    uint32_t n_out_channel = 4;
    int init_level = 5;

    vector<uint32_t> input_shapes = {32, 64, 512};
    vector<uint32_t> kernel_shapes = {1, 3, 5};
    vector<uint32_t> skips = {2, 4};
    vector<uint32_t> strides = {1, 2};

    for (uint32_t s : input_shapes) {
        uint32_t input_shape = s;
        SECTION("input_shape=" + str({input_shape})) {
            for (uint32_t k : kernel_shapes) {
                uint32_t kernel_shape = k;
                SECTION("kernel_shape=" + str({kernel_shape})) {
                    for (uint32_t s0 : skips) {
                        uint32_t skip = s0;
                        uint32_t n_channel_per_ct = div_ceil(this->N / 2, input_shape * skip);
                        SECTION("skip=" + str({skip})) {
                            for (uint32_t s1 : strides) {
                                uint32_t stride = s1;
                                SECTION("stride=" + str({stride})) {
                                    Array<double, 3> conv0_weight =
                                        gen_random_array<3>({n_out_channel, n_in_channel, kernel_shape}, 1.0);
                                    Array<double, 1> conv0_bias = gen_random_array<1>({n_out_channel}, 1.0);
                                    Array<double, 2> input_array =
                                        gen_random_array<2>({n_in_channel, input_shape}, 1.0);

                                    Feature1DEncrypted input_feature(&this->context, init_level, skip);
                                    input_feature.pack(input_array);
                                    Conv1DPackedLayer conv0_layer(this->context.get_parameter(), input_shape,
                                                                  conv0_weight, conv0_bias, stride, skip,
                                                                  n_channel_per_ct, init_level);
                                    conv0_layer.prepare_weight();

                                    Feature1DEncrypted output_feature(&this->context, init_level - 1, skip * stride);
                                    output_feature.shape = input_shape / stride;
                                    output_feature.skip = skip * stride;
                                    output_feature.n_channel = n_out_channel;
                                    output_feature.n_channel_per_ct = n_channel_per_ct;
                                    for (int i = 0; i < div_ceil(n_out_channel, n_channel_per_ct); i++) {
                                        output_feature.data.push_back(this->context.new_ciphertext(
                                            init_level - 1, this->param.get_default_scale()));
                                    }

                                    fs::path project_path = base_path /
                                                            ("conv1d_input_shape_" + to_string(input_shape) +
                                                             "_kernel_shape_" + to_string(kernel_shape) + "_skip_" +
                                                             to_string(skip) + "_stride_" + to_string(stride)) /
                                                            ("level_" + to_string(init_level)) / "server";

                                    auto arg_names = read_arg_names(project_path);
                                    vector<CxxVectorArgument> cxx_args;
                                    cxx_args.push_back({arg_names[0], &input_feature.data});
                                    cxx_args.push_back({arg_names[1], &conv0_layer.weight_pt});
                                    cxx_args.push_back({arg_names[2], &conv0_layer.bias_pt});
                                    cxx_args.push_back({arg_names[3], &output_feature.data});

                                    this->run(project_path, cxx_args);

                                    Array<double, 2> output_mg = output_feature.unpack();
                                    Array<double, 2> plain_output = conv0_layer.plaintext_call(input_array);

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
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "multiplexed_conv1d", "", HeteroProcessors) {
    uint32_t n_in_channel = 16;
    uint32_t n_out_channel = 32;
    int init_level = 5;

    vector<uint32_t> input_shapes = {32, 64, 512};
    vector<uint32_t> kernel_shapes = {1, 3, 5};
    vector<uint32_t> skips = {2, 4};
    vector<uint32_t> strides = {1, 2};

    for (uint32_t s : input_shapes) {
        uint32_t input_shape = s;
        SECTION("input_shape=" + str({input_shape})) {
            for (uint32_t k : kernel_shapes) {
                if (k > input_shape) {
                    continue;
                }
                uint32_t kernel_shape = k;
                SECTION("kernel_shape=" + str({kernel_shape})) {
                    for (uint32_t s0 : skips) {
                        uint32_t skip = s0;
                        uint32_t n_channel_per_ct = div_ceil(this->N / 2, input_shape);
                        SECTION("skip=" + str({skip})) {
                            for (uint32_t s1 : strides) {
                                uint32_t stride = s1;
                                SECTION("stride=" + str({stride})) {
                                    Array<double, 3> conv0_weight =
                                        gen_random_array<3>({n_out_channel, n_in_channel, kernel_shape}, 1.0);
                                    Array<double, 1> conv0_bias = gen_random_array<1>({n_out_channel}, 1.0);
                                    Array<double, 2> input_array =
                                        gen_random_array<2>({n_in_channel, input_shape}, 1.0);

                                    Feature1DEncrypted input_feature(&this->context, init_level, skip);
                                    input_feature.pack_multiplexed(input_array, false, this->param.get_default_scale());
                                    ParMultiplexedConv1DPackedLayer conv0_layer(
                                        this->context.get_parameter(), input_shape, conv0_weight, conv0_bias, stride,
                                        skip, n_channel_per_ct, init_level);
                                    conv0_layer.prepare_weight();

                                    bool needs_rearrange = (skip > 1 || stride > 1);
                                    int output_level = needs_rearrange ? init_level - 2 : init_level - 1;
                                    uint32_t n_block_per_ct = div_ceil(n_channel_per_ct, skip);
                                    uint32_t n_output_cts = needs_rearrange ?
                                                                div_ceil(n_out_channel, n_channel_per_ct) :
                                                                div_ceil(n_out_channel, n_block_per_ct);

                                    Feature1DEncrypted output_feature(&this->context, output_level, skip * stride);
                                    output_feature.shape = input_shape / stride;
                                    output_feature.skip = skip * stride;
                                    output_feature.n_channel = n_out_channel;
                                    output_feature.n_channel_per_ct = n_channel_per_ct;
                                    for (uint32_t i = 0; i < n_output_cts; i++) {
                                        output_feature.data.push_back(this->context.new_ciphertext(
                                            output_level, this->param.get_default_scale()));
                                    }

                                    uint32_t n_select_pt = min(n_block_per_ct, n_out_channel);
                                    vector<CkksPlaintextRingt> select_pt_subset;
                                    for (int i = 0; i < n_select_pt; i++) {
                                        select_pt_subset.push_back(move(conv0_layer.block_select_pt[i]));
                                    }

                                    fs::path project_path =
                                        base_path /
                                        ("multiplexed_conv1d_input_shape_" + to_string(input_shape) + "_kernel_shape_" +
                                         to_string(kernel_shape) + "_skip_" + to_string(skip) + "_stride_" +
                                         to_string(stride)) /
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
                                    Array<double, 2> plain_output = conv0_layer.plaintext_call(input_array);

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
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "par_block_qkt_v_attention", "", HeteroProcessors) {
    double default_scale = this->context.get_parameter().get_default_scale();

    auto run_attention_test = [&](uint32_t seq_len, uint32_t n_heads, uint32_t head_dim, int init_level) {
        uint32_t d = head_dim;
        uint32_t total_dim = n_heads * head_dim;

        // Use small scale to keep values manageable through two multiplications
        Array<double, 2> Q_mat = gen_random_array<2>({seq_len, total_dim}, 1.0);
        Array<double, 2> K_mat = gen_random_array<2>({seq_len, total_dim}, 0.1);
        Array<double, 2> V_mat = gen_random_array<2>({seq_len, total_dim}, 1.0);

        // Pack Q, K, V using par_block_col_major_pack
        FeatureMatEncrypted Q_enc(&this->context, init_level);
        Q_enc.shape = {seq_len, head_dim};
        Q_enc.matmul_block_size = d;
        Q_enc.par_block_col_major_pack(Q_mat, d, n_heads, false, default_scale);

        FeatureMatEncrypted K_enc(&this->context, init_level);
        K_enc.shape = {seq_len, head_dim};
        K_enc.matmul_block_size = d;
        K_enc.par_block_col_major_pack(K_mat, d, n_heads, false, default_scale);

        FeatureMatEncrypted V_enc(&this->context, init_level);
        V_enc.shape = {seq_len, head_dim};
        V_enc.matmul_block_size = d;
        V_enc.par_block_col_major_pack(V_mat, d, n_heads, false, default_scale);

        // Step 1: Transpose K -> K^T (per head: seq_len x head_dim -> head_dim x seq_len)
        ParBlockColMajorTranspose kt_transpose(this->context.get_parameter(), {seq_len, head_dim}, d, n_heads,
                                               init_level);
        kt_transpose.precompute_diagonals();
        FeatureMatEncrypted KT_enc = kt_transpose.run(this->context, K_enc);
        // KT_enc at level init_level - 1, shape = {head_dim, seq_len}

        // Step 2: Drop Q to match K^T level
        FeatureMatEncrypted Q_dropped = Q_enc.drop_level(1);
        Q_dropped.matmul_block_size = d;

        // Step 3: Q * K^T  (per head: seq_len x head_dim @ head_dim x seq_len = seq_len x seq_len)
        ParBlockColMajorCCMM qkt_ccmm(this->context.get_parameter(), {seq_len, head_dim}, {head_dim, seq_len}, d,
                                      n_heads, init_level - 1);
        qkt_ccmm.precompute_diagonals();
        FeatureMatEncrypted attn_enc = qkt_ccmm.run(this->context, Q_dropped, KT_enc);
        // attn_enc at level init_level - 4, shape = {seq_len, seq_len}

        // Step 4: Drop V to match attention scores level
        FeatureMatEncrypted V_dropped = V_enc.drop_level(4);
        V_dropped.matmul_block_size = d;

        // Step 5: attn_scores * V  (per head: seq_len x seq_len @ seq_len x head_dim = seq_len x head_dim)
        ParBlockColMajorCCMM attnv_ccmm(this->context.get_parameter(), {seq_len, seq_len}, {seq_len, head_dim}, d,
                                        n_heads, init_level - 4);
        attnv_ccmm.precompute_diagonals();
        FeatureMatEncrypted result_enc = attnv_ccmm.run(this->context, attn_enc, V_dropped);
        // result_enc at level init_level - 7 = 0, shape = {seq_len, head_dim}

        // Unpack result
        auto result = result_enc.par_block_col_major_unpack(seq_len, head_dim, d, n_heads);

        // Compute expected: per head, Q_h @ K_h^T @ V_h
        Array<double, 2> expected({(uint64_t)seq_len, (uint64_t)total_dim});
        for (uint32_t h = 0; h < n_heads; h++) {
            for (uint32_t i = 0; i < seq_len; i++) {
                for (uint32_t j = 0; j < head_dim; j++) {
                    double sum = 0.0;
                    for (uint32_t k1 = 0; k1 < seq_len; k1++) {
                        double attn_score = 0.0;
                        for (uint32_t k2 = 0; k2 < head_dim; k2++) {
                            attn_score += Q_mat.get(i, h * head_dim + k2) * K_mat.get(k1, h * head_dim + k2);
                        }
                        sum += attn_score * V_mat.get(k1, h * head_dim + j);
                    }
                    expected.set(i, h * head_dim + j, sum);
                }
            }
        }

        print_double_message(result.to_array_1d().data(), "output_mg", 10);
        print_double_message(expected.to_array_1d().data(), "output_mg_expected", 10);

        auto compare_result = compare(expected, result);
        std::cout << "max_error=" << compare_result.max_error << " max_abs=" << compare_result.max_abs
                  << " rmse=" << compare_result.rmse << " rms=" << compare_result.rms << std::endl;
        REQUIRE(compare_result.max_error < 0.05 * compare_result.max_abs);
        REQUIRE(compare_result.rmse < 0.01 * compare_result.rms);
    };

    SECTION("G=1: seq_len=53, n_heads=3, head_dim=16") {
        // Level budget: transpose(1) + CCMM(3) + CCMM(3) = 7
        run_attention_test(53, 3, 16, 7);
    }

    // G=2 disabled: head_dim=64 causes n_h_padded*d^2 > n_slot, requires larger N
    // SECTION("G=2: seq_len=83, n_heads=3, head_dim=64") {
    //     // n_slot=8192, d=64, d^2=4096, n_h_padded=4, n_h_padded*d^2=16384 > 8192
    //     // S=8192/4096=2, G=4/2=2
    //     // Level budget: transpose(1) + CCMM(3) + CCMM(3) = 7
    //     run_attention_test(83, 3, 64, 7);
    // }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "par_cpmm_square", "", HeteroProcessors) {
    double default_scale = this->context.get_parameter().get_default_scale();

    auto run_square_test = [&](uint32_t m, uint32_t n_heads, uint32_t head_dim, int init_level) {
        uint32_t d = head_dim;
        uint32_t total_dim = n_heads * head_dim;

        Array<double, 2> A_mat = gen_random_array<2>({m, total_dim}, 0.5);
        Array<double, 2> W_mat = gen_random_array<2>({total_dim, total_dim}, 0.1);

        FeatureMatEncrypted A_enc(&this->context, init_level);
        A_enc.shape = {m, head_dim};
        A_enc.matmul_block_size = d;
        A_enc.par_block_col_major_pack(A_mat, d, n_heads, false, default_scale);

        ParBlockColMajorCPMM cpmm(this->context.get_parameter(), {m, head_dim}, W_mat, d, n_heads, init_level);
        cpmm.precompute_diagonals();
        FeatureMatEncrypted result_enc = cpmm.run(this->context, A_enc);

        auto result = result_enc.par_block_col_major_unpack(m, head_dim, d, n_heads);

        Array<double, 2> expected({(uint64_t)m, (uint64_t)total_dim});
        for (uint32_t i = 0; i < m; i++) {
            for (uint32_t j = 0; j < total_dim; j++) {
                double sum = 0.0;
                for (uint32_t k = 0; k < total_dim; k++) {
                    sum += A_mat.get(i, k) * W_mat.get(k, j);
                }
                expected.set(i, j, sum);
            }
        }

        print_double_message(result.to_array_1d().data(), "cpmm_output", 10);
        print_double_message(expected.to_array_1d().data(), "cpmm_expected", 10);

        auto compare_result = compare(expected, result);
        std::cout << "max_error=" << compare_result.max_error << " max_abs=" << compare_result.max_abs
                  << " rmse=" << compare_result.rmse << " rms=" << compare_result.rms << std::endl;
        REQUIRE(compare_result.max_error < 0.05 * compare_result.max_abs);
        REQUIRE(compare_result.rmse < 0.01 * compare_result.rms);
    };

    SECTION("G=1: m=53, n_heads=3, head_dim=16") {
        run_square_test(53, 3, 16, 2);
    }

    SECTION("G=2: m=83, n_heads=3, head_dim=64") {
        // n_slot=8192, d=64, d^2=4096, n_h_padded=4, n_h_padded*d^2=16384 > 8192
        // S=8192/4096=2, G=4/2=2
        run_square_test(83, 3, 64, 2);
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "par_cpmm_expand_reduce", "", HeteroProcessors) {
    double default_scale = this->context.get_parameter().get_default_scale();

    auto run_expand_reduce_test = [&](uint32_t m, uint32_t n_heads, uint32_t head_dim, uint32_t K, int init_level) {
        uint32_t d = head_dim;
        uint32_t total_dim = n_heads * head_dim;
        uint32_t expanded_dim = K * total_dim;

        Array<double, 2> A_mat = gen_random_array<2>({m, total_dim}, 0.5);
        Array<double, 2> W1_mat = gen_random_array<2>({total_dim, expanded_dim}, 0.1);
        Array<double, 2> W2_mat = gen_random_array<2>({expanded_dim, total_dim}, 0.1);

        FeatureMatEncrypted A_enc(&this->context, init_level);
        A_enc.shape = {m, head_dim};
        A_enc.matmul_block_size = d;
        A_enc.par_block_col_major_pack(A_mat, d, n_heads, false, default_scale);

        // Expand: A @ W1
        ParBlockColMajorCPMM expand_cpmm(this->context.get_parameter(), {m, head_dim}, W1_mat, d, n_heads, init_level);
        expand_cpmm.precompute_diagonals();
        FeatureMatEncrypted mid_enc = expand_cpmm.run(this->context, A_enc);

        // Reduce: (A @ W1) @ W2
        ParBlockColMajorCPMM reduce_cpmm(this->context.get_parameter(), {m, head_dim}, W2_mat, d, n_heads,
                                         mid_enc.level);
        reduce_cpmm.precompute_diagonals();
        FeatureMatEncrypted result_enc = reduce_cpmm.run(this->context, mid_enc);

        auto result = result_enc.par_block_col_major_unpack(m, head_dim, d, n_heads);

        // Expected: (A @ W1) @ W2
        Array<double, 2> expected({(uint64_t)m, (uint64_t)total_dim});
        std::vector<double> mid(m * expanded_dim, 0.0);
        for (uint32_t i = 0; i < m; i++) {
            for (uint32_t j = 0; j < expanded_dim; j++) {
                double sum = 0.0;
                for (uint32_t k = 0; k < total_dim; k++) {
                    sum += A_mat.get(i, k) * W1_mat.get(k, j);
                }
                mid[i * expanded_dim + j] = sum;
            }
        }
        for (uint32_t i = 0; i < m; i++) {
            for (uint32_t j = 0; j < total_dim; j++) {
                double sum = 0.0;
                for (uint32_t k = 0; k < expanded_dim; k++) {
                    sum += mid[i * expanded_dim + k] * W2_mat.get(k, j);
                }
                expected.set(i, j, sum);
            }
        }

        print_double_message(result.to_array_1d().data(), "mlp_output", 10);
        print_double_message(expected.to_array_1d().data(), "mlp_expected", 10);

        auto compare_result = compare(expected, result);
        std::cout << "max_error=" << compare_result.max_error << " max_abs=" << compare_result.max_abs
                  << " rmse=" << compare_result.rmse << " rms=" << compare_result.rms << std::endl;
        REQUIRE(compare_result.max_error < 0.05 * compare_result.max_abs);
        REQUIRE(compare_result.rmse < 0.01 * compare_result.rms);
    };

    SECTION("G=1: m=53, n_heads=3, head_dim=16, K=4") {
        run_expand_reduce_test(53, 3, 16, 4, 4);
    }

    // G=2 disabled: head_dim=64 causes n_h_padded*d^2 > n_slot, requires larger N
    // SECTION("G=2: m=83, n_heads=3, head_dim=64, K=4") {
    //     // n_slot=8192, d=64, d^2=4096, n_h_padded=4, n_h_padded*d^2=16384 > 8192
    //     // S=8192/4096=2, G=4/2=2
    //     run_expand_reduce_test(83, 3, 64, 4, 4);
    // }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "add_layer", "", HeteroProcessors) {
    int init_level = 2;
    Duo skip = {1, 1};

    auto run_add_test = [&](uint32_t n_channel, uint32_t s) {
        Duo input_shape = {s, s};
        uint32_t n_channel_per_ct = div_ceil(this->n_slot, (input_shape[0] * input_shape[1]));
        uint32_t n_ct = div_ceil(n_channel, n_channel_per_ct);

        Array<double, 3> input_x0 = gen_random_array<3>({n_channel, input_shape[0], input_shape[1]}, 1.0);
        Array<double, 3> input_x1 = gen_random_array<3>({n_channel, input_shape[0], input_shape[1]}, 1.0);

        Feature2DEncrypted x0_enc(&this->context, init_level, skip);
        x0_enc.pack_multiplexed(input_x0, false, this->param.get_default_scale());

        Feature2DEncrypted x1_enc(&this->context, init_level, skip);
        x1_enc.pack_multiplexed(input_x1, false, this->param.get_default_scale());

        // Pre-allocate output (add doesn't consume levels)
        Feature2DEncrypted output_feature(&this->context, init_level);
        for (uint32_t i = 0; i < n_ct; i++) {
            output_feature.data.push_back(this->context.new_ciphertext(init_level, this->param.get_default_scale()));
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

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "avgpool2d_layer", "", HeteroProcessors) {
    int init_level = 3;
    Duo skip = {1, 1};

    auto run_avgpool_test = [&](uint32_t n_channel, uint32_t s, const Duo& stride) {
        Duo input_shape = {s, s};
        uint32_t n_channel_per_ct = div_ceil(this->n_slot, (input_shape[0] * input_shape[1]));
        uint32_t n_ct = div_ceil(n_channel, n_channel_per_ct);

        Array<double, 3> input_array = gen_random_array<3>({n_channel, input_shape[0], input_shape[1]}, 1.0);

        Feature2DEncrypted input_feature(&this->context, init_level, skip);
        input_feature.pack_multiplexed(input_array, false, this->param.get_default_scale());

        // Prepare select_tensor_pt via Avgpool2DLayer
        Avgpool2DLayer avgpool(input_shape, stride);
        avgpool.prepare_weight(this->param, n_channel_per_ct, n_channel, init_level, skip, input_shape);

        // Pre-allocate output (rescale consumes one level)
        uint32_t out_channels_per_ct = n_channel_per_ct * stride[0] * stride[1];
        uint32_t n_packed_out_channel = div_ceil(n_channel, out_channels_per_ct);
        Feature2DEncrypted output_feature(&this->context, init_level - 1);
        for (uint32_t i = 0; i < n_packed_out_channel; i++) {
            output_feature.data.push_back(
                this->context.new_ciphertext(init_level - 1, this->param.get_default_scale()));
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
        output_feature.skip = {skip[0] * stride[0], skip[1] * stride[1]};
        output_feature.n_channel = n_channel;
        output_feature.n_channel_per_ct = n_channel_per_ct * stride[0] * stride[1];
        output_feature.shape = {input_shape[0] / stride[0], input_shape[1] / stride[1]};
        auto result_mg = output_feature.unpack_multiplexed();

        auto result_expected = avgpool.plaintext_call_multiplexed(input_array);

        print_double_message(result_mg.to_array_1d().data(), "output_mg", 10);
        print_double_message(result_expected.to_array_1d().data(), "plain_output", 10);

        auto compare_result = compare(result_expected, result_mg);
        REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
        REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
    };

    vector<uint32_t> shapes = {8, 16, 32, 64};
    vector<uint32_t> channels = {4, 10, 15, 32, 37};
    vector<Duo> strides = {{2, 2}, {4, 4}, {8, 8}};

    for (const auto& stride : strides) {
        SECTION("stride=" + to_string(stride[0])) {
            for (uint32_t n_channel : channels) {
                SECTION("n_channel=" + to_string(n_channel)) {
                    for (uint32_t s : shapes) {
                        if (s < stride[0])
                            continue;  // shape must be >= stride
                        SECTION("shape=" + to_string(s) + "x" + to_string(s)) {
                            run_avgpool_test(n_channel, s, stride);
                        }
                    }
                }
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "adaptive_avgpool2d_layer", "", HeteroProcessors) {
    int init_level = 3;
    Duo skip = {1, 1};

    auto run_adaptive_avgpool_test = [&](uint32_t n_channel, uint32_t s, const Duo& stride) {
        Duo input_shape = {s, s};
        uint32_t n_channel_per_ct = div_ceil(this->n_slot, (input_shape[0] * input_shape[1]));
        uint32_t n_ct = div_ceil(n_channel, n_channel_per_ct);

        Array<double, 3> input_array = gen_random_array<3>({n_channel, input_shape[0], input_shape[1]}, 1.0);

        Feature2DEncrypted input_feature(&this->context, init_level, skip);
        input_feature.pack_multiplexed(input_array, false, this->param.get_default_scale());

        // No prepare_weight needed for adaptive avgpool
        Avgpool2DLayer avgpool(input_shape, stride);

        // Pre-allocate output — NO level consumed (no mult/rescale), same number of CTs
        Feature2DEncrypted output_feature(&this->context, init_level);
        for (uint32_t i = 0; i < n_ct; i++) {
            output_feature.data.push_back(this->context.new_ciphertext(init_level, this->param.get_default_scale()));
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
        output_feature.skip = {skip[0] * stride[0], skip[1] * stride[1]};
        output_feature.invalid_fill = {stride[0], stride[1]};
        output_feature.n_channel = n_channel;
        output_feature.n_channel_per_ct = n_channel_per_ct;
        output_feature.shape = {input_shape[0] / stride[0], input_shape[1] / stride[1]};
        auto result_mg = output_feature.unpack_multiplexed();

        auto result_expected = avgpool.plaintext_call(input_array);

        print_double_message(result_mg.to_array_1d().data(), "output_mg", 10);
        print_double_message(result_expected.to_array_1d().data(), "plain_output", 10);

        auto compare_result = compare(result_expected, result_mg);
        REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
        REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
    };

    vector<uint32_t> channels = {4, 10, 15, 32, 37};
    vector<uint32_t> shapes = {8, 16, 32, 64};
    vector<Duo> strides = {{2, 2}, {4, 4}, {8, 8}};

    for (const auto& stride : strides) {
        SECTION("stride=" + to_string(stride[0])) {
            for (uint32_t n_channel : channels) {
                SECTION("n_channel=" + to_string(n_channel)) {
                    for (uint32_t s : shapes) {
                        if (s < stride[0])
                            continue;
                        SECTION("shape=" + to_string(s) + "x" + to_string(s)) {
                            run_adaptive_avgpool_test(n_channel, s, stride);
                        }
                    }
                }
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "interleaved_avgpool2d_layer", "", HeteroProcessors) {
    int init_level = 3;

    auto run_interleaved_avgpool_test = [&](uint32_t n_channel, const Duo& stride, const Duo& block_shape,
                                            uint32_t mult) {
        Duo input_shape = {block_shape[0] * mult, block_shape[1] * mult};
        Duo block_expansion = {input_shape[0] / block_shape[0], input_shape[1] / block_shape[1]};

        Array<double, 3> input_array = gen_random_array<3>({n_channel, input_shape[0], input_shape[1]}, 1.0);

        Feature2DEncrypted input_feature(&this->context, init_level);
        Duo total_stride = block_expansion;
        input_feature.pack_interleaved(input_array, block_shape, total_stride, false, this->param.get_default_scale());

        Avgpool2DLayer avgpool(block_shape, stride);

        // No level consumed (only adds)
        uint32_t out_size = input_feature.data.size() / (stride[0] * stride[1]);
        Feature2DEncrypted output_feature(&this->context, init_level);
        for (uint32_t i = 0; i < out_size; i++) {
            output_feature.data.push_back(this->context.new_ciphertext(init_level, this->param.get_default_scale()));
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
        Duo out_expansion = {block_expansion[0] / stride[0], block_expansion[1] / stride[1]};
        output_feature.n_channel = n_channel;
        output_feature.n_channel_per_ct = 1;
        output_feature.skip = {1, 1};
        output_feature.shape = {input_shape[0] / stride[0], input_shape[1] / stride[1]};
        auto result_mg = output_feature.unpack_interleaved(block_shape, out_expansion);

        auto result_expected = avgpool.plaintext_call(input_array);

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

    for (const auto& stride : strides) {
        SECTION("stride=" + to_string(stride[0])) {
            for (uint32_t n_channel : channels) {
                SECTION("ch=" + to_string(n_channel)) {
                    for (uint32_t mult : multipliers) {
                        if (mult < stride[0])
                            continue;  // block_expansion must be >= stride
                        SECTION("mult=" + to_string(mult)) {
                            run_interleaved_avgpool_test(n_channel, stride, block_shape, mult);
                        }
                    }
                }
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
        x1_enc.pack_multiplexed(input_x1, false, this->param.get_default_scale());

        Feature2DEncrypted x2_enc(&this->context, init_level, skip);
        x2_enc.pack_multiplexed(input_x2, false, this->param.get_default_scale());

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
        uint32_t n_channel_per_ct = div_ceil(this->n_slot, (input_shape[0] * input_shape[1]));
        uint32_t n_ct = div_ceil(n_channel, n_channel_per_ct);

        Array<double, 3> input_array = gen_random_array<3>({n_channel, input_shape[0], input_shape[1]}, 1.0);
        Array<double, 1> weight = gen_random_array<1>({n_channel}, 1.0);

        Feature2DEncrypted input_feature(&this->context, init_level, skip);
        input_feature.pack_multiplexed(input_array, false, this->param.get_default_scale());

        // Prepare weight plaintexts
        MultScalarLayer mult_layer(this->param, input_shape, weight, skip, n_channel_per_ct, init_level);
        mult_layer.prepare_weight();

        // Pre-allocate output (mult_scalar consumes one level due to rescale)
        Feature2DEncrypted output_feature(&this->context, init_level - 1);
        for (uint32_t i = 0; i < n_ct; i++) {
            output_feature.data.push_back(
                this->context.new_ciphertext(init_level - 1, this->param.get_default_scale()));
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
        Duo block_shape = {input_shape[0] / stride[0], input_shape[1] / stride[1]};
        input_feature.pack_interleaved(input_array, block_shape, stride, false, this->param.get_default_scale());

        UpsampleLayer upsample(this->param, stride, upsample_factor, init_level, n_channel, n_channel_per_ct);
        upsample.prepare_data();
        Feature2DEncrypted result_enc = upsample.run(this->context, input_feature);

        Duo out_stride = {stride[0] * upsample_factor[0], stride[1] * upsample_factor[1]};
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
        uint32_t n_channel_per_ct = div_ceil(this->n_slot, (input_shape[0] * input_shape[1]));

        Array<double, 3> input_array = gen_random_array<3>({n_channel, input_shape[0], input_shape[1]}, 1.0);

        Feature2DEncrypted input_feature(&this->context, init_level, skip);
        input_feature.pack_multiplexed(input_array, false, this->param.get_default_scale());

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
