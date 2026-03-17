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
#include <iomanip>

#include "data_structs/feature.h"
#include "fhe_layers/conv2d_packed_layer.h"
#include "fhe_layers/poly_relu2d.h"
#include "fhe_layers/multiplexed_conv2d_pack_layer.h"
#include "fhe_layers/multiplexed_conv2d_pack_layer_depthwise.h"
#include "fhe_layers/activation_layer.h"
#include "fhe_layers/conv2d_depthwise.h"
#include "fhe_layers/dense_packed_layer.h"
#include "fhe_layers/block_col_major_ccmm.h"
#include "fhe_layers/block_col_major_cpmm.h"
#include "fhe_layers/block_col_major_transpose.h"
#include "fhe_layers/conv1d_packed_layer.h"
#include "fhe_layers/multiplexed_conv1d_pack_layer.h"
#include "ut_util.h"
#include <cxx_sdk_v2/cxx_fhe_task.h>

using namespace cxx_sdk_v2;
using namespace std;

string base_path = "../hetero";

struct TaskMetrics {
    std::string test_name;
    std::string task_config;
    int n;
    std::string processor_type;
    double execution_time_ms;
};

std::string extract_task_config(const string& project_path, const string& base_path) {
    string config = project_path.substr(base_path.length() + 1);
    config = config.substr(0, config.length() - 8);
    return config;
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

template <typename T> class HeteroFixture {
public:
    HeteroFixture()
        : N{16384}, n_slot{N / 2}, param{CkksParameter::create_parameter(N)},
          context{CkksContext::create_random_context(param)}, level(3), min_level{0}, max_level{param.get_max_level()},
          default_scale{param.get_default_scale()} {
        context.gen_rotation_keys();
    }

    ~HeteroFixture() {
        MetricsCollector::save_to_csv("hetero_performance_results.csv");
    }

    uint64_t run(const string& project_path, const vector<CxxVectorArgument>& cxx_args) {
        auto start_time = std::chrono::high_resolution_clock::now();
        uint64_t result;
        std::string processor_type;

        if constexpr (is_same_v<T, ProcessorCpu>) {
            processor_type = "CPU";
            FheTaskCpu fhe_task(project_path);
            result = fhe_task.run(&context, cxx_args);
#ifdef INFERENCE_SDK_ENABLE_GPU
        } else if constexpr (is_same_v<T, ProcessorGpu>) {
            processor_type = "GPU";
            FheTaskGpu fhe_task(project_path);
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
    CkksParameter param;
    CkksContext context;
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
            input_feature.pack(input_array, false, this->param.get_default_scale());

            Feature2DEncrypted output_feature(&this->context, init_level);

            for (int i = 0; i < 1; i++) {
                output_feature.data.push_back(
                    this->context.new_ciphertext(init_level - 1, this->param.get_default_scale()));
            }

            vector<CxxVectorArgument> cxx_args = {
                CxxVectorArgument{"input_node", &input_feature.data},
                CxxVectorArgument{"output_ct", &output_feature.data},
            };

            string project_path = base_path + "/CKKS_square_" + to_string(input_shape[0]) + "_" +
                                  to_string(input_shape[1]) + "/level_" + to_string(init_level) + "/server/";
            cout << "project_path=" << project_path << endl;
            this->run(project_path, cxx_args);

            output_feature.skip = {1, 1};
            output_feature.n_channel = 1;
            output_feature.n_channel_per_ct = this->n_slot / (s * s);
            output_feature.shape = {s, s};
            auto output_mg = output_feature.unpack();

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

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "conv_1ch_s1", "", HeteroProcessors) {
    uint32_t n_in_channel = 1;
    uint32_t n_out_channel = 1;
    Duo stride = {1, 1};
    Duo skip = {1, 1};
    int init_level = 2;

    vector<uint32_t> input_shapes = {4, 8, 16, 32, 64};
    vector<uint32_t> kernel_shapes = {1, 3, 5};

    for (uint32_t s : input_shapes) {
        Duo input_shape = {s, s};
        SECTION("input_shape=" + str(input_shape)) {
            for (uint32_t k : kernel_shapes) {
                Duo kernel_shape = {k, k};
                SECTION("kernel_shape=" + str(kernel_shape)) {
                    Array<double, 4> conv0_weight =
                        gen_random_array<4>({n_out_channel, n_in_channel, kernel_shape[0], kernel_shape[1]}, 0.1);
                    Array<double, 1> conv0_bias = gen_random_array<1>({n_out_channel}, 0);
                    Array<double, 3> input_array =
                        gen_random_array<3>({n_in_channel, input_shape[0], input_shape[1]}, 1.0);

                    uint32_t n_channel_per_ct = div_ceil(this->n_slot, (input_shape[0] * input_shape[1]));

                    Feature2DEncrypted input_feature(&this->context, init_level);
                    input_feature.pack(input_array, false, this->param.get_default_scale());

                    Conv2DPackedLayer conv0_layer(this->context.get_parameter(), input_shape, conv0_weight, conv0_bias,
                                                  stride, skip, n_channel_per_ct, init_level);
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

                    vector<CxxVectorArgument> cxx_args = {
                        CxxVectorArgument{"input_0", &input_feature.data},
                        CxxVectorArgument{"convw__conv1_Conv", &conv0_layer.weight_pt_},
                        CxxVectorArgument{"convb__conv1_Conv", &conv0_layer.bias_pt_},
                        CxxVectorArgument{"output", &output_feature.data},
                    };

                    string project_path = base_path + "/CKKS_conv2d_" + to_string(n_in_channel) + "_in_" +
                                          to_string(n_out_channel) + "_out_channel_1_stride_" +
                                          to_string(input_shape[0]) + "_" + to_string(input_shape[1]) + "_" +
                                          to_string(kernel_shape[0]) + "_" + to_string(kernel_shape[1]) + "/level_" +
                                          to_string(init_level) + "/server/";

                    this->run(project_path, cxx_args);

                    auto output_mg = output_feature.unpack();

                    auto plain_output = conv0_layer.run_plaintext(input_array);

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

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "conv_1ch_s2", "", HeteroProcessors) {
    uint32_t n_in_channel = 1;
    uint32_t n_out_channel = 1;
    Duo stride = {2, 2};
    Duo skip = {1, 1};
    int init_level = 2;

    vector<uint32_t> input_shapes = {32, 64};
    vector<uint32_t> kernel_shapes = {1, 3, 5};

    for (uint32_t s : input_shapes) {
        Duo input_shape = {s, s};
        SECTION("input_shape=" + str(input_shape)) {
            for (uint32_t k : kernel_shapes) {
                Duo kernel_shape = {k, k};
                SECTION("kernel_shape=" + str(kernel_shape)) {
                    Array<double, 4> conv0_weight =
                        gen_random_array<4>({n_out_channel, n_in_channel, kernel_shape[0], kernel_shape[1]}, 0.1);
                    Array<double, 1> conv0_bias = gen_random_array<1>({n_out_channel}, 0.1);
                    Array<double, 3> input_array =
                        gen_random_array<3>({n_in_channel, input_shape[0], input_shape[1]}, 1.0);

                    uint32_t n_channel_per_ct = div_ceil(this->n_slot, (input_shape[0] * input_shape[1]));

                    Feature2DEncrypted input_feature(&this->context, init_level);
                    input_feature.pack(input_array, false, this->param.get_default_scale());

                    Conv2DPackedLayer conv0_layer(this->context.get_parameter(), input_shape, conv0_weight, conv0_bias,
                                                  stride, skip, n_channel_per_ct, init_level);
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

                    vector<CxxVectorArgument> cxx_args = {
                        CxxVectorArgument{"input_0", &input_feature.data},
                        CxxVectorArgument{"convw__conv1_Conv", &conv0_layer.weight_pt_},
                        CxxVectorArgument{"convb__conv1_Conv", &conv0_layer.bias_pt_},
                        CxxVectorArgument{"output", &output_feature.data},
                    };

                    string project_path = base_path + "/CKKS_conv2d_1_in_1_out_channel_2_stride_" +
                                          to_string(input_shape[0]) + "_" + to_string(input_shape[1]) + "_" +
                                          to_string(kernel_shape[0]) + "_" + to_string(kernel_shape[1]) + "/level_" +
                                          to_string(init_level) + "/server/";

                    this->run(project_path, cxx_args);

                    Array<double, 3> output_array = output_feature.unpack();
                    Array<double, 3> output_array_plain_call = conv0_layer.run_plaintext(input_array);

                    auto compare_result = compare(output_array_plain_call, output_array);
                    REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
                    REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
                }
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "conv_mch_s1", "", HeteroProcessors) {
    Duo input_shape = {32, 32};
    Duo kernel_shape = {3, 3};
    Duo stride = {1, 1};
    Duo skip = {1, 1};
    uint32_t n_channel_per_ct = div_ceil(this->n_slot, (input_shape[0] * input_shape[1]));
    int init_level = 2;

    vector<uint32_t> nc_ins = {1, 3, 4, 16, 17};
    vector<uint32_t> nc_outs = {1, 3, 4, 32, 33};

    for (uint32_t n_in_channel : nc_ins) {
        SECTION("n_in_channel=" + to_string(n_in_channel)) {
            for (uint32_t n_out_channel : nc_outs) {
                SECTION("n_out_channel=" + to_string(n_out_channel)) {
                    Array<double, 4> conv0_weight =
                        gen_random_array<4>({n_out_channel, n_in_channel, kernel_shape[0], kernel_shape[1]}, 0.1);
                    Array<double, 1> conv0_bias = gen_random_array<1>({n_out_channel}, 0.1);
                    Array<double, 3> x_mg = gen_random_array<3>({n_in_channel, input_shape[0], input_shape[1]}, 1.0);

                    Conv2DPackedLayer conv_layer(this->context.get_parameter(), input_shape, conv0_weight, conv0_bias,
                                                 stride, skip, n_channel_per_ct, init_level);
                    conv_layer.prepare_weight();

                    Feature2DEncrypted input_feature(&this->context, init_level);
                    input_feature.pack(x_mg, false, this->param.get_default_scale());

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

                    vector<CxxVectorArgument> cxx_args = {
                        CxxVectorArgument{"input_0", &input_feature.data},
                        CxxVectorArgument{"convw__conv1_Conv", &conv_layer.weight_pt_},
                        CxxVectorArgument{"convb__conv1_Conv", &conv_layer.bias_pt_},
                        CxxVectorArgument{"output", &output_feature.data},
                    };

                    string project_path = base_path + "/CKKS_conv2d_" + to_string(n_in_channel) + "_in_" +
                                          to_string(n_out_channel) + "_out_channel_1_stride_32_32_3_3/level_" +
                                          to_string(init_level) + "/server/";

                    this->run(project_path, cxx_args);

                    auto y_mg = output_feature.unpack();
                    auto y_expected = conv_layer.run_plaintext(x_mg);

                    auto compare_result = compare(y_expected, y_mg);
                    REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
                    REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
                }
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "conv_mch_s2", "", HeteroProcessors) {
    Duo input_shape = {32, 32};
    Duo kernel_shape = {3, 3};
    Duo stride = {2, 2};
    Duo skip = {1, 1};
    uint32_t n_channel_per_ct = div_ceil(this->n_slot, (input_shape[0] * input_shape[1]));
    int init_level = 2;

    vector<uint32_t> nc_ins = {1, 3, 4, 16, 17};
    vector<uint32_t> nc_outs = {1, 3, 4, 32, 33};

    for (uint32_t n_in_channel : nc_ins) {
        SECTION("n_in_channel=" + to_string(n_in_channel)) {
            for (uint32_t n_out_channel : nc_outs) {
                SECTION("n_out_channel=" + to_string(n_out_channel)) {
                    Array<double, 4> conv0_weight =
                        gen_random_array<4>({n_out_channel, n_in_channel, kernel_shape[0], kernel_shape[1]}, 0.1);
                    Array<double, 1> conv0_bias = gen_random_array<1>({n_out_channel}, 0.1);
                    Array<double, 3> x_mg = gen_random_array<3>({n_in_channel, input_shape[0], input_shape[1]}, 1.0);

                    Conv2DPackedLayer conv_layer(this->context.get_parameter(), input_shape, conv0_weight, conv0_bias,
                                                 stride, skip, n_channel_per_ct, init_level);
                    conv_layer.prepare_weight();

                    Feature2DEncrypted input_feature(&this->context, init_level);
                    input_feature.pack(x_mg, false, this->param.get_default_scale());

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

                    vector<CxxVectorArgument> cxx_args = {
                        CxxVectorArgument{"input_0", &input_feature.data},
                        CxxVectorArgument{"convw__conv1_Conv", &conv_layer.weight_pt_},
                        CxxVectorArgument{"convb__conv1_Conv", &conv_layer.bias_pt_},
                        CxxVectorArgument{"output", &output_feature.data},
                    };

                    string project_path = base_path + "/CKKS_conv2d_" + to_string(n_in_channel) + +"_in_" +
                                          to_string(n_out_channel) + "_out_channel_2_stride_32_32_3_3/level_" +
                                          to_string(init_level) + "/server/";

                    this->run(project_path, cxx_args);

                    auto y_mg = output_feature.unpack();
                    auto y_expected = conv_layer.run_plaintext(x_mg);

                    auto compare_result = compare(y_expected, y_mg);
                    REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
                    REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
                }
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "dw_32ch_s1_32x32_k3", "", HeteroProcessors) {
    Duo input_shape = {32, 32};
    Duo kernel_shape = {3, 3};
    uint32_t n_in_channel = 4;
    uint32_t n_out_channel = 4;
    Duo stride = {1, 1};
    Duo skip = {1, 1};
    uint32_t n_channel_per_ct = div_ceil(this->n_slot, (input_shape[0] * input_shape[1]));
    int init_level = 5;

    Array<double, 4> conv0_weight =
        gen_random_array<4>({n_out_channel, n_in_channel, kernel_shape[0], kernel_shape[1]}, 0.1);
    Array<double, 1> conv0_bias = gen_random_array<1>({n_out_channel}, 0);
    Array<double, 3> input = gen_random_array<3>({n_out_channel, input_shape[0], input_shape[1]}, 1);

    Conv2DPackedDepthwiseLayer conv(this->context.get_parameter(), input_shape, conv0_weight, conv0_bias, stride, skip,
                                    n_channel_per_ct, init_level);
    conv.prepare_weight();

    Feature2DEncrypted f2d(&this->context, init_level);
    f2d.pack(input, false, this->param.get_default_scale());

    Feature2DEncrypted output_feature(&this->context, init_level - 1);
    output_feature.shape[0] = input_shape[0] / stride[0];
    output_feature.shape[1] = input_shape[1] / stride[1];
    output_feature.skip[0] = skip[0] * stride[0];
    output_feature.skip[1] = skip[1] * stride[1];
    output_feature.n_channel = n_out_channel;
    output_feature.n_channel_per_ct = n_channel_per_ct;
    for (int i = 0; i < div_ceil(n_out_channel, n_channel_per_ct); i++) {
        output_feature.data.push_back(this->context.new_ciphertext(init_level - 1, this->param.get_default_scale()));
    }

    vector<CxxVectorArgument> cxx_args = {
        CxxVectorArgument{"input_0", &f2d.data},
        CxxVectorArgument{"convw__conv1_Conv", &conv.weight_pt_},
        CxxVectorArgument{"convb__conv1_Conv", &conv.bias_pt_},
        CxxVectorArgument{"output", &output_feature.data},
    };

    string project_path =
        base_path + "/CKKS_dw_conv2d_4_in_4_out_channel_2_stride_32_32_3_3/level_" + to_string(init_level) + "/server/";

    this->run(project_path, cxx_args);

    Array<double, 3> output_mg = output_feature.unpack();
    Array<double, 3> plain_output = conv.run_plaintext(input);

    print_double_message(output_mg.to_array_1d().data(), "output_mg", 10);
    print_double_message(plain_output.to_array_1d().data(), "plain_output", 10);

    auto compare_result = compare(plain_output, output_mg);
    REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
    REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "dw_4ch_s2_32x32_k3", "", HeteroProcessors) {
    Duo input_shape = {32, 32};
    Duo kernel_shape = {3, 3};
    uint32_t n_in_channel = 4;
    uint32_t n_out_channel = 4;
    Duo stride = {2, 2};
    Duo skip = {1, 1};
    uint32_t n_channel_per_ct = div_ceil(this->n_slot, (input_shape[0] * input_shape[1]));
    int init_level = 5;

    auto input = gen_random_array<3>({n_in_channel, input_shape[0], input_shape[1]}, 1);
    Array<double, 4> conv0_weight =
        gen_random_array<4>({n_out_channel, n_in_channel, kernel_shape[0], kernel_shape[1]}, 0.1);
    auto conv0_bias = gen_random_array<1>({n_out_channel}, 0);

    Feature2DEncrypted f2d(&this->context, init_level);
    f2d.pack(input, false, this->param.get_default_scale());

    Conv2DPackedDepthwiseLayer conv(this->context.get_parameter(), input_shape, conv0_weight, conv0_bias, stride, skip,
                                    n_channel_per_ct, init_level);
    conv.prepare_weight();

    Feature2DEncrypted output_feature(&this->context, init_level - 1);
    output_feature.shape[0] = input_shape[0] / stride[0];
    output_feature.shape[1] = input_shape[1] / stride[1];
    output_feature.skip[0] = skip[0] * stride[0];
    output_feature.skip[1] = skip[1] * stride[1];
    output_feature.n_channel = n_out_channel;
    output_feature.n_channel_per_ct = n_channel_per_ct;
    for (int i = 0; i < div_ceil(n_out_channel, n_channel_per_ct); i++) {
        output_feature.data.push_back(this->context.new_ciphertext(init_level - 1, this->param.get_default_scale()));
    }

    vector<CxxVectorArgument> cxx_args = {
        CxxVectorArgument{"input_0", &f2d.data},
        CxxVectorArgument{"convw__conv1_Conv", &conv.weight_pt_},
        CxxVectorArgument{"convb__conv1_Conv", &conv.bias_pt_},
        CxxVectorArgument{"output", &output_feature.data},
    };

    string project_path =
        base_path + "/CKKS_dw_conv2d_4_in_4_out_channel_2_stride_32_32_3_3/level_" + to_string(init_level) + "/server/";

    this->run(project_path, cxx_args);

    auto output_mg = output_feature.unpack();
    auto plain_output = conv.run_plaintext(input);

    print_double_message(output_mg.to_array_1d().data(), "output_mg", 10);
    print_double_message(plain_output.to_array_1d().data(), "plain_output", 10);

    auto compare_result = compare(output_mg, plain_output);
    REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
    REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "mux_conv_s1_32x32_k3", "", HeteroProcessors) {
    Duo input_shape = {32, 32};
    Duo kernel_shape = {3, 3};
    Duo stride = {1, 1};
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

                    ParMultiplexedConv2DPackedLayer conv_layer(this->context.get_parameter(), input_shape, conv0_weight,
                                                               conv0_bias, stride, skip, n_channel_per_ct, init_level,
                                                               1.0);
                    conv_layer.prepare_weight_for_post_skip_rotation();

                    Feature2DEncrypted input_feature(&this->context, init_level, skip);
                    input_feature.par_mult_pack(input_array, false, this->context.get_parameter().get_default_scale());

                    Feature2DEncrypted output_feature(&this->context, init_level - 1);
                    output_feature.shape[0] = input_shape[0] / stride[0];
                    output_feature.shape[1] = input_shape[1] / stride[1];
                    output_feature.skip[0] = skip[0] * stride[0];
                    output_feature.skip[1] = skip[1] * stride[1];
                    output_feature.n_channel = n_out_channel;
                    output_feature.n_channel_per_ct = (n_channel_per_ct * stride[0] * stride[1]);
                    for (int i = 0; i < div_ceil(n_out_channel, (n_channel_per_ct * stride[0] * stride[1])); i++) {
                        output_feature.data.push_back(
                            this->context.new_ciphertext(init_level - 1, this->param.get_default_scale()));
                    }

                    vector<CxxVectorArgument> cxx_args = {
                        CxxVectorArgument{"input_0", &input_feature.data},
                        CxxVectorArgument{"convw__conv1_Conv", &conv_layer.weight_pt},
                        CxxVectorArgument{"convb__conv1_Conv", &conv_layer.bias_pt},
                        CxxVectorArgument{"output", &output_feature.data},
                    };

                    string project_path = base_path + "/CKKS_multiplexed_conv2d_" + to_string(n_in_channel) + "_in_" +
                                          to_string(n_out_channel) + "_out_channel_1_stride_32_32_3_3/level_" +
                                          to_string(init_level) + "/server/";

                    this->run(project_path, cxx_args);

                    auto y_mg = output_feature.par_mult_unpack();
                    auto y_expected = conv_layer.run_plaintext(input_array);

                    auto compare_result = compare(y_expected, y_mg);
                    REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
                    REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
                }
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "mux_conv_s2_32x32_k3", "", HeteroProcessors) {
    Duo input_shape = {32, 32};
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

                    ParMultiplexedConv2DPackedLayer conv_layer(this->context.get_parameter(), input_shape, conv0_weight,
                                                               conv0_bias, stride, skip, n_channel_per_ct, init_level,
                                                               1.0);
                    conv_layer.prepare_weight_for_post_skip_rotation();

                    Feature2DEncrypted input_feature(&this->context, init_level, skip);
                    input_feature.par_mult_pack(input_array, false, this->context.get_parameter().get_default_scale());

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

                    vector<CxxVectorArgument> cxx_args = {
                        CxxVectorArgument{"input_0", &input_feature.data},
                        CxxVectorArgument{"convm__conv1_Conv", &conv_layer.mask_pt},
                        CxxVectorArgument{"convw__conv1_Conv", &conv_layer.weight_pt},
                        CxxVectorArgument{"convb__conv1_Conv", &conv_layer.bias_pt},
                        CxxVectorArgument{"output", &output_feature.data},
                    };

                    string project_path = base_path + "/CKKS_multiplexed_conv2d_" + to_string(n_in_channel) + "_in_" +
                                          to_string(n_out_channel) + "_out_channel_2_stride_32_32_3_3/level_" +
                                          to_string(init_level) + "/server/";

                    this->run(project_path, cxx_args);

                    auto y_mg = output_feature.par_mult_unpack();
                    auto y_expected = conv_layer.run_plaintext(input_array);

                    auto compare_result = compare(y_expected, y_mg);
                    REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
                    REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
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
                    input_feature.par_mult_pack(input_array, false, this->context.get_parameter().get_default_scale());

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

                    vector<CxxVectorArgument> cxx_args = {
                        CxxVectorArgument{"input_0", &input_feature.data},
                        CxxVectorArgument{"convm__conv1_Conv", &dw_conv_layer.mask_pt},
                        CxxVectorArgument{"convw__conv1_Conv", &dw_conv_layer.weight_pt},
                        CxxVectorArgument{"convb__conv1_Conv", &dw_conv_layer.bias_pt},
                        CxxVectorArgument{"output", &output_feature.data},
                    };

                    string project_path = base_path + "/CKKS_multiplexed_dw_conv2d_" + to_string(n_in_channel) +
                                          "_in_" + to_string(n_out_channel) + "_out_channel_2_stride_64_64_3_3/level_" +
                                          to_string(init_level) + "/server/";

                    this->run(project_path, cxx_args);

                    auto y_mg = output_feature.par_mult_unpack();
                    auto y_expected = dw_conv_layer.run_plaintext(input_array);

                    auto compare_result = compare(y_expected, y_mg);
                    REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
                    REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
                }
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "fc_cyclic", "", HeteroProcessors) {
    Duo w_shape = {128, 512};
    Duo b_shape = {128, 1};
    Duo skip = {1, 1};
    uint32_t init_level = 2;

    Array<double, 1> x_mg = gen_random_array<1>({w_shape[1]}, 0.1);
    Array<double, 2> weight = gen_random_array<2>({w_shape[0], w_shape[1]}, 1);
    Array<double, 1> bias = gen_random_array<1>({b_shape[0]}, 1);

    vector<uint32_t> input_shapes = {1, 2, 4, 8, 16};
    for (uint32_t s : input_shapes) {
        Duo input_shape = {s, s};
        uint32_t n_channel_per_ct = div_ceil(this->n_slot, input_shape[0] * input_shape[1]);
        SECTION("input_shape=" + str(input_shape)) {
            DensePackedLayer fc_layer(this->context.get_parameter(), input_shape, skip, weight, bias, n_channel_per_ct,
                                      init_level, 0);
            fc_layer.prepare_weight1();

            Feature0DEncrypted x_ct(&this->context, init_level);
            x_ct.skip = input_shape[0] * input_shape[1];
            x_ct.pack_cyclic(x_mg.to_array_1d(), false, this->param.get_default_scale());

            Feature0DEncrypted output_feature(&this->context, init_level - 1);
            output_feature.skip = x_ct.skip;
            output_feature.n_channel = w_shape[0];
            output_feature.n_channel_per_ct = n_channel_per_ct;
            for (int i = 0; i < div_ceil(output_feature.n_channel, n_channel_per_ct); i++) {
                output_feature.data.push_back(
                    this->context.new_ciphertext(init_level - 1, this->param.get_default_scale()));
            }

            vector<CxxVectorArgument> cxx_args = {
                CxxVectorArgument{"input_node", &x_ct.data},
                CxxVectorArgument{"weight_pt", &fc_layer.weight_pt},
                CxxVectorArgument{"bias_pt", &fc_layer.bias_pt},
                CxxVectorArgument{"output_ct", &output_feature.data},
            };

            string project_path = base_path + "/CKKS_fc_prepare_weight1_1D_pack_cyclic_" + to_string(input_shape[0]) +
                                  "_" + to_string(input_shape[1]) + "/level_" + to_string(init_level) + "/server/";
            this->run(project_path, cxx_args);

            DecryptType dec_type = DecryptType::SPARSE;
            auto output_mg = output_feature.unpack(dec_type);
            Array<double, 1> plain_output = fc_layer.plaintext_call(x_mg);

            print_double_message(output_mg.to_array_1d().data(), "output_mg", 128);
            print_double_message(plain_output.to_array_1d().data(), "plain_output", 128);

            auto compare_result = compare(plain_output, output_mg);
            REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
            REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "fc_skip", "", HeteroProcessors) {
    Duo input_shape = {1, 1};
    Duo w_shape = {10, 4096};
    Duo b_shape = {10, 1};
    uint32_t init_level = 2;

    Array<double, 1> input_array = gen_random_array<1>({w_shape[1]}, 0.1);
    Array<double, 2> weight = gen_random_array<2>({w_shape[0], w_shape[1]}, 1);
    Array<double, 1> bias = gen_random_array<1>({b_shape[0]}, 1);

    vector<uint32_t> skip_shapes = {2, 4, 8, 16};
    for (uint32_t s : skip_shapes) {
        Duo skip = {s, s};
        SECTION("skip=" + str(skip)) {
            uint32_t n_channel_per_ct = div_ceil(this->n_slot, skip[0] * skip[1]);
            DensePackedLayer fc_layer(this->context.get_parameter(), input_shape, skip, weight, bias, n_channel_per_ct,
                                      init_level, 0);
            fc_layer.prepare_weight1();
            Feature0DEncrypted x_ct(&this->context, init_level);
            x_ct.skip = skip[0] * skip[1];
            x_ct.pack_skip(input_array, false);

            Feature0DEncrypted output_feature(&this->context, init_level - 1);
            output_feature.skip = x_ct.skip;
            output_feature.n_channel = w_shape[0];
            output_feature.n_channel_per_ct = n_channel_per_ct;
            for (int i = 0; i < div_ceil(output_feature.n_channel, n_channel_per_ct); i++) {
                output_feature.data.push_back(
                    this->context.new_ciphertext(init_level - 1, this->param.get_default_scale()));
            }

            vector<CxxVectorArgument> cxx_args = {
                CxxVectorArgument{"input_node", &x_ct.data},
                CxxVectorArgument{"weight_pt", &fc_layer.weight_pt},
                CxxVectorArgument{"bias_pt", &fc_layer.bias_pt},
                CxxVectorArgument{"output_ct", &output_feature.data},
            };

            string project_path = base_path + "/CKKS_fc_prepare_weight1_1D_pack_skip_" + to_string(skip[0]) + "_" +
                                  to_string(skip[1]) + "/level_" + to_string(init_level) + "/server/";
            this->run(project_path, cxx_args);

            DecryptType dec_type = DecryptType::SPARSE;
            auto output_mg = output_feature.unpack(dec_type);

            auto plain_output = fc_layer.plaintext_call(input_array);

            print_double_message(output_mg.to_array_1d().data(), "output_mg", 10);
            print_double_message(plain_output.to_array_1d().data(), "plain_output", 10);

            auto compare_result = compare(plain_output, output_mg);
            REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
            REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
        }
    }
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "fc_fc", "", HeteroProcessors) {
    int init_level = 2;

    // fc0
    uint32_t input_channel = 1024;
    uint32_t output_channel = 1024;
    Duo dense_shape = {4, 4};
    Duo skip = {1, 1};

    Array<double, 2> weight0 = gen_random_array<2>({output_channel, input_channel}, 1);
    Array<double, 1> bias0 = gen_random_array<1>({output_channel}, 1);

    // fc1
    uint32_t output_channel1 = 128;
    Duo dense_shape1 = {1, 1};
    Duo skip1 = {dense_shape[0] * skip[0], dense_shape[1] * skip[1]};
    Array<double, 2> weight1 = gen_random_array<2>({output_channel1, output_channel}, 1);
    Array<double, 1> bias1 = gen_random_array<1>({output_channel1}, 1);

    // input data
    Array<double, 1> input = gen_random_array<1>({input_channel}, 0.1);

    Feature0DEncrypted input_feature(&this->context, init_level);
    input_feature.skip = dense_shape[0] * dense_shape[1];
    input_feature.pack_cyclic(input.to_array_1d(), false, this->param.get_default_scale());
    input_feature.n_channel = input_channel;
    input_feature.n_channel_per_ct = div_ceil(this->n_slot, dense_shape[0] * dense_shape[1]);

    DensePackedLayer dense(this->context.get_parameter(), dense_shape, skip, weight0, bias0,
                           input_feature.n_channel_per_ct, init_level, 0);
    dense.prepare_weight1();

    DensePackedLayer dense1(this->context.get_parameter(), dense_shape1, skip1, weight1, bias1,
                            input_feature.n_channel_per_ct, init_level - 1, 0);
    dense1.prepare_weight1();

    Feature0DEncrypted output_feature(&this->context, init_level - 2);
    output_feature.skip = skip1[0] * skip1[1];
    output_feature.n_channel = output_channel1;
    output_feature.n_channel_per_ct = div_ceil(this->n_slot, dense_shape1[0] * dense_shape1[1]);
    for (int i = 0; i < div_ceil(output_feature.n_channel, output_feature.n_channel_per_ct); i++) {
        output_feature.data.push_back(this->context.new_ciphertext(init_level - 2, this->param.get_default_scale()));
    }

    vector<CxxVectorArgument> cxx_args = {
        CxxVectorArgument{"input_node", &input_feature.data}, CxxVectorArgument{"weight_pt0", &dense.weight_pt},
        CxxVectorArgument{"bias_pt0", &dense.bias_pt},        CxxVectorArgument{"weight_pt1", &dense1.weight_pt},
        CxxVectorArgument{"bias_pt1", &dense1.bias_pt},       CxxVectorArgument{"output_ct", &output_feature.data}};

    string project_path = base_path + "/CKKS_fc_fc_" + to_string(input_channel) + "_" + to_string(output_channel) +
                          "_" + to_string(output_channel1) + "/level_" + to_string(init_level) + "/server/";

    this->run(project_path, cxx_args);

    Array<double, 1> output_mg = output_feature.unpack(DecryptType::SPARSE);

    Array<double, 1> output_plain_0 = dense.plaintext_call(input);
    Array<double, 1> output_plain_1 = dense1.plaintext_call(output_plain_0);

    print_double_message(output_mg.to_array_1d().data(), "output_mg", 10);
    print_double_message(output_plain_1.to_array_1d().data(), "plain_output", 10);
    ArrayComparison result = compare(output_plain_1, output_mg);
    REQUIRE(result.max_error < 5.0e-2 * result.max_abs);
    REQUIRE(result.rmse < 1.0e-2 * result.rms);
}

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "poly_bsgs", "", HeteroProcessors) {
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
            input_feature.par_mult_pack(input_array, false, this->context.get_parameter().get_default_scale());

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

            vector<CxxVectorArgument> cxx_args;
            cxx_args.push_back(CxxVectorArgument{"input_node", &input_feature.data});
            for (int i = 0; i <= order; i++) {
                cxx_args.push_back(CxxVectorArgument{"weight_pt" + to_string(i), &polyx.weight_pt[i]});
            }
            cxx_args.push_back(CxxVectorArgument{"output_ct", &output_feature.data});

            string project_path = base_path + "/CKKS_poly_relu_bsgs_" + to_string(n_channel) + "_channel_order_" +
                                  to_string(order) + "/level_" + to_string(init_level);

            this->run(project_path, cxx_args);

            auto output_mg = output_feature.par_mult_unpack();
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
                    input_feature.skip = skip_val;
                    input_feature.n_channel = n_channel;
                    input_feature.pack_skip(input_array, false);

                    // Create PolyRelu for Feature0D: input_shape={1,1}, skip={skip_val, 1}
                    Duo input_shape = {1, 1};
                    Duo skip = {skip_val, 1};
                    PolyRelu polyx(this->context.get_parameter(), input_shape, order, weight, skip, n_channel_per_ct,
                                   init_level, {1, 1}, {1, 1}, true);
                    polyx.prepare_weight_for_feature0d();

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

                    vector<CxxVectorArgument> cxx_args;
                    cxx_args.push_back(CxxVectorArgument{"input_node", &input_feature.data});
                    for (int i = 0; i <= (int)order; i++) {
                        cxx_args.push_back(CxxVectorArgument{"weight_pt" + to_string(i), &polyx.weight_pt[i]});
                    }
                    cxx_args.push_back(CxxVectorArgument{"output_ct", &output_feature.data});

                    string project_path = base_path + "/CKKS_poly_relu_bsgs_feature0d_" + to_string(n_channel) +
                                          "_channel_order_" + to_string(order) + "_skip_" + to_string(skip_val) +
                                          "/level_" + to_string(init_level);

                    this->run(project_path, cxx_args);

                    auto output_mg = output_feature.unpack(DecryptType::SPARSE);
                    auto output_mg_expected = polyx.run_plaintext_for_non_absorb_case_0d(input_array);

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

                                            Feature2DEncrypted A_enc(&this->context, init_level);
                                            A_enc.shape = {m, n};
                                            A_enc.matmul_block_size = d;
                                            A_enc.block_col_major_pack(
                                                A_mat, d, false, this->context.get_parameter().get_default_scale());

                                            Feature2DEncrypted B_enc(&this->context, init_level);
                                            B_enc.shape = {n, p};
                                            B_enc.matmul_block_size = d;
                                            B_enc.block_col_major_pack(
                                                B_mat, d, false, this->context.get_parameter().get_default_scale());

                                            BlockColMajorCCMM ccmm(this->context.get_parameter(), A_enc.shape,
                                                                   B_enc.shape, A_enc.matmul_block_size,
                                                                   B_enc.matmul_block_size, A_enc.level, B_enc.level);
                                            ccmm.precompute_diagonals();
                                            Feature2DEncrypted C_enc = ccmm.run(this->context, A_enc, B_enc);

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

                                            Feature2DEncrypted A_enc(&this->context, init_level);
                                            A_enc.shape = {m, n};
                                            A_enc.matmul_block_size = d;
                                            A_enc.block_col_major_pack(
                                                A_mat, d, false, this->context.get_parameter().get_default_scale());

                                            BlockColMajorCPMM cpmm(this->context.get_parameter(), A_enc.shape, {n, p},
                                                                   B_mat, d, A_enc.level);
                                            cpmm.precompute_diagonals();
                                            Feature2DEncrypted C_enc = cpmm.run(this->context, A_enc);

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

                                    Feature2DEncrypted A_enc(&this->context, init_level);
                                    A_enc.shape = {m, n};
                                    A_enc.matmul_block_size = d;
                                    A_enc.block_col_major_pack(A_mat, d, false,
                                                               this->context.get_parameter().get_default_scale());

                                    BlockColMajorTranspose bt(this->context.get_parameter(), A_enc.shape,
                                                              A_enc.matmul_block_size, A_enc.level);
                                    bt.precompute_diagonals();
                                    Feature2DEncrypted AT_enc = bt.run(this->context, A_enc);

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

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "poly_relu_bsgs", "", HeteroProcessors) {
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
    input_feature.par_mult_pack(input_array, false, this->context.get_parameter().get_default_scale());

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

    // Build cxx_args matching Python naming: poly0_weight_pt{i}, poly1_weight_pt{i}
    vector<CxxVectorArgument> cxx_args;
    cxx_args.push_back(CxxVectorArgument{"input_node", &input_feature.data});
    for (int i = 0; i <= order0; i++) {
        cxx_args.push_back(CxxVectorArgument{"poly0_weight_pt" + to_string(i), &poly0.weight_pt[i]});
    }
    for (int i = 0; i <= order1; i++) {
        cxx_args.push_back(CxxVectorArgument{"poly1_weight_pt" + to_string(i), &poly1.weight_pt[i]});
    }
    cxx_args.push_back(CxxVectorArgument{"output_ct", &output_feature.data});

    string project_path = base_path + "/CKKS_poly_relu_bsgs_" + to_string(n_channel) + "_channel_order_" +
                          to_string(order0) + "_" + to_string(order1) + "/level_" + to_string(init_level);

    this->run(project_path, cxx_args);

    auto output_mg = output_feature.par_mult_unpack();

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

                                    vector<CxxVectorArgument> cxx_args;
                                    cxx_args.push_back(CxxVectorArgument{"input_node", &input_feature.data});
                                    cxx_args.push_back(CxxVectorArgument{"weight_pt", &conv0_layer.weight_pt});
                                    cxx_args.push_back(CxxVectorArgument{"bias_pt", &conv0_layer.bias_pt});
                                    cxx_args.push_back(CxxVectorArgument{"output_ct", &output_feature.data});

                                    string project_path = base_path + "/conv1d_input_shape_" + to_string(input_shape) +
                                                          "_kernel_shape_" + to_string(kernel_shape) + "_skip_" +
                                                          to_string(skip) + "_stride_" + to_string(stride) + "/level_" +
                                                          to_string(init_level) + "/server/";

                                    this->run(project_path, cxx_args);

                                    Array<double, 2> output_mg = output_feature.unpack();
                                    Array<double, 2> plain_output = conv0_layer.plaintext_call(input_array);

                                    print_double_message(output_mg.to_array_1d().data(), "output_mg", 10);
                                    print_double_message(plain_output.to_array_1d().data(), "plain_output", 10);

                                    auto compare_result = compare(plain_output.to_array_2d(), output_mg.to_array_2d());
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
                                    input_feature.par_mult_pack(input_array, false, this->param.get_default_scale());
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

                                    vector<CxxVectorArgument> cxx_args;
                                    cxx_args.push_back(CxxVectorArgument{"input_node", &input_feature.data});
                                    cxx_args.push_back(CxxVectorArgument{"weight_pt", &conv0_layer.weight_pt});
                                    cxx_args.push_back(CxxVectorArgument{"bias_pt", &conv0_layer.bias_pt});
                                    if (needs_rearrange) {
                                        cxx_args.push_back(CxxVectorArgument{"block_select_pt", &select_pt_subset});
                                    }
                                    cxx_args.push_back(CxxVectorArgument{"output_ct", &output_feature.data});

                                    string project_path =
                                        base_path + "/multiplexed_conv1d_input_shape_" + to_string(input_shape) +
                                        "_kernel_shape_" + to_string(kernel_shape) + "_skip_" + to_string(skip) +
                                        "_stride_" + to_string(stride) + "/level_" + to_string(init_level) + "/server/";

                                    this->run(project_path, cxx_args);

                                    Array<double, 2> output_mg = output_feature.par_mult_unpack();
                                    Array<double, 2> plain_output = conv0_layer.plaintext_call(input_array);

                                    print_double_message(output_mg.to_array_1d().data(), "output_mg", 10);
                                    print_double_message(plain_output.to_array_1d().data(), "plain_output", 10);

                                    auto compare_result = compare(plain_output.to_array_2d(), output_mg.to_array_2d());
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
