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
#include "fhe_layers/par_block_col_major_add_pt.h"
#include "fhe_layers/conv1d_packed_layer.h"
#include "fhe_layers/multiplexed_conv1d_pack_layer.h"
#include "fhe_layers/multiplexed_conv1d_depthwise_pack_layer.h"
#include "fhe_layers/inverse_multiplexed_conv2d_layer.h"
#include "fhe_layers/inverse_multiplexed_conv2d_layer_depthwise.h"
#include "fhe_layers/add_layer.h"
#include "fhe_layers/avgpool2d_layer.h"
#include "fhe_layers/avgpool1d_layer.h"
#include "fhe_layers/concat_layer.h"
#include "fhe_layers/mult_scaler.h"
#include "fhe_layers/upsample_layer.h"
#include "fhe_layers/upsample_nearest_layer.h"
#include "fhe_layers/block_col_major_layernorm.h"
#include "fhe_layers/par_block_col_major_layernorm.h"
#include "fhe_layers/block_col_major_polyactrn.h"
#include "fhe_layers/par_block_col_major_polyactrn.h"
#include "data_structs/feature_mat.h"
#include "ut_util.h"
#include <cxx_sdk_v2/cxx_fhe_task.h>
#include <cxx_sdk_v2/cxx_argument.h>
#include <lattisense/lib/nlohmann/json.hpp>
#include <any>
#include <unordered_map>

using namespace std;
using namespace lattisense;
using namespace fhe_ops_lib;
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

#define FOR_EACH_SECTION(var_decl, range, section_name)                                                                \
    for (var_decl : range)                                                                                             \
    SECTION(section_name)

#define FOR_EACH_SECTION_IF(var_decl, range, condition, section_name)                                                  \
    for (var_decl : range)                                                                                             \
        if (!(condition)) {                                                                                            \
        } else                                                                                                         \
            SECTION(section_name)

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "sq", "", HeteroProcessors) {
    int init_level = 2;
    vector<Duo> input_shapes = {{16, 16}, {32, 32}, {64, 64}};

    FOR_EACH_SECTION(const Duo& input_shape, input_shapes, "input_shape=" + str(input_shape)) {
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
        output_feature.n_channel_per_ct = this->n_slot / prod(input_shape);
        output_feature.shape = input_shape;
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
        input_feature.pack_multiple_channel(input_array, false, this->param.get_default_scale());

        Conv2DPackedLayer conv0_layer(this->context.get_parameter(), input_shape, move(conv0_weight), move(conv0_bias),
                                      stride, skip, n_channel_per_ct, init_level);
        conv0_layer.prepare_weight();

        Feature2DEncrypted output_feature(&this->context, init_level - 1);
        output_feature.shape = input_shape / stride;
        output_feature.skip = skip * stride;
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

                    Conv2DPackedDepthwiseLayer conv(this->context.get_parameter(), input_shape, move(conv0_weight),
                                                    move(conv0_bias), stride, skip, n_channel_per_ct, init_level);
                    conv.prepare_weight();

                    Feature2DEncrypted f2d(&this->context, init_level);
                    f2d.pack_multiple_channel(input, false, this->param.get_default_scale());

                    Feature2DEncrypted output_feature(&this->context, init_level - 1);
                    output_feature.shape = input_shape / stride;
                    output_feature.skip = skip * stride;
                    output_feature.n_channel = n_channel;
                    output_feature.n_channel_per_ct = n_channel_per_ct;
                    for (int i = 0; i < div_ceil(n_channel, n_channel_per_ct); i++) {
                        output_feature.data.push_back(
                            this->context.new_ciphertext(init_level - 1, this->param.get_default_scale()));
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

        MultiplexedConv2DPackedLayer conv_layer(this->context.get_parameter(), input_shape, move(conv0_weight),
                                                move(conv0_bias), stride, skip, n_channel_per_ct, init_level, 1.0);
        conv_layer.prepare_weight_for_post_skip_rotation();

        Feature2DEncrypted input_feature(&this->context, init_level, skip);
        input_feature.pack_multiplexed(input_array, false, this->context.get_parameter().get_default_scale());

        Feature2DEncrypted output_feature(&this->context, output_level);
        output_feature.shape = input_shape / stride;
        output_feature.skip = skip * stride;
        output_feature.n_channel = n_out_channel;
        output_feature.n_channel_per_ct = (n_channel_per_ct * prod(stride));
        for (int i = 0; i < div_ceil(n_out_channel, (n_channel_per_ct * prod(stride))); i++) {
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

            MultiplexedConv2DPackedLayerDepthwise dw_conv_layer(this->context.get_parameter(), input_shape,
                                                                move(conv0_weight), move(conv0_bias), stride, skip,
                                                                n_channel_per_ct, init_level, 1.0);
            dw_conv_layer.prepare_weight();

            Feature2DEncrypted input_feature(&this->context, init_level, skip);
            input_feature.pack_multiplexed(input_array, false, this->context.get_parameter().get_default_scale());

            Feature2DEncrypted output_feature(&this->context, init_level - 2);
            output_feature.shape = input_shape / stride;
            output_feature.skip = skip * stride;
            output_feature.n_channel = n_out_channel;
            output_feature.n_channel_per_ct = (n_channel_per_ct * prod(stride));
            for (int i = 0; i < div_ceil(n_out_channel, (n_channel_per_ct * prod(stride))); i++) {
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

                                InverseMultiplexedConv2DLayer conv_layer(this->context.get_parameter(), input_shape,
                                                                         move(conv0_weight), move(conv0_bias), padding,
                                                                         stride, block_shape, init_level, 1.0);
                                conv_layer.prepare_weight();

                                Feature2DEncrypted input_feature(&this->context, init_level, skip);
                                Duo total_stride = stride * stride_next;
                                input_feature.pack_interleaved(input_array, block_shape, total_stride, false,
                                                               this->context.get_parameter().get_default_scale());

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

                        InverseMultiplexedConv2DLayer conv_layer(this->context.get_parameter(), input_shape,
                                                                 move(conv0_weight), move(conv0_bias), padding, stride,
                                                                 block_shape, init_level, 1.0);
                        conv_layer.prepare_weight();

                        Feature2DEncrypted input_feature(&this->context, init_level, skip);
                        Duo effective_stride = input_shape / block_shape;
                        Duo total_stride = effective_stride;
                        input_feature.pack_interleaved(input_array, block_shape, total_stride, false,
                                                       this->context.get_parameter().get_default_scale());

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
                                this->context.get_parameter(), input_shape, move(conv0_weight), move(conv0_bias),
                                padding, stride, block_shape, init_level, 1.0);
                            conv_layer.prepare_weight();

                            // Pack input using interleaved packing
                            Feature2DEncrypted input_feature(&this->context, init_level, skip);
                            Duo total_stride = stride * stride_next;
                            input_feature.pack_interleaved(input_array, block_shape, total_stride, false,
                                                           this->context.get_parameter().get_default_scale());

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
        input_feature.pack(input_1d, false, this->param.get_default_scale(), skip_0d);

        DensePackedLayer dense(this->context.get_parameter(), move(weight), move(bias), n_channel_per_ct, init_level,
                               0);
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
        input_feature.pack_multiplexed(input_3d, false, this->param.get_default_scale());

        ReshapeLayer reshape(this->param);
        Feature0DEncrypted input_0d = reshape.call(this->context, input_feature);

        uint32_t block_size = prod(cfg.shape * cfg.skip);
        uint32_t n_blocks_per_ct = div_ceil((uint32_t)this->n_slot, block_size);

        DensePackedLayer dense(this->context.get_parameter(), move(weight), move(bias), n_blocks_per_ct, init_level, 0);
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
    input_feature.pack(input_1d, false, this->param.get_default_scale(), /*skip=*/1);

    // fc0: multiplexed — block=[4*1]*[4*1]=16 slots, n_num_pre_ct=ceil(8192/16)=512
    uint32_t block_size0 = prod(dense_shape * skip0);
    uint32_t n_num_pre_ct0 = div_ceil((uint32_t)this->n_slot, block_size0);
    DensePackedLayer dense0(this->context.get_parameter(), move(weight0), move(bias0), n_num_pre_ct0, init_level, 0);
    dense0.prepare_weight_for_2d_multiplexed(dense_shape, skip0);

    uint32_t skip_0d1 = prod(skip1);
    uint32_t n_channel_per_ct1 = this->n_slot / skip_0d1;
    DensePackedLayer dense1(this->context.get_parameter(), move(weight1), move(bias1), n_channel_per_ct1,
                            init_level - 1, 0);
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
        input_feature.pack_multiplexed(input_array, false, this->context.get_parameter().get_default_scale());

        PolyRelu2D polyx(this->context.get_parameter(), {input_shape[0], input_shape[1]}, order, move(weight), skip,
                         n_channel_per_ct, init_level);
        polyx.prepare_weight_bsgs();

        int output_level = init_level - PolyRelu2D::compute_bsgs_level_cost(order);
        Feature2DEncrypted output_feature(&this->context, output_level);
        output_feature.skip = skip;
        output_feature.shape = input_shape;
        output_feature.n_channel = n_channel;
        output_feature.n_channel_per_ct = input_feature.n_channel_per_ct;
        for (int i = 0; i < div_ceil(n_channel, n_channel_per_ct); i++) {
            output_feature.data.push_back(this->context.new_ciphertext(output_level, this->param.get_default_scale()));
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
                input_feature.pack(input_array, false, this->param.get_default_scale(), skip_val);

                // Create PolyRelu0D for Feature0D
                PolyRelu0D polyx(this->context.get_parameter(), move(weight), init_level, order, skip_val);
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
                        input_feature.pack(input_array, false, this->param.get_default_scale());

                        PolyRelu1D polyx(this->context.get_parameter(), move(weight), init_level, order, skip_val,
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
                            output_feature.data.push_back(
                                this->context.new_ciphertext(output_level, this->param.get_default_scale()));
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
                        input_feature.pack_multiplexed(input_array, false, this->param.get_default_scale());

                        PolyRelu1D polyx(this->context.get_parameter(), move(weight), init_level, order, skip_val,
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
                            output_feature.data.push_back(
                                this->context.new_ciphertext(output_level, this->param.get_default_scale()));
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
    return [](ExecutionContext& exec_ctx, const std::unordered_map<NodeIndex, std::any>& inputs, std::any& output,
              const ComputeNode& self) -> void {
        CkksContext* ctx_ptr = exec_ctx.get_arithmetic_context<CkksContext>();
        if (!ctx_ptr)
            ctx_ptr = exec_ctx.get_arithmetic_context<CkksBtpContext>();
        if (!ctx_ptr)
            throw std::runtime_error("encode_pt: cannot get CkksContext");
        if (!self.custom_prop.has_value())
            throw std::runtime_error("encode_pt: missing custom_prop");

        const auto& attrs = self.custom_prop->attributes;
        const std::string op_class = attrs.at("op_class").get<std::string>();
        const std::string type = attrs.at("type").get<std::string>();

        NodeIndex in_idx = self.input_nodes[0]->index;
        auto cd_ptr = std::any_cast<std::shared_ptr<CustomData>>(inputs.at(in_idx));
        void* layer_ptr = cd_ptr->get_typed_data<void>();

        CkksPlaintextRingt pt = [&]() -> CkksPlaintextRingt {
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
                return layer->generate_gamma_pt(*ctx_ptr, attrs.value("mb", 0), attrs.value("bj", 0),
                                                attrs.value("g", 0));
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
            throw std::runtime_error("encode_pt: unknown op_class: " + op_class);
        }();

        output = std::make_shared<CkksPlaintextRingt>(std::move(pt));
    };
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
        A_enc.par_block_col_major_pack(A, d, cfg.n_heads, d, false, res.param.get_default_scale());

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
        uint32_t n_slot = res.param.get_n() / 2;
        uint32_t G = (n_slot >= n_h_padded * d * d) ? 1 : n_h_padded / (n_slot / (d * d));
        uint32_t n_out = num_block_rows_A * G;

        for (uint32_t i = 0; i < n_out; i++)
            out_cts.push_back(res.context.new_ciphertext(cfg.level - 2, res.param.get_default_scale()));

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
        A_enc.par_block_col_major_pack(A, d, cfg.n_heads, d, false, res.param.get_default_scale());

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
        uint32_t n_slot = res.param.get_n() / 2;
        uint32_t G = (n_slot >= n_h_padded * d * d) ? 1 : n_h_padded / (n_slot / (d * d));
        uint32_t n_out = num_block_rows_A * G;

        for (uint32_t i = 0; i < n_out; i++)
            out_cts.push_back(res.context.new_ciphertext(cfg.level - 2, res.param.get_default_scale()));

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
    A_enc.par_block_col_major_pack(A, d, n_heads, head_dim, false, res.param.get_default_scale());

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
    uint32_t n_slot = res.param.get_n() / 2;
    uint32_t G = (n_slot >= n_h_padded * d * d) ? 1 : n_h_padded / (n_slot / (d * d));
    uint32_t n_out = K_col * num_block_rows_A * G;

    for (uint32_t i = 0; i < n_out; i++)
        out_cts.push_back(res.context.new_ciphertext(level - 2, res.param.get_default_scale()));

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
    A_enc.par_block_col_major_pack(A, d, n_heads, head_dim, false, res.param.get_default_scale());

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
    uint32_t n_slot = res.param.get_n() / 2;
    uint32_t G = (n_slot >= n_h_padded * d * d) ? 1 : n_h_padded / (n_slot / (d * d));
    uint32_t n_out = num_block_rows_A * G;  // REDUCE: single output megablock

    for (uint32_t i = 0; i < n_out; i++)
        out_cts.push_back(res.context.new_ciphertext(level - 2, res.param.get_default_scale()));

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
    Q_enc.par_block_col_major_pack(Q_mat, d, n_heads, d, false, res.param.get_default_scale());
    FeatureMatEncrypted K_enc(&res.context, init_level);
    K_enc.par_block_col_major_pack(K_mat, d, n_heads, d, false, res.param.get_default_scale());
    FeatureMatEncrypted V_enc(&res.context, init_level);
    V_enc.par_block_col_major_pack(V_mat, d, n_heads, d, false, res.param.get_default_scale());

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
        out_cts.push_back(res.context.new_ciphertext(init_level - 7, res.param.get_default_scale()));

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
    X_enc.par_block_col_major_pack(X_mat, d, n_heads, d, false, res.param.get_default_scale());

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
    uint32_t n_slot = res.param.get_n() / 2;
    uint32_t G = (n_slot >= n_h_padded * d * d) ? 1 : n_h_padded / (n_slot / (d * d));
    uint32_t n_out = num_block_rows_out * G;
    for (uint32_t i = 0; i < n_out; i++)
        out_cts.push_back(res.context.new_ciphertext(0, res.param.get_default_scale()));

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
    A_enc.par_block_col_major_pack(A_mat, d, n_heads, d, false, this->param.get_default_scale());

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
    uint32_t n_slot = this->param.get_n() / 2;
    uint32_t G = (n_slot >= n_h_padded * d * d) ? 1 : n_h_padded / (n_slot / (d * d));
    uint32_t n_out = num_block_rows_A * G;

    for (uint32_t i = 0; i < n_out; i++)
        out_cts.push_back(this->context.new_ciphertext(init_level - 4, this->param.get_default_scale()));

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
    input_feature.pack_multiplexed(input_array, false, this->context.get_parameter().get_default_scale());

    // Layer 0: p0(x)
    PolyRelu2D poly0(this->context.get_parameter(), input_shape, order0, move(weight0), skip, n_channel_per_ct,
                     init_level);
    poly0.prepare_weight_bsgs();

    // Layer 1: sign(x) ≈ p1(p0(x))
    PolyRelu2D poly1(this->context.get_parameter(), input_shape, order1, move(weight1), skip, n_channel_per_ct,
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
                    Conv1DPackedLayer conv0_layer(this->context.get_parameter(), input_shape, move(conv0_weight),
                                                  move(conv0_bias), stride, skip, n_channel_per_ct, init_level);
                    conv0_layer.prepare_weight();

                    Feature1DEncrypted output_feature(&this->context, init_level - 1, skip * stride);
                    output_feature.shape = input_shape / stride;
                    output_feature.skip = skip * stride;
                    output_feature.n_channel = n_out_channel;
                    output_feature.n_channel_per_ct = n_channel_per_ct;
                    for (int i = 0; i < div_ceil(n_out_channel, n_channel_per_ct); i++) {
                        output_feature.data.push_back(
                            this->context.new_ciphertext(init_level - 1, this->param.get_default_scale()));
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
                    input_feature.pack_multiplexed(input_array, false, this->param.get_default_scale());
                    MultiplexedConv1DPackedLayer conv0_layer(this->context.get_parameter(), input_shape,
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
                        output_feature.data.push_back(
                            this->context.new_ciphertext(output_level, this->param.get_default_scale()));
                    }

                    uint32_t n_select_pt = min(n_block_per_ct, n_out_channel);
                    vector<CkksPlaintextRingt> select_pt_subset;
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
    x0_enc.par_block_col_major_pack(x0, d, n_heads, d, false, res.param.get_default_scale());
    FeatureMatEncrypted x1_enc(&res.context, init_level);
    x1_enc.par_block_col_major_pack(x1, d, n_heads, d, false, res.param.get_default_scale());

    ParBlockColMajorAdd add_layer(res.param);
    auto expected = add_layer.run_plaintext(x0, x1);

    vector<CkksCiphertext> x0_cts, x1_cts, out_cts;
    for (auto& ct : x0_enc.data)
        x0_cts.push_back(ct.copy());
    for (auto& ct : x1_enc.data)
        x1_cts.push_back(ct.copy());
    for (size_t i = 0; i < x0_enc.data.size(); i++)
        out_cts.push_back(res.context.new_ciphertext(init_level, res.param.get_default_scale()));

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
    A_enc.par_block_col_major_pack(A, d, n_heads, d, false, res.param.get_default_scale());

    vector<CkksCiphertext> in_cts, out_cts;
    vector<CustomData> layer_data;
    for (auto& ct : A_enc.data)
        in_cts.push_back(ct.copy());
    layer_data.emplace_back(static_cast<void*>(layer_ptr.get()));
    for (size_t i = 0; i < A_enc.data.size(); i++)
        out_cts.push_back(res.context.new_ciphertext(init_level, res.param.get_default_scale()));

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
        input_feature.pack_multiplexed(input_array, false, this->param.get_default_scale());

        // Prepare select_tensor_pt via Avgpool2DLayer
        Avgpool2DLayer avgpool(input_shape, stride);
        avgpool.prepare_weight(this->param, n_channel_per_ct, n_channel, init_level, skip, input_shape);

        // Pre-allocate output (rescale consumes one level)
        uint32_t out_channels_per_ct = n_channel_per_ct * prod(stride);
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

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "interleaved_avgpool2d_layer", "", HeteroProcessors) {
    int init_level = 3;

    auto run_interleaved_avgpool_test = [&](uint32_t n_channel, const Duo& stride, const Duo& block_shape,
                                            uint32_t mult) {
        Duo input_shape = block_shape * mult;
        Duo block_expansion = input_shape / block_shape;

        Array<double, 3> input_array = gen_random_array<3>({n_channel, input_shape[0], input_shape[1]}, 1.0);

        Feature2DEncrypted input_feature(&this->context, init_level);
        Duo total_stride = block_expansion;
        input_feature.pack_interleaved(input_array, block_shape, total_stride, false, this->param.get_default_scale());

        Avgpool2DLayer avgpool(block_shape, stride);

        // No level consumed (only adds)
        uint32_t out_size = input_feature.data.size() / prod(stride);
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
        input_feature.pack_multiplexed(input_array, false, this->param.get_default_scale());

        Avgpool1DLayer avgpool(s, stride);

        Feature1DEncrypted output_feature(&this->context, init_level, skip * stride, stride);
        for (uint32_t i = 0; i < n_ct; i++) {
            output_feature.data.push_back(this->context.new_ciphertext(init_level, this->param.get_default_scale()));
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
        input_feature.pack_multiplexed(input_array, false, this->param.get_default_scale());

        Avgpool1DLayer avgpool(s, stride);
        avgpool.prepare_weight(this->param, n_channel_per_ct, n_channel, init_level, skip, s);

        uint32_t out_channels_per_ct = n_channel_per_ct * stride;
        uint32_t n_packed_out_channel = div_ceil(n_channel, out_channels_per_ct);
        Feature1DEncrypted output_feature(&this->context, init_level - 1, skip * stride);
        for (uint32_t i = 0; i < n_packed_out_channel; i++) {
            output_feature.data.push_back(
                this->context.new_ciphertext(init_level - 1, this->param.get_default_scale()));
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
        uint32_t n_channel_per_ct = div_ceil(this->n_slot, prod(input_shape));
        uint32_t n_ct = div_ceil(n_channel, n_channel_per_ct);

        Array<double, 3> input_array = gen_random_array<3>({n_channel, input_shape[0], input_shape[1]}, 1.0);
        Array<double, 1> weight = gen_random_array<1>({n_channel}, 1.0);

        Feature2DEncrypted input_feature(&this->context, init_level, skip);
        input_feature.pack_multiplexed(input_array, false, this->param.get_default_scale());

        // Prepare weight plaintexts
        MultScalarLayer mult_layer(this->param, input_shape, move(weight), skip, n_channel_per_ct, init_level);
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
                        input_feature.pack_multiplexed(input_array, false, this->param.get_default_scale());

                        MultiplexedDWConv1DPackedLayer conv_layer(this->context.get_parameter(), input_shape,
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
                            output_feature.data.push_back(
                                this->context.new_ciphertext(output_level, this->param.get_default_scale()));
                        }

                        uint32_t n_select_pt = min(n_channel_per_ct, n_channel);
                        vector<CkksPlaintextRingt> select_pt_subset;
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
        input_feature.pack_multiplexed(input_2d, false, this->param.get_default_scale());

        // Wrap as Feature0DEncrypted (run_1d_multiplexed only uses .data)
        Feature0DEncrypted input_0d(&this->context, init_level);
        input_0d.data = move(input_feature.data);
        input_0d.n_channel = cfg.n_channel;
        input_0d.n_channel_per_ct = n_valid_per_ct;
        input_0d.skip = 1;

        // Layer
        DensePackedLayer dense(this->context.get_parameter(), move(weight), move(bias), n_block_per_ct, init_level, 0);
        dense.prepare_weight_for_1d_multiplexed(cfg.shape, cfg.skip, cfg.invalid_fill);

        Feature0DEncrypted output(&this->context, init_level - 1);
        output.skip = block_size;
        output.n_channel = cfg.n_out;
        output.n_channel_per_ct = n_block_per_ct;
        for (uint32_t i = 0; i < n_packed_out; i++) {
            output.data.push_back(this->context.new_ciphertext(init_level - 1, this->param.get_default_scale()));
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
        input_feature.pack_interleaved(input_array, block_shape, stride, false, this->param.get_default_scale());

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

TEMPLATE_LIST_TEST_CASE_METHOD(HeteroFixture, "block_col_major_layernorm", "", HeteroProcessors) {
    // LayerNorm needs higher levels; use local context with N=32768
    const int N = 32768;
    CkksParameter param = CkksParameter::create_parameter(N);
    CkksContext context = CkksContext::create_random_context(param);
    context.gen_rotation_keys();
    double default_scale = param.get_default_scale();

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
        X_enc.block_col_major_pack(X_mat, d, false, default_scale);

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
    CkksParameter param = CkksParameter::create_parameter(N);
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
    X_enc.par_block_col_major_pack(X_mat, d, n_heads, d, false, param.get_default_scale());

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
    uint32_t n_slot = param.get_n() / 2;
    uint32_t G = (n_slot >= n_h_padded * d * d) ? 1 : n_h_padded / (n_slot / (d * d));
    uint32_t n_out = num_block_rows * num_block_cols * G;
    for (uint32_t i = 0; i < n_out; i++)
        out_cts.push_back(context.new_ciphertext(0, param.get_default_scale()));

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
        X_enc.block_col_major_pack(X_mat, d, false, this->default_scale);

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
        X_enc.par_block_col_major_pack(X_mat, d, n_heads, d, false, res.param.get_default_scale());

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
        uint32_t n_slot = res.param.get_n() / 2;
        uint32_t G = (n_slot >= n_h_padded * d * d) ? 1 : n_h_padded / (n_slot / (d * d));
        uint32_t n_out = num_block_rows * num_block_cols * G;
        int out_level = init_level - 1 - (degree == 4 ? 3 : 2);
        for (uint32_t i = 0; i < n_out; i++)
            out_cts.push_back(res.context.new_ciphertext(out_level, res.param.get_default_scale()));

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
        A_enc.par_block_col_major_pack(A_mat, d, n_heads, d, false, res.param.get_default_scale());

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
        uint32_t n_slot = res.param.get_n() / 2;
        uint32_t G = (n_slot >= n_h_padded * d * d) ? 1 : n_h_padded / (n_slot / (d * d));
        uint32_t n_out = num_block_rows_A * G;
        int out_level = reduce_level - 2;
        for (uint32_t i = 0; i < n_out; i++)
            out_cts.push_back(res.context.new_ciphertext(out_level, res.param.get_default_scale()));

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
    A_enc.par_block_col_major_pack(A, d, n_heads, d, false, res.param.get_default_scale());

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
    uint32_t n_slot = res.param.get_n() / 2;
    uint32_t G = (n_slot >= n_h_padded * d * d) ? 1 : n_h_padded / (n_slot / (d * d));
    uint32_t n_out = num_block_rows_A * G;

    for (uint32_t i = 0; i < n_out; i++)
        out_cts.push_back(res.context.new_ciphertext(level - 2, res.param.get_default_scale()));

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
    A_enc.par_block_col_major_pack(A, d, n_heads, K, false, res.param.get_default_scale());

    static vector<CkksCiphertext> in_cts, out_cts;
    static vector<CustomData> layer_data;
    in_cts.clear();
    out_cts.clear();
    layer_data.clear();

    for (auto& ct : A_enc.data)
        in_cts.push_back(ct.copy());
    layer_data.emplace_back(static_cast<void*>(layer_ptr.get()));

    for (uint32_t i = 0; i < in_cts.size(); i++)
        out_cts.push_back(res.context.new_ciphertext(level - 1, res.param.get_default_scale()));

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
    A_enc.par_block_col_major_pack(A, d, n_heads, N_dim, false, res.param.get_default_scale());
    FeatureMatEncrypted B_enc(&res.context, level);
    B_enc.par_block_col_major_pack(B, d, n_heads, P, false, res.param.get_default_scale());

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
    uint32_t n_slot = res.param.get_n() / 2;
    uint32_t G = (n_slot >= n_h_padded * d * d) ? 1 : n_h_padded / (n_slot / (d * d));
    n_out *= G;

    for (uint32_t i = 0; i < n_out; i++)
        out_cts.push_back(res.context.new_ciphertext(level - 3, res.param.get_default_scale()));

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
        X_enc.par_block_col_major_pack(X, d, n_heads, d, false, res.param.get_default_scale());

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
        uint32_t n_slot = res.param.get_n() / 2;
        uint32_t G = (n_slot >= n_h_padded * d * d) ? 1 : n_h_padded / (n_slot / (d * d));
        uint32_t n_out = num_block_rows * num_block_cols * G;
        int out_level = level - (degree == 4 ? 3 : 2);

        for (uint32_t i = 0; i < n_out; i++)
            out_cts.push_back(res.context.new_ciphertext(out_level, res.param.get_default_scale()));

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
