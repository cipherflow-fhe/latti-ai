#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "fhe_mpc.h"
#include "SCI/src/globals.h"
#include "inference_task/inference_process.h"
#include "inference_task/mpc_task_meta_data.h"
#include "ut_util.h"

#include <iomanip>
#include <signal.h>
#include <stdexcept>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace std;
using namespace sci;
using namespace lattisense;

int party = SERVER;
int port = 12309;
string address = "127.0.0.1";
int num_threads = 1;
bool auto_start_client_task = true;

vector<uint32_t> save_vec(uint32_t is_2d, MpcLayerType type, uint32_t pool_type = 0, uint32_t pack = 0) {
    vector<uint32_t> info;

    info.push_back(is_2d);
    info.push_back((uint32_t)type);
    info.push_back(pool_type);
    info.push_back(pack);
    return info;
}

Array1D read_plain_face(string file_path, string name, double factor, uint32_t skip) {
    Array1D tensor;
    ifstream filestream(file_path);
    if (!filestream.is_open()) {
        printf("%s, open file failed. please check.", file_path.c_str());
        exit(0);
    }
    string line;
    string cell;
    while (getline(filestream, line)) {
        stringstream line_stream(line);
        string c_name = "0";
        while (getline(line_stream, cell, ',')) {
            if (cell.c_str() == name) {
                c_name = name;
                continue;
            }
            if (c_name == name) {
                double x;
                sscanf(cell.c_str(), "%lf", &x);
                tensor.push_back(x * factor);
                for (int i = 0; i < skip - 1; i++) {
                    tensor.push_back(0);
                }
            }
        }
    }
    return tensor;
}

double compute_distance(vector<double> x, vector<double> y) {
    double sum = 0;
    for (int i = 0; i < x.size(); i++) {
        sum += pow((x[i] - y[i]), 2);
    }
    sum = pow(sum, 0.5);
    return sum;
}

void print_double_message(const double* data, const string& name, int size) {
    cout << name;
    for (int i = 0; i < size; i++) {
        cout << data[i] << ",";
    }
    cout << endl;
}

void write_file_common(const string& file_path, const Array1D& data) {
    ofstream file(file_path);
    for (double value : data) {
        file << value << endl;
    }
}

ArrayComparison compare(const Array1D& expected, const Array1D& output) {
    return compare(Array<double, 1>::from_array_1d(expected), Array<double, 1>::from_array_1d(output));
}

class ClientTaskProcess {
public:
    ClientTaskProcess() {
        if (!auto_start_client_task) {
            return;
        }

        pid = fork();
        if (pid < 0) {
            throw runtime_error("failed to fork client_task");
        }
        if (pid == 0) {
            prctl(PR_SET_PDEATHSIG, SIGTERM);
            if (getppid() == 1) {
                _exit(1);
            }
            execl("../inference_task/client_task", "client_task", (char*)nullptr);
            _exit(1);
        }
    }

    ~ClientTaskProcess() {
        if (pid <= 0) {
            return;
        }
        int status;
        if (waitpid(pid, &status, WNOHANG) == 0) {
            kill(pid, SIGTERM);
            waitpid(pid, &status, 0);
        }
    }

private:
    pid_t pid = -1;
};

class MpcFixture {
public:
    MpcFixture() : parameter{CkksParameter::create_parameter(8192)} {
        srand(time(NULL));
        party = SERVER;
        port = 12309;
        address = "127.0.0.1";
        num_threads = 1;
        bitlength = RING_MOD_BIT;
        StartComputation();
        data_trans.io_in = io;
        context = data_trans.recv_public_context();
    }

protected:
    ClientTaskProcess client_task;
    CkksParameter parameter;
    DataTransmission data_trans;
    CkksContext context;

    int slot_size = 4096;
    int scale_ord = DEFAULT_SCALE_BIT;
    double pt_range = 128.0;
    uint64_t ring_mod = RING_MOD;
};

TEST_CASE_METHOD(MpcFixture, "multi_channel_mpc_refresh_test") {
    int level = 5;
    uint32_t n_channel = 4;
    vector<uint32_t> shapes = {16};
    Duo skip = {1, 1};
    for (uint32_t s : shapes) {
        SECTION("shape=(" + to_string(s) + ',' + to_string(s) + ')') {
            Duo shape = {s, s};
            MpcTaskMetaData meta_data;
            meta_data.append(MpcProtoType::enc_to_share_for_multi_channel_pack, {"u8", "u8", "u8"}, (uint8_t)level, 0,
                             0);
            meta_data.append(MpcProtoType::share_to_enc_for_multi_channel_pack, {"u8", "u32", "duo"}, (uint8_t)level,
                             n_channel, skip);
            meta_data.append(MpcProtoType::end, {});
            Bytes meta_data_bytes = meta_data.serialize();
            data_trans.send_bytes(meta_data_bytes);

            Array<double, 3> x_mg = gen_random_array<3>({n_channel, shape[0], shape[1]}, 1.0);
            x_mg.set(0, 0, 0, -2);
            cout << "x_mg=" << x_mg.get_data()[0] << endl;

            Feature2DEncrypted x_e(&context, level, skip);
            x_e.pack_multiplexed(x_mg, false, DEFAULT_SCALE);

            auto x_share0 = server_enc_to_share_multi_pack(context, x_e, scale_ord, ring_mod,PackType::MultiplexedPacking);

            Feature2DShare y_share0(ring_mod, scale_ord);
            y_share0.data = x_share0.data.copy();

            auto y_ct = server_share_to_enc_multi_pack(context, y_share0, scale_ord, ring_mod, level, PackType::MultiplexedPacking);
            Array<double, 3> y_mg = y_ct.unpack_multiplexed();
            for (int i = 0; i < y_mg.get_size(); i++) {
                y_mg.set(i, y_mg.get_data()[i] / T_SCALE);
            }

            auto compare_res = compare(x_mg, y_mg);
            REQUIRE(compare_res.max_error < 5.0e-2 * compare_res.max_abs);
        }
    }
}

TEST_CASE_METHOD(MpcFixture, "big_size_e2s_s2e_no_conv_test") {
    int init_level = 5;
    uint32_t n_channel = 16;

    Duo input_shape = {256, 256};
    Duo block_shape = {64, 64};
    Duo block_expansion = {(uint32_t)ceil(input_shape[0] / (double)block_shape[0]),
                           (uint32_t)ceil(input_shape[1] / (double)block_shape[1])};

    auto x_mg = gen_random_array<3>({n_channel, input_shape[0], input_shape[1]}, 0.1);

    Feature2DEncrypted x_e(&context, init_level, {1, 1});
    x_e.pack_interleaved(x_mg, block_shape, block_expansion, false,
                         context.get_parameter().get_default_scale());

    MpcTaskMetaData meta_data;
    meta_data.append(MpcProtoType::enc_to_share_for_multi_channel_pack, {"u8", "u8", "u8", "u8"}, (uint8_t)x_e.level, 0,
                     0, (uint8_t)PackType::InterleavedPacking);
    meta_data.append(MpcProtoType::share_to_enc_for_multi_channel_pack, {"u8", "u32", "duo", "u8"}, (uint8_t)x_e.level,
                     x_e.n_channel, x_e.skip, (uint8_t)PackType::InterleavedPacking);
    meta_data.append(MpcProtoType::end, {});
    Bytes meta_data_bytes = meta_data.serialize();
    data_trans.send_bytes(meta_data_bytes);

    auto x_share0 = server_enc_to_share_multi_pack(context, x_e, scale_ord, ring_mod, PackType::InterleavedPacking);

    Feature2DShare y_share0(ring_mod, scale_ord);
    y_share0.data = x_share0.data.copy();

    auto y_ct = server_share_to_enc_multi_pack(context, y_share0, scale_ord, ring_mod, x_e.level,
                                               PackType::InterleavedPacking);
    Array<double, 3> y_mg = y_ct.unpack_interleaved(block_shape, block_expansion);
    for (int i = 0; i < y_mg.get_size(); i++) {
        y_mg.set(i, y_mg.get_data()[i] / T_SCALE);
    }

    auto compare_res = compare(x_mg, y_mg);
    REQUIRE(compare_res.max_error < 5.0e-2 * compare_res.max_abs);
}

TEST_CASE_METHOD(MpcFixture,"big_size_conv_test") {
    // CkksParameter parameter = CkksParameter::create_parameter(8192);
    // CkksContext context = CkksContext::create_random_context(parameter);
    // context.gen_rotation_keys();

    int init_level = 5;
    uint32_t n_in_channel = 16;
    uint32_t n_out_channel = 32;

    Duo input_shape = {256, 256};
    Duo kernel_shape = {3, 3};
    Duo stride = {2, 2};
    Duo block_shape = {64, 64};

    Duo block_expansion = {(uint32_t)ceil(input_shape[0] / (double)block_shape[0]),
                           (uint32_t)ceil(input_shape[1] / (double)block_shape[1])};
    Duo next_stride = {(uint32_t)ceil(block_expansion[0] / (double)stride[0]),
                       (uint32_t)ceil(block_expansion[1] / (double)stride[1])};

    auto input_array_vec = gen_random_array<3>({n_in_channel, input_shape[0], input_shape[1]}, 0.1);
    auto conv0_weight = gen_random_array<4>({n_out_channel, n_in_channel, kernel_shape[0], kernel_shape[1]}, 0.1);
    auto conv0_bias = gen_random_array<1>({n_out_channel}, 1);

    Feature2DEncrypted f2d(&context, init_level, {1, 1});
    f2d.pack_interleaved(input_array_vec, block_shape, block_expansion, false,
                         context.get_parameter().get_default_scale());

    Array<int, 1> padding({2});
    padding.set(0, -1);
    padding.set(1, -1);
    InverseMultiplexedConv2DLayer conv(context.get_parameter(), input_shape, conv0_weight.copy(), conv0_bias.copy(),
                                       padding, stride, block_shape, init_level);
    conv.prepare_weight();

    auto y_ct = conv.run(context, f2d);
    Array<double, 3> y_mg = y_ct.unpack_interleaved(block_shape, next_stride);
    auto y_expected = conv.run_plaintext(input_array_vec);

    auto compare_res = compare(y_expected, y_mg);
    REQUIRE(compare_res.max_error < 5.0e-2 * compare_res.max_abs);
}

TEST_CASE_METHOD(MpcFixture, "big_size_e2s_s2e_test") {
    int init_level = 5;

    uint32_t n_in_channel = 16;
    uint32_t n_out_channel = 32;

    Duo skip = {1, 1};

    Duo input_shape = {256, 256};
    Duo kernel_shape = {3, 3};
    Duo stride = {2, 2};

    Duo block_shape = {64, 64};
    uint32_t n_channel_per_ct = div_ceil(context.get_parameter().get_n() / 2, (input_shape[0] * input_shape[1]));

    Duo block_expansion = {(uint32_t)ceil(input_shape[0] / (double)block_shape[0]),
                           (uint32_t)ceil(input_shape[1] / (double)block_shape[1])};
    Duo next_stride = {(uint32_t)ceil(block_expansion[0] / (double)stride[0]),
                       (uint32_t)ceil(block_expansion[1] / (double)stride[1])};

    auto input_array_vec = gen_random_array<3>({n_in_channel, input_shape[0], input_shape[1]}, 0.1);
    auto conv0_weight = gen_random_array<4>({n_out_channel, n_in_channel, kernel_shape[0], kernel_shape[1]}, 0.1);
    auto conv0_bias = gen_random_array<1>({n_out_channel}, 1);

    Feature2DEncrypted f2d(&context, init_level, {1, 1});
    f2d.pack_interleaved(input_array_vec, block_shape, block_expansion, false,
                               context.get_parameter().get_default_scale());
    cout << "run-------" << endl;
    Array<int, 1> padding({2});
    padding.set(0, -1);
    padding.set(1, -1);
    InverseMultiplexedConv2DLayer conv(context.get_parameter(), input_shape, conv0_weight.copy(), conv0_bias.copy(),
                                       padding, stride, block_shape, init_level);
    conv.prepare_weight();

    auto x_e = conv.run(context, f2d);

    MpcTaskMetaData meta_data;
    meta_data.append(MpcProtoType::enc_to_share_for_multi_channel_pack, {"u8", "u8", "u8", "u8"}, (uint8_t)x_e.level, 0,
                     0, (uint8_t)PackType::InterleavedPacking);
    meta_data.append(MpcProtoType::share_to_enc_for_multi_channel_pack, {"u8", "u32", "duo", "u8"}, (uint8_t)x_e.level,
                     x_e.n_channel, x_e.skip, (uint8_t)PackType::InterleavedPacking);
    meta_data.append(MpcProtoType::end, {});
    Bytes meta_data_bytes = meta_data.serialize();
    data_trans.send_bytes(meta_data_bytes);
    auto x_share0 =
        server_enc_to_share_multi_pack(context, x_e, scale_ord, ring_mod, PackType::InterleavedPacking);

    Feature2DShare y_share0(ring_mod, scale_ord);
    y_share0.data = x_share0.data.copy();

    auto y_ct = server_share_to_enc_multi_pack(context, y_share0, scale_ord, ring_mod, init_level,
                                               PackType::InterleavedPacking);
    Array<double, 3> y_mg = y_ct.unpack_interleaved(block_shape, next_stride);
    for (int i = 0; i < y_mg.get_size(); i++) {
        y_mg.set(i, y_mg.get_data()[i] / T_SCALE);
    }
    auto x_mg = conv.run_plaintext(input_array_vec);
    for (int i = 0; i < 20; i++) {
        cout << "y_mg=" << y_mg.get_data()[i] << "x_mg=" << x_mg.get_data()[i] << endl;
    }

    auto compare_res = compare(x_mg, y_mg);
    REQUIRE(compare_res.max_error < 5.0e-2 * compare_res.max_abs);
    // }
}

TEST_CASE_METHOD(MpcFixture, "Feature2DEncrypted to shares and back") {
    int level = 3;
    uint32_t n_channel = 4;
    vector<uint32_t> shapes = {16, 32};

    for (uint32_t s : shapes) {
        SECTION("shape=(" + to_string(s) + ',' + to_string(s) + ')') {
            Duo shape = {s, s};

            MpcTaskMetaData meta_data;
            meta_data.append(MpcProtoType::enc_to_share, {"u8", "u8", "u8"}, (uint8_t)level, 0, 0);
            meta_data.append(MpcProtoType::share_to_enc, {"u8", "u32"}, (uint8_t)level, n_channel);
            meta_data.append(MpcProtoType::end, {});
            Bytes meta_data_bytes = meta_data.serialize();
            data_trans.send_bytes(meta_data_bytes);

            Array<double, 3> x_mg = gen_random_array<3>({n_channel, shape[0], shape[1]}, 1.0);
            Feature2DEncrypted x_e(&context, level);
            x_e.pack_multiple_channel(x_mg);

            Feature2DEncrypted x_share1_enc(&context, x_e.level);
            Feature2DShare x_share0(ring_mod, scale_ord);
            x_e.split_to_shares(&x_share1_enc, &x_share0);
            Bytes x_share1_enc_bytes = x_share1_enc.serialize();
            data_trans.send_bytes(x_share1_enc_bytes);

            for (int i = 0; i < x_share0.data.get_size(); i++) {
                uint64_t temp = (x_share0.data.get(i) * T_SCALE) % RING_MOD;
                x_share0.data.set(i, temp);
            }
            MPC mpc(scale_ord, ring_mod, pt_range);
            Bytes b1 = mpc.wrap_protocol(x_share0.data.to_array_1d(), data_trans.io_in, otpack, party);
            Bytes y_share1_bytes = data_trans.receive_bytes();
            Feature2DEncrypted y_share1_enc(&context, x_e.level);
            y_share1_enc.deserialize(y_share1_bytes);
            y_share1_enc.decompress();
            Bytes y_share2_bytes = data_trans.receive_bytes();
            Feature2DEncrypted y_share2_enc(&context, x_e.level);
            y_share2_enc.deserialize(y_share2_bytes);
            Feature2DEncrypted y_ct = y_share1_enc.combine_with_share_new_protocol(x_share0, y_share2_enc, b1);

            Array<double, 3> y_mg = y_ct.unpack_multiple_channel();
            for (int i = 0; i < y_mg.get_size(); i++) {
                y_mg.set(i, y_mg.get_data()[i] / T_SCALE);
            }
            auto compare_res = compare(x_mg, y_mg);
            REQUIRE(compare_res.max_error < 5.0e-2 * compare_res.max_abs);
        }
    }
}

TEST_CASE_METHOD(MpcFixture, "Feature0DEncrypted to shares relu0d and back") {
    int level = 1;
    vector<uint32_t> n_channels = {4096};
    for (uint32_t n_channel : n_channels) {
        SECTION("n_channel=(" + to_string(n_channel) + ')') {
            MpcTaskMetaData meta_data;
            meta_data.append(MpcProtoType::enc_to_share_0d, {"u8", "u32", "u8", "u8"}, (uint8_t)level, n_channel, 0, 0);
            meta_data.append(MpcProtoType::relu_0d, {});
            meta_data.append(MpcProtoType::share_to_enc_0d, {"u8", "u32"}, (uint8_t)level, n_channel);
            meta_data.append(MpcProtoType::end, {});
            Bytes meta_data_bytes = meta_data.serialize();
            data_trans.send_bytes(meta_data_bytes);

            Array<double, 1> x_mg = gen_random_array<1>({n_channel}, 1.0);
            x_mg.set(0, 1.2);
            Feature0DEncrypted x_e(&context, level);
            x_e.pack(x_mg);
            x_e.skip = 1;
            x_e.n_channel = n_channel;

            Feature0DEncrypted x_share1_enc(&context, x_e.level);
            Feature0DShare x_share0(ring_mod, scale_ord);
            x_e.split_to_shares(&x_share1_enc, &x_share0);

            vector<uint8_t> x_share1_enc_bytes = x_share1_enc.serialize();
            data_trans.send_bytes(x_share1_enc_bytes);
            cout << "byte=" << x_e.data[0].serialize(parameter).size() << endl;
            MPC mpc(scale_ord, ring_mod, pt_range);
            data_trans.io_in->flush();

            // relu
            ReluLayerServer act(scale_ord, ring_mod, pt_range);
            Feature0DShare y_share0(ring_mod, scale_ord);
            act.run(x_share0, y_share0);

            for (int i = 0; i < y_share0.data.get_size(); i++) {
                uint64_t temp = (y_share0.data.get(i) * T_SCALE) % RING_MOD;
                y_share0.data.set(i, temp);
            }

            auto b1 = mpc.wrap_protocol(y_share0.data.to_array_1d(), data_trans.io_in, otpack, party);
            data_trans.io_in->flush();

            Bytes y_share1_bytes = data_trans.receive_bytes();
            Feature0DEncrypted y_share1_enc(&context, x_e.level);
            y_share1_enc.deserialize(y_share1_bytes);
            y_share1_enc.decompress();

            Bytes y_share2_bytes = data_trans.receive_bytes();
            cout << " y byte=" << x_share1_enc_bytes.size() << endl;
            Feature0DEncrypted y_share2_enc(&context, x_e.level);
            y_share2_enc.deserialize(y_share2_bytes);
            auto y_ct = y_share1_enc.combine_with_share_new_protocol(y_share0, y_share2_enc, b1);

            Array<double, 1> y_mg = y_ct.unpack();

            for (int i = 0; i < y_mg.get_size(); i++) {
                y_mg.set(i, y_mg.get_data()[i] / T_SCALE);
            }
            print_double_message(y_mg.get_data(), "ct_res=", 10);
            print_double_message(x_mg.get_data(), "real_res=", 10);
            Array<double, 1> y_mg_expected = act.run_relu_plaintext(x_mg);
            auto compare_res = compare(y_mg_expected, y_mg);
            fprintf(stderr, "max_erro=%f\n", compare_res.max_error);
            REQUIRE(compare_res.max_error < 5.0e-2 * compare_res.max_abs);
        }
    }
}

TEST_CASE_METHOD(MpcFixture, "feature 2d relu: change shape and skip") {
    srand(time(NULL));
    int level = 1;
    uint32_t n_channel = 4;
    vector<uint32_t> shapes = {4, 32};
    vector<uint32_t> skips = {1};
    for (uint32_t s : shapes) {
        Duo shape = {s, s};
        SECTION("shape=" + str(shape)) {
            for (uint32_t sk : skips) {
                Duo skip = {sk, sk};
                SECTION("skip=" + str(skip)) {
                    MpcTaskMetaData meta_data;
                    meta_data.append(MpcProtoType::enc_to_share, {"u8", "u8", "u8"}, (uint8_t)level, 0, 0);
                    meta_data.append(MpcProtoType::relu, {});
                    meta_data.append(MpcProtoType::share_to_enc, {"u8", "u32"}, (uint8_t)level, n_channel);
                    meta_data.append(MpcProtoType::end, {});
                    Bytes meta_data_bytes = meta_data.serialize();
                    data_trans.send_bytes(meta_data_bytes);

                    // encrypt
                    Array<double, 3> x_mg = gen_random_array<3>({n_channel, shape[0], shape[1]}, 1.0);
                    x_mg.get_data()[0] = 0.1;
                    vector<double> temp_vec(4096, 0.1);
                    Feature2DEncrypted x_e(&context, level);
                    x_e.pack_multiple_channel(x_mg, false, DEFAULT_SCALE);

                    x_e.skip = skip;
                    x_e.shape = {x_e.shape[0] / skip[0], x_e.shape[1] / skip[1]};

                    // enc_to_share
                    Feature2DEncrypted x_share1_enc(&context, x_e.level);
                    Feature2DShare x_share0(ring_mod, scale_ord);
                    x_e.split_to_shares(&x_share1_enc, &x_share0);
                    vector<uint8_t> x_share1_enc_bytes = x_share1_enc.serialize();
                    data_trans.send_bytes(x_share1_enc_bytes);

                    // relu
                    ReluLayerServer act(scale_ord, ring_mod, pt_range);
                    Feature2DShare y_share0(ring_mod, scale_ord);
                    act.run(x_share0, y_share0);

                    
                    for (int i = 0; i < y_share0.data.get_size(); i++) {
                        uint64_t temp = (y_share0.data.get(i) * T_SCALE) % RING_MOD;
                        y_share0.data.set(i, temp);
                    }

                    MPC mpc(scale_ord, ring_mod, pt_range);
                    auto b1 = mpc.wrap_protocol(y_share0.data.to_array_1d(), data_trans.io_in, otpack, party);

                    Bytes y_share1_bytes = data_trans.receive_bytes();
                    Feature2DEncrypted y_share1_enc(&context, x_e.level);
                    y_share1_enc.deserialize(y_share1_bytes);
                    y_share1_enc.decompress();

                    Bytes y_share2_bytes = data_trans.receive_bytes();
                    Feature2DEncrypted y_share2_enc(&context, x_e.level);
                    y_share2_enc.deserialize(y_share2_bytes);

                    auto y_ct = y_share1_enc.combine_with_share_new_protocol(y_share0, y_share2_enc, b1);
                    Array<double, 3> y_mg = y_ct.unpack_multiple_channel();

                    // compare
                    Array<double, 3> x_mg_skip = x_mg.apply_skip(skip);
                    Array<double, 3> y_mg_expected = act.run_relu_plaintext(x_mg_skip);
                    print_double_message(y_mg_expected.get_data(), "expected", 10);
                    for (int i = 0; i < y_mg.get_size(); i++) {
                        y_mg.set(i, y_mg.get_data()[i] / T_SCALE);
                    }
                    print_double_message(y_mg.get_data(), "y_mg", 10);
                    auto compare_res = compare(y_mg_expected, y_mg);
                    cout << "max_erro=" << compare_res.max_error << endl;
                    cout << "rms_erro=" << compare_res.rmse << endl;
                    cout << "default scale=" << context.get_parameter().get_default_scale() << endl;
                    REQUIRE(compare_res.max_error < 5.0e-2 * compare_res.max_abs);
                }
            }
        }
    }
}

TEST_CASE_METHOD(MpcFixture, "feature 2d relu: change n_channel") {
    int level = 3;
    Duo shape = {32, 32};
    vector<uint32_t> ncs = {1, 3, 4, 5, 33};
    Duo skip = {1, 1};

    for (uint32_t n_channel : ncs) {
        SECTION("n_channel=" + to_string(n_channel)) {
            srand(0);
            MpcTaskMetaData meta_data;
            meta_data.append(MpcProtoType::enc_to_share, {"u8", "u8", "u8"}, (uint8_t)level, 0, 0);
            meta_data.append(MpcProtoType::relu, {});
            meta_data.append(MpcProtoType::share_to_enc, {"u8", "u32"}, (uint8_t)level, n_channel);
            meta_data.append(MpcProtoType::end, {});
            Bytes meta_data_bytes = meta_data.serialize();
            data_trans.send_bytes(meta_data_bytes);

            // encrypt
            Array<double, 3> x_mg = gen_random_array<3>({n_channel, shape[0], shape[1]}, 1.0);
            Feature2DEncrypted x_e(&context, level);
            x_e.pack_multiple_channel(x_mg);
            
            x_e.skip = skip;
            x_e.shape = {x_e.shape[0] / skip[0], x_e.shape[1] / skip[1]};

            // enc_to_share
            Feature2DEncrypted x_share1_enc(&context, x_e.level);
            Feature2DShare x_share0(ring_mod, scale_ord);
            x_e.split_to_shares(&x_share1_enc, &x_share0);
            vector<uint8_t> x_share1_enc_bytes = x_share1_enc.serialize();
            data_trans.send_bytes(x_share1_enc_bytes);

            // relu
            ReluLayerServer act(scale_ord, ring_mod, pt_range);
            Feature2DShare y_share0(ring_mod, scale_ord);
            act.run(x_share0, y_share0);

          
            for (int i = 0; i < y_share0.data.get_size(); i++) {
                y_share0.data[i] = (y_share0.data[i] * T_SCALE) % RING_MOD;
            }
            MPC mpc(scale_ord, ring_mod, pt_range);
            Bytes b1 = mpc.wrap_protocol(y_share0.data.to_array_1d(), data_trans.io_in, otpack, party);
            Bytes y_share1_bytes = data_trans.receive_bytes();
            Feature2DEncrypted y_share1_enc(&context, x_e.level);
            y_share1_enc.deserialize(y_share1_bytes);
            y_share1_enc.decompress();
            Bytes y_share2_bytes = data_trans.receive_bytes();
            Feature2DEncrypted y_share2_enc(&context, x_e.level);
            y_share2_enc.deserialize(y_share2_bytes);
            Feature2DEncrypted y_ct = y_share1_enc.combine_with_share_new_protocol(y_share0, y_share2_enc, b1);

            // decrypt
            Array<double, 3> y_mg = y_ct.unpack_multiple_channel();

            // compare
            Array<double, 3> x_mg_skip = x_mg.apply_skip(skip);
            Array<double, 3> y_mg_expected = act.run_relu_plaintext(x_mg_skip);

            for (int i = 0; i < y_mg.get_size(); i++) {
                y_mg[i] = y_mg[i] / T_SCALE;
            }

            auto compare_res = compare(y_mg_expected, y_mg);
            cout << "max_erro=" << compare_res.max_error << endl;
            REQUIRE(compare_res.max_error < 5.0e-2 * compare_res.max_abs);
        }
    }
}

TEST_CASE_METHOD(MpcFixture, "feature 2d maxpool") {
    srand(1);
    int level = 4;
    uint32_t pool_type = MAXPOOL;
    uint32_t n_channel = 4;
    Duo pool_stride = {2, 2};

    vector<uint32_t> shapes = {32, 64};
    vector<uint32_t> kernel_shapes = {2, 3};
    for (uint32_t s : shapes) {
        Duo shape = {s, s};
        SECTION("shape=" + str(shape)) {
            for (uint32_t ks : kernel_shapes) {
                Duo kernel_shape = {ks, ks};
                SECTION("kernel_shape=" + str(kernel_shape)) {
                    Duo pad = {0, 0};
                    MpcTaskMetaData meta_data;
                    meta_data.append(MpcProtoType::enc_to_share, {"u8", "u8", "u8"}, (uint8_t)level, 0, 0);
                    meta_data.append(MpcProtoType::max_pool, {"duo", "duo"}, kernel_shape, pool_stride);
                    meta_data.append(MpcProtoType::share_to_enc, {"u8", "u32"}, (uint8_t)level, n_channel);
                    meta_data.append(MpcProtoType::end, {});
                    Bytes meta_data_bytes = meta_data.serialize();
                    data_trans.send_bytes(meta_data_bytes);

                    auto x_mg = gen_random_array<3>({n_channel, shape[0], shape[1]}, 1.0);
                    Feature2DEncrypted x_e(&context, level);
                    x_e.pack_multiple_channel(x_mg);

                    Feature2DEncrypted x_share1_enc(&context, x_e.level);
                    Feature2DShare x_share0(ring_mod, scale_ord);
                    x_e.split_to_shares(&x_share1_enc, &x_share0);
                    Bytes x_share1_enc_bytes = x_share1_enc.serialize();
                    data_trans.send_bytes(x_share1_enc_bytes);

                    PoolLayerServer pool(kernel_shape, pool_stride, scale_ord, ring_mod, pool_type, pt_range);
                    Feature2DShare y_share0(ring_mod, scale_ord);
                    pool.run(x_e, x_share0, y_share0);

                    for (int i = 0; i < y_share0.data.get_size(); i++) {
                        uint64_t temp = (y_share0.data.get(i) * T_SCALE) % RING_MOD;
                        y_share0.data.set(i, temp);
                    }
                    MPC mpc(scale_ord, ring_mod, pt_range);
                    auto b1 = mpc.wrap_protocol(y_share0.data.to_array_1d(), data_trans.io_in, otpack, party);

                    Bytes y_share1_bytes = data_trans.receive_bytes();
                    Feature2DEncrypted y_share1_enc(&context, x_e.level);
                    y_share1_enc.deserialize(y_share1_bytes);
                    y_share1_enc.decompress();

                    Bytes y_share2_bytes = data_trans.receive_bytes();
                    Feature2DEncrypted y_share2_enc(&context, x_e.level);
                    y_share2_enc.deserialize(y_share2_bytes);

                    auto y_share1_dec = y_share1_enc.unpack_multiple_channel();
                    auto y_share2_dec = y_share2_enc.unpack_multiple_channel();

                    auto y_ct = y_share1_enc.combine_with_share_new_protocol(y_share0, y_share2_enc, b1);

                    Array<double, 3> y_mg = y_ct.unpack_multiple_channel();

                    for (int i = 0; i < y_mg.get_size(); i++) {
                        y_mg.set(i, y_mg.get_data()[i] / T_SCALE);
                    }

                    Array<double, 3> y_mg_expected = pool.run_maxpool_plaintext(x_mg);
                    print_double_message(y_mg_expected.get_data(), "expected", 20);
                    print_double_message(y_mg.get_data(), "y_mg", 20);
                    auto compare_res = compare(y_mg_expected, y_mg);
                    printf("max_erro=%f", compare_res.max_error);
                    REQUIRE(compare_res.max_error < 5.0e-2 * compare_res.max_abs);
                }
            }
        }
    }
}

TEST_CASE_METHOD(MpcFixture, "feature 2d avgpool") {
    srand(1);
    int level = 4;
    uint32_t pool_type = AVGPOOL;
    uint32_t n_channel = 4;
    Duo pool_stride = {2, 2};

    vector<uint32_t> shapes = {32, 64};
    vector<uint32_t> kernel_shapes = {2, 3};
    for (uint32_t s : shapes) {
        Duo shape = {s, s};
        SECTION("shape=" + str(shape)) {
            for (uint32_t ks : kernel_shapes) {
                Duo kernel_shape = {ks, ks};
                SECTION("kernel_shape=" + str(kernel_shape)) {
                    Duo pad = {0, 0};
                    MpcTaskMetaData meta_data;
                    meta_data.append(MpcProtoType::enc_to_share, {"u8", "u8", "u8"}, (uint8_t)level, 0, 0);
                    meta_data.append(MpcProtoType::avg_pool, {"duo", "duo"}, kernel_shape, pool_stride);
                    meta_data.append(MpcProtoType::share_to_enc, {"u8", "u32"}, (uint8_t)level, n_channel);
                    meta_data.append(MpcProtoType::end, {});
                    Bytes meta_data_bytes = meta_data.serialize();
                    data_trans.send_bytes(meta_data_bytes);

                    auto x_mg = gen_random_array<3>({n_channel, shape[0], shape[1]}, 1.0);
                    Feature2DEncrypted x_e(&context, level);
                    x_e.pack_multiple_channel(x_mg);

                    Feature2DEncrypted x_share1_enc(&context, x_e.level);
                    Feature2DShare x_share0(ring_mod, scale_ord);
                    x_e.split_to_shares(&x_share1_enc, &x_share0);
                    Bytes x_share1_enc_bytes = x_share1_enc.serialize();
                    data_trans.send_bytes(x_share1_enc_bytes);

                    PoolLayerServer pool(kernel_shape, pool_stride, scale_ord, ring_mod, pool_type, pt_range);

                    Feature2DShare y_share0(ring_mod, scale_ord);
                    pool.run(x_e, x_share0, y_share0);

                    for (int i = 0; i < y_share0.data.get_size(); i++) {
                        uint64_t temp = (y_share0.data.get(i) * T_SCALE) % RING_MOD;
                        y_share0.data.set(i, temp);
                    }
                    MPC mpc(scale_ord, ring_mod, pt_range);
                    auto b1 = mpc.wrap_protocol(y_share0.data.to_array_1d(), data_trans.io_in, otpack, party);

                    Bytes y_share1_bytes = data_trans.receive_bytes();
                    Feature2DEncrypted y_share1_enc(&context, x_e.level);
                    y_share1_enc.deserialize(y_share1_bytes);
                    y_share1_enc.decompress();

                    Bytes y_share2_bytes = data_trans.receive_bytes();
                    Feature2DEncrypted y_share2_enc(&context, x_e.level);
                    y_share2_enc.deserialize(y_share2_bytes);
                    auto y_ct = y_share1_enc.combine_with_share_new_protocol(y_share0, y_share2_enc, b1);

                    auto y_mg = y_ct.unpack_multiple_channel();

                    auto y_mg_expected = pool.run_avgpool_plaintext(x_mg);

                    for (int i = 0; i < y_mg.get_size(); i++) {
                        y_mg.set(i, y_mg.get_data()[i] / (T_SCALE * kernel_shape[0] * kernel_shape[1]));
                    }
                    auto compare_res = compare(y_mg_expected, y_mg);
                    cout << "max_erro=" << compare_res.max_error << endl;
                    REQUIRE(compare_res.max_error < 5.0e-2 * compare_res.max_abs);
                }
            }
        }
    }
}

TEST_CASE_METHOD(MpcFixture, "feature 2d maxpool relu") {
    srand(1);
    int level = 4;
    Duo pool_stride = {2, 2};
    uint32_t pool_type = MAXPOOL;
    uint32_t n_channel = 4;

    vector<uint32_t> kernel_shapes = {2, 3};
    vector<uint32_t> shapes = {32, 64};
    for (uint32_t s : shapes) {
        Duo shape = {s, s};
        SECTION("shape=" + str(shape)) {
            for (uint32_t ks : kernel_shapes) {
                Duo kernel_shape = {ks, ks};
                SECTION("kernel_shape=" + str(kernel_shape)) {
                    Duo pad = {0, 0};
                    MpcTaskMetaData meta_data;
                    meta_data.append(MpcProtoType::enc_to_share, {"u8", "u8", "u8"}, (uint8_t)level, 0, 0);
                    meta_data.append(MpcProtoType::max_pool, {"duo", "duo"}, kernel_shape, pool_stride);
                    meta_data.append(MpcProtoType::relu, {});
                    meta_data.append(MpcProtoType::share_to_enc, {"u8", "u32"}, (uint8_t)level, n_channel);
                    meta_data.append(MpcProtoType::end, {});
                    Bytes meta_data_bytes = meta_data.serialize();
                    data_trans.send_bytes(meta_data_bytes);

                    auto x_mg = gen_random_array<3>({n_channel, shape[0], shape[1]}, 1.0);
                    Feature2DEncrypted x_e(&context, level);
                    x_e.pack_multiple_channel(x_mg);

                    // to_share
                    Feature2DEncrypted x_share1_enc(&context, x_e.level);
                    Feature2DShare x_share0(ring_mod, scale_ord);
                    x_e.split_to_shares(&x_share1_enc, &x_share0);
                    Bytes x_share1_enc_bytes = x_share1_enc.serialize();
                    data_trans.send_bytes(x_share1_enc_bytes);

                    // pool
                    PoolLayerServer pool(kernel_shape, pool_stride, scale_ord, ring_mod, pool_type, pt_range);
                    Feature2DShare y_share0(ring_mod, scale_ord);
                    pool.run(x_e, x_share0, y_share0);

                    // relu
                    ReluLayerServer act(scale_ord, ring_mod, pt_range);
                    Feature2DShare y_share1(ring_mod, scale_ord);
                    act.run(y_share0, y_share1);

                    for (int i = 0; i < y_share1.data.get_size(); i++) {
                        uint64_t temp = (y_share1.data.get(i) * T_SCALE) % RING_MOD;
                        y_share1.data.set(i, temp);
                    }

                    MPC mpc(scale_ord, ring_mod, pt_range);
                    auto b1 = mpc.wrap_protocol(y_share1.data.to_array_1d(), data_trans.io_in, otpack, party);

                    Bytes y_share1_bytes = data_trans.receive_bytes();
                    Feature2DEncrypted y_share1_enc(&context, x_e.level);
                    y_share1_enc.deserialize(y_share1_bytes);
                    y_share1_enc.decompress();

                    Bytes y_share2_bytes = data_trans.receive_bytes();
                    Feature2DEncrypted y_share2_enc(&context, x_e.level);
                    y_share2_enc.deserialize(y_share2_bytes);
                    auto y_ct = y_share1_enc.combine_with_share_new_protocol(y_share1, y_share2_enc, b1);

                    auto y_mg = y_ct.unpack_multiple_channel();

                    for (int i = 0; i < y_mg.get_size(); i++) {
                        y_mg.set(i, y_mg.get_data()[i] / (T_SCALE));
                    }

                    auto y_mg_vec = y_mg.reshape<1>({0});

                    auto y_mg_expected = pool.run_maxpool_plaintext(x_mg);
                    auto z_mg_expected = act.run_relu_plaintext(y_mg_expected.reshape<1>({0}));

                    auto compare_res = compare(z_mg_expected, y_mg_vec);
                    cout << "max_erro=" << compare_res.max_error << endl;
                    REQUIRE(compare_res.max_error < 5.0e-2 * compare_res.max_abs);
                }
            }
        }
    }
}

TEST_CASE_METHOD(MpcFixture, "feature 2d avgpool relu") {
    srand(1);
    int level = 4;
    Duo pool_stride = {2, 2};
    uint32_t pool_type = AVGPOOL;
    uint32_t n_channel = 4;

    vector<uint32_t> kernel_shapes = {2, 3};
    vector<uint32_t> shapes = {32, 64};
    for (uint32_t s : shapes) {
        Duo shape = {s, s};
        SECTION("shape=" + str(shape)) {
            for (uint32_t ks : kernel_shapes) {
                Duo kernel_shape = {ks, ks};
                SECTION("kernel_shape=" + str(kernel_shape)) {
                    Duo pad = {0, 0};
                    MpcTaskMetaData meta_data;
                    meta_data.append(MpcProtoType::enc_to_share, {"u8", "u8", "u8"}, (uint8_t)level, 0, 0);
                    meta_data.append(MpcProtoType::avg_pool, {"duo", "duo"}, kernel_shape, pool_stride);
                    meta_data.append(MpcProtoType::relu, {});
                    meta_data.append(MpcProtoType::share_to_enc, {"u8", "u32"}, (uint8_t)level, n_channel);
                    meta_data.append(MpcProtoType::end, {});
                    Bytes meta_data_bytes = meta_data.serialize();
                    data_trans.send_bytes(meta_data_bytes);

                    auto x_mg = gen_random_array<3>({n_channel, shape[0], shape[1]}, 1.0);
                    Feature2DEncrypted x_e(&context, level);
                    x_e.pack_multiple_channel(x_mg);

                    // to_share
                    Feature2DEncrypted x_share1_enc(&context, x_e.level);
                    Feature2DShare x_share0(ring_mod, scale_ord);
                    x_e.split_to_shares(&x_share1_enc, &x_share0);
                    Bytes x_share1_enc_bytes = x_share1_enc.serialize();
                    data_trans.send_bytes(x_share1_enc_bytes);

                    // pool
                    PoolLayerServer pool(kernel_shape, pool_stride, scale_ord, ring_mod, pool_type, pt_range);
                    Feature2DShare y_share0(ring_mod, scale_ord);
                    pool.run(x_e, x_share0, y_share0);

                    // relu
                    ReluLayerServer act(scale_ord, ring_mod, pt_range);
                    Feature2DShare y_share1(ring_mod, scale_ord);
                    act.run(y_share0, y_share1);

                    for (int i = 0; i < y_share1.data.get_size(); i++) {
                        uint64_t temp = (y_share1.data.get(i) * T_SCALE) % RING_MOD;
                        y_share1.data.set(i, temp);
                    }

                    MPC mpc(scale_ord, ring_mod, pt_range);
                    auto b1 = mpc.wrap_protocol(y_share1.data.to_array_1d(), data_trans.io_in, otpack, party);
                    printf("b1=%lu\n", b1[0]);

                    Bytes y_share1_bytes = data_trans.receive_bytes();
                    Feature2DEncrypted y_share1_enc(&context, x_e.level);
                    y_share1_enc.deserialize(y_share1_bytes);
                    y_share1_enc.decompress();

                    Bytes y_share2_bytes = data_trans.receive_bytes();
                    Feature2DEncrypted y_share2_enc(&context, x_e.level);
                    y_share2_enc.deserialize(y_share2_bytes);
                    auto y_ct = y_share1_enc.combine_with_share_new_protocol(y_share1, y_share2_enc, b1);

                    Array<double, 3> y_mg = y_ct.unpack_multiple_channel();

                    for (int i = 0; i < y_mg.get_size(); i++) {
                        y_mg.set(i, y_mg.get_data()[i] / (T_SCALE * kernel_shape[0] * kernel_shape[1]));
                    }

                    auto y_mg_expected = pool.run_avgpool_plaintext(x_mg);
                    auto z_mg_expected = act.run_relu_plaintext(y_mg_expected);
                    auto compare_res = compare(z_mg_expected, y_mg);
                    cout << "max_erro=" << compare_res.max_error << endl;
                    REQUIRE(compare_res.max_error < 5.0e-2 * compare_res.max_abs);
                }
            }
        }
    }
}

TEST_CASE_METHOD(MpcFixture, "conv relu") {
    srand(time(NULL));
    uint32_t n_in_channel = 1;
    uint32_t n_out_channel = 1;
    Duo stride = {1, 1};
    Duo skip = {1, 1};

    int level = 2;
    vector<uint32_t> input_shapes = {32, 64};
    vector<uint32_t> kernel_shapes = {1, 3, 5};
    cout << "668ok" << endl;

    for (uint32_t s : input_shapes) {
        Duo input_shape = {s, s};
        SECTION("input_shape=" + str(input_shape)) {
            uint32_t n_channel_per_ct = slot_size / (input_shape[0] * input_shape[1]);
            for (uint32_t k : kernel_shapes) {
                Duo kernel_shape = {k, k};
                SECTION("kernel_shape=" + str(kernel_shape)) {
                    Array<double, 4> conv0_weight =
                        gen_random_array<4>({n_out_channel, n_in_channel, kernel_shape[0], kernel_shape[1]}, 0.1);
                    Array<double, 1> conv0_bias = gen_random_array<1>({n_out_channel}, 0.1);
                    Array<double, 4> conv1_weight =
                        gen_random_array<4>({n_out_channel, n_in_channel, kernel_shape[0], kernel_shape[1]}, 1);
                    Array<double, 1> conv1_bias = gen_random_array<1>({n_out_channel}, 1);
                    Array<double, 3> input_array =
                        gen_random_array<3>({n_in_channel, input_shape[0], input_shape[1]}, 1.0);

                    Feature2DEncrypted input_feature(&context, level);
                    cout << "686ok" << endl;
                    for (int i = 0; i < 5; i++)
                        cout << context.get_parameter().get_q(i) << endl;

                    input_feature.pack_multiple_channel(input_array, false, DEFAULT_SCALE);
                    Conv2DPackedLayer conv0_layer(context.get_parameter(), input_shape, conv0_weight.copy(), conv0_bias.copy(),
                                                  stride, skip, n_channel_per_ct, level);
                    conv0_layer.prepare_weight();
                    cout << "pack ok=" << endl;
                    cout << "input_feature res=" << context.decode(context.decrypt(input_feature.data[0]))[0] << endl;
                    Conv2DPackedLayer conv1_layer(context.get_parameter(), input_shape, conv1_weight.copy(), conv1_bias.copy(),
                                                  stride, skip, n_channel_per_ct, level);
                    conv1_layer.prepare_weight();

                    // conv
                    Feature2DEncrypted x_e = conv0_layer.run(context, input_feature);
                    print_double_message(context.decode(context.decrypt(x_e.data[0])).data(), "conv_res", 10);
                    cout << "@@@@@x_e_scale=" << x_e.data[0].get_scale() << endl;

                    MpcTaskMetaData meta_data;
                    meta_data.append(MpcProtoType::enc_to_share, {"u8", "u8", "u8"}, (uint8_t)x_e.level, 0, 0);
                    meta_data.append(MpcProtoType::relu, {});
                    meta_data.append(MpcProtoType::share_to_enc, {"u8", "u32"}, (uint8_t)x_e.level, n_out_channel);
                    meta_data.append(MpcProtoType::end, {});
                    Bytes meta_data_bytes = meta_data.serialize();
                    data_trans.send_bytes(meta_data_bytes);

                    // enc_to_share
                    Feature2DEncrypted x_share1_enc(&context, x_e.level);
                    Feature2DShare x_share0(ring_mod, scale_ord);
                    x_e.split_to_shares(&x_share1_enc, &x_share0);
                    cout << "x_e res=" << x_share1_enc.unpack_multiple_channel().get(0, 0, 0) << ", share=" << x_share0.data[0] << endl;
                    vector<uint8_t> x_share1_enc_bytes = x_share1_enc.serialize();
                    data_trans.send_bytes(x_share1_enc_bytes);

                    // relu
                    ReluLayerServer act(scale_ord, ring_mod, pt_range);
                    Feature2DShare y_share0(ring_mod, scale_ord);
                    act.run(x_share0, y_share0);
                    cout << "act run=" << y_share0.data[0] << endl;

                    for (int i = 0; i < y_share0.data.get_size(); i++) {
                        uint64_t temp = (y_share0.data.get(i) * T_SCALE) % RING_MOD;
                        y_share0.data.set(i, temp);
                    }

                    MPC mpc(scale_ord, ring_mod, pt_range);
                    auto b1 = mpc.wrap_protocol(y_share0.data.to_array_1d(), data_trans.io_in, otpack, party);
                    printf("b1=%lu\n", b1[0]);

                    Bytes y_share1_bytes = data_trans.receive_bytes();
                    Feature2DEncrypted y_share1_enc(&context, x_e.level);
                    y_share1_enc.deserialize(y_share1_bytes);
                    y_share1_enc.decompress();

                    Bytes y_share2_bytes = data_trans.receive_bytes();
                    Feature2DEncrypted y_share2_enc(&context, x_e.level);
                    y_share2_enc.deserialize(y_share2_bytes);
                    auto y_ct = y_share1_enc.combine_with_share_new_protocol(y_share0, y_share2_enc, b1);
                    Array<double, 3> relu_array_pt = y_ct.unpack_multiple_channel();

                    for (int i = 0; i < relu_array_pt.get_size(); i++) {
                        double temp = relu_array_pt.get(i) / T_SCALE;
                        relu_array_pt.set(i, temp);
                    }

                    auto conv0_array = conv0_layer.run_plaintext(input_array);
                    Array<double, 3> relu_array = act.run_relu_plaintext(conv0_array);
                    print_double_message(conv0_array.get_data(), "real conv0", 15);
                    auto compare_result_relu = compare(relu_array, relu_array_pt);
                    cout << "max_erro_relu=" << compare_result_relu.max_error << endl;
                }
            }
        }
    }
}

TEST_CASE_METHOD(MpcFixture, "f2d to share and f0d back") {
    int level = 3;
    uint32_t n_channel = 4;
    vector<uint32_t> shapes = {32};

    for (uint32_t s : shapes) {
        SECTION("shape=(" + to_string(s) + ',' + to_string(s) + ')') {
            Duo shape = {s, s};
            uint32_t fc_n_channel = n_channel * s * s;
            MpcTaskMetaData meta_data;
            meta_data.append(MpcProtoType::enc_to_share, {"u8", "u8", "u8"}, (uint8_t)level, 0, 0);
            meta_data.append(MpcProtoType::share_2d_to_0d, {});
            meta_data.append(MpcProtoType::share_to_enc_0d, {"u8", "u32"}, (uint8_t)level, fc_n_channel);
            meta_data.append(MpcProtoType::end, {});
            Bytes meta_data_bytes = meta_data.serialize();
            data_trans.send_bytes(meta_data_bytes);

            Array<double, 3> x_mg = gen_random_array<3>({n_channel, shape[0], shape[1]}, 1.0);
            Feature2DEncrypted x_e(&context, level);
            x_e.pack_multiple_channel(x_mg);

            Feature2DEncrypted x_share1_enc(&context, x_e.level);
            Feature2DShare x_share0(ring_mod, scale_ord);
            x_e.split_to_shares(&x_share1_enc, &x_share0);
            vector<uint8_t> x_share1_enc_bytes = x_share1_enc.serialize();
            data_trans.send_bytes(x_share1_enc_bytes);

            Feature0DShare x_share0_0d(ring_mod, scale_ord);
            x_share0_0d.data = std::move(x_share0.data);

            MPC mpc(scale_ord, ring_mod, pt_range);
            data_trans.io_in->flush();

            for (int i = 0; i < x_share0_0d.data.get_size(); i++) {
                uint64_t temp = (x_share0_0d.data.get(i) * T_SCALE) % RING_MOD;
                x_share0_0d.data.set(i, temp);
            }

            auto b1 = mpc.wrap_protocol(x_share0_0d.data.to_array_1d(), data_trans.io_in, otpack, party);
            data_trans.io_in->flush();

            Bytes y_share1_bytes = data_trans.receive_bytes();
            Feature0DEncrypted y_share1_enc(&context, x_e.level);
            y_share1_enc.deserialize(y_share1_bytes);
            y_share1_enc.decompress();

            Bytes y_share2_bytes = data_trans.receive_bytes();
            Feature0DEncrypted y_share2_enc(&context, x_e.level);
            y_share2_enc.deserialize(y_share2_bytes);
            // y_share2_enc.decompress();
            auto y_ct = y_share1_enc.combine_with_share_new_protocol(x_share0_0d, y_share2_enc, b1);
            Array1D y_mg = y_ct.unpack().to_array_1d();

            for (int i = 0; i < y_mg.size(); i++) {
                y_mg[i] = y_mg[i] / T_SCALE;
            }

            Array1D y_mg_expected = array_3d_to_1d(x_mg.to_array_3d());
            auto compare_res = compare(y_mg_expected, y_mg);
            cout << "erro=" << compare_res.max_error << endl;
            REQUIRE(compare_res.max_error < 5.0e-2 * compare_res.max_abs);
        }
    }
}

TEST_CASE_METHOD(MpcFixture, "conv maxpool conv") {

    srand(time(NULL));
    uint32_t n_in_channel = 4;
    uint32_t n_out_channel = 4;
    Duo stride = {1, 1};
    Duo skip = {1, 1};

    int level = 2;
    Duo input_shape = {16, 16};
    Duo kernel_shape0 = {3, 3};
    Duo kernel_shape1 = {3, 3};
    vector<uint32_t> pool_kernel_shapes = {2};
    vector<uint32_t> pool_strides = {2};
    uint32_t pool_type = MAXPOOL;
    uint32_t n_channel_per_ct = slot_size / (input_shape[0] * input_shape[1]);

    for (uint32_t ks : pool_kernel_shapes) {
        Duo pool_kernel_shape = {ks, ks};
        SECTION("pool_kernel_shape=" + str(pool_kernel_shape)) {
            for (uint32_t s : pool_strides) {
                Duo pool_stride = {s, s};
                Duo pad = {0, 0};
                uint32_t pooled_shape0 = (input_shape[0] - pool_kernel_shape[0] + 2 * pad[0]) / pool_stride[0] + 1;
                uint32_t pooled_shape1 = (input_shape[1] - pool_kernel_shape[1] + 2 * pad[1]) / pool_stride[1] + 1;
                
                assert((pooled_shape0 & (pooled_shape0 - 1)) == 0);
                Duo conv1_input_shape = {pooled_shape0, pooled_shape1};
                uint32_t conv1_n_channel_per_ct = slot_size / (conv1_input_shape[0] * conv1_input_shape[1]);
                SECTION("pool_stride=" + str(pool_stride)) {
                    Array<double, 4> conv0_weight =
                        gen_random_array<4>({n_out_channel, n_in_channel, kernel_shape0[0], kernel_shape0[1]}, 0.1);
                    Array<double, 1> conv0_bias = gen_random_array<1>({n_out_channel}, 0.1);
                    Array<double, 4> conv1_weight =
                        gen_random_array<4>({n_out_channel, n_in_channel, kernel_shape1[0], kernel_shape1[1]}, 0.1);
                    Array<double, 1> conv1_bias = gen_random_array<1>({n_out_channel}, 0.1);
                    Array<double, 3> input_array =
                        gen_random_array<3>({n_in_channel, input_shape[0], input_shape[1]}, 1.0);

                    Feature2DEncrypted input_feature(&context, level);
                    input_feature.pack_multiple_channel(input_array, false, DEFAULT_SCALE);

                    Conv2DPackedLayer conv0_layer(context.get_parameter(), input_shape, conv0_weight.copy(), conv0_bias.copy(),
                                                  stride, skip, n_channel_per_ct, level);
                    conv0_layer.prepare_weight();

                    
                    Array<double, 4> conv1_weight_scale(conv1_weight.get_shape());
                    for (int i = 0; i < conv1_weight.get_size(); i++) {
                        conv1_weight_scale.set(i, conv1_weight.get(i) / T_SCALE);
                    }

                    Conv2DPackedLayer conv1_layer(context.get_parameter(), conv1_input_shape, conv1_weight_scale.copy(),
                                                  conv1_bias.copy(), stride, skip, conv1_n_channel_per_ct, level - 1);
                    Conv2DPackedLayer conv1_layer_pt(parameter, conv1_input_shape, conv1_weight.copy(), conv1_bias.copy(), stride,
                                                     skip, conv1_n_channel_per_ct, level - 1);
                    conv1_layer.prepare_weight();

                    // conv
                    Feature2DEncrypted x_e = conv0_layer.run(context, input_feature);
                    auto x_e_res = context.decode(context.decrypt(x_e.data[0]));
                    print_double_message(x_e_res.data(), "conv0=", 10);

                    MpcTaskMetaData meta_data;
                    meta_data.append(MpcProtoType::enc_to_share, {"u8", "u8", "u8"}, (uint8_t)x_e.level, 0, 0);
                    meta_data.append(MpcProtoType::max_pool, {"duo", "duo"}, pool_kernel_shape, pool_stride);
                    meta_data.append(MpcProtoType::share_to_enc, {"u8", "u32"}, (uint8_t)x_e.level, n_out_channel);
                    meta_data.append(MpcProtoType::end, {});
                    Bytes meta_data_bytes = meta_data.serialize();
                    data_trans.send_bytes(meta_data_bytes);

                    // enc_to_share
                    Feature2DEncrypted x_share1_enc(&context, x_e.level);
                    Feature2DShare x_share0(ring_mod, scale_ord);
                    x_e.split_to_shares(&x_share1_enc, &x_share0);
                    vector<uint8_t> x_share1_enc_bytes = x_share1_enc.serialize();
                    data_trans.send_bytes(x_share1_enc_bytes);

                    // maxpool
                    PoolLayerServer pool(pool_kernel_shape, pool_stride, scale_ord, ring_mod, pool_type, pt_range);
                    Feature2DShare y_share0(ring_mod, scale_ord);
                    pool.run(x_e, x_share0, y_share0);

                    for (int i = 0; i < y_share0.data.get_size(); i++) {
                        uint64_t temp = (y_share0.data.get(i) * T_SCALE) % RING_MOD;
                        y_share0.data.set(i, temp);
                    }

                    MPC mpc(scale_ord, ring_mod, pt_range);
                    auto b1 = mpc.wrap_protocol(y_share0.data.to_array_1d(), data_trans.io_in, otpack, party);
                    printf("b1=%lu\n", b1[0]);

                    Bytes y_share1_bytes = data_trans.receive_bytes();
                    Feature2DEncrypted y_share1_enc(&context, x_e.level);
                    y_share1_enc.deserialize(y_share1_bytes);
                    y_share1_enc.decompress();

                    Bytes y_share2_bytes = data_trans.receive_bytes();
                    Feature2DEncrypted y_share2_enc(&context, x_e.level);
                    y_share2_enc.deserialize(y_share2_bytes);
                    auto y_ct = y_share1_enc.combine_with_share_new_protocol(y_share0, y_share2_enc, b1);
                    auto y_res = context.decode(context.decrypt(y_ct.data[0]));
                    print_double_message(y_res.data(), "conv0-max-res=", 10);
                    // conv1
                    Feature2DEncrypted z_ct = conv1_layer.run(context, y_ct);
                    Array<double, 3> output_array = z_ct.unpack_multiple_channel();

                    Array<double, 3> conv0_array = conv0_layer.run_plaintext(input_array);
                    Array<double, 3> pool_array = pool.run_maxpool_plaintext(conv0_array);
                    print_double_message(pool_array.get_data(), "pt-res=", 10);
                    Array<double, 3> conv1_array = conv1_layer_pt.run_plaintext(pool_array);

                    auto compare_result = compare(conv1_array, output_array);
                    cout << "max_erro=" << compare_result.max_error << endl;

                    if (compare_result.max_abs < 1.0e-5) {
                        REQUIRE(compare_result.max_error < 1.0e-5);
                    } else {
                        REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
                    }
                    REQUIRE(compare_result.rmse < 1.0e-2 * compare_result.rms);
                }
            }
        }
    }
}

TEST_CASE_METHOD(MpcFixture, "conv avgpool conv") {
    srand(time(NULL));
    uint32_t n_in_channel = 1;
    uint32_t n_out_channel = 1;
    Duo stride = {1, 1};
    Duo skip = {1, 1};

    int level = 2;
    Duo input_shape = {32, 32};
    Duo kernel_shape0 = {3, 3};
    Duo kernel_shape1 = {3, 3};
    vector<uint32_t> pool_kernel_shapes = {4};
    vector<uint32_t> pool_strides = {4};
    uint32_t pool_type = AVGPOOL;
    uint32_t n_channel_per_ct = slot_size / (input_shape[0] * input_shape[1]);

    for (uint32_t ks : pool_kernel_shapes) {
        Duo pool_kernel_shape = {ks, ks};
        SECTION("pool_kernel_shape=" + str(pool_kernel_shape)) {
            for (uint32_t s : pool_strides) {
                Duo pool_stride = {s, s};
                Duo pad = {0, 0};
                uint32_t pooled_shape0 = (input_shape[0] - pool_kernel_shape[0] + 2 * pad[0]) / pool_stride[0] + 1;
                uint32_t pooled_shape1 = (input_shape[1] - pool_kernel_shape[1] + 2 * pad[1]) / pool_stride[1] + 1;
                
                assert((pooled_shape0 & (pooled_shape0 - 1)) == 0);
                Duo conv1_input_shape = {pooled_shape0, pooled_shape1};
                uint32_t conv1_n_channel_per_ct = slot_size / (conv1_input_shape[0] * conv1_input_shape[1]);
                SECTION("pool_stride=" + str(pool_stride)) {
                    Array<double, 4> conv0_weight =
                        gen_random_array<4>({n_out_channel, n_in_channel, kernel_shape0[0], kernel_shape0[1]}, 0.1);
                    Array<double, 1> conv0_bias = gen_random_array<1>({n_out_channel}, 0.1);
                    Array<double, 4> conv1_weight =
                        gen_random_array<4>({n_out_channel, n_in_channel, kernel_shape1[0], kernel_shape1[1]}, 0.1);
                    Array<double, 1> conv1_bias = gen_random_array<1>({n_out_channel}, 0.1);
                    Array<double, 3> input_array =
                        gen_random_array<3>({n_in_channel, input_shape[0], input_shape[1]}, 0.0);

                    Feature2DEncrypted input_feature(&context, level);
                    input_feature.pack_multiple_channel(input_array);

                    Conv2DPackedLayer conv0_layer(context.get_parameter(), input_shape, conv0_weight.copy(), conv0_bias.copy(),
                                                  stride, skip, n_channel_per_ct, level);
                    conv0_layer.prepare_weight();

                    Array<double, 4> conv1_weight_scale(conv1_weight.get_shape());
                    for (int i = 0; i < conv1_weight.get_size(); i++) {
                        conv1_weight_scale.set(i, conv1_weight.get(i) /
                                                      (T_SCALE * pool_kernel_shape[0] * pool_kernel_shape[1]));
                    }

                    Conv2DPackedLayer conv1_layer(context.get_parameter(), conv1_input_shape, conv1_weight_scale.copy(),
                                                  conv1_bias.copy(), stride, skip, conv1_n_channel_per_ct, level - 1);
                    Conv2DPackedLayer conv1_layer_pt(context.get_parameter(), conv1_input_shape, conv1_weight.copy(),
                                                     conv1_bias.copy(), stride, skip, conv1_n_channel_per_ct, level - 1);
                    conv1_layer.prepare_weight();

                    // conv
                    Feature2DEncrypted x_e = conv0_layer.run(context, input_feature);

                    MpcTaskMetaData meta_data;
                    meta_data.append(MpcProtoType::enc_to_share, {"u8", "u8", "u8"}, (uint8_t)x_e.level, 0, 0);
                    meta_data.append(MpcProtoType::avg_pool, {"duo", "duo"}, pool_kernel_shape, pool_stride);
                    meta_data.append(MpcProtoType::share_to_enc, {"u8", "u32"}, (uint8_t)x_e.level, n_out_channel);
                    meta_data.append(MpcProtoType::end, {});
                    Bytes meta_data_bytes = meta_data.serialize();
                    data_trans.send_bytes(meta_data_bytes);

                    // enc_to_share
                    Feature2DEncrypted x_share1_enc(&context, x_e.level);
                    Feature2DShare x_share0(ring_mod, scale_ord);
                    x_e.split_to_shares(&x_share1_enc, &x_share0);
                    vector<uint8_t> x_share1_enc_bytes = x_share1_enc.serialize();
                    data_trans.send_bytes(x_share1_enc_bytes);

                    // maxpool
                    PoolLayerServer pool(pool_kernel_shape, pool_stride, scale_ord, ring_mod, pool_type, pt_range);
                    Feature2DShare y_share0(ring_mod, scale_ord);
                    pool.run(x_e, x_share0, y_share0);

                    for (int i = 0; i < y_share0.data.get_size(); i++) {
                        uint64_t temp = (y_share0.data.get(i) * T_SCALE) % RING_MOD;
                        y_share0.data.set(i, temp);
                    }

                    MPC mpc(scale_ord, ring_mod, pt_range);
                    auto b1 = mpc.wrap_protocol(y_share0.data.to_array_1d(), data_trans.io_in, otpack, party);
                    printf("b1=%lu\n", b1[0]);

                    Bytes y_share1_bytes = data_trans.receive_bytes();
                    Feature2DEncrypted y_share1_enc(&context, x_e.level);
                    y_share1_enc.deserialize(y_share1_bytes);
                    y_share1_enc.decompress();

                    Bytes y_share2_bytes = data_trans.receive_bytes();
                    Feature2DEncrypted y_share2_enc(&context, x_e.level);
                    y_share2_enc.deserialize(y_share2_bytes);
                    auto y_ct = y_share1_enc.combine_with_share_new_protocol(y_share0, y_share2_enc, b1);

                    // conv1
                    Feature2DEncrypted z_ct = conv1_layer.run(context, y_ct);
                    Array<double, 3> output_array = z_ct.unpack_multiple_channel();

                    Array<double, 3> conv0_array = conv0_layer.run_plaintext(input_array);
                    Array<double, 3> pool_array = pool.run_avgpool_plaintext(conv0_array);
                    Array<double, 3> conv1_array = conv1_layer_pt.run_plaintext(pool_array);

                    auto compare_result = compare(conv1_array, output_array);
                    cout << "compare_result=" << compare_result.max_error << endl;

                    
                    if (compare_result.max_abs < 1.0e-5) {
                        REQUIRE(compare_result.max_error < 1.0e-5);
                    } else {
                        REQUIRE(compare_result.max_error < 5.0e-2 * compare_result.max_abs);
                    }
                }
            }
        }
    }
}

Feature0DEncrypted
compute_distance_fhe(Feature0DEncrypted& x_e, CkksContext& context, int level, string file_path, string name) {
    Array2D vec_list;
    Array1D vec1 = read_plain_face(file_path, name, 1.0, 1);
    vec1 = L2_normal(vec1);
    vec_list.push_back(vec1);
    DataTransmission data_trans(io);
    MpcTaskMetaData meta_data;
    Duo array_num = {1, 128};
    meta_data.append(MpcProtoType::enc_to_share_0d, {"u8", "u32", "u8", "u8"}, (uint8_t)level, 128, 0, 0);
    meta_data.append(MpcProtoType::distance, {"duo"}, array_num);
    meta_data.append(MpcProtoType::share_to_enc_0d, {"u8", "u32"}, (uint8_t)level, x_e.n_channel);
    meta_data.append(MpcProtoType::end, {});
    Bytes meta_data_bytes = meta_data.serialize();
    data_trans.send_bytes(meta_data_bytes);
    // to_share
    int scale_ord = DEFAULT_SCALE_BIT;
    double pt_range = 128;
    uint64_t ring_mod = RING_MOD;
    Feature0DEncrypted x_share1_enc(&context, x_e.level);
    Feature0DShare x_share0(ring_mod, scale_ord);
    x_e.split_to_shares(&x_share1_enc, &x_share0);
    Bytes x_share1_enc_bytes = x_share1_enc.serialize();
    data_trans.send_bytes(x_share1_enc_bytes);

    // distance
    DistanceLayerServer dist(scale_ord, ring_mod, pt_range);
    Feature0DShare y_share0(ring_mod, scale_ord);
    Array2DUint input_y_mat;
    for (int i = 0; i < vec_list.size(); i++) {
        auto input_y_vec_uint = array_1d_double_to_uint64(vec_list[i], scale_ord, ring_mod);
        input_y_mat.push_back(input_y_vec_uint);
    }

    dist.run(x_share0, input_y_mat, y_share0, false);
    cout << "dist res=" << y_share0.data[0] << endl;

    for (int i = 0; i < y_share0.data.get_size(); i++) {
        uint64_t temp = (y_share0.data.get(i) * T_SCALE) % RING_MOD;
        y_share0.data.set(i, temp);
    }

    MPC mpc(scale_ord, ring_mod, pt_range);
    data_trans.io_in->flush();
    auto b1 = mpc.wrap_protocol(y_share0.data.to_array_1d(), data_trans.io_in, otpack, party);
    data_trans.io_in->flush();

    Bytes y_share1_bytes = data_trans.receive_bytes();
    Feature0DEncrypted y_share1_enc(&context, x_e.level);
    y_share1_enc.deserialize(y_share1_bytes);
    y_share1_enc.decompress();

    Bytes y_share2_bytes = data_trans.receive_bytes();
    Feature0DEncrypted y_share2_enc(&context, x_e.level);
    y_share2_enc.deserialize(y_share2_bytes);
    auto f0d = y_share1_enc.combine_with_share_new_protocol(y_share0, y_share2_enc, b1);

    auto pt_res = x_e.unpack().to_array_1d();
    auto pt_res_L2 = L2_normal(pt_res);
    auto real_sidt = compute_distance(pt_res_L2, vec1);
    return f0d;
}

TEST_CASE_METHOD(MpcFixture, "test distance") {
    srand(time(NULL));
    std::cout << std::fixed << std::setprecision(10);
    int level = 1;
    Array1D vec2 = read_plain_face("/home/zhongy/encrypted-inference/inference/"
                                   "face_gui_v2/plain_face1.txt",
                                   "Alice_Fisher_0001", 1.0, 1);
    // Aaron_Peirsol_0001
    Array1D vec1 = read_plain_face("/home/zhongy/encrypted-inference/inference/"
                                   "face_gui_v2/plain_face1.txt",
                                   "Aaron_Peirsol_0001", 1.0, 1);
    Array1D vec3 = read_plain_face("/home/zhongy/encrypted-inference/inference/"
                                   "face_gui_v2/plain_face1.txt",
                                   "Aaron_Peirsol_0002", 1.0, 1);
    vec2 = L2_normal(vec2);
    vec3 = L2_normal(vec3);
    Array2D vec_list;
    vec_list.push_back(vec2);
    uint32_t n_channel = vec1.size();
    Feature0DEncrypted x_e(&context, level);
    x_e.pack(Array<double, 1>::from_array_1d(vec1), false, DEFAULT_SCALE);
    cout << "x_e[0]=" << vec1[0] << endl;
    x_e.skip = 1;
    x_e.n_channel = n_channel;
    Duo array_num = {(uint32_t)vec_list.size(), n_channel};

    MpcTaskMetaData meta_data;
    meta_data.append(MpcProtoType::enc_to_share_0d, {"u8", "u32", "u8", "u8"}, (uint8_t)level, n_channel, 0, 0);
    meta_data.append(MpcProtoType::distance, {"duo"}, array_num);
    meta_data.append(MpcProtoType::recovery_share, {});
    meta_data.append(MpcProtoType::end, {});
    Bytes meta_data_bytes = meta_data.serialize();
    data_trans.send_bytes(meta_data_bytes);

    // to_share
    Feature0DEncrypted x_share1_enc(&context, x_e.level);
    Feature0DShare x_share0(ring_mod, scale_ord);
    x_e.split_to_shares(&x_share1_enc, &x_share0);
    Bytes x_share1_enc_bytes = x_share1_enc.serialize();
    data_trans.send_bytes(x_share1_enc_bytes);

    // distance
    DistanceLayerServer dist(scale_ord, ring_mod, pt_range);
    Feature0DShare y_share0(ring_mod, scale_ord);
    Array2DUint input_y_mat;
    for (int i = 0; i < vec_list.size(); i++) {
        auto input_y_vec_uint = array_1d_double_to_uint64(vec_list[i], scale_ord, ring_mod);
        input_y_mat.push_back(input_y_vec_uint);
    }

    dist.run(x_share0, input_y_mat, y_share0, false);
    cout << "dist res=" << y_share0.data[0] << endl;
    // recovery share
    Array1DUint recv_vec(y_share0.data.get_size(), 0);
    io->recv_data(recv_vec.data(), recv_vec.size() * sizeof(uint64_t));
    auto res = recovery_share(recv_vec, y_share0.data.to_array_1d(), ring_mod, scale_ord);

    auto vec1_L2 = L2_normal(vec1);
    Array1D real_dist_vec;
    for (int i = 0; i < vec_list.size(); i++) {
        auto real_dist = compute_distance(vec1_L2, vec_list[i]);
        real_dist_vec.push_back(real_dist * real_dist);
    }

    auto compare_res = compare(res, real_dist_vec);
    fprintf(stderr, "res=%f,%f", res[0], real_dist_vec[0]);
    REQUIRE(compare_res.max_error < 0.01);
}

TEST_CASE_METHOD(MpcFixture, "test distance fc") {
    // srand(time);
    srand(time(NULL));
    Array1D vec2 = read_plain_face("/home/zhongy/encrypted-inference/inference/"
                                   "face_gui_v2/plain_face1.txt",
                                   "Alice_Fisher_0001", 1.0, 1);
    // Aaron_Peirsol_0001
    Array1D vec1 = read_plain_face("/home/zhongy/encrypted-inference/inference/"
                                   "face_gui_v2/plain_face1.txt",
                                   "Aaron_Peirsol_0001", 1.0, 1);
    Array1D vec3 = read_plain_face("/home/zhongy/encrypted-inference/inference/"
                                   "face_gui_v2/plain_face1.txt",
                                   "Aaron_Peirsol_0002", 1.0, 1);
    vec2 = L2_normal(vec2);
    vec3 = L2_normal(vec3);
    Array2D vec_list;
    vec_list.push_back(vec2);
    uint32_t n_channel = vec1.size();

    Duo array_shape = {4, 4};
    Duo fc_skip = {1, 1};
    Array<double, 1> input_array = gen_random_array<1>({512}, 0.1);
    Array<double, 2> fc_weight = gen_random_array<2>({128, 512}, 1);
    Array<double, 1> fc_bias = gen_random_array<1>({128}, 1);
    uint32_t fc_n_channel_per_ct = slot_size / (array_shape[0] * array_shape[1]);
    uint32_t fc_level = 2;

    DensePackedLayer fc_layer(context.get_parameter(), fc_weight.copy(), fc_bias.copy(), fc_n_channel_per_ct,
                              fc_level, 0);
    fc_layer.normal_dense = false;
    Feature0DEncrypted x_in(&context, fc_level);
    x_in.skip = 1;
    cout << "1316 skip=" << x_in.skip << endl;
    x_in.pack(input_array, false, DEFAULT_SCALE);
    cout << "x_in scale=" << x_in.data[0].get_scale() << endl;
    fc_layer.prepare_weight_0d_skip(1);
    cout << "prepare ok " << endl;
    auto x_e = fc_layer.run_0d_skip(context, x_in);
    cout << "x_e scale=" << x_e.data[0].get_scale() << endl;
    x_e.skip = array_shape[0] * array_shape[1];
    x_e.n_channel_per_ct = 4096 / (array_shape[0] * array_shape[1]);
    auto fc_pt = x_e.unpack().to_array_1d();
    print_double_message(fc_pt.data(), "fc_pt", 10);
    auto real_res = fc_layer.run_plaintext(input_array).to_array_1d();

    auto compare_res = compare(fc_pt, real_res);
    fprintf(stderr, "real_res=%f\n", compare_res.max_error);

    string file_name = "/home/zhongy/encrypted-inference/inference/face_gui_v2/plain_face1.txt";
    string name = "Alice_Fisher_0001";
    auto res = compute_distance_fhe(x_e, context, x_e.level, file_name, name);
    auto res_pt = res.unpack();
    fprintf(stderr, "ct_res=%f\n", res_pt[0] / T_SCALE);
    Array1D compare_vec = {res_pt[0] / T_SCALE};

    auto vec1_L2 = L2_normal(real_res);
    Array1D real_dist_vec;
    for (int i = 0; i < vec_list.size(); i++) {
        auto real_dist = compute_distance(vec1_L2, vec_list[i]);
        real_dist_vec.push_back(real_dist * real_dist);
    }
    cout << "real_dist=" << real_dist_vec[0] << endl;
    auto compare_res_1 = compare(compare_vec, real_dist_vec);
    REQUIRE(compare_res_1.max_error < 0.05);
}

TEST_CASE_METHOD(MpcFixture, "test relu6") {
    int level = 1;
    uint32_t n_channel = 4;
    Duo shape = {32, 32};
    Duo skip = {1, 1};

    srand(0);
    MpcTaskMetaData meta_data;
    meta_data.append(MpcProtoType::enc_to_share, {"u8", "u8", "u8"}, (uint8_t)level, 0, 0);
    meta_data.append(MpcProtoType::relu6, {});
    meta_data.append(MpcProtoType::share_to_enc, {"u8", "u32"}, (uint8_t)level, n_channel);
    meta_data.append(MpcProtoType::end, {});
    Bytes meta_data_bytes = meta_data.serialize();
    data_trans.send_bytes(meta_data_bytes);

    // encrypt
    Array<double, 3> x_mg = gen_random_array<3>({n_channel, shape[0], shape[1]}, 10.0);
    Feature2DEncrypted x_e(&context, level);
    x_e.pack_multiple_channel(x_mg);
    x_e.skip = skip;
    x_e.shape = {x_e.shape[0] / skip[0], x_e.shape[1] / skip[1]};

    // enc_to_share
    Feature2DEncrypted x_share1_enc(&context, x_e.level);
    Feature2DShare x_share0(ring_mod, scale_ord);
    x_e.split_to_shares(&x_share1_enc, &x_share0);
    vector<uint8_t> x_share1_enc_bytes = x_share1_enc.serialize();
    data_trans.send_bytes(x_share1_enc_bytes);

    // relu6:
    Relu6LayerServer act(scale_ord, ring_mod, pt_range);
    Feature2DShare y_share0(ring_mod, scale_ord);
    act.run(x_share0, y_share0);
    cout << "server relu6 data=" << y_share0.data[0] << endl;

    for (int i = 0; i < y_share0.data.get_size(); i++) {
        uint64_t temp = (y_share0.data.get(i) * T_SCALE) % RING_MOD;
        y_share0.data.set(i, temp);
    }

    MPC mpc(scale_ord, ring_mod, pt_range);
    auto b1 = mpc.wrap_protocol(y_share0.data.to_array_1d(), data_trans.io_in, otpack, party);
    printf("b1=%lu\n", b1[0]);

    Bytes y_share1_bytes = data_trans.receive_bytes();
    Feature2DEncrypted y_share1_enc(&context, x_e.level);
    y_share1_enc.deserialize(y_share1_bytes);
    y_share1_enc.decompress();

    Bytes y_share2_bytes = data_trans.receive_bytes();
    Feature2DEncrypted y_share2_enc(&context, x_e.level);
    y_share2_enc.deserialize(y_share2_bytes);
    auto y_ct = y_share1_enc.combine_with_share_new_protocol(y_share0, y_share2_enc, b1);
    Array<double, 3> y_mg = y_ct.unpack_multiple_channel();

    Array<double, 3> y_true = act.run_relu6_plaintext(x_mg);

    for (int i = 0; i < y_mg.get_size(); i++) {
        double temp = y_mg.get(i) / T_SCALE;
        y_mg.set(i, temp);
    }
    cout << "y_mg=" << y_mg.get(0, 0, 0);
    auto compare_res = compare(y_true, y_mg);
    REQUIRE(compare_res.max_error < 5.0e-2 * compare_res.max_abs);
}

TEST_CASE_METHOD(MpcFixture, "test_argmax") {
    int level = 3;

    vector<uint32_t> n_channels = {1024, 4096};

    for (uint32_t n_channel : n_channels) {
        SECTION("n_channel=(" + to_string(n_channel) + ')') {
            MpcTaskMetaData meta_data;
            meta_data.append(MpcProtoType::enc_to_share_0d, {"u8", "u32", "u8", "u8"}, (uint8_t)level, n_channel, 0, 0);
            meta_data.append(MpcProtoType::argmax, {});
            meta_data.append(MpcProtoType::share_to_enc_0d, {"u8", "u32"}, (uint8_t)level, n_channel);
            meta_data.append(MpcProtoType::end, {});
            Bytes meta_data_bytes = meta_data.serialize();
            data_trans.send_bytes(meta_data_bytes);

            vector<double> x_mg_vec(n_channel, 0);
            auto x_mg = Array<double, 1>::from_array_1d(x_mg_vec);
            x_mg.set(2, 1.2);
            Feature0DEncrypted x_e(&context, level);
            x_e.pack(x_mg);
            x_e.skip = 1;
            x_e.n_channel = n_channel;

            Feature0DEncrypted x_share1_enc(&context, x_e.level);
            Feature0DShare x_share0(ring_mod, scale_ord);
            x_e.split_to_shares(&x_share1_enc, &x_share0);

            vector<uint8_t> x_share1_enc_bytes = x_share1_enc.serialize();
            data_trans.send_bytes(x_share1_enc_bytes);

            MPC mpc(scale_ord, ring_mod, pt_range);
            data_trans.io_in->flush();
            // relu
            ArgMaxLayer argmax(scale_ord, ring_mod, pt_range);
            Feature0DShare y_share0(ring_mod, scale_ord);
            argmax.run(x_share0, y_share0);

            for (int i = 0; i < y_share0.data.get_size(); i++) {
                uint64_t temp = (y_share0.data.get(i) * T_SCALE) % RING_MOD;
                y_share0.data.set(i, temp);
            }

            auto b1 = mpc.wrap_protocol(y_share0.data.to_array_1d(), data_trans.io_in, otpack, party);
            data_trans.io_in->flush();

            Bytes y_share1_bytes = data_trans.receive_bytes();
            Feature0DEncrypted y_share1_enc(&context, x_e.level);
            y_share1_enc.deserialize(y_share1_bytes);
            y_share1_enc.decompress();

            Bytes y_share2_bytes = data_trans.receive_bytes();
            Feature0DEncrypted y_share2_enc(&context, x_e.level);
            y_share2_enc.deserialize(y_share2_bytes);
            auto y_ct = y_share1_enc.combine_with_share_new_protocol(y_share0, y_share2_enc, b1);

            Array<double, 1> y_mg = y_ct.unpack();

            for (int i = 0; i < y_mg.get_size(); i++) {
                y_mg.set(i, y_mg.get_data()[i] / T_SCALE);
            }
            print_double_message(y_mg.get_data(), "ct_res=", 10);
            Array<double, 1>& y_mg_expected = x_mg;

            int index = argmax.argmax_plaintext_call(x_mg.to_array_1d());
            REQUIRE(abs(index - y_mg[0]) < 5.0e-2);
        }
    }
}
TEST_CASE_METHOD(MpcFixture, "test_div") {
    int level = 3;

    vector<uint32_t> n_channels = {1024, 4096};

    for (uint32_t n_channel : n_channels) {
        SECTION("n_channel=(" + to_string(n_channel) + ')') {
            vector<double> div_num(n_channel, 0.5);
            div_num = gen_random_array_positive<1>({n_channel}, 1).to_array_1d();
            MpcTaskMetaData meta_data;
            meta_data.append(MpcProtoType::enc_to_share_0d, {"u8", "u32", "u8", "u8"}, (uint8_t)level, n_channel, 0, 0);
            meta_data.append(MpcProtoType::div, {});
            meta_data.append(MpcProtoType::share_to_enc_0d, {"u8", "u32"}, (uint8_t)level, n_channel);
            meta_data.append(MpcProtoType::end, {});
            Bytes meta_data_bytes = meta_data.serialize();
            data_trans.send_bytes(meta_data_bytes);

            vector<double> x_mg_vec(n_channel, 1);
            auto x_mg = Array<double, 1>::from_array_1d(x_mg_vec);
            x_mg = gen_random_array_positive<1>({n_channel}, 1);
            x_mg.set(2, 1.2);
            Feature0DEncrypted x_e(&context, level);
            x_e.pack(x_mg, false, DEFAULT_SCALE);
            x_e.skip = 1;
            x_e.n_channel = n_channel;

            auto x_share0 = server_enc_to_share(context, x_e, scale_ord, ring_mod);

            // div
            DivLayerServer div_layer(scale_ord, ring_mod, pt_range);
            Feature0DShare y_share0(ring_mod, scale_ord);
            div_layer.run(x_share0, div_num, y_share0);

            auto y_ct = share_to_enc(y_share0, context, scale_ord, ring_mod, pt_range, x_e.level);
            Array<double, 1> y_mg = y_ct.unpack();

            for (int i = 0; i < y_mg.get_size(); i++) {
                y_mg.set(i, y_mg.get_data()[i] / T_SCALE);
            }
            print_double_message(y_mg.get_data(), "ct_res=", 10);
            Array<double, 1>& y_mg_expected = x_mg;
            y_mg_expected = div_layer.div_plaintext_call(y_mg_expected, div_num);
            print_double_message(y_mg_expected.get_data(), "pt_res=", 10);
            auto compare_res = compare(y_mg_expected, y_mg);
            write_file_common("y_mg", y_mg.to_array_1d());
            write_file_common("y_mg_expect", y_mg_expected.to_array_1d());
            REQUIRE(compare_res.max_error < 5.0e-2 * compare_res.max_abs);
        }
    }
}

TEST_CASE_METHOD(MpcFixture, "test_div_reciprocal") {
    int level = 3;
    vector<uint32_t> n_channels = {1024, 4096};

    for (uint32_t n_channel : n_channels) {
        SECTION("n_channel=(" + to_string(n_channel) + ')') {
            vector<double> div_num(n_channel, 0.5);
            div_num = gen_random_array_positive<1>({n_channel}, 1).to_array_1d();
            MpcTaskMetaData meta_data;
            meta_data.append(MpcProtoType::enc_to_share_0d, {"u8", "u32", "u8", "u8"}, (uint8_t)level, n_channel, 0, 0);
            meta_data.append(MpcProtoType::reciprocal, {});
            meta_data.append(MpcProtoType::share_to_enc_0d, {"u8", "u32"}, (uint8_t)level, n_channel);
            meta_data.append(MpcProtoType::end, {});
            Bytes meta_data_bytes = meta_data.serialize();
            data_trans.send_bytes(meta_data_bytes);

            vector<double> x_mg_vec(n_channel, 1);
            auto x_mg = Array<double, 1>::from_array_1d(x_mg_vec);
            x_mg = gen_random_array_positive<1>({n_channel}, 1);
            x_mg.set(2, 1.2);
            Feature0DEncrypted x_e(&context, level);
            x_e.pack(x_mg, false, DEFAULT_SCALE);
            x_e.skip = 1;
            x_e.n_channel = n_channel;

            auto x_share0 = server_enc_to_share(context, x_e, scale_ord, ring_mod);

            // div_reciprocal
            DivLayerServer div_layer(scale_ord, ring_mod, pt_range);
            Feature0DShare y_share0(ring_mod, scale_ord);
            div_layer.run_reciprocal(x_share0, y_share0);

            auto y_ct = share_to_enc(y_share0, context, scale_ord, ring_mod, pt_range, x_e.level);

            Array<double, 1> y_mg = y_ct.unpack();

            for (int i = 0; i < y_mg.get_size(); i++) {
                y_mg.set(i, y_mg.get_data()[i] / T_SCALE);
            }
            print_double_message(y_mg.get_data(), "ct_res=", 10);
            Array<double, 1>& y_mg_expected = x_mg;
            y_mg_expected = div_layer.div_reciprocal_plaintext_call(y_mg_expected);
            print_double_message(y_mg_expected.get_data(), "pt_res=", 10);
            auto compare_res = compare(y_mg_expected, y_mg);
            REQUIRE(compare_res.max_error < 5.0e-2 * compare_res.max_abs);
        }
    }
}

TEST_CASE_METHOD(MpcFixture, "test_sqrt") {
    int level = 3;
    vector<uint32_t> n_channels = {1024, 4096};

    for (uint32_t n_channel : n_channels) {
        SECTION("n_channel=(" + to_string(n_channel) + ')') {
            vector<double> div_num(n_channel, 0.5);
            div_num = gen_random_array_positive<1>({n_channel}, 1).to_array_1d();
            MpcTaskMetaData meta_data;
            meta_data.append(MpcProtoType::enc_to_share_0d, {"u8", "u32", "u8", "u8"}, (uint8_t)level, n_channel, 0, 0);
            meta_data.append(MpcProtoType::sqrt, {});
            meta_data.append(MpcProtoType::share_to_enc_0d, {"u8", "u32"}, (uint8_t)level, n_channel);
            meta_data.append(MpcProtoType::end, {});
            Bytes meta_data_bytes = meta_data.serialize();
            data_trans.send_bytes(meta_data_bytes);

            vector<double> x_mg_vec(n_channel, 1);
            auto x_mg = Array<double, 1>::from_array_1d(x_mg_vec);
            x_mg = gen_random_array_positive<1>({n_channel}, 1);
            x_mg.set(2, 1.2);
            Feature0DEncrypted x_e(&context, level);
            x_e.pack(x_mg, false, DEFAULT_SCALE);
            x_e.skip = 1;
            x_e.n_channel = n_channel;

            auto x_share0 = server_enc_to_share(context, x_e, scale_ord, ring_mod);

            // sqrt
            SqrtLayer sqrt_layer(scale_ord, ring_mod, pt_range);
            Feature0DShare y_share0(ring_mod, scale_ord);
            sqrt_layer.run(x_share0, y_share0);

            auto y_ct = share_to_enc(y_share0, context, scale_ord, ring_mod, pt_range, x_e.level);

            Array<double, 1> y_mg = y_ct.unpack();

            for (int i = 0; i < y_mg.get_size(); i++) {
                y_mg.set(i, y_mg.get_data()[i] / T_SCALE);
            }
            print_double_message(y_mg.get_data(), "ct_res=", 10);
            Array<double, 1>& y_mg_expected = x_mg;
            y_mg_expected = sqrt_layer.sqrt_plaintext_call(y_mg_expected);
            print_double_message(y_mg_expected.get_data(), "pt_res=", 10);
            auto compare_res = compare(y_mg_expected, y_mg);
            REQUIRE(compare_res.max_error < 5.0e-2 * compare_res.max_abs);
        }
    }
}

TEST_CASE_METHOD(MpcFixture, "test_softmax") {
    int level = 3;
    vector<uint32_t> n_channels = {150};

    for (uint32_t n_channel : n_channels) {
        SECTION("n_channel=(" + to_string(n_channel) + ')') {
            vector<double> div_num(n_channel, 0.5);
            div_num = gen_random_array_positive<1>({n_channel}, 1).to_array_1d();
            MpcTaskMetaData meta_data;
            meta_data.append(MpcProtoType::enc_to_share_0d, {"u8", "u32", "u8", "u8"}, (uint8_t)level, n_channel, 0, 0);
            meta_data.append(MpcProtoType::softmax, {});
            meta_data.append(MpcProtoType::share_to_enc_0d, {"u8", "u32"}, (uint8_t)level, n_channel);
            meta_data.append(MpcProtoType::end, {});
            Bytes meta_data_bytes = meta_data.serialize();
            data_trans.send_bytes(meta_data_bytes);

            vector<double> x_mg_vec(n_channel, 1);
            auto x_mg = Array<double, 1>::from_array_1d(x_mg_vec);
            x_mg = gen_random_array_positive<1>({n_channel}, 1);
            Feature0DEncrypted x_e(&context, level);
            x_e.pack(x_mg, false, DEFAULT_SCALE);
            x_e.skip = 1;
            x_e.n_channel = n_channel;

            auto x_share0 = server_enc_to_share(context, x_e, scale_ord, ring_mod);
            // softmax
            SoftMaxLayerServer softmax_layer(scale_ord, ring_mod, pt_range);
            Feature0DShare y_share0(ring_mod, scale_ord);
            // EndComputation();
            softmax_layer.run(x_share0, y_share0);
            // EndComputation();
            auto y_ct = share_to_enc(y_share0, context, scale_ord, ring_mod, pt_range, x_e.level);

            Array<double, 1> y_mg = y_ct.unpack();

            for (int i = 0; i < y_mg.get_size(); i++) {
                y_mg.set(i, y_mg.get_data()[i] / T_SCALE);
            }
            print_double_message(y_mg.get_data(), "ct_res=", 10);
            Array<double, 1>& y_mg_expected = x_mg;
            y_mg_expected = softmax_layer.softmax_plaintext_call(y_mg_expected);
            print_double_message(y_mg_expected.get_data(), "pt_res=", 10);
            auto compare_res = compare(y_mg_expected, y_mg);
            REQUIRE(compare_res.max_error < 5.0e-2 * compare_res.max_abs);
        }
    }
}
