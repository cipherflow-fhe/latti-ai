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

#include "mpc/fhe_mpc.h"
#include "mpc/mpc_numeric.h"
#include "mpc_wrapper/enc_share_conversion.h"
#include "mpc_wrapper/mpc_data_transmission.h"
#include "data_structs/feature.h"
#include "mpc/mpc_session.h"

#include <cstring>
#include <cstdlib>

using namespace std;
using namespace lattisense;

namespace {

constexpr int kDefaultMpcTestPort = 12309;
constexpr int kLevel = 5;
constexpr uint32_t kNChannel = 4;
const Duo kShape = {16, 16};

bool is_integer_arg(const char* value) {
    if (value == nullptr || *value == '\0') {
        return false;
    }
    if (*value == '-' || *value == '+') {
        value++;
    }
    while (*value != '\0') {
        if (*value < '0' || *value > '9') {
            return false;
        }
        value++;
    }
    return true;
}

int get_test_port(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (is_integer_arg(argv[i])) {
            return atoi(argv[i]);
        }
    }
    if (const char* env_port = getenv("MPC_TEST_PORT")) {
        return atoi(env_port);
    }
    return kDefaultMpcTestPort;
}

void init_mpc_party(int port_in) {
    ::mpc::init_party(CLIENT, port_in);
}

void scale_share_by_t(Feature2DShare& share) {
    for (int i = 0; i < share.data.get_size(); i++) {
        share.data.set(i, (share.data.get(i) * mpc::T_SCALE) % share.ring_mod);
    }
}

CkksContext make_test_context() {
    return CkksContext::create_random_context(CkksParameter::create_parameter(8192));
}

void run_relu_client(int port_in) {
    init_mpc_party(port_in);

    DataTransmission dt = ::mpc::data_transmission();
    CkksContext context = make_test_context();
    MpcDataTransmission(dt).send_public_context(context);

    Bytes x_share1_bytes = dt.receive_bytes();
    Feature2DEncrypted x_share1_enc(&context, kLevel, {1, 1}, {1, 1}, PackType::MultipleChannelPacking);
    x_share1_enc.deserialize(x_share1_bytes);

    map<string, unique_ptr<CkksContext>> ckks_contexts;
    CkksContext* context_in = &context;
    CkksContext* context_out = &context;
    EncToShareClient enc_to_share_client(ckks_contexts, context_in, context_out);
    Feature2DShare x_share1 = enc_to_share_client.decrypt_to_share(x_share1_enc, PackType::MultipleChannelPacking);

    ReluLayerClient relu(mpc::DEFAULT_SCALE_BIT, mpc::RING_MOD, 128.0);
    Feature2DShare y_share1(mpc::RING_MOD, mpc::DEFAULT_SCALE_BIT);
    y_share1.data = decltype(y_share1.data)::move_from_array_1d(
        relu.run(mpc::Array<uint64_t, 1>::from_array_1d(x_share1.data.to_array_1d())).move_to_array_1d());
    y_share1.shape = x_share1.shape;
    scale_share_by_t(y_share1);

    Feature2DEncrypted y_share1_enc(&context, kLevel, {1, 1}, {1, 1}, PackType::MultipleChannelPacking);
    ShareToEncClient share_to_enc_client(context);
    Array<uint64_t, 1> y_share1_add_mod =
        share_to_enc_client.encrypt_from_share(y_share1_enc, y_share1, kNChannel, kShape,
                                               PackType::MultipleChannelPacking);

    MPC mpc_protocol(mpc::DEFAULT_SCALE_BIT, mpc::RING_MOD, 128.0);
    auto b0 = mpc_protocol.wrap_protocol(y_share1_add_mod.to_array_1d(), ::mpc::current_party());

    Array<double, 1> b0_mult_mod_div_s({b0.size()});
    for (int i = 0; i < b0.size(); i++) {
        b0_mult_mod_div_s.set(i, double(b0[i] * mpc::RING_MOD) / mpc::DEFAULT_SCALE);
    }

    CkksContext& extra_context = context.get_extra_level_context();
    Feature2DEncrypted wrap_enc(&extra_context, kLevel + 1, {1, 1}, {1, 1}, PackType::MultipleChannelPacking);
    wrap_enc.pack_multiple_channel(b0_mult_mod_div_s.reshape<3>({kNChannel, kShape[0], kShape[1]}), false,
                                   mpc::DEFAULT_SCALE);

    dt.send_bytes(y_share1_enc.serialize());
    dt.send_bytes(wrap_enc.serialize());
    dt.flush();
}

void test_new_mpc_refresh_client(int port_in) {
    init_mpc_party(port_in);
    cout << "refresh client: RecodeBigComplex path" << endl;

    int N = 8192;
    CkksParameter param = CkksParameter::create_parameter(N);
    CkksContext context = CkksContext::create_random_context(param, MAX_LEVEL, true);
    context.gen_rotation_keys();

    DataTransmission dt = ::mpc::data_transmission();
    dt.send_bytes(context.serialize());

    Bytes recv_ct_bytes = dt.receive_bytes();
    CkksCiphertext recv_ct = CkksCiphertext::deserialize(recv_ct_bytes);

    CkksPlaintext recv_pt = context.decrypt(recv_ct);

    constexpr int refreshed_level = 3;
    double scale = context.get_parameter().get_default_scale();
    CkksPlaintext recode_pt = context.recode_big_complex(recv_pt, refreshed_level, scale);

    CkksCiphertext send_ct = context.encrypt_symmetric(recode_pt);
    // CkksPlaintext recode_pt = context.encode(context.decode(recv_pt),refreshed_level,scale);
    // CkksCiphertext send_ct = context.encrypt_symmetric(recode_pt);
    dt.send_bytes(send_ct.serialize(context.get_parameter()));
    dt.flush();
}


}  // namespace

int main(int argc, char** argv) {
    int port_in = get_test_port(argc, argv);
    if (argc > 1 && strcmp(argv[1], "refresh") == 0) {
        test_new_mpc_refresh_client(port_in);
        return 0;
    }

    run_relu_client(port_in);
    return 0;
}
