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

#include "fhe_mpc.h"
#include "SCI/src/globals.h"
#include "data_structs/feature.h"

#include <cstring>
#include <cstdlib>

using namespace std;
using namespace sci;
using namespace lattisense;

int party = CLIENT;
int port = 12309;
string address = "127.0.0.1";
int num_threads = 1;

namespace {

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
    return port;
}

void init_mpc_party(int port_in) {
    party = CLIENT;
    port = port_in;
    address = "127.0.0.1";
    num_threads = 1;
    bitlength = RING_MOD_BIT;
    StartComputation();
}

void scale_share_by_t(Feature2DShare& share) {
    for (int i = 0; i < share.data.get_size(); i++) {
        share.data.set(i, (share.data.get(i) * T_SCALE) % share.ring_mod);
    }
}

CkksContext make_test_context() {
    return CkksContext::create_random_context(CkksParameter::create_parameter(8192));
}

void run_relu_client(int port_in) {
    init_mpc_party(port_in);

    DataTransmission dt(io);
    CkksContext context = make_test_context();
    dt.send_public_context(context);

    Bytes x_share1_bytes = dt.receive_bytes();
    Feature2DEncrypted x_share1_enc(&context, kLevel, {1, 1}, {1, 1}, PackType::MultipleChannelPacking);
    x_share1_enc.deserialize(x_share1_bytes);

    Feature2DShare x_share1(RING_MOD, DEFAULT_SCALE_BIT);
    x_share1_enc.decrypt_to_share(&x_share1, PackType::MultipleChannelPacking);

    ReluLayerClient relu(DEFAULT_SCALE_BIT, RING_MOD, 128.0);
    Feature2DShare y_share1(RING_MOD, DEFAULT_SCALE_BIT);
    relu.run(x_share1, y_share1);
    scale_share_by_t(y_share1);

    Feature2DEncrypted y_share1_enc(&context, kLevel, {1, 1}, {1, 1}, PackType::MultipleChannelPacking);
    Array<uint64_t, 1> y_share1_add_mod =
        y_share1_enc.encrypt_from_share(y_share1, kNChannel, kShape, PackType::MultipleChannelPacking);

    MPC mpc(DEFAULT_SCALE_BIT, RING_MOD, 128.0);
    auto b0 = mpc.wrap_protocol(y_share1_add_mod.to_array_1d(), dt.io_in, otpack, party);

    Array<double, 1> b0_mult_mod_div_s({b0.size()});
    for (int i = 0; i < b0.size(); i++) {
        b0_mult_mod_div_s.set(i, double(b0[i] * RING_MOD) / DEFAULT_SCALE);
    }

    CkksContext& extra_context = context.get_extra_level_context();
    Feature2DEncrypted wrap_enc(&extra_context, kLevel + 1, {1, 1}, {1, 1}, PackType::MultipleChannelPacking);
    wrap_enc.pack_multiple_channel(b0_mult_mod_div_s.reshape<3>({kNChannel, kShape[0], kShape[1]}), false,
                                   DEFAULT_SCALE);

    dt.send_bytes(y_share1_enc.serialize());
    dt.send_bytes(wrap_enc.serialize());
    dt.io_in->flush();
}

void test_new_mpc_refresh_client(int port_in) {
    init_mpc_party(port_in);
    cout << "refresh client: RecodeBigComplex path" << endl;

    int N = 8192;
    CkksParameter param = CkksParameter::create_parameter(N);
    CkksContext context = CkksContext::create_random_context(param, MAX_LEVEL, true);
    context.gen_rotation_keys();

    DataTransmission dt(io);
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
    dt.io_in->flush();
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
