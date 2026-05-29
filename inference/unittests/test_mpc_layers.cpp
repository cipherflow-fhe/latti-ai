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
#include "fhe_mpc.h"
#include "SCI/src/globals.h"
#include "data_structs/feature.h"
#include "ut_util.h"

#include <cstdlib>

using namespace std;
using namespace sci;
using namespace lattisense;

int party = SERVER;
int port = 12309;
string address = "127.0.0.1";
int num_threads = 1;

namespace {

constexpr int kLevel = 5;
constexpr uint32_t kNChannel = 4;
const Duo kShape = {16, 16};

int get_test_port() {
    if (const char* env_port = getenv("MPC_TEST_PORT")) {
        return atoi(env_port);
    }
    return port;
}

void init_mpc_party(int port_in) {
    party = SERVER;
    port = port_in;
    address = "127.0.0.1";
    num_threads = 1;
    bitlength = RING_MOD_BIT;
    StartComputation();
}

Array<double, 3> relu_plaintext(const Array<double, 3>& input) {
    return input.apply([](double e) { return e > 0.0 ? e : 0.0; });
}

void scale_share_by_t(Feature2DShare& share) {
    for (int i = 0; i < share.data.get_size(); i++) {
        share.data.set(i, (share.data.get(i) * T_SCALE) % share.ring_mod);
    }
}

Array<double, 3> make_relu_input() {
    Array<double, 3> input({kNChannel, kShape[0], kShape[1]});
    for (uint32_t c = 0; c < kNChannel; c++) {
        for (uint32_t i = 0; i < kShape[0]; i++) {
            for (uint32_t j = 0; j < kShape[1]; j++) {
                double value = (static_cast<int>((c * 31 + i * 7 + j * 3) % 17) - 8) / 8.0;
                input.set(c, i, j, value);
            }
        }
    }
    input.set(0,0,0,-3);
    return input;
}

Array<double, 3> run_relu_server(int port_in) {
    init_mpc_party(port_in);

    DataTransmission dt(io);
    CkksContext context = dt.recv_public_context();
    double d_s =  context.get_parameter().get_default_scale();
    cout<<"d_s="<< d_s<<endl;
    auto input = make_relu_input();

    Feature2DEncrypted input_enc(&context, kLevel, {1, 1}, {1, 1}, PackType::MultipleChannelPacking);
    input_enc.pack_multiple_channel(input, false, DEFAULT_SCALE);

    Feature2DEncrypted x_share1_enc(&context, kLevel, {1, 1}, {1, 1}, PackType::MultipleChannelPacking);
    Feature2DShare x_share0(RING_MOD, DEFAULT_SCALE_BIT);
    input_enc.split_to_shares_for_multi_channel_pack(&x_share1_enc, &x_share0);
    dt.send_bytes(x_share1_enc.serialize());

    ReluLayerServer relu(DEFAULT_SCALE_BIT, RING_MOD, 128.0);
    Feature2DShare y_share0(RING_MOD, DEFAULT_SCALE_BIT);
    relu.run(x_share0, y_share0);
    scale_share_by_t(y_share0);

    MPC mpc(DEFAULT_SCALE_BIT, RING_MOD, 128.0);
    auto b1 = mpc.wrap_protocol(y_share0.data.to_array_1d(), dt.io_in, otpack, party);

    Bytes y_share1_bytes = dt.receive_bytes();
    Feature2DEncrypted y_share1_enc(&context, kLevel, {1, 1}, {1, 1}, PackType::MultipleChannelPacking);
    y_share1_enc.deserialize(y_share1_bytes);
    y_share1_enc.decompress();

    Bytes y_share2_bytes = dt.receive_bytes();
    CkksContext& extra_context = context.get_extra_level_context();
    Feature2DEncrypted y_share2_enc(&extra_context, kLevel + 1, {1, 1}, {1, 1}, PackType::MultipleChannelPacking);
    y_share2_enc.deserialize(y_share2_bytes);

    auto y_ct = y_share1_enc.combine_with_share_new_protocol_for_multi_pack(y_share0, y_share2_enc, b1);
    auto output = y_ct.unpack_multiple_channel();
    for (int i = 0; i < output.get_size(); i++) {
        output.set(i, output.get(i) / T_SCALE);
    }
    return output;
}

}  // namespace

TEST_CASE("relu_mpc_multiple_channel_pack", "[mpc][relu]") {
    auto output = run_relu_server(get_test_port());
    cout<<"res_ct="<<output.to_array_3d()[0][0][0]<<endl;
    auto expected = relu_plaintext(make_relu_input());
    cout<<"expected="<<expected.to_array_3d()[0][0][0]<<endl;
    auto compare_res = compare(expected, output);
    REQUIRE(compare_res.max_error < 5.0e-2 * max(compare_res.max_abs, 1.0));
}
