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

#include "mpc_adapter/enc_share_conversion.h"

#include <cmath>
#include <random>

#include "mpc/fhe_mpc.h"
#include "mpc_array_bridge.h"
#include "mpc/mpc_session.h"
#include "util.h"

using namespace std;
using namespace lattisense;

namespace {

vector<CkksPlaintext> multi_pack_to_pt(const Array<double, 3>& feature_mg,
                                       Feature2DEncrypted& f2d,
                                       int n_channel,
                                       Duo shape,
                                       Duo skip,
                                       CkksContext& context,
                                       int level,
                                       double scale_in,
                                       PackType pack_type) {
    CkksContext* old_context = f2d.context;
    int old_level = f2d.level;
    f2d.context = &context;
    f2d.level = level;

    vector<CkksPlaintext> pt_vec;
    if (pack_type == PackType::MultipleChannelPacking) {
        pt_vec = f2d.encode_multiple_channel(feature_mg, scale_in);
    } else if (pack_type == PackType::MultiplexedPacking) {
        pt_vec = f2d.encode_multiplexed(feature_mg, scale_in);
    } else {
        Duo block_expansion = {(uint32_t)ceil(shape[0] / (double)BLOCK_SHAPE[0]),
                               (uint32_t)ceil(shape[1] / (double)BLOCK_SHAPE[1])};
        pt_vec = f2d.encode_interleaved(feature_mg, BLOCK_SHAPE, block_expansion, scale_in);
    }

    f2d.context = old_context;
    f2d.level = old_level;
    return pt_vec;
}

}  // namespace

EncToShareClient::EncToShareClient(map<string, unique_ptr<CkksContext>>& ckks_contexts,
                                   CkksContext*& context_in,
                                   CkksContext*& context_out,
                                   int scale_ord,
                                   uint64_t ring_mod)
    : ckks_contexts_(ckks_contexts),
      context_in_(context_in),
      context_out_(context_out),
      scale_ord_(scale_ord),
      ring_mod_(ring_mod) {}

ShareToEncClient::ShareToEncClient(CkksContext& context_out, int scale_ord, uint64_t ring_mod, double pt_range)
    : context_out_(context_out), scale_ord_(scale_ord), ring_mod_(ring_mod), pt_range_(pt_range) {}

EncToShareServer::EncToShareServer(CkksContext& context, int scale_ord, uint64_t ring_mod)
    : context_(context), scale_ord_(scale_ord), ring_mod_(ring_mod) {}

ShareToEncServer::ShareToEncServer(CkksContext& context, int scale_ord, uint64_t ring_mod, double pt_range)
    : context_(context), scale_ord_(scale_ord), ring_mod_(ring_mod), pt_range_(pt_range) {}

Feature2DShare EncToShareClient::decrypt_to_share(const Feature2DEncrypted& x_enc, PackType pack_type) const {
    Feature2DShare share(ring_mod_, scale_ord_);
    share.shape = x_enc.shape;

    Array<double, 3> x_double_matrix;
    if (pack_type == PackType::MultiplexedPacking) {
        x_double_matrix = x_enc.unpack_multiplexed();
    } else if (pack_type == PackType::MultipleChannelPacking) {
        x_double_matrix = x_enc.unpack_multiple_channel();
    } else if (pack_type == PackType::InterleavedPacking) {
        Duo block_expansion = {(uint32_t)ceil(x_enc.shape[0] / (double)BLOCK_SHAPE[0]),
                               (uint32_t)ceil(x_enc.shape[1] / (double)BLOCK_SHAPE[1])};
        x_double_matrix = x_enc.unpack_interleaved(BLOCK_SHAPE, block_expansion);
    }

    share.data = array_double_to_uint64(x_double_matrix, share.scale_ord, share.ring_mod).reshape<1>({0});
    return share;
}

Feature2DShare EncToShareClient::decrypt_to_share_simple(const Feature2DEncrypted& x_enc, PackType pack_type) const {
    Feature2DShare share(ring_mod_, scale_ord_);
    share.shape = x_enc.shape;

    Array<double, 3> x_double_matrix;
    if (pack_type == PackType::MultiplexedPacking) {
        x_double_matrix = x_enc.unpack_multiplexed();
    } else if (pack_type == PackType::MultipleChannelPacking) {
        x_double_matrix = x_enc.unpack_multiple_channel();
    } else if (pack_type == PackType::InterleavedPacking) {
        Duo block_expansion = {(uint32_t)ceil(x_enc.shape[0] / (double)BLOCK_SHAPE[0]),
                               (uint32_t)ceil(x_enc.shape[1] / (double)BLOCK_SHAPE[1])};
        x_double_matrix = x_enc.unpack_interleaved(BLOCK_SHAPE, block_expansion);
    }

    share.data_double.resize({x_double_matrix.get_size()});
    for (int i = 0; i < x_double_matrix.get_size(); i++) {
        share.data_double.set(i, x_double_matrix.get_data()[i]);
    }
    return share;
}

Feature0DShare EncToShareClient::decrypt_to_share(const Feature0DEncrypted& x_enc, int n_channel) const {
    (void)n_channel;
    Feature0DShare share(ring_mod_, scale_ord_);
    Array<double, 1> x_double_vec = x_enc.unpack();
    share.data = array_double_to_uint64(x_double_vec, share.scale_ord, share.ring_mod);
    return share;
}

void EncToShareServer::split_to_shares(const Feature2DEncrypted& x_enc,
                                       Feature2DEncrypted* share0,
                                       Feature2DShare* share1) const {
    int n_slot = x_enc.context->get_parameter().get_n() / 2;
    double share_scale = DEFAULT_SCALE;
    int feature_bitlength = DEFAULT_SCALE_BIT + 1;
    int sigma = SIGMA;

    Duo pre_skip_shape = x_enc.shape * x_enc.skip;
    size_t n_share_feature = x_enc.n_channel * x_enc.shape[0] * x_enc.shape[1];
    size_t n_mask = x_enc.n_channel * pre_skip_shape[0] * pre_skip_shape[1];

    vector<double> mask_d(n_mask);
    vector<int64_t> r(n_mask);
    for (int i = 0; i < n_mask; i++) {
        r[i] = int64_t(gen_random_uint(feature_bitlength + sigma)) - int64_t(1ull << (feature_bitlength + sigma - 1));
        mask_d[i] = double(r[i]) / share_scale;
    }

    share0->n_channel = x_enc.n_channel;
    share0->n_channel_per_ct = x_enc.n_channel_per_ct;
    share0->shape = x_enc.shape;
    share0->skip = x_enc.skip;
    share0->level = x_enc.level;
    share0->data.clear();
    vector<double> mask_d_span(mask_d);
    for (int i = 0; i < x_enc.data.size(); i++) {
        size_t start = i * n_slot;
        size_t length = i == x_enc.data.size() - 1 ? (mask_d_span.size() - start) : n_slot;
        vector<double> mask_mg_vec(mask_d_span.begin() + start, mask_d_span.begin() + start + length);
        CkksPlaintext mask_pt = x_enc.context->encode(mask_mg_vec, x_enc.level, DEFAULT_SCALE);
        CkksCiphertext share0_ct = x_enc.context->add_plain(x_enc.data[i], mask_pt);
        share0->data.push_back(move(share0_ct));
    }

    share1->shape = x_enc.shape;
    share1->data.resize({n_share_feature});
    for (int i = 0; i < x_enc.n_channel; i++) {
        for (int j = 0; j < x_enc.shape[0]; j++) {
            for (int k = 0; k < x_enc.shape[1]; k++) {
                int skipped_index = i * x_enc.shape[0] * x_enc.shape[1] + j * x_enc.shape[1] + k;
                int pre_skip_index = i * pre_skip_shape[0] * pre_skip_shape[1] +
                                     j * pre_skip_shape[1] * x_enc.skip[0] + k * x_enc.skip[1];
                share1->data[skipped_index] =
                    (-int64_t(r[pre_skip_index]) % share1->ring_mod + share1->ring_mod) % share1->ring_mod;
            }
        }
    }
}

void EncToShareServer::split_to_shares_simple(const Feature2DEncrypted& x_enc,
                                              Feature2DEncrypted* share0,
                                              Feature2DShare* share1) const {
    int n_slot = x_enc.context->get_parameter().get_n() / 2;
    int sigma = SIGMA;

    Duo pre_skip_shape = x_enc.shape * x_enc.skip;
    size_t n_share_feature = x_enc.n_channel * x_enc.shape[0] * x_enc.shape[1];
    size_t n_mask = x_enc.n_channel * pre_skip_shape[0] * pre_skip_shape[1];

    vector<double> mask_d(n_mask);
    vector<double> r(n_mask);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(-pow(2, DATA_BIT + sigma), pow(2, DATA_BIT + sigma));
    for (int i = 0; i < n_mask; i++) {
        r[i] = dis(gen);
        mask_d[i] = -r[i];
    }

    share0->n_channel = x_enc.n_channel;
    share0->n_channel_per_ct = x_enc.n_channel_per_ct;
    share0->shape = x_enc.shape;
    share0->skip = x_enc.skip;
    share0->level = x_enc.level;
    share0->data.clear();
    vector<double> mask_d_span(mask_d);
    for (int i = 0; i < x_enc.data.size(); i++) {
        size_t start = i * n_slot;
        size_t length = i == x_enc.data.size() - 1 ? (mask_d_span.size() - start) : n_slot;
        vector<double> mask_mg_vec(mask_d_span.begin() + start, mask_d_span.begin() + start + length);
        CkksPlaintext mask_pt = x_enc.context->encode(mask_mg_vec, x_enc.level, DEFAULT_SCALE);
        CkksCiphertext share0_ct = x_enc.context->add_plain(x_enc.data[i], mask_pt);
        share0->data.push_back(move(share0_ct));
    }

    share1->shape = x_enc.shape;
    share1->data_double.resize({n_share_feature});
    for (int i = 0; i < x_enc.n_channel; i++) {
        for (int j = 0; j < x_enc.shape[0]; j++) {
            for (int k = 0; k < x_enc.shape[1]; k++) {
                int skipped_index = i * x_enc.shape[0] * x_enc.shape[1] + j * x_enc.shape[1] + k;
                int pre_skip_index = i * pre_skip_shape[0] * pre_skip_shape[1] +
                                     j * pre_skip_shape[1] * x_enc.skip[0] + k * x_enc.skip[1];
                share1->data_double[skipped_index] = r[pre_skip_index];
            }
        }
    }
}

void EncToShareServer::split_to_shares_for_multi_channel_pack(const Feature2DEncrypted& x_enc,
                                                              Feature2DEncrypted* share0,
                                                              Feature2DShare* share1,
                                                              PackType pack_type) const {
    double share_scale = DEFAULT_SCALE;
    int feature_bitlength = DEFAULT_SCALE_BIT + 1;
    int sigma = SIGMA;
    size_t n_mask = x_enc.n_channel * x_enc.shape[0] * x_enc.shape[1];

    vector<double> mask_d(n_mask);
    vector<int64_t> r(n_mask);
    for (int i = 0; i < n_mask; i++) {
        r[i] = int64_t(gen_random_uint(feature_bitlength + sigma)) - int64_t(1ull << (feature_bitlength + sigma - 1));
        mask_d[i] = double(r[i]) / share_scale;
    }
    share0->n_channel = x_enc.n_channel;
    share0->n_channel_per_ct = x_enc.n_channel_per_ct;
    share0->shape = x_enc.shape;
    share0->skip = x_enc.skip;
    share0->level = x_enc.level;
    share0->data.clear();
    auto mask_d_array = Array<double, 1>::from_array_1d(mask_d).reshape<3>(
        {x_enc.n_channel, x_enc.shape[0], x_enc.shape[1]});
    auto mask_pt = multi_pack_to_pt(mask_d_array, *share0, x_enc.n_channel, x_enc.shape, x_enc.skip, *x_enc.context,
                                    x_enc.level, DEFAULT_SCALE, pack_type);
    for (int i = 0; i < x_enc.data.size(); i++) {
        CkksCiphertext share0_ct = x_enc.context->add_plain(x_enc.data[i], mask_pt[i]);
        share0->data.push_back(move(share0_ct));
    }

    share1->shape = x_enc.shape;
    share1->data.resize({n_mask});
    for (int i = 0; i < n_mask; i++) {
        share1->data[i] = (-int64_t(r[i]) % share1->ring_mod + share1->ring_mod) % share1->ring_mod;
    }
}

void EncToShareServer::split_to_shares_for_multi_channel_pack_simple(const Feature2DEncrypted& x_enc,
                                                                     Feature2DEncrypted* share0,
                                                                     Feature2DShare* share1,
                                                                     PackType pack_type) const {
    int sigma = SIGMA;
    size_t n_share_feature = x_enc.n_channel * x_enc.shape[0] * x_enc.shape[1];

    vector<double> mask_d(n_share_feature);
    vector<double> r(n_share_feature);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(-pow(2, DATA_BIT + sigma), pow(2, DATA_BIT + sigma));
    for (int i = 0; i < n_share_feature; i++) {
        r[i] = dis(gen);
        mask_d[i] = -r[i];
    }

    share0->n_channel = x_enc.n_channel;
    share0->n_channel_per_ct = x_enc.n_channel_per_ct;
    share0->shape = x_enc.shape;
    share0->skip = x_enc.skip;
    share0->level = x_enc.level;
    share0->data.clear();
    share0->packing_type = pack_type;

    auto mask_d_array = Array<double, 1>::from_array_1d(mask_d).reshape<3>(
        {x_enc.n_channel, x_enc.shape[0], x_enc.shape[1]});
    auto mask_pt = multi_pack_to_pt(mask_d_array, *share0, x_enc.n_channel, x_enc.shape, x_enc.skip, *x_enc.context,
                                    x_enc.level, DEFAULT_SCALE, pack_type);
    for (int i = 0; i < x_enc.data.size(); i++) {
        CkksCiphertext share0_ct = x_enc.context->add_plain(x_enc.data[i], mask_pt[i]);
        share0->data.push_back(move(share0_ct));
    }

    share1->shape = x_enc.shape;
    share1->data_double.resize({n_share_feature});
    for (int i = 0; i < n_share_feature; i++) {
        share1->data_double[i] = r[i];
    }
}

void EncToShareServer::split_to_shares(const Feature0DEncrypted& x_enc,
                                       Feature0DEncrypted* share0,
                                       Feature0DShare* share1) const {
    int n_slot = x_enc.context->get_parameter().get_n() / 2;
    double share_scale = pow(2, share1->scale_ord);
    int feature_bitlength = DEFAULT_SCALE_BIT + 1;
    int sigma = SIGMA;
    share0->n_channel = x_enc.n_channel;
    share0->n_channel_per_ct = x_enc.n_channel_per_ct;
    share0->skip = x_enc.skip;
    share0->level = x_enc.level;
    share0->data.clear();
    vector<vector<double>> mask_d_mat;
    vector<vector<int64_t>> r_mat;
    for (int i = 0; i < x_enc.data.size(); i++) {
        vector<double> mask_d(n_slot);
        vector<int64_t> r(n_slot);
        for (int j = 0; j < n_slot; j++) {
            r[j] =
                int64_t(gen_random_uint(feature_bitlength + sigma)) - int64_t(1ull << (feature_bitlength + sigma - 1));
            mask_d[j] = double(r[j]) / share_scale;
        }
        mask_d_mat.push_back(mask_d);
        r_mat.push_back(r);
        CkksPlaintext mask_pt = x_enc.context->encode(mask_d, x_enc.level, DEFAULT_SCALE);
        CkksCiphertext share0_ct = x_enc.context->add_plain(x_enc.data[i], mask_pt);

        share0->data.push_back(move(share0_ct));
    }

    share1->data.resize({x_enc.n_channel});
    int T = 0;
    for (int i = 0; i < mask_d_mat.size(); i++) {
        for (int j = 0; j < x_enc.n_channel_per_ct; j++) {
            if (T >= x_enc.n_channel) {
                break;
            }
            uint64_t neg_r = (-r_mat[i][j * x_enc.skip] % share1->ring_mod + share1->ring_mod) % share1->ring_mod;
            share1->data.set(T, neg_r);
            T = T + 1;
        }
    }
}

void EncToShareServer::split_to_shares_reshape(const Feature0DEncrypted& x_enc,
                                               Feature0DEncrypted* share0,
                                               Feature0DShare* share1) const {
    int n_slot = x_enc.context->get_parameter().get_n() / 2;
    double share_scale = pow(2, share1->scale_ord);
    int feature_bitlength = DEFAULT_SCALE_BIT + 1;
    int sigma = SIGMA;
    share0->n_channel = x_enc.n_channel;
    share0->n_channel_per_ct = x_enc.n_channel_per_ct;
    share0->skip = x_enc.skip;
    share0->level = x_enc.level;
    share0->data.clear();
    vector<vector<double>> mask_d_mat;
    vector<vector<int64_t>> r_mat;
    for (int i = 0; i < x_enc.data.size(); i++) {
        vector<double> mask_d(n_slot);
        vector<int64_t> r(n_slot);
        for (int j = 0; j < n_slot; j++) {
            r[j] =
                int64_t(gen_random_uint(feature_bitlength + sigma)) - int64_t(1ull << (feature_bitlength + sigma - 1));
            mask_d[j] = double(r[j]) / share_scale;
        }
        r_mat.push_back(r);
        mask_d_mat.push_back(mask_d);
        CkksPlaintext mask_pt = x_enc.context->encode(mask_d, x_enc.level, DEFAULT_SCALE);
        CkksCiphertext share0_ct = x_enc.context->add_plain(x_enc.data[i], mask_pt);
        share0->data.push_back(move(share0_ct));
    }

    share1->data.resize({x_enc.n_channel});
    int T = 0;

    for (int i = 0; i < mask_d_mat.size(); i++) {
        for (int j = 0; j < div_ceil(x_enc.n_channel, x_enc.data.size()); j++) {
            if (T >= x_enc.n_channel) {
                break;
            }
            uint64_t neg_r = (-r_mat[i][j * x_enc.skip] % share1->ring_mod + share1->ring_mod) % share1->ring_mod;
            share1->data.set(T, neg_r);
            T += 1;
        }
    }
}

Feature2DShare EncToShareClient::client_enc_to_share(const Bytes& meta_data_bytes) {
    DataTransmission data_trans = ::mpc::data_transmission();

    uint8_t level;
    uint8_t param_id_in;
    uint8_t param_id_out;
    bytes_to_va(meta_data_bytes, {"u8", "u8", "u8"}, &level, &param_id_in, &param_id_out);

    string param_in = param_to_string(param_id_in);
    string param_out = param_to_string(param_id_out);
    context_in_ = ckks_contexts_[param_in].get();
    context_out_ = ckks_contexts_[param_out].get();

    Bytes x_e_bytes = data_trans.receive_bytes();
    Feature2DEncrypted x_e(context_in_, level);
    x_e.deserialize(x_e_bytes);
    return decrypt_to_share(x_e, PackType::MultipleChannelPacking);
}

Feature2DShare EncToShareClient::client_enc_to_share_simple(const Bytes& meta_data_bytes) {
    DataTransmission data_trans = ::mpc::data_transmission();

    uint8_t level;
    uint8_t param_id_in;
    uint8_t param_id_out;
    bytes_to_va(meta_data_bytes, {"u8", "u8", "u8"}, &level, &param_id_in, &param_id_out);

    string param_in = param_to_string(param_id_in);
    string param_out = param_to_string(param_id_out);
    context_in_ = ckks_contexts_[param_in].get();
    context_out_ = ckks_contexts_[param_out].get();

    Bytes x_e_bytes = data_trans.receive_bytes();
    Feature2DEncrypted x_e(context_in_, level);
    x_e.deserialize(x_e_bytes);
    return decrypt_to_share_simple(x_e, PackType::MultipleChannelPacking);
}

Feature2DShare EncToShareClient::client_enc_to_share_for_multi_channel_pack(const Bytes& meta_data_bytes) {
    DataTransmission data_trans = ::mpc::data_transmission();
    uint8_t level;
    uint8_t param_id_in;
    uint8_t param_id_out;
    uint8_t temp_int = 0;
    bytes_to_va(meta_data_bytes, {"u8", "u8", "u8", "u8"}, &level, &param_id_in, &param_id_out, &temp_int);
    PackType pack_type = (PackType)temp_int;

    string param_in = param_to_string(param_id_in);
    string param_out = param_to_string(param_id_out);
    context_in_ = ckks_contexts_[param_in].get();
    context_out_ = ckks_contexts_[param_out].get();

    Bytes x_e_bytes = data_trans.receive_bytes();
    Feature2DEncrypted x_e(context_in_, level);
    x_e.deserialize(x_e_bytes);
    return decrypt_to_share(x_e, pack_type);
}

Feature2DShare EncToShareClient::client_enc_to_share_for_multi_channel_pack_simple(const Bytes& meta_data_bytes) {
    DataTransmission data_trans = ::mpc::data_transmission();
    uint8_t level;
    uint8_t param_id_in;
    uint8_t param_id_out;
    uint8_t temp_int = 0;
    bytes_to_va(meta_data_bytes, {"u8", "u8", "u8", "u8"}, &level, &param_id_in, &param_id_out, &temp_int);
    PackType pack_type = (PackType)temp_int;

    string param_in = param_to_string(param_id_in);
    string param_out = param_to_string(param_id_out);
    context_in_ = ckks_contexts_[param_in].get();
    context_out_ = ckks_contexts_[param_out].get();

    Bytes x_e_bytes = data_trans.receive_bytes();
    Feature2DEncrypted x_e(context_in_, level);
    x_e.deserialize(x_e_bytes);
    return decrypt_to_share_simple(x_e, pack_type);
}

Feature0DShare EncToShareClient::client_enc_to_share_0d(const Bytes& meta_data_bytes) {
    DataTransmission data_trans = ::mpc::data_transmission();
    uint8_t level;
    uint32_t n_channal;
    uint8_t param_in_id;
    uint8_t param_out_id;

    bytes_to_va(meta_data_bytes, {"u8", "u32", "u8", "u8"}, &level, &n_channal, &param_in_id, &param_out_id);
    string param_in = param_to_string(param_in_id);
    string param_out = param_to_string(param_out_id);
    context_in_ = ckks_contexts_.at(param_in).get();
    context_out_ = ckks_contexts_.at(param_out).get();

    Bytes x_e_bytes = data_trans.receive_bytes();
    if (context_in_ == nullptr) {
        cout << "wrong ptr" << endl;
    }
    Feature0DEncrypted x_e(context_in_, level);
    x_e.deserialize(x_e_bytes);
    return decrypt_to_share(x_e, n_channal);
}

void ShareToEncClient::client_share_to_enc(Feature2DShare& share, const Bytes& meta_data_bytes) {
    DataTransmission data_trans = ::mpc::data_transmission();
    uint8_t level;
    uint32_t n_channel;
    bytes_to_va(meta_data_bytes, {"u8", "u32"}, &level, &n_channel);

    Feature2DEncrypted x_e(&context_out_, level, {1, 1}, {1, 1}, PackType::MultipleChannelPacking);
    for (int i = 0; i < share.data.get_size(); i++) {
        share.data.set(i, share.data.get(i) * T_SCALE % ring_mod_);
    }
    auto data_process = x_e.encrypt_from_share(share, n_channel, share.shape, PackType::MultipleChannelPacking);

    MPC mpc(scale_ord_, ring_mod_, pt_range_);
    auto b0 = mpc.wrap_protocol(data_process.to_array_1d(), ::mpc::current_party());

    Array<double, 1> b0_mult_mod_div_s_mg(share.data.get_shape());
    double scale = DEFAULT_SCALE;
    for (int i = 0; i < b0.size(); i++) {
        double temp_res = double(b0[i] * ring_mod_) / scale;
        b0_mult_mod_div_s_mg.set(i, temp_res);
    }
    CkksContext& ctx_extra = context_out_.get_extra_level_context();
    Feature2DEncrypted send_ct(&ctx_extra, level + 1, {1, 1}, {1, 1}, PackType::MultipleChannelPacking);
    send_ct.pack_multiple_channel(
        b0_mult_mod_div_s_mg.reshape<3>({(uint64_t)n_channel, (uint64_t)share.shape[0], (uint64_t)share.shape[1]}),
        false, DEFAULT_SCALE);
    data_trans.send_bytes(x_e.serialize());
    data_trans.send_bytes(send_ct.serialize());
}

void ShareToEncClient::client_share_to_enc_simple(Feature2DShare& share, const Bytes& meta_data_bytes) {
    DataTransmission data_trans = ::mpc::data_transmission();
    uint8_t level;
    uint32_t n_channel;
    bytes_to_va(meta_data_bytes, {"u8", "u32"}, &level, &n_channel);

    Feature2DEncrypted x_e(&context_out_, level, {1, 1}, {1, 1}, PackType::MultipleChannelPacking);
    x_e.encrypt_from_share_simple(share, n_channel, share.shape, PackType::MultipleChannelPacking,
                                  MPC_REFRESH_USE_RECODE);

    data_trans.send_bytes(x_e.serialize());
}

void ShareToEncClient::client_share_to_enc_for_multi_channel_pack(Feature2DShare& share,
                                                                  const Bytes& meta_data_bytes) {
    DataTransmission data_trans = ::mpc::data_transmission();
    uint8_t level;
    uint32_t n_channel;
    Duo skip;

    uint8_t temp_int = 0;
    bytes_to_va(meta_data_bytes, {"u8", "u32", "duo", "u8"}, &level, &n_channel, &skip, &temp_int);
    PackType pack_type = (PackType)temp_int;
    Feature2DEncrypted x_e(&context_out_, level, skip);
    for (int i = 0; i < share.data.get_size(); i++) {
        share.data.set(i, share.data.get(i) * T_SCALE % ring_mod_);
    }
    auto data_process = x_e.encrypt_from_share(share, n_channel, share.shape, pack_type);

    MPC mpc(scale_ord_, ring_mod_, pt_range_);
    auto b0 = mpc.wrap_protocol(data_process.to_array_1d(), ::mpc::current_party());

    Array<double, 1> b0_mult_mod_div_s_mg(share.data.get_shape());
    double scale = DEFAULT_SCALE;
    for (int i = 0; i < b0.size(); i++) {
        double temp_res = double(b0[i] * ring_mod_) / scale;
        b0_mult_mod_div_s_mg.set(i, temp_res);
    }
    CkksContext& ctx_extra = context_out_.get_extra_level_context();
    Feature2DEncrypted send_ct(&ctx_extra, level + 1, skip, {1, 1}, pack_type);
    auto send_mg =
        b0_mult_mod_div_s_mg.reshape<3>({(uint64_t)n_channel, (uint64_t)share.shape[0], (uint64_t)share.shape[1]});
    if (pack_type == PackType::MultipleChannelPacking) {
        send_ct.pack_multiple_channel(send_mg, false, DEFAULT_SCALE);
    } else if (pack_type == PackType::MultiplexedPacking) {
        send_ct.pack_multiplexed(send_mg, false, DEFAULT_SCALE);
    } else if (pack_type == PackType::InterleavedPacking) {
        Duo block_expansion = {(uint32_t)ceil(share.shape[0] / (double)BLOCK_SHAPE[0]),
                               (uint32_t)ceil(share.shape[1] / (double)BLOCK_SHAPE[1])};
        send_ct.pack_interleaved(send_mg, BLOCK_SHAPE, block_expansion, false, DEFAULT_SCALE);
    }
    data_trans.send_bytes(x_e.serialize());
    data_trans.send_bytes(send_ct.serialize());
}

void ShareToEncClient::client_share_to_enc_for_multi_channel_pack_simple(Feature2DShare& share,
                                                                         const Bytes& meta_data_bytes) {
    DataTransmission data_trans = ::mpc::data_transmission();
    uint8_t level;
    uint32_t n_channel;
    Duo skip;

    uint8_t temp_int = 0;
    bytes_to_va(meta_data_bytes, {"u8", "u32", "duo", "u8"}, &level, &n_channel, &skip, &temp_int);
    PackType pack_type = (PackType)temp_int;
    Feature2DEncrypted x_e(&context_out_, level, skip, {1, 1}, pack_type);
    x_e.encrypt_from_share_simple(share, n_channel, share.shape, pack_type, MPC_REFRESH_USE_RECODE);

    data_trans.send_bytes(x_e.serialize());
}

void ShareToEncClient::client_share_to_enc_0d(Feature0DShare& share, const Bytes& meta_data_bytes) {
    DataTransmission data_trans = ::mpc::data_transmission();
    uint8_t level = 0;
    uint32_t n_channel = 0;
    bytes_to_va(meta_data_bytes, {"u8", "u32"}, &level, &n_channel);
    Feature0DEncrypted x_e(&context_out_, level);
    x_e.skip = 1;
    for (int i = 0; i < share.data.get_size(); i++) {
        share.data.set(i, share.data.get(i) * T_SCALE % ring_mod_);
    }
    auto data_process = x_e.encrypt_from_share(share, n_channel);
    MPC mpc(scale_ord_, ring_mod_, pt_range_);
    data_trans.flush();
    auto b0 = mpc.wrap_protocol(data_process.to_array_1d(), ::mpc::current_party());
    data_trans.flush();

    Array<double, 1> send_mg(share.data.get_shape());
    double scale = DEFAULT_SCALE;
    for (int i = 0; i < b0.size(); i++) {
        double temp_res = double(b0[i] * ring_mod_) / scale;
        send_mg.set(i, temp_res);
    }
    CkksContext& ctx_extra = context_out_.get_extra_level_context();
    Feature0DEncrypted send_ct(&ctx_extra, level + 1);
    send_ct.skip = 1;
    double encode_scale = pow(2, DEFAULT_SCALE_BIT);
    send_ct.pack_cyclic(send_mg.to_array_1d(), false, encode_scale);
    data_trans.send_bytes(x_e.serialize());
    data_trans.send_bytes(send_ct.serialize());
}

Feature2DShare EncToShareServer::server_enc_to_share_multi_pack(const Feature2DEncrypted& x_enc,
                                                                 PackType pack_type) {
    DataTransmission data_trans = ::mpc::data_transmission();

    Feature2DEncrypted x_share1_enc(&context_, x_enc.level, x_enc.skip, x_enc.invalid_fill, pack_type);
    Feature2DShare x_share0(ring_mod_, scale_ord_);

    split_to_shares_for_multi_channel_pack(x_enc, &x_share1_enc, &x_share0, pack_type);
    data_trans.send_bytes(x_share1_enc.serialize());

    return x_share0;
}

Feature2DShare EncToShareServer::server_enc_to_share_multi_pack_simple(const Feature2DEncrypted& x_enc,
                                                                       PackType pack_type) {
    DataTransmission data_trans = ::mpc::data_transmission();

    Feature2DEncrypted x_share1_enc(&context_, x_enc.level, x_enc.skip, x_enc.invalid_fill, pack_type);
    Feature2DShare x_share0(ring_mod_, scale_ord_);

    split_to_shares_for_multi_channel_pack_simple(x_enc, &x_share1_enc, &x_share0, pack_type);
    data_trans.send_bytes(x_share1_enc.serialize());

    return x_share0;
}

Feature2DEncrypted ShareToEncServer::server_share_to_enc_multi_pack(Feature2DShare& y_share0,
                                                                    int level,
                                                                    PackType pack_type) {
    DataTransmission data_trans = ::mpc::data_transmission();
    for (int i = 0; i < y_share0.data.get_size(); i++) {
        y_share0.data.set(i, (y_share0.data.get(i) * T_SCALE) % ring_mod_);
    }

    MPC mpc(scale_ord_, ring_mod_, pt_range_);
    auto b1 = mpc.wrap_protocol(y_share0.data.to_array_1d(), SERVER);

    Feature2DEncrypted y_share1_enc(&context_, level, {1, 1}, {1, 1}, pack_type);
    y_share1_enc.deserialize(data_trans.receive_bytes());
    y_share1_enc.packing_type = pack_type;
    y_share1_enc.decompress();

    CkksContext& extra_context = context_.get_extra_level_context();
    Feature2DEncrypted y_share2_enc(&extra_context, level + 1, y_share1_enc.skip, {1, 1}, pack_type);
    y_share2_enc.deserialize(data_trans.receive_bytes());
    y_share2_enc.packing_type = pack_type;

    Feature2DEncrypted y_ct =
        y_share1_enc.combine_with_share_new_protocol_for_multi_pack(y_share0, y_share2_enc, b1, pack_type);
    y_ct.packing_type = pack_type;
    return y_ct;
}

Feature2DEncrypted ShareToEncServer::server_share_to_enc_multi_pack_simple(Feature2DShare& y_share0,
                                                                           int level,
                                                                           PackType pack_type) {
    DataTransmission data_trans = ::mpc::data_transmission();
    Feature2DEncrypted y_share1_enc(&context_, level, {1, 1}, {1, 1}, pack_type);
    y_share1_enc.deserialize(data_trans.receive_bytes());
    y_share1_enc.packing_type = pack_type;
    y_share1_enc.decompress();

    Feature2DEncrypted y_ct = y_share1_enc.combine_with_share_simple_for_multi_pack(y_share0, pack_type);
    y_ct.packing_type = pack_type;
    return y_ct;
}

Feature2DEncrypted ShareToEncServer::server_share_to_enc_simple(Feature2DShare& y_share0, int level) {
    DataTransmission data_trans = ::mpc::data_transmission();
    Feature2DEncrypted y_share1_enc(&context_, level, {1, 1}, {1, 1}, PackType::MultipleChannelPacking);
    y_share1_enc.deserialize(data_trans.receive_bytes());
    y_share1_enc.packing_type = PackType::MultipleChannelPacking;
    y_share1_enc.decompress();

    Feature2DEncrypted y_ct = y_share1_enc.combine_with_share_simple(y_share0);
    y_ct.packing_type = PackType::MultipleChannelPacking;
    return y_ct;
}

Feature2DShare EncToShareServer::server_enc_to_share(const Feature2DEncrypted& x_enc) {
    DataTransmission data_trans = ::mpc::data_transmission();

    Feature2DEncrypted x_share1_enc(&context_, x_enc.level, x_enc.skip, x_enc.invalid_fill,
                                    PackType::MultipleChannelPacking);
    Feature2DShare x_share0(ring_mod_, scale_ord_);

    split_to_shares(x_enc, &x_share1_enc, &x_share0);
    data_trans.send_bytes(x_share1_enc.serialize());

    return x_share0;
}

Feature2DShare EncToShareServer::server_enc_to_share_simple(const Feature2DEncrypted& x_enc) {
    DataTransmission data_trans = ::mpc::data_transmission();

    Feature2DEncrypted x_share1_enc(&context_, x_enc.level, x_enc.skip, x_enc.invalid_fill,
                                    PackType::MultipleChannelPacking);
    Feature2DShare x_share0(ring_mod_, scale_ord_);

    split_to_shares_simple(x_enc, &x_share1_enc, &x_share0);
    data_trans.send_bytes(x_share1_enc.serialize());

    return x_share0;
}

Feature0DShare EncToShareServer::server_enc_to_share(const Feature0DEncrypted& x_enc) {
    DataTransmission data_trans = ::mpc::data_transmission();

    Feature0DShare x_share0(ring_mod_, scale_ord_);
    Feature0DEncrypted x_share1_enc(&context_, x_enc.level);

    split_to_shares(x_enc, &x_share1_enc, &x_share0);
    data_trans.send_bytes(x_share1_enc.serialize());

    return x_share0;
}

Feature0DEncrypted ShareToEncServer::share_to_enc(Feature0DShare& y_share0, int level) {
    DataTransmission data_trans = ::mpc::data_transmission();
    for (int i = 0; i < y_share0.data.get_size(); i++) {
        y_share0.data.set(i, (y_share0.data.get(i) * T_SCALE) % ring_mod_);
    }

    MPC mpc(scale_ord_, ring_mod_, pt_range_);
    auto b1 = mpc.wrap_protocol(y_share0.data.to_array_1d(), SERVER);

    Feature0DEncrypted y_share1_enc(&context_, level);
    y_share1_enc.deserialize(data_trans.receive_bytes());
    y_share1_enc.decompress();

    CkksContext& extra_context = context_.get_extra_level_context();
    Feature0DEncrypted y_share2_enc(&extra_context, level + 1);
    y_share2_enc.deserialize(data_trans.receive_bytes());

    return y_share1_enc.combine_with_share_new_protocol(y_share0, y_share2_enc, b1);
}
