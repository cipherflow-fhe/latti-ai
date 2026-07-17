#include "mpc_wrapper/inference_process_mpc_server.h"

#include <iostream>

#include "mpc_wrapper/enc_share_conversion.h"
#include "mpc/fhe_mpc.h"
#include "mpc_array_bridge.h"

using namespace std;
using namespace lattisense;

namespace {

const vector<MpcProtoType> REFRESH_MPC = {MpcProtoType::enc_to_share_for_multi_channel_pack_simple,
                                          MpcProtoType::share_to_enc_for_multi_channel_pack_simple};

const vector<MpcProtoType> RELU2D_MPC = {MpcProtoType::enc_to_share_for_multi_channel_pack, MpcProtoType::relu,
                                         MpcProtoType::share_to_enc_for_multi_channel_pack};

const vector<MpcProtoType> MAXPOOL2D_MPC = {MpcProtoType::enc_to_share_for_multi_channel_pack,
                                            MpcProtoType::max_pool,
                                            MpcProtoType::share_to_enc_for_multi_channel_pack};

uint8_t ckks_parameter_id_to_u8(const string& param_id) {
    if (param_id.rfind("param", 0) == 0) {
        return static_cast<uint8_t>(stoi(param_id.substr(5)));
    }
    return static_cast<uint8_t>(stoi(param_id));
}

}  // namespace

void InitMpc::init_mpc_layer(const InitInferenceProcess& init,
                             const vector<MpcProtoType>& operations,
                             const json& layer,
                             MpcTaskMetaData& meta_data) {
    FeatureNode feature_input0(init.json_features[layer["feature_input"][0].get<string>()]);
    FeatureNode feature_output0(init.json_features[layer["feature_output"][0].get<string>()]);
    PackType pack_type = PackType::MultipleChannelPacking;
    if (init.pack_style == "multiplexed") {
        pack_type = choose_pack_type(feature_input0.shape, init.block_shape);
    }

    uint8_t param_id_in = ckks_parameter_id_to_u8(feature_input0.ckks_parameter_id);
    uint8_t param_id_out = ckks_parameter_id_to_u8(feature_output0.ckks_parameter_id);

    for (MpcProtoType operation : operations) {
        if (operation == MpcProtoType::enc_to_share_for_multi_channel_pack ||
            operation == MpcProtoType::enc_to_share_for_multi_channel_pack_simple) {
            meta_data.append(operation, {"u8", "u8", "u8", "u8"}, (uint8_t)feature_input0.level, param_id_in,
                             param_id_out, (uint8_t)pack_type);
        } else if (operation == MpcProtoType::share_to_enc_for_multi_channel_pack ||
                   operation == MpcProtoType::share_to_enc_for_multi_channel_pack_simple) {
            meta_data.append(operation, {"u8", "u32", "duo", "u8"}, (uint8_t)feature_output0.level,
                             feature_output0.channel, &feature_output0.skip, (uint8_t)pack_type);
        } else if (operation == MpcProtoType::relu) {
            meta_data.append(operation, {});
        } else if (operation == MpcProtoType::max_pool) {
            Duo kernel_shape = {layer["kernel_shape"][0], layer["kernel_shape"][1]};
            Duo stride = {layer["stride"][0], layer["stride"][1]};
            meta_data.append(operation, {"duo", "duo"}, &kernel_shape, &stride);
        } else {
            throw runtime_error("unsupported mpc operation for init_mpc_layer");
        }
    }
}

void InitMpc::init_mpc_refresh_layer(const InitInferenceProcess& init, const string& key, const json& layer) {
    MpcTaskMetaData meta_data;
    init_mpc_layer(init, REFRESH_MPC, layer, meta_data);
    meta_data_map_[key] = MakeU<MpcTaskMetaData>(move(meta_data));
    ckks_mpc_refresh_[key] = MakeU<MpcRefreshLayerServer>(mpc::DEFAULT_SCALE_BIT, mpc::RING_MOD, 128.0);
}

void InitMpc::init_relu2d_layer(const InitInferenceProcess& init, const string& key, const json& layer) {
    MpcTaskMetaData meta_data;
    init_mpc_layer(init, RELU2D_MPC, layer, meta_data);
    meta_data_map_[key] = MakeU<MpcTaskMetaData>(move(meta_data));
    ckks_relu2d_[key] = MakeU<ReluLayerServer>(mpc::DEFAULT_SCALE_BIT, mpc::RING_MOD, 128.0);
}

void InitMpc::init_maxpool2d_layer(const InitInferenceProcess& init, const string& key, const json& layer) {
    MpcTaskMetaData meta_data;
    init_mpc_layer(init, MAXPOOL2D_MPC, layer, meta_data);
    meta_data_map_[key] = MakeU<MpcTaskMetaData>(move(meta_data));

    Duo kernel_shape = {layer["kernel_shape"][0], layer["kernel_shape"][1]};
    Duo stride = {layer["stride"][0], layer["stride"][1]};
    ckks_maxpool2d_[key] =
        MakeU<PoolLayerServer>(kernel_shape, stride, mpc::DEFAULT_SCALE_BIT, mpc::RING_MOD, MAXPOOL, 128.0);
}

const MpcTaskMetaData& InitMpc::meta_data(const string& key) const {
    return *meta_data_map_.at(key);
}

MpcRefreshLayerServer& InitMpc::refresh_layer(const string& key) {
    return *ckks_mpc_refresh_.at(key);
}

ReluLayerServer& InitMpc::relu2d_layer(const string& key) {
    return *ckks_relu2d_.at(key);
}

PoolLayerServer& InitMpc::maxpool2d_layer(const string& key) {
    return *ckks_maxpool2d_.at(key);
}

Feature2DEncrypted InferenceMpcServer::calculate_mpc_refresh(const InitInferenceProcess& init,
                                                             InitMpc& init_mpc,
                                                             const map<string, UPtr<CkksContext>>& ckks_contexts,
                                                             const FeatureEncrypted& feature_node,
                                                             const string& key,
                                                             const json& layer) {
    constexpr int scale_ord = mpc::DEFAULT_SCALE_BIT;
    constexpr uint64_t ring_mod = mpc::RING_MOD;
    constexpr double pt_range = 128.0;

    FeatureNode feature_input(init.json_features[layer["feature_input"][0].get<string>()]);
    FeatureNode feature_output(init.json_features[layer["feature_output"][0].get<string>()]);
    if (feature_node.dim != 2) {
        throw runtime_error("mpc_refresh currently expects Feature2DEncrypted input");
    }

    const Feature2DEncrypted& x_enc = dynamic_cast<const Feature2DEncrypted&>(feature_node);
    CkksContext& context_in = *ckks_contexts.at(feature_input.ckks_parameter_id);
    CkksContext& context_out = *ckks_contexts.at(feature_output.ckks_parameter_id);
    PackType pack_type = PackType::MultipleChannelPacking;
    if (init.pack_style == "multiplexed") {
        pack_type = choose_pack_type(feature_input.shape, init.block_shape);
    }
    cout << "[mpc_refresh][server] key=" << key << ", pack_type=" << static_cast<int>(pack_type)
         << ", input_shape=(" << feature_input.shape[0] << "," << feature_input.shape[1] << ")"
         << ", output_shape=(" << feature_output.shape[0] << "," << feature_output.shape[1] << ")" << endl;

    EncToShareServer enc_to_share_server(context_in, scale_ord, ring_mod);
    Feature2DShare x_share0 = enc_to_share_server.server_enc_to_share_multi_pack_simple(x_enc, pack_type);
    Feature2DShare y_share0(ring_mod, scale_ord);
    y_share0.shape = x_share0.shape;
    if (x_share0.data.get_size() > 0) {
        assign_share_data_from_mpc_array(
            y_share0.data, init_mpc.refresh_layer(key).run(share_data_to_mpc_array(x_share0.data)));
    }
    if (x_share0.data_double.get_size() > 0) {
        auto x_data_double = mpc::Array<double, 1>::from_array_1d(x_share0.data_double.to_array_1d());
        y_share0.data_double = decltype(y_share0.data_double)::move_from_array_1d(
            init_mpc.refresh_layer(key).run_double(x_data_double).move_to_array_1d());
    }

    ShareToEncServer share_to_enc_server(context_out, scale_ord, ring_mod, pt_range);
    Feature2DEncrypted y_ct =
        share_to_enc_server.server_share_to_enc_multi_pack_simple(y_share0, feature_output.level, pack_type);
    y_ct.packing_type = pack_type;
    return y_ct;
}

Feature2DEncrypted InferenceMpcServer::calculate_relu2d(const InitInferenceProcess& init,
                                                        InitMpc& init_mpc,
                                                        const map<string, UPtr<CkksContext>>& ckks_contexts,
                                                        const FeatureEncrypted& feature_node,
                                                        const string& key,
                                                        const json& layer) {
    constexpr int scale_ord = mpc::DEFAULT_SCALE_BIT;
    constexpr uint64_t ring_mod = mpc::RING_MOD;
    constexpr double pt_range = 128.0;

    FeatureNode feature_input(init.json_features[layer["feature_input"][0].get<string>()]);
    FeatureNode feature_output(init.json_features[layer["feature_output"][0].get<string>()]);
    if (feature_node.dim != 2) {
        throw runtime_error("relu2d currently expects Feature2DEncrypted input");
    }

    const Feature2DEncrypted& x_enc = dynamic_cast<const Feature2DEncrypted&>(feature_node);
    CkksContext& context_in = *ckks_contexts.at(feature_input.ckks_parameter_id);
    CkksContext& context_out = *ckks_contexts.at(feature_output.ckks_parameter_id);
    PackType pack_type = PackType::MultipleChannelPacking;
    if (init.pack_style == "multiplexed") {
        pack_type = choose_pack_type(feature_input.shape, init.block_shape);
    }
    cout << "[relu2d][server] key=" << key << ", pack_type=" << static_cast<int>(pack_type)
         << ", input_shape=(" << feature_input.shape[0] << "," << feature_input.shape[1] << ")"
         << ", output_shape=(" << feature_output.shape[0] << "," << feature_output.shape[1] << ")" << endl;

    EncToShareServer enc_to_share_server(context_in, scale_ord, ring_mod);
    Feature2DShare x_share0 = enc_to_share_server.server_enc_to_share_multi_pack(x_enc, pack_type);
    Feature2DShare y_share0(ring_mod, scale_ord);
    y_share0.shape = x_share0.shape;
    assign_share_data_from_mpc_array(y_share0.data,
                                     init_mpc.relu2d_layer(key).run(share_data_to_mpc_array(x_share0.data)));

    ShareToEncServer share_to_enc_server(context_out, scale_ord, ring_mod, pt_range);
    Feature2DEncrypted y_ct =
        share_to_enc_server.server_share_to_enc_multi_pack(y_share0, feature_output.level, pack_type);
    y_ct.packing_type = pack_type;
    return y_ct;
}

Feature2DEncrypted InferenceMpcServer::calculate_maxpool2d(const InitInferenceProcess& init,
                                                           InitMpc& init_mpc,
                                                           const map<string, UPtr<CkksContext>>& ckks_contexts,
                                                           const FeatureEncrypted& feature_node,
                                                           const string& key,
                                                           const json& layer) {
    constexpr int scale_ord = mpc::DEFAULT_SCALE_BIT;
    constexpr uint64_t ring_mod = mpc::RING_MOD;
    constexpr double pt_range = 128.0;

    FeatureNode feature_input(init.json_features[layer["feature_input"][0].get<string>()]);
    FeatureNode feature_output(init.json_features[layer["feature_output"][0].get<string>()]);
    if (feature_node.dim != 2) {
        throw runtime_error("maxpool2d currently expects Feature2DEncrypted input");
    }

    const Feature2DEncrypted& x_enc = dynamic_cast<const Feature2DEncrypted&>(feature_node);
    CkksContext& context_in = *ckks_contexts.at(feature_input.ckks_parameter_id);
    CkksContext& context_out = *ckks_contexts.at(feature_output.ckks_parameter_id);
    PackType pack_type = PackType::MultipleChannelPacking;
    if (init.pack_style == "multiplexed") {
        pack_type = choose_pack_type(feature_input.shape, init.block_shape);
    }
    cout << "[maxpool2d][server] key=" << key << ", pack_type=" << static_cast<int>(pack_type)
         << ", input_shape=(" << feature_input.shape[0] << "," << feature_input.shape[1] << ")"
         << ", output_shape=(" << feature_output.shape[0] << "," << feature_output.shape[1] << ")" << endl;

    EncToShareServer enc_to_share_server(context_in, scale_ord, ring_mod);
    Feature2DShare x_share0 = enc_to_share_server.server_enc_to_share_multi_pack(x_enc, pack_type);

    Feature2DShare y_share0(ring_mod, scale_ord);
    const int input_area = static_cast<int>(x_share0.shape[0] * x_share0.shape[1]);
    if (input_area <= 0) {
        throw runtime_error("maxpool2d input share has invalid shape");
    }
    const int num_matrix = static_cast<int>(x_share0.data.get_size()) / input_area;
    assign_share_data_from_mpc_array(
        y_share0.data,
        init_mpc.maxpool2d_layer(key).run(x_share0.data.to_array_1d(), x_share0.shape, num_matrix));
    y_share0.shape = init_mpc.maxpool2d_layer(key).output_shape(x_share0.shape);

    ShareToEncServer share_to_enc_server(context_out, scale_ord, ring_mod, pt_range);
    Feature2DEncrypted y_ct =
        share_to_enc_server.server_share_to_enc_multi_pack(y_share0, feature_output.level, pack_type);
    y_ct.packing_type = pack_type;
    return y_ct;
}
