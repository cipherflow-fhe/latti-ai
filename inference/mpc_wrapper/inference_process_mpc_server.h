#pragma once

#include "data_structs/feature.h"
#include "mpc/mpc_refresh_layer.h"
#include "mpc/mpc_task_meta_data.h"
#include "util.h"

class InitInferenceProcess;

class InitMpc {
public:
    std::map<std::string, UPtr<MpcTaskMetaData>> meta_data_map_;
    std::map<std::string, UPtr<MpcRefreshLayerServer>> ckks_mpc_refresh_;

    void init_mpc_layer(const InitInferenceProcess& init,
                        const std::vector<MpcProtoType>& operations,
                        const json& layer,
                        MpcTaskMetaData& meta_data);
    void init_mpc_refresh_layer(const InitInferenceProcess& init, const std::string& key, const json& layer);
    const MpcTaskMetaData& meta_data(const std::string& key) const;
    MpcRefreshLayerServer& refresh_layer(const std::string& key);
};

class InferenceMpcServer {
public:
    Feature2DEncrypted calculate_mpc_refresh(
        const InitInferenceProcess& init,
        InitMpc& init_mpc,
        const std::map<std::string, UPtr<fhe_ops_lib::CkksContext>>& ckks_contexts,
        const FeatureEncrypted& feature_node,
        const std::string& key,
        const json& layer);
};
