#pragma once

#include <map>
#include <memory>
#include <string>

#include "fhe_ops_lib/fhe_lib_v2.h"
#include "mpc/mpc_numeric.h"
#include "mpc/mpc_types.h"

using fhe_ops_lib::CkksContext;

class InferenceMpcClient {
public:
    explicit InferenceMpcClient(std::map<std::string, std::unique_ptr<CkksContext>>& ckks_contexts);

    mpc::Array1DUint run();

private:
    std::map<std::string, std::unique_ptr<CkksContext>>& ckks_contexts_;
    int scale_ord_ = mpc::DEFAULT_SCALE_BIT;
    double pt_range_ = 128.0;
    uint64_t ring_mod_ = mpc::RING_MOD;
};
