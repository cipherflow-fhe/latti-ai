#pragma once

#include "mpc/mpc_types.h"

template <typename ShareData>
mpc::Array<uint64_t, 1> share_data_to_mpc_array(const ShareData& data) {
    return mpc::Array<uint64_t, 1>::from_array_1d(data.to_array_1d());
}

template <typename ShareData>
void assign_share_data_from_mpc_array(ShareData& dst, mpc::Array<uint64_t, 1>&& data) {
    dst = ShareData::move_from_array_1d(data.move_to_array_1d());
}
