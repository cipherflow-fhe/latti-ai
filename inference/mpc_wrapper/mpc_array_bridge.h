#pragma once

#include "mpc/mpc_types.h"
#include "util/ndarray.h"

template <typename T, int dim>
mpc::Array<T, dim> to_mpc_array(const Array<T, dim>& input) {
    mpc::Array<T, dim> output(input.get_shape());
    output.set_data(input.to_array_1d());
    return output;
}

template <typename T, int dim>
Array<T, dim> to_latti_array(mpc::Array<T, dim>&& input) {
    Array<T, dim> output(input.get_shape());
    output.move_data(input.move_to_array_1d());
    return output;
}

template <typename ShareData>
mpc::Array<uint64_t, 1> share_data_to_mpc_array(const ShareData& data) {
    return mpc::Array<uint64_t, 1>::from_array_1d(data.to_array_1d());
}

template <typename ShareData>
void assign_share_data_from_mpc_array(ShareData& dst, mpc::Array<uint64_t, 1>&& data) {
    dst = ShareData::move_from_array_1d(data.move_to_array_1d());
}
