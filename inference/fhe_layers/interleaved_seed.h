// Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "layer.h"

// Public layout operations shared by multiplexed and inverse seed weights.
namespace interleaved_seed {

// Dense support is sparse in the channel residues inside each physical pixel.
// Packing moves it right by delta; CKKS Rot(+delta) moves it back left.
inline uint32_t tile_offset(const Duo& shape, const Duo& skip, const Duo& invalid_fill, uint32_t tile) {
    if (!prod(shape) || !prod(skip) || !prod(invalid_fill) || skip[0] % invalid_fill[0] ||
        skip[1] % invalid_fill[1] || tile >= prod(invalid_fill)) {
        throw std::invalid_argument("invalid dense tile geometry or index");
    }
    return (tile / invalid_fill[1]) * (skip[0] / invalid_fill[0]) * shape[1] * skip[1] +
           (tile % invalid_fill[1]) * (skip[1] / invalid_fill[1]);
}

inline std::vector<double> tile_mask(const Duo& shape, const Duo& skip, const Duo& invalid_fill,
                                      uint32_t slots, uint32_t tile, bool bias) {
    const uint32_t delta = tile_offset(shape, skip, invalid_fill, tile);
    const Duo physical = shape * skip;
    const uint32_t block_size = prod(physical);
    if (slots == 0 || slots % block_size) throw std::invalid_argument("dense tiles require full physical blocks");
    std::vector<double> mask(slots, 0.0);
    for (uint32_t i = 0; i < slots; ++i) {
        const uint32_t h = (i % block_size) / physical[1], w = i % physical[1];
        if (bias ? (h == 0 && w == 0) :
            (h % skip[0] < skip[0] / invalid_fill[0] && w % skip[1] < skip[1] / invalid_fill[1])) {
            mask[i + delta] = 1.0;
        }
    }
    return mask;
}

template <typename GenerateValues>
std::vector<double> pack_tiles(uint32_t count, uint32_t seed_index, const Duo& shape, const Duo& skip,
                                 const Duo& invalid_fill, uint32_t slots, GenerateValues generate) {
    const uint32_t group = prod(invalid_fill);
    if (!group || seed_index >= div_ceil(count, group)) throw std::out_of_range("dense tile seed index");
    std::vector<double> seed(slots, 0.0);
    for (uint32_t sub = 0; sub < group && seed_index * group + sub < count; ++sub) {
        const auto source = generate(seed_index * group + sub);
        if (source.size() != slots) throw std::invalid_argument("dense tile source slot count");
        const uint32_t delta = tile_offset(shape, skip, invalid_fill, sub);
        for (uint32_t i = 0; i < slots; ++i) seed[(i + delta) % slots] += source[i];
    }
    return seed;
}

inline bool valid_stride(const Duo& stride, const Duo& shape) {
    return stride[0] && stride[1] && !(stride[0] & (stride[0] - 1)) &&
           !(stride[1] & (stride[1] - 1)) && stride != Duo{1, 1} &&
           shape[0] % stride[0] == 0 && shape[1] % stride[1] == 0;
}

inline std::vector<double> mask(const Duo& shape, const Duo& skip, uint32_t blocks, uint32_t slots,
                                const Duo& stride, int subgrid, const DuoInt& step, bool boundary) {
    std::vector<double> values(slots, 0.0);
    const Duo physical = shape * skip;
    for (uint32_t t = 0; t < blocks; ++t) {
        for (uint32_t h = 0; h < physical[0]; ++h) {
            for (uint32_t w = 0; w < physical[1]; ++w) {
                const uint32_t index = t * prod(physical) + h * physical[1] + w;
                if (index >= slots) continue;
                if (boundary) {
                    const int64_t row = static_cast<int64_t>(h) + step[0] * static_cast<int64_t>(skip[0]);
                    const int64_t col = static_cast<int64_t>(w) + step[1] * static_cast<int64_t>(skip[1]);
                    values[index] = row >= 0 && row < physical[0] && col >= 0 && col < physical[1];
                } else {
                    values[index] = (h / skip[0]) % stride[0] == subgrid / stride[1] &&
                                    (w / skip[1]) % stride[1] == subgrid % stride[1];
                }
            }
        }
    }
    return values;
}

struct InverseKernelPosition {
    Duo kernel;
    DuoInt step;
};

inline InverseKernelPosition inverse_kernel_position(const Duo& kernel_shape, const Duo& ct_stride,
                                                     const Duo& output_step, const DuoInt& pad, int index) {
    const Duo input_step = ct_stride * output_step;
    int count = 0;
    for (const Duo& r2 : duo_range(output_step)) {
        for (const Duo& seg : duo_range(ct_stride)) {
            if (seg[0] >= kernel_shape[0] || seg[1] >= kernel_shape[1]) continue;
            const Duo split = (kernel_shape + ct_stride - Duo{1, 1} - seg) / ct_stride;
            for (const Duo& uv : duo_range(split)) {
                if (count++ != index) continue;
                const DuoInt val = to_int(seg) - pad + to_int(ct_stride) * to_int(uv + r2);
                const DuoInt begin = (val % input_step + to_int(input_step)) % input_step;
                return {uv * ct_stride + seg, (val - begin) / to_int(input_step)};
            }
        }
    }
    throw std::out_of_range("inverse seed kernel index");
}

template <typename GenerateValues>
std::vector<double> pack_kernels(uint32_t count, uint32_t seed_index, const Duo& stride,
                                  const Duo& shape, const Duo& skip, uint32_t blocks, uint32_t slots,
                                  GenerateValues generate) {
    std::vector<double> seed(slots, 0.0);
    const uint32_t group = prod(stride);
    for (uint32_t sub = 0; sub < group && seed_index * group + sub < count; ++sub) {
        const auto source = generate(seed_index * group + sub);
        const auto select = mask(shape, skip, blocks, slots, stride, sub, {0, 0}, false);
        for (uint32_t i = 0; i < slots; ++i) seed[i] += source[i] * select[i];
    }
    return seed;
}

// The C++ layer's CT*PT path restores plaintext slots locally. Encrypted
// parameters are restored homomorphically by the Python-generated Runner graph.
inline ls::CkksPlaintext restore_plaintext(ls::CkksContext& ctx, const ls::CkksPlaintext& seed,
                                               const std::vector<double>& select,
                                               const std::vector<double>& boundary, const Duo& shape,
                                               const Duo& skip, const Duo& stride, int subgrid,
                                               int level, double scale) {
    std::vector<double> values;
    ctx.decode(seed, values);
    for (size_t i = 0; i < values.size(); ++i) values[i] *= i < select.size() ? select[i] : 0.0;
    const uint32_t residue[2] = {static_cast<uint32_t>(subgrid) / stride[1],
                                static_cast<uint32_t>(subgrid) % stride[1]};
    const uint32_t units[2] = {skip[0] * shape[1] * skip[1], skip[1]};
    for (int dim = 0; dim < 2; ++dim) {
        for (uint32_t bit = 0; (1u << bit) < stride[dim]; ++bit) {
            int64_t step = (1u << bit) * units[dim];
            if (((residue[dim] >> bit) & 1) == 0) step = -step;
            const auto previous = values;
            const int64_t size = values.size();
            for (int64_t i = 0; i < size; ++i) values[i] += previous[(i + step + size) % size];
        }
    }
    for (size_t i = 0; i < values.size(); ++i) values[i] *= i < boundary.size() ? boundary[i] : 0.0;
    return fhe_api::encode_ringt(ctx, values, scale);
}

}  // namespace interleaved_seed
