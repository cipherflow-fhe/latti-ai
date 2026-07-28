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

#include "batch_dense_packed_layer.h"
#include "layer_util.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

using namespace std;
using namespace lattisense;

BatchDensePackedLayer::BatchDensePackedLayer(const CkksParameter& param_in,
                                             const Duo& shape_A,
                                             const Duo& shape_P,
                                             const Array<double, 2>& P_mat_in,
                                             uint32_t block_size,
                                             uint32_t level_A,
                                             Array<double, 1>&& bias)
    : Layer(param_in), P_mat_(P_mat_in.copy()), bias_(move(bias)) {
    if (shape_A[0] == 0 || shape_A[1] == 0 || shape_P[0] == 0 || shape_P[1] == 0)
        throw invalid_argument("BatchDensePackedLayer dimensions must be non-zero");
    if (shape_A[1] != shape_P[0])
        throw invalid_argument("BatchDensePackedLayer inner dimensions do not match");
    if (P_mat_in.get_shape()[0] != shape_P[0] || P_mat_in.get_shape()[1] != shape_P[1])
        throw invalid_argument("BatchDensePackedLayer weight shape does not match shape_P");
    if (block_size == 0 || (block_size & (block_size - 1)) != 0)
        throw invalid_argument("BatchDensePackedLayer block_size must be a power of two");

    batch_size_ = shape_A[0];
    input_dim_ = shape_A[1];
    output_dim_ = shape_P[1];
    block_size_ = block_size;
    level_ = level_A;

    n_slot_ = param_.get_n() / 2;
    chunk_size_ = block_size_ * block_size_;
    if (n_slot_ < chunk_size_ || n_slot_ % chunk_size_ != 0)
        throw invalid_argument("BatchDensePackedLayer requires n_slot divisible by block_size^2");

    chunks_per_ct_ = n_slot_ / chunk_size_;
    n_batch_blocks_ = div_ceil(batch_size_, block_size_);
    n_input_blocks_ = div_ceil(input_dim_, block_size_);
    n_output_blocks_ = div_ceil(output_dim_, block_size_);
    batch_ct_groups_ = div_ceil(n_batch_blocks_, chunks_per_ct_);
    // Each active group owns a baby/wrap rotation cache. Bound this layer's
    // concurrency to four so larger batches do not multiply peak ciphertext
    // memory without bound.
    batch_group_threads_ = std::min<uint32_t>(batch_ct_groups_, 4);

    // The standard sqrt choice is faster here in practice: giant-step
    // rotations happen after plaintext products and are substantially more
    // expensive than the input-side baby rotations on this backend.
    bsgs_baby_step_ = static_cast<uint32_t>(ceil(sqrt(static_cast<double>(block_size_))));
    bsgs_giant_steps_ = div_ceil(block_size_, bsgs_baby_step_);

    if (bias_.get_size() > 0) {
        if (bias_.get_size() != output_dim_)
            throw invalid_argument("BatchDensePackedLayer bias shape does not match output dimension");
        has_bias_ = true;
    }
}

int BatchDensePackedLayer::get_block_index(uint32_t block_col, uint32_t group, uint32_t groups_per_col) {
    return static_cast<int>(block_col * groups_per_col + group);
}

vector<double>
BatchDensePackedLayer::build_diagonal(uint32_t input_block, uint32_t output_block, uint32_t k, bool wrapping) const {
    vector<double> result(n_slot_, 0.0);
    uint32_t rotation = k * block_size_;
    uint32_t wrap_begin = chunk_size_ - rotation;

    for (uint32_t chunk = 0; chunk < chunks_per_ct_; chunk++) {
        uint32_t chunk_base = chunk * chunk_size_;
        for (uint32_t col = 0; col < block_size_; col++) {
            uint32_t weight_row = input_block * block_size_ + (col + k) % block_size_;
            uint32_t weight_col = output_block * block_size_ + col;
            double value = 0.0;
            if (weight_row < input_dim_ && weight_col < output_dim_)
                value = P_mat_.get(weight_row, weight_col);

            for (uint32_t row = 0; row < block_size_; row++) {
                uint32_t local_slot = row + block_size_ * col;
                bool is_wrap = rotation != 0 && local_slot >= wrap_begin;
                if (is_wrap == wrapping)
                    result[chunk_base + local_slot] = value;
            }
        }
    }
    return result;
}

vector<double> BatchDensePackedLayer::build_bias(uint32_t output_block, uint32_t group) const {
    vector<double> result(n_slot_, 0.0);
    if (!has_bias_)
        return result;

    for (uint32_t chunk = 0; chunk < chunks_per_ct_; chunk++) {
        uint32_t batch_block = group * chunks_per_ct_ + chunk;
        uint32_t valid_rows = 0;
        if (batch_block < n_batch_blocks_)
            valid_rows = std::min(block_size_, batch_size_ - batch_block * block_size_);
        uint32_t chunk_base = chunk * chunk_size_;
        for (uint32_t col = 0; col < block_size_; col++) {
            uint32_t output_col = output_block * block_size_ + col;
            double value = output_col < output_dim_ ? bias_[output_col] : 0.0;
            for (uint32_t row = 0; row < valid_rows; row++)
                result[chunk_base + row + block_size_ * col] = value;
        }
    }
    return result;
}

vector<double> BatchDensePackedLayer::shift_plaintext_right(const vector<double>& values, uint32_t rotation) const {
    vector<double> shifted(values.size(), 0.0);
    if (values.empty())
        return shifted;
    rotation %= values.size();
    for (size_t i = 0; i < values.size(); i++) {
        size_t src = (i + values.size() - rotation) % values.size();
        shifted[i] = values[src];
    }
    return shifted;
}

void BatchDensePackedLayer::precompute_diagonals() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    if (has_bias_) {
        bias_pt_.resize(n_output_blocks_ * batch_ct_groups_);
        for (uint32_t output_block = 0; output_block < n_output_blocks_; output_block++) {
            for (uint32_t group = 0; group < batch_ct_groups_; group++) {
                uint32_t bias_idx = output_block * batch_ct_groups_ + group;
                bias_pt_[bias_idx] = ctx.encode_ringt(build_bias(output_block, group), param_.get_default_scale());
            }
        }
    }
    // Diagonals are generated per input/output block in run() to avoid
    // retaining thousands of full RNS plaintexts for large matrices.
    diagonals_prepared_ = true;
}

CkksCiphertext BatchDensePackedLayer::block_matmul(CkksContext& ctx,
                                                   const RotatedInput& rotations,
                                                   const vector<CkksPlaintextMul>& diagonal_now,
                                                   const vector<CkksPlaintextMul>& diagonal_wrap) const {
    if (diagonal_now.size() != block_size_ || diagonal_wrap.size() != block_size_)
        throw runtime_error("BatchDensePackedLayer diagonal count mismatch");
    if (rotations.baby_now.size() != bsgs_baby_step_ || rotations.baby_wrap.size() != bsgs_baby_step_)
        throw runtime_error("BatchDensePackedLayer rotation count mismatch");

    CkksCiphertext result(0);
    bool initialized = false;

    for (uint32_t giant_index = 0; giant_index < bsgs_giant_steps_; giant_index++) {
        CkksCiphertext inner_now(0);
        CkksCiphertext inner_wrap(0);
        bool now_initialized = false;
        bool wrap_initialized = false;
        uint32_t begin = giant_index * bsgs_baby_step_;
        uint32_t end = std::min(block_size_, begin + bsgs_baby_step_);

        for (uint32_t b = 0; b < end - begin; b++) {
            uint32_t k = begin + b;
            auto now_term = ctx.mult_plain_mul(rotations.baby_now[b], diagonal_now[k]);
            if (!now_initialized) {
                inner_now = move(now_term);
                now_initialized = true;
            } else {
                inner_now = ctx.add(inner_now, now_term);
            }

            // k=0 has an all-zero wrapping diagonal and does not need the
            // otherwise unnecessary negative rotation.
            if (k != 0) {
                auto wrap_term = ctx.mult_plain_mul(rotations.baby_wrap[b], diagonal_wrap[k]);
                if (!wrap_initialized) {
                    inner_wrap = move(wrap_term);
                    wrap_initialized = true;
                } else {
                    inner_wrap = ctx.add(inner_wrap, wrap_term);
                }
            }
        }

        CkksCiphertext inner = move(inner_now);
        if (wrap_initialized)
            inner = ctx.add(inner, inner_wrap);
        // Both branches have the same giant-step compensation. Add them
        // before rotating so one giant rotation serves both branches.
        if (giant_index > 0) {
            int giant_rotation = static_cast<int>(giant_index * bsgs_baby_step_ * block_size_);
            inner = ctx.rotate(inner, giant_rotation);
        }
        if (!initialized) {
            result = move(inner);
            initialized = true;
        } else {
            result = ctx.add(result, inner);
        }
    }

    return ctx.rescale(result, ctx.get_parameter().get_default_scale());
}

BatchDensePackedLayer::RotatedInput
BatchDensePackedLayer::precompute_input_rotations(CkksContext& ctx, const CkksCiphertext& input) const {
    RotatedInput result;
    // The same rotations are used by every output block. Keep them local to
    // one batch group so memory remains bounded by O(bsgs_baby_step_).
    result.baby_now = populate_rotations_1_side(ctx, input, bsgs_baby_step_ - 1, block_size_);

    // For the wrapping branch, the -chunk_size offset makes the subsequent
    // giant rotation equivalent to a chunk-local rotation.
    vector<int32_t> wrap_steps;
    wrap_steps.reserve(bsgs_baby_step_);
    for (uint32_t b = 0; b < bsgs_baby_step_; b++)
        wrap_steps.push_back(static_cast<int32_t>(b * block_size_) - static_cast<int>(chunk_size_));
    auto wrap_map = ctx.rotate(input, wrap_steps);
    result.baby_wrap.reserve(bsgs_baby_step_);
    for (int32_t step : wrap_steps)
        result.baby_wrap.push_back(move(wrap_map.at(step)));
    return result;
}

void BatchDensePackedLayer::encode_diagonals(CkksContext& ctx,
                                             uint32_t input_block,
                                             uint32_t output_block,
                                             vector<CkksPlaintextMul>& diagonal_now,
                                             vector<CkksPlaintextMul>& diagonal_wrap) const {
    double diagonal_scale = param_.get_q(level_);
    diagonal_now.clear();
    diagonal_wrap.clear();
    diagonal_now.reserve(block_size_);
    diagonal_wrap.reserve(block_size_);

    for (uint32_t k = 0; k < block_size_; k++) {
        uint32_t giant = (k / bsgs_baby_step_) * bsgs_baby_step_ * block_size_;
        auto now = shift_plaintext_right(build_diagonal(input_block, output_block, k, false), giant);
        auto wrapping = shift_plaintext_right(build_diagonal(input_block, output_block, k, true), giant);
        auto now_ringt = ctx.encode_ringt(now, diagonal_scale);
        auto wrapping_ringt = ctx.encode_ringt(wrapping, diagonal_scale);
        diagonal_now.push_back(ctx.ringt_to_mul(now_ringt, level_));
        diagonal_wrap.push_back(ctx.ringt_to_mul(wrapping_ringt, level_));
    }
}

Feature0DEncrypted BatchDensePackedLayer::run(CkksContext& ctx, const Feature0DEncrypted& A) {
    uint32_t expected_input_cts = n_input_blocks_ * batch_ct_groups_;
    if (!A.data_compressed.empty())
        throw invalid_argument("BatchDensePackedLayer requires decompressed input ciphertexts");
    if (!diagonals_prepared_)
        throw runtime_error("BatchDensePackedLayer::precompute_diagonals() was not called");
    if (A.level != level_ || !A.batch_packed || A.batch_size != batch_size_ || A.batch_feature_dim != input_dim_ ||
        A.batch_block_size != block_size_ || A.data.size() != expected_input_cts) {
        throw invalid_argument("BatchDensePackedLayer input shape or ciphertext layout mismatch");
    }

    Feature0DEncrypted result(&ctx, A.level - 1);
    result.data.resize(n_output_blocks_ * batch_ct_groups_);
    result.batch_packed = true;
    result.batch_size = batch_size_;
    result.batch_feature_dim = output_dim_;
    result.batch_block_size = block_size_;
    result.n_channel = output_dim_;
    result.n_channel_per_ct = block_size_;
    result.skip = 1;

    CkksContext diagonal_context = CkksContext::create_empty_context(param_);
    for (uint32_t input_block = 0; input_block < n_input_blocks_; input_block++) {
        // Convert each diagonal to the multiplication representation once.
        // The resulting plaintexts are read-only and reused by all batch
        // groups and by every output block in this input-block pass.
        vector<vector<CkksPlaintextMul>> diagonals_now(n_output_blocks_);
        vector<vector<CkksPlaintextMul>> diagonals_wrap(n_output_blocks_);
        for (uint32_t output_block = 0; output_block < n_output_blocks_; output_block++) {
            encode_diagonals(diagonal_context, input_block, output_block, diagonals_now[output_block],
                             diagonals_wrap[output_block]);
        }

        // A batch ciphertext's rotations do not depend on the output block.
        // Generate them once, then consume the same rotation set for all
        // output blocks. This is the main BSGS reuse across output channels.
        parallel_for(
            static_cast<int>(batch_ct_groups_), batch_group_threads_, ctx, [&](CkksContext& ctx_copy, int group_index) {
                uint32_t group = static_cast<uint32_t>(group_index);
                uint32_t input_index = input_block * batch_ct_groups_ + group;
                auto rotations = precompute_input_rotations(ctx_copy, A.data[input_index]);
                for (uint32_t output_block = 0; output_block < n_output_blocks_; output_block++) {
                    uint32_t output_index = output_block * batch_ct_groups_ + group;
                    auto term =
                        block_matmul(ctx_copy, rotations, diagonals_now[output_block], diagonals_wrap[output_block]);
                    if (input_block == 0) {
                        result.data[output_index] = move(term);
                    } else {
                        result.data[output_index] = ctx_copy.add(result.data[output_index], term);
                    }
                }
            });
    }

    if (has_bias_) {
        for (uint32_t output_block = 0; output_block < n_output_blocks_; output_block++) {
            parallel_for(static_cast<int>(batch_ct_groups_), batch_group_threads_, ctx,
                         [&](CkksContext& ctx_copy, int group_index) {
                             uint32_t group = static_cast<uint32_t>(group_index);
                             uint32_t output_index = output_block * batch_ct_groups_ + group;
                             uint32_t bias_index = output_index;
                             result.data[output_index] =
                                 ctx_copy.add_plain_ringt(result.data[output_index], bias_pt_[bias_index]);
                         });
        }
    }

    result.n_channel = output_dim_;
    result.n_channel_per_ct = block_size_;
    return result;
}

Array<double, 2> BatchDensePackedLayer::run_plaintext(const Array<double, 2>& A) const {
    if (A.get_shape()[0] != batch_size_ || A.get_shape()[1] != input_dim_)
        throw invalid_argument("BatchDensePackedLayer plaintext input shape mismatch");

    Array<double, 2> result({static_cast<uint64_t>(batch_size_), static_cast<uint64_t>(output_dim_)});
    for (uint32_t batch = 0; batch < batch_size_; batch++) {
        for (uint32_t output = 0; output < output_dim_; output++) {
            double value = has_bias_ ? bias_[output] : 0.0;
            for (uint32_t input = 0; input < input_dim_; input++)
                value += A.get(batch, input) * P_mat_.get(input, output);
            result.set(batch, output, value);
        }
    }
    return result;
}
