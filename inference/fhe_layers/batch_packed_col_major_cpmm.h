/*
 * Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#pragma once

#include "layer.h"
#include "../data_structs/feature_mat.h"

// Ciphertext-plaintext matrix multiplication for FeatureMatEncrypted values
// packed with FeatureMatEncrypted::batch_block_col_major_pack().
//
// The logical operation is A(B x K) @ P(K x H) = C(B x H). Different batch
// blocks are stored in different d*d chunks of one ciphertext. Since a normal
// CKKS rotation is global, the block kernel splits every rotated diagonal into
// non-wrapping and wrapping plaintext diagonals and uses two global rotations
// to realize a chunk-local rotation without an extra level-consuming mask.
class BatchPackedColMajorCPMM : public Layer {
public:
    BatchPackedColMajorCPMM(const ls::CkksParameter& param_in,
                            const Duo& shape_A,
                            const Duo& shape_P,
                            const Array<double, 2>& P_mat_in,
                            uint32_t block_size,
                            uint32_t level_A,
                            Array<double, 1>&& bias = Array<double, 1>());

    void precompute_diagonals();
    FeatureMatEncrypted run(ls::CkksContext& ctx, const FeatureMatEncrypted& A);
    Array<double, 2> run_plaintext(const Array<double, 2>& A) const;

private:
    struct RotatedInput {
        std::vector<ls::CkksCiphertext> baby_now;
        std::vector<ls::CkksCiphertext> baby_wrap;
    };

    static int get_block_index(uint32_t block_col, uint32_t group, uint32_t groups_per_col);

    std::vector<double> build_diagonal(uint32_t input_block, uint32_t output_block, uint32_t k, bool wrapping) const;
    std::vector<double> shift_plaintext_right(const std::vector<double>& values, uint32_t rotation) const;
    std::vector<double> build_bias(uint32_t output_block, uint32_t group) const;
    void encode_diagonals(ls::CkksContext& ctx,
                          uint32_t input_block,
                          uint32_t output_block,
                          std::vector<ls::CkksPlaintextMul>& diagonal_now,
                          std::vector<ls::CkksPlaintextMul>& diagonal_wrap) const;

    RotatedInput precompute_input_rotations(ls::CkksContext& ctx, const ls::CkksCiphertext& input) const;

    ls::CkksCiphertext block_matmul(ls::CkksContext& ctx,
                                    const RotatedInput& rotations,
                                    const std::vector<ls::CkksPlaintextMul>& diagonal_now,
                                    const std::vector<ls::CkksPlaintextMul>& diagonal_wrap) const;

    uint32_t batch_size_;
    uint32_t input_dim_;
    uint32_t output_dim_;
    uint32_t block_size_;
    uint32_t n_slot_;
    uint32_t chunk_size_;
    uint32_t chunks_per_ct_;
    uint32_t n_batch_blocks_;
    uint32_t n_input_blocks_;
    uint32_t n_output_blocks_;
    uint32_t batch_ct_groups_;
    uint32_t batch_group_threads_;
    uint32_t bsgs_baby_step_;
    uint32_t bsgs_giant_steps_;
    Array<double, 2> P_mat_;
    Array<double, 1> bias_;
    bool has_bias_ = false;

    std::vector<ls::CkksPlaintextRingt> bias_pt_;
    bool diagonals_prepared_ = false;
};
