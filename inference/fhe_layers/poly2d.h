#pragma once

#include "poly_base.h"
#include "../data_structs/feature2d.h"
#include <vector>

class Poly2D : public PolyBase {
public:
    Poly2D(const ls::CkksParameter& param,
           const Duo& input_shape,
           int order,
           Array<double, 2>&& weight,
           const Duo& skip,
           uint32_t n_channel_per_ct,
           uint32_t level,
           const Duo& zero_skip = {1, 1},
           const Duo& block_expansion = {1, 1},
           bool is_ordinary_pack = false);

    void prepare_weight() override {
        prepare_weight_stockmeyer();
    }
    void prepare_weight_lazy() override {
        prepare_weight_stockmeyer_lazy();
    }
    void prepare_weight_stockmeyer();
    void prepare_weight_stockmeyer_lazy();

    ls::CkksPlaintextRingt generate_weight_pt_for_stockmeyer(ls::CkksContext& ctx, int coeff_idx, int ct_idx) const;
    std::vector<ls::CkksCiphertext> run_core_stockmeyer(ls::CkksContext& ctx, const std::vector<ls::CkksCiphertext>& x);
    Feature2DEncrypted run_stockmeyer(ls::CkksContext& ctx, const Feature2DEncrypted& x);

    static int compute_stockmeyer_level_cost(int order) {
        return PolyBase::compute_stockmeyer_level_cost(order);
    }

    Array<double, 3> run_plaintext(const Array<double, 3>& x) const;
    Array<double, 3> run_plaintext_for_non_absorb_case(const Array<double, 3>& x) const {
        return run_plaintext(x);
    }

    Array<double, 2> weight;
    Duo input_shape;
    Duo skip;
    Duo zero_skip;
    Duo block_expansion;
    Duo block_shape;
    Duo pre_skip;
    uint32_t n_channel_per_ct;
    uint32_t n_block_per_ct;
    bool is_ordinary_pack;
    std::vector<std::vector<ls::CkksPlaintextRingt>> weight_pt;

private:
    int cached_skip_prod;
    int cached_channel;
    int cached_total_block_size;
    std::vector<double> build_weight_vec(int coeff_idx, int ct_idx) const;
};
