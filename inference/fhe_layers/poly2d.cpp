#include "poly2d.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>

using namespace std;
using namespace lattisense;

Poly2D::Poly2D(const CkksParameter& param,
               const Duo& input_shape_in,
               int order_in,
               Array<double, 2>&& weight_in,
               const Duo& skip_in,
               uint32_t n_channel_per_ct_in,
               uint32_t level_in,
               const Duo& zero_skip_in,
               const Duo& block_expansion_in,
               bool is_ordinary_pack_in)
    : PolyBase(param, level_in, order_in), weight(move(weight_in)), input_shape(input_shape_in), skip(skip_in),
      zero_skip(zero_skip_in), block_expansion(block_expansion_in), n_channel_per_ct(n_channel_per_ct_in),
      is_ordinary_pack(is_ordinary_pack_in) {
    if ((input_shape[0] & (input_shape[0] - 1)) != 0 || (input_shape[1] & (input_shape[1] - 1)) != 0)
        throw invalid_argument("input_shape must be powers of 2");
    if ((skip[0] & (skip[0] - 1)) != 0 || (skip[1] & (skip[1] - 1)) != 0)
        throw invalid_argument("skip must be powers of 2");
    n_block_per_ct = div_ceil(n_channel_per_ct, skip[0] * skip[1]);
    pre_skip = {skip[0] * zero_skip[0], skip[1] * zero_skip[1]};
    block_shape = {input_shape[0] * skip[0] / block_expansion[0], input_shape[1] * skip[1] / block_expansion[1]};
    if ((block_shape[0] & (block_shape[0] - 1)) != 0 || (block_shape[1] & (block_shape[1] - 1)) != 0)
        throw invalid_argument("block_shape must be powers of 2");
    cached_skip_prod = skip[0] * skip[1];
    cached_channel = weight.get_shape()[1];
    cached_total_block_size = n_block_per_ct * block_shape[0] * block_shape[1];
}

vector<double> Poly2D::build_weight_vec(int idx, int packed_idx) const {
    vector<double> out(N / 2, 0.0);
    int n_packed = div_ceil(cached_channel, n_channel_per_ct) * block_expansion[0] * block_expansion[1];
    if (packed_idx < 0 || packed_idx >= n_packed)
        throw out_of_range("Poly2D ciphertext index out of range");
    if (is_ordinary_pack) {
        for (int ch = 0; ch < (int)n_channel_per_ct; ++ch) {
            int channel_idx = packed_idx * n_channel_per_ct + ch;
            if (channel_idx >= cached_channel)
                continue;
            for (int i = 0; i < input_shape[0]; ++i)
                for (int j = 0; j < input_shape[1]; ++j)
                    out[ch * block_shape[0] * block_shape[1] + i * block_shape[1] * skip[0] + j * skip[1]] =
                        weight.get(idx, channel_idx);
        }
    } else {
        for (int linear = 0; linear < cached_total_block_size; ++linear) {
            int block = linear / (block_shape[0] * block_shape[1]);
            int residual = linear % (block_shape[0] * block_shape[1]);
            int i = residual / block_shape[1], j = residual % block_shape[1];
            int channel_idx = packed_idx * n_channel_per_ct / block_expansion[0] / block_expansion[1] +
                              block * cached_skip_prod + skip[1] * (i % skip[0]) + j % skip[1];
            if (channel_idx >= cached_channel || i % pre_skip[0] >= skip[0] || j % pre_skip[1] >= skip[1])
                continue;
            out[linear] = weight.get(idx, channel_idx);
        }
    }
    return out;
}

CkksPlaintextRingt Poly2D::generate_weight_pt_for_stockmeyer(CkksContext& ctx, int coeff_idx, int ct_idx) const {
    return ctx.encode_ringt(build_weight_vec(coeff_idx, ct_idx), get_stockmeyer_coeff_scale().at(coeff_idx));
}

void Poly2D::prepare_weight_stockmeyer() {
    init_stockmeyer();
    int n_ct = div_ceil(cached_channel, n_channel_per_ct) * block_expansion[0] * block_expansion[1];
    weight_pt.clear();
    weight_pt.resize(order + 1);
    CkksContext ctx = CkksContext::create_empty_context(param_);
    parallel_for(order + 1, th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        weight_pt[idx].resize(n_ct);
        for (int ct = 0; ct < n_ct; ++ct)
            weight_pt[idx][ct] = generate_weight_pt_for_stockmeyer(ctx_copy, idx, ct);
    });
}

void Poly2D::prepare_weight_stockmeyer_lazy() {
    init_stockmeyer();
    weight_pt.clear();
}

vector<CkksCiphertext> Poly2D::run_core_stockmeyer(CkksContext& ctx, const vector<CkksCiphertext>& x) {
    init_stockmeyer();
    vector<CkksCiphertext> result(x.size());
    const auto& coeff_levels = get_stockmeyer_level_order();
    const auto& powers = get_stockmeyer_powers();
    const auto& baby_levels = get_stockmeyer_baby_poly_output_level();
    const auto& baby_scales = get_stockmeyer_baby_poly_output_scale();
    const int n_baby = get_stockmeyer_n_baby_polys();
    const int baby_steps = get_stockmeyer_baby_steps();

    struct Node {
        CkksCiphertext ct;
        int constant = -1;
        int level = -1;
        double scale = 0;
        bool has_ct = false;
    };
    parallel_for(x.size(), th_nums, ctx, [&](CkksContext& c, int xi) {
        map<int, CkksCiphertext> xp;
        xp[1] = x[xi].copy();
        for (const auto& kv : powers) {
            if (kv.first <= 1)
                continue;
            auto h = xp.at(kv.second.decomp_a).copy();
            xp[kv.first] = c.rescale(c.relinearize(c.mult(h, h)), c.get_parameter().get_default_scale());
        }
        auto drop = [&](CkksCiphertext v, int lv) {
            while (v.get_level() > lv)
                v = c.drop_level(v);
            if (v.get_level() != lv)
                throw runtime_error("Poly2D Stockmeyer level mismatch");
            return v;
        };
        auto coeff_mul = [&](int idx, int lv) {
            auto it = coeff_levels.find(idx);
            if (it == coeff_levels.end() || it->second != lv)
                throw runtime_error("Poly2D coefficient level mismatch");
            if (weight_pt.empty())
                return c.ringt_to_mul(generate_weight_pt_for_stockmeyer(c, idx, xi), lv);
            return c.ringt_to_mul(weight_pt[idx][xi], lv);
        };
        auto coeff_add = [&](const CkksCiphertext& v, int idx) {
            if (weight_pt.empty())
                return c.add_plain_ringt(v, generate_weight_pt_for_stockmeyer(c, idx, xi));
            return c.add_plain_ringt(v, weight_pt[idx][xi]);
        };
        vector<Node> nodes;
        for (int b = 0; b < n_baby; ++b) {
            int base = b * baby_steps, target = baby_levels[b];
            double scale = baby_scales[b];
            Node node;
            node.level = target;
            node.scale = scale;
            CkksCiphertext acc;
            bool initialized = false;
            if (base + 1 <= order) {
                auto p = drop(xp.at(1).copy(), target + 1);
                auto term = c.rescale(c.mult_plain_mul(p, coeff_mul(base + 1, p.get_level())), scale);
                acc = move(term);
                initialized = true;
            }
            if (base + 2 <= order) {
                auto p = drop(xp.at(2).copy(), target + 1);
                auto term = c.rescale(c.mult_plain_mul(p, coeff_mul(base + 2, p.get_level())), scale);
                if (initialized)
                    acc = c.add(acc, term);
                else
                    acc = move(term);
                initialized = true;
            }
            if (base + 3 <= order) {
                double c3_scale = (scale / powers.at(2).scale) * c.get_parameter().get_q(target + 1);
                auto c3x = drop(xp.at(1).copy(), target + 2);
                c3x = c.rescale(c.mult_plain_mul(c3x, coeff_mul(base + 3, c3x.get_level())), c3_scale);
                auto x2 = drop(xp.at(2).copy(), target + 1);
                auto term = c.rescale(c.relinearize(c.mult(c3x, x2)), scale);
                if (initialized)
                    acc = c.add(acc, term);
                else
                    acc = move(term);
                initialized = true;
            }
            if (initialized) {
                node.ct = coeff_add(acc, base);
                node.has_ct = true;
            } else
                node.constant = base;
            nodes.push_back(move(node));
        }
        int combine_power = baby_steps;
        while (nodes.size() > 1) {
            vector<Node> next;
            for (size_t i = 0; i < nodes.size(); i += 2) {
                if (i + 1 == nodes.size()) {
                    next.push_back(move(nodes[i]));
                    continue;
                }
                Node left = move(nodes[i]), right = move(nodes[i + 1]);
                if (!left.has_ct)
                    throw runtime_error("Poly2D empty left baby polynomial");
                auto giant = drop(xp.at(combine_power).copy(), left.level + 1);
                CkksCiphertext term;
                if (right.has_ct) {
                    auto rc = drop(right.ct.copy(), left.level + 1);
                    term = c.rescale(c.relinearize(c.mult(rc, giant)), left.scale);
                } else {
                    term = c.rescale(c.mult_plain_mul(giant, coeff_mul(right.constant, giant.get_level())), left.scale);
                }
                left.ct = c.add(drop(left.ct.copy(), left.level), term);
                next.push_back(move(left));
            }
            nodes = move(next);
            combine_power *= 2;
        }
        if (nodes.empty() || !nodes[0].has_ct)
            throw runtime_error("Poly2D Stockmeyer has no result");
        result[xi] = move(nodes[0].ct);
    });
    return result;
}

Feature2DEncrypted Poly2D::run_stockmeyer(CkksContext& ctx, const Feature2DEncrypted& x) {
    Feature2DEncrypted result(&ctx, get_stockmeyer_output_level());
    result.shape = x.shape;
    result.skip = x.skip;
    result.n_channel = x.n_channel;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.data = run_core_stockmeyer(ctx, x.data);
    result.level = result.data.empty() ? get_stockmeyer_output_level() : result.data[0].get_level();
    return result;
}

Array<double, 3> Poly2D::run_plaintext(const Array<double, 3>& x) const {
    Array<double, 3> result(x.get_shape());
    for (int ch = 0; ch < (int)x.get_shape()[0]; ++ch)
        for (int i = 0; i < (int)x.get_shape()[1]; ++i)
            for (int j = 0; j < (int)x.get_shape()[2]; ++j) {
                double v = x.get(ch, i, j), p = 0;
                for (int k = order; k >= 0; --k)
                    p = p * v + weight.get(k, ch);
                result.set(ch, i, j, p);
            }
    return result;
}
