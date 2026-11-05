// Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "fhe_api.h"
#include "inference_task/inference_process.h"
#include "cxx_sdk_v2/cxx_fhe_task.h"
#include "cxx_sdk_v2/cxx_argument.h"
#include <atomic>
#include <cstdlib>
#include <mutex>

using namespace std;
using namespace lattisense;

namespace {

void require_close(CkksContext& ctx, const FeatureMatEncrypted& actual,
                   const Array<double, 2>& expected, uint32_t d, uint32_t heads,
                   uint32_t cols, int level) {
    double scale = ctx.parameter().default_scale();
    REQUIRE(actual.level == level);
    for (const auto& ct : actual.data) {
        REQUIRE(ct.level() == level);
        REQUIRE(ct.scale() / scale == Approx(1.0).epsilon(1e-9));
    }
    auto unpacked = actual.par_block_col_major_unpack(expected.get_shape()[0], cols, d, heads);
    REQUIRE(unpacked.get_shape() == expected.get_shape());
    double error = 0;
    for (uint32_t i = 0; i < expected.get_size(); i++)
        error = max(error, abs(unpacked.get(i) - expected.get(i)));
    INFO("max_abs_error=" << error);
    REQUIRE(isfinite(error));
    REQUIRE(error < 2e-4);
}

// Use the same PT dispatcher as the production encode_pt executor. The checking
// callback observes actual CT metadata after any GPU/CPU ABI transfer.
vector<CkksCiphertext> run_lazy(CkksContext& ctx, const string& path,
                               FeatureMatEncrypted& a, FeatureMatEncrypted& b,
                               vector<Layer*> layers, int level, bool binary, size_t output_count = 0) {
    unordered_map<string, ExecutorFunc> executors;
    atomic<size_t> encoded{0}, checked{0};
    mutex probe_mutex;
    executors["encode_pt"] = [&](ExecutionContext& c, unordered_map<NodeId, any>& in, const ComputeNode& node) {
        auto* context = c.get_arithmetic_context<CkksContext>();
        auto data = any_cast<shared_ptr<CustomData>>(in.at(node.input_nodes[0]->id));
        auto* layer = static_cast<Layer*>(data->get_typed_data<void>());
        const auto& attrs = node.custom_prop->attributes;
        const string op_class = attrs.at("op_class");
        auto ln_pt = [&](auto* ln) {
            return ln->generate_pt(*context, attrs.value("pt_idx", 0), attrs.value("bi", 0),
                                   attrs.value("bj", 0), attrs.value("g", 0));
        };
        CkksPlaintext pt;
        if (op_class == "ParBlockColMajorLNStats")
            pt = ln_pt(static_cast<ParBlockColMajorLNStats*>(layer));
        else if (op_class == "ParBlockColMajorLNXCentered")
            pt = ln_pt(static_cast<ParBlockColMajorLNXCentered*>(layer));
        else if (op_class == "ParBlockColMajorLNMinimaxInit")
            pt = ln_pt(static_cast<ParBlockColMajorLNMinimaxInit*>(layer));
        else if (op_class == "ParBlockColMajorLNGoldschmidt")
            pt = ln_pt(static_cast<ParBlockColMajorLNGoldschmidt*>(layer));
        else if (op_class == "ParBlockColMajorLNAffine")
            pt = ln_pt(static_cast<ParBlockColMajorLNAffine*>(layer));
        else
            pt = generate_matrix_parameter(*context, layer, attrs);
        if (node.custom_prop->type == "encode_secret") {
            lock_guard<mutex> lock(probe_mutex);
            vector<double> values;
            ctx.decode(pt, values);
            auto encoded = fhe_api::encode(ctx, values, attrs.at("parameter_level").get<int>(), pt.scale());
            in[node.output_nodes[0]->id] = make_shared<CkksCiphertext>(ctx.encrypt(encoded));
        } else {
            in[node.output_nodes[0]->id] = make_shared<CkksPlaintext>(move(pt));
        }
        ++encoded;
    };
    executors["encode_secret"] = executors["encode_pt"];
    executors["check_contract"] = [&](ExecutionContext& ec, unordered_map<NodeId, any>& in, const ComputeNode& node) {
        auto* context = ec.get_arithmetic_context<CkksContext>();
        if (!context)
            throw runtime_error("contract check: missing CKKS context");
        const auto& attrs = node.custom_prop->attributes;
        auto x = any_cast<shared_ptr<CkksCiphertext>>(in.at(node.input_nodes[0]->id));
        auto second = node.input_nodes.size() == 1 ? 0 : 1;
        double rhs_scale;
        int rhs_level;
        if (attrs.at("plaintext").get<bool>()) {
            auto pt = any_cast<shared_ptr<CkksPlaintext>>(in.at(node.input_nodes[second]->id));
            rhs_scale = pt->scale();
            rhs_level = pt->is_ringt() ? x->level() : pt->level();
        } else {
            auto y = any_cast<shared_ptr<CkksCiphertext>>(in.at(node.input_nodes[second]->id));
            rhs_scale = y->scale();
            rhs_level = y->level();
        }
        if (x->level() != rhs_level)
            throw runtime_error("contract check: arithmetic operands have different levels");
        if (attrs.at("operation") != "mult" && abs(x->scale() / rhs_scale - 1) > 1e-9)
            throw runtime_error("contract check: add/sub operands have different scales");
        in[node.output_nodes[0]->id] = make_shared<CkksCiphertext>(x->copy());
        ++checked;
    };
    vector<CkksCiphertext> output;
    for (size_t i = 0; i < (output_count ? output_count : a.data.size()); i++)
        output.push_back(CkksCiphertext(ctx.parameter(), level));
    vector<vector<CustomData>> sources(layers.size());
    vector<CxxVectorArgument> args{{"a", &a.data}};
    for (size_t i = 0; i < layers.size(); i++) {
        sources[i].emplace_back(static_cast<void*>(layers[i]));
        args.push_back({i == 0 ? "layer" : "layer" + to_string(i), &sources[i]});
    }
    if (binary)
        args.push_back({"b", &b.data});
    args.push_back({"output", &output});
    bool gpu = getenv("PCM_BERT_TEST_GPU") != nullptr;
#ifdef INFERENCE_SDK_ENABLE_GPU
    if (gpu) {
        FheTaskGpu task(path);
        task.bind_custom_executors(executors);
        task.run(&ctx, args);
    } else
#else
    if (gpu)
        throw runtime_error("GPU test requested in a CPU build");
#endif
    {
        FheTaskCpu task(path);
        task.bind_custom_executors(executors);
        task.run(&ctx, args);
    }
    INFO("encoded=" << encoded << " checked=" << checked);
    REQUIRE(encoded > 0);
    if (!gpu)
        REQUIRE(checked > 0);
    return output;
}

}  // namespace

TEST_CASE("PCM row/chunk seeds restore independent matrix diagonals", "[pcm_seed]") {
    auto param = CkksParameter::create_parameter(13, RingType::ConjugateInvariant);
    const uint32_t L = 3, slots = param.max_slots();
    auto ctx = CkksContext::create_random_context(param);
    ctx.gen_rotation_keys();
    REQUIRE(double(param.q()[L]) != param.default_scale());
    // Repeated chunks, head padding, multiple groups, u=1 and non-square BSGS.
    for (auto cfg : vector<array<uint32_t, 4>>{{8, 3, 24, 24}, {8, 3, 24, 43},
            {8, 3, 43, 24}, {32, 7, 224, 2}, {64, 3, 192, 192}}) {
        auto [d, heads, rows, cols] = cfg;
        CAPTURE(d, heads, rows, cols);
        uint32_t hp = 1;
        while (hp < heads) hp *= 2;
        const uint32_t u = min(hp, slots / (d*d)), chunk = u*d*d, rho = slots/chunk;
        if (u == 1) hp = heads;
        const uint32_t groups = hp/u, n = heads*d;
        const uint32_t kr = (rows+n-1)/n, kc = (cols+n-1)/n, blocks = max(kr, kc);
        const uint32_t requests = blocks*groups*heads*d, capacity = d*rho;
        const uint32_t baby = uint32_t(ceil(sqrt(double(d))));
        Array<double, 2> w({rows, cols});
        for (uint32_t i = 0; i < rows; ++i)
            for (uint32_t j = 0; j < cols; ++j) w.set(i, j, (1.0+i*cols+j)/(rows*cols));
        ParBlockColMajorCPMM layer(param, {d+3, d}, w, d, heads, L);
        REQUIRE(layer.row_chunk_seed_count() == (requests+capacity-1)/capacity);
        vector<vector<double>> seeds;
        for (uint32_t sid = 0; sid < layer.row_chunk_seed_count(); ++sid)
            seeds.push_back(layer.row_chunk_seed_values(sid));
        double integer_error = 0, recovery_error = 0, diagonal_error = 0;
        for (uint32_t ridx = 0; ridx < requests; ++ridx) {
            uint32_t index = ridx;
            const uint32_t k = index%d; index /= d;
            const uint32_t bp = index%heads; index /= heads;
            const uint32_t g = index%groups, mb = index/groups;
            const uint32_t sid = ridx/capacity, position = ridx%capacity;
            vector<double> expected(slots);
            for (uint32_t s = 0; s < slots; ++s) {
                const uint32_t j = (s%chunk)/(d*u), h = s%u;
                const uint32_t row = (kr>1 ? mb*n : 0)+(g*u+h)*d+(j+k%baby)%d;
                const uint32_t col = (kc>1 ? mb*n : 0)+bp*d+(j+d-(k/baby)*baby%d)%d;
                expected[s] = g*u+h < heads && row < rows && col < cols ? w.get(row, col) : 0;
                const uint32_t source = (position/d)*chunk+(position%d+d*j)*u+h;
                integer_error = max(integer_error, abs(seeds[sid][source]-expected[s]));
            }
            // Boundary requests exercise every seed, chunk and both signed row directions.
            if (ridx%capacity != 0 && ridx%capacity != capacity-1 && ridx+1 != requests &&
                ridx != d-1 && ridx != d) continue;
            const auto& seed_pt_2 = layer.generate_row_chunk_seed_pt(ctx, sid);
            std::vector<double> seed_values_2;
            ctx.decode(seed_pt_2, seed_values_2);
            auto encoded_seed_2 = fhe_api::encode(ctx, seed_values_2, L+1, seed_pt_2.scale());
            auto seed = ctx.encrypt(std::move(encoded_seed_2));
            auto restored = ctx.mult(seed,
                ParBlockColMajorCPMM::selection_for_config(ctx, d, heads, L, position));
            for (uint32_t bit = 0; (1u<<bit) < d; ++bit) {
                const int step = int((1u<<bit)*u);
                restored = ctx.add(restored, ctx.rotate(restored, position%d & (1u<<bit) ? step : -step));
            }
            for (uint32_t bit = 0; (1u<<bit) < rho; ++bit) {
                const int step = int((1u<<bit)*chunk);
                restored = ctx.add(restored, ctx.rotate(restored, position/d & (1u<<bit) ? step : -step));
            }
            restored = ctx.rescale(restored);
            REQUIRE(restored.level() == L);
            REQUIRE(restored.scale()/param.q()[L] == Approx(1.0).epsilon(1e-9));
            vector<double> values; ctx.decode(ctx.decrypt(restored), values);
            const auto& seed_pt_3 = layer.generate_diag_pt(ctx, mb, g, bp, k);
            std::vector<double> seed_values_3;
            ctx.decode(seed_pt_3, seed_values_3);
            auto encoded_seed_3 = fhe_api::encode(ctx, seed_values_3, L, seed_pt_3.scale());
            vector<double> reference; ctx.decode(encoded_seed_3, reference);
            for (uint32_t s = 0; s < slots; ++s) {
                recovery_error = max(recovery_error, abs(values[s]-expected[s]));
                diagonal_error = max(diagonal_error, abs(values[s]-reference[s]));
            }
        }
        INFO("recovery=" << recovery_error << " diagonal=" << diagonal_error);
        REQUIRE(integer_error == 0);
        REQUIRE(recovery_error < 1e-4);
        REQUIRE(diagonal_error < 1e-4);
        for (uint32_t t = requests%capacity; t && t < capacity; ++t)
            for (uint32_t j = 0; j < d; ++j)
                for (uint32_t h = 0; h < u; ++h)
                    REQUIRE(seeds.back()[(t/d)*chunk+(t%d+d*j)*u+h] == 0);
    }
}

TEST_CASE("PCM BERT LayerNorm stages preserve actual scale and lazy arithmetic contracts", "[pcm_bert]") {
    auto param = CkksParameter::create_parameter(14, RingType::ConjugateInvariant);
    const int L = 6;
    auto ctx = CkksContext::create_random_context(param);
    ctx.gen_rotation_keys();
    REQUIRE(double(param.q()[L]) != param.default_scale());
    const char* task_root = getenv("PCM_BERT_TASK_ROOT");
    uint32_t cfg_index = 0;
    for (auto cfg : vector<array<uint32_t, 3>>{{35, 12, 32}, {11, 3, 8}}) {
        auto [rows, heads, d] = cfg;
        Duo shape{rows, heads * 2 * d}, head{rows, 2 * d};
        ParBlockColMajorLayout layout(shape, head, heads, d, param.max_slots());
        Array<double, 2> x({rows, shape[1]}), scalar({rows, 1}), other({rows, 1});
        for (uint32_t i = 0; i < rows; i++) {
            scalar.set(i, 0, 0.5 + 0.002 * i);
            other.set(i, 0, 0.3 + 0.003 * i);
            for (uint32_t j = 0; j < shape[1]; j++)
                x.set(i, j, 0.03 + 0.001 * i + 0.0001 * j);
        }
        FeatureMatEncrypted full(&ctx, L), compact(&ctx, L), b(&ctx, L);
        full.par_block_col_major_pack(x, d, heads, 2 * d, param.default_scale());
        auto pack_compact = [&](FeatureMatEncrypted& target, const Array<double, 2>& values) {
            target.data.clear();
            for (uint32_t bi = 0; bi < layout.br; bi++) {
                vector<double> slots(param.max_slots());
                for (uint32_t s = 0; s < slots.size(); s++) {
                    uint32_t row = bi * d + (s % layout.chunk / layout.u) % d;
                    if (row < rows)
                        slots[s] = values.get(row, 0);
                }
                target.data.push_back(ctx.encrypt(fhe_api::encode(ctx, slots, L, param.default_scale())));
            }
        };
        pack_compact(compact, scalar);
        pack_compact(b, other);
        auto check = [&](const string& name, vector<CkksCiphertext> actual,
                         const Array<double, 2>& expected, Layer* layer,
                         FeatureMatEncrypted& input, int cost, bool binary, bool is_compact) {
            INFO("stage=" << name << " cfg=" << cfg_index);
            auto verify = [&](const vector<CkksCiphertext>& cts) {
                REQUIRE(cts.size() == (is_compact ? layout.br : layout.total_cts()));
                for (const auto& ct : cts) {
                    REQUIRE(ct.level() == L - cost);
                    REQUIRE(ct.scale() / param.default_scale() == Approx(1.0).epsilon(1e-9));
                }
                if (is_compact) {
                    double error = 0;
                    for (uint32_t bi = 0; bi < cts.size(); bi++) {
                        std::vector<double> values; ctx.decode(ctx.decrypt(cts[bi]), values);
                        for (uint32_t s = 0; s < values.size(); s++) {
                            uint32_t row = bi * d + (s % layout.chunk / layout.u) % d;
                            if (row < rows)
                                error = max(error, abs(values[s] - expected.get(row, 0)));
                        }
                    }
                    INFO("compact max_abs_error=" << error);
                    REQUIRE(isfinite(error));
                    REQUIRE(error < 2e-4);
                } else {
                    FeatureMatEncrypted matrix(&ctx, L - cost);
                    matrix.shape = shape;
                    matrix.head_shape = head;
                    matrix.matmul_block_size = d;
                    for (const auto& ct : cts)
                        matrix.data.push_back(ct.copy());
                    require_close(ctx, matrix, expected, d, heads, 2 * d, L - cost);
                }
            };
            verify(actual);
            if (task_root) {
                auto lazy = run_lazy(ctx, string(task_root) + "/" + to_string(cfg_index) + "/" + name,
                                     input, b, {layer}, L - cost, binary, actual.size());
                verify(lazy);
                double error = 0;
                for (uint32_t i = 0; i < actual.size(); i++) {
                    std::vector<double> eager_values; ctx.decode(ctx.decrypt(actual[i]), eager_values);
                    std::vector<double> lazy_values; ctx.decode(ctx.decrypt(lazy[i]), lazy_values);
                    for (uint32_t s = 0; s < eager_values.size(); s++)
                        error = max(error, abs(eager_values[s] - lazy_values[s]));
                }
                REQUIRE(error < 2e-4);
            }
        };
        ParBlockColMajorLNStats stats(param, shape, d, heads, L, 1e-5, 0.7);
        stats.prepare_weight();
        check("lnstats", stats.run(ctx, full), stats.run_plaintext(x), &stats, full, 4, false, true);
        ParBlockColMajorLNXCentered center(param, shape, d, heads, L);
        center.prepare_weight();
        check("lncenter", center.run(ctx, full), center.run_plaintext(x), &center, full, 2, false, false);
        ParBlockColMajorLNMinimaxInit init(param, d, L, 0.8, -0.2, 0.03);
        init.prepare_weight();
        check("lninit", init.run(ctx, compact.data), init.run_plaintext(scalar), &init, compact, 2, false, true);
        ParBlockColMajorLNGoldschmidt gs(param, d, L);
        gs.prepare_weight();
        check("lngs", gs.run(ctx, compact.data, b.data), gs.run_plaintext(scalar, other),
              &gs, compact, 3, true, true);
        Array<double, 1> gamma({shape[1]}), beta({shape[1]});
        for (uint32_t j = 0; j < shape[1]; j++) {
            gamma.set(j, 0.7 + 0.0001 * j);
            beta.set(j, -0.01 + 0.0001 * j);
        }
        ParBlockColMajorLNAffine affine(param, shape, d, heads, L, 0.9, move(gamma), move(beta));
        affine.prepare_weight();
        check("lnaffine", affine.run(ctx, full.data, b.data).data, affine.run_plaintext(x, other),
              &affine, full, 2, true, false);
        cfg_index++;
    }
}

TEST_CASE("PCM BERT primitives, eager/lazy parity and arithmetic contracts", "[pcm_bert]") {
    auto param = CkksParameter::create_parameter(14, RingType::ConjugateInvariant);
    const int L = 6;
    auto ctx = CkksContext::create_random_context(param);
    ctx.gen_rotation_keys();
    REQUIRE(double(param.q()[L]) != param.default_scale());
    const char* task_root = getenv("PCM_BERT_TASK_ROOT");
    if (!task_root)
        WARN("native checks only; generate tasks with gen_pcm_bert_layers.py and set PCM_BERT_TASK_ROOT for lazy checks");
    uint32_t cfg_index = 0;
    for (auto cfg : vector<array<uint32_t, 5>>{{35, 12, 37, 32, 1}, {11, 3, 9, 8, 2}}) {
        auto [rows, heads, cols, d, K] = cfg;
        Duo shape{rows, heads * cols * K}, head{rows, cols};
        CAPTURE(rows, heads, cols, d, K);
        Array<double, 2> x({rows, shape[1]}), y({rows, shape[1]});
        for (uint32_t i = 0; i < rows; i++)
            for (uint32_t j = 0; j < shape[1]; j++) {
                x.set(i, j, 0.03 + 0.001 * i + 0.0001 * j);
                y.set(i, j, 0.5 + 0.002 * i + 0.0002 * j);
            }
        FeatureMatEncrypted a(&ctx, L), b(&ctx, L);
        a.par_block_col_major_pack(x, d, heads, cols, param.default_scale());
        b.par_block_col_major_pack(y, d, heads, cols, param.default_scale());
        auto check = [&](const string& name, FeatureMatEncrypted actual,
                         const Array<double, 2>& expected, vector<Layer*> layers, bool binary = false) {
            INFO("layer=" << name);
            int output_level = name == "invinit" ? L : L - 2;
            if (name.rfind("poly", 0) == 0)
                output_level = L - MatPolyBase::compute_stockmeyer_level_cost(stoul(name.substr(4)));
            else if (name == "gelu_composition")
                output_level = L - 5;
            require_close(ctx, actual, expected, d, heads, cols, output_level);
            if (task_root) {
                auto result = run_lazy(ctx, string(task_root) + "/" + to_string(cfg_index) + "/" + name,
                                       a, b, layers, output_level, binary);
                auto eager_values = actual.par_block_col_major_unpack(rows, cols, d, heads);
                actual.data = move(result);
                require_close(ctx, actual, expected, d, heads, cols, output_level);
                require_close(ctx, actual, eager_values, d, heads, cols, output_level);
            }
        };
        ParBlockColMajorMultipleSquare square(param, shape, head, heads, L, d);
        square.prepare_weight();
        check("square", square.run(ctx, a), square.run_plaintext(x), {&square});
        ParBlockColMajorHeadColSum sum(param, shape, head, heads, L, d);
        sum.prepare_weight();
        // Poison every padding slot, including padded heads, rows and columns.
        ParBlockColMajorLayout layout(shape, head, heads, d, param.max_slots());
        for (uint32_t idx = 0; idx < a.data.size(); idx++) {
            auto noise = layout.constant(idx, 0.0);
            for (uint32_t slot = 0; slot < noise.size(); slot++)
                if (layout.column(idx, slot) < 0)
                    noise[slot] = 7;
            a.data[idx] = ctx.add(a.data[idx], ctx.encrypt(fhe_api::encode(ctx, noise, L, param.default_scale())));
        }
        check("headsum", sum.run(ctx, a), sum.run_plaintext(x), {&sum});
        a.par_block_col_major_pack(x, d, heads, cols, param.default_scale());
        ParBlockColMajorInverseInit init(param, shape, head, heads, L, d);
        init.prepare_weight();
        check("invinit", init.run(ctx, a), init.run_plaintext(x), {&init});
        ParBlockColMajorInverseIter iter(param, shape, head, heads, L, d);
        iter.prepare_weight();
        check("inviter", iter.run(ctx, a, b), iter.run_plaintext(x, y), {&iter}, true);
        ParBlockColMajorCtMul mul(param, shape, head, heads, L, d);
        mul.prepare_weight();
        check("ctmul", mul.run(ctx, a, b), mul.run_plaintext(x, y), {&mul}, true);
        ParBlockColMajorPolyMultCt gelu(param, shape, head, heads, L, d);
        gelu.prepare_weight();
        check("gelumult", gelu.run(ctx, a, b), gelu.run_plaintext(x, y), {&gelu}, true);
        for (uint32_t order : {1u, 2u, 3u, 7u, 15u}) {
            Array<double, 2> coeff({order + 1, shape[1]});
            for (uint32_t k = 0; k <= order; k++)
                for (uint32_t j = 0; j < shape[1]; j++)
                    coeff.set(k, j, K == 1 && k % 3 == 0 ? 0 : (0.1 + 0.00001 * j) / (k + 1));
            ParBlockColMajorPoly poly(param, shape, head, heads, L, move(coeff), order, d);
            poly.prepare_weight();
            check("poly" + to_string(order), poly.run(ctx, a), poly.run_plaintext(x), {&poly});
        }
        // Complete three-polynomial GELU composition plus the ordered (t,x)
        // product. Low degrees keep this unit test bootstrap-free; actual BERT
        // order-15 coefficients and refresh scheduling are covered end to end.
        vector<unique_ptr<ParBlockColMajorPoly>> polys;
        FeatureMatEncrypted intermediate(&ctx, L);
        intermediate.par_block_col_major_pack(x, d, heads, cols, param.default_scale());
        auto expected = x.copy();
        vector<Layer*> stages;
        for (uint32_t i = 0; i < 3; i++) {
            Array<double, 2> coeff({2, shape[1]});
            for (uint32_t j = 0; j < shape[1]; j++) {
                coeff.set(0, j, 0.02 * i);
                coeff.set(1, j, 0.3 + 0.1 * i);
            }
            polys.push_back(make_unique<ParBlockColMajorPoly>(param, shape, head, heads, L - i,
                                                              move(coeff), 1, d));
            polys.back()->prepare_weight();
            expected = polys.back()->run_plaintext(expected);
            intermediate = polys.back()->run(ctx, intermediate);
            stages.push_back(polys.back().get());
        }
        ParBlockColMajorPolyMultCt tail(param, shape, head, heads, L - 3, d);
        tail.prepare_weight();
        auto original = a.drop_level(3);
        expected = tail.run_plaintext(expected, x);
        stages.push_back(&tail);
        check("gelu_composition", tail.run(ctx, intermediate, original), expected, stages);
        cfg_index++;
    }
}
