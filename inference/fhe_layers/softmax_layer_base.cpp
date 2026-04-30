#include "softmax_layer_base.h"
#include <algorithm>
#include <cmath>
#include <cassert>

namespace cxx_sdk_v2 {
// 基础版无自举
// 对齐密文级别
static void align_levels(ls::CkksContext& ctx, ls::CkksCiphertext& a, ls::CkksCiphertext& b) {
    int target = std::min(a.get_level(), b.get_level());
    while (a.get_level() > target) a = ctx.drop_level(a);
    while (b.get_level() > target) b = ctx.drop_level(b);
}

// 广播一个槽位的值到所有槽位（假设初始有效值在槽位0）
ls::CkksCiphertext SoftmaxLayerbase::broadcast_slots(ls::CkksContext& ctx, ls::CkksCiphertext ct, uint32_t skip) {
    // 同时确保在调用处传入正确的 skip（通常为 Feature0DEncrypted::skip）
    ls::CkksCiphertext res = std::move(ct);
    for (uint32_t step = 1; step < n_channel_per_ct_; step <<= 1) {
        uint32_t physical_shift = step * skip;
        auto rotated = ctx.rotate(res, -(int)physical_shift);
        align_levels(ctx, res, rotated);
        res = ctx.add(res, rotated);
    }
    return res;
}
// 乘法 + 重线化 + 重缩放
static ls::CkksCiphertext mul_relin_rescale(ls::CkksContext& ctx, ls::CkksCiphertext a, ls::CkksCiphertext b) {
    align_levels(ctx, a, b);
    ls::CkksCiphertext3 prod3 = ctx.mult(a, b);
    ls::CkksCiphertext prod = ctx.relinearize(prod3);
    prod = ctx.rescale(prod, ctx.get_parameter().get_default_scale());
    return prod;
}
// double target_scale = a.get_scale() * b.get_scale() / ctx.get_parameter().get_q(a.get_level());

SoftmaxLayerbase::SoftmaxLayerbase(const ls::CkksParameter& param,
                           const Array<double, 1>& exp_coeffs,
                           const Array<double, 1>& inv_coeffs,
                           uint32_t n_channel_per_ct,
                           uint32_t level_in,
                           int exp_order,
                           int inv_order,
                           int ciphertext_skip, int total_n_channel, double input_min, double input_max)
    : Layer(param), n_channel_per_ct_(n_channel_per_ct), exp_order_(exp_order), inv_order_(inv_order), ciphertext_skip_(ciphertext_skip) {

    exp_domain_a_ = input_min;  // 根据实际输入范围设定
    exp_domain_b_ = input_max;
    uint64_t exp_coeff_len = exp_coeffs.get_shape()[0];
    exp_chebyshev_coeffs_.resize(exp_coeff_len);
    for (uint64_t i = 0; i < exp_coeff_len; ++i)
        exp_chebyshev_coeffs_[i] = exp_coeffs[i]; // 直接存 Chebyshev 系数


    inv_domain_a_ = (double)total_n_channel * std::exp(input_min) + 0.1;
    inv_domain_b_ = (double)total_n_channel * std::exp(input_max) + 0.5;
    uint64_t inv_coeff_len = inv_coeffs.get_shape()[0];
    inv_chebyshev_coeffs_.resize(inv_coeff_len);
    for (uint64_t i = 0; i < inv_coeff_len; ++i) {
        inv_chebyshev_coeffs_[i] = inv_coeffs[i];   // 直接存 Chebyshev 系数
    }
    assert(inv_coeff_len >= 3 && "inv_coeffs must contain at least [a, b, c0]"); // 系数至少三个，要合规
}

// 跨槽求和（将 n_channel_per_ct 个逻辑槽位累加到第一个槽位）
ls::CkksCiphertext SoftmaxLayerbase::sum_slots(ls::CkksContext& ctx, ls::CkksCiphertext ct, uint32_t skip) {
    ls::CkksCiphertext res = std::move(ct);
    for (uint32_t step = 1; step < n_channel_per_ct_; step <<= 1) {
        uint32_t physical_shift = step * skip;
        auto rotated = ctx.rotate(res, (int)physical_shift);   // 正向旋转
        align_levels(ctx, res, rotated);
        res = ctx.add(res, rotated);
    }
    return res;
}

Feature0DEncrypted SoftmaxLayerbase::run(ls::CkksContext& ctx, const Feature0DEncrypted& x) {
    uint64_t slots = ctx.get_parameter().get_n() / 2;
    double base_scale = ctx.get_parameter().get_default_scale();
    auto input_plain = x.unpack();
    printf("[DEBUG] Input to exp: %.6f %.6f %.6f %.6f\n",
        input_plain[0], input_plain[1], input_plain[2], input_plain[3]);
    // Step 1: 计算 exp(x_i)
    Feature0DEncrypted exp_x(&ctx, 0);
    exp_x.n_channel        = x.n_channel;
    exp_x.n_channel_per_ct = x.n_channel_per_ct;
    exp_x.skip             = x.skip;
    exp_x.pack_type        = x.pack_type;
    // 得到的还是一个密文向量
    for (size_t i = 0; i < x.data.size(); ++i) {
        auto ct = ctx.poly_eval_chebyshev(
            x.data[i],
            exp_chebyshev_coeffs_,
            exp_domain_a_,
            exp_domain_b_,
            slots,  
            base_scale
        );
        if (i == 0) exp_x.level = ct.get_level();
        exp_x.data.push_back(std::move(ct));
    }
    printf("[LEVEL] after exp: %d (consumed %d)\n", 
           exp_x.data[0].get_level(), x.data[0].get_level() - exp_x.data[0].get_level());
    
    // 检查输入非空
    assert(!exp_x.data.empty());
    // ★ 调试：验证 exp 结果
    auto exp_plain = exp_x.unpack();
    printf("[DEBUG] After exp: %.6f %.6f %.6f %.6f\n",
           exp_plain[0], exp_plain[1], exp_plain[2], exp_plain[3]);
    // Step 2: 对每个 ciphertext 先在槽内归约（sum_slots），再把这些归约结果跨密文相加得到全局和
    // global_sum是一个全局和密文，只关心他的第0槽
    // reduced是单密文和，是每个exp_x.data内部槽位的和，存储在第0槽
    // 用first标记第一次循环，直接赋值给global_sum，后续循环则累加到global_sum（此时仍是第0槽有效）
    ls::CkksCiphertext global_sum;
    bool first = true;
    for (auto& ct : exp_x.data) {
        // 先在密文内将该 ciphertext 的多个通道累加到其首槽
        ls::CkksCiphertext reduced = sum_slots(ctx, ct.copy(), x.skip);
        if (first) {
            auto pt = ctx.decrypt(reduced);
            auto decoded_slots = ctx.decode(pt);
            printf("[DEBUG] After sum_slots[0]: slot0=%.6f, slot1=%.6f\n",
                   decoded_slots[0], decoded_slots[1]);
            global_sum = reduced.copy();
            first = false;
        } else {
            align_levels(ctx, global_sum, reduced);
            global_sum = ctx.add(global_sum, reduced);
        }
    }
    // ★ 调试：验证 global_sum 槽0
    auto pt_sum = ctx.decrypt(global_sum);
    auto slots_sum = ctx.decode(pt_sum);
    printf("[DEBUG] After all sum[0]: slot0=%.6f, slot1=%.6f\n",
                   slots_sum[0], slots_sum[1]);
    // mask：encode时level必须与global_sum一致
    std::vector<double> mask_vec(slots, 0.0);
    mask_vec[0] = 1.0;
    auto mask_pt = ctx.encode(mask_vec, global_sum.get_level(), base_scale);
    global_sum = ctx.mult_plain(global_sum, mask_pt);
    global_sum = ctx.rescale(global_sum, base_scale);  

    // 构造全局和的 Feature0DEncrypted sum_feat，也是一个密文向量，只是当前只有第一个密文值是有效的
    Feature0DEncrypted sum_feat(&ctx, global_sum.get_level());
    sum_feat.n_channel = 1;
    sum_feat.n_channel_per_ct = 1;
    sum_feat.skip = 1; // 单槽布局
    sum_feat.pack_type = exp_x.pack_type;
    sum_feat.data.push_back(std::move(global_sum));
    sum_feat.level = sum_feat.data[0].get_level();

    assert(sum_feat.data.size() == 1);

    printf("[DEBUG] inv_domain: a=%.4f b=%.4f\n", inv_domain_a_, inv_domain_b_);
    printf("[LEVEL] before inv (sum_feat): %d\n", sum_feat.level);

    ls::CkksCiphertext inv_sum = ctx.poly_eval_chebyshev(sum_feat.data[0], inv_chebyshev_coeffs_, 
                                inv_domain_a_, inv_domain_b_, slots, base_scale);
    printf("[LEVEL] after inv: %d (consumed %d)\n", inv_sum.get_level(), sum_feat.level - inv_sum.get_level());

    auto pt_inv = ctx.decrypt(inv_sum);
    auto slots_inv = ctx.decode(pt_inv);
    printf("[DEBUG] inv_sum (chebyshev): slot0=%.6f (expected ~%.6f)  slot1=%.6f\n",
           slots_inv[0], 1.0 / slots_sum[0], slots_inv[1]);  // slots_sum 来自前面解密的全局和

    // 引入迭代计算提高精度
    auto w = sum_feat.data[0].copy();      // x 的副本
    auto y = inv_sum.copy();               // 初始猜测值的副本  
    // 动态水位对齐 (Critical Step!)，如果 x 和 y_init 的 Level 不等，必须降到一致
    if (w.get_level() > y.get_level()) {
        w = ctx.drop_level(w, w.get_level() - y.get_level());
    } else if (y.get_level() > w.get_level()) {
        y = ctx.drop_level(y, y.get_level() - w.get_level());
    }
    // 增加范围处理，能够支持超过2的输入，但是会额外消耗一层
    CkksCiphertext3 a_3 = ctx.mult(w, y);
    CkksCiphertext a_new = ctx.relinearize(a_3);
    w = ctx.rescale(a_new, base_scale);

    // goldschmidt迭代，一次迭代消耗一层
    for (int i = 0; i < 2; ++i) {
        // 1. 计算 r = 2.0 - w
        // 构造匹配 w 当前 Level 和 Scale 的常数 2.0
        std::vector<double> two_vec(slots, 2.0);
        std::vector<double> one_vec(slots, 1.0);
        CkksPlaintext two_pt = ctx.encode(two_vec, w.get_level(), w.get_scale());
        CkksPlaintext one_pt = ctx.encode(one_vec, w.get_level(), w.get_scale());
        auto zero_ct = ctx.sub(w, w);  // 0 (密文)
        auto minus_w = ctx.sub(zero_ct, w);  // 0-w=-w（密文）
        auto r = ctx.add_plain(minus_w, two_pt);  // r = 2.0 - w

        // 2. 更新 w = w * r (使 w 逐渐逼近 1)
        CkksCiphertext3 w_3 = ctx.mult(w, r);
        CkksCiphertext w_new = ctx.relinearize(w_3);
        w_new = ctx.rescale(w_new, base_scale);

        // 3. 更新 y = y * r (使 y 逐渐逼近 1/x)
        CkksCiphertext3 y_3 = ctx.mult(y, r);
        CkksCiphertext y_new = ctx.relinearize(y_3);
        y_new = ctx.rescale(y_new, base_scale);

        // 4. 步进赋值
        w = std::move(w_new);
        y = std::move(y_new);
    }
    // 最后的 y 就是更高精度的 1/sum 密文，赋值给 inv_sum 以便后续使用
    inv_sum = std::move(y);
    
    // 调试输出（可保留）
    auto pt_inv_after = ctx.decrypt(inv_sum);
    auto slots_inv_after = ctx.decode(pt_inv_after);
    printf("[DEBUG] inv_sum (chebyshev): slot0=%.6f (expected ~%.6f)  slot1=%.6f\n",
           slots_inv_after[0], 1.0 / slots_sum[0], slots_inv_after[1]);  // slots_sum 来自前面解密的全局和

    auto mask_pt2 = ctx.encode(mask_vec, inv_sum.get_level(), base_scale);
    inv_sum = ctx.mult_plain(inv_sum, mask_pt2);
    inv_sum = ctx.rescale(inv_sum, base_scale);

    auto pt_inv_mask = ctx.decrypt(inv_sum);
    auto slots_inv_mask = ctx.decode(pt_inv_mask);
    printf("[DEBUG] inv_sum (after mask): slot0=%.6f (expected ~%.6f)  slot1=%.6f\n",
           slots_inv_mask[0], 1.0 / slots_sum[0], slots_inv_mask[1]);  // slots_sum 来自前面解密的全局和
    
    auto pt_bts = ctx.decrypt(inv_sum);
    auto slots_bts = ctx.decode(pt_bts);
    printf("[DEBUG] after bts_2: slot0=%.6f (expected ~%.6f)  slot1=%.6f\n",
           slots_bts[0], 1.0 / slots_sum[0], slots_bts[1]);

    ls::CkksCiphertext broadcast_inv = broadcast_slots(ctx, std::move(inv_sum), x.skip);
    pt_sum = ctx.decrypt(broadcast_inv);
    slots_sum = ctx.decode(pt_sum);
    printf("[DEBUG] inv_sum broadcast_slots: slot0=%.6f, slot1=%.6f\n", slots_sum[0], slots_sum[1]);
    // Step 5: 归一化 - 对每个 ciphertext 做乘法 (exp_ct * broadcast_inv)
    Feature0DEncrypted result(&ctx, 0);
    result.n_channel = x.n_channel;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.skip = x.skip;
    result.pack_type = exp_x.pack_type;

    for (size_t i = 0; i < exp_x.data.size(); ++i) {
        // 直接使用同一个 broadcast_inv 与每个 exp ciphertext 相乘
        ls::CkksCiphertext inv_arg = (i + 1 == exp_x.data.size())
            ? std::move(broadcast_inv)
            : broadcast_inv.copy();
        ls::CkksCiphertext exp_arg = (i + 1 == exp_x.data.size())
            ? std::move(exp_x.data[i])
            : exp_x.data[i].copy();
        auto prod = mul_relin_rescale(ctx, std::move(exp_arg), std::move(inv_arg));

        auto pt_mut = ctx.decrypt(prod);
        auto slots_mut = ctx.decode(pt_mut);
        printf("[DEBUG] after last mult: slot0=%.6f, slot1=%.6f\n", slots_mut[0], slots_mut[1]);

        if (i == 0) result.level = prod.get_level();
        result.data.push_back(std::move(prod));
    }
   
    return result;
}

} // namespace cxx_sdk_v2
