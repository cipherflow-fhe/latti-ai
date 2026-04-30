#pragma once
#include "poly_relu_base.h" // 必须包含，复用 BSGS 逻辑
#include "../data_structs/feature.h"
#include "../data_structs/feature0d.h"
#include "layer.h"
#include <memory>
#include <vector>

namespace cxx_sdk_v2 {
// 基础版无自举
class SoftmaxLayerbase : public Layer {
public:
    SoftmaxLayerbase(const ls::CkksParameter& param,
                 const Array<double, 1>& exp_coeffs,
                 const Array<double, 1>& inv_coeffs,
                 uint32_t n_channel_per_ct,
                 uint32_t level_in,
                 int exp_order = 0,
                 int inv_order = 0,
                 int ciphertext_skip = 1, int total_n_channel = 4, double input_max = 2.0, double input_min = 0.0);

    // LattiAI 要求的标准入口
    Feature0DEncrypted run(ls::CkksContext& ctx, const Feature0DEncrypted& x);

    uint32_t get_depth() const;

private:
    uint32_t n_channel_per_ct_;
    int exp_order_, inv_order_;
    // 新成员：输入密文 skip（每个逻辑通道在物理槽位间隔）
    int ciphertext_skip_ = 1;
    // 槽位旋转累加
    ls::CkksCiphertext sum_slots(ls::CkksContext& ctx, ls::CkksCiphertext ct, uint32_t skip);
    // 内部辅助：将首槽结果广播至所有有效通道
    ls::CkksCiphertext broadcast_slots(ls::CkksContext& ctx, ls::CkksCiphertext ct, uint32_t skip);

    std::vector<double> inv_chebyshev_coeffs_;  // 新增
    double inv_domain_a_;                        // 新增
    double inv_domain_b_;                        // 新增
    std::vector<double> exp_chebyshev_coeffs_;
    double exp_domain_a_;
    double exp_domain_b_;
};

} // namespace cxx_sdk_v2
