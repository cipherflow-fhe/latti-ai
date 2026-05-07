#include <iostream>
#include <vector>
#include <cmath>
#include "fhe_ops_lib/fhe_lib_v2.h"

using namespace fhe_ops_lib;

int main() {
    // 1. 参数设置
    uint64_t N = 1 << 14;               // 16384
    CkksParameter param = CkksParameter::create_parameter(N);
    int level = 5;
    CkksContext ctx = CkksContext::create_random_context(param, level);
    double scale = 1LL << 40;           // 2^40

    // 2. 明文输入（与设计文档一致）
    std::vector<double> plain_input = {2.0, 1.0, 0.5, 0.2};
    size_t n = plain_input.size();

    // 3. 加密每个元素（独立密文，不使用打包）
    std::vector<CkksCiphertext> ctxt_vec;
    for (double v : plain_input) {
        CkksPlaintext pt = ctx.encode({v}, level, scale);
        ctxt_vec.push_back(ctx.encrypt_asymmetric(pt));
    }

    // 4. 指数近似（二阶泰勒：1 + x + x²/2）
    std::vector<CkksCiphertext> exp_vec;
    for (auto& ct : ctxt_vec) {
        // 计算 x²
        CkksCiphertext3 x2_3 = ctx.mult(ct, ct);
        CkksCiphertext x2 = ctx.relinearize(x2_3);
        x2 = ctx.rescale(x2, 1e-20);
        // 1 + x
        CkksPlaintext one_pt = ctx.encode({1.0}, level, scale);
        CkksCiphertext exp_ct = ctx.add_plain(ct, one_pt);
        // + x²/2
        CkksPlaintext half_pt = ctx.encode({0.5}, level, scale);
        exp_ct = ctx.add(exp_ct, ctx.mult_plain(x2, half_pt));
        exp_vec.push_back(exp_ct);
    }

    // 5. 求和（所有指数密文相加）
    CkksCiphertext sum_ct = exp_vec[0];
    for (size_t i = 1; i < n; ++i) {
        sum_ct = ctx.add(sum_ct, exp_vec[i]);
    }

    // 6. 倒数近似（常数 0.5）
    CkksPlaintext inv_pt = ctx.encode({0.5}, level, scale);
    CkksCiphertext inv_ct = ctx.encrypt_asymmetric(inv_pt);

    // 7. 归一化：每个指数乘以倒数
    std::vector<CkksCiphertext> result_ct;
    for (auto& exp_ct : exp_vec) {
        CkksCiphertext3 mul3 = ctx.mult(exp_ct, inv_ct);
        CkksCiphertext mul = ctx.relinearize(mul3);
        mul = ctx.rescale(mul, 1e-20);
        result_ct.push_back(mul);
    }

    // 8. 解密并输出
    std::cout << "=== Homomorphic Softmax (Independent Test) ===" << std::endl;
    for (size_t i = 0; i < n; ++i) {
        CkksPlaintext pres = ctx.decrypt(result_ct[i]);
        std::vector<double> dec = ctx.decode(pres);
        std::cout << "p" << i << " = " << dec[0] << std::endl;
    }

    // 9. 明文 Softmax 参考
    std::vector<double> exp_plain(n);
    double sum_exp = 0.0;
    for (size_t i = 0; i < n; ++i) {
        exp_plain[i] = std::exp(plain_input[i]);
        sum_exp += exp_plain[i];
    }
    std::cout << "\n=== Plain Softmax ===" << std::endl;
    for (size_t i = 0; i < n; ++i) {
        std::cout << "p" << i << " = " << exp_plain[i] / sum_exp << std::endl;
    }

    return 0;
}
