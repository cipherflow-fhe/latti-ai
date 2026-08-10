#include <iostream>
#include <vector>
#include <cmath>
#include "fhe_ops_lib/fhe_lib_v2.h"

using namespace fhe_ops_lib;

int main() {
    uint64_t N = 1 << 14;
    CkksParameter param = CkksParameter::create_parameter(N);
    int level = 5;
    CkksContext ctx = CkksContext::create_random_context(param, level);
    double scale = 1LL << 40;

    // 明文输入
    double x_val = 2.0;
    std::vector<double> plain = {x_val};
    CkksPlaintext pt_x = ctx.encode(plain, level, scale);
    CkksCiphertext ct_x = ctx.encrypt_asymmetric(pt_x);

    // 计算 x^2
    CkksCiphertext3 x2_3 = ctx.mult(ct_x, ct_x);
    CkksCiphertext x2 = ctx.relinearize(x2_3);
    x2 = ctx.rescale(x2, scale);
    // x^2/2
    CkksPlaintext half = ctx.encode({0.5}, level, scale);
    CkksCiphertext x2_half = ctx.mult_plain(x2, half);
    // 1 + x
    CkksPlaintext one = ctx.encode({1.0}, level, scale);
    CkksCiphertext one_plus_x = ctx.add_plain(ct_x, one);
    // exp_approx = (1 + x) + x^2/2
    CkksCiphertext exp_ct = ctx.add(one_plus_x, x2_half);

    // 求和（只有一个元素，直接移动）
    CkksCiphertext sum_ct = std::move(exp_ct);

    // 倒数近似：常数 0.5
    CkksPlaintext inv_plain = ctx.encode({0.5}, level, scale);
    CkksCiphertext inv_ct = ctx.encrypt_asymmetric(inv_plain);

    // 归一化
    CkksCiphertext3 res3 = ctx.mult(sum_ct, inv_ct);
    CkksCiphertext res = ctx.relinearize(res3);
    res = ctx.rescale(res, scale);

    // 解密
    CkksPlaintext pres = ctx.decrypt(res);
    std::vector<double> decoded = ctx.decode(pres);
    std::cout << "Homomorphic Softmax (single element, approx): " << decoded[0] << std::endl;
    std::cout << "Plain Softmax (single element): 1.0" << std::endl;

    return 0;
}
