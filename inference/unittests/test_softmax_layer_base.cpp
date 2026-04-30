#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "softmax_layer_base.h"
#include "fixture.hpp"        // 提供 CkksCpuFixture
#include "ut_util.h"
#include <cmath>
#include <numeric>
#include "data_structs/feature0d.h"
#include <vector>
#include <functional>
#include <stdexcept>
#include <random>
#include <string>
using namespace std;
using namespace cxx_sdk_v2;

// 基础版无自举
// 计算函数在 [a,b] 上的 Chebyshev 插值系数
// 对应 numpy.polynomial.Chebyshev.interpolate(func, degree, domain)
static std::vector<double> get_chebyshev_coeffs(
    const std::function<double(double)>& func,
    int degree,
    double a, double b)
{
    int n = degree + 1;  // 插值点数

    // Chebyshev 节点（标准域 [-1,1] 上）
    std::vector<double> nodes(n);
    for (int k = 0; k < n; k++) {
        nodes[k] = std::cos(M_PI * (2*k + 1) / (2.0 * n));
    }

    // 映射到 [a,b] 并计算函数值
    std::vector<double> fx(n);
    for (int k = 0; k < n; k++) {
        double x = 0.5 * (b - a) * nodes[k] + 0.5 * (a + b);
        fx[k] = func(x);
    }

    // 计算 Chebyshev 系数
    std::vector<double> coeffs(n);
    for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += fx[k] * std::cos(M_PI * j * (2*k + 1) / (2.0 * n));
        }
        coeffs[j] = (2.0 / n) * sum;
    }
    coeffs[0] /= 2.0;  // c0 减半，与 numpy 约定一致

    return coeffs;
}

// 封装两种函数类型，对应 Python 版本接口
static std::vector<double> get_chebyshev_coeffs_optimized(
    const std::string& func_type,
    int degree,
    double domain_a, double domain_b)
{
    if (func_type == "reciprocal") {
        return get_chebyshev_coeffs([](double x){ return 1.0 / x; }, degree, domain_a, domain_b);
    } else if (func_type == "exp") {
        return get_chebyshev_coeffs([](double x){ return std::exp(x); }, degree, domain_a, domain_b);
    } else {
        throw std::runtime_error("Unsupported function type: " + func_type);
    }
}

// 明文 softmax
static vector<double> plain_softmax(const vector<double>& x) {
    vector<double> result(x.size());
    double sum_exp = 0.0;
    for (double v : x) sum_exp += std::exp(v);
    for (size_t i = 0; i < x.size(); ++i) result[i] = std::exp(x[i]) / sum_exp;
    return result;
}

// ──────────────────────────────────────────────────────────────────────────────
// 复用 CkksCpuFixture 提供的上下文和参数  public CkksCpuFixture
// ──────────────────────────────────────────────────────────────────────────────
class SoftmaxTestFixturebase : public CkksN65536Fixture   {
public:
    SoftmaxTestFixturebase(
        double input_min = -2.0,
        double input_max = 0.0,
        uint32_t n_ch = 4,
        int exp_deg = 7,
        int inv_deg = 4)
        : CkksN65536Fixture () {
        max_level = context.get_parameter().get_max_level();

        double inv_a = (double)n_ch * std::exp(input_min) + 0.1;
        double inv_b = (double)n_ch * std::exp(input_max) + 0.5;
        auto exp_vec = get_chebyshev_coeffs_optimized("exp", exp_deg, input_min, input_max);
        auto inv_vec = get_chebyshev_coeffs_optimized("reciprocal", inv_deg, inv_a, inv_b);

        exp_coeffs = std::make_shared<Array<double,1>>(make_array(exp_vec));
        inv_coeffs = std::make_shared<Array<double,1>>(make_array(inv_vec));
        exp_order = exp_deg;
        inv_order = inv_deg;
    }
    static Array<double, 1> make_array(const std::vector<double>& vec) {
        Array<double, 1> arr({static_cast<uint64_t>(vec.size())});
        for (size_t i = 0; i < vec.size(); ++i) arr[static_cast<uint64_t>(i)] = vec[i];
        return arr;  // 触发移动构造
    }

    Feature0DEncrypted encrypt_input(const vector<double>& input, int level_in,
                                  uint32_t n_ch_per_ct, uint32_t skip_in = 1) {
        uint64_t slots = context.get_parameter().get_n() / 2;
        uint32_t n_ct  = (input.size() + n_ch_per_ct - 1) / n_ch_per_ct;

        Feature0DEncrypted feat(&context, level_in);
        feat.n_channel        = input.size();
        feat.n_channel_per_ct = n_ch_per_ct;
        feat.skip             = skip_in;
        feat.pack_type        = 0;
        feat.level            = level_in;

        for (uint32_t ci = 0; ci < n_ct; ci++) {
            // 每个密文填 n_ch_per_ct 个值
            vector<double> slot_vec(slots, 0.0);
            for (uint32_t i = 0; i < n_ch_per_ct; i++) {
                uint32_t idx = ci * n_ch_per_ct + i;
                if (idx < input.size())
                    slot_vec[i * skip_in] = input[idx];
            }
            // 用 pack() 源码里确认的 API
            auto pt = context.encode(slot_vec, level_in, default_scale);
            auto ct = context.encrypt_asymmetric(pt);  // 来自 pack() 第53行
            feat.data.push_back(std::move(ct));
        }
        return feat;
    }
    // 解密 Feature0DEncrypted 为 vector<double>
    vector<double> decrypt_output(const Feature0DEncrypted& feat) {
        Array<double, 1> arr = feat.unpack();
        vector<double> result(feat.n_channel);
        for (uint32_t i = 0; i < feat.n_channel; ++i) result[i] = arr[static_cast<uint64_t>(i)];
        return result;
    }

    // 供普通 TEST_CASE 访问
    ls::CkksContext& get_context() { return context; }
    const ls::CkksParameter& get_param() { return context.get_parameter(); }
// protected:
public:
    std::shared_ptr<Array<double, 1>> exp_coeffs;
    std::shared_ptr<Array<double, 1>> inv_coeffs;
    int max_level;  // 新增声明
    int exp_order, inv_order;
};

// 计时器
struct ScopedTimer {
    using Clock = std::chrono::steady_clock;
    const char* label_;
    Clock::time_point start_;
    explicit ScopedTimer(const char* label) : label_(label), start_(Clock::now()) {}
    ~ScopedTimer() {
        double ms = std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
        printf("[TIMER] %s: %.3f ms\n", label_, ms);
    }
};

// ──────────────────────────────────────────────────────────────────────────────
// TEST CASE 1: 基础功能（单密文，4 通道）
// ──────────────────────────────────────────────────────────────────────────────
TEST_CASE_METHOD(SoftmaxTestFixturebase, "Softmax basic functionality", "[softmax]") {
    printf("[DEBUG] max_level=%d\n", max_level);
    const uint32_t n_channel_per_ct = 4;
    const int level_in = 19;   // 足够容纳深度

    SoftmaxLayerbase layer(context.get_parameter(), *exp_coeffs, *inv_coeffs, n_channel_per_ct, 
                        level_in, exp_order, inv_order, 1, 4, -2.0, 0.0);

    // 随机生成 4 个在 [-2.0, 0.0] 内的输入
    std::mt19937 rng(42);  // 固定种子，方便复现
    std::uniform_real_distribution<double> dist(-2.0, 0.0);  // 设定范围
    std::vector<double> input(4); // 设定输入尺寸
    for (auto& v : input) v = dist(rng);

    printf("[DEBUG] Random input: %.6f %.6f %.6f %.6f\n",
           input[0], input[1], input[2], input[3]);

    // std::vector<double> input = {-2.0, -1.5, -1.0, -0.5};  // 指数和约 0.14+0.22+0.37+0.61=1.34
    auto encrypted_input = encrypt_input(input, level_in, n_channel_per_ct);
    auto result = layer.run(context, encrypted_input);
    auto decrypted = decrypt_output(result);

    auto expected = plain_softmax(input);
    // 统计误差
    auto got_arr = SoftmaxTestFixturebase::make_array(decrypted);
    auto exp_arr = SoftmaxTestFixturebase::make_array(expected);
    auto cmp = compare(exp_arr, got_arr);
    printf("[STATS softmax] max_err=%.2e  rmse=%.2e  max_abs=%.2e\n", cmp.max_error, cmp.rmse, cmp.max_abs);
    constexpr double kRelTol = 5.0e-2;   // 5% relative error
    REQUIRE(cmp.max_error < kRelTol * cmp.max_abs);
    constexpr double kTol = 1e-3;
    for (size_t i = 0; i < input.size(); ++i) {
        INFO("index = " << i);
        CHECK(std::fabs(decrypted[i] - expected[i]) < kTol);
    }
    // 归一化检查
    double sum = std::accumulate(decrypted.begin(), decrypted.end(), 0.0);
    CHECK(std::fabs(sum - 1.0) < kTol);
    
}
// 特殊测试，将输入限制在 ReLU 输出范围，验证 softmax 在非负输入上的表现（更贴近实际推理场景）
TEST_CASE("Softmax after ReLU simulation", "[softmax]") {
    const uint32_t n_channel_per_ct = 4;
    const uint32_t total_inputs = 16;   // 实际输入总数
    const int level_in = 19;
    // 输入范围 [0.0, 4.0]，4 通道，exp 用 7 阶 Chebyshev，倒数用 4 阶
    SoftmaxTestFixturebase fix(-1.0, 3.0, total_inputs, 7, 4);
    // 2. 构建 SoftmaxLayerbase，传入 Chebyshev 系数等参数
    // SoftmaxLayerbase layer(fix.get_param(), *fix.exp_coeffs, *fix.inv_coeffs, n_channel_per_ct,
    //     level_in, fix.exp_order, fix.inv_order,
    //     1,                                              // ciphertext_skip
    //     total_inputs,                                   // total_n_channel
    //     0.0,                                            // input_min
    //     4.0                                             // input_max
    // );
    SoftmaxLayerbase layer(fix.get_param(), *fix.exp_coeffs, *fix.inv_coeffs, n_channel_per_ct,
        level_in, fix.exp_order, fix.inv_order,
        1,                                              // ciphertext_skip
        total_inputs,                                   // total_n_channel
        -1.0,                                            // input_min
        3.0                                             // input_max
    );

    // 模拟ReLU输出（非负值）
    // 随机生成 8 个在 [0.0, 4.0] 内的输入
    std::mt19937 rng(123);  // 不同种子与上面区分
    std::uniform_real_distribution<double> dist(-1.0, 3.0); // 设定实际输入范围，注意要与上面的一致（3个位置）
    std::vector<double> input(total_inputs);
    for (auto& v : input) v = dist(rng);

    printf("[DEBUG] Random input:");
    for (auto v : input) printf(" %.4f", v);
    printf("\n");

    // 加密 → 运行 Softmax 层 → 解密
    auto encrypted = fix.encrypt_input(input, level_in, n_channel_per_ct);
    auto result = layer.run(fix.get_context(), encrypted);
    auto decrypted = fix.decrypt_output(result);
    
    auto expected = plain_softmax(input);
    // 统计误差
    auto got_arr = SoftmaxTestFixturebase::make_array(decrypted);
    auto exp_arr = SoftmaxTestFixturebase::make_array(expected);
    auto cmp = compare(exp_arr, got_arr);
    printf("[STATS softmax] max_err=%.2e  rmse=%.2e  max_abs=%.2e\n", cmp.max_error, cmp.rmse, cmp.max_abs);
    constexpr double kRelTol = 5.0e-2;   // 5% relative error
    REQUIRE(cmp.max_error < kRelTol * cmp.max_abs);
    constexpr double kTol = 1e-3;
    for (size_t i = 0; i < input.size(); ++i) {
        INFO("index = " << i << ", got=" << decrypted[i] << ", expected=" << expected[i]);
        CHECK(std::fabs(decrypted[i] - expected[i]) < kTol);
    }
    double sum = std::accumulate(decrypted.begin(), decrypted.end(), 0.0);
    CHECK(std::fabs(sum - 1.0) < kTol);
}

