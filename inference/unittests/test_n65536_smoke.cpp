// 放在 inference/test/ 下，CMakeLists.txt 加一行 add_executable
#include <cstdio>
#include <chrono>
#include <cxx_sdk_v2/cxx_fhe_task.h>

namespace ls = cxx_sdk_v2;

int main() {
    using clk = std::chrono::steady_clock;

    auto t0 = clk::now();
    printf("[1/4] create_parameter(65536)...\n"); fflush(stdout);
    auto param = ls::CkksParameter::create_parameter(65536);
    printf("    max_level = %d, default_scale = %.0f\n",
           param.get_max_level(), param.get_default_scale());

    auto t1 = clk::now();
    printf("[2/4] create_random_context...\n"); fflush(stdout);
    auto ctx = ls::CkksContext::create_random_context(param);

    auto t2 = clk::now();
    std::vector<int32_t> rots = {+1, -1, +2, -2, +4, -4, +8, -8};
    printf("[3/4] gen_rotation_keys_for_rotations({+1,-1,+2,-2,+4,-4,+8,-8}, false)\n");
    printf("这一步最容易 OOM\n"); fflush(stdout);
    // ctx.gen_rotation_keys();
    ctx.gen_rotation_keys_for_rotations(rots, /*include_swap_rows=*/false);

    auto t3 = clk::now();
    printf("[4/4] one encode/encrypt/mult/rescale round-trip...\n"); fflush(stdout);
    uint64_t slots = param.get_n() / 2;
    std::vector<double> v(slots, 0.5);
    auto pt = ctx.encode(v, param.get_max_level(), param.get_default_scale());
    auto ct = ctx.encrypt_asymmetric(pt);
    auto ct2 = ctx.mult(ct, ct);
    auto ct3 = ctx.relinearize(ct2);
    auto ct4 = ctx.rescale(ct3, param.get_default_scale());
    auto pt_out = ctx.decrypt(ct4);
    auto out = ctx.decode(pt_out);
    printf("    decoded[0] = %.6f (expect 0.25)\n", out[0]);

    auto ms = [](auto a, auto b){
        return std::chrono::duration<double, std::milli>(b-a).count();
    };
    printf("\n[TIMING]\n");
    printf("  create_parameter   : %.1f ms\n", ms(t0, t1));
    printf("  create_context     : %.1f ms\n", ms(t1, t2));
    printf("  gen_rotation_keys  : %.1f ms\n", ms(t2, t3));
    printf("  one mult round     : %.1f ms\n", ms(t3, clk::now()));
    return 0;
}