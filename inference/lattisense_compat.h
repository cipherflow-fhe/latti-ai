// lattisense v6 兼容层：为旧 API 提供等价包装。
// latti-ai-new 全量迁移到 v6 原生 API 后本文件应删除。
#pragma once

#include <cxx_sdk_v2/cxx_fhe_task.h>

#include <complex>
#include <vector>

namespace fhe_ops_lib {
using CkksCompressedCiphertext = CkksCiphertext;
}  // namespace fhe_ops_lib

namespace ls_compat {

using namespace fhe_ops_lib;

// encode_ringt：构造 ringt metadata 的 CkksPlaintext
inline CkksPlaintext encode_ringt(CkksContext& ctx,
                                  const std::vector<double>& values,
                                  double scale) {
    const auto& param = ctx.parameter();
    CkksPlaintext pt(param,
                     plaintext_metadata(/*is_ringt=*/true,
                                        /*is_batched=*/true,
                                        /*level=*/0,
                                        /*log_slots=*/param.log_max_slots(),
                                        /*scale=*/scale));
    ctx.encode(values, pt);
    return pt;
}

inline CkksPlaintext encode_ringt(CkksContext& ctx,
                                  const std::vector<std::complex<double>>& values,
                                  double scale) {
    const auto& param = ctx.parameter();
    CkksPlaintext pt(param, plaintext_metadata(true, true, 0, param.log_max_slots(), scale));
    ctx.encode(values, pt);
    return pt;
}


// encode 三参：返回 CkksPlaintext（关键：返回值，不是 void）
inline CkksPlaintext encode(CkksContext& ctx,
                            const std::vector<double>& values,
                            uint32_t level,
                            double scale) {
    const auto& param = ctx.parameter();
    CkksPlaintext pt(param,
                     plaintext_metadata(/*is_ringt=*/false,
                                        /*is_batched=*/true,
                                        (int)level,
                                        param.log_max_slots(),
                                        scale,
                                        /*is_ntt=*/true));
    ctx.encode(values, pt);
    return pt;
}

// decode 单参：返回 vector<double>
inline std::vector<double> decode(CkksContext& ctx, const CkksPlaintext& pt) {
    std::vector<double> out;
    ctx.decode(pt, out);
    return out;
}

// serialize 丢参数
inline Bytes serialize(const CkksCiphertext& ct) {
    return ct.serialize();
}

// encrypt_symmetric 系列 → 用 encrypt 兜底
inline CkksCiphertext encrypt_symmetric(CkksContext& ctx, const CkksPlaintext& pt) {
    return ctx.encrypt(pt);
}
inline CkksCiphertext encrypt_symmetric_compressed(CkksContext& ctx, const CkksPlaintext& pt) {
    return ctx.encrypt(pt);
}
inline CkksCiphertext encrypt_asymmetric(CkksContext& ctx, const CkksPlaintext& pt) {
    return ctx.encrypt(pt);
}
inline CkksCiphertext compressed_ciphertext_to_ciphertext(const CkksCiphertext& ct) {
    return CkksCiphertext::deserialize(ct.serialize());
}


// ringt_to_pt: v6 里没有独立转换；ringt 转普通 pt 的语义在 encrypt/encode 内部处理。
// 兼容层里做一次序列化拷贝直通。
inline CkksPlaintext ringt_to_pt(CkksContext& /*ctx*/, const CkksPlaintext& pt, int /*level*/) {
    return CkksPlaintext::deserialize(pt.serialize());
}

}  // namespace ls_compat
