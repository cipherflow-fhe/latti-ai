# 旧符号 -> 新符号映射表

## 类型映射

| 旧 | 新 | 说明 |
|---|---|---|
| CkksPlaintextRingt | CkksPlaintext + plaintext_metadata(true, ...) | metadata.is_ringt=true，level=0，is_ntt=false |
| CkksPlaintextMul  | CkksPlaintext + plaintext_metadata(false, ..., is_ntt=true) | metadata.is_ringt=false，is_ntt=true，指定 level |
| CkksBtpContext    | CkksContext + enable_bootstrapping=true | create_random_context(param, type, true) 或 set_enable_bootstrapping(true) |
| CkksBtpParameter  | CkksParameter | BTP 参数由 CkksContext 内部派生，无需单独类型 |
| NodeIndex         | NodeId (= std::string) | mega_ag.h |
| BfvCiphertext3    | BfvCiphertext(param, level, degree=2) | 用 degree 参数 |
| BfvPlaintextRingt | BfvPlaintext + metadata.is_ringt=true | 同 CkksPlaintextRingt |
| BfvPlaintextMul   | BfvPlaintext + metadata.is_ntt=true | 同 CkksPlaintextMul |

## 方法映射

| 旧 | 新 | 说明 |
|---|---|---|
| ctx.ringt_to_mul(pt, level) | 删除，直接 ctx.mult(ct, pt) | mult 根据 pt.metadata 自动处理 ringt/mul |
| param.get_max_level() | param.max_level() | |
| param.get_default_scale() | param.default_scale() | |
| ctx.get_parameter() | ctx.parameter() | |
| btp_ctx.bootstrap(x) | ckks_ctx.bootstrap(x) | 需 ctx.enable_bootstrapping()==true |
| ctx.get_copy(tid) | （需确认）| 无对应，可能需上下文按线程复制 |

## 文件名映射

| 旧 | 新 | 说明 |
|---|---|---|
| mega_ag.json | compiled_mega_ag.json + fhe_parameter.json | 拆成图结构 + 参数 |
| task_signature.json | task_signature.json | 保留 |
| phase == "offline" | 无 offline phase | 只有 online |
| ckks_btp_swk | 无 | 自举密钥由 BootstrappingEvaluationKeys 统一管理 |

## 关键 API

### CkksContext 创建

```cpp
// 旧：
CkksBtpParameter btp_param = CkksBtpParameter::create_parameter();
CkksBtpContext btp_ctx = CkksBtpContext::create_random_context(btp_param);

// 新：
CkksParameter param = CkksParameter::create_parameter(log_n);
CkksContext ctx = CkksContext::create_random_context(param, EncryptorType::PUBLIC_KEY, /*enable_bootstrapping=*/true);
// 或
ctx.set_enable_bootstrapping(true);
