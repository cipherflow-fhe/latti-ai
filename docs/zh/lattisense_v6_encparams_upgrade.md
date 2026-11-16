# 参数加密分支本地集成（2026-11-06）

在 `feat/poly-maxpooling` 保留已有 PolyMaxPool 工作，合入
`origin/feat/v6-adapt-encparams` 的 `9619f42`。
Lattisense 固定为 `83e58218437eee1afb60f47f786e80a020eede63`，
Lattigo 固定为 `22bff324e146968c186106524c06cea607462b05`。
递归子模块按上游锁定版本更新；本次仅本地集成。

## 适配内容

- 保留 `--maxpool-config`、alpha/B/upper_bound、ONNX padding、阶段拆分、DP 和 live-set 元数据。
- 新版 `MatPolyBase` 与 MaxPool 的 `PolyBase` 并存，矩阵路径沿用上游实现。
- MaxPool Generator 统一使用 `frontend.*`，避免重复模块造成 CKKS 参数和节点类型不一致。
- ring-t 明文 metadata 使用 level 0，编码 scale 取实际使用层的 q；完整 RNS 系数保留各自的 level/scale。
- `plaintext_eager` 保留阶段参数绑定；`plaintext_lazy` 和 `encrypted_offline`
  通过输入为空的 `encode_pt` 在运行端生成公开的固定多项式系数、归一化常数及布局 mask。
  这些常数不进入加密模型参数存储，不作为额外在线输入。
- 普通推理与参数加密 Runner 都支持相同的 MaxPool 编码描述。
- 回归使用新版 CI residual 参数 `PN15Default`（32768 slots、12 个乘法层）。
  旧参数实验报告和旧 `/tmp` 任务不代表本次新版验证结果。

## 验证

- CPU Release 全量构建通过，目录 `/tmp/latti-v3-build`。
- 21 项 MaxPool 测试通过，包含 Retina 空间布局的真实 DP → Generator → DAG/manifest，
  以及 alpha14 阶段在明文 eager/lazy、参数加密 eager/runtime_lazy 下的图生成。
- 8 项 MaxPool 编译器测试通过。
- 16 项编译/部署/范围归一化测试通过。
- `[ckks_copy]` 通过。
- `[maxpool_public]` 通过：在原生 v6 密文上验证 ring-t/full-RNS 的常数、稀疏 mask、
  prefix mask 乘法及解密结果，共 49159 项断言。

复现 Python 测试时使用：

```bash
export PYTHONPATH=.:inference/lattisense:training:training/model_compiler
.venv/bin/python -m unittest discover -s inference/unittests -t . -p 'test_*maxpool*.py'
.venv/bin/python -m unittest discover -s training/model_compiler/unittests -p 'test_maxpool*.py'
.venv/bin/python -m unittest inference.unittests.test_compile_model_variant inference.unittests.test_server_provisioned_runner_generator inference.unittests.test_range_norm_maxpool
```

本次没有使用 135 服务器的数据执行六任务端到端验收，没有构建或验证 GPU，
也没有重新测量完整 MaxPool 比较树的 bootstrap 精度。
参数加密模式的新公开常数路径覆盖编译器生成的 `maxpool2d_*` 阶段；
历史 fused MaxPool 任务应重新编译为阶段形式后使用该部署方式。
