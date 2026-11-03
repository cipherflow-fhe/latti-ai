# 新版底层集成基线（2026-11-03）

主仓库集成 `origin/feat/lattisense-adapt` 的 `d5fcca9603790768cff3aaccade588eff99f829a`，保留本地 PolyMaxPool 工作。Lattisense 使用该分支指定的 `898bc396cb247426e5c76ab2ae7702558b0c3766`，Lattigo 使用 `22bff324e146968c186106524c06cea607462b05`，GPU core 使用 `aea1ef873aed3e199aa785a0e4f42d8a42b9351f`。这些是本次集成的明确版本，未声明为远端今后的最新提交。

旧版本工作保留在主仓库提交 `7879bd9` 和 Lattisense 分支 `feat/poly-maxpooling-local`。新版 SDK 及其子模块保持上游原样；应用层适配新版统一明文类型、metadata、残余参数和 rescale 接口。

## 已验证

- CPU Release 完整 CMake 构建通过，构建目录 `/tmp/latti-v3-build`。
- `test_e2e_single_layer '[ckks_copy]'` 通过：degree 1、2 密文复制保留 scale 和 level。
- Python 的 MaxPool 手动配置、alpha Generator、stage dispatch、interleaved reduction 共 13 项测试通过。

复现构建：

```bash
cmake -S . -B /tmp/latti-v3-build -DLATTISENSE_ENABLE_GPU=OFF -DCMAKE_BUILD_TYPE=Release
GOCACHE=/tmp/latti-go-v3-cache cmake --build /tmp/latti-v3-build -j4
```

## 后续 MaxPool 验收范围

编译成功和上述测试不能证明完整 MaxPool 密文精度或完整 CNN 验收通过。GPU 未构建或测试。旧 45/50-bit 实验参数、`scripts/prepare_maxpool_ckks_test.py` 与 `scripts/test_maxpool_precision_parameters.py` 仍依赖旧版预设，需迁移至新版 residual 参数设计后再使用。旧 `/tmp` 任务数据应重新生成；历史诊断报告不代表新版运行结果。不要直接将旧 SDK 的固定 bootstrap 参数实现覆盖进新版。
