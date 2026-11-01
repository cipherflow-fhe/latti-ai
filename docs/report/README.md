# 参数压缩实现与验收记录

依据：[参数密文压缩与 runtime-lazy GPU 验收设计规范](../spec/parameter_compression_runtime_lazy_design.md)。

ResNet20 与 MobileNetV2 均完成 `[1,1]` / `[4,4]` 对照，各组 1 次预热、3 次正式 GPU runtime-lazy 运行，结构与完整 logits 数值门槛通过。

- [代码修改摘要](code_change_summary.md)
- [实测验收结论及限制](acceptance_summary.md)
- [ResNet20 原始指标](resnet20_runtime_lazy.json)、[MobileNetV2 原始指标](mobilenet_runtime_lazy.json)
- [逐层计数、level 与实际存储审计](layout_storage_audit.json)、[逐参数族存储](parameter_family_storage.json)
- [源码与依赖版本/hash](source_provenance.json)、[命令、构建和复查说明](raw/validation_commands.md)
- [依赖恢复与 task 清理记录](raw/cleanup.json)

`raw/` 保存完整 logits、CPU/GPU monitor CSV、命令/returncode、日志与失败重试证据。早期 smoke 不参与正式结论。
验收生成的 task 已删除；没有提交权重、密文、密钥或依赖修改。实现与报告在同一次 latti-ai 提交中，不 push。
