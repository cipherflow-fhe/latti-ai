# PolyMaxPool 手动配置与验收入口

保持原模型 `nn.MaxPool2d`。ReLU 的 degree/upper_bound 仍由原有替换入口设置；MaxPool 的参数独立配置，不改变 ReLU 参数，也不自动改变全网 CKKS 预设。

创建 `maxpool.json`，键为 ONNX → pt.json 后的层 ID（必须与实际文件一致）：

```json
{
  "_max_pool_MaxPool": {"alpha": 7, "upper_bound": 1.0, "B": 1.0}
}
```

```bash
python training/run_compile.py --input cnn.onnx --output compiled_cnn \
  --style multiplexed --maxpool-config maxpool.json
```

alpha 从已验证固定表选取 degrees 和 coefficients（当前 7～14），不在线拟合。upper_bound 和 B 必须有限且为正。指定的层必须是原始 3×3、stride=2、padding=1 MaxPool；未知层、额外参数和不支持的配置报错。未指定层沿用已有参数及默认值。

running_max 仍来自校准并写入原始图对应层，本配置不接受它，也不会以测试数据代替校准。缺少校准数据时，本入口不会自动补齐。upper_bound 用于输入缩放，B 用于比较差值的缩放；具体任务的近似效果需用户验证。

编译器在 Lowering 前应用参数，随后复用现有分阶段图和 DP 自举规划。使用分阶段 Generator；旧 fused MaxPool 后端不支持非默认配置。论文理论 depth 不等于具体 Runtime 的 Level 消耗。高 alpha 不保证现有全网 CKKS 参数下的精度，参数选择仍需人工验证。

## 验证记录与限制

- 论文系数：独立验证记录目前位于本地忽略目录 `archive/maxpool/`。
- Generator、alpha=7/14 的 CKKS 复验和 Retina GPU 实验记录位于本地忽略目录 `archive/maxpool/reports/`，不属于运行时核心提交。50-bit 为实验配置，不作为默认参数或安全认证结论。
- 完整 Retina CNN 密文端到端验收仍需独立执行，单算子成功不代表整网完成。

阶段诊断文档保留为历史实验记录。部分诊断脚本对多项式输出 feature 的索引存在选中 carry 分量的疑点，其 P0/P1 归因不能直接作为最终结论。提高 scale 改善早期误差，但现有 alpha14 完整链路的最终误差并未同比下降，不能断言所有误差都来自 scale。

## 本轮上传整理

实验脚本和临时补丁位于本地忽略目录 `archive/maxpool/experimental-scripts/` 与 `archive/maxpool/root-experiments/`。最终提交只应包含核心实现、核心测试和本使用说明；Retina-DKD 及其他外部工作区不属于本次提交。
