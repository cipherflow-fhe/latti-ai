# PolyMaxPool 手动配置与验收入口

推荐在 PyTorch 中通过 `prepare_for_fhe()` 替换 MaxPool，与 ReLU 一起准备模型：

```python
from training.nn_tools import prepare_for_fhe, PolyMaxPool2d, export_to_onnx

prepare_for_fhe(model, maxpool_alpha=14, maxpool_upper_bound=0.1, maxpool_B=1.0)
# model(x) 此时已经执行可求梯度的明文多项式 MaxPool。
# 根据验证集精度决定是否微调；本函数不自动训练模型。
```

MaxPool 参数独立于 ReLU 的 degree/upper_bound。支持 kernel=3、stride=2、padding=1，拒绝 ceil_mode、return_indices 和 dilation 等不支持的语义。

每个 PolyMaxPool2d 保存标量 `running_max` buffer（默认 1）。训练模式下首次前向记录输入绝对最大值，后续取历史最大值；eval 模式冻结。无需微调时也应使用代表性数据校准输入范围，可保持全模型 eval，仅将 PolyMaxPool2d 设为 train，在 no_grad 下运行校准集，随后全部设为 eval。校准是范围统计，不是权重更新。不要把默认值或导出用随机输入当作实际校准。

明文计算与当前编译 DAG 使用相同的输入缩放、固定符号近似系数、逐轮 B 缩放、4 轮 deferred-half 归约和尺度恢复。越界 tap 复制窗口锚点，与现有密文访问计划一致；所有输入符号均适用。计算为多项式近似，不等于精确 MaxPool。应先检查明文任务精度，再决定是否微调。高阶多项式在 float32 下仍可能有数值误差。

ONNX 保留 `nn_tools::PolyMaxPool2d` 自定义节点及 alpha、upper_bound、B、冻结的 running_max。读取后展开计算阶段，交给全图统一 Level/Bootstrap 规划。导出 helper 中精确 MaxPool 仅用于追踪输出形状；实际 PyTorch forward 使用多项式，自定义节点由编译器解释，通用 ONNX Runtime 不能直接执行它。

## 原始 ONNX 的兼容入口

已有的普通 ONNX MaxPool 仍可通过下面的配置入口处理。ReLU 参数不受影响，也不自动改变全网 CKKS 预设。

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

普通 ONNX 的 running_max 仍来自校准并写入原始图对应层，本配置不接受它，也不会以测试数据代替校准。缺少校准数据时，本入口不会自动补齐。upper_bound 用于输入缩放，B 用于比较差值的缩放；具体任务的近似效果需用户验证。

编译器在 Lowering 前应用参数，随后复用现有分阶段图和 DP 自举规划。使用分阶段 Generator；旧 fused MaxPool 后端不支持非默认配置。论文理论 depth 不等于具体 Runtime 的 Level 消耗。高 alpha 不保证现有全网 CKKS 参数下的精度，参数选择仍需人工验证。

## 验证记录与限制

- 论文系数：独立验证记录目前位于本地忽略目录 `archive/maxpool/`。
- Generator、alpha=7/14 的 CKKS 复验和 Retina GPU 实验记录位于本地忽略目录 `archive/maxpool/reports/`，不属于运行时核心提交。50-bit 为实验配置，不作为默认参数或安全认证结论。
- 完整 Retina CNN 密文端到端验收仍需独立执行，单算子成功不代表整网完成。

阶段诊断文档保留为历史实验记录。部分诊断脚本对多项式输出 feature 的索引存在选中 carry 分量的疑点，其 P0/P1 归因不能直接作为最终结论。提高 scale 改善早期误差，但现有 alpha14 完整链路的最终误差并未同比下降，不能断言所有误差都来自 scale。

## 本轮上传整理

实验脚本和临时补丁位于本地忽略目录 `archive/maxpool/experimental-scripts/` 与 `archive/maxpool/root-experiments/`。最终提交只应包含核心实现、核心测试和本使用说明；Retina-DKD 及其他外部工作区不属于本次提交。
