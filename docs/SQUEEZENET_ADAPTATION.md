# SqueezeNet FHE 适配技术记录

## 背景与目标

本次工作的目的是验证 Latti-AI 框架对非官方模型的通用支持能力。Latti-AI 是一个全同态加密（FHE）推理框架，官方已提供了 ResNet-20 和 MobileNetV2 两种示例模型。我尝试对其官方未覆盖的 SqueezeNet 模型进行适配，以探究框架在自定义模型上的可行性。

测试环境为 AutoDL 云实例，主要配置如下：
- GPU：NVIDIA RTX 5090
- CUDA 12.8，GCC 12.3.0，CMake 3.30.5
- Python 3.10，PyTorch 2.5.1

## 模型适配过程

我独立编写了适用于 CIFAR-10 数据集的 SqueezeNet 模型定义文件（`squeezenet_cifar10.py`）。原始 SqueezeNet 为 224×224 的 ImageNet 设计，因此进行了以下调整：
- 将第一层卷积核从 7×7 改为 3×3，步长从 2 改为 1，以适配 32×32 的输入尺寸。
- 将分类器中的 `AdaptiveAvgPool2d` 替换为核大小为 3 的 `AvgPool2d`，以避免导出 ONNX 时产生动态 Shape 算子。
- 将原分类器中的全连接层（`view` + `Linear`）改为全卷积结构（`Conv2d` + `AvgPool2d`），进一步确保整个计算图在 FHE 编译时无动态维度操作。

基线训练时，我发现使用官方 ResNet-20 示例中的学习率 0.1 会导致损失出现 NaN，经过调整，将学习率降至 0.001 后训练稳定。最终训练 200 轮，最佳准确率为 81.71%。

我尝试了官方的微调流程以恢复替换算子后的精度，但训练初期损失再次变为 NaN，即使降低 `upper_bound` 到 1.5 或更换多项式模块亦未能解决。因此，最终模型是未经微调、直接替换算子后使用的，预计会有一定的精度损失。

## 算子替换与 ONNX 导出问题

FHE 推理不支持 ReLU 和 MaxPool 等非线性操作。利用框架提供的 `replace_activation_with_poly` 和 `replace_maxpool_with_avgpool` 工具，我将所有 ReLU 替换为多项式激活（`Simple_Polyrelu`），所有 MaxPool 替换为 AvgPool。

然而，导出的 ONNX 模型在 FHE 编译阶段遇到了多个算子不支持的错误。这些问题需要通过分析 ONNX 图的结构来逐一解决。

### Shape 算子不支持

编译器首次运行时报告 `Current operator Shape is not supported`。Shape 算子通常由 `view` 或自适应池化操作引入。虽然我已经将分类器改为全卷积结构，但导出的 ONNX 中仍存在少量 Shape 节点。

为了解决此问题，我编写了后处理脚本：利用 ONNX 的形状推理功能获取所有中间张量的固定形状，然后将 Shape 节点全部替换为输出已知形状的 Constant 节点。

### Sub 算子不支持

编译器随后报告 `Current operator Sub is not supported`。Sub 节点来自多项式激活的内部实现。我编写了相应的转换脚本：遍历 ONNX 图，将所有 Sub 节点替换为 Add 节点，同时对减法操作的第二个输入取负。若该输入是一个常量初始化器，则直接修改其值为负数；若是中间张量，则插入一个乘法节点将其乘以 -1。

### Mul 节点常量缺失

在修复上述问题后，编译器开始频繁报出 `Mul node ... missing constant or feature input` 的错误。例如 `node_mul_5` 的输入 `val_2` 和 `features_1_c1` 等。经过诊断，这些是多项式系数，它们在 ONNX 中以 initializer 的形式存在，而编译器只识别通过 Constant 节点提供的常量。

解决方法是：扫描所有 Mul 节点的输入，将其中属于标量初始化器的值额外创建为同名的 Constant 节点（同时保留原初始化器以避免影响权重访问）。此外，我注意到编译器内部使用的常量名称存在点号和下划线两种格式（如 `features.1.c1` 与 `features_1_c1`），因此修改了 `onnx_to_json.py`，在构建常量字典时主动收集所有初始化器，并为每个常量注册两种名称格式，确保编译器在任何命名约定下均能找到所需常量。

## 编译器增强

为使 SqueezeNet 能够通过 FHE 编译，我对编译器的前端代码 `training/model_export/onnx_to_json.py` 做了两处增强：

1. **初始化器常量收集**：在原常量字典构建完成后，增加一个遍历步骤，将所有 ONNX 初始化器的值按原名称和下划线别名两种格式存入常量字典。这解决了乘法系数被误判为缺失的问题。

2. **特征形状补全**：添加了一个 `_fill_feature_shapes` 函数，利用 ONNX 的形状推理接口，为特征字典中缺失 `shape` 字段的特征补充形状信息。该函数在写入最终的 `pt.json` 文件之前调用，避免了后续指令生成阶段因缺少 `shape` 字段而失败。

## FHE 编译与指令生成

经过上述修复后，使用 `run_compile.py` 成功完成了 SqueezeNet 的 FHE 编译。编译过程生成了完整的加密计算图（`pt.json`）以及任务配置文件（`task/server/nn_layers_ct_0.json` 等）。随后运行 `gen_mega_ag.py` 也成功生成了底层执行指令。

为了验证这些结果的可复现性，我编写了一个验证脚本 `verify_squeezenet_compile.sh`，它能够检查编译产物的存在性、验证 ONNX 模型的完整性，并重新执行指令生成。该脚本可以用于快速复现我的编译成果。

## 推理失败分析

尽管编译和指令生成均已成功，但在运行加密推理时，服务端加载模型阶段抛出了一个 JSON 类型异常：[json.exception.type_error.302] type must be number, but is array
无论是 GPU 模式还是 CPU 模式，该错误均稳定出现。然而，同环境下官方提供的 ResNet-20 示例在 CPU 模式下可以正常运行并获得 PASS 结果，说明推理框架本身是正常的。

### 诊断尝试

我首先检查了 `nn_layers_ct_0.json` 文件，发现一些特征缺少 `shape` 或 `skip` 字段，于是编写了后处理脚本来补充这些字段。同时，将 SqueezeNet 任务目录下的 `ckks_parameter.json` 和 `task_config.json` 替换为官方示例中的对应文件，以排除配置差异。

为了进一步定位，我编写了类型对比脚本，将官方 ResNet-20 的 `nn_layers_ct_0.json` 与 SqueezeNet 的同名文件进行递归对比。结果发现存在大量字段类型不一致的情况，例如官方中 `output.skip` 为整数，而 SqueezeNet 中为数组；`output.shape` 在官方中不存在，在 SqueezeNet 中是一个数组。我尝试对所有不一致的字段进行类型强制对齐，但错误依然存在。

### 根本原因推断

经过多轮实验和排查，我认为该问题的根源在于 Latti-AI 的编译器在处理 SqueezeNet 这种结构的模型时，生成的 JSON 字段类型与 C++ 推理引擎的预期不完全一致。部分字段（如 `skip`、`shape`）的类型在推理引擎中隐式假定为标量，但 SqueezeNet 更深的计算图导致编译器在某些分支下产生了数组。这是一个框架层面类型系统的不一致问题，需要通过修改编译器或推理引擎的源代码来根治，已经超出了应用层适配的能力范围。

## 文件清单

| 文件 | 说明 |
|------|------|
| `examples/test_cifar10/model/squeezenet_cifar10.py` | SqueezeNet 模型定义 |
| `examples/test_cifar10/model/my_poly.py` | 自定义多项式激活函数（备选） |
| `examples/test_cifar10/train.py` | 修改后的训练脚本（增加 `--arch` 参数） |
| `training/model_export/onnx_to_json.py` | 修改后的编译器前端（常量收集、形状补全） |
| `scripts/verify_squeezenet_compile.sh` | 编译成果验证脚本 |
| `docs/squeezenet-adaptation.md` | 本文档 |

## 总结

通过本次工作，我独立将 SqueezeNet 模型接入了 Latti-AI 框架，并成功通过 FHE 编译和指令生成，但在最终推理阶段因框架内部的类型问题未能完成。这个过程让我对 FHE 推理的编译链、ONNX 模型结构以及跨语言类型系统有了较深的理解，也积累了复杂工程问题的排查经验。希望本文档能为后续尝试适配其他自定义模型的开发者提供一些参考。
