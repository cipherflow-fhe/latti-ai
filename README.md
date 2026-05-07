# SqueezeNet 全同态加密推理适配与编译器增强

## 项目简介

本项目基于 Latti-AI 框架，独立完成了 SqueezeNet 模型的全同态加密（FHE）推理适配工作。主要工作包括：模型结构适配、ONNX 计算图修复、编译器前端增强，以及 FHE 编译与指令生成。核心推理引擎和密码学库未作修改。

## 环境要求

Python 3.10, PyTorch 2.1+, ONNX 1.16+
GCC 12, CMake 3.30+
CUDA 12.8（可选）
测试环境：AutoDL 云实例，RTX 5090，90GB RAM

## 快速开始

### 验证 SqueezeNet 编译产物与指令生成

bash scripts/verify_squeezenet_compile.sh

### 运行官方 ResNet-20 推理

python3 inference/interface/gen_mega_ag.py --task-dir examples/test_cifar10/task
./build/examples/inference --task-dir examples/test_cifar10/task \
  --input examples/test_cifar10/task/client/img.csv --verify
  
## 原创性说明

以下文件为本人独立编写或实质性修改，修改位置均标注于源码注释中：examples/test_cifar10/model/squeezenet_cifar10.py（SqueezeNet 模型定义）、examples/test_cifar10/model/my_poly.py（自定义多项式激活）、training/model_export/onnx_to_json.py（常量收集与形状补全）、examples/test_cifar10/train.py（增加 --arch 参数）、scripts/verify_squeezenet_compile.sh（验证脚本）、docs/squeezenet-adaptation.md（技术记录）。其余文件来自官方仓库。

## 完成度

模型适配、算子替换、ONNX 图修复、FHE 编译、指令生成均已完成。密文推理未通过，原因定位于编译器与推理引擎的类型系统不一致，需框架底层修复。详见 docs/squeezenet-adaptation.md

## 技术挑战

Shape/Sub 算子不支持	ONNX 后处理消除或等价转换
Mul 常量缺失	初始化器转 Constant 节点，编译器常量收集增强
推理 JSON 类型错误	框架底层类型不一致，已缓解，彻底解决需改源码

## 参考资料

Latti-AI: https://github.com/cipherflow-fhe/latti-ai
SqueezeNet: https://arxiv.org/abs/1602.07360