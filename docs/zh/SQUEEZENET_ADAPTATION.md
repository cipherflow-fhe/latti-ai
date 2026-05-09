# SqueezeNet FHE 适配技术记录

## 模型修改
- 三处 MaxPool2d(kernel=3,stride=2) 改为 kernel=2,stride=2
- 删除分类器末尾的 ReLU 激活
- Kaiming 初始化 mode 从 fan_out 改为 fan_in
- 分类器池化改用 AdaptiveAvgPool2d(1)

## ONNX 图修复
- 用后处理脚本将 Shape 节点替换为 Constant 节点
- 将 Sub 算子转换为 Add + 负数常量
- 将标量初始化器转为 Constant 节点，解决 Mul 常量缺失

## 编译器增强
- 在 onnx_to_json.py 中增加收集 ONNX 初始化器常量的逻辑
- 添加形状补全函数，防止 JSON 缺少 shape 字段

## 编译结果
- FHE 编译成功，128 组实验全部通过
- 指令生成正常完成

## 推理情况
- 服务端加载模型时出现 JSON 类型错误（含 null 值）
- 已通过脚本补全缺失字段、覆盖官方加密参数文件，但仍未彻底解决