#!/bin/bash
echo "====== 1. 检查编译产物 ======"
ls -R /root/latti-ai/runs/cifar10_squeezenet/compile_output/task/

echo ""
echo "====== 2. 验证 ONNX 文件 ======"
python3 -c "import onnx; m=onnx.load('/root/latti-ai/runs/cifar10_squeezenet/compile_output/squeezenet_dual_constant.onnx'); print('ONNX 有效，共有', len(m.graph.node), '个节点')"

echo ""
echo "====== 3. 重新生成推理指令 ======"
python3 inference/interface/gen_mega_ag.py --task-dir /root/latti-ai/runs/cifar10_squeezenet/compile_output/task

echo ""
echo "====== 验证完成 ======"
echo "上述步骤若无报错，证明 SqueezeNet FHE 编译和指令生成成功。"
