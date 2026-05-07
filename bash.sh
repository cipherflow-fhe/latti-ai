#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# 1. 编译 Go SDK 动态库
echo "==== Building Go SDK Shared Library ===="
cd inference/lattisense/fhe_ops_lib/lattigo/go_sdk
go build -buildmode=c-shared -o liblattigo.so main.go bootstrap.go c_struct_import_export.go conversion.go multiparty.go
cd "$PROJECT_ROOT"

# 2. 运行 Python 端的计算图生成单元测试
echo "==== Running Python Graph Generation Tests ===="
python3 -m unittest inference.unittests.test_gen_layers.TestLayerExport.test_softmax_graph_gen_single -v
python3 -m unittest inference.unittests.test_gen_layers.TestLayerExport.test_softmax_graph_gen -v

# 3. 编译 C++ 测试工程
echo "==== Compiling C++ Unit Tests ===="
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make test_fhe_layers_hetero -j$(nproc)
cd "$PROJECT_ROOT"

# 4. 运行 C++ 算子推理单元测试
echo "==== Running C++ Softmax Tests ===="
./build/inference/unittests/test_fhe_layers_hetero "*softmax_layer*"

echo "==== All Softmax Tests Completed Successfully ===="
