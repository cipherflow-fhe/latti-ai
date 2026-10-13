# Python Bindings (pybind11)

通用 Python 绑定，将 C++ `InferenceClient` / `InferenceServer` API 导出为 Python 模块 `latti_inference`。

## 编译

前置条件：latti-ai 项目已编译（`build/` 目录存在）。

```bash
cd inference/interface/python
pip install pybind11
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
make -j$(nproc)
```

编译产物：`build/latti_inference.cpython-3xx-x86_64-linux-gnu.so`

使用 GPU 构建时，在 CMake 配置中指定 `-DLATTI_AI_BUILD=/path/to/latti-ai/build/gpu`；该目录也需要以 Release 编译。

## 使用

```python
import latti_inference

# 客户端：密钥生成、加密、解密
client = latti_inference.InferenceClient('/path/to/task/client')
client.setup()  # 默认逐密钥生成，发布 /path/to/task/client/eval_context.bin
encrypted = client.encrypt({'input': 'input.csv'})  # -> dict[str, bytes]

# 服务端：模型加载、密文推理
server = latti_inference.InferenceServer('/path/to/task/server', use_gpu=True)
server.import_eval_context_file('/path/to/task/client/eval_context.bin')
server.load_model()
result = server.evaluate(encrypted)  # -> dict[str, bytes]

# 解密
decrypted = client.decrypt(result)  # -> dict[str, DecryptedOutput]
output = decrypted['output']
print(output.output[: output.num_outputs])
```

## 部署

部署时同时携带 Python 扩展、`liblattisense.so` 及其动态库依赖，保持运行时包的相对目录结构，并将 `inference/interface/python/build` 加入 Python 模块搜索路径。`client.export_eval_context_to_file(path)` 可将已有客户端的评估键逐键生成到指定文件；默认 `setup()` 后使用文件接口传递评估键。
