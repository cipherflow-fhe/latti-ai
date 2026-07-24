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

## 使用

```python
import latti_inference

# 客户端：密钥生成、加密、解密
client = latti_inference.InferenceClient('/path/to/task/client')
client.setup()
eval_ctx = client.export_eval_context()  # -> bytes
encrypted = client.encrypt({'input': 'input.csv'})  # -> dict[str, bytes]

# 服务端：模型加载、密文推理
server = latti_inference.InferenceServer('/path/to/task/server', use_gpu=True)
server.import_eval_context(eval_ctx)
server.load_model()
result = server.evaluate(encrypted)  # -> dict[str, bytes]

# 解密
decrypted = client.decrypt(result)  # -> dict[str, DecryptedOutput]
output = decrypted['output']
print(output.output[: output.num_outputs])
```

## 部署

将编译产物 `.so` 拷贝到应用项目的 `lib/` 目录即可使用。
