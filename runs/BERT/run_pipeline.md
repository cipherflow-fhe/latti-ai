# GPU Pipeline

```bash
cmake -B build -DINFERENCE_SDK_ENABLE_GPU=ON -DLATTISENSE_CUDA_ARCH=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
python training/run_compile.py --input=./runs/BERT/model/trained_poly.onnx --output=./runs/BERT --style=multiplexed --config training/config/BERT_base_config.json
python inference/interface/gen_mega_ag.py --task-dir ./runs/BERT/task
./build/examples/inference --task-dir ./runs/BERT/task --input runs/BERT/input.csv --verify --gpu
```
