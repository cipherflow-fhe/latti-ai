# LattiAI API Reference

## Overview

LattiAI provides a complete pipeline for running neural network inference on encrypted data using Fully Homomorphic Encryption (FHE). The pipeline consists of three stages:

```
Adaptation                          Compilation         Inference
(nn_tools + model_export)     →   (model_compiler)  →  (C++ Runtime)

PyTorch model → ONNX → JSON        JSON → ERG          Encrypted
  + poly         graph spec         (compiled DAG)      execution
  activations
```

### Module Map

| Module | Language | Source | Purpose |
|--------|----------|--------|---------|
| `nn_tools` | Python | `training/nn_tools/` | Replace activations with polynomial approximations; replace max pooling with average pooling; export to ONNX and H5 |
| `model_export` | Python | `training/model_export/` | Convert ONNX models to JSON computation graphs (part of Adaptation stage) |
| `model_compiler` | Python | `training/model_compiler/` | Compile JSON graphs into encrypted computation DAG |
| Inference Runtime | C++ | `inference/` | Execute encrypted inference using CKKS FHE |

---

## 1. Model Adaptation

Source: `training/nn_tools/`, `training/model_export/`

### 1.1 Activation Classes

#### RangeNorm2d

```python
RangeNorm2d(num_features=0, upper_bound=3.0, eps=1e-3, momentum=0.1)
```

Per-channel range normalization. Normalizes input to `[-upper_bound, upper_bound]` using a running estimate of the per-channel absolute maximum.

Supports lazy initialization: pass `num_features=0` to defer buffer creation until the first forward call, where the channel count is inferred from `x.shape[1]`.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `num_features` | `int` | `0` | Number of channels (0 for lazy init) |
| `upper_bound` | `float` | `3.0` | Normalization upper bound |
| `eps` | `float` | `1e-3` | Small constant to avoid division by zero |
| `momentum` | `float` | `0.1` | Momentum for running-max update |

---

#### Simple_Polyrelu

```python
Simple_Polyrelu(scale_before=1.0, scale_after=1.0, degree=4, activation='relu')
```

Polynomial activation approximating ReLU or SiLU via Hermite expansion.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `scale_before` | `float` | `1.0` | Input scaling factor |
| `scale_after` | `float` | `1.0` | Output scaling factor |
| `degree` | `int` | `4` | Polynomial degree (2 or 4) |
| `activation` | `str` | `'relu'` | Target activation (`'relu'` or `'silu'`) |

**Raises:** `ValueError` if `activation` is not `'relu'` or `'silu'`, or if `degree` is not 2 or 4.

---

#### RangeNormPoly2d

```python
RangeNormPoly2d(num_features=0, upper_bound=3.0, degree=4, activation='relu')
```

Combined range normalization + polynomial activation. Applies per-channel range normalization, then a polynomial activation, and rescales back. Exported as a single `nn_tools::RangeNormPoly2d` custom op in ONNX.

Supports lazy initialization: pass `num_features=0` (default) to defer buffer creation until the first forward call.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `num_features` | `int` | `0` | Number of channels (0 for lazy init) |
| `upper_bound` | `float` | `3.0` | Normalization upper bound |
| `degree` | `int` | `4` | Polynomial degree (2 or 4) |
| `activation` | `str` | `'relu'` | Target activation (`'relu'` or `'silu'`) |

**Example:**

```python
from nn_tools import RangeNormPoly2d

# Use as a drop-in activation replacement
activation = RangeNormPoly2d(upper_bound=3.0, degree=4, activation='relu')
```

---

### 1.2 Replacement Functions

#### replace_activation_with_poly()

```python
replace_activation_with_poly(model, old_cls=nn.ReLU, new_module_factory=RangeNormPoly2d, upper_bound=3.0, degree=4)
```

Replace all instances of `old_cls` activation with `RangeNormPoly2d` in-place.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | `nn.Module` | required | PyTorch model (modified in-place) |
| `old_cls` | `Type[nn.Module]` | `nn.ReLU` | Activation class to replace. Supported: `nn.ReLU`, `nn.SiLU` |
| `new_module_factory` | `Callable` | `RangeNormPoly2d` | Constructor `(upper_bound, degree, activation) -> nn.Module` |
| `upper_bound` | `float` | `3.0` | Normalization upper bound |
| `degree` | `int` | `4` | Polynomial degree (2 or 4) |

**Returns:** The same model with activations replaced.

**Raises:** `ValueError` if `old_cls` is not `nn.ReLU` or `nn.SiLU`.

**Example:**

```python
from nn_tools import replace_activation_with_poly

model = resnet20()
replace_activation_with_poly(model, old_cls=nn.ReLU)
# or replace SiLU activations
replace_activation_with_poly(model, old_cls=nn.SiLU, degree=4)
```

---

#### replace_activation()

```python
replace_activation(module, old_cls=nn.ReLU, new_module_factory=RangeNormPoly2d,
                   upper_bound=3.0, degree=4, activation='relu')
```

Generic in-place activation replacement. Replace all `old_cls` activations with instances created by `new_module_factory`.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `module` | `nn.Module` | required | Model to modify in-place |
| `old_cls` | `Type[nn.Module]` | `nn.ReLU` | Activation class to replace |
| `new_module_factory` | `Callable` | `RangeNormPoly2d` | Constructor `(upper_bound, degree, activation) -> nn.Module` |
| `upper_bound` | `float` | `3.0` | Normalization upper bound |
| `degree` | `int` | `4` | Polynomial degree |
| `activation` | `str` | `'relu'` | Target activation name (`'relu'` or `'silu'`) |

---

#### replace_maxpool_with_avgpool()

```python
replace_maxpool_with_avgpool(model)
```

Replace all `nn.MaxPool2d` with `nn.AvgPool2d` in-place. 

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | `nn.Module` | required | PyTorch model (modified in-place) |

**Returns:** The same model with MaxPool layers replaced.

**Example:**

```python
from nn_tools import replace_maxpool_with_avgpool

model = resnet18()
replace_maxpool_with_avgpool(model)
```

---

#### count_activations()

```python
count_activations(module, activation_cls=nn.ReLU)
```

Count the number of `activation_cls` instances in `module`.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `module` | `nn.Module` | required | PyTorch model |
| `activation_cls` | `Type[nn.Module]` | `nn.ReLU` | Activation class to count |

**Returns:** `int` — Number of matching activations.

---

### 1.3 Export Functions

#### export_to_onnx()

```python
export_to_onnx(model, save_path, input_size=(1, 3, 32, 32), opset_version=13,
               dynamic_batch=True, remove_identity=True, save_h5=True, verbose=True)
```

Export a PyTorch model to ONNX. BatchNorm is kept as a full operator (not folded into Conv).

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | `nn.Module` | required | PyTorch model |
| `save_path` | `str` | required | Output `.onnx` file path |
| `input_size` | `Tuple[int, ...]` | `(1, 3, 32, 32)` | Input tensor shape |
| `opset_version` | `int` | `13` | ONNX opset version |
| `dynamic_batch` | `bool` | `True` | Enable dynamic batch-size axis |
| `remove_identity` | `bool` | `True` | Remove Identity ops after export |
| `save_h5` | `bool` | `True` | Also save weights to an H5 file |
| `verbose` | `bool` | `True` | Log progress information |

**Returns:** `str` — Path to the saved ONNX file.

---

#### fuse_and_export_h5()

```python
fuse_and_export_h5(model, h5_path, upper_bound=3.0, degree=4, eps=1e-3, verbose=True)
```

Fuse Conv+BN, absorb polynomial scale, and export all weights to H5.

Automatically handles:

1. `Conv2d` + `BatchNorm2d` pairs → fused `(weight, bias)`
2. `RangeNormPoly2d` → per-channel polynomial coefficients
3. `Linear` → `(weight, bias)`
4. Standalone `Conv2d` (no BN) → `(weight, bias)`

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | `nn.Module` | required | Trained model (with `RangeNormPoly2d` activations) |
| `h5_path` | `str` | required | Output H5 file path |
| `upper_bound` | `float` | `3.0` | `RangeNormPoly2d` upper bound |
| `degree` | `int` | `4` | Polynomial degree |
| `eps` | `float` | `1e-3` | `RangeNormPoly2d` epsilon |
| `verbose` | `bool` | `True` | Log progress information |

**Returns:** `str` — `h5_path`.

---

#### save_onnx_weights_to_h5()

```python
save_onnx_weights_to_h5(onnx_path, h5_path=None, verbose=True)
```

Extract all weights from an ONNX model and save to H5.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `onnx_path` | `str` | required | Path to `.onnx` model |
| `h5_path` | `str` | `None` | Output H5 path (default: `<onnx_stem>_weights.h5`) |
| `verbose` | `bool` | `True` | Log progress information |

**Returns:** `str` — Path to the saved H5 file.

---

#### remove_identity_nodes()

```python
remove_identity_nodes(onnx_path)
```

Remove all Identity operators from an ONNX model file.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `onnx_path` | `str` | required | Path to `.onnx` file (modified in-place) |

**Returns:** `int` — Number of removed Identity operators.

---

#### load_h5_weights()

```python
load_h5_weights(h5_path)
```

Load weights from an H5 file.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `h5_path` | `str` | required | Path to H5 file |

**Returns:** `dict` — `{name: numpy_array}` dictionary.

---

### 1.4 ONNX-to-JSON Converter (model_export)

#### onnx_to_json()

```python
onnx_to_json(onnx_filename, output_filename, style)
```

Convert an ONNX model file to the JSON format used by the encrypted inference engine.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `onnx_filename` | `str` | required | Path to the input `.onnx` model |
| `output_filename` | `str` | required | Path to the output `.json` file |
| `style` | `str` | required | Packing style: `'ordinary'` or `'multiplexed'` |

---

### 1.5 Graph Data Model

#### FeatureNode

```python
FeatureNode(key, dim, channel, scale, skip, ckks_parameter_id, shape)
```

Data node in the computation graph. Represents an intermediate feature tensor.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `key` | `str` | Node identifier |
| `dim` | `int` | Feature dimensionality (0, 1, or 2) |
| `channel` | `int` | Number of channels |
| `scale` | `float` | Scaling factor |
| `skip` | `list` | Skip values in ciphertext packing |
| `ckks_parameter_id` | `str` | CKKS parameter set identifier |
| `shape` | `list` | Spatial dimensions |

**Methods:**

- `to_json() -> dict` — Serialize to JSON-compatible dict.

---

#### ComputeNode

```python
ComputeNode(layer_id, layer_type, feature_input, feature_output)
```

Base class for operation nodes in the computation graph.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `layer_id` | `str` | Layer identifier |
| `layer_type` | `str` | Operation type string |
| `feature_input` | `list[FeatureNode]` | Input feature nodes |
| `feature_output` | `list[FeatureNode]` | Output feature nodes |

**Methods:**

- `to_json() -> dict` — Serialize to JSON-compatible dict.
- `from_onnx_node(node, features, ...) -> ComputeNode` — Class method on subclasses to construct from ONNX node.

---

### 1.6 Supported Operations

| Class | Source File | ONNX Op(s) |
|-------|-----------|-------------|
| `ConvComputeNode` | `operations/Conv.py` | Conv |
| `ConvTransposeComputeNode` | `operations/ConvTranspose.py` | ConvTranspose |
| `DenseComputeNode` | `operations/Dense.py` | Gemm |
| `BatchNormComputeNode` | `operations/BatchNorm.py` | BatchNormalization |
| `ReluComputeNode` | `operations/Relu.py` | Relu |
| `Simple_PolyreluComputeNode` | `operations/Simple_Polyrelu.py` | RangeNormPoly2d |
| `PolyReluComputeNode` | `operations/PolyRelu.py` | PolyReluListIndependent |
| `AveragePoolComputeNode` | `operations/AveragePool.py` | AveragePool / GlobalAveragePool |
| `MaxPoolComputeNode` | `operations/MaxPool.py` | MaxPool |
| `ReshapeComputeNode` | `operations/Reshape.py` | Reshape / Flatten |
| `DropoutComputeNode` | `operations/Dropout.py` | Dropout |
| `ConstMulComputeNode` | `operations/ConstMul.py` | Mul |
| `SigmoidComputeNode` | `operations/Sigmoid.py` | Sigmoid |
| `AddComputeNode` | `operations/Add.py` | Add |
| `AddMorphComputeNode` | `operations/AddMorph.py` | AddMorph |
| `ConcatComputeNode` | `operations/Concat.py` | Concat |
| `SplitComputeNode` | `operations/Split.py` | Split |
| `ResizeComputeNode` | `operations/Resize.py` | Resize |
| `MatMulComputeNode` | `operations/MatMul.py` | MatMul |
| `IdentityComputeNode` | `operations/Identity.py` | Identity |
| `RangeNorm2dComputeNode` | `operations/RangeNorm2d.py` | RangeNorm2d |
| `RangeNormComputeNode` | `operations/RangeNorm.py` | RangeNorm |
| `SquareComputeNode` | `operations/Square.py` | Pow |
| `SoftmaxComputeNode` | `operations/Softmax.py` | Softmax |
| `ArgmaxComputeNode` | `operations/Argmax.py` | Argmax |
| `RNNComputeNode` | `operations/RNN.py` | MyRNN |
| `Relu6ComputeNode` | `operations/Relu6.py` | Clip |

All operation nodes inherit from `ComputeNode` and implement `from_onnx_node()` for construction from ONNX graph nodes.

---

## 2. model_compiler — Encrypted Graph Compiler

Source: `training/run_compile.py`, `training/model_compiler/`

### 2.1 CLI Interface (run_compile.py)

```bash
python run_compile.py -i <input> [-o <output>] [options]
```

Compile a model using the graph splitter tool. Supports both ONNX model files (`.onnx`) and pre-converted JSON files (`.json`).

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-i/--input` | `str` | required | Input `.onnx` or `.json` file |
| `-o/--output` | `str` | input dir | Output directory |
| `--poly_n` | `int` | config | Polynomial modulus degree: `8192`, `16384`, or `65536` |
| `--style` | `str` | config | Computation style: `ordinary` or `multiplexed` |
| `--graph_type` | `str` | config | Graph type: `btp` |
| `--num_experiments` | `int` | `128` | Number of parallel compilation experiments |
| `--num_workers` | `int` | `16` | Number of worker processes |
| `--temperature` | `float` | `1.0` | Randomization temperature |

**Examples:**

```bash
# Compile from JSON
python run_compile.py -i pt.json -o ./output --poly_n 65536 --style ordinary

# Compile from ONNX (auto-converts to JSON first)
python run_compile.py -i model.onnx -o ./output --style multiplexed

# Compile with custom parallelism
python run_compile.py -i pt.json --num_experiments 256 --num_workers 32
```

---

### 2.2 Programmatic API

#### init_config_with_args()

```python
init_config_with_args(poly_n=None, style=None, graph_type=None)
```

Initialize compiler configuration. Overrides values from `config.json` with provided arguments.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `poly_n` | `int` | `None` | Polynomial modulus degree (`POLY_N`) |
| `style` | `str` | `None` | Computation style (`STYLE`) |
| `graph_type` | `str` | `None` | Graph type (`GRAPH_TYPE`) |

---

#### run_parallel()

```python
run_parallel(num_experiments, input_file_path,
             output_dir, temperature, num_workers=16)
```

Run multiple compilations in parallel and select the best result.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `num_experiments` | `int` | required | Number of parallel compilation runs |
| `input_file_path` | `Path` | required | Input `pt.json` file path |
| `output_dir` | `Path` | required | Output directory |
| `temperature` | `float` | required | Randomization temperature |
| `num_workers` | `int` | `16` | Number of worker processes |

---

### 2.3 Compiler Data Model

Source: `training/model_compiler/components.py`

#### FeatureNode

```python
FeatureNode(key, dim, channel, scale=1.0, ckks_parameter_id='param0',
            ckks_scale=1, shape=[1, 1])
```

Data node in the compiler graph. Represents an encrypted feature tensor with CKKS parameters.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `key` | `str` | required | Node identifier |
| `dim` | `int` | required | Feature dimensionality |
| `channel` | `int` | required | Number of channels |
| `scale` | `float` | `1.0` | Scaling factor |
| `ckks_parameter_id` | `str` | `'param0'` | CKKS parameter set ID |
| `ckks_scale` | `float` | `1` | CKKS encoding scale |
| `shape` | `list` | `[1, 1]` | Spatial dimensions |

---

#### ComputeNode

```python
ComputeNode(layer_id, layer_type, channel_input, channel_output,
            ckks_parameter_id_input='param0', ckks_parameter_id_output='param0')
```

Base computation node in the compiler graph.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `layer_id` | `str` | required | Layer identifier |
| `layer_type` | `str` | required | Operation type |
| `channel_input` | `int` | required | Input channel count |
| `channel_output` | `int` | required | Output channel count |
| `ckks_parameter_id_input` | `str` | `'param0'` | Input CKKS parameter set |
| `ckks_parameter_id_output` | `str` | `'param0'` | Output CKKS parameter set |

---

#### EncryptParameterNode

```python
EncryptParameterNode(poly_modulus_degree, coeff_modulus_bit_length,
                     special_prime_bit_length)
```

CKKS encryption parameter specification.

| Name | Type | Description |
|------|------|-------------|
| `poly_modulus_degree` | `int` | Polynomial modulus degree (N) |
| `coeff_modulus_bit_length` | `int` | Coefficient modulus bit length |
| `special_prime_bit_length` | `int` | Special prime bit length |

---

#### LayerAbstractGraph

```python
LayerAbstractGraph(parent_graph=None)
```

DAG-based computation graph. Uses `networkx.DiGraph` internally with `FeatureNode` and `ComputeNode` as nodes.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `parent_graph` | `LayerAbstractGraph` | `None` | Parent graph (for hierarchical decomposition) |

**Key attributes:**

- `dag` — `networkx.DiGraph` containing `FeatureNode` and `ComputeNode` instances
- `graph_id` — Unique graph identifier
- `is_mpc` — Whether this subgraph contains MPC operations

**Key methods:**

- `get_leading_feature_nodes() -> list[FeatureNode]` — Get input feature nodes (in-degree = 0)

---


#### task_config.json

Server/client inference task configuration including:

- `task_type` — Task type identifier
- `pack_style` — Packing style (`ordinary` or `multiplexed`)
- `server_task` — Server task execution parameters
- `block_shape` — Block dimensions for ciphertext packing
- `is_absorb_polyrelu` — Whether polynomial activation coefficients are absorbed

#### ckks_parameter.json

CKKS encryption parameters including:

- `poly_modulus_degree` — Polynomial modulus degree (N)
- `coeff_modulus_bit_length` — Coefficient modulus chain
- `special_prime_bit_length` — Special prime for key-switching

#### ergs/erg0.json

Compiled encrypted computation graph containing:

- `input_feature` — List of graph input feature IDs
- `output_feature` — List of graph output feature IDs
- `feature` — Dictionary of `FeatureNode` specifications (dimensions, channels, packing parameters, CKKS levels)
- `layer` — Dictionary of `ComputeNode` specifications (operation types, connections, kernel parameters)

---

## 3. Inference Runtime (C++)

Source: `inference/` (excluding `lattisense/`)

### 3.1 Core Classes

#### InitInferenceProcess

```cpp
class InitInferenceProcess {
    InitInferenceProcess(const string& project_path_in, bool is_fpga = true);
    virtual void init_parameters(bool is_bootstrapping = false);
    virtual void load_model_prepare();
};
```

Loads the compiled graph and model parameters from disk. Initializes all FHE layer objects with their weights.

| Member | Type | Description |
|--------|------|-------------|
| `project_path` | `filesystem::path` | Root path of the compiled task |
| `task_type` | `string` | Task type identifier |
| `pack_style` | `string` | Packing style |
| `ckks_parameters` | `map<string, unique_ptr<CkksParameter>>` | CKKS parameter sets |
| `is_lazy` | `bool` | Enable lazy weight preparation |

The class holds maps of all initialized FHE layer instances:

- `ckks_conv2ds` — `Conv2DPackedLayer` instances
- `ckks_dw_conv2ds` — `Conv2DPackedDepthwiseLayer` instances
- `ckks_denses` — `DensePackedLayer` instances
- `ckks_poly_relu` — `PolyRelu` instances
- `ckks_adds` — `AddLayer` instances
- `ckks_avgpool` — `Avgpool2DLayer` instances
- `ckks_concat` — `ConcatLayer` instances
- `ckks_upsample` — `UpsampleLayer` instances
- `ckks_multiplexed_conv2ds` — `ParMultiplexedConv2DPackedLayer` instances

---

#### InferenceProcess

```cpp
class InferenceProcess {
    InferenceProcess(InitInferenceProcess* fp_in, bool is_fpga_in);
    void run_task(bool is_mpc = false);
    void run_task_sdk(bool is_mpc = false);
    void run_task_plaintext(bool is_mpc = false);
    void set_feature(const string& feature_id, unique_ptr<FeatureEncrypted> feature);
    const FeatureEncrypted& get_feature(const string& feature_id);
};
```

Executes encrypted inference using layers initialized by `InitInferenceProcess`.

| Member | Type | Description |
|--------|------|-------------|
| `fp` | `InitInferenceProcess*` | Pointer to initialization process |
| `compute_device` | `ComputeDevice` | Execution device (default: `CPU`) |
| `intermediate_result` | `map<string, unique_ptr<FeatureEncrypted>>` | Feature storage during inference |
| `ckks_contexts` | `map<string, unique_ptr<CkksContext>>` | CKKS contexts |

---

#### Feature2DEncrypted

```cpp
class Feature2DEncrypted : public FeatureEncrypted {
    Feature2DEncrypted(CkksContext* context_in, int ct_level, Duo skip_in = {1, 1});

    virtual void pack(const Array<double, 3>& feature_mg,
                      bool is_symmetric = false, double scale_in = DEFAULT_SCALE);
    virtual Array<double, 3> unpack() const;

    Bytes serialize() const;
    void deserialize(const Bytes& bytes) override;
};
```

Encrypted 2D feature map. Stores CKKS ciphertexts for a multi-channel 2D spatial feature.

| Member | Type | Description |
|--------|------|-------------|
| `shape` | `Duo` | Spatial dimensions `[H, W]` |
| `skip` | `Duo` | Skip values for packing |
| `data` | `vector<CkksCiphertext>` | Encrypted data (one ciphertext per packed channel group) |
| `n_channel` | `uint32_t` | Number of channels |
| `n_channel_per_ct` | `uint32_t` | Channels packed per ciphertext |
| `level` | `uint32_t` | Current CKKS level |

**Key methods:**

- `pack()` / `unpack()` — Encrypt/decrypt feature data
- `single_pack()` / `mult_pack()` — Alternative packing strategies
- `serialize()` / `deserialize()` — Binary serialization
- `split_to_shares()` / `combine_with_share()` — Secret sharing for MPC

---

#### ComputeDevice

```cpp
enum class ComputeDevice { CPU, GPU, FPGA };
```

Execution device for inference.

---

### 3.2 FHE Layers

| Layer Class | Header | Description |
|-------------|--------|-------------|
| `Conv2DPackedLayer` | `fhe_layers/conv2d_packed_layer.h` | Encrypted Conv2D with channel packing |
| `Conv2DPackedDepthwiseLayer` | `fhe_layers/conv2d_depthwise.h` | Encrypted depthwise Conv2D |
| `ParMultiplexedConv2DPackedLayer` | `fhe_layers/multiplexed_conv2d_pack_layer.h` | Multiplexed Conv2D with parallel packing |
| `Avgpool2DLayer` | `fhe_layers/avgpool2d_layer.h` | Encrypted average pooling |
| `PolyRelu` | `fhe_layers/poly_relu2d.h` | Polynomial activation (Hermite expansion) |
| `DensePackedLayer` | `fhe_layers/dense_packed_layer.h` | Encrypted fully-connected layer |
| `AddLayer` | `fhe_layers/add_layer.h` | Ciphertext element-wise addition |
| `ConcatLayer` | `fhe_layers/concat_layer.h` | Ciphertext channel concatenation |
| `UpsampleLayer` | `fhe_layers/upsample_layer.h` | Ciphertext upsampling (zero-insertion) |
| `MultScalarLayer` | `fhe_layers/mult_scaler.h` | Scalar multiplication on ciphertexts |
| `ReshapeLayer` | `fhe_layers/reshape_layer.h` | Reshape 2D features to 0D |
| `SquareLayer` | `fhe_layers/activation_layer.h` | Element-wise squaring |

---

#### Conv2DPackedLayer

```cpp
Conv2DPackedLayer(const CkksParameter& param,
                  const Duo& input_shape,
                  const Array<double, 4>& weight,
                  const Array<double, 1>& bias,
                  const Duo& stride,
                  const Duo& skip,
                  uint32_t n_channel_per_ct,
                  uint32_t level,
                  double residual_scale = 1.0);
```

Encrypted 2D convolution with channel packing. Packs multiple channels into a single ciphertext to reduce ciphertext count.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `param` | `const CkksParameter&` | — | CKKS encryption parameters |
| `input_shape` | `const Duo&` | — | Spatial dimensions (H, W) of the input feature |
| `weight` | `const Array<double, 4>&` | — | Convolution kernel weights (out_ch, in_ch, kH, kW) |
| `bias` | `const Array<double, 1>&` | — | Bias vector (out_ch) |
| `stride` | `const Duo&` | — | Convolution stride (sH, sW) |
| `skip` | `const Duo&` | — | Skip factor for packed ciphertext layout |
| `n_channel_per_ct` | `uint32_t` | — | Number of channels packed per ciphertext |
| `level` | `uint32_t` | — | CKKS multiplicative level |
| `residual_scale` | `double` | `1.0` | Scaling factor for residual connections |

**Key methods:**

- `void prepare_weight()` — Pre-encode weights as plaintexts
- `void prepare_weight_lazy()` — Defer weight encoding until first use
- `Feature2DEncrypted run(CkksContext& ctx, const Feature2DEncrypted& x)` — Execute convolution on encrypted input

---

#### Conv2DPackedDepthwiseLayer

```cpp
Conv2DPackedDepthwiseLayer(const CkksParameter& param,
                           const Duo& input_shape,
                           const Array<double, 4>& weight,
                           const Array<double, 1>& bias,
                           const Duo& stride,
                           const Duo& skip,
                           uint32_t n_channel_per_ct,
                           uint32_t level,
                           double residual_scale = 1.0);
```

Encrypted depthwise 2D convolution. Each input channel is convolved with its own filter independently, reducing computational cost compared to standard convolution.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `param` | `const CkksParameter&` | — | CKKS encryption parameters |
| `input_shape` | `const Duo&` | — | Spatial dimensions (H, W) of the input feature |
| `weight` | `const Array<double, 4>&` | — | Depthwise kernel weights (ch, 1, kH, kW) |
| `bias` | `const Array<double, 1>&` | — | Bias vector (ch) |
| `stride` | `const Duo&` | — | Convolution stride (sH, sW) |
| `skip` | `const Duo&` | — | Skip factor for packed ciphertext layout |
| `n_channel_per_ct` | `uint32_t` | — | Number of channels packed per ciphertext |
| `level` | `uint32_t` | — | CKKS multiplicative level |
| `residual_scale` | `double` | `1.0` | Scaling factor for residual connections |

**Key methods:**

- `void prepare_weight()` — Pre-encode weights as plaintexts
- `void prepare_weight_lazy()` — Defer weight encoding until first use
- `Feature2DEncrypted run(CkksContext& ctx, const Feature2DEncrypted& x)` — Execute depthwise convolution on encrypted input
- `Array<double, 3> run_plaintext(const Array<double, 3>& x, double multiplier = 1.0)` — Execute on plaintext input (for debugging)

---

#### ParMultiplexedConv2DPackedLayer

```cpp
ParMultiplexedConv2DPackedLayer(const CkksParameter& param,
                                const Duo& input_shape,
                                const Array<double, 4>& weight,
                                const Array<double, 1>& bias,
                                const Duo& stride,
                                const Duo& skip,
                                uint32_t n_channel_per_ct,
                                uint32_t level,
                                double residual_scale = 1.0,
                                const Duo& upsample_factor = {1, 1});
```

Multiplexed 2D convolution with parallel channel packing. Supports upsampling by interleaving zero-valued slots, enabling efficient resolution changes within the convolution.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `param` | `const CkksParameter&` | — | CKKS encryption parameters |
| `input_shape` | `const Duo&` | — | Spatial dimensions (H, W) of the input feature |
| `weight` | `const Array<double, 4>&` | — | Convolution kernel weights (out_ch, in_ch, kH, kW) |
| `bias` | `const Array<double, 1>&` | — | Bias vector (out_ch) |
| `stride` | `const Duo&` | — | Convolution stride (sH, sW) |
| `skip` | `const Duo&` | — | Skip factor for packed ciphertext layout |
| `n_channel_per_ct` | `uint32_t` | — | Number of channels packed per ciphertext |
| `level` | `uint32_t` | — | CKKS multiplicative level |
| `residual_scale` | `double` | `1.0` | Scaling factor for residual connections |
| `upsample_factor` | `const Duo&` | `{1, 1}` | Upsampling factor (uH, uW) for resolution changes |

**Key methods:**

- `void prepare_weight_for_post_skip_rotation()` — Pre-encode weights for post-skip rotation mode
- `void prepare_weight_for_post_skip_rotation_lazy()` — Defer encoding until first use
- `Feature2DEncrypted run(CkksContext& ctx, const Feature2DEncrypted& x)` — Execute multiplexed convolution on encrypted input
- `Feature2DEncrypted run_for_post_skip_rotation(CkksContext& ctx, const Feature2DEncrypted& x)` — Execute with post-skip rotation optimization

---

#### Avgpool2DLayer

```cpp
Avgpool2DLayer(const Duo& shape, const Duo& stride);
```

Encrypted average pooling layer. Computes the average of elements within a sliding window over the spatial dimensions.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `shape` | `const Duo&` | — | Pooling window size (pH, pW) |
| `stride` | `const Duo&` | — | Pooling stride (sH, sW) |

**Key methods:**

- `void prepare_weight(const CkksParameter& param, int n_channel_per_ct, int level, const Duo& skip, const Duo& shape)` — Pre-encode pooling parameters for multiplexed mode
- `Feature2DEncrypted run(CkksContext& ctx, const Feature2DEncrypted& x)` — Execute average pooling on encrypted input
- `Feature2DEncrypted run_adaptive_avgpool(CkksContext& ctx, const Feature2DEncrypted& x)` — Execute adaptive average pooling (global)
- `Feature2DEncrypted run_multiplexed_avgpool(CkksContext& ctx, const Feature2DEncrypted& x)` — Execute on multiplexed-packed input
- `Feature2DEncrypted run_split_avgpool(CkksContext& ctx, const Feature2DEncrypted& x, const Duo block_expansion)` — Execute split average pooling with block expansion

---

#### PolyRelu

```cpp
PolyRelu(const CkksParameter& param,
         const Duo& input_shape,
         const int order,
         const Array<double, 2>& weight,
         const Duo& skip,
         uint32_t n_channel_per_ct,
         uint32_t level,
         const Duo& upsample_factor = {1, 1},
         const Duo& block_expansion = {1, 1});
```

Polynomial activation function using Hermite expansion. Approximates ReLU (or other activations) with a polynomial of configurable degree, which can be evaluated directly on ciphertexts.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `param` | `const CkksParameter&` | — | CKKS encryption parameters |
| `input_shape` | `const Duo&` | — | Spatial dimensions (H, W) of the input feature |
| `order` | `int` | — | Polynomial degree for the activation approximation |
| `weight` | `const Array<double, 2>&` | — | Polynomial coefficients (order+1, channels) |
| `skip` | `const Duo&` | — | Skip factor for packed ciphertext layout |
| `n_channel_per_ct` | `uint32_t` | — | Number of channels packed per ciphertext |
| `level` | `uint32_t` | — | CKKS multiplicative level |
| `upsample_factor` | `const Duo&` | `{1, 1}` | Upsampling factor for multiplexed layout |
| `block_expansion` | `const Duo&` | `{1, 1}` | Block expansion factor |

**Key methods:**

- `void prepare_weight()` — Pre-encode polynomial coefficients as plaintexts
- `void prepare_weight_lazy()` — Defer encoding until first use
- `Feature2DEncrypted run(CkksContext& ctx, const Feature2DEncrypted& x)` — Evaluate polynomial activation on encrypted input
- `Feature2DEncrypted run_for_non_absorb_case(CkksContext& ctx, const Feature2DEncrypted& x)` — Evaluate without absorbing coefficients into adjacent layers
- `Array<double, 3> run_plaintext(const Array<double, 3>& x)` — Execute on plaintext input (for debugging)

---

#### DensePackedLayer

```cpp
DensePackedLayer(const CkksParameter& param,
                 const Duo& input_shape,
                 const Duo& skip,
                 const Array<double, 2>& weight,
                 const Array<double, 1>& bias,
                 uint32_t pack,
                 uint32_t level,
                 int mark,
                 double residual_scale = 1.0);
```

Encrypted fully-connected (dense) layer with packed ciphertext layout. Supports both single-pack and mult-pack strategies for matrix-vector multiplication on encrypted data.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `param` | `const CkksParameter&` | — | CKKS encryption parameters |
| `input_shape` | `const Duo&` | — | Spatial dimensions (H, W) of the input feature |
| `skip` | `const Duo&` | — | Skip factor for packed ciphertext layout |
| `weight` | `const Array<double, 2>&` | — | Weight matrix (out_features, in_features) |
| `bias` | `const Array<double, 1>&` | — | Bias vector (out_features) |
| `pack` | `uint32_t` | — | Packing factor for output features |
| `level` | `uint32_t` | — | CKKS multiplicative level |
| `mark` | `int` | — | Layer marker for scheduling |
| `residual_scale` | `double` | `1.0` | Scaling factor for residual connections |

**Key methods:**

- `void prepare_weight1()` — Pre-encode weights for single-pack mode
- `void prepare_weight1_lazy()` — Defer single-pack weight encoding until first use
- `void prepare_weight_for_mult_pack()` — Pre-encode weights for mult-pack mode
- `void prepare_weight_for_mult_pack_lazy()` — Defer mult-pack weight encoding until first use
- `Feature0DEncrypted call(CkksContext& ctx, const Feature2DEncrypted& x)` — Execute on 2D encrypted input (single-pack)
- `Feature0DEncrypted call(CkksContext& ctx, const Feature0DEncrypted& x)` — Execute on 0D encrypted input (single-pack)
- `Feature0DEncrypted run_mult_park(CkksContext& ctx, const Feature2DEncrypted& x)` — Execute on 2D encrypted input (mult-pack)
- `Feature0DEncrypted run_mult_park(CkksContext& ctx, const Feature0DEncrypted& x)` — Execute on 0D encrypted input (mult-pack)
- `Array<double, 1> plaintext_call(const Array<double, 1>& x, double multiplier = 1.0)` — Execute on plaintext input (for debugging)

---

#### AddLayer

```cpp
AddLayer(const CkksParameter& param);
```

Element-wise addition of two encrypted feature maps. Used for residual/skip connections in neural networks.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `param` | `const CkksParameter&` | — | CKKS encryption parameters |

**Key methods:**

- `Feature2DEncrypted run(CkksContext& ctx, const Feature2DEncrypted& x0, const Feature2DEncrypted& x1)` — Add two encrypted feature maps element-wise
- `Array<double, 3> run_plaintext(const Array<double, 3>& x0, const Array<double, 3>& x1)` — Add two plaintext feature maps (for debugging)

---

#### ConcatLayer

```cpp
ConcatLayer();
```

Channel concatenation of encrypted feature maps. Appends the ciphertext vectors of multiple inputs along the channel dimension.

**Key methods:**

- `Feature2DEncrypted run(CkksContext& ctx, const Feature2DEncrypted& x1, const Feature2DEncrypted& x2)` — Concatenate two encrypted feature maps along channels
- `Feature2DEncrypted run_multiple_inputs(CkksContext& ctx, const std::vector<Feature2DEncrypted>& inputs)` — Concatenate multiple encrypted feature maps

---

#### UpsampleLayer

```cpp
UpsampleLayer(const CkksParameter& param,
              const Duo& stride,
              const Duo& upsample_factor,
              const int& level,
              const int& n_channel,
              const int& n_channel_per_ct);
```

Encrypted upsampling via zero-insertion. Increases spatial resolution by inserting zeros between existing elements in the ciphertext packing layout.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `param` | `const CkksParameter&` | — | CKKS encryption parameters |
| `stride` | `const Duo&` | — | Current stride of the packed layout |
| `upsample_factor` | `const Duo&` | — | Upsampling factor (uH, uW) |
| `level` | `const int&` | — | CKKS multiplicative level |
| `n_channel` | `const int&` | — | Total number of channels |
| `n_channel_per_ct` | `const int&` | — | Number of channels per ciphertext |

**Key methods:**

- `void prepare_data()` — Pre-compute rotation keys and masks for upsampling
- `Feature2DEncrypted run(CkksContext& ctx, const Feature2DEncrypted& x)` — Execute upsampling on encrypted input

---

#### MultScalarLayer

```cpp
MultScalarLayer(const CkksParameter& param,
                const Duo& input_shape,
                const Array<double, 1>& weight,
                const Duo& skip,
                uint32_t n_channel_per_ct,
                uint32_t level,
                const Duo& upsample_factor = {1, 1},
                const Duo& block_expansion = {1, 1});
```

Scalar multiplication on encrypted feature maps. Multiplies each channel by a per-channel scalar weight encoded as a plaintext.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `param` | `const CkksParameter&` | — | CKKS encryption parameters |
| `input_shape` | `const Duo&` | — | Spatial dimensions (H, W) of the input feature |
| `weight` | `const Array<double, 1>&` | — | Per-channel scalar weights |
| `skip` | `const Duo&` | — | Skip factor for packed ciphertext layout |
| `n_channel_per_ct` | `uint32_t` | — | Number of channels packed per ciphertext |
| `level` | `uint32_t` | — | CKKS multiplicative level |
| `upsample_factor` | `const Duo&` | `{1, 1}` | Upsampling factor for multiplexed layout |
| `block_expansion` | `const Duo&` | `{1, 1}` | Block expansion factor |

**Key methods:**

- `void prepare_weight()` — Pre-encode scalar weights as plaintexts
- `Feature2DEncrypted run(CkksContext& ctx, const Feature2DEncrypted& x)` — Execute scalar multiplication on encrypted input
- `Array<double, 3> run_plaintext(const Array<double, 3>& x)` — Execute on plaintext input (for debugging)

---

#### ReshapeLayer

```cpp
ReshapeLayer(const CkksParameter& param);
```

Reshape encrypted 2D features to 0D (flattened) representation. Converts a `Feature2DEncrypted` into a `Feature0DEncrypted` for use with fully-connected layers.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `param` | `const CkksParameter&` | — | CKKS encryption parameters |

**Key methods:**

- `Feature0DEncrypted call(CkksContext& ctx, const Feature2DEncrypted& x)` — Reshape 2D encrypted feature to 0D

---

#### SquareLayer

```cpp
SquareLayer(const CkksParameter& param);
```

Element-wise squaring activation on encrypted data. Computes `x^2` for each element, consuming one multiplicative level.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `param` | `const CkksParameter&` | — | CKKS encryption parameters |

**Key methods:**

- `Feature2DEncrypted call(CkksContext& ctx, const Feature2DEncrypted& x)` — Square each element of a 2D encrypted feature
- `Feature0DEncrypted call(CkksContext& ctx, const Feature0DEncrypted& x)` — Square each element of a 0D encrypted feature

---

### 3.3 Code Generation (Python)

#### gen_custom_task()

```python
gen_custom_task(task_path, n=16384, use_gpu=True, style='ordinary')
```

Generate GPU-accelerated inference instructions (mega_ag format) from a compiled task.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `task_path` | `str` | required | Path to the `ergs/` directory containing `erg0.json` |
| `n` | `int` | `16384` | Polynomial modulus degree |
| `use_gpu` | `bool` | `True` | Generate GPU-optimized instructions |
| `style` | `str` | `'ordinary'` | Packing style: `'ordinary'` or `'multiplexed'` |

---

### 3.4 Instruction Generation Layers (Python)

Source: `inference/model_generator/layers/`

These Python layer classes generate FHE computation instructions (graph nodes) for the inference runtime. Each class mirrors a corresponding C++ FHE layer (Section 3.2) and produces the instruction DAG that the runtime executes.

| Layer Class | Source | Description |
|---|---|---|
| `Conv2DPackedLayer` | `layers/conv_pack.py` | Packed Conv2D instruction generation |
| `Conv2DepthwiseLayer` | `layers/conv_dw.py` | Depthwise Conv2D instruction generation |
| `MultConv2DPackedLayer` | `layers/mult_conv.py` | Multiplexed Conv2D instruction generation |
| `MultConv2DPackedDepthwiseLayer` | `layers/mult_conv_dw.py` | Multiplexed depthwise Conv2D instruction generation |
| `DensePackedLayer` | `layers/dense_pack.py` | Packed fully-connected instruction generation |
| `Avgpool_layer` | `layers/avgpool.py` | Average pooling instruction generation |
| `PolyReluLayer` | `layers/poly_relu.py` | Polynomial activation instruction generation |
| `MultScalarLayer` | `layers/mult_scalar.py` | Scalar multiplication instruction generation |
| `Square_layer` | `layers/square_pack.py` | Element-wise squaring instruction generation |
| `UpsampleNearestLayer` | `layers/upsample_layer.py` | Nearest-neighbor upsampling instruction generation |
| `ConcatLayer` | `layers/concat_layer.py` | Channel concatenation instruction generation |
| `AddLayer` | `layers/add_pack.py` | Element-wise addition instruction generation |
| `InverseMultiplexedConv2d` | `layers/inverse_multiplexed_conv2d_layer.py` | Inverse multiplexed Conv2D instruction generation |

---

#### Conv2DPackedLayer

```python
Conv2DPackedLayer(n_out_channel, n_in_channel, input_shape, kernel_shape,
                  stride, skip, pack, n_packed_in_channel, n_packed_out_channel)
```

Source: `layers/conv_pack.py`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `n_out_channel` | `int` | Number of output channels |
| `n_in_channel` | `int` | Number of input channels |
| `input_shape` | `list[int]` | Spatial dimensions (must be powers of 2) |
| `kernel_shape` | `list[int]` | Kernel dimensions |
| `stride` | `list[int]` | Convolution stride (must be powers of 2) |
| `skip` | `list[int]` | Skip values (must be powers of 2) |
| `pack` | `int` | Packing factor |
| `n_packed_in_channel` | `int` | Number of packed input channels |
| `n_packed_out_channel` | `int` | Number of packed output channels |

**Raises:** `ValueError` if `input_shape`, `stride`, or `skip` are not powers of 2.

**Key methods:**

- `call(x, weight_pt, bias_pt) -> list[CkksCiphertextNode]` — Generate convolution instructions
- `call_custom_compute(x, conv_data_source) -> list[CkksCiphertextNode]` — Generate with on-demand weight encoding

---

#### Conv2DepthwiseLayer

```python
Conv2DepthwiseLayer(n_out_channel, n_in_channel, input_shape, kernel_shape,
                    stride, skip, pack, n_packed_in_channel, n_packed_out_channel)
```

Source: `layers/conv_dw.py`

**Parameters:** Same as `Conv2DPackedLayer`.

**Raises:** `ValueError` if `input_shape`, `stride`, or `skip` are not powers of 2.

**Key methods:**

- `call(x, weight_pt, bias_pt) -> list[DataNode]`
- `call_custom_compute(x, conv_data_source) -> list[DataNode]`

---

#### MultConv2DPackedLayer

```python
MultConv2DPackedLayer(n_out_channel, n_in_channel, input_shape, kernel_shape,
                      stride, skip, n_channel_per_ct, n_packed_in_channel,
                      n_packed_out_channel, upsample_factor=[1, 1])
```

Source: `layers/mult_conv.py`

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `n_out_channel` | `int` | required | Number of output channels |
| `n_in_channel` | `int` | required | Number of input channels |
| `input_shape` | `list[int]` | required | Spatial dimensions (must be powers of 2) |
| `kernel_shape` | `list[int]` | required | Kernel dimensions |
| `stride` | `list[int]` | required | Convolution stride (must be powers of 2) |
| `skip` | `list[int]` | required | Skip values (must be powers of 2) |
| `n_channel_per_ct` | `int` | required | Channels packed per ciphertext |
| `n_packed_in_channel` | `int` | required | Number of packed input channels |
| `n_packed_out_channel` | `int` | required | Number of packed output channels |
| `upsample_factor` | `list[int]` | `[1, 1]` | Upsampling factor |

**Raises:** `ValueError` if `input_shape`, `stride`, or `skip` are not powers of 2.

**Key methods:**

- `call(x, weight_pt, bias_pt, mast_pt) -> list[CkksCiphertextNode]`
- `call_custom_compute(x, conv_data_source) -> list[CkksCiphertextNode]`

---

#### MultConv2DPackedDepthwiseLayer

```python
MultConv2DPackedDepthwiseLayer(n_out_channel, n_in_channel, input_shape, kernel_shape,
                               stride, skip, n_channel_per_ct, n_packed_in_channel,
                               n_packed_out_channel, upsample_factor=[1, 1])
```

Source: `layers/mult_conv_dw.py`

**Parameters:** Same as `MultConv2DPackedLayer`.

**Raises:** `ValueError` if `input_shape`, `stride`, or `skip` are not powers of 2.

**Key methods:**

- `call(x, weight_pt, bias_pt, mast_pt) -> list[CkksCiphertextNode]`
- `call_custom_compute(x, conv_data_source) -> list[CkksCiphertextNode]`

---

#### DensePackedLayer

```python
DensePackedLayer(n_out_channel, n_in_channel, input_shape, skip, pack,
                 n_packed_in_feature, n_packed_out_feature)
```

Source: `layers/dense_pack.py`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `n_out_channel` | `int` | Number of output features |
| `n_in_channel` | `int` | Number of input features |
| `input_shape` | `list[int]` | Spatial dimensions (must be powers of 2) |
| `skip` | `list[int]` | Skip values (must be powers of 2) |
| `pack` | `int` | Packing factor |
| `n_packed_in_feature` | `int` | Number of packed input features |
| `n_packed_out_feature` | `int` | Number of packed output features |

**Raises:** `ValueError` if `input_shape` or `skip` are not powers of 2.

**Key methods:**

- `call(x, weight_pt, bias_pt) -> list[CkksCiphertextNode]` — Single-pack mode
- `call_mult_pack(x, weight_pt, bias_pt, n) -> list` — Mult-pack mode
- `call_custom_compute(x, dense_data_source) -> list[CkksCiphertextNode]`
- `call_mult_pack_custom_compute(x, dense_data_source, n) -> list`

---

#### Avgpool_layer

```python
Avgpool_layer(stride, shape, channel=1, skip=[1, 1])
```

Source: `layers/avgpool.py`

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `stride` | `list[int]` | required | Pooling stride (must be powers of 2) |
| `shape` | `list[int]` | required | Spatial dimensions (must be powers of 2) |
| `channel` | `int` | `1` | Number of channels |
| `skip` | `list[int]` | `[1, 1]` | Skip values (must be powers of 2) |

**Raises:** `ValueError` if `shape`, `stride`, or `skip` are not powers of 2.

**Key methods:**

- `call(x) -> list[DataNode]` — Standard average pooling
- `run_adaptive_avgpool(x, n) -> list[DataNode]` — Adaptive (global) average pooling

---

#### PolyReluLayer

```python
PolyReluLayer(input_shape, order, skip, n_channel_per_ct,
              upsample_factor=[1, 1], block_expansion=[1, 1])
```

Source: `layers/poly_relu.py`

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `input_shape` | `list[int]` | required | Spatial dimensions (must be powers of 2) |
| `order` | `int` | required | Polynomial degree |
| `skip` | `list[int]` | required | Skip values (must be powers of 2) |
| `n_channel_per_ct` | `int` | required | Channels packed per ciphertext |
| `upsample_factor` | `list[int]` | `[1, 1]` | Upsampling factor |
| `block_expansion` | `list[int]` | `[1, 1]` | Block expansion factor |

**Raises:** `ValueError` if `input_shape`, `skip`, or derived `block_shape` are not powers of 2.

**Key methods:**

- `call(x, weight_pt) -> list` — Generate polynomial activation instructions
- `call_custom_compute(x, poly_data_source, layer_id='') -> list` — With on-demand weight encoding

---

#### MultScalarLayer

```python
MultScalarLayer()
```

Source: `layers/mult_scalar.py`

No constructor parameters.

**Key methods:**

- `call(x1, pt_scale1) -> list[list[DataNode]]` — Scalar multiplication

---

#### Square_layer

```python
Square_layer(level)
```

Source: `layers/square_pack.py`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `level` | `int` | CKKS multiplicative level |

**Key methods:**

- `call(x) -> list[CkksCiphertextNode]` — Element-wise squaring

---

#### UpsampleNearestLayer

```python
UpsampleNearestLayer(shape, skip, upsample_factor, n_channel_per_ct, level)
```

Source: `layers/upsample_layer.py`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `shape` | `list[int]` | Spatial dimensions (must be powers of 2) |
| `skip` | `list[int]` | Skip values (must be powers of 2) |
| `upsample_factor` | `list[int]` | Upsampling factor |
| `n_channel_per_ct` | `int` | Channels packed per ciphertext |
| `level` | `int` | CKKS multiplicative level |

**Raises:** `ValueError` if `shape` or `skip` are not powers of 2.

**Key methods:**

- `call_custom_compute(x, data_source, n_channel) -> list[CkksCiphertextNode]` — Generate upsampling instructions

---

#### ConcatLayer

```python
ConcatLayer()
```

Source: `layers/concat_layer.py`

No constructor parameters.

**Key methods:**

- `call(x1, x2) -> list[CkksCiphertextNode]` — Concatenate two feature maps
- `call_multiple_inputs(inputs) -> list[CkksCiphertextNode]` — Concatenate multiple feature maps

---

#### AddLayer

```python
AddLayer()
```

Source: `layers/add_pack.py`

No constructor parameters.

**Key methods:**

- `call(x1, x2, scale1, scale2, pt_scale1=None, pt_scale2=None) -> list[DataNode]` — Element-wise addition with optional scaling
- `mult_and_add(x1, x2, pt_scale1, pt_scale2=None) -> DataNode` — Multiply-then-add helper

---

#### InverseMultiplexedConv2d

```python
InverseMultiplexedConv2d(n_out_channel, n_in_channel, input_shape, padding,
                         kernel_shape, stride, stride_next, skip, block_shape)
```

Source: `layers/inverse_multiplexed_conv2d_layer.py`

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `n_out_channel` | `int` | Number of output channels |
| `n_in_channel` | `int` | Number of input channels |
| `input_shape` | `list[int]` | Spatial dimensions (must be powers of 2) |
| `padding` | `list[int]` | Padding values |
| `kernel_shape` | `list[int]` | Kernel dimensions |
| `stride` | `list[int]` | Convolution stride (must be powers of 2) |
| `stride_next` | `list[int]` | Next layer stride (must be powers of 2) |
| `skip` | `list[int]` | Skip values (must be powers of 2) |
| `block_shape` | `list[int]` | Block dimensions (must be powers of 2) |

**Raises:** `ValueError` if `input_shape`, `stride`, `stride_next`, `skip`, or `block_shape` are not powers of 2.

**Key methods:**

- `call(x, weight_pt, bias_pt, N) -> list[CkksCiphertextNode]`
- `call_custom_compute(x, conv_data_source, N) -> list[CkksCiphertextNode]`
