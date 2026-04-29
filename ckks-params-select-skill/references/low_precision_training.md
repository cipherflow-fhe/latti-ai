# Low-Precision Training for CKKS Parameter Optimization

## Why Lower Precision?

CKKS scale (per-level fixed-point precision) must be large enough to faithfully represent the model's weights. Lower-precision weights need a smaller scale, which shrinks the ciphertext modulus Q and may enable a smaller ring dimension N — yielding significant inference speedup.

| Weight Precision | CKKS Scale (bits) | Potential N Reduction | Speedup |
|---|---|---|---|
| float32 | 30-34 | Baseline | 1x |
| FP8 (float8_e4m3fn) | 21-24 | 2x smaller N possible | ~3-4x |
| INT8 (8-bit integer) | 21-24 | 2x smaller N possible | ~3-4x |

## Accuracy Constraint

The lower-precision model must stay within an acceptable accuracy drop of the float32 baseline (typically <=0.5%). If it exceeds this threshold, fall back to float32 parameters.

## General Methodology

### 1. Quantization-Aware Training (QAT) — Recommended

Train with quantized weights from the start, simulating quantization during forward passes.

Key principles:
- **Per-channel scaling**: Scale each output channel independently to maximize dynamic range
- **In-place quantization**: After each optimizer step, quantize weights to the target format
- **Stochastic rounding**: Probabilistic rounding reduces systematic quantization bias
- **Exclude normalization parameters**: BN/LN params stay float32 (critical for training stability)
- **Gradient clipping**: Stabilize training under quantization noise
- **Warm-start**: Load float32 baseline checkpoint, then fine-tune with quantized weights

Example per-channel scaling for FP8:
```python
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0

def fp8_quantize_per_channel(w):
    dims = list(range(1, w.dim()))  # all dims except output channel
    amax = w.abs().amax(dim=dims, keepdim=True).clamp(min=1e-12)
    scale = FP8_MAX / amax
    w_fp8 = (w * scale).to(torch.float8_e4m3fn).float() / scale
    return w_fp8
```

### 2. Post-Training Quantization (PTQ) — Simpler but less accurate

Load a pre-trained float32 model and quantize weights in-place. No retraining needed, but accuracy drop may be larger.

### Decision Flow

```
1. Train float32 baseline model, measure accuracy
2. Train lower-precision model (QAT preferred, PTQ as fallback)
3. Measure accuracy drop
4. If drop <= threshold (e.g., 0.5%):
     -> Use lower-precision parameters (smaller N, faster inference)
   Else:
     -> Fall back to float32 parameters
```

## CKKS Scale Selection After Low-Precision Training

| Weight characteristics | Recommended log_default_scale |
|---|---|
| Small model, tight weight range | 21 bits |
| Medium model, wider weight range | 24 bits |
| Large model, diverse weight ranges | 24-26 bits |

Start with the minimum scale for your precision level. If accuracy benchmarks show high error, increase scale and re-evaluate.