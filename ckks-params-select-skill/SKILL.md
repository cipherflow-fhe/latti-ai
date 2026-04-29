---
name: ckks-param-selector-latti-ai
description: Select optimal CKKS parameters for FHE encrypted inference based on model characteristics and user priorities (security or efficiency). This skill should be used when the user needs to choose CKKS parameters (N, Q, P, scale, levels), estimate security levels, assess bootstrapping needs, or evaluate trade-offs between parameter choices. Triggers on mentions of "CKKS parameters", "FHE parameters", "polynomial degree", "security level", "N selection", "bootstrapping parameters", "multiplicative depth", or when planning encrypted inference deployment.
---

# CKKS Parameter Selector

## Overview

Guide users through selecting CKKS (Fully Homomorphic Encryption) parameters for encrypted neural network inference. The workflow balances security, performance, and model accuracy based on the user's priorities.

The core principle: **N (ring dimension) should be the smallest power of 2 that satisfies both the model's depth requirement and the user's security target.** Everything else follows from that.

## Workflow

```
1. Gather requirements    → What model? What priority? What security target?
2. Analyze model depth    → Extract multiplicative depth (levels consumed)
3. Assess weight precision → Can lower-precision training shrink parameters?
4. Recommend parameters   → Compute minimum N, select scale, build Q/P chains
5. Verify security        → Estimate concrete security bits
```

## User Modes

Adapt interaction based on expertise:

**Guided mode** (ML developer): Walk through each phase with explanations. Load `references/ckks_basics.md` for concept explanations as needed.

**Expert mode** (FHE expert): Skip directly to computation and verification. Focus on results, not concepts.

Detect mode from the user's first message. Explicit parameter mentions (N, Q, log_scale) indicate expert mode. Model names or "I want to deploy..." indicate guided mode.

## Phase 1: Gather Requirements

Ask the user for:

1. **Model**: ONNX file path, compiled DAG path, or model architecture description
2. **Priority**: Security (target bit level) or Efficiency (fastest inference)
3. **Security target**: If security priority, ask for target bits (128/192/256). Default: 128-bit.
4. **Weight precision**: float32 (standard) or willing to train a lower-precision version

## Phase 2: Analyze Model Depth

Determine the multiplicative depth (number of HE levels consumed) from the model.

**From ONNX model**: Run `scripts/analyze_model_depth.py --onnx <path>`

**From compiled DAG**: Run `scripts/analyze_model_depth.py --dag <path>`

**From model description**: Estimate depth using these rules:

| Operation | Levels Consumed | Notes |
|---|---|---|
| Conv2D / Dense / MatMul / Gemm | 1 | Linear layers |
| Polynomial activation (degree-d) | ceil(log2(d)) | degree-4 = 2 levels |
| Square | 1 | x^2 |
| Element-wise multiply (ciphertext * ciphertext) | 1 | Two encrypted inputs |
| Add / Reshape / Pool / BatchNorm | 0 | Pass-through |
| Scalar multiply / add (plaintext) | 0 | No level consumption |
| Bootstrapping | Restores levels | But adds significant latency |
**Key insight**: The depth determines the minimum number of CKKS levels needed, which constrains the minimum size of Q (ciphertext modulus chain).

## Phase 3: Weight Precision Assessment

If the user prioritizes efficiency, evaluate whether lower-precision training can reduce parameter sizes.

### The Core Trade-off

CKKS scale (per-level precision) must be large enough to represent the weights faithfully:
- **float32 weights**: scale = 30-34 bits per level
- **FP8 / INT8 weights**: scale = 21-24 bits per level

Smaller scale means each Q prime is smaller, so more levels fit within the same log2(QP) budget — or equivalently, the same depth can be achieved with a smaller N (faster inference).

### Practical Impact

| Weight Precision | Scale (bits) | Effect on N | Speed |
|---|---|---|---|
| float32 | 30-34 | Larger N required | Baseline |
| FP8 (8-bit float) | 21-24 | May enable 2x smaller N | ~3-4x faster |
| INT8 (8-bit integer) | 21-24 | May enable 2x smaller N | ~3-4x faster |

### Viability Checklist

Lower-precision training is viable when:
1. The model is small-to-medium (Conv-heavy models work best)
2. The baseline accuracy is high enough to absorb a small drop (typically ≤0.5%)
3. Training infrastructure supports the target precision (e.g., `torch.float8_e4m3fn`)

If lower-precision training is chosen, see `references/low_precision_training.md` for methodological guidance.

## Phase 4: Recommend Parameters

### Step 1: Determine log_scale

Based on weight precision from Phase 3:
- float32 weights → log_scale = 30-34
- Low-precision weights → log_scale = 21-24

### Step 2: Compute minimum log2(QP)

```
log2(Q) ≈ (depth + 1) * log_scale    # one prime per level + output
log2(P) ≈ log_scale * 2              # key-switching primes (1-2 primes)
log2(QP) ≈ log2(Q) + log2(P)
```

### Step 3: Find minimum N

Look up the HE Standard security bounds in `references/he_standard_security_table.md`. Find the smallest power-of-2 N where:

```
log2(QP) <= security_bound[N]    # for the target security level
```
HE Standard approximate bounds for 128-bit security:
| N | max log2(QP) |
|---|---|
| 8192 | 218 |
| 16384 | 438 |
| 32768 | 881 |
| 65536 | 1761 |

### Step 4: Handle Depth Overflow

If depth exceeds available levels for the smallest viable N:
1. **Increase N** — provides more levels but slower (each doubling of N ≈ 3-4x slower)
2. **Reduce scale** — if weight precision allows, smaller scale fits more levels
3. **Use bootstrapping** — refreshes levels mid-computation, but adds overhead
4. **Simplify model** — reduce depth (fewer layers, lower polynomial degree)

### Step 5: Build Q/P Prime Chains

Each prime q_i must be NTT-friendly: q_i ≡ 1 (mod 2N), and close to 2^log_scale in size.

Guidelines:
- Q chain: one prime per level, all ≈ 2^log_scale bits
- P chain: 1-4 primes for key-switching, same NTT-friendly constraint
- Bootstrapping: additional primes for StC (Subsum-to-Coeff), Sine evaluation, CtS (Coeff-to-Subsum)

### Automated Tools

If parameter sets are provided via a JSON catalog (`--catalog <path>`):

```bash
# Recommend from a catalog of known parameter sets
python scripts/recommend_params.py --depth 9 --weight-type fp8 --priority efficiency --catalog params.json

# Analyze model depth
python scripts/analyze_model_depth.py --onnx model.onnx
```

## Phase 5: Verify Security

### Quick Check (HE Standard Approximation)

Compute `log2(QP) / N` and compare against HE Standard bounds per N value. See `references/he_standard_security_table.md` for the full table.

This gives a conservative lower bound — actual security is typically higher.

### Exact Estimation (Recommended)

Use lattice-estimator with SageMath to get concrete bit-level security across multiple attack vectors:
```bash
# With lattice-estimator installed
python scripts/estimate_security.py --q <prime1> <prime2> ... --p <prime1> ... --N <N>

# Or from a catalog
python scripts/estimate_security.py --catalog params.json --preset <name>
```

The estimator runs 4 attacks and reports the minimum: primal_usvp, primal_bdd, dual, dual_hybrid. The weakest attack determines the security level.

### Distributions

CKKS security estimation models the RLWE problem with:
- **Secret distribution**: Approximated as DiscreteGaussian(3.19) per HE Standard convention (corresponds to ternary {-1, 0, 1} with density ~1/3)
- **Error distribution**: DiscreteGaussian(σ), typically σ ≈ 3.2

### If Security Is Insufficient

1. Bump N to next power of 2
2. Reduce log_scale (if weight precision allows)
3. Reduce depth (simplify model architecture)
4. Re-run verification

## Output

After completing all phases, provide:

1. **Recommended parameters**: N, log_scale, number of levels, Q primes, P primes
2. **Security report**: concrete bit-level estimate with attack breakdown
3. **Trade-off analysis**: comparison of viable options with speed/security trade-offs
4. **Next steps**: integration guidance for the user's specific FHE framework

## Resources

### scripts/

- `estimate_security.py` — Estimate security bits for parameter sets using lattice-estimator or HE Standard approximation. Accepts explicit primes or a JSON catalog.
- `analyze_model_depth.py` — Extract multiplicative depth from ONNX models or compiled JSON DAGs. Reports level-consuming operations and bootstrapping needs.
- `recommend_params.py` — Recommend parameter sets based on depth, weight type, and priority from a JSON catalog. Outputs comparison table.

### references/

- `ckks_basics.md` — CKKS parameter concepts (N, Q, P, scale, levels, slots, packing strategies).
- `he_standard_security_table.md` — HE Standard security bounds per N for 128-bit and 192-bit targets. LWE attack methodology.
- `low_precision_training.md` — Methodology for training lower-precision models to enable smaller FHE parameters.
- `parameter_catalog_template.json` — Template JSON format for defining known parameter sets. Use this to create a catalog for your specific FHE framework.