# CKKS Parameter Basics

## Core Parameters

### N (polynomial modulus degree)
- Must be a power of 2: 1024, 2048, 4096, 8192, 16384, 32768, 65536
- Number of CKKS plaintext slots = N/2
- Larger N = more security for same Q, more slots for SIMD parallelism, but slower computation
- Common N values: 8192, 16384, 32768, 65536

### Q (ciphertext modulus chain)
- Product of prime numbers q_0, q_1, ..., q_L
- Each prime corresponds to one CKKS "level"
- Each HE multiplication consumes one level (drops the top prime)
- Primes must be NTT-friendly: each q_i ≡ 1 (mod 2N)
- Larger total Q = more levels/precision, but less security for same N

### P (key-switching / special modulus)
- Additional primes used for key-switching and bootstrapping operations
- Not consumed during computation; auxiliary modulus
- P primes also must be NTT-friendly

### log2(QP) (total modulus bit-length)
- Sum of bit-lengths of all Q and P primes
- Key security metric: ratio log2(QP)/N determines security level

### Scale (per-level precision)
- Each level carries a fixed-point scale, typically a power of 2
- log_default_scale = bit-length of scale (e.g., 21, 24, 30, 34, 40, 45)
- Larger scale = more arithmetic precision per level, but consumes more of Q
- Scale choice depends on weight precision: FP8 weights need 21-24 bits, float32 weights need 30-34 bits

### Levels (multiplicative depth)
- Number of HE multiplications possible before bootstrapping
- max_level = len(Q) - 1
- Each multiplication, square, or polynomial activation consumes one level
- Bootstrapping restores levels but is expensive

## Key Relationships

```
log2(Q) >= levels * log_default_scale + log_scale_output
log2(QP) = log2(Q) + log2(P)
security_bits ~ f(log2(QP) / N)    # approximate, depends on exact primes
```

## Distributions for Security Estimation

CKKS security estimation models the RLWE problem with:
- **Secret distribution**: typically ternary {-1, 0, 1} with density ~1/3, approximated as DiscreteGaussian(3.19) per HE Standard convention.
- **Error distribution**: DiscreteGaussian(sigma ~3.2), approximated as DiscreteGaussian(3.19) for security estimation.

## Slot Layout

CKKS encodes N/2 complex numbers (practically N/2 real numbers) into one ciphertext.
The "slots" enable SIMD parallelism — one HE operation processes all slots simultaneously.
For common N values:
- N=8192 -> 4096 slots
- N=16384 -> 8192 slots
- N=32768 -> 16384 slots
- N=65536 -> 32768 slots

## Packing Strategies

Common packing strategies:

1. **Ordinary packing**: Each ciphertext holds multiple channels. Simple but may waste slots.
2. **Multiplexed packing**: Channels are interleaved for higher slot utilization. Better throughput for most models.
3. **Interleaved packing**: Alternative interleaving for specific layer shapes.

## Bootstrapping

When the computation graph depth exceeds available levels, bootstrapping is inserted to refresh ciphertexts:
- Consumes several levels itself (bootstrapping circuit depth)
- After bootstrapping, levels are restored to near-maximum
- Parameters with sparse secrets (e.g., H=192 non-zero coefficients) enable efficient bootstrapping
- Bootstrapping is typically triggered automatically by the model compiler when depth exceeds available levels