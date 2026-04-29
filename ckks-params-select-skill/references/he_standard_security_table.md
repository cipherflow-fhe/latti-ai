# HE Standard Security Reference

## Security Levels (HomomorphicEncryption.org Standard)

| Security Level | Use Case |
|---|---|
| 128-bit | General use, short-to-medium term |
| 192-bit | Long-term security |
| 256-bit | Highest security, sensitive data |

## Maximum log2(QP) by N for Target Security

These are approximate upper bounds. Always verify with lattice-estimator using exact prime values.

### 128-bit Security

| N | logN | Max log2(QP) | Slots |
|---|---|---|---|
| 1024 | 10 | 27 | 512 |
| 2048 | 11 | 54 | 1024 |
| 4096 | 12 | 109 | 2048 |
| 8192 | 13 | 218 | 4096 |
| 16384 | 14 | 438 | 8192 |
| 32768 | 15 | 881 | 16384 |
| 65536 | 16 | 1761 | 32768 |

### 192-bit Security

| N | logN | Max log2(QP) | Slots |
|---|---|---|---|
| 1024 | 10 | 19 | 512 |
| 2048 | 11 | 37 | 1024 |
| 4096 | 12 | 75 | 2048 |
| 8192 | 13 | 152 | 4096 |
| 16384 | 14 | 305 | 8192 |
| 32768 | 15 | 611 | 16384 |
| 65536 | 16 | 1227 | 32768 |

## Security Estimation Methodology

CKKS security is based on the Ring-LWE (RLWE) problem. The concrete security level is estimated by modeling the best known lattice attacks:

1. **primal_usvp** (Unique-SVP): Primal attack via unique shortest vector
2. **primal_bdd** (BDD): Primal attack via bounded distance decoding
3. **dual**: Dual lattice attack
4. **dual_hybrid**: Dual hybrid attack (often the weakest)

Security = minimum bits across all attacks.

### LWE Parameters for CKKS

- n = N (ring dimension)
- q = Q * P (total ciphertext modulus)
- Secret distribution: typically ternary {-1, 0, 1} with density ~1/3
- Error distribution: DiscreteGaussian(sigma ~3.2)

The ternary secret is commonly approximated as DiscreteGaussian(3.19) per HE Standard convention for estimation purposes. Adjust based on your FHE library's actual distributions.