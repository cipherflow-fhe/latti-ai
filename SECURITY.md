# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| latest  | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in LattiAI, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please send an email to **info@cipherflow.cn** with:

- A description of the vulnerability
- Steps to reproduce the issue
- Affected versions
- Any potential impact assessment

We will acknowledge receipt within 3 business days and aim to provide an initial assessment within 10 business days.

## Scope

This security policy covers:

- The LattiAI inference framework (`inference/`)
- The model compiler and training tools (`training/`)
- CI/CD configurations and Docker setup

The [LattiSense](https://github.com/cipherflow-fhe/lattisense) submodule (`inference/lattisense/`) has its own security policy.

## Cryptographic Notice

LattiAI implements privacy-preserving inference using the CKKS Fully Homomorphic Encryption scheme. The cryptographic primitives are provided by the LattiSense library. Security of the overall system depends on proper parameter selection and correct usage of the encryption pipeline. If you believe there is a weakness in the cryptographic implementation, please include detailed technical analysis in your report.
