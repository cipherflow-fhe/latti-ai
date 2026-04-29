# Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from scipy.integrate import quad
from scipy.special import factorial
import matplotlib.pyplot as plt
import warnings
import os
import torch
import torch.nn as nn


# =====================
# Probabilist's Hermite polynomials
# =====================
def hermite_prob(n, x):
    """Compute probabilist's Hermite polynomial He_n(x)."""
    if n == 0:
        return np.ones_like(x) if hasattr(x, '__len__') else 1.0
    elif n == 1:
        return x
    elif n == 2:
        return x**2 - 1
    elif n == 3:
        return x**3 - 3 * x
    elif n == 4:
        return x**4 - 6 * x**2 + 3
    elif n == 5:
        return x**5 - 10 * x**3 + 15 * x
    else:
        return x * hermite_prob(n - 1, x) - (n - 1) * hermite_prob(n - 2, x)


# =====================
# Activation function implementation
# =====================
def relu(x):
    """ReLU function."""
    return np.maximum(0, x) if hasattr(x, '__len__') else max(0, x)


def silu(x, threshold=30):
    """
    Numerically stable SiLU function.
    Args:
        threshold: use approximation when |x| > threshold
    """
    if np.isscalar(x):
        if x > threshold:
            return x
        elif x < -threshold:
            return 0.0
        return x / (1 + np.exp(-x)) if x >= 0 else x * np.exp(x) / (1 + np.exp(x))
    else:
        result = np.zeros_like(x)
        mask_pos = x > threshold
        mask_neg = x < -threshold
        mask_mid = ~(mask_pos | mask_neg)
        x_mid = x[mask_mid]
        result[mask_pos] = x[mask_pos]
        result[mask_neg] = 0.0
        result[mask_mid] = np.where(
            x_mid >= 0, x_mid / (1 + np.exp(-x_mid)), x_mid * np.exp(x_mid) / (1 + np.exp(x_mid))
        )
        return result


# =====================
# Coefficient computation (with parameter support)
# =====================
def compute_coefficients(func, max_n=5, tol=1e-8, limit=20, **func_kwargs):
    """
    Compute Hermite expansion coefficients for a given function.
    Args:
        func: target function (any callable)
        max_n: maximum order
        tol: integration tolerance
        limit: integration limit
        func_kwargs: extra arguments passed to func
    """
    coefficients = []

    for n in range(max_n + 1):

        def integrand_scalar(x):
            """quad 要求返回 Python float；这里显式标量化避免 1-d ndarray。"""
            x_scalar = float(x)
            # 保证传给 activation 的是 1-d，兼容依赖向量输入的 numpy_func
            x_arr = np.array([x_scalar], dtype=float)
            if func_kwargs:
                fx = func(x_arr, **func_kwargs)
            else:
                fx = func(x_arr)

            fx_scalar = float(np.asarray(fx, dtype=float).reshape(-1)[0])
            h_scalar = float(np.asarray(hermite_prob(n, x_arr), dtype=float).reshape(-1)[0])
            gauss = float(np.exp(-(x_scalar**2) / 2))
            return fx_scalar * h_scalar * gauss

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            I_n, _ = quad(
                integrand_scalar,
                -limit,
                limit,
                epsabs=tol,
                limit=1000,  # increase integration subintervals
            )

        c_n = I_n / (np.sqrt(2 * np.pi) * np.sqrt(factorial(n)))
        coefficients.append(c_n)

    return coefficients


# =====================
# Visualization utility
# =====================
def plot_activation(func, func_name, coeffs, x_range=(-4, 4), save_dir='activation_plots', **func_kwargs):
    """Plot activation function and its Hermite approximation."""
    os.makedirs(save_dir, exist_ok=True)
    x = np.linspace(x_range[0], x_range[1], 500)
    # Pass extra arguments when calling the function
    if func_kwargs:
        y_true = func(x, **func_kwargs)
    else:
        y_true = func(x)

    # Compute approximation at different orders
    approx = {}
    for n_terms in [6]:
        if n_terms <= len(coeffs) - 1:
            approx[n_terms] = sum(
                c * hermite_prob(n, x) / np.sqrt(factorial(n)) for n, c in enumerate(coeffs[: n_terms + 1])
            )

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, 'k-', lw=3, label=f'True {func_name}')
    for n, y_approx in approx.items():
        plt.plot(x, y_approx, '--', label=f'Hermite N={n}', alpha=0.8)
    plt.title(f'{func_name} Function Approximation')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    # plt.show()
    filename = f'{func_name.replace(" ", "_").replace("(", "").replace(")", "")}_approx.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f'Plot saved to: {filepath}')

    # Return approximation error for analysis
    rel_error = np.sqrt(np.mean((y_true - approx[5]) ** 2)) if 5 in approx else None
    return rel_error


def get_hermite_coeffs_for_module(module_cls, degree=4, **kwargs):
    """Compute Hermite expansion coefficients for any nn.Module activation.

    Instantiates the module, evaluates it via PyTorch forward pass,
    and computes Hermite coefficients via numerical integration.

    Args:
        module_cls: nn.Module class (e.g. nn.GELU, nn.Tanh, nn.Mish).
        degree:     Polynomial degree.
        **kwargs:   Extra arguments passed to compute_coefficients
                    (e.g. tol, limit).

    Returns:
        Tuple of (degree+1) floats: (a_0, a_1, ..., a_degree),
        ready to use as hermite_coeffs in Simple_Polyrelu.
    """
    module = module_cls()
    module.eval()

    def numpy_func(x):
        x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        with torch.no_grad():
            result = module(torch.as_tensor(x_arr, dtype=torch.float64))
        return result.numpy()

    raw_coeffs = compute_coefficients(numpy_func, max_n=degree, **kwargs)
    return tuple(c / np.sqrt(factorial(n)) for n, c in enumerate(raw_coeffs))


# =====================
# Main program
# =====================
if __name__ == '__main__':
    # Configuration
    MAX_N = 10
    INTEGRAL_LIMIT = 20
    X_RANGE = (-5, 5)
    SAVE_DIR = 'activation_plots'  # plot save directory
    # Test all activation functions
    activations = [
        # (relu, "ReLU", {}),
        (silu, 'SiLU', {'threshold': 30})
    ]

    results = {}

    for func, name, kwargs in activations:
        print(f'\n{"=" * 40}')
        print(f'Computing Hermite coefficients for {name}')
        print(f'{"=" * 40}')

        # Compute coefficients
        coeffs = compute_coefficients(func, max_n=MAX_N, limit=INTEGRAL_LIMIT, **kwargs)

        # Print coefficients
        print('Coefficients:')
        for n, c in enumerate(coeffs):
            print(f'  f_{n} = {c:.8f}')

        # Plot and save
        rel_error = plot_activation(func, name, coeffs, x_range=X_RANGE, save_dir=SAVE_DIR, **kwargs)

        if rel_error is not None:
            print(f'{name} order-5 approximation RMSE: {rel_error:.6f}')
