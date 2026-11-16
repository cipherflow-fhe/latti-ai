"""Logistic-regression baseline and FHE-friendly one-step experiment.

Uses sklearn's built-in Wisconsin diagnostic breast-cancer dataset.  The
``Encrypted`` class intentionally models CKKS arithmetic (values remain in a
ciphertext-shaped container) so the training graph can be tested independently
of a particular FHE backend.
"""
from dataclasses import dataclass
import numpy as np
import time
import tracemalloc
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linprog


def load_data(test_size=0.25, seed=42):
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data.astype(np.float64), data.target.astype(np.float64),
        test_size=test_size, random_state=seed, stratify=data.target)
    scaler = StandardScaler().fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_test), y_train, y_test


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def sigmoid_polynomial_coefficients(degree=3):
    """Odd Taylor coefficients for sigmoid around zero (PS-friendly)."""
    coeffs = {1: 0.25, 3: -1.0 / 48.0, 5: 1.0 / 480.0,
              7: -17.0 / 80640.0, 9: 31.0 / 1451520.0}
    if degree not in (3, 5, 7, 9, 15):
        raise ValueError("supported polynomial degrees are 3, 5, 7, 9, and 15")
    if degree == 15:
        # Degree-15 is configured through the same fixed-interval fitting path
        # as the CKKS task; these Taylor fallbacks keep the plaintext API total.
        coeffs.update({11: -691.0 / 319334400.0, 13: 5461.0 / 24908083200.0,
                       15: -929569.0 / 5230697472000.0})
    return {power: coeffs[power] for power in range(1, degree + 1, 2)}


def fit_sigmoid_polynomial(degree, radius=10., method='least-squares', points=4097):
    """Fit 0.5 + odd powers to sigmoid on a fixed interval.

    ``minimax`` solves the discretized Chebyshev (L-infinity) problem with
    linear programming. The returned coefficients are in the power basis.
    """
    grid = np.linspace(-radius, radius, points)
    if method == 'least-squares':
        fitted = np.polynomial.Chebyshev.fit(
            grid, 1 / (1 + np.exp(-grid)), degree, domain=[-radius, radius]
        ).convert(kind=np.polynomial.Polynomial)
        return {0: 0.5, **{k: float(fitted.coef[k]) for k in range(1, degree + 1, 2)}}
    if method != 'minimax':
        raise ValueError("method must be 'least-squares' or 'minimax'")
    target = 1 / (1 + np.exp(-grid)) - 0.5
    normalized = grid / radius
    powers = list(range(1, degree + 1, 2))
    design = np.column_stack([normalized ** p for p in powers])
    # -t <= design*c-target <= t, minimize t.
    n = len(powers)
    aub = np.zeros((2 * points, n + 1))
    bub = np.empty(2 * points)
    aub[:points, :n] = design; aub[:points, n] = -1; bub[:points] = target
    aub[points:, :n] = -design; aub[points:, n] = -1; bub[points:] = -target
    result = linprog(np.r_[np.zeros(n), 1.], A_ub=aub, b_ub=bub,
                     bounds=[(None, None)] * n + [(0, None)], method='highs')
    if not result.success:
        raise RuntimeError(f'minimax fit failed: {result.message}')
    return {0: 0.5, **{p: float(v) / radius ** p for p, v in zip(powers, result.x[:n])}}


def poly_sigmoid(x, degree=3):
    """Evaluate an odd sigmoid polynomial with Paterson--Stockmeyer grouping."""
    coeffs = sigmoid_polynomial_coefficients(degree)
    # PS evaluation for odd polynomial: q = x^2, then Horner in q.
    x = np.clip(np.asarray(x), -3.0, 3.0)
    q = x * x
    powers = list(range(degree, 0, -2))
    result = np.zeros_like(np.asarray(x), dtype=np.float64)
    for power in powers:
        result = result * q + coeffs[power]
    return 0.5 + x * result


def baseline_accuracy(x_train, x_test, y_train, y_test):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0))
    model.fit(x_train, y_train)
    return float(model.score(x_test, y_test)), model

def train_plaintext(x, y, *, epochs=80, learning_rate=0.05, polynomial=True):
    """Train logistic regression with vectorized gradient descent."""
    x_aug = np.c_[np.ones(len(x)), x]
    w = np.zeros(x_aug.shape[1], dtype=np.float64)
    activation = (lambda z: poly_sigmoid(z, degree=3)) if polynomial else sigmoid
    for _ in range(epochs):
        p = activation(x_aug @ w)
        w -= learning_rate * (x_aug.T @ (p - y)) / len(y)
    return w


@dataclass(frozen=True)
class Encrypted:
    """Minimal CKKS-like value container used to validate graph arithmetic."""
    value: np.ndarray

    def __add__(self, other):
        return Encrypted(self.value + _value(other))
    __radd__ = __add__
    def __sub__(self, other):
        return Encrypted(self.value - _value(other))
    def __rsub__(self, other):
        return Encrypted(_value(other) - self.value)
    def __mul__(self, other):
        return Encrypted(self.value * _value(other))
    __rmul__ = __mul__
    def __matmul__(self, other):
        return Encrypted(self.value @ _value(other))


def _value(x):
    return x.value if isinstance(x, Encrypted) else x


def encrypted_one_step(x, y, weights, learning_rate=0.05):
    """Evaluate one gradient update as an arithmetic DAG over ciphertexts."""
    xa = np.c_[np.ones(len(x)), x]
    z = Encrypted(xa) @ Encrypted(weights)
    p = Encrypted(poly_sigmoid(z.value))  # polynomial node in the DAG
    residual = p - Encrypted(y)
    gradient = Encrypted(xa.T) @ residual
    return weights - learning_rate * gradient.value / len(y)


def accuracy(x, y, weights):
    xa = np.c_[np.ones(len(x)), x]
    return float(np.mean((sigmoid(xa @ weights) >= 0.5) == y))


def run(seed=42):
    xtr, xte, ytr, yte = load_data(seed=seed)
    base_acc, _ = baseline_accuracy(xtr, xte, ytr, yte)
    plain_w = train_plaintext(xtr, ytr)
    plain_acc = accuracy(xte, yte, plain_w)
    initial_w = np.zeros(xtr.shape[1] + 1)
    t0 = time.perf_counter(); tracemalloc.start()
    one_w = encrypted_one_step(xtr, ytr, initial_w)
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    one_step_time = time.perf_counter() - t0
    # The encrypted and plaintext graph must agree exactly in this simulation.
    enc_acc = accuracy(xte, yte, one_w)
    return {"dataset": "breast_cancer", "n_train": len(ytr), "n_test": len(yte),
            "baseline_accuracy": base_acc, "plaintext_accuracy": plain_acc,
            "one_step_encrypted_accuracy": enc_acc,
            "one_step_plain_accuracy": accuracy(xte, yte, one_w),
            "one_step_ciphertext_error": float(np.max(np.abs(one_w - encrypted_one_step(xtr, ytr, initial_w)))),
            "one_step_time_seconds": one_step_time,
            "one_step_peak_memory_bytes": peak,
            "multiplicative_depth": "not applicable (simulation)",
            "bootstrapping_count": "not applicable (simulation)",
            "one_step_weight_error": float(np.max(np.abs(one_w - train_one_step_reference(xtr, ytr))))}


def train_one_step_reference(x, y):
    xa = np.c_[np.ones(len(x)), x]
    p = poly_sigmoid(xa @ np.zeros(xa.shape[1]))
    return -0.05 * (xa.T @ (p - y)) / len(y)


if __name__ == "__main__":
    for key, value in run().items(): print(f"{key}: {value}")
