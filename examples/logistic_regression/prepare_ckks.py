"""Prepare a fixed split, matched plaintext references and an unrolled CKKS task."""
import argparse
import json
import numpy as np
import time
import tracemalloc
import sklearn
from sklearn.linear_model import LogisticRegression
from experiment import load_data, sigmoid_polynomial_coefficients, fit_sigmoid_polynomial
from ckks_step import build
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='runs/logistic/ckks_step')
    parser.add_argument('--degree', type=int, default=3, choices=(3, 5, 7, 9, 15))
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--approximation', choices=('taylor', 'least-squares', 'minimax'), default='taylor')
    parser.add_argument('--radius', type=float, default=10.)
    parser.add_argument('--checkpoints', action='store_true')
    args = parser.parse_args()
    if args.steps < 1 or args.learning_rate <= 0 or args.radius <= 0:
        parser.error('steps, learning-rate and radius must be positive')
    x, xt, y, yt = load_data()
    xa = np.c_[np.ones(len(x)), x]
    initial = np.zeros(xa.shape[1])
    data = {f'x{j}': xa[:, j].tolist() for j in range(xa.shape[1])}
    data.update({f'w{j}': [float(initial[j])] for j in range(len(initial))})
    data['labels'] = y.tolist()
    coefficients = {0: 0.5, **sigmoid_polynomial_coefficients(args.degree)}
    if args.approximation in ('least-squares', 'minimax'):
        coefficients = fit_sigmoid_polynomial(args.degree, args.radius, args.approximation, 4097)
    for step in range(args.steps):
        data.update({f'step{step}_coefficient{k}': [coefficients.get(k, 0.)] for k in range(args.degree+1)})
    w = initial.copy()
    ranges = []
    history = []
    tracemalloc.start()
    start = time.perf_counter()
    for step in range(args.steps):
        z = xa @ w
        ranges.append([float(z.min()), float(z.max())])
        probabilities = sum(c * z**k for k, c in coefficients.items())
        w -= args.learning_rate * (xa.T @ (probabilities - y)) / len(y)
        if not np.all(np.isfinite(w)):
            raise ValueError(f'Plaintext training diverged at step {step + 1}')
        history.append(w.tolist())
    data['plaintext_training_seconds'] = time.perf_counter() - start
    data['plaintext_traced_peak_bytes'] = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    start = time.perf_counter()
    baseline = LogisticRegression(max_iter=2000, random_state=0).fit(x, y)
    data['sklearn_baseline'] = dict(test_accuracy=float(baseline.score(xt, yt)),
        fit_seconds=time.perf_counter() - start, solver='lbfgs', C=1., max_iter=2000,
        n_iter=baseline.n_iter_.tolist(), note='L2 regularization; convergence-oriented reference, not matched updates')
    exact_w = initial.copy()
    start = time.perf_counter()
    for _ in range(args.steps):
        z = xa @ exact_w
        p = 1 / (1 + np.exp(-z))
        exact_w -= args.learning_rate * (xa.T @ (p - y)) / len(y)
    test_aug = np.c_[np.ones(len(xt)), xt]
    data['exact_sigmoid_reference'] = dict(
        test_accuracy=float(np.mean((test_aug @ exact_w >= 0) == yt)),
        training_seconds=time.perf_counter() - start,
        weight_max_difference_from_polynomial=float(np.max(np.abs(exact_w - w))))
    if args.approximation == 'least-squares' and np.max(np.abs(ranges)) > args.radius:
        raise ValueError('Training logits exceed the fixed polynomial fitting interval')
    build(args.output, x.shape[1], args.degree, learning_rate=args.learning_rate,
          batch_size=len(y), steps=args.steps, checkpoints=args.checkpoints)
    manifest_path = Path(args.output, 'experiment.json')
    manifest = json.loads(manifest_path.read_text())
    manifest.update(approximation=args.approximation, polynomial_coefficients=coefficients,
                    approximation_radius=args.radius, fitting_grid_points=4097,
                    numpy_version=np.__version__, sklearn_version=sklearn.__version__)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    data['expected_weight_history'] = history
    data['expected_weights'] = w.tolist()
    data['logit_ranges'] = ranges
    data['x_test'] = xt.tolist()
    data['y_test'] = yt.tolist()
    Path(args.output, 'data.json').write_text(json.dumps(data))
