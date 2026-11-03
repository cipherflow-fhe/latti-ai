"""Independent plaintext checks; numerical evidence, not interval proofs."""

import math

import mpmath as mp
import numpy as np

from training.nn_tools.paper_maxpool_polynomials import (
    PAPER_CONFIG, approximate_max, evaluate_composite_sign, evaluate_odd_poly,
    paper_coefficients, validate_metadata,
)
from training.nn_tools.fixed_maxpool_profiles import get_maxpool_profile


def _eval(coeffs, x):
    result = mp.mpf(0)
    for c in reversed(coeffs):
        result = result * x + c
    return result


def real_roots_in_interval(coeffs, lo, hi):
    """Numerically isolate roots via derivative monotonic intervals.

    Coefficients use ascending ordinary powers. Recursive critical points
    partition the interval, including repeated roots at critical points.
    This uses high precision but is not certified interval arithmetic.
    """
    coeffs = list(coeffs)
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs.pop()
    if len(coeffs) <= 1:
        return []
    if len(coeffs) == 2:
        root = -coeffs[0] / coeffs[1]
        return [root] if lo <= root <= hi else []
    critical = real_roots_in_interval(
        [i * coeffs[i] for i in range(1, len(coeffs))], lo, hi)
    points = [lo] + [r for r in critical if lo < r < hi] + [hi]
    tol = mp.power(10, -(mp.mp.dps // 2))
    roots = [x for x in points if abs(_eval(coeffs, x)) < tol]
    for left, right in zip(points, points[1:]):
        fl, fr = _eval(coeffs, left), _eval(coeffs, right)
        if fl * fr >= 0:
            continue
        for _ in range(mp.mp.dps * 4):
            mid = (left + right) / 2
            fm = _eval(coeffs, mid)
            if right - left < tol:
                break
            if fl * fm <= 0:
                right = mid
            else:
                left, fl = mid, fm
        roots.append((left + right) / 2)
    ordered = []
    for root in sorted(roots):
        if not ordered or abs(root - ordered[-1]) > tol * 10:
            ordered.append(root)
    return ordered


def validate_paper_coefficients(alpha, dps=80, samples=257):
    """Propagate full positive images using all numerical critical points."""
    if dps < 50 or samples < 2:
        raise ValueError('require dps >= 50 and samples >= 2')
    validate_metadata(alpha)
    config = PAPER_CONFIG[alpha]
    with mp.workdps(dps):
        coeffs = [[mp.mpf(str(c)) for c in stage]
                  for stage in paper_coefficients(alpha)]
        epsilon = mp.mpf(config['zeta']) / 2**alpha
        lo, hi = epsilon, mp.mpf(1)
        domains = []
        for stage in coeffs:
            derivative = [mp.mpf(0)] * (2 * len(stage) - 1)
            for j, c in enumerate(stage):
                derivative[2*j] = (2*j+1) * c
            critical = real_roots_in_interval(derivative, lo, hi)
            values = [evaluate_odd_poly(x, stage) for x in [lo, hi] + critical]
            out_lo, out_hi = min(values), max(values)
            domains.append({
                'positive_input': [str(lo), str(hi)],
                'positive_output': [str(out_lo), str(out_hi)],
                'critical_points': [str(x) for x in critical],
                'stage_sign_max_error': str(max(abs(out_lo-1), abs(out_hi-1))),
            })
            if out_lo <= 0:
                raise ValueError('stage image crosses zero: sign domain assumption fails')
            lo, hi = out_lo, out_hi
        error = max(abs(lo-1), abs(hi-1))
        float_error = mp.mpf(0)
        for i in range(samples):
            x = float(epsilon + (1-epsilon)*i/(samples-1))
            # Compare exactly the same binary64 input, not nearby decimals.
            reference = mp.mpf(x)
            for stage in coeffs:
                reference = evaluate_odd_poly(reference, stage)
            float_error = max(float_error, abs(mp.mpf(float(evaluate_composite_sign(x, alpha))) - reference))
        return {
            'alpha': alpha, 'zeta': config['zeta'], 'epsilon': float(epsilon),
            'degrees': list(config['degrees']), 'paper_depth': config['depth'],
            'actual_ckks_level_cost': None,
            'source': 'paper_appendix_A_user_transcription',
            'odd_coefficients': [[str(c) for c in s] for s in coeffs],
            'stage_domains': domains, 'measured_max_error': str(error),
            'sign_target': str(mp.mpf(2)**(-(alpha-1))),
            'numerical_sign_bound_pass': bool(error <= mp.mpf(2)**(-(alpha-1))),
            'float64_reference_max_difference': str(float_error),
            'float64_samples': samples, 'decimal_precision': dps,
            'validation_method': 'recursive numerical derivative root isolation; not a certified proof',
        }


def paper_maxpool2d_reference(x, alpha, bound, kernel_size=3, stride=2, padding=1):
    """NCHW NumPy reference with paper normalization and valid-tap masks.

    Missing taps bypass comparisons, matching negative-infinity padding
    without evaluating a polynomial at infinity or introducing zero taps.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 4 or not np.isfinite(x).all():
        raise ValueError('expected finite NCHW input')
    if not math.isfinite(bound) or bound <= 0 or np.max(np.abs(x)) > bound:
        raise ValueError('bound must be positive and cover the input')
    if kernel_size <= 0 or stride <= 0 or not 0 <= padding <= kernel_size//2:
        raise ValueError('invalid pooling parameters')
    rounds = (kernel_size**2 - 1).bit_length()
    denominator = .5 - (rounds-1)*2.**(-alpha)
    if denominator <= 0:
        raise ValueError('invalid normalization denominator')
    scale = bound / denominator
    h, w = x.shape[-2:]
    oh, ow = (h+2*padding-kernel_size)//stride+1, (w+2*padding-kernel_size)//stride+1
    if min(oh, ow) <= 0:
        raise ValueError('empty output')
    out = np.empty(x.shape[:2] + (oh, ow))
    for row in range(oh):
        for col in range(ow):
            taps = []
            for ky in range(kernel_size):
                for kx in range(kernel_size):
                    iy, ix = row*stride+ky-padding, col*stride+kx-padding
                    taps.append(x[..., iy, ix]/scale+.5 if 0 <= iy < h and 0 <= ix < w else None)
            while len(taps) > 1:
                next_taps = []
                for i in range(0, len(taps)-1, 2):
                    a, b = taps[i:i+2]
                    next_taps.append(b if a is None else a if b is None else approximate_max(a, b, alpha))
                if len(taps) % 2:
                    next_taps.append(taps[-1])
                taps = next_taps
            out[..., row, col] = scale * (taps[0]-.5)
    return out


def build_maxpool_execution_config(alpha, upper_bound, input_bound, comparison_B=1.0,
                                   *, k_first=1.0, k_later=1.05,
                                   eps=1e-3):
    """Prepare user-facing MaxPool parameters for the deferred-half path.

    ``input_bound`` is the calibrated absolute input range. ``upper_bound``
    is the normalized scale, while ``k_first``/``k_later`` describe the
    existing alpha=7 round convention. Coefficients remain fixed; only P0's
    input scaling is generated here. ``comparison_B`` is the user-selected
    first-round difference scale; later rounds retain the deferred-half rule.
    """
    profile = get_maxpool_profile(alpha)
    for name, value in [('upper_bound', upper_bound), ('input_bound', input_bound),
                        ('comparison_B', comparison_B),
                        ('k_first', k_first), ('k_later', k_later)]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be positive and finite')
    if not math.isfinite(eps) or eps < 0:
        raise ValueError('eps must be finite and nonnegative')
    degrees = profile['degrees']
    coeffs = [[float(c) for c in stage] for stage in profile['odd_coefficients']]
    rounds = []
    for round_index in range(1):  # round scaling is expanded by the tree planner
        base = comparison_B
        rounds.append({'round_index': round_index, 'base_B': base,
                       'effective_B': base, 'p0_scale': 1/(2*base)})
    return {
        'alpha': alpha, 'degrees': degrees, 'paper_depth': profile['paper_depth'],
        'upper_bound': float(upper_bound), 'input_bound': float(input_bound),
        'running_max': float(input_bound), 'eps': float(eps),
        'input_scale': float(input_bound / upper_bound + eps),
        'input_normalization_multiplier': float(1/(input_bound / upper_bound + eps)),
        'k_first': float(k_first), 'k_later': float(k_later),
        'comparison_B': float(comparison_B),
        'stages': [{'degree': d, 'odd_coefficients': c} for d, c in zip(degrees, coeffs)],
        'round_template': rounds,
        'source': profile['config_version'],
    }


def configure_poly_maxpool(alpha, upper_bound, B, *, running_max,
                           k_later=1.05, eps=1e-3):
    """Build an execution config from the public three-parameter API.

    Users choose ``alpha``, ``upper_bound`` and ``B``. ``running_max`` is
    model calibration metadata supplied by conversion code, never a per-input
    inference option. Degree combinations and coefficients are selected only
    through the validated alpha profile.
    """
    return build_maxpool_execution_config(
        alpha, upper_bound, running_max, comparison_B=B,
        k_first=B, k_later=k_later, eps=eps,
    )


def deferred_half_binary_max(a, b, config, round_index=0):
    """Plaintext equivalent of one existing deferred-half binary Max node.

    The node returns ``2*max(a,b)`` approximately; callers divide by two per
    reduction round. This mirrors the current CKKS graph's deferred half.
    """
    if not isinstance(round_index, int) or round_index < 0:
        raise ValueError('invalid round index')
    base = config['comparison_B'] if round_index == 0 else config['k_later']
    bound = base if round_index == 0 else base * (2 ** round_index)
    coeffs = config['stages']
    difference = (a - b) / (2 * bound)
    sign_value = difference
    for stage in coeffs:
        sign_value = evaluate_odd_poly(sign_value, stage['odd_coefficients'])
    return a + b + (a - b) * sign_value


def deferred_half_maxpool2d_reference(x, alpha, input_bound, upper_bound=1.0,
                                      kernel_size=3, stride=2, padding=1,
                                      comparison_B=1.0, k_first=1.0,
                                      k_later=1.05, eps=1e-3):
    """NCHW plaintext model of the current deferred-half MaxPool schedule.

    This is intentionally separate from :func:`paper_maxpool2d_reference`:
    it models latti-ai's per-round B scaling and deferred division, so its
    output can be compared directly with the future CKKS implementation.
    """
    config = build_maxpool_execution_config(alpha, upper_bound, input_bound,
                                            comparison_B=comparison_B,
                                            k_first=k_first, k_later=k_later,
                                            eps=eps)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 4 or not np.isfinite(x).all() or np.max(np.abs(x)) > input_bound:
        raise ValueError('input is outside calibrated input_bound')
    h, w = x.shape[-2:]
    oh, ow = (h+2*padding-kernel_size)//stride+1, (w+2*padding-kernel_size)//stride+1
    if min(oh, ow) <= 0:
        raise ValueError('empty output')
    normalized = x / config['input_scale']
    out = np.empty(x.shape[:2] + (oh, ow))
    for row in range(oh):
        for col in range(ow):
            taps = []
            for ky in range(kernel_size):
                for kx in range(kernel_size):
                    iy, ix = row*stride+ky-padding, col*stride+kx-padding
                    taps.append(normalized[..., iy, ix] if 0 <= iy < h and 0 <= ix < w else None)
            for r in range((len(taps)-1).bit_length()):
                nxt = []
                for i in range(0, len(taps)-1, 2):
                    a, b = taps[i:i+2]
                    nxt.append(b if a is None else a if b is None else
                               deferred_half_binary_max(a, b, config, r))
                if len(taps) % 2:
                    # Deferred-half representation doubles every live value,
                    # including an unpaired carry, before the next round.
                    nxt.append(None if taps[-1] is None else 2 * taps[-1])
                taps = nxt
            out[..., row, col] = config['input_scale'] * taps[0] / (2 ** ((len(taps) and (kernel_size**2-1).bit_length())))
    return out, config
