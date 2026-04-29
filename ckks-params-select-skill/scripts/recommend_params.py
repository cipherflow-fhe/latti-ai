#!/usr/bin/env python3
"""Recommend CKKS parameters based on model requirements and user priorities.

Given the model's multiplicative depth, weight type, and the user's priority
(security or efficiency), recommend the best CKKS parameter set from a
user-provided catalog or compute minimum requirements from first principles.

Usage:
  python recommend_params.py --depth 9 --weight-type fp8 --priority efficiency
  python recommend_params.py --depth 9 --weight-type float32 --priority security --security-target 128
  python recommend_params.py --depth 9 --catalog params.json --priority security
  python recommend_params.py --depth 17 --weight-type float32 --priority security --catalog params.json
"""

import argparse
import json
import math
import sys


# HE Standard approximate bounds for common N values.
# These map N -> max log2(QP) for the given security level.
SECURITY_BOUNDS_128 = {
    8192: 218,
    16384: 438,
    32768: 881,
    65536: 1761,
}

SECURITY_BOUNDS_192 = {
    8192: 152,
    16384: 305,
    32768: 611,
    65536: 1227,
}


def compute_min_qp_bits(depth, log_scale):
    """Compute minimum log2(QP) needed for given depth and scale.

    Q needs: depth * log_scale (one scale per level) + output_scale
    P needs: key-switching primes, typically 1-2 primes of log_scale size.
    """
    log_q = (depth + 1) * log_scale
    log_p = log_scale * 2  # approximate: 1-2 P primes
    return log_q + log_p


def find_min_N(log_qp, security_target=128):
    """Find smallest N that achieves the target security level.

    Returns None if log_qp exceeds all known bounds.
    """
    if security_target >= 192:
        bounds = SECURITY_BOUNDS_192
    else:
        bounds = SECURITY_BOUNDS_128

    for N in sorted(bounds.keys()):
        if log_qp <= bounds[N]:
            return N
    return None

def recommend_from_catalog(depth, weight_type, priority, catalog, security_target=128):
    """Recommend parameter sets from a user-provided catalog.

    Args:
        depth: Model multiplicative depth (levels needed).
        weight_type: 'float32', 'fp8', or other precision identifier.
        priority: 'security' or 'efficiency'.
        catalog: Dict from loaded JSON catalog (see parameter_catalog_template.json).
        security_target: Minimum security bits.

    Returns:
        List of recommendation dicts, sorted by suitability.
    """
    sets = catalog.get('parameter_sets', {})
    speed_table = catalog.get('relative_speed', {})

    recommendations = []

    for name, spec in sets.items():
        spec_wt = spec.get('weight_type', 'float32')

        # Check weight type compatibility
        if weight_type and spec_wt != weight_type:
            continue

        # Check depth
        needs_btp = spec.get('needs_btp', False)
        max_level = spec.get('max_level', spec.get('levels', 0))

        if not needs_btp and depth > max_level:
            continue

        # Score: lower = better
        if priority == 'efficiency':
            score = speed_table.get(name, spec.get('N', 99999))
        else:
            # Security priority: prefer smallest viable N
            score = spec['N']

        recommendations.append({
            'name': name,
            'N': spec['N'],
            'max_level': max_level,
            'log_scale': spec.get('log_scale', 0),
            'weight_type': spec_wt,
            'needs_btp': needs_btp,
            'description': spec.get('description', ''),
            'depth_margin': max_level - depth if not needs_btp else float('inf'),
            'relative_speed': speed_table.get(name, None),
            'score': score,
        })

    recommendations.sort(key=lambda r: r['score'])
    return recommendations

def compute_custom_params(depth, weight_type, security_target=128):
    """Compute a custom parameter set from first principles.

    Returns dict with recommended N, log_scale, and verification data.
    """
    if weight_type in ('fp8', 'int8', 'low_precision'):
        log_scale = 21  # minimum for 8-bit weights
    else:
        log_scale = 30  # minimum for float32

    log_qp = compute_min_qp_bits(depth, log_scale)
    min_N = find_min_N(log_qp, security_target)

    return {
        'depth': depth,
        'log_scale': log_scale,
        'log_qp_estimate': log_qp,
        'min_N': min_N,
        'security_target': security_target,
        'feasible': min_N is not None,
    }


def main():
    parser = argparse.ArgumentParser(description='Recommend CKKS parameters')
    parser.add_argument('--depth', type=int, required=True,
                        help='Model multiplicative depth')
    parser.add_argument('--weight-type', default='float32',
                        help='Weight precision type (e.g., float32, fp8, int8)')
    parser.add_argument('--priority', choices=['security', 'efficiency'], default='security',
                        help='Optimization priority')
    parser.add_argument('--security-target', type=int, default=128,
                        help='Minimum security bits (default: 128)')
    parser.add_argument('--catalog', default=None,
                        help='Path to JSON catalog file with known parameter sets')
    parser.add_argument('--json', action='store_true',
                        help='Output as JSON')
    args = parser.parse_args()

    custom = compute_custom_params(args.depth, args.weight_type, args.security_target)

    recs = []
    if args.catalog:
        with open(args.catalog, 'r') as f:
            catalog = json.load(f)
        recs = recommend_from_catalog(args.depth, args.weight_type, args.priority,
                                      catalog, args.security_target)

    if args.json:
        output = {
            'recommendations': recs,
            'custom_estimate': custom,
        }
        print(json.dumps(output, indent=2))
        return

    print('=' * 72)
    print('CKKS Parameter Recommendation')
    print('=' * 72)
    print(f'  Depth:           {args.depth}')
    print(f'  Weight type:     {args.weight_type}')
    print(f'  Priority:        {args.priority}')
    print(f'  Security target: >={args.security_target} bits')
    print()
    if recs:
        print('--- Recommended Parameter Sets ---')
        for i, rec in enumerate(recs):
            marker = ' <-- BEST' if i == 0 else ''
            btp_note = ' (bootstrapping)' if rec['needs_btp'] else ''
            speed_str = f'{rec["relative_speed"]:.1f}x' if rec['relative_speed'] else 'N/A'
            print(f'\n  [{i + 1}] {rec["name"]}{marker}')
            print(f'      N={rec["N"]}, max_level={rec["max_level"]}, log_scale={rec["log_scale"]}')
            print(f'      Depth margin: {rec["depth_margin"]}{btp_note}')
            print(f'      Relative speed: {speed_str} (1.0 = fastest)')
            if rec['description']:
                print(f'      {rec["description"]}')
    else:
        if args.catalog:
            print('  No suitable parameter set found in catalog.')
        print('  Use the custom estimate below to build a new parameter set.')

    print()
    print('--- Custom Parameter Estimate ---')
    print(f'  Minimum log2(QP): {custom["log_qp_estimate"]:.0f} bits')
    min_N_str = str(custom['min_N']) if custom['min_N'] else 'N/A (exceeds known bounds)'
    print(f'  Minimum N for {args.security_target}-bit security: {min_N_str}')
    if custom['feasible']:
        print(f'  Recommended log_scale: {custom["log_scale"]}')
    print()

    # Low-precision suggestion for float32
    if args.weight_type == 'float32' and args.priority == 'efficiency':
        print('--- Low-Precision Optimization Suggestion ---')
        print('  Training with FP8/INT8 weights can enable smaller parameters:')
        lp_custom = compute_custom_params(args.depth, 'fp8', args.security_target)
        print(f'    FP8 log2(QP): {lp_custom["log_qp_estimate"]:.0f} bits '
              f'(vs {custom["log_qp_estimate"]:.0f} for float32)')
        if lp_custom['min_N'] and custom['min_N']:
            if lp_custom['min_N'] < custom['min_N']:
                print(f'    FP8 min N: {lp_custom["min_N"]} (vs {custom["min_N"]} for float32) -- '
                      f'{custom["min_N"] / lp_custom["min_N"]:.1f}x smaller ring!')
        print()


if __name__ == '__main__':
    main()