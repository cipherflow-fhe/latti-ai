#!/usr/bin/env python3
"""Estimate CKKS security level for given parameter sets using lattice-estimator.

Requires SageMath and lattice-estimator for exact estimation:
  conda activate sage
  git clone https://github.com/malb/lattice-estimator.git

Usage:
  python estimate_security.py --q 16957441 2752513 ... --p 17006593 --N 8192
  python estimate_security.py --catalog params.json --preset MyParamSet
  python estimate_security.py --catalog params.json --preset all
"""

import argparse
import json
import math
import sys


def log2_qp(q_primes, p_primes):
    """Compute log2(Q*P) from explicit prime lists."""
    total = 1
    for qi in q_primes:
        total *= qi
    for pi in p_primes:
        total *= pi
    return math.log2(total)


def estimate_one(name, params, use_estimator=False):
    """Estimate security for a single parameter set.

    Args:
        name: Parameter set name.
        params: Dict with 'N', 'q' (list of Q primes), 'p' (list of P primes),
                and optional 'log_scale', 'levels'.
        use_estimator: If True, run lattice-estimator (requires sage).

    Returns:
        Dict with log_qp, security assessment, and optional attack results.
    """
    log_qp = log2_qp(params['q'], params['p'])
    N = params['N']

    result = {
        'name': name,
        'N': N,
        'logN': int(math.log2(N)),
        'slots': N // 2,
        'levels': params.get('levels', len(params['q']) - 1),
        'log_scale': params.get('log_scale', 0),
        'log_qp': log_qp,
        'num_q_primes': len(params['q']),
        'num_p_primes': len(params['p']),
    }

    if use_estimator:
        try:
            sys.path.insert(0, '')
            from estimator import LWE, ND

            Xs = ND.DiscreteGaussian(3.19, n=1)
            Xe = ND.DiscreteGaussian(3.19)

            qp = 1
            for qi in params['q']:
                qp *= qi
            for pi in params['p']:
                qp *= pi
                lwe_params = LWE.Parameters(n=N, q=qp, Xs=Xs, Xe=Xe, tag=name)

            attacks = {
                'usvp': LWE.primal_usvp,
                'bdd': LWE.primal_bdd,
                'dual': LWE.dual,
                'dual_hybrid': LWE.dual_hybrid,
            }

            min_bits = float('inf')
            min_attack = ''
            attack_results = {}

            for attack_name, attack_fn in attacks.items():
                try:
                    res = attack_fn(lwe_params)
                    rop = res.get('rop', None)
                    if rop is not None:
                        bits = math.log2(float(rop))
                        attack_results[attack_name] = bits
                        if bits < min_bits:
                            min_bits = bits
                            min_attack = attack_name
                except Exception as e:
                    attack_results[attack_name] = f'FAILED: {e}'

            result['attacks'] = attack_results
            result['min_security'] = min_bits if min_bits != float('inf') else None
            result['min_attack'] = min_attack
        except ImportError:
            print('Warning: lattice-estimator not available. Using HE Standard approximation.')
            result['min_security'] = None
            result['min_attack'] = 'estimator_unavailable'

    # HE Standard approximate check (always computed)
    # Per-N bounds from the HE Standard tables for common N values.
    # These are conservative upper bounds; actual security may be higher.
    SECURITY_BOUNDS_128 = {8192: 218, 16384: 438, 32768: 881, 65536: 1761}
    SECURITY_BOUNDS_192 = {8192: 152, 16384: 305, 32768: 611, 65536: 1227}

    if N in SECURITY_BOUNDS_192 and log_qp <= SECURITY_BOUNDS_192[N]:
        result['approx_security'] = '>=192 bits'
    elif N in SECURITY_BOUNDS_128 and log_qp <= SECURITY_BOUNDS_128[N]:
        result['approx_security'] = '>=128 bits'
    elif log_qp / N <= 0.0464:
        result['approx_security'] = '>=80 bits'
    else:
        result['approx_security'] = '<80 bits (INSECURE)'

    return result


def load_catalog(catalog_path):
    """Load parameter sets from a JSON catalog file.

    Expected format (see references/parameter_catalog_template.json):
    {
        "parameter_sets": {
            "SetName": {
                "N": 8192,
                "levels": 9,
                "log_scale": 21,
                "q": [0x1, 0x2, ...],
                "p": [0x1, ...],
                "description": "Optional description"
            }
        },
        "security_bounds": {
            "128": {"8192": 218, ...},
            "192": {"8192": 152, ...}
        }
    }
    """
    with open(catalog_path, 'r') as f:
        data = json.load(f)
    return data
def main():
    parser = argparse.ArgumentParser(description='Estimate CKKS security level')
    parser.add_argument('--catalog', help='Path to JSON catalog file with parameter sets')
    parser.add_argument('--preset', nargs='*', default=None,
                        help='Preset name(s) from catalog, or "all"')
    parser.add_argument('--q', type=int, nargs='+', help='Q primes (explicit)')
    parser.add_argument('--p', type=int, nargs='+', help='P primes (explicit)')
    parser.add_argument('--N', type=int, help='Polynomial modulus degree')
    parser.add_argument('--log-scale', type=int, default=0, help='Log scale bits')
    parser.add_argument('--levels', type=int, default=0, help='Multiplicative levels')
    parser.add_argument('--use-estimator', action='store_true',
                        help='Run lattice-estimator (requires sage + lattice-estimator)')
    parser.add_argument('--estimator-path', default=None,
                        help='Path to lattice-estimator directory')
    args = parser.parse_args()

    if args.use_estimator and args.estimator_path:
        sys.path.insert(0, args.estimator_path)

    param_sets = []

    if args.q and args.p and args.N:
        param_sets.append({
            'name': f'Custom(N={args.N})',
            'N': args.N,
            'log_scale': args.log_scale or 0,
            'levels': args.levels or 0,
            'q': args.q,
            'p': args.p,
        })
    elif args.catalog:
        catalog = load_catalog(args.catalog)
        sets = catalog.get('parameter_sets', {})
        if not sets:
            print(f'Error: No parameter_sets found in {args.catalog}')
            sys.exit(1)

        presets = args.preset or list(sets.keys())
        if 'all' in presets:
            presets = list(sets.keys())
        for name in presets:
            if name in sets:
                param_sets.append({'name': name, **sets[name]})
            else:
                print(f'Unknown preset: {name}. Available: {", ".join(sets.keys())}')
                sys.exit(1)
    else:
        print('Provide --catalog <path> [--preset <name>] or --q ... --p ... --N <int>')
        parser.print_help()
        sys.exit(1)

    print('=' * 72)
    print('CKKS Security Estimation')
    print('=' * 72)

    results = []
    for ps in param_sets:
        name = ps['name']
        params = {k: v for k, v in ps.items() if k != 'name'}
        result = estimate_one(name, params, use_estimator=args.use_estimator)
        results.append(result)

    for r in results:
        print(f'\n--- {r["name"]} ---')
        print(f'  N = {r["N"]},  logN = {r["logN"]},  slots = {r["slots"]}')
        print(f'  levels = {r["levels"]},  log(scale) = {r["log_scale"]}')
        print(f'  log2(QP) = {r["log_qp"]:.1f}')
        print(f'  |Q| = {r["num_q_primes"]} primes,  |P| = {r["num_p_primes"]} primes')
        print(f'  Approx: {r["approx_security"]}')

        if 'attacks' in r:
            print()
            for attack_name, bits in r['attacks'].items():
                if isinstance(bits, float):
                    print(f'    {attack_name:15s} -> {bits:.1f} bits')
                else:
                    print(f'    {attack_name:15s} -> {bits}')

            if r.get('min_security') is not None:
                print(f'\n  >>> Minimum security: {r["min_security"]:.1f} bits (attack: {r["min_attack"]})')

        print()
        # Summary table
    if len(results) > 1:
        print('=' * 72)
        print('Summary')
        print('=' * 72)
        print(f'{"Name":25s} {"N":>6s} {"log2(QP)":>10s} {"Scale":>6s} {"Lvl":>4s} {"Security":>12s}')
        print('-' * 72)
        for r in results:
            sec = f'{r["min_security"]:.1f}b' if r.get('min_security') else r['approx_security']
            print(f'{r["name"]:25s} {r["N"]:>6d} {r["log_qp"]:>10.1f} {r["log_scale"]:>6d} {r["levels"]:>4d} {sec:>12s}')
        print()


if __name__ == '__main__':
    main()