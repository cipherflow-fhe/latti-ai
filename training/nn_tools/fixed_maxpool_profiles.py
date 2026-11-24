"""Versioned, numerically validated plaintext profiles; no online generation.

This lookup does not enable a profile in the CKKS runtime. Runtime evaluation
plans and level costs must be validated separately before integration.
"""

from copy import deepcopy
from decimal import Decimal
from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=1)
def _profiles():
    path = Path(__file__).with_name('paper_maxpool_profiles.json')
    data = json.loads(path.read_text(encoding='utf-8'))
    if data['schema_version'] != 1 or data['coefficient_order'] != 'ascending_odd_powers':
        raise ValueError('unsupported fixed MaxPool profile format')
    profiles = {}
    for entry in data['profiles']:
        alpha = entry['alpha']
        if alpha in profiles:
            raise ValueError('duplicate fixed MaxPool alpha')
        degrees, coefficients = entry['degrees'], entry['odd_coefficients']
        if len(degrees) != len(coefficients) or not degrees:
            raise ValueError('invalid fixed MaxPool stage count')
        for degree, stage in zip(degrees, coefficients):
            if degree % 2 != 1 or len(stage) != (degree+1)//2:
                raise ValueError('invalid fixed MaxPool coefficient layout')
            if not all(Decimal(c).is_finite() for c in stage) or Decimal(stage[-1]) == 0:
                raise ValueError('invalid fixed MaxPool coefficient value')
        error = Decimal(entry['measured_max_error'])
        if (entry['numerical_sign_bound_pass'] is not True
                or not error.is_finite() or error < 0
                or error > Decimal(2)**(-(alpha-1))):
            raise ValueError('fixed MaxPool profile has not passed numerical validation')
        profiles[alpha] = dict(entry, config_version=data['config_version'],
                               validation_status=data['validation_status'])
    return profiles


def available_maxpool_alphas():
    """Return only configurations with saved coefficients and validation."""
    return tuple(sorted(_profiles()))


def get_maxpool_profile(alpha):
    """Return a defensive copy; decimal coefficients remain lossless strings.

    Alpha 4..6 have table metadata but are deliberately unavailable here.
    Missing profiles raise an error rather than generating or substituting one.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, int):
        raise TypeError('alpha must be an integer')
    if alpha not in _profiles():
        raise ValueError(f'no validated fixed MaxPool profile for alpha={alpha}; '
                         f'available: {available_maxpool_alphas()}')
    return deepcopy(_profiles()[alpha])
