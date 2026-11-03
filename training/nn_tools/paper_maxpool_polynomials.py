"""Paper minimax composite polynomials for encrypted MaxPool.

The coefficients are in ascending odd-power order: ``[c1, c3, ...]``.
They approximate sign after composition; they are not ReLU Hermite
coefficients.  This module is intentionally independent from the CKKS
compiler and generator.
"""

from __future__ import annotations

import math
from typing import Iterable


PAPER_CONFIG = {
    4: {"zeta": 5, "degrees": [5], "depth": 4},
    5: {"zeta": 5, "degrees": [13], "depth": 5},
    6: {"zeta": 10, "degrees": [3, 7], "depth": 6},
    7: {"zeta": 11, "degrees": [7, 7], "depth": 7},
    8: {"zeta": 12, "degrees": [7, 15], "depth": 8},
    9: {"zeta": 13, "degrees": [15, 15], "depth": 9},
    10: {"zeta": 13, "degrees": [7, 7, 13], "depth": 11},
    11: {"zeta": 15, "degrees": [7, 7, 27], "depth": 12},
    12: {"zeta": 15, "degrees": [7, 15, 27], "depth": 13},
    13: {"zeta": 16, "degrees": [15, 15, 27], "depth": 14},
    14: {"zeta": 17, "degrees": [15, 27, 29], "depth": 15},
}

# Appendix A, alpha=7..14 (user-supplied transcription). Alpha=4..6 have no coefficients
# until their source (Appendix/author repository) is verified.
PAPER_ODD_COEFFS = {
    7: [
        [7.30445164958251, -34.6825871108659, 59.8596518298826, -31.8755225906466],
        [2.40085652217597, -2.63125454261783, 1.54912674773593, -0.331172956504304],
    ],
    8: [
        [8.83133072022416, -46.4575039895512, 83.0282234720408, -44.9928477828070],
        [3.94881885083263, -12.9103010992282, 28.0865362174658, -35.5969148965137, 26.5159370881337, -11.4184889368449, 2.62558443881334, -0.249172299998642],
    ],
    9: [
        [18.0966285718807, -434.038703274886, 4154.97103545696, -18684.6943613149, 44165.7177889329, -56552.7928983401, 37115.6122725781, -9782.41933892781],
        [3.79753323360856, -11.7718157771192, 24.9771086678346, -31.5238841603993, 23.7294863126722, -10.4331800195923, 2.46743976260838, -0.242130100247617],
    ],
    10: [
        [10.8541842577442, -62.2833925211098, 114.369227820443, -62.8023496973074],
        [4.13976170985111, -5.84997640211679, 2.94376255659280, -0.454530437460152],
        [3.29956739043733, -7.84227260291355, 12.8907764115564, -12.4917112584486, 6.94167991428074, -2.04298067399942, 0.246407138926031],
    ],
    11: [
        [11.2590667402954, -65.4692933329973, 120.694634277757, -66.4019695377825],
        [4.70477624210883, -6.79884851596681, 3.31525104382873, -0.489362936859897],
        [5.36334257654953, -35.5169555441962, 177.807304115644, -592.297395415024, 1348.91691889362, -2158.76445084938, 2473.65685586918, -2049.13542536248, 1227.39317090559, -525.826175134002, 156.930558712840, -30.9658595645912, 3.62894000814968, -0.191160283749939],
    ],
    12: [
        [11.5523042357223, -67.7794513440968, 125.283740404562, -69.0142908232934],
        [9.65167636181626, -61.6939174538469, 155.170351652298, -182.697582383214, 112.910726525406, -37.7752411770263, 6.47503909732344, -0.445613365723361],
        [5.25888355571745, -33.7233593794284, 164.983085013457, -541.408891406992, 1222.96207997963, -1952.01910566479, 2240.84021378300, -1866.34916983170, 1127.22117843121, -488.070474638380, 147.497846308920, -29.5171048879526, 3.51269520930994, -0.188101836557879],
    ],
    13: [
        [24.5589415425004, -669.660449716894, 6672.99848301339, -30603.6656163898, 73188.4032987787, -94443.3217050084, 62325.4094212546, -16494.6744117805],
        [9.35625636035439, -59.1638963933626, 148.860930626448, -175.812874878582, 109.111299685955, -36.6768839978755, 6.31846290311294, -0.437113415082177],
        [5.07813569758861, -30.7329918137186, 144.109746812809, -459.661688826142, 1021.52064470459, -1620.56256708877, 1864.67646416570, -1567.49300877143, 960.970309093422, -424.326161871646, 131.278509256003, -26.9812576626115, 3.30651387315565, -0.182742944627533],
    ],
    14: [
        [24.9052143193754, -682.383057582430, 6809.42845390599, -31250.7100017105, 74765.9388363757, -96504.6838475839, 63697.7923778246, -16860.2621347190],
        [16.8285511926011, -339.811750495659, 2790.69998793847, -11351.4151573790, 26623.0010283745, -39384.0328661975, 38788.4230348060, -26239.5303844988, 12365.6207016532, -4053.36460089999, 906.042880951087, -131.687649208288, 11.2176079033623, -0.424938020467471],
        [5.31755497689391, -35.4371531531577, 184.122441329140, -655.386830146253, 1638.78335428060, -2953.86237048226, 3908.06423362418, -3834.96739165131, 2799.60654766517, -1512.86231886692, 596.160139340009, -166.321739302958, 31.0988369739884, -3.49349374506190, 0.178142156956495],
    ],
}


def evaluate_odd_poly(x, odd_coeffs: Iterable[float]):
    """Evaluate ``c1*x + c3*x**3 + ...`` using Horner in ``x**2``."""
    coeffs = list(odd_coeffs)
    if not coeffs:
        raise ValueError("odd_coeffs must not be empty")
    result = coeffs[-1]
    for coeff in reversed(coeffs[:-1]):
        result = coeff + x * x * result
    return x * result


def paper_coefficients(alpha: int):
    """Return a defensive copy of supplied Appendix-A coefficient data."""
    if alpha not in PAPER_CONFIG:
        raise ValueError(f"unsupported alpha: {alpha}")
    if alpha not in PAPER_ODD_COEFFS:
        raise KeyError(f"coefficients are not loaded for alpha={alpha}")
    return [list(v) for v in PAPER_ODD_COEFFS[alpha]]


def evaluate_composite_sign(x, alpha: int):
    """Evaluate the paper composition in listed stage order."""
    values = x
    for coeffs in paper_coefficients(alpha):
        values = evaluate_odd_poly(values, coeffs)
    return values


def approximate_max(a, b, alpha: int):
    """Evaluate the paper two-input polynomial Max approximation."""
    difference = a - b
    return (a + b + difference * evaluate_composite_sign(difference, alpha)) / 2


def initial_gap(alpha: int) -> float:
    """Return epsilon = zeta * 2**(-alpha)."""
    cfg = PAPER_CONFIG[alpha]
    return cfg["zeta"] * 2.0 ** (-alpha)


def validate_metadata(alpha: int) -> None:
    cfg = PAPER_CONFIG[alpha]
    coeffs = paper_coefficients(alpha)
    if len(coeffs) != len(cfg["degrees"]):
        raise ValueError("coefficient stage count does not match degree table")
    for values, degree in zip(coeffs, cfg["degrees"]):
        if degree % 2 == 0 or len(values) != (degree + 1) // 2:
            raise ValueError(f"invalid odd coefficient count for degree {degree}")
        if not all(math.isfinite(v) for v in values):
            raise ValueError("non-finite coefficient")
