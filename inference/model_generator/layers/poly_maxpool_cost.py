"""Primitive-operation counts for fixed-profile interleaved stages.

Counts follow Poly2D's Stockmeyer implementation, including zero coefficients
(which the eager evaluator still executes). Timing comes from framework tables;
relinearization is included in the framework's ciphertext-multiply timing.
"""
import math
from functools import lru_cache

from training.model_compiler.maxpool_plan import build_maxpool_plan, parse_maxpool_split_stage
from inference.model_generator.layers.poly_maxpool_taps import build_interleaved_maxpool_tap_masks
from inference.model_generator.layers.poly_base import PolyBase
from training.nn_tools.fixed_maxpool_profiles import get_maxpool_profile


def get_maxpool_compare_cost(alpha=7):
    return sum(PolyBase.compute_stockmeyer_level_cost(d)
               for d in get_maxpool_profile(alpha)['degrees']) + 1


@lru_cache(maxsize=32)
def _tap_counts(shape, slots):
    plan = build_maxpool_plan(
        {'kernel_shape': [3, 3], 'stride': [2, 2], 'padding': [1, 1]},
        [2 * v for v in shape])
    specs = build_interleaved_maxpool_tap_masks(plan, shape, 1, slots, 1.)
    return (sum(len(t) for t in specs),
            sum(bool(s['rotation_step']) for t in specs for s in t),
            sum(len(t) - 1 for t in specs))


def get_maxpool_stage_op_count(stage, shape, channels, slots, level, alpha=7, polynomial_index=None):
    """Return counts grouped by the level at which each operation executes."""
    shape = tuple(shape)
    plane = math.prod(shape)
    if channels <= 0 or len(shape) != 2 or plane <= 0 or slots % plane:
        raise ValueError('Invalid MaxPool cost layout')
    costs = {}

    def record(lv, count=1, **ops):
        if lv < 0:
            raise ValueError('Insufficient level for MaxPool stage')
        bucket = costs.setdefault(lv, dict.fromkeys(('rotate', 'mult_plain', 'mult', 'add', 'rescale'), 0))
        for name, value in ops.items():
            bucket[name] += count * value

    degrees = get_maxpool_profile(alpha)['degrees']

    def poly(start, count, degree):
        # Mirror Poly2D's eager baby/giant execution, including zero terms
        # and the square powers which are constructed even if unused.
        groups = (degree + 4) // 4
        span = 1 << (groups - 1).bit_length()
        power = 2
        while power <= max(2, 2 * span):
            lv = start - (power.bit_length() - 2)
            record(lv, count, mult=1, rescale=1)
            power *= 2
        targets = [None] * groups

        def assign(first, width, size, target):
            if width == 1:
                targets[first] = target
                return
            half = width // 2
            assign(first, half, min(size, half), target)
            if size > half:
                assign(first + half, half, size - half, target + 1)

        assign(0, span, groups, start - PolyBase.compute_stockmeyer_level_cost(degree))
        nodes = []
        for group, target in enumerate(targets):
            base = 4 * group
            terms = 0
            for offset in (1, 2):
                if base + offset <= degree:
                    record(target + 1, count, mult_plain=1, rescale=1)
                    if terms:
                        record(target, count, add=1)
                    terms += 1
            if base + 3 <= degree:
                record(target + 2, count, mult_plain=1, rescale=1)
                record(target + 1, count, mult=1, rescale=1)
                if terms:
                    record(target, count, add=1)
                terms += 1
            if terms:
                record(target, count, add=1)  # constant term
            nodes.append((bool(terms), target))
        while len(nodes) > 1:
            next_nodes = []
            for i in range(0, len(nodes), 2):
                if i + 1 == len(nodes):
                    next_nodes.append(nodes[i])
                    continue
                _, target = nodes[i]
                record(target + 1, count, rescale=1,
                       **{'mult' if nodes[i + 1][0] else 'mult_plain': 1})
                record(target, count, add=1)
                next_nodes.append((True, target))
            nodes = next_nodes

    if stage == 'primitive_poly':
        poly(level, channels, degrees[polynomial_index])
    elif stage == 'normalize_taps':
        if level < 1:
            raise ValueError('Insufficient level for MaxPool normalization')
        branches, rotations, additions = _tap_counts(shape, slots)
        record(level, channels, mult_plain=branches, rescale=branches, rotate=rotations)
        record(level - 1, channels, add=additions)
    elif stage.startswith('reduce_round_'):
        split = parse_maxpool_split_stage(stage)
        r = split[0] if split else int(stage.removeprefix('reduce_round_')) - 1
        plan = build_maxpool_plan(
            {'kernel_shape': [3, 3], 'stride': [2, 2], 'padding': [1, 1]},
            [2 * v for v in shape])
        if split:
            phase = split[1]
            if phase != 'finish' and phase >= len(degrees):
                raise ValueError('Invalid MaxPool polynomial stage')
            compare_cost = 1 if phase == 'finish' else PolyBase.compute_stockmeyer_level_cost(degrees[phase])
        else:
            compare_cost = get_maxpool_compare_cost(alpha)
        if not 0 <= r < len(plan['rounds']) or level < compare_cost:
            raise ValueError('Invalid MaxPool round or insufficient level')
        comparisons = len(plan['rounds'][r]['pairs']) * channels
        if split:
            if phase == 'finish':
                record(level, comparisons, mult=1, rescale=1)
                record(level - 1, comparisons, add=1)
            else:
                poly(level, comparisons, degrees[phase])
                if phase == 0:
                    record(level, add=2 * comparisons + len(plan['rounds'][r]['carry']) * channels)
            return costs
        current = level
        for degree in degrees:
            poly(current, comparisons, degree)
            current -= PolyBase.compute_stockmeyer_level_cost(degree)
        record(level, comparisons, add=2)  # difference and sum
        record(current, comparisons, mult=1, rescale=1)
        record(current - 1, add=comparisons + len(plan['rounds'][r]['carry']) * channels)
    elif stage in {'repack', 'restore'}:
        if level < 1:
            raise ValueError('Insufficient level for MaxPool packing/restoration')
        outputs = math.ceil(channels / (slots // plane))
        count = channels if stage == 'repack' else outputs
        record(level, count, mult_plain=1, rescale=1)
        if stage == 'repack':
            record(level - 1, channels - outputs, rotate=1, add=1)
    else:
        raise ValueError(f'Unknown MaxPool cost stage: {stage}')
    return costs


def get_maxpool_primitive_op_count(primitive, shape, channels, slots, level, alpha=7):
    op = primitive['op']
    if op == 'poly':
        return get_maxpool_stage_op_count('primitive_poly', shape, channels, slots, level,
                                         alpha, primitive['polynomial'])
    plane = math.prod(shape)
    outputs = math.ceil(channels / (slots // plane))
    counts = dict.fromkeys(('rotate', 'mult_plain', 'mult', 'add', 'rescale'), 0)
    if op == 'rotate':
        steps = primitive['steps']
        counts['rotate'] = sum(bool(steps[i % len(steps)]) for i in range(channels))
    elif op in ('add', 'sub', 'double'):
        counts['add'] = channels
    elif op == 'pack':
        counts['add'] = channels - outputs
    elif op == 'mult':
        counts.update(mult=channels, rescale=channels)
    elif op in ('mask', 'select', 'restore'):
        count = outputs if op == 'restore' else channels
        counts.update(mult_plain=count, rescale=count)
    elif op == 'source':
        counts['add'] = 6 * channels  # explicit zero dependencies for other blocks
    else:
        raise ValueError(f'Unknown MaxPool primitive cost: {op}')
    return {level: counts}
