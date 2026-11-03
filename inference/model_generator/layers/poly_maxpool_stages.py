"""Executable Python stage dispatch for the Retina interleaved MaxPool.

Plaintext binding is supplied by the caller. Bootstrap remains an ordinary
framework operation between calls, never an implicit operation in this module.
"""
import math

from inference.lattisense.frontend.custom_task import add, sub, mult, rescale, relin, drop_level, rotate_cols
from training.model_compiler.maxpool_plan import parse_maxpool_split_stage
from inference.model_generator.layers.poly_maxpool_taps import (
    call_interleaved_normalized_maxpool_taps,
    build_interleaved_maxpool_tap_masks,
)
from inference.model_generator.layers.poly2d import Poly2D
from training.nn_tools.fixed_maxpool_profiles import get_maxpool_profile
from inference.model_generator.layers.poly_maxpool_generic import (
    flatten_maxpool_state, call_maxpool_round_stage,
)


def _maxpool_stage_coefficients(alpha, round_index, comparison_B):
    """Return fixed profile coefficients with only the first stage scaled.

    The profile coefficients are immutable sign approximations. The input
    difference scale belongs to the first polynomial stage; later composition
    stages consume its output and are left unchanged. This helper is kept
    separate from CKKS execution so alpha=7 can be regression-tested first.
    """
    if not math.isfinite(comparison_B) or comparison_B <= 0:
        raise ValueError('MaxPool comparison B must be positive')
    profile = get_maxpool_profile(alpha)
    stages = [list(map(float, values)) for values in profile['odd_coefficients']]
    if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
        raise ValueError('Invalid MaxPool round index')
    bound = comparison_B * (1 if round_index == 0 else 1.05) * (2 ** round_index)
    stages[0] = [value / (2 * bound) ** (2 * i + 1)
                 for i, value in enumerate(stages[0])]
    return {
        'alpha': alpha,
        'degrees': list(profile['degrees']),
        'stages': stages,
        'comparison_B': bound,
        'source': profile['config_version'],
    }


def bind_maxpool_stage_plaintexts(stage, plan, channels, slots, level, running_max, make_pt,
                                  alpha=7, comparison_B=1.0, upper_bound=1.0):
    """Declare plaintext arguments and explicit encoding descriptors.

    make_pt(suffix, count, descriptor) returns bound plaintext nodes. The
    descriptor contains the encoding level and scale, never ciphertext data.
    """
    if not math.isfinite(running_max) or running_max <= 0:
        raise ValueError('MaxPool requires positive calibrated running_max')
    if not math.isfinite(upper_bound) or upper_bound <= 0:
        raise ValueError('MaxPool upper_bound must be positive and finite')
    # Validate the configuration on every stage, including normalization.
    get_maxpool_profile(alpha)
    if not math.isfinite(comparison_B) or comparison_B <= 0:
        raise ValueError('MaxPool comparison B must be positive')
    normalization = upper_bound / running_max
    plane = math.prod(plan['output_shape'])
    if stage == 'normalize_taps':
        specs = build_interleaved_maxpool_tap_masks(
            plan, plan['output_shape'], channels, slots, normalization,
        )
        masks = []
        for t, branches in enumerate(specs):
            columns = []
            for b, spec in enumerate(branches):
                # Store each shared mask only once, independent of channels.
                mask = spec['mask']
                indices = mask.nonzero()[0].tolist()
                columns.append(make_pt(f'tap_{t}_branch_{b}', channels, {
                    'level': level, 'kind': 'mask', 'value': normalization,
                    'indices': indices,
                }))
            masks.append([[column[c] for column in columns] for c in range(channels)])
        return {'tap_specs': specs, 'masks': masks}
    if stage.startswith('reduce_round_'):
        split = parse_maxpool_split_stage(stage)
        r = split[0] if split else int(stage[len('reduce_round_'):]) - 1
        if not 0 <= r < len(plan['rounds']):
            raise ValueError('Invalid reduction stage')
        profile = _maxpool_stage_coefficients(alpha, r, comparison_B)
        if split and split[1] == 'finish':
            if level < 1:
                raise ValueError('MaxPool finish requires one level')
            # This stage uses only ciphertexts; declare no unused plaintext.
            return {'level_cost': 1}
        if split:
            index = split[1]
            if index >= len(profile['degrees']):
                raise ValueError('Invalid MaxPool polynomial index')
            profile['degrees'] = [profile['degrees'][index]]
            profile['stages'] = [profile['stages'][index]]
        costs = [Poly2D.compute_stockmeyer_level_cost(d) for d in profile['degrees']]
        # Include the final difference * sign ciphertext multiplication.
        cost = sum(costs) + (0 if split else 1)
        if level < cost:
            raise ValueError(f'MaxPool alpha={alpha} round requires {cost} levels; got {level}')
        result = {'sign_stages': [], 'level_cost': cost}
        start = level
        for index, (degree, values, stage_cost) in enumerate(zip(
                profile['degrees'], profile['stages'], costs)):
            name = f'p{index}'
            coeffs = [value for v in values for value in (0., v)]
            poly = Poly2D(plan['output_shape'], degree, [1, 1], 1)
            poly.init_stockmeyer(start)
            result[name] = [make_pt(f'{name}_degree_{i}', channels, {
                'kind': 'constant', 'value': value,
                'level': poly.cached_stockmeyer_level_order[i],
                'scale': poly.cached_stockmeyer_coeff_scale[i],
            }) for i, value in enumerate(coeffs)]
            result['sign_stages'].append({'degree': degree, 'weights': result[name]})
            start -= stage_cost
        return result
    if stage == 'repack':
        return {'select': make_pt('select', channels, {
            'kind': 'prefix', 'length': plane, 'value': 1., 'level': level,
        })}
    if stage == 'restore':
        return {'restore': make_pt('restore', math.ceil(channels / (slots // plane)), {
            'kind': 'constant', 'value': running_max / upper_bound / 2**plan['compare_depth'], 'level': level,
        })}
    raise ValueError(f'Unknown MaxPool stage: {stage}')


def call_maxpool_stage(stage, nodes, plan, channels, slots, plaintexts):
    """Execute one stage using caller-bound plaintext nodes.

    normalize: {'tap_specs': ..., 'masks': [tap][channel][branch]}
    round: {'p0': [degree][channel], 'p1': [degree][channel]}
    repack: {'select': [channel]} (ones on the dense spatial plane only)
    restore: {'restore': [output ciphertext]} (running_max / 16)
    """
    shape = plan['output_shape']
    plane = math.prod(shape)
    if not nodes or len({ct.level for ct in nodes}) != 1:
        raise ValueError('MaxPool stage requires nonempty aligned inputs')
    start = nodes[0].level
    if plane <= 0 or slots < plane or slots % plane:
        raise ValueError('Invalid MaxPool output plane or slot capacity')
    if stage == 'normalize_taps':
        result = flatten_maxpool_state(call_interleaved_normalized_maxpool_taps(
            nodes, plaintexts['tap_specs'], plaintexts['masks'], channels,
        ), channels)
        cost = 1
    elif stage.startswith('reduce_round_'):
        split = parse_maxpool_split_stage(stage)
        if split:
            result = _call_split_round(nodes, plan, split, plaintexts, channels)
        else:
            index = int(stage[len('reduce_round_'):]) - 1
            result = call_maxpool_round_stage(nodes, plan, index, plaintexts, channels)
        cost = plaintexts.get('level_cost', 7)
    elif stage == 'repack':
        if plane <= 0 or slots % plane or len(nodes) != channels:
            raise ValueError('Invalid dense MaxPool output packing')
        if len(plaintexts['select']) != channels:
            raise ValueError('Missing MaxPool channel selection masks')
        per_ct = slots // plane
        result = []
        for first in range(0, channels, per_ct):
            value = None
            for channel in range(first, min(first + per_ct, channels)):
                branch = rescale(mult(nodes[channel], plaintexts['select'][channel]))
                shift = -(channel - first) * plane
                if shift:
                    branch = rotate_cols(branch, [shift])[0]
                value = branch if value is None else add(value, branch)
            result.append(value)
        cost = 1
    elif stage == 'restore':
        expected = math.ceil(channels / (slots // plane))
        if len(nodes) != expected or len(plaintexts['restore']) != expected:
            raise ValueError('Incorrect MaxPool restore ciphertext count')
        result = [rescale(mult(ct, pt)) for ct, pt in zip(nodes, plaintexts['restore'])]
        cost = 1
    else:
        raise ValueError(f'Unknown MaxPool stage: {stage}')
    if any(ct.level != start - cost for ct in result):
        raise ValueError(f'Incorrect output level for MaxPool stage {stage}')
    return result


def _call_split_round(nodes, plan, split, plaintexts, channels):
    """State order: [sum,difference,sign] per pair, then doubled carries.

    Every component is level-aligned and survives ordinary external Bootstrap.
    No refresh is selected or executed here.
    """
    from inference.model_generator.layers.poly_maxpool2d_layer import _evaluate_sign_chain
    r, phase = split
    if not 0 <= r < len(plan['rounds']):
        raise ValueError('Invalid MaxPool round')
    round_plan = plan['rounds'][r]
    pairs, carries = round_plan['pairs'], round_plan['carry']
    groups = [nodes[i:i + channels] for i in range(0, len(nodes), channels)]
    expected = (2 * len(pairs) + len(carries)) if phase == 0 else (3 * len(pairs) + len(carries))
    if len(nodes) != expected * channels:
        raise ValueError('Incomplete MaxPool split live state')
    output_level = nodes[0].level - plaintexts['level_cost']

    def align(group):
        return [drop_level(ct, ct.level - output_level) if ct.level != output_level else ct
                for ct in group]

    result = []
    for pair_index, (i, j) in enumerate(pairs):
        if phase == 0:
            summation = [add(a, b) for a, b in zip(groups[i], groups[j])]
            difference = [sub(a, b) for a, b in zip(groups[i], groups[j])]
            sign = difference
        else:
            summation, difference, sign = groups[3 * pair_index:3 * pair_index + 3]
        if phase == 'finish':
            product = [rescale(relin(mult(a, b))) for a, b in zip(difference, sign)]
            result.append([add(a, b) for a, b in zip(align(summation), product)])
        else:
            sign = _evaluate_sign_chain(sign, plaintexts['sign_stages'],
                                        plan['output_shape'], [1, 1], 1)
            result.extend([align(summation), align(difference), sign])
    if phase == 0:
        carry_groups = [[add(ct, ct) for ct in groups[index]] for index in carries]
    else:
        carry_groups = groups[3 * len(pairs):]
    result.extend(align(group) for group in carry_groups)
    return flatten_maxpool_state(result, channels)
