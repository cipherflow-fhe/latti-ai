"""Individual MaxPool graph operations, with no implicit level/refresh policy."""
import math
from functools import lru_cache

from frontend.custom_task import add, sub, mult, rescale, relin, rotate_cols
from inference.model_generator.layers.poly_maxpool_stages import bind_maxpool_stage_plaintexts
from inference.model_generator.layers.poly_maxpool_taps import build_interleaved_maxpool_tap_masks
from inference.model_generator.layers.poly_maxpool2d_layer import _evaluate_sign_chain
from training.model_compiler.maxpool_plan import build_maxpool_plan
from training.nn_tools.fixed_maxpool_profiles import get_maxpool_profile


@lru_cache(maxsize=32)
def _tap_masks(shape, slots, normalization):
    plan = build_maxpool_plan({'kernel_shape': [3, 3], 'stride': [2, 2],
                              'padding': [1, 1]}, [2 * v for v in shape])
    return build_interleaved_maxpool_tap_masks(plan, shape, 1, slots, normalization)


def maxpool_primitive_level_cost(primitive, alpha):
    from inference.model_generator.layers.poly_base import PolyBase
    op = primitive['op']
    if op == 'poly':
        degree = get_maxpool_profile(alpha)['degrees'][primitive['polynomial']]
        return PolyBase.compute_stockmeyer_level_cost(degree)
    if op in ('mask', 'select', 'restore', 'mult'):
        return 1
    if op in ('source', 'rotate', 'add', 'sub', 'double', 'pack'):
        return 0
    raise ValueError(f'Unknown MaxPool primitive: {op}')


def bind_maxpool_primitive(layer, plan, channels, slots, level, make_pt):
    primitive = layer['maxpool_primitive']
    op = primitive['op']
    get_maxpool_profile(layer.get('alpha', 7))
    for name in ('running_max', 'upper_bound', 'B'):
        value = float(layer.get(name, 1.))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'MaxPool {name} must be positive and finite')
    if op == 'mask':
        normalization = float(layer['upper_bound']) / float(layer['running_max'])
        specs = _tap_masks(tuple(plan['output_shape']), slots, normalization)
        spec = specs[primitive['tap']][primitive['branch']]
        return {'weights': make_pt('mask', channels, {
            'kind': 'mask', 'indices': spec['mask'].nonzero()[0].tolist(),
            'value': normalization, 'level': level})}
    stage = {'select': 'repack', 'restore': 'restore'}.get(op)
    if op == 'poly':
        stage = f"reduce_round_{primitive['round'] + 1}_poly_{primitive['polynomial']}"
    if stage is None:
        return {}
    return bind_maxpool_stage_plaintexts(
        stage, plan, channels, slots, level, float(layer['running_max']), make_pt,
        alpha=layer.get('alpha', 7), comparison_B=layer.get('B', 1.),
        upper_bound=layer.get('upper_bound', 1.))


def call_maxpool_primitive(primitive, inputs, plan, channels, slots, plaintexts):
    """Inputs are ordered compiler features; each feature is a CT vector."""
    op = primitive['op']
    nodes = inputs[0]
    if op == 'source':
        if len(nodes) != channels * 4:
            raise ValueError('MaxPool expects four interleaved blocks per channel')
        selected = nodes[primitive['source_block']::4]
        # A compiler feature represents the complete live input set.  Keep
        # the non-selected blocks as explicit zero dependencies so the
        # frontend does not reject them as unused data nodes.
        for block in range(4):
            if block == primitive['source_block']:
                continue
            for index, (chosen, unused) in enumerate(zip(selected, nodes[block::4])):
                selected_value = sub(unused, unused)
                selected[index] = add(chosen, selected_value)
        return selected
    if op == 'rotate':
        steps = primitive['steps']
        if len(steps) not in (1, len(nodes)):
            raise ValueError('MaxPool rotation count differs from ciphertext count')
        return [rotate_cols(ct, [steps[i % len(steps)]])[0] if steps[i % len(steps)] else ct
                for i, ct in enumerate(nodes)]
    if op in ('add', 'sub', 'mult'):
        if len(inputs) != 2 or len(nodes) != len(inputs[1]):
            raise ValueError('MaxPool binary inputs must have equal sizes')
        if op == 'mult':
            return [rescale(relin(mult(a, b))) for a, b in zip(*inputs)]
        fn = add if op == 'add' else sub
        return [fn(a, b) for a, b in zip(*inputs)]
    if op == 'double':
        return [add(ct, ct) for ct in nodes]
    if op == 'poly':
        return _evaluate_sign_chain(nodes, plaintexts['sign_stages'], plan['output_shape'], [1, 1], 1)
    if op in ('mask', 'select', 'restore'):
        key = {'mask': 'weights', 'select': 'select', 'restore': 'restore'}[op]
        weights = plaintexts[key]
        if len(nodes) != len(weights):
            raise ValueError('MaxPool plaintext count differs from ciphertext count')
        return [rescale(mult(ct, pt)) for ct, pt in zip(nodes, weights)]
    if op == 'pack':
        per_ct = slots // math.prod(plan['output_shape'])
        result = []
        for first in range(0, len(nodes), per_ct):
            value = nodes[first]
            for ct in nodes[first + 1:first + per_ct]:
                value = add(value, ct)
            result.append(value)
        return result
    raise ValueError(f'Unknown MaxPool primitive: {op}')
