# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Numerical PCM generators matching the native RingT parameter encoders."""

import math

import h5py
import numpy as np


SUPPORTED = {
    ('InputZero', 'zero'),
    ('ParBlockColMajorCPMM', 'row_chunk_seed'),
    ('ParBlockColMajorCPMM', 'bias_pt'),
    ('ParBlockColMajorLNAffine', 'pt'),
    ('ParBlockColMajorLNStats', 'pt'),
    ('ParBlockColMajorLNMinimaxInit', 'pt'),
    ('ParBlockColMajorPolyActRNGamma', 'gamma_pt'),
    ('ParBlockColMajorPolyActRNPoly', 'coeff_pt'),
    ('ParBlockColMajorPoly', 'stockmeyer_coeff_pt'),
    ('ParBlockColMajorAddPt', 'add_pt'),
}


def _geometry(config, slots):
    d, heads = config['matmul_block_size'], config['n_heads']
    if type(d) is not int or d < 1 or d & (d - 1) or heads < 1 or slots < d * d:
        raise ValueError('Invalid PCM parameter geometry')
    padded = 1 << (heads - 1).bit_length()
    u = min(padded, slots // (d * d))
    chunk = u * d * d
    if u == 1:
        padded = heads
    if slots % chunk or padded % u:
        raise ValueError('Invalid PCM parameter chunk geometry')
    return d, heads, padded, u, chunk, padded // u, slots // chunk


def matrix_argument(signature, graph, config, parameters, weights):
    if config.get('mat_pack_style') != 'par_block_col_major':
        raise ValueError('Unsupported matrix parameter layout')
    layer = graph['layer'][signature['layer_id']]
    feature = graph['feature'][layer['feature_input'][0]]
    profile = parameters[feature['ckks_parameter_id']]
    generator = signature['generator']
    op, kind = generator['op_class'], generator['type']
    if (op, kind) not in SUPPORTED or signature.get('type') != 'ct':
        raise ValueError('Unsupported PCM parameter generator')
    size, level, scale = signature['size'], signature['level'], signature.get('scale')
    if (
        not size
        or any(type(n) is not int or n <= 0 for n in size)
        or type(level) is not int
        or not 0 <= level < len(profile['q'])
        or not isinstance(scale, (int, float))
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError('Invalid PCM parameter signature')
    slots = profile['n_slots']
    d, heads, padded, u, chunk, groups, rho = _geometry(config, slots)
    path = None
    if op == 'ParBlockColMajorCPMM':
        path = layer.get('weight_path') if kind == 'row_chunk_seed' else layer.get('bias_path')
    elif op == 'ParBlockColMajorLNAffine':
        path = layer.get('weight_path') if generator['pt_idx'] == 0 else layer.get('bias_path')
    elif op in (
        'ParBlockColMajorPolyActRNGamma',
        'ParBlockColMajorPolyActRNPoly',
        'ParBlockColMajorPoly',
        'ParBlockColMajorAddPt',
    ):
        path = layer.get('weight_path', layer.get('path', layer.get('coeffs_path')))
    if path is not None and (path not in weights or weights[path].size * 8 > 32 * 1024**2):
        raise ValueError('Missing or oversized PCM parameter H5 dataset: ' + path)
    if kind == 'row_chunk_seed':
        shape = layer['weight_shape']
        head_shape = feature['head_shape']
        if head_shape[1] != d or weights[path].size != math.prod(shape):
            raise ValueError('PCM seed weight shape differs from graph')
        n = heads * d
        row_blocks, col_blocks = math.ceil(shape[0] / n), math.ceil(shape[1] / n)
        if min(row_blocks, col_blocks) != 1:
            raise ValueError('Unsupported PCM seed matrix geometry')
        blocks = max(row_blocks, col_blocks)
        count = math.ceil(blocks * groups * heads / rho)
        expected = dict(
            kind='pcm_row_chunk_v1',
            weight_shape=shape,
            head_shape=head_shape,
            d=d,
            H=heads,
            Hp=padded,
            u=u,
            C=chunk,
            G=groups,
            rho=rho,
            K=blocks,
            request_count=blocks * groups * heads * d,
            requests_per_seed=d * rho,
            seed_count=count,
            use_site_level=signature['use_site_level'],
            seed_level=level,
            scale=scale,
            request_order_version=1,
        )
        if signature.get('layout') != expected or generator.get('pcm_seed_layout') != expected:
            raise ValueError('PCM seed layout differs from actual graph')
        seed = generator.get('seed_id')
        if type(seed) is not int or not 0 <= seed < count:
            raise ValueError('PCM parameter seed index out of range')
    return dict(
        signature,
        count=math.prod(size),
        slots=slots,
        param_kind='matrix',
        _parameter_path=path,
        use_site_level=signature.get('use_site_level', level),
    )


def matrix_values(item, graph, config, path):
    layer = graph['layer'][item['layer_id']]
    feature = graph['feature'][layer['feature_input'][0]]
    generator = item['generator']
    op, kind, slots = generator['op_class'], generator['type'], item['slots']
    d, heads, padded, u, chunk, groups, rho = _geometry(config, slots)
    positions = np.arange(slots)
    local = positions % chunk
    head = generator.get('g', 0) * u + local % u
    row = generator.get('bi', 0) * d + local // u % d
    col = generator.get('bj', 0) * d + local // u // d
    rows, columns = feature['shape']
    columns_per_head = feature.get('head_shape', [rows, columns // heads])[1]
    if op in ('ParBlockColMajorLNAffine', 'ParBlockColMajorAddPt'):
        columns_per_head = columns // heads
    global_col = generator.get('mb', 0) * heads * columns_per_head + head * columns_per_head + col
    valid = (head < heads) & (row < rows) & (col < columns_per_head)
    values = None
    if item['_parameter_path'] is not None:
        with h5py.File(path) as weights:
            values = np.asarray(weights[item['_parameter_path']], dtype=np.float64)
    result = np.zeros(slots)
    if op == 'InputZero':
        return result
    if op == 'ParBlockColMajorLNStats':
        params = config.get('layernorm_param', {})
        inv_var = layer.get('inv_var', params.get('inv_var', 1 / params.get('var_std_bound', 4.0) ** 2))
        index = generator['pt_idx']
        if index not in (2, 3):
            raise ValueError('Unsupported encrypted PCM statistics parameter')
        value = inv_var if index == 2 else layer.get('epsilon', params.get('eps', 1e-5)) * inv_var
        return np.full(slots, value)
    if op == 'ParBlockColMajorLNMinimaxInit':
        params = config.get('layernorm_param', {})
        coeffs = params.get('minimax_init_coeffs', [6.19067182, -16.15885111, 11.52830778])
        return np.full(slots, layer.get('c' + str(generator['pt_idx']), coeffs[generator['pt_idx']]))
    if kind == 'row_chunk_seed':
        values = values.reshape(layer['weight_shape'])
        layout = item['layout']
        request = generator['seed_id'] * (d * rho) + positions // chunk * d + local // u % d
        k = request % d
        bp = request // d % heads
        g = request // d // heads % groups
        mb = request // d // heads // groups
        j, h = local // u // d, local % u
        baby = math.ceil(math.sqrt(d))
        source_row = (g * u + h) * d + (j + k % baby) % d
        source_col = bp * d + (j + d - (k // baby) * baby % d) % d
        n = heads * d
        if math.ceil(values.shape[0] / n) > math.ceil(values.shape[1] / n):
            source_row = source_row + mb * n
        elif math.ceil(values.shape[1] / n) > math.ceil(values.shape[0] / n):
            source_col = source_col + mb * n
        selected = (
            (request < layout['request_count'])
            & (source_row < values.shape[0])
            & (source_col < values.shape[1])
            & ((g * u + h) * d + (j + k % baby) % d < n)
        )
        result[selected] = values[source_row[selected], source_col[selected]]
        return result
    if op == 'ParBlockColMajorCPMM':
        columns_per_head = feature['head_shape'][1]
        global_col = generator.get('mb', 0) * heads * columns_per_head + head * columns_per_head + col
        valid = (head < heads) & (row < rows) & (col < columns_per_head)
        if values is None:
            return result
    elif op == 'ParBlockColMajorPolyActRNGamma':
        if columns == 1:
            return np.full(slots, values.reshape(-1)[0])
    elif op in ('ParBlockColMajorPolyActRNPoly', 'ParBlockColMajorPoly'):
        degree = generator['coeff_idx']
        values = values.reshape(layer.get('order', 4) + 1, -1)[degree]
        if op == 'ParBlockColMajorPolyActRNPoly' and degree != 0:
            valid = (head < heads) & (col < columns_per_head)
    elif op == 'ParBlockColMajorLNAffine':
        if generator['pt_idx'] == 0:
            params = config.get('layernorm_param', {})
            values = values * layer.get('inv_std', params.get('inv_std', 1 / params.get('var_std_bound', 4.0)))
    elif op == 'ParBlockColMajorAddPt':
        values = values.reshape(feature['shape'])
        selected = valid & (global_col < values.shape[1])
        result[selected] = values[row[selected], global_col[selected]]
        return result
    values = values.reshape(-1)
    valid &= global_col < values.size
    result[valid] = values[global_col[valid]]
    return result
