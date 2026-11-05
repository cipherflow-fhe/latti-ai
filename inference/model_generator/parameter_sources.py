# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Parameter provenance shared by the matrix compiler and task generator."""

import copy
import hashlib
import json


def public_source(kind, recipe):
    content = json.dumps(recipe, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return dict(visibility='public', source_kind=kind, source_refs=[],
                public_recipe=dict(version=1, content=recipe,
                                   sha256=hashlib.sha256(content.encode()).hexdigest()))


def secret_source(kind, *refs):
    return dict(visibility='secret', source_kind=kind,
                source_refs=[ref for ref in refs if ref])


def parameter_key(attributes):
    key = attributes['type']
    for field in ('pt_idx', 'coeff_idx'):
        if field in attributes:
            key += f':{field}={int(attributes[field])}'
    return key


def pcm_parameter_scale(attributes, input_levels, parameter):
    """Match the private C++ generators, including non-default intermediate scales."""
    op, kind = attributes['op_class'], attributes['type']
    level, scale = input_levels[0], float(parameter.default_scale)
    q = parameter.q
    # These rules are the same scale contracts used by the C++ RingT encoders.
    if op == 'ParUpperDiagonalLNStats':
        return [q[level], q[level], q[level-1]/scale*q[level-2], scale][int(attributes['pt_idx'])]
    if op.startswith('ParUpperDiagonal'):
        op = op.replace('ParUpperDiagonal', 'ParBlockColMajor', 1)
    if op == 'ParBlockColMajorLNXCentered':
        return float(q[level-int(attributes['pt_idx'])])
    if op == 'ParBlockColMajorLNGoldschmidt':
        return [scale/q[level]*scale/q[level-1]*scale,
                q[level]/scale*q[level]/scale*q[level-1]/scale*q[level-2]][int(attributes['pt_idx'])]
    if op in {'ParBlockColMajorMultipleSquare', 'ParBlockColMajorCtMul'}:
        return float(q[level]/scale*q[level-1])
    if op == 'ParBlockColMajorHeadColSum':
        return float(q[level-int(kind == 'mask_pt')]) if attributes['op_class'].startswith('ParBlock') else float(q[level])
    if op in {'ParBlockColMajorAddPt', 'ParBlockColMajorInverseInit'}:
        return scale
    if op == 'ParBlockColMajorInverseIter':
        return float(q[level]) if kind == 'two_pt' else float((q[level]/scale)**2*q[level-1])
    if op == 'ParBlockColMajorPolyMultCt':
        return scale if kind == 'half_pt' else float(q[level-1]/scale*q[level])
    if kind == 'row_chunk_seed':
        return float(q[level])
    if kind in {'bias_pt', 'add_pt'}:
        return scale
    if op == 'ParBlockColMajorPolyActRNGamma':
        return float(q[level])
    if op == 'ParBlockColMajorPolyActRNPoly':
        degree = int(attributes['coeff_idx'])
        return [scale, q[level], (q[level]/scale)*q[level-1],
                ((q[level]/scale)*q[level]/scale)*q[level-2],
                (((q[level]/scale)*q[level]/scale)*q[level-1]/scale)*q[level-2]][degree]
    if op == 'ParBlockColMajorPoly':
        return float(attributes['parameter_scale'])
    if op == 'ParBlockColMajorLNStats':
        return [q[level], q[level-1], (q[level-2]/scale)*q[level-3], scale][int(attributes['pt_idx'])]
    if op == 'ParBlockColMajorLNMinimaxInit':
        return [scale, q[level], (q[level]/scale)*q[level-1]][int(attributes['pt_idx'])]
    if op == 'ParBlockColMajorLNAffine':
        level_y = input_levels[1]
        return (q[level_y]/scale)*q[level_y-1] if int(attributes['pt_idx']) == 0 else scale
    raise ValueError(f'Unknown secret PCM parameter scale: {op}/{kind}')


def pcm_parameter_sources(layer):
    """Record private imported values; explicit transform records take precedence."""
    sources = copy.deepcopy(layer.get('parameter_sources', {}))
    kind = layer.get('type')
    refs = [layer.get(key, '') for key in ('weight_path', 'bias_path', 'gamma_path',
                                          'coeffs_path', 'running_max_path', 'path')]
    keys = {
        'parcpmm': ('row_chunk_seed', 'bias_pt'),
        'pcm_add_pt': ('add_pt',),
        'pcmgamma': ('gamma_pt',),
        'pcmpoly': ('coeff_pt',),
        'pcmgeneralpoly': ('stockmeyer_coeff_pt',),
        'pcmaffine': ('pt:pt_idx=0', 'pt:pt_idx=1'),
        'pcmstats': ('pt:pt_idx=2', 'pt:pt_idx=3'),
        'pcminit': ('pt:pt_idx=0', 'pt:pt_idx=1', 'pt:pt_idx=2'),
    }.get(kind, ())
    for key in keys:
        sources.setdefault(key, secret_source('calibration' if kind == 'pcmstats' else 'onnx_initializer', *refs))
    return sources


def resolve_pcm_parameter_source(layer, attributes):
    """Only explicit geometric generators are public without a source record."""
    op, kind = attributes['op_class'], attributes['type']
    structural = {
        'ParBlockColMajorCPMM': {'row_chunk_selection', 'mask_h0_pt'},
        'ParBlockColMajorCCMM': {'mask_A', 'mask_B', 'mask_C'},
        'ParBlockColMajorTranspose': {'mask'},
        'ParBlockColMajorMultipleSquare': {'valid_mask_pt'},
        'ParBlockColMajorHeadColSum': {'head0_mask_pt'},
        'ParBlockColMajorInverseInit': {'two_pt'},
        'ParBlockColMajorInverseIter': {'two_pt', 'norm_pt'},
        'ParBlockColMajorCtMul': {'valid_mask_pt'},
        'ParBlockColMajorPolyMultCt': {'half_pt', 'norm_pt'},
    }
    # These classes have no learned or calibrated generator inputs.
    geometric_classes = {'ParBlockColMajorCCMM', 'ParBlockColMajorTranspose',
                         'ParBlockColMajorMultipleSquare', 'ParBlockColMajorHeadColSum',
                         'ParBlockColMajorInverseInit', 'ParBlockColMajorInverseIter',
                         'ParBlockColMajorCtMul', 'ParBlockColMajorPolyMultCt',
                         'ParBlockColMajorLNXCentered', 'ParBlockColMajorLNGoldschmidt'}
    if (op in geometric_classes or kind in structural.get(op, set()) or
            (op == 'ParBlockColMajorLNStats' and int(attributes.get('pt_idx', -1)) in (0, 1))):
        return public_source('shape', dict(generator=op, type=kind,
                                           pt_idx=attributes.get('pt_idx')))
    records = layer.get('parameter_sources', {})
    record = copy.deepcopy(records.get(parameter_key(attributes), records.get(kind)))
    if record is None:
        record = secret_source('h5_parameter')
        record['missing_provenance'] = True
    if record.get('visibility') == 'public':
        recipe = record.get('public_recipe')
        if not recipe or 'content' not in recipe:
            raise ValueError(f"Public PCM parameter lacks recipe: {parameter_key(attributes)}")
        expected = public_source(record['source_kind'], recipe['content'])['public_recipe']
        if recipe != expected:
            raise ValueError(f"Invalid PCM public recipe hash/version: {parameter_key(attributes)}")
    elif record.get('visibility') != 'secret':
        raise ValueError(f"Invalid PCM parameter visibility: {parameter_key(attributes)}")
    return record
