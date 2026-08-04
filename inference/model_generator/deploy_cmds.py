# Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inference.lattisense.frontend.custom_task import *
from inference.model_generator.layers.activation_layer import *
from inference.model_generator.layers.add_pack import *
from inference.model_generator.layers.avgpool1d_layer import *
from inference.model_generator.layers.avgpool2d_layer import *
from inference.model_generator.layers.concat_layer import *
from inference.model_generator.layers.conv1d_packed_layer import *
from inference.model_generator.layers.conv2d_depthwise import *
from inference.model_generator.layers.conv2d_packed_layer import *
from inference.model_generator.layers.dense_packed_layer import *
from inference.model_generator.layers.inverse_multiplexed_conv2d_layer import *
from inference.model_generator.layers.inverse_multiplexed_depthwise_conv2d_layer import *
from inference.model_generator.layers.mult_scaler import *
from inference.model_generator.layers.multiplexed_conv1d_pack_layer import *
from inference.model_generator.layers.multiplexed_dw_conv1d_pack_layer import *
from inference.model_generator.layers.multiplexed_conv2d_pack_layer import *
from inference.model_generator.layers.multiplexed_conv2d_pack_layer_depthwise import *
from inference.model_generator.layers.poly_relu0d import *
from inference.model_generator.layers.poly_relu1d import *
from inference.model_generator.layers.poly_relu2d import *
from inference.model_generator.layers.upsample_layer import *
from inference.model_generator.layers.par_block_col_major_ccmm import ParBlockColMajorCCMM
from inference.model_generator.layers.par_block_col_major_cpmm import ParBlockColMajorCPMM
from inference.model_generator.layers.par_block_col_major_add_pt import ParBlockColMajorAddPt
from inference.model_generator.layers.par_block_col_major_layernorm import (
    ParBlockColMajorLNAffine,
    ParBlockColMajorLNGoldschmidt,
    ParBlockColMajorLNMinimaxInit,
    ParBlockColMajorLNStats,
    ParBlockColMajorLNXCentered,
)
from inference.model_generator.layers.par_block_col_major_polyactrn import (
    ParBlockColMajorPolyActRNGamma,
    ParBlockColMajorPolyActRNPoly,
)
from inference.model_generator.layers.par_block_col_major_transpose import ParBlockColMajorTranspose
from inference.model_generator.layers.par_lower_diag_ccmm import ParLowerDiagCCMM
from inference.model_generator.layers.par_lower_diag_pcmm import ParLowerDiagPCMM
from inference.model_generator.layers.par_lower_diag_transpose import ParLowerDiagTranspose
from inference.model_generator.layers.par_lower_diagonal_add_pt import ParLowerDiagonalAddPt
from inference.model_generator.layers.par_upper_diagonal_layernorm import (
    ParUpperDiagonalLNAffine,
    ParUpperDiagonalLNGoldschmidt,
    ParUpperDiagonalLNMinimaxInit,
    ParUpperDiagonalLNStats,
    ParUpperDiagonalLNXCentered,
)
from inference.model_generator.layers.par_upper_diagonal_polyact import (
    ParUpperDiagonalPolyActRNGamma,
    ParUpperDiagonalPolyActRNPoly,
)
from inference.model_generator.layers.par_upper_diagonal_poly import ParUpperDiagonalPoly
from inference.model_generator.layers.par_upper_diagonal_poly_mult_ct import ParUpperDiagonalPolyMultCt
from inference.model_generator.layers.par_upper_diagonal_softmax import (
    ParUpperDiagonalAddPt,
    ParUpperDiagonalGELU,
    ParUpperDiagonalHeadColSum,
    ParUpperDiagonalInverseInit,
    ParUpperDiagonalInverseIter,
    ParUpperDiagonalMultipleSquare,
)
from training.model_compiler.components import (
    N16QP1546H192H32,
    PN13QP218,
    PN14QP438,
    PN15QP880,
    PN16QP1761,
)


def read_config(config_path):
    with open(config_path, 'r', encoding='utf8') as fp:
        config_ctx = json.load(fp)
    return config_ctx


_FHE_PARAMS = {
    'PN13QP218': PN13QP218,
    'PN14QP438': PN14QP438,
    'PN15QP880': PN15QP880,
    'PN16QP1761': PN16QP1761,
    'N16QP1546H192H32': N16QP1546H192H32,
}

INPUT_MODES = {'ciphertext', 'plaintext'}
PARAMETER_MODES = {'plaintext_lazy', 'plaintext_eager', 'encrypted_offline'}


def set_param(param_name):
    if param_name not in _FHE_PARAMS:
        raise ValueError(f'Unsupported FHE parameter name: {param_name!r}')
    fhe = _FHE_PARAMS[param_name]
    if param_name == 'N16QP1546H192H32':
        param = CkksBtpParam.create_default_param()
    else:
        param = Param.create_ckks_custom_param(
            n=fhe.poly_modulus_degree,
            q=fhe.q,
            p=fhe.p,
        )
        param.scale = 1 << fhe.log_default_scale
    set_fhe_param(param)


def gen_custom_task(
    task_path,
    param_name='PN14QP438',
    use_gpu=True,
    style='ordinary',
    lazy=False,
    parameter_mode=None,
    input_mode='ciphertext',
    output_dir=None,
):
    if parameter_mode is None:
        parameter_mode = 'plaintext_lazy' if lazy else 'plaintext_eager'
    if parameter_mode not in PARAMETER_MODES:
        raise ValueError(f'Unsupported parameter_mode: {parameter_mode!r}. Expected one of {sorted(PARAMETER_MODES)}')
    if input_mode not in INPUT_MODES:
        raise ValueError(f'Unsupported input_mode: {input_mode!r}. Expected one of {sorted(INPUT_MODES)}')
    if parameter_mode == 'plaintext_lazy':
        lazy = True
    elif parameter_mode in {'plaintext_eager', 'encrypted_offline'}:
        lazy = False
    output_dir = task_path if output_dir is None else output_dir

    n = _FHE_PARAMS[param_name].poly_modulus_degree
    set_param(param_name)
    task_config_info = read_config(os.path.join(task_path, 'task_config.json'))
    try:
        block_shape = task_config_info['block_shape']
    except Exception:
        block_shape = [64, 64]
    config_info = read_config(os.path.join(task_path, 'nn_layers_ct_0.json'))
    input_args = list()
    offline_input_args = list()
    feature_id_to_nodes_map = {}
    par_feature_shapes = {}
    pdm_feature_head_shapes = {}
    task_output_feature_ids = config_info['output_feature']
    graph_input_feature_ids = set(config_info['input_feature'])
    feature_consumers = {}
    for consumer_layer_id, lyr in config_info['layer'].items():
        for fid in lyr.get('feature_input', []):
            feature_consumers.setdefault(fid, []).append((consumer_layer_id, lyr))

    def _require_positive_task_config_int(field_name, layer_id, layer_type):
        if field_name not in task_config_info:
            raise ValueError(f"Layer '{layer_id}' ({layer_type}) requires '{field_name}' in task_config.json")
        try:
            value = int(task_config_info[field_name])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Layer '{layer_id}' ({layer_type}) requires positive integer '{field_name}' in task_config.json, "
                f'got {task_config_info[field_name]!r}'
            ) from exc
        if value <= 0:
            raise ValueError(
                f"Layer '{layer_id}' ({layer_type}) requires positive '{field_name}' in task_config.json, got {value}"
            )
        return value

    def _matmul_block_size():
        return _require_positive_task_config_int('matmul_block_size', 'par block-col-major', 'task')

    def _par_input_shape(feat, n_heads, split_rows=False, feature_id=None):
        if 'head_shape' in feat:
            return tuple(feat['head_shape'])
        if feat.get('data_type') == 'feature_mat':
            name = f" '{feature_id}'" if feature_id is not None else ''
            raise ValueError(f'feature_mat{name} must define head_shape for par block-col-major layers')
        rows, cols = feat['shape']
        rows_per_head = rows // n_heads if split_rows else rows
        return (rows_per_head, cols // n_heads)

    def _par_ct_count(shape_per_head, block_size, G):
        return math.ceil(shape_per_head[0] / block_size) * math.ceil(shape_per_head[1] / block_size) * G

    def _par_group_count(block_size, n_heads, n_slot):
        n_h_padded = 1
        while n_h_padded < n_heads:
            n_h_padded <<= 1
        if n_slot >= n_h_padded * block_size * block_size:
            return 1
        S = n_slot // (block_size * block_size)
        if S == 1:
            n_h_padded = n_heads
        return n_h_padded // S

    def _feature_mat_ct_info(feat, n_heads, n_slot, split_rows=False, block_size=None, feature_id=None):
        shape_per_head = _par_input_shape(feat, n_heads, split_rows=split_rows, feature_id=feature_id)
        block_size = int(block_size or _matmul_block_size())
        G = _par_group_count(block_size, n_heads, n_slot)
        return shape_per_head, block_size, G, _par_ct_count(shape_per_head, block_size, G)

    def _head_dim(layer_id, layer_type):
        return _require_positive_task_config_int('head_dim', layer_id, layer_type)

    def _normalize_pdm_head_shape(head_shape):
        head_shape = tuple(int(v) for v in head_shape)
        if len(head_shape) != 2:
            return head_shape
        if task_config_info.get('mat_pack_style') != 'par_diagonal_pack':
            return head_shape
        try:
            head_dim = int(task_config_info.get('head_dim', 0))
        except (TypeError, ValueError):
            return head_shape
        if head_dim > 0 and head_shape[0] == head_dim and head_shape[1] != head_dim:
            return (head_shape[1], head_shape[0])
        return head_shape

    def _pdm_head_shape(feat, feature_id, layer_id, layer_type):
        if feature_id in pdm_feature_head_shapes:
            return pdm_feature_head_shapes[feature_id]
        if 'head_shape' not in feat:
            raise ValueError(f"feature_mat '{feature_id}' must define head_shape for {layer_type} layer '{layer_id}'")
        head_shape = _normalize_pdm_head_shape(feat['head_shape'])
        if len(head_shape) != 2 or head_shape[0] <= 0 or head_shape[1] <= 0:
            raise ValueError(
                f"feature_mat '{feature_id}' has invalid head_shape for {layer_type} layer '{layer_id}': "
                f'{feat["head_shape"]!r}'
            )
        pdm_feature_head_shapes[feature_id] = head_shape
        return head_shape

    def _feature_input_is_plaintext(feature_id):
        return input_mode == 'plaintext' and feature_id in graph_input_feature_ids

    def _make_feature_nodes(feature_id, count, level):
        if _feature_input_is_plaintext(feature_id):
            return [CkksPlaintextRingtNode(feature_id + f'input{k}') for k in range(count)]
        return [CkksCiphertextNode(feature_id + f'input{k}', level=level) for k in range(count)]

    encrypted_parameter_arg_counts = {}
    encrypted_parameter_effective_ids = set()

    def _flatten_nodes(nodes):
        if isinstance(nodes, list):
            result = []
            for item in nodes:
                result.extend(_flatten_nodes(item))
            return result
        return [nodes]

    def _argument_level(nodes):
        for node in _flatten_nodes(nodes):
            if hasattr(node, 'level'):
                return int(node.level)
        return None

    def _append_parameter_arg(arg_id, nodes):
        if parameter_mode == 'encrypted_offline':
            effective_id = arg_id
            count = encrypted_parameter_arg_counts.get(arg_id, 0)
            encrypted_parameter_arg_counts[arg_id] = count + 1
            if count > 0:
                level = _argument_level(nodes)
                suffix = f'__L{level}' if level is not None else f'__U{count}'
                effective_id = f'{arg_id}{suffix}'
                extra = 1
                while effective_id in encrypted_parameter_effective_ids:
                    effective_id = f'{arg_id}{suffix}__U{extra}'
                    extra += 1
            encrypted_parameter_effective_ids.add(effective_id)
            arg = Argument(effective_id, nodes)
            offline_input_args.append(arg)
        else:
            arg = Argument(arg_id, nodes)
            input_args.append(arg)

    def _unsupported_encrypted_parameter_layer(layer_id, layer_type):
        raise ValueError(
            f"Layer '{layer_id}' ({layer_type}) is not supported by "
            'parameter_mode="encrypted_offline" in Phase 3. Supported parameterized paths are conv2d variants, '
            '0D/special dense, mult_scalar, and polyrelu/polyact. Public structural constants remain plaintext.'
        )

    def _register_feature_nodes(feature_id, count, level):
        count = int(count)
        if count <= 0:
            raise ValueError(f"feature '{feature_id}' requires positive ciphertext count, got {count}")
        if feature_id in feature_id_to_nodes_map:
            nodes = feature_id_to_nodes_map[feature_id]
            if len(nodes) != count:
                raise ValueError(
                    f"feature '{feature_id}' is already registered with {len(nodes)} ciphertexts, expected {count}"
                )
            return nodes
        nodes = _make_feature_nodes(feature_id, count, level)
        feature_id_to_nodes_map[feature_id] = nodes
        input_args.append(Argument(feature_id, nodes))
        return nodes

    def _remember_pdm_output_head_shape(feature_id):
        feat = config_info['feature'][feature_id]
        if 'head_shape' in feat:
            pdm_feature_head_shapes[feature_id] = _normalize_pdm_head_shape(feat['head_shape'])

    def _pdm_upper_layer(layer_type, shape, head_shape, n_heads, layer_id, layer_config):
        if layer_type == 'pdmstats':
            return ParUpperDiagonalLNStats(shape, head_shape, n_heads, n // 2)
        if layer_type == 'pdmcenter':
            return ParUpperDiagonalLNXCentered(shape, head_shape, n_heads, n // 2)
        if layer_type == 'pdminit':
            return ParUpperDiagonalLNMinimaxInit(shape, head_shape, n_heads, n // 2)
        if layer_type == 'pdmgs':
            return ParUpperDiagonalLNGoldschmidt(shape, head_shape, n_heads, n // 2)
        if layer_type == 'pdmaffine':
            return ParUpperDiagonalLNAffine(shape, head_shape, n_heads, n // 2)
        if layer_type == 'pdmgamma':
            return ParUpperDiagonalPolyActRNGamma(shape, head_shape, n_heads, n // 2)
        if layer_type == 'pdmpoly':
            degree = int(layer_config.get('degree', layer_config.get('order', 2)))
            if degree not in (2, 4):
                raise ValueError(f"pdmpoly layer '{layer_id}' only supports degree 2 or 4, got {degree}")
            return ParUpperDiagonalPolyActRNPoly(shape, head_shape, n_heads, n // 2, degree)
        if layer_type == 'pdmupperaddpt':
            return ParUpperDiagonalAddPt(shape, head_shape, n_heads, n // 2)
        if layer_type == 'pdmupperpoly':
            return ParUpperDiagonalPoly(shape, head_shape, n_heads, n // 2, int(layer_config.get('order', 15)))
        if layer_type == 'pdmmulsquare':
            return ParUpperDiagonalMultipleSquare(shape, head_shape, n_heads, n // 2)
        if layer_type == 'pdmheadcolsum':
            return ParUpperDiagonalHeadColSum(shape, head_shape, n_heads, n // 2)
        if layer_type == 'pdminvinit':
            return ParUpperDiagonalInverseInit(shape, head_shape, n_heads, n // 2)
        if layer_type == 'pdminviter':
            return ParUpperDiagonalInverseIter(shape, head_shape, n_heads, n // 2)
        if layer_type == 'pdmctmul':
            return ParUpperDiagonalGELU(shape, head_shape, n_heads, n // 2)
        if layer_type == 'pdmupperpolymultct':
            return ParUpperDiagonalPolyMultCt(shape, head_shape, n_heads, n // 2)
        raise ValueError(f'Unsupported PDM upper layer type: {layer_type}')

    def _pdm_upper_shape(feat, head_shape):
        return (head_shape[0], int(feat['shape'][0]))

    def _pdm_lower_shape_and_head(feat, feature_id, layer_id, layer_type):
        head_shape = _pdm_head_shape(feat, feature_id, layer_id, layer_type)
        return (int(feat['shape'][0]), head_shape[0]), (head_shape[1], head_shape[0])

    def _pdm_transpose_shape(feat, feature_id, layer_id, layer_type):
        head_shape = _pdm_head_shape(feat, feature_id, layer_id, layer_type)
        return (head_shape[1], head_shape[0])

    def _pdm_single_input_total_ct_layer(layer_type, feat, feature_id, layer_id, layer_config):
        n_heads = _require_positive_task_config_int('n_heads', layer_id, layer_type)
        head_shape = _pdm_head_shape(feat, feature_id, layer_id, layer_type)
        if layer_type == 'pdm_add_pt':
            shape, lower_head_shape = _pdm_lower_shape_and_head(feat, feature_id, layer_id, layer_type)
            return ParLowerDiagonalAddPt(shape, lower_head_shape, n_heads, n // 2)
        return _pdm_upper_layer(
            layer_type, _pdm_upper_shape(feat, head_shape), head_shape, n_heads, layer_id, layer_config
        )

    def _pdm_feature_ct_count(feat, feature_id, layer_id, layer_type):
        n_heads = _require_positive_task_config_int('n_heads', layer_id, layer_type)
        head_shape = _pdm_head_shape(feat, feature_id, layer_id, layer_type)
        layer = ParUpperDiagonalPolyActRNGamma(_pdm_upper_shape(feat, head_shape), head_shape, n_heads, n // 2)
        return layer.total_cts

    def _pdmpcmm_shapes(layer_config, feat_in, feat_out, input_fid, layer_id):
        head_shape = _pdm_head_shape(feat_in, input_fid, layer_id, layer_config['type'])
        weight_shape = layer_config.get('weight_shape')
        if weight_shape:
            W_T_shape = tuple(int(v) for v in weight_shape)
            if len(W_T_shape) != 2 or W_T_shape[0] <= 0 or W_T_shape[1] <= 0:
                raise ValueError(f"pdmpcmm layer '{layer_id}' has invalid weight_shape: {weight_shape!r}")
        else:
            W_T_shape = (int(feat_out['shape'][0]), int(feat_in['shape'][0]))
        shape_X_T = (W_T_shape[1], head_shape[0])
        return shape_X_T, W_T_shape

    def _make_pdmccmm_layer(layer_id, layer_type, fid0, fid1, feat0, feat1, n_heads):
        head0 = _pdm_head_shape(feat0, fid0, layer_id, layer_type)
        head1 = _pdm_head_shape(feat1, fid1, layer_id, layer_type)
        lower0 = (head0[1], head0[0])
        lower1 = (head1[1], head1[0])
        candidates = [
            (fid0, head0, fid1, head1),
            (fid1, head1, fid0, head0),
            (fid0, lower0, fid1, head1),
            (fid1, lower1, fid0, head0),
            (fid0, head0, fid1, lower1),
            (fid1, head1, fid0, lower0),
            (fid0, lower0, fid1, lower1),
            (fid1, lower1, fid0, lower0),
        ]
        errors = []
        for fid_A, shape_A, fid_B, shape_B in candidates:
            try:
                layer = ParLowerDiagCCMM(shape_A, shape_B, n_heads, _head_dim(layer_id, layer_type), n // 2)
                return layer, fid_A, fid_B
            except (AssertionError, ValueError) as exc:
                errors.append(f'{shape_A} @ {shape_B}: {exc}')
        raise ValueError(f"pdmccmm layer '{layer_id}' has unsupported input head shapes: {errors}")

    def _pdm_consumer_input_ct_count(feature_id, consumer_layer_id, consumer):
        consumer_type = consumer['type']
        feat = config_info['feature'][feature_id]
        n_heads = _require_positive_task_config_int('n_heads', consumer_layer_id, consumer_type)
        if consumer_type == 'pdmtranspose':
            shape = _pdm_transpose_shape(feat, feature_id, consumer_layer_id, consumer_type)
            layer = ParLowerDiagTranspose(shape, n_heads, _head_dim(consumer_layer_id, consumer_type), n // 2)
            return layer.m_c
        if consumer_type == 'pdmpcmm':
            feat_out = config_info['feature'][consumer['feature_output'][0]]
            shape_X_T, W_T_shape = _pdmpcmm_shapes(consumer, feat, feat_out, feature_id, consumer_layer_id)
            layer = ParLowerDiagPCMM(
                shape_X_T,
                W_T_shape,
                n_heads,
                _head_dim(consumer_layer_id, consumer_type),
                n // 2,
                has_bias='bias_path' in consumer,
            )
            return layer.K_col * layer.m_c
        if consumer_type == 'pdmccmm':
            fid0, fid1 = consumer['feature_input'][:2]
            feat0 = config_info['feature'][fid0]
            feat1 = config_info['feature'][fid1]
            layer, fid_A, fid_B = _make_pdmccmm_layer(
                consumer_layer_id, consumer_type, fid0, fid1, feat0, feat1, n_heads
            )
            if feature_id == fid_A:
                return layer.m_c
            if feature_id == fid_B:
                return layer.m_c if layer.is_kqt else layer.n_c
            return None
        if consumer_type in {
            'pdm_add_pt',
            'pdmstats',
            'pdmcenter',
            'pdminit',
            'pdmgs',
            'pdmaffine',
            'pdmgamma',
            'pdmpoly',
            'pdmupperaddpt',
            'pdmupperpoly',
            'pdmmulsquare',
            'pdmheadcolsum',
            'pdminvinit',
        }:
            return _pdm_single_input_total_ct_layer(
                consumer_type, feat, feature_id, consumer_layer_id, consumer
            ).total_cts
        if consumer_type in {'pdminviter', 'pdmctmul', 'pdmupperpolymultct'}:
            return _pdm_single_input_total_ct_layer(
                consumer_type, feat, feature_id, consumer_layer_id, consumer
            ).total_cts
        if consumer_type in {'add', 'add2d'}:
            return _pdm_feature_ct_count(feat, feature_id, consumer_layer_id, consumer_type)
        return None

    def _pdm_consumed_ct_count(feature_id, default_count):
        counts = []
        for consumer_layer_id, consumer in feature_consumers.get(feature_id, []):
            count = _pdm_consumer_input_ct_count(feature_id, consumer_layer_id, consumer)
            if count is not None:
                counts.append(int(count))
        if not counts:
            return default_count
        return max(counts)

    # Pre-add all graph-level input ciphertexts first so they precede weights in input_args.
    # This ensures the C++ signature position-matching works correctly for multi-input models.
    # Only needed when there are multiple graph-level inputs; single-input models are handled
    # correctly by the lazy-loading in the main loop (which also computes the right CT count
    # for big_size layers).
    if len(config_info['input_feature']) > 1:
        # First-use lookup: {input_fid: (first consumer layer id, first consumer layer config)}
        first_use = {}
        for consumer_layer_id, lyr in config_info['layer'].items():
            for fid in lyr.get('feature_input', []):
                if fid in config_info['input_feature'] and fid not in first_use:
                    first_use[fid] = (consumer_layer_id, lyr)

        for input_fid in config_info['input_feature']:
            feat = config_info['feature'][input_fid]
            level = int(feat['level'])
            consumer_info = first_use.get(input_fid)
            consumer_layer_id = None
            consumer = None
            if consumer_info is not None:
                consumer_layer_id, consumer = consumer_info
            if feat.get('data_type') == 'feature_mat':
                if consumer is None:
                    continue
                consumer_type = consumer['type']
                n_heads = _require_positive_task_config_int('n_heads', consumer_layer_id, consumer_type)
                if consumer_type in {'partranspose', 'parccmm', 'parcpmm'} and n_heads <= 1:
                    raise ValueError(f"feature_mat input '{input_fid}' only supports par matrix ops with n_heads > 1")
                if consumer_type == 'partranspose':
                    shape_per_head = _par_input_shape(feat, n_heads, split_rows=False, feature_id=input_fid)
                    par_feature_shapes[input_fid] = shape_per_head
                    block_size = _matmul_block_size()
                    G = _par_group_count(block_size, n_heads, n // 2)
                    n_packed = _par_ct_count(shape_per_head, block_size, G)
                elif consumer_type == 'parccmm':
                    idx = consumer['feature_input'].index(input_fid)
                    shape_per_head = _par_input_shape(feat, n_heads, split_rows=(idx == 1), feature_id=input_fid)
                    par_feature_shapes[input_fid] = shape_per_head
                    block_size = _matmul_block_size()
                    G = _par_group_count(block_size, n_heads, n // 2)
                    n_packed = _par_ct_count(shape_per_head, block_size, G)
                elif consumer_type == 'parcpmm':
                    shape_per_head = _par_input_shape(feat, n_heads, split_rows=False, feature_id=input_fid)
                    par_feature_shapes[input_fid] = shape_per_head
                    block_size = _matmul_block_size()
                    if shape_per_head[1] > block_size:
                        raise ValueError(f"parcpmm input '{input_fid}' per-head width exceeds matmul_block_size")
                    G = _par_group_count(block_size, n_heads, n // 2)
                    n_packed = math.ceil(shape_per_head[0] / block_size) * G
                elif consumer_type == 'pdmtranspose':
                    shape = _pdm_transpose_shape(feat, input_fid, consumer_layer_id, consumer_type)
                    layer = ParLowerDiagTranspose(shape, n_heads, _head_dim(consumer_layer_id, consumer_type), n // 2)
                    n_packed = layer.m_c
                elif consumer_type == 'pdmpcmm':
                    feat_out = config_info['feature'][consumer['feature_output'][0]]
                    shape_X_T, W_T_shape = _pdmpcmm_shapes(consumer, feat, feat_out, input_fid, consumer_layer_id)
                    layer = ParLowerDiagPCMM(
                        shape_X_T,
                        W_T_shape,
                        n_heads,
                        _head_dim(consumer_layer_id, consumer_type),
                        n // 2,
                        has_bias='bias_path' in consumer,
                    )
                    n_packed = layer.K_col * layer.m_c
                elif consumer_type == 'pdmccmm':
                    fid0, fid1 = consumer['feature_input'][:2]
                    feat0 = config_info['feature'][fid0]
                    feat1 = config_info['feature'][fid1]
                    layer, fid_A, fid_B = _make_pdmccmm_layer(
                        consumer_layer_id,
                        consumer_type,
                        fid0,
                        fid1,
                        feat0,
                        feat1,
                        n_heads,
                    )
                    if input_fid == fid_A:
                        n_packed = layer.m_c
                    elif input_fid == fid_B:
                        n_packed = layer.m_c if layer.is_kqt else layer.n_c
                    else:
                        raise ValueError(f"pdmccmm layer '{consumer_layer_id}' does not consume input '{input_fid}'")
                elif consumer_type in {
                    'pdm_add_pt',
                    'pdmstats',
                    'pdmcenter',
                    'pdminit',
                    'pdmgs',
                    'pdmaffine',
                    'pdmgamma',
                    'pdmpoly',
                    'pdmupperaddpt',
                    'pdmupperpoly',
                    'pdmmulsquare',
                    'pdmheadcolsum',
                    'pdminvinit',
                    'pdminviter',
                    'pdmctmul',
                    'pdmupperpolymultct',
                }:
                    layer = _pdm_single_input_total_ct_layer(
                        consumer_type, feat, input_fid, consumer_layer_id, consumer
                    )
                    n_packed = layer.total_cts
                elif consumer_type in {'pcmstats', 'pcmcenter', 'pcmgamma', 'pcmpoly', 'add', 'add2d', 'pcm_add_pt'}:
                    shape_per_head, _, _, n_packed = _feature_mat_ct_info(feat, n_heads, n // 2, feature_id=input_fid)
                    par_feature_shapes[input_fid] = shape_per_head
                else:
                    raise ValueError(
                        f"feature_mat input '{input_fid}' is consumed by unsupported matrix layer "
                        f"'{consumer_type}'; use parcpmm/parccmm/partranspose, pdmpcmm/pdmccmm/pdmtranspose, "
                        'PCM/PDM stages, or add2d'
                    )
            else:
                pack = int(feat['pack_num'])
                n_in_channel_fid = int(feat['channel'])
                n_packed = math.ceil(n_in_channel_fid / pack)
                # Mirror the big_size expansion logic from the per-layer loop below:
                # big_size conv2d / avgpool / mult_scalar (2D only) consume inputs as
                #   n_in_channel * block_expansion[0] * block_expansion[1]
                # ciphertexts, not the default ceil(channel / pack_num).
                if (
                    consumer is not None
                    and consumer.get('is_big_size', False)
                    and feat.get('dim', 2) == 2
                    and (
                        consumer['type'] == 'conv2d'
                        or 'avgpool' in consumer['type']
                        or consumer['type'] == 'mult_scalar'
                    )
                ):
                    input_shape = feat['shape']
                    be0 = math.ceil(input_shape[0] / block_shape[0])
                    be1 = math.ceil(input_shape[1] / block_shape[1])
                    n_packed = n_in_channel_fid * be0 * be1
            _register_feature_nodes(input_fid, n_packed, level)

    _PAR_MATRIX_LAYER_TYPES = {'parcpmm', 'parccmm', 'partranspose', 'pcm_add_pt'}
    _PCM_LAYER_TYPES = {'pcmstats', 'pcmcenter', 'pcminit', 'pcmgs', 'pcmaffine', 'pcmgamma', 'pcmpoly'}
    _PDM_MATRIX_LAYER_TYPES = {'pdmpcmm', 'pdmccmm', 'pdmtranspose', 'pdm_add_pt'}
    _PDM_LAYER_TYPES = {
        'pdmstats',
        'pdmcenter',
        'pdminit',
        'pdmgs',
        'pdmaffine',
        'pdmgamma',
        'pdmpoly',
        'pdmupperaddpt',
        'pdmupperpoly',
        'pdmmulsquare',
        'pdmheadcolsum',
        'pdminvinit',
        'pdminviter',
        'pdmctmul',
        'pdmupperpolymultct',
    }
    _FEATURE_MAT_LAYER_TYPES = _PAR_MATRIX_LAYER_TYPES | _PCM_LAYER_TYPES | _PDM_MATRIX_LAYER_TYPES | _PDM_LAYER_TYPES
    _UNSUPPORTED_MATRIX_LAYER_TYPES = {'cpmm', 'qkvcpmm', 'ccmm', 'transpose'}
    _ENCRYPTED_OFFLINE_UNSUPPORTED_LAYER_TYPES = _FEATURE_MAT_LAYER_TYPES

    for layer_id, layer_config in config_info['layer'].items():
        if layer_config['type'] == 'relu2d':
            continue
        if parameter_mode == 'encrypted_offline' and layer_config['type'] in _ENCRYPTED_OFFLINE_UNSUPPORTED_LAYER_TYPES:
            _unsupported_encrypted_parameter_layer(layer_id, layer_config['type'])
        layer_input_feature_ids = layer_config['feature_input']
        layer_output_feature_ids = layer_config['feature_output']

        # Matrix layer types have no channel_input/skip/pack. Non-par matrix ops are unsupported;
        # par input registration is deferred to their own elif branches below.
        if layer_config['type'] in _UNSUPPORTED_MATRIX_LAYER_TYPES:
            raise ValueError(
                f"Layer '{layer_id}' has unsupported non-par matrix type '{layer_config['type']}'; "
                'use parcpmm/parccmm/partranspose or pdmpcmm/pdmccmm/pdmtranspose'
            )
        input_feat0 = config_info['feature'][layer_input_feature_ids[0]]
        if layer_config['type'] in _FEATURE_MAT_LAYER_TYPES or input_feat0.get('data_type') == 'feature_mat':
            level = int(input_feat0['level'])
        else:
            groups = 1
            n_in_channel = int(layer_config['channel_input'])
            n_out_channel = int(layer_config['channel_output'])

            skip = input_feat0['skip']
            pack = int(input_feat0['pack_num'])
            level = int(input_feat0['level'])
            n_packed_in_channel = math.ceil(n_in_channel / pack)
            n_packed_out_channel = math.ceil(n_out_channel / pack)

            # For big_conv/big_avgpool/big_mult_scalar (2D only)
            if (
                (
                    layer_config['type'] == 'conv2d'
                    or 'avgpool' in layer_config['type']
                    or layer_config['type'] == 'mult_scalar'
                )
                and layer_config.get('is_big_size', False)
                and config_info['feature'][layer_input_feature_ids[0]].get('dim', 2) == 2
            ):
                input_shape = config_info['feature'][layer_input_feature_ids[0]]['shape']
                block_expansion = (
                    math.ceil(input_shape[0] / block_shape[0]),
                    math.ceil(input_shape[1] / block_shape[1]),
                )
                n_packed_in_channel = n_in_channel * block_expansion[0] * block_expansion[1]

            if layer_config['type'] == 'conv1d':
                _input_shape_1d = config_info['feature'][layer_input_feature_ids[0]]['shape'][0]
                _skip_1d = skip[0] if isinstance(skip, list) else skip
                if style == 'multiplexed':
                    n_packed_in_channel = math.ceil(n_in_channel / math.ceil(n // 2 / _input_shape_1d))
                else:
                    n_packed_in_channel = math.ceil(n_in_channel / int(n // 2 // _input_shape_1d // _skip_1d))

            for input_node in layer_input_feature_ids:
                if input_node not in feature_id_to_nodes_map.keys():
                    x = _make_feature_nodes(input_node, n_packed_in_channel, level)
                    feature_id_to_nodes_map.update({input_node: x})
                    input_args.append(Argument(input_node, x))

        if layer_config['type'] == 'reshape':
            layer_output_nodes = feature_id_to_nodes_map[layer_input_feature_ids[0]]
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif layer_config['type'] == 'conv2d':
            input_shape = config_info['feature'][layer_input_feature_ids[0]]['shape']
            groups = layer_config['groups']
            kernel_shape = layer_config['kernel_shape']
            stride = layer_config['stride']
            is_big_conv = layer_config['is_big_size']
            block_expansion = (math.ceil(input_shape[0] / block_shape[0]), math.ceil(input_shape[1] / block_shape[1]))
            next_stride = [block_expansion[0] // stride[0], block_expansion[1] // stride[1]]
            padding = [-1, -1]
            if is_big_conv:
                if groups == n_out_channel and groups != 1:
                    big_conv = InverseMultiplexedDepthwiseConv2DLayer(
                        n_out_channel,
                        input_shape,
                        padding,
                        kernel_shape,
                        stride,
                        block_shape,
                    )
                else:
                    big_conv = InverseMultiplexedConv2DLayer(
                        n_out_channel,
                        n_in_channel,
                        input_shape,
                        padding,
                        kernel_shape,
                        stride,
                        block_shape,
                    )

                if lazy:
                    conv_data_source = CustomDataNode(type='conv_data_source', id=f'{layer_id}')
                    layer_output_nodes = big_conv.call_custom_compute(
                        feature_id_to_nodes_map[layer_input_feature_ids[0]],
                        conv_data_source,
                        n,
                    )
                    feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
                    input_args.append(Argument(f'{layer_id}', [conv_data_source]))
                elif parameter_mode == 'encrypted_offline':
                    weight_ct, bias_ct, repack_mask_pt = big_conv.make_param_ct_nodes(layer_id, level)
                    _append_parameter_arg(f'convw_{layer_id}', weight_ct)
                    _append_parameter_arg(f'convb_{layer_id}', bias_ct)
                    if repack_mask_pt is not None:
                        input_args.append(Argument(f'repack_mask_{layer_id}', [repack_mask_pt]))
                    layer_output_nodes = big_conv.call_param_ct(
                        feature_id_to_nodes_map[layer_input_feature_ids[0]],
                        weight_ct,
                        bias_ct,
                        n,
                        repack_mask_pt=repack_mask_pt,
                        input_is_plaintext=_feature_input_is_plaintext(layer_input_feature_ids[0]),
                    )
                    feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
                else:
                    weight_pt, bias_pt, repack_mask_pt = big_conv.make_pt_nodes(layer_id)
                    layer_output_nodes = big_conv.call(
                        feature_id_to_nodes_map[layer_input_feature_ids[0]],
                        weight_pt,
                        bias_pt,
                        n,
                        repack_mask_pt=repack_mask_pt,
                    )
                    feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
                    input_args.append(Argument(f'convw_{layer_id}', weight_pt))
                    input_args.append(Argument(f'convb_{layer_id}', bias_pt))
                    if repack_mask_pt is not None:
                        input_args.append(Argument(f'repack_mask_{layer_id}', [repack_mask_pt]))
            else:
                if style == 'ordinary':
                    if groups == n_out_channel and groups != 1:
                        conv0_layer = Conv2DPackedDepthwiseLayer(
                            n_out_channel,
                            n_in_channel,
                            input_shape,
                            kernel_shape,
                            stride,
                            skip,
                            pack,
                            n_packed_in_channel,
                            n_packed_out_channel,
                        )
                    else:
                        conv0_layer = Conv2DPackedLayer(
                            n_out_channel,
                            n_in_channel,
                            input_shape,
                            kernel_shape,
                            stride,
                            skip,
                            pack,
                            n_packed_in_channel,
                            n_packed_out_channel,
                        )

                    if lazy:
                        conv_data_source = CustomDataNode(type='conv_data_source', id=f'{layer_id}')
                        layer_output_nodes = conv0_layer.call_custom_compute(
                            feature_id_to_nodes_map[layer_input_feature_ids[0]], conv_data_source
                        )
                        feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
                        input_args.append(Argument(f'{layer_id}', [conv_data_source]))
                    elif parameter_mode == 'encrypted_offline':
                        weight_ct, bias_ct = conv0_layer.make_param_ct_nodes(layer_id, level)
                        _append_parameter_arg(f'convw_{layer_id}', weight_ct)
                        _append_parameter_arg(f'convb_{layer_id}', bias_ct)
                        layer_output_nodes = conv0_layer.call_param_ct(
                            feature_id_to_nodes_map[layer_input_feature_ids[0]],
                            weight_ct,
                            bias_ct,
                            input_is_plaintext=_feature_input_is_plaintext(layer_input_feature_ids[0]),
                        )
                        feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
                    else:
                        weight_pt, bias_pt = conv0_layer.make_pt_nodes(layer_id)
                        input_args.append(Argument(f'convw_{layer_id}', weight_pt))
                        input_args.append(Argument(f'convb_{layer_id}', bias_pt))
                        layer_output_nodes = conv0_layer.call(
                            feature_id_to_nodes_map[layer_input_feature_ids[0]], weight_pt, bias_pt
                        )
                        feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
                if style == 'multiplexed':
                    n_in_channel_per_ct = pack
                    if groups == n_out_channel and groups != 1:
                        conv0_layer = MultiplexedConv2DPackedLayerDepthwise(
                            n_out_channel,
                            n_in_channel,
                            input_shape,
                            kernel_shape,
                            stride,
                            skip,
                            n_in_channel_per_ct,
                            n_packed_in_channel,
                            n_packed_out_channel,
                        )
                    else:
                        conv0_layer = MultiplexedConv2DPackedLayer(
                            n_out_channel,
                            n_in_channel,
                            input_shape,
                            kernel_shape,
                            stride,
                            skip,
                            pack,
                            n_packed_in_channel,
                            n_packed_out_channel,
                        )

                    if lazy:
                        conv_data_source = CustomDataNode(type='conv_data_source', id=f'{layer_id}')
                        layer_output_nodes = conv0_layer.call_custom_compute(
                            feature_id_to_nodes_map[layer_input_feature_ids[0]], conv_data_source
                        )
                        feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
                        input_args.append(Argument(f'{layer_id}', [conv_data_source]))
                    elif parameter_mode == 'encrypted_offline':
                        weight_ct, bias_ct, mask_pt = conv0_layer.make_param_ct_nodes(layer_id, level)
                        if mask_pt:
                            input_args.append(Argument(f'convm_{layer_id}', mask_pt))
                        _append_parameter_arg(f'convw_{layer_id}', weight_ct)
                        _append_parameter_arg(f'convb_{layer_id}', bias_ct)
                        layer_output_nodes = conv0_layer.call_param_ct(
                            feature_id_to_nodes_map[layer_input_feature_ids[0]],
                            weight_ct,
                            bias_ct,
                            mask_pt,
                            input_is_plaintext=_feature_input_is_plaintext(layer_input_feature_ids[0]),
                        )
                        feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
                    else:
                        weight_pt, bias_pt, mask_pt = conv0_layer.make_pt_nodes(layer_id)
                        if mask_pt:
                            input_args.append(Argument(f'convm_{layer_id}', mask_pt))
                        input_args.append(Argument(f'convw_{layer_id}', weight_pt))
                        input_args.append(Argument(f'convb_{layer_id}', bias_pt))
                        layer_output_nodes = conv0_layer.call(
                            feature_id_to_nodes_map[layer_input_feature_ids[0]], weight_pt, bias_pt, mask_pt
                        )
                        feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif layer_config['type'] in ('batchnorm2d', 'dropout', 'constmul', 'identity'):
            layer_output_nodes = feature_id_to_nodes_map[layer_input_feature_ids[0]]
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif layer_config['type'] == 'square2d':
            act_layer = SquareLayer(level)
            layer_output_nodes = act_layer.call(feature_id_to_nodes_map[layer_input_feature_ids[0]])
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif layer_config['type'] in ('poly_relu2d', 'polyact'):
            feat = config_info['feature'][layer_input_feature_ids[0]]
            level = int(feat['level'])
            order = layer_config['order']
            dim = feat['dim']

            feature_id_in_nodes = feature_id_to_nodes_map[layer_input_feature_ids[0]]
            drop_level_n = feature_id_in_nodes[0].level - level
            if level < feature_id_in_nodes[0].level:
                feature_id_in_nodes = [drop_level(node, drop_level_n) for node in feature_id_in_nodes]

            if dim == 0:
                skip_0d = feat['skip'] if not isinstance(feat['skip'], list) else feat['skip'][0]
                n_channel_per_ct_0d = int(n // 2 // skip_0d)
                polyrelu = PolyRelu0D(order, skip_0d, n_channel_per_ct_0d)
                if lazy:
                    poly_data_source = CustomDataNode(type='poly_data_source', id=f'{layer_id}')
                    layer_output_nodes = polyrelu.call_bsgs_feature0d_lazy(
                        feature_id_in_nodes, poly_data_source, layer_id
                    )
                    input_args.append(Argument(f'{layer_id}', [poly_data_source]))
                elif parameter_mode == 'encrypted_offline':
                    weight_ct = polyrelu.make_param_ct_nodes(
                        layer_id, len(feature_id_in_nodes), feature_id_in_nodes[0].level
                    )
                    layer_output_nodes = polyrelu.call_bsgs_param_ct(feature_id_in_nodes, weight_ct)
                else:
                    weight_pt = polyrelu.make_pt_nodes(layer_id, len(feature_id_in_nodes))
                    layer_output_nodes = polyrelu.call_bsgs_feature0d(feature_id_in_nodes, weight_pt)

            elif dim == 1:
                shape_1d = feat['shape'][0]
                skip_1d = feat['skip'] if not isinstance(feat['skip'], list) else feat['skip'][0]
                if lazy:
                    poly_data_source = CustomDataNode(type='poly_data_source', id=f'{layer_id}')
                if style == 'multiplexed':
                    n_channel_per_ct_1d = int(n // 2 // shape_1d)
                    polyrelu = PolyRelu1D(shape_1d, order, skip_1d, n_channel_per_ct_1d)
                    if lazy:
                        layer_output_nodes = polyrelu.call_bsgs_mux_lazy(
                            feature_id_in_nodes, poly_data_source, layer_id
                        )
                    elif parameter_mode == 'encrypted_offline':
                        weight_ct = polyrelu.make_param_ct_nodes(
                            layer_id, len(feature_id_in_nodes), feature_id_in_nodes[0].level
                        )
                        layer_output_nodes = polyrelu.call_bsgs_param_ct(feature_id_in_nodes, weight_ct)
                    else:
                        weight_pt = polyrelu.make_pt_nodes(layer_id, len(feature_id_in_nodes))
                        layer_output_nodes = polyrelu.call_bsgs_mux(feature_id_in_nodes, weight_pt)
                else:
                    n_channel_per_ct_1d = int(n // 2 // shape_1d // skip_1d)
                    polyrelu = PolyRelu1D(shape_1d, order, skip_1d, n_channel_per_ct_1d)
                    if lazy:
                        layer_output_nodes = polyrelu.call_bsgs_skip_lazy(
                            feature_id_in_nodes, poly_data_source, layer_id
                        )
                    elif parameter_mode == 'encrypted_offline':
                        weight_ct = polyrelu.make_param_ct_nodes(
                            layer_id, len(feature_id_in_nodes), feature_id_in_nodes[0].level
                        )
                        layer_output_nodes = polyrelu.call_bsgs_param_ct(feature_id_in_nodes, weight_ct)
                    else:
                        weight_pt = polyrelu.make_pt_nodes(layer_id, len(feature_id_in_nodes))
                        layer_output_nodes = polyrelu.call_bsgs_skip(feature_id_in_nodes, weight_pt)
                if lazy:
                    input_args.append(Argument(f'{layer_id}', [poly_data_source]))

            else:  # dim == 2
                input_shape = feat['shape']
                polyrelu = PolyRelu2D(input_shape, order, skip, pack)
                if lazy:
                    poly_data_source = CustomDataNode(type='poly_data_source', id=f'{layer_id}')
                    layer_output_nodes = polyrelu.call_bsgs_lazy(feature_id_in_nodes, poly_data_source, layer_id)
                    input_args.append(Argument(f'{layer_id}', [poly_data_source]))
                elif parameter_mode == 'encrypted_offline':
                    weight_ct = polyrelu.make_param_ct_nodes(
                        layer_id, len(feature_id_in_nodes), feature_id_in_nodes[0].level
                    )
                    layer_output_nodes = polyrelu.call_bsgs_param_ct(feature_id_in_nodes, weight_ct)
                else:
                    weight_pt = polyrelu.make_pt_nodes(layer_id, len(feature_id_in_nodes))
                    layer_output_nodes = polyrelu.call_bsgs_feature2d(feature_id_in_nodes, weight_pt)

            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
            if not lazy:
                if parameter_mode == 'encrypted_offline':
                    for i in range(len(weight_ct)):
                        _append_parameter_arg(f'poly_reluw_{layer_id}_{i}', weight_ct[i])
                else:
                    for i in range(len(weight_pt)):
                        input_args.append(Argument(f'poly_reluw_{layer_id}_{i}', weight_pt[i]))

        elif layer_config['type'] == 'conv1d':
            if parameter_mode == 'encrypted_offline':
                _unsupported_encrypted_parameter_layer(layer_id, layer_config['type'])
            input_shape = config_info['feature'][layer_input_feature_ids[0]]['shape'][0]
            kernel_shape = layer_config['kernel_shape'][0]
            stride = layer_config['stride'][0]
            groups = layer_config['groups']
            skip_1d = skip[0] if isinstance(skip, list) else skip
            if style == 'multiplexed':
                n_channel_per_ct = math.ceil(n // 2 / input_shape)
                n_packed_in_channel = math.ceil(n_in_channel / n_channel_per_ct)
                n_packed_out_channel = math.ceil(n_out_channel / n_channel_per_ct)
                if groups == n_out_channel and groups != 1:
                    n_packed_ct = math.ceil(n_out_channel / n_channel_per_ct)
                    conv1d = MultiplexedDWConv1DPackedLayer(
                        n_out_channel,
                        input_shape,
                        kernel_shape,
                        stride,
                        skip_1d,
                        n_channel_per_ct,
                        n_packed_ct,
                    )
                    if lazy:
                        conv_data_source = CustomDataNode(type='conv_data_source', id=f'{layer_id}')
                        layer_output_nodes = conv1d.call_custom_compute(
                            feature_id_to_nodes_map[layer_input_feature_ids[0]], conv_data_source
                        )
                        feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
                        input_args.append(Argument(f'{layer_id}', [conv_data_source]))
                    else:
                        weight_pt, bias_pt, block_select_pt = conv1d.make_pt_nodes(layer_id)
                        layer_output_nodes = conv1d.call(
                            feature_id_to_nodes_map[layer_input_feature_ids[0]],
                            weight_pt,
                            bias_pt,
                            block_select_pt if block_select_pt else None,
                        )
                        feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
                        input_args.append(Argument(f'convw_{layer_id}', weight_pt))
                        input_args.append(Argument(f'convb_{layer_id}', bias_pt))
                        if block_select_pt:
                            input_args.append(Argument(f'convm_{layer_id}', block_select_pt))
                else:
                    conv1d = MultiplexedConv1DPackedLayer(
                        n_out_channel,
                        n_in_channel,
                        input_shape,
                        kernel_shape,
                        stride,
                        skip_1d,
                        n_channel_per_ct,
                        n_packed_in_channel,
                        n_packed_out_channel,
                    )
                    if lazy:
                        conv_data_source = CustomDataNode(type='conv_data_source', id=f'{layer_id}')
                        layer_output_nodes = conv1d.call_custom_compute(
                            feature_id_to_nodes_map[layer_input_feature_ids[0]], conv_data_source
                        )
                        feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
                        input_args.append(Argument(f'{layer_id}', [conv_data_source]))
                    else:
                        weight_pt, bias_pt, block_select_pt = conv1d.make_pt_nodes(layer_id)
                        layer_output_nodes = conv1d.call(
                            feature_id_to_nodes_map[layer_input_feature_ids[0]],
                            weight_pt,
                            bias_pt,
                            block_select_pt if block_select_pt else None,
                        )
                        feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
                        input_args.append(Argument(f'convw_{layer_id}', weight_pt))
                        input_args.append(Argument(f'convb_{layer_id}', bias_pt))
                        if block_select_pt:
                            input_args.append(Argument(f'convm_{layer_id}', block_select_pt))
            else:
                n_channel_per_ct = int(n // 2 // input_shape // skip_1d)
                n_pack_in_channel = math.ceil(n_in_channel / n_channel_per_ct)
                n_packed_out_channel = math.ceil(n_out_channel / (n_channel_per_ct * stride))
                conv1d = Conv1DPackedLayer(
                    n_out_channel,
                    n_in_channel,
                    input_shape,
                    kernel_shape,
                    stride,
                    skip_1d,
                    n_channel_per_ct,
                    n_pack_in_channel,
                    n_packed_out_channel,
                )
                if lazy:
                    conv_data_source = CustomDataNode(type='conv_data_source', id=f'{layer_id}')
                    layer_output_nodes = conv1d.call_custom_compute(
                        feature_id_to_nodes_map[layer_input_feature_ids[0]], conv_data_source
                    )
                    feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
                    input_args.append(Argument(f'{layer_id}', [conv_data_source]))
                else:
                    weight_pt, bias_pt = conv1d.make_pt_nodes(layer_id)
                    layer_output_nodes = conv1d.call(
                        feature_id_to_nodes_map[layer_input_feature_ids[0]], weight_pt, bias_pt
                    )
                    feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
                    input_args.append(Argument(f'convw_{layer_id}', weight_pt))
                    input_args.append(Argument(f'convb_{layer_id}', bias_pt))

        elif layer_config['type'] == 'mult_scalar':
            mult_scalar_layer = MultScalarLayer()
            input_nodes = feature_id_to_nodes_map[layer_input_feature_ids[0]]
            if lazy:
                conv_data_source = CustomDataNode(type='conv_data_source', id=f'{layer_id}')
                layer_output_nodes = mult_scalar_layer.call_custom_compute(input_nodes, conv_data_source)
                feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
                input_args.append(Argument(f'{layer_id}', [conv_data_source]))
            elif parameter_mode == 'encrypted_offline':
                weight_ct = mult_scalar_layer.make_param_ct_nodes(layer_id, len(input_nodes), level)
                layer_output_nodes = mult_scalar_layer.call_param_ct(input_nodes, weight_ct)
                feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
                _append_parameter_arg(f'mult_scalar_{layer_id}', weight_ct)
            else:
                pt = mult_scalar_layer.make_pt_nodes(layer_id, len(input_nodes))
                layer_output_nodes = mult_scalar_layer.call(input_nodes, pt)
                feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
                input_args.append(Argument(f'mult_scalar_{layer_id}', pt))

        elif layer_config['type'] == 'mult_coeff':
            raise ValueError(
                f"Layer '{layer_id}' has type 'mult_coeff', which should have been absorbed "
                f"into adjacent layers or converted to 'mult_scalar' during compilation."
            )

        elif layer_config['type'] == 'drop_level':
            input_fid = layer_input_feature_ids[0]
            output_fid = layer_output_feature_ids[0]
            level_in = config_info['feature'][input_fid]['level']
            level_out = config_info['feature'][output_fid]['level']
            drop_level_n = level_in - level_out
            input_nodes = feature_id_to_nodes_map[input_fid]
            if (
                task_config_info.get('mat_pack_style') == 'par_diagonal_pack'
                and config_info['feature'][output_fid].get('data_type') == 'feature_mat'
            ):
                consumed_count = _pdm_consumed_ct_count(output_fid, len(input_nodes))
                if consumed_count > len(input_nodes):
                    raise ValueError(
                        f"drop_level layer '{layer_id}' output '{output_fid}' needs {consumed_count} ciphertexts, "
                        f'but input has only {len(input_nodes)}'
                    )
                input_nodes = input_nodes[:consumed_count]
            layer_output_nodes = [drop_level(node, drop_level_n) for node in input_nodes]
            feature_id_to_nodes_map.update({output_fid: layer_output_nodes})

        elif layer_config['type'] == 'bootstrapping':
            layer_output_nodes = []
            for node in feature_id_to_nodes_map[layer_input_feature_ids[0]]:
                if node.level > 0:
                    node = drop_level(node, node.level)
                layer_output_nodes.append(bootstrap(node))
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif layer_config['type'] in ('add', 'add2d'):
            is_feature_mat_add = any(
                config_info['feature'][fid].get('data_type') == 'feature_mat' for fid in layer_input_feature_ids
            )
            if is_feature_mat_add:
                if not all(
                    config_info['feature'][fid].get('data_type') == 'feature_mat' for fid in layer_input_feature_ids
                ):
                    raise ValueError('feature_mat add expects all inputs to be feature_mat')
                n_heads = task_config_info.get('n_heads', 1)
                if n_heads <= 1:
                    raise ValueError('feature_mat add expects matrix-packed inputs with n_heads > 1')

                is_pdm_add = task_config_info.get('mat_pack_style') == 'par_diagonal_pack'
                block_size = None if is_pdm_add else _matmul_block_size()
                shape_per_head = None
                if not is_pdm_add:
                    shape_per_head = par_feature_shapes.get(layer_input_feature_ids[0])
                    if shape_per_head is None:
                        shape_per_head, _, _, _ = _feature_mat_ct_info(
                            input_feat0, n_heads, n // 2, feature_id=layer_input_feature_ids[0]
                        )
                        par_feature_shapes[layer_input_feature_ids[0]] = shape_per_head

                for input_fid in layer_input_feature_ids:
                    if input_fid not in feature_id_to_nodes_map:
                        feat = config_info['feature'][input_fid]
                        if is_pdm_add:
                            n_cts = _pdm_feature_ct_count(feat, input_fid, layer_id, layer_config['type'])
                            _register_feature_nodes(input_fid, n_cts, int(feat['level']))
                        else:
                            input_shape, _, _, n_cts = _feature_mat_ct_info(
                                feat, n_heads, n // 2, block_size=block_size, feature_id=input_fid
                            )
                            par_feature_shapes[input_fid] = input_shape
                            _register_feature_nodes(input_fid, n_cts, int(feat['level']))

                lhs = feature_id_to_nodes_map[layer_input_feature_ids[0]]
                rhs = feature_id_to_nodes_map[layer_input_feature_ids[1]]
                if len(lhs) == len(rhs):
                    add_layer = AddLayer()
                    layer_output_nodes = add_layer.call(lhs, rhs)
                elif is_pdm_add and len(lhs) > len(rhs) and len(lhs) % len(rhs) == 0:
                    layer_output_nodes = [add(ct, rhs[i % len(rhs)]) for i, ct in enumerate(lhs)]
                elif is_pdm_add and len(rhs) > len(lhs) and len(rhs) % len(lhs) == 0:
                    layer_output_nodes = [add(lhs[i % len(lhs)], ct) for i, ct in enumerate(rhs)]
                else:
                    raise ValueError(
                        f"feature_mat add layer '{layer_id}' input ciphertext counts do not match: "
                        f'{len(lhs)} vs {len(rhs)}'
                    )
                if is_pdm_add:
                    _remember_pdm_output_head_shape(layer_output_feature_ids[0])
                else:
                    par_feature_shapes[layer_output_feature_ids[0]] = shape_per_head
            else:
                add_layer = AddLayer()
                layer_output_nodes = add_layer.call(
                    feature_id_to_nodes_map[layer_input_feature_ids[0]],
                    feature_id_to_nodes_map[layer_input_feature_ids[1]],
                )
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif 'concat2d' in layer_config['type']:
            # Check if any input has n_channel not divisible by n_channel_per_ct
            input_n_channels = []
            input_packs = []
            has_uneven = False
            for input_fid in layer_input_feature_ids:
                feat = config_info['feature'][input_fid]
                n_ch = int(feat['channel'])
                input_n_channels.append(n_ch)
                input_packs.append(int(feat['pack_num']))
                if n_ch % pack != 0:
                    has_uneven = True

            mixed_pack = len(set(input_packs)) > 1

            if mixed_pack:
                # Mixed-pack path: inputs come with different pack_num; neither the
                # fast merge nor the uneven-same-pack path works. Use the general
                # mask + rotate + add repack routine, driven by each input's own
                # pack/skip and the output feature's pack/skip.
                out_feat = config_info['feature'][layer_output_feature_ids[0]]
                out_pack = int(out_feat['pack_num'])
                out_skip_raw = out_feat.get('skip', 1)
                out_skip = int(out_skip_raw[0] if isinstance(out_skip_raw, list) else out_skip_raw)
                input_skips = []
                for input_fid in layer_input_feature_ids:
                    sk = config_info['feature'][input_fid].get('skip', 1)
                    input_skips.append(int(sk[0] if isinstance(sk, list) else sk))

                concat_layer = ConcatLayer()
                input_node_lists = [feature_id_to_nodes_map[fid] for fid in layer_input_feature_ids]
                total_channels = sum(input_n_channels)
                mask_pts = [CkksPlaintextRingtNode(f'concat_mask_{layer_id}_{i}') for i in range(total_channels)]
                layer_output_nodes = concat_layer.call_multiple_inputs_mixed_pack(
                    input_node_lists,
                    input_n_channels,
                    input_packs,
                    input_skips,
                    out_pack,
                    out_skip,
                    mask_pts,
                )
                feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
                input_args.append(Argument(f'concat_mask_{layer_id}', mask_pts))
            elif has_uneven:
                # Uneven path: per-channel mask+rotate+add
                concat_layer = ConcatLayer()
                input_node_lists = [feature_id_to_nodes_map[fid] for fid in layer_input_feature_ids]
                first_feat = config_info['feature'][layer_input_feature_ids[0]]
                # For dim=0 features there is no H/W; synthesise a virtual 1D
                # layout so the 2D uneven algorithm still produces correct slot
                # offsets. With shape=[1, skip_scalar] and skip=[1, 1]:
                #   block_size = skip_scalar
                #   src_slot_base = local_ch * skip_scalar
                # which matches the physical packing of a dim=0 feature.
                if first_feat.get('dim', 2) == 0:
                    feat_skip = first_feat.get('skip', 1)
                    skip_scalar = int(feat_skip[0] if isinstance(feat_skip, list) else feat_skip)
                    input_shape = [1, skip_scalar]
                    use_skip = [1, 1]
                else:
                    input_shape = first_feat['shape']
                    use_skip = skip
                total_channels = sum(input_n_channels)

                # Create mask plaintext nodes for each global channel
                mask_pts = [CkksPlaintextRingtNode(f'concat_mask_{layer_id}_{i}') for i in range(total_channels)]

                layer_output_nodes = concat_layer.call_multiple_inputs_uneven(
                    input_node_lists, input_n_channels, pack, input_shape, use_skip, mask_pts
                )
                feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
                input_args.append(Argument(f'concat_mask_{layer_id}', mask_pts))
            else:
                # Fast path: concat is a runtime-only operation, just merge node lists
                layer_output_nodes = []
                for input_fid in layer_input_feature_ids:
                    layer_output_nodes.extend(feature_id_to_nodes_map[input_fid])
                feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif 'upsample_nearest' in layer_config['type']:
            input_shape = config_info['feature'][layer_input_feature_ids[0]]['shape']
            upsample_factor = layer_config['upsample_factor']
            upsample_layer = UpsampleNearestLayer(
                shape=input_shape,
                skip=skip,
                upsample_factor=upsample_factor,
                n_channel_per_ct=pack,
                level=level,
            )
            if lazy:
                upsample_data_source = CustomDataNode(type='upsample_data_source', id=f'{layer_id}')
                layer_output_nodes = upsample_layer.call_custom_compute(
                    feature_id_to_nodes_map[layer_input_feature_ids[0]],
                    upsample_data_source,
                    n_in_channel,
                )
                input_args.append(Argument(f'{layer_id}', [upsample_data_source]))
            else:
                select_tensor_pt = upsample_layer.make_pt_nodes(layer_id, n_in_channel)
                layer_output_nodes = upsample_layer.call(
                    feature_id_to_nodes_map[layer_input_feature_ids[0]],
                    select_tensor_pt,
                    n_in_channel,
                )
                input_args.append(Argument(f'upsample_select_pt_{layer_id}', select_tensor_pt))
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif 'avgpool' in layer_config['type']:
            input_shape = config_info['feature'][layer_input_feature_ids[0]]['shape']
            stride = layer_config['stride']
            is_adaptive = layer_config.get('is_adaptive_avgpool', True)
            is_big_size = layer_config.get('is_big_size', False)
            is_1d = layer_config['type'] == 'avgpool1d'
            if is_1d:
                skip_1d = skip[0] if isinstance(skip, list) else skip
                avgpool = Avgpool1DLayer(stride[0], input_shape[0], channel=n_in_channel, skip=skip_1d)
            else:
                avgpool = Avgpool2DLayer(stride, input_shape, channel=n_in_channel, skip=skip)
            if is_big_size:
                block_expansion = (
                    [
                        math.ceil(input_shape[0] / block_shape[0]),
                    ]
                    if is_1d
                    else [
                        math.ceil(input_shape[0] / block_shape[0]),
                        math.ceil(input_shape[1] / block_shape[1]),
                    ]
                )
                # Check if output < block_shape (repack needed)
                output_shape = [input_shape[i] // stride[i] for i in range(len(stride))]
                need_repack = any(output_shape[i] < block_shape[i] for i in range(len(stride)))
                repack_mask_pt = None
                if need_repack:
                    repack_mask_pt = CkksPlaintextRingtNode(f'repack_mask_{layer_id}')
                layer_output_nodes = avgpool.call_interleaved_avgpool(
                    feature_id_to_nodes_map[layer_input_feature_ids[0]],
                    block_expansion,
                    N=n,
                    repack_mask_pt=repack_mask_pt,
                    block_shape=block_shape if need_repack else None,
                )
                if repack_mask_pt is not None:
                    input_args.append(Argument(f'repack_mask_{layer_id}', [repack_mask_pt]))
            else:
                if is_adaptive:
                    # level_cost=0: only rotations + adds, normalization absorbed into adjacent layers
                    if style == 'ordinary':
                        layer_output_nodes = avgpool.call(feature_id_to_nodes_map[layer_input_feature_ids[0]])
                    else:
                        layer_output_nodes = avgpool.run_adaptive_avgpool(
                            feature_id_to_nodes_map[layer_input_feature_ids[0]], n=n
                        )
                else:
                    # level_cost=1: non-adaptive avgpool needs mult+rescale (select_tensor)
                    if is_1d and lazy:
                        avg_data_source = CustomDataNode(type='avg_data_source', id=f'{layer_id}')
                        layer_output_nodes = avgpool.call_custom_compute_multiplexed_avgpool(
                            feature_id_to_nodes_map[layer_input_feature_ids[0]],
                            avg_data_source,
                            n_in_channel,
                            n,
                        )
                        input_args.append(Argument(f'{layer_id}', [avg_data_source]))
                    elif is_1d:
                        n_channel_per_ct = int(math.ceil(n / 2 / input_shape[0]))
                        out_channels_per_ct = n_channel_per_ct * stride[0]
                        n_select_pt = min(n_in_channel, out_channels_per_ct)
                        select_tensor_pt = [
                            CkksPlaintextRingtNode(f'select_pt_{layer_id}_{i}') for i in range(n_select_pt)
                        ]
                        layer_output_nodes = avgpool.call_multiplexed_avgpool(
                            feature_id_to_nodes_map[layer_input_feature_ids[0]],
                            select_tensor_pt,
                            n_in_channel,
                            n_channel_per_ct,
                        )
                        input_args.append(Argument(f'select_tensor_pt_{layer_id}', select_tensor_pt))
                    elif lazy:
                        avg_data_source = CustomDataNode(type='avg_data_source', id=f'{layer_id}')
                        layer_output_nodes = avgpool.call_custom_compute_multiplexed_avgpool(
                            feature_id_to_nodes_map[layer_input_feature_ids[0]],
                            avg_data_source,
                            n_in_channel,
                            n,
                        )
                        input_args.append(Argument(f'{layer_id}', [avg_data_source]))
                    else:
                        n_channel_per_ct = int(math.ceil(n / 2 / (input_shape[0] * input_shape[1])))
                        out_channels_per_ct = n_channel_per_ct * stride[0] * stride[1]
                        n_select_pt = min(n_in_channel, out_channels_per_ct)
                        select_tensor_pt = [
                            CkksPlaintextRingtNode(f'select_pt_{layer_id}_{i}') for i in range(n_select_pt)
                        ]
                        layer_output_nodes = avgpool.call_multiplexed_avgpool(
                            feature_id_to_nodes_map[layer_input_feature_ids[0]],
                            select_tensor_pt,
                            n_in_channel,
                            n_channel_per_ct,
                        )
                        input_args.append(Argument(f'select_tensor_pt_{layer_id}', select_tensor_pt))

            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif layer_config['type'] == 'fc0':
            if 'special_info' not in config_info['feature'][layer_input_feature_ids[0]]:
                # call_skip_0d path — matching test_fc_fc_feature0d Layer 1
                skip_0d = config_info['feature'][layer_input_feature_ids[0]]['skip']
                n_channel_per_ct = int(n // 2 // skip_0d)
                pack_0d = n_channel_per_ct
                n_packed_in_feature = math.ceil(n_in_channel / n_channel_per_ct)
                n_packed_out_feature = math.ceil(n_out_channel / n_channel_per_ct)
                fc_layer = DensePackedLayer(
                    n_out_channel,
                    n_in_channel,
                    [1, 1],
                    [1, 1],
                    pack_0d,
                    n_packed_in_feature,
                    n_packed_out_feature,
                )
                if lazy:
                    dense_data_source = CustomDataNode(type='fc_data_source', id=f'{layer_id}')
                    input_args.append(Argument(f'{layer_id}', [dense_data_source]))
                    layer_output_nodes = fc_layer.call_skip_0d_custom_compute(
                        feature_id_to_nodes_map[layer_input_feature_ids[0]], dense_data_source, skip_0d
                    )
                elif parameter_mode == 'encrypted_offline':
                    weight_ct, bias_ct = fc_layer.make_param_ct_nodes_skip_0d(layer_id, level)
                    _append_parameter_arg(f'densew_{layer_id}', weight_ct)
                    _append_parameter_arg(f'denseb_{layer_id}', bias_ct)
                    layer_output_nodes = fc_layer.call_param_ct_skip_0d(
                        feature_id_to_nodes_map[layer_input_feature_ids[0]],
                        weight_ct,
                        bias_ct,
                        skip_0d,
                        input_is_plaintext=_feature_input_is_plaintext(layer_input_feature_ids[0]),
                    )
                else:
                    weight_pt, bias_pt = fc_layer.make_pt_nodes_skip_0d(layer_id)
                    input_args.append(Argument(f'densew_{layer_id}', weight_pt))
                    input_args.append(Argument(f'denseb_{layer_id}', bias_pt))
                    layer_output_nodes = fc_layer.call_skip_0d(
                        feature_id_to_nodes_map[layer_input_feature_ids[0]], weight_pt, bias_pt, skip_0d
                    )
            else:
                special_info = config_info['feature'][layer_input_feature_ids[0]]['special_info']
                special_shape = special_info['shape']
                special_skip = special_info['skip']
                invalid_fill = special_info.get('invalid_fill', [1, 1])
                if len(special_shape) == 1:
                    # 1D multiplexed: special_shape=[L], special_skip=[skip]
                    shape_1d = int(special_shape[0])
                    skip_1d = int(special_skip[0])
                    invalid_fill_1d = int(invalid_fill[0])
                    block_stride = skip_1d  # skip already contains invalid_fill
                    block_size = shape_1d * block_stride
                    n_block_per_ct = int(n // 2) // block_size
                    valid_sub = skip_1d // invalid_fill_1d
                    n_channel_per_ct_1d = n_block_per_ct * valid_sub
                    dense = DensePackedLayer(
                        n_out_channel,
                        n_in_channel,
                        [shape_1d, 1],
                        [skip_1d, 1],
                        n_channel_per_ct_1d,
                        math.ceil(n_in_channel / n_channel_per_ct_1d),
                        math.ceil(n_out_channel / n_block_per_ct),
                        invalid_fill=[invalid_fill_1d, 1],
                    )
                    if lazy:
                        dense_data_source = CustomDataNode(type='fc_data_source', id=f'{layer_id}')
                        input_args.append(Argument(f'{layer_id}', [dense_data_source]))
                        layer_output_nodes = dense.call_1d_multiplexed_custom_compute(
                            feature_id_to_nodes_map[layer_input_feature_ids[0]], dense_data_source, n
                        )
                    elif parameter_mode == 'encrypted_offline':
                        weight_ct, bias_ct = dense.make_param_ct_nodes_1d_multiplexed(layer_id, n, level)
                        _append_parameter_arg(f'densew_{layer_id}', weight_ct)
                        _append_parameter_arg(f'denseb_{layer_id}', bias_ct)
                        layer_output_nodes = dense.call_param_ct_1d_multiplexed(
                            feature_id_to_nodes_map[layer_input_feature_ids[0]],
                            weight_ct,
                            bias_ct,
                            n,
                            input_is_plaintext=_feature_input_is_plaintext(layer_input_feature_ids[0]),
                        )
                    else:
                        weight_pt, bias_pt = dense.make_pt_nodes_1d_multiplexed(layer_id, n)
                        input_args.append(Argument(f'densew_{layer_id}', weight_pt))
                        input_args.append(Argument(f'denseb_{layer_id}', bias_pt))
                        layer_output_nodes = dense.call_1d_multiplexed(
                            feature_id_to_nodes_map[layer_input_feature_ids[0]], weight_pt, bias_pt, n
                        )
                else:
                    # 2D multiplexed: special_shape=[H, W], special_skip=[s0, s1]
                    dense = DensePackedLayer(
                        n_out_channel,
                        n_in_channel,
                        special_shape,
                        special_skip,
                        math.ceil(n // 2 / (special_shape[0] * special_skip[0] * special_shape[1] * special_skip[1])),
                        n_in_channel,
                        n_out_channel,
                        invalid_fill=invalid_fill,
                    )
                    if lazy:
                        dense_data_source = CustomDataNode(type='fc_data_source', id=f'{layer_id}')
                        input_args.append(Argument(f'{layer_id}', [dense_data_source]))
                        layer_output_nodes = dense.call_multiplexed_custom_compute(
                            feature_id_to_nodes_map[layer_input_feature_ids[0]], dense_data_source, n
                        )
                    elif parameter_mode == 'encrypted_offline':
                        weight_ct, bias_ct = dense.make_param_ct_nodes_multiplexed(layer_id, n, level)
                        _append_parameter_arg(f'densew_{layer_id}', weight_ct)
                        _append_parameter_arg(f'denseb_{layer_id}', bias_ct)
                        layer_output_nodes = dense.call_param_ct_multiplexed(
                            feature_id_to_nodes_map[layer_input_feature_ids[0]],
                            weight_ct,
                            bias_ct,
                            n,
                            input_is_plaintext=_feature_input_is_plaintext(layer_input_feature_ids[0]),
                        )
                    else:
                        weight_pt, bias_pt = dense.make_pt_nodes_multiplexed(layer_id, n)
                        input_args.append(Argument(f'densew_{layer_id}', weight_pt))
                        input_args.append(Argument(f'denseb_{layer_id}', bias_pt))
                        layer_output_nodes = dense.call_multiplexed(
                            feature_id_to_nodes_map[layer_input_feature_ids[0]], weight_pt, bias_pt, n
                        )
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif layer_config['type'] == 'partranspose':
            n_heads = task_config_info.get('n_heads', 1)
            input_fid = layer_input_feature_ids[0]
            feat_in = config_info['feature'][input_fid]
            shape_per_head = par_feature_shapes.get(input_fid)
            if shape_per_head is None:
                shape_per_head = _par_input_shape(feat_in, n_heads, split_rows=False, feature_id=input_fid)
                par_feature_shapes[input_fid] = shape_per_head
            block_size = _matmul_block_size()

            partranspose_layer = ParBlockColMajorTranspose(shape_per_head, block_size, n_heads, n // 2)
            G = partranspose_layer.G
            n_cts_in = partranspose_layer.num_blocks * G

            # Register par-type input (deferred from the top of the loop)
            if input_fid not in feature_id_to_nodes_map:
                x = _make_feature_nodes(input_fid, n_cts_in, level)
                feature_id_to_nodes_map[input_fid] = x
                input_args.append(Argument(input_fid, x))

            data_source = CustomDataNode(type='partranspose_data_source', id=f'{layer_id}')
            input_args.append(Argument(f'{layer_id}', [data_source]))
            layer_output_nodes = partranspose_layer.call_custom_compute(
                feature_id_to_nodes_map[input_fid],
                data_source,
            )
            feature_id_to_nodes_map[layer_output_feature_ids[0]] = layer_output_nodes
            par_feature_shapes[layer_output_feature_ids[0]] = (shape_per_head[1], shape_per_head[0])

        elif layer_config['type'] == 'pcm_add_pt':
            n_heads = task_config_info.get('n_heads', 1)
            feat_in = config_info['feature'][layer_input_feature_ids[0]]
            shape_full = tuple(feat_in['shape'])
            block_size = _matmul_block_size()

            add_pt_layer = ParBlockColMajorAddPt(shape_full, block_size, n_heads, n // 2)

            n_cts_in = add_pt_layer.total_cts

            input_fid = layer_input_feature_ids[0]
            if input_fid not in feature_id_to_nodes_map:
                x = _make_feature_nodes(input_fid, n_cts_in, level)
                feature_id_to_nodes_map[input_fid] = x
                input_args.append(Argument(input_fid, x))

            data_source = CustomDataNode(type='pcm_add_pt_data_source', id=f'{layer_id}')
            input_args.append(Argument(f'{layer_id}', [data_source]))
            layer_output_nodes = add_pt_layer.call_custom_compute(
                feature_id_to_nodes_map[input_fid],
                data_source,
            )
            feature_id_to_nodes_map[layer_output_feature_ids[0]] = layer_output_nodes

        elif layer_config['type'] == 'pdmtranspose':
            n_heads = _require_positive_task_config_int('n_heads', layer_id, layer_config['type'])
            input_fid = layer_input_feature_ids[0]
            feat_in = config_info['feature'][input_fid]
            shape = _pdm_transpose_shape(feat_in, input_fid, layer_id, layer_config['type'])
            layer = ParLowerDiagTranspose(shape, n_heads, _head_dim(layer_id, layer_config['type']), n // 2)

            if input_fid not in feature_id_to_nodes_map:
                _register_feature_nodes(input_fid, layer.m_c, level)

            data_source = CustomDataNode(type='pdmtranspose_data_source', id=f'{layer_id}')
            input_args.append(Argument(f'{layer_id}', [data_source]))
            layer_output_nodes = layer.call_custom_compute(feature_id_to_nodes_map[input_fid], data_source)
            feature_id_to_nodes_map[layer_output_feature_ids[0]] = layer_output_nodes
            _remember_pdm_output_head_shape(layer_output_feature_ids[0])

        elif layer_config['type'] == 'pdm_add_pt':
            input_fid = layer_input_feature_ids[0]
            feat_in = config_info['feature'][input_fid]
            layer = _pdm_single_input_total_ct_layer(layer_config['type'], feat_in, input_fid, layer_id, layer_config)

            if input_fid not in feature_id_to_nodes_map:
                _register_feature_nodes(input_fid, layer.total_cts, level)

            data_source = CustomDataNode(type='pdm_add_pt_data_source', id=f'{layer_id}')
            input_args.append(Argument(f'{layer_id}', [data_source]))
            layer_output_nodes = layer.call_custom_compute(feature_id_to_nodes_map[input_fid], data_source)
            feature_id_to_nodes_map[layer_output_feature_ids[0]] = layer_output_nodes
            _remember_pdm_output_head_shape(layer_output_feature_ids[0])

        elif layer_config['type'] == 'pdmpcmm':
            n_heads = _require_positive_task_config_int('n_heads', layer_id, layer_config['type'])
            input_fid = layer_input_feature_ids[0]
            feat_in = config_info['feature'][input_fid]
            feat_out = config_info['feature'][layer_output_feature_ids[0]]
            shape_X_T, W_T_shape = _pdmpcmm_shapes(layer_config, feat_in, feat_out, input_fid, layer_id)

            layer = ParLowerDiagPCMM(
                shape_X_T,
                W_T_shape,
                n_heads,
                _head_dim(layer_id, layer_config['type']),
                n // 2,
                has_bias='bias_path' in layer_config,
            )
            n_cts_in = layer.K_col * layer.m_c

            if input_fid not in feature_id_to_nodes_map:
                _register_feature_nodes(input_fid, n_cts_in, level)

            data_source = CustomDataNode(type='pdmpcmm_data_source', id=f'{layer_id}')
            input_args.append(Argument(f'{layer_id}', [data_source]))
            layer_output_nodes = layer.call_custom_compute(feature_id_to_nodes_map[input_fid], data_source)
            feature_id_to_nodes_map[layer_output_feature_ids[0]] = layer_output_nodes
            _remember_pdm_output_head_shape(layer_output_feature_ids[0])

        elif layer_config['type'] == 'pdmccmm':
            n_heads = _require_positive_task_config_int('n_heads', layer_id, layer_config['type'])
            fid0 = layer_input_feature_ids[0]
            fid1 = layer_input_feature_ids[1]
            feat0 = config_info['feature'][fid0]
            feat1 = config_info['feature'][fid1]
            layer, fid_A, fid_B = _make_pdmccmm_layer(layer_id, layer_config['type'], fid0, fid1, feat0, feat1, n_heads)

            feat_A = config_info['feature'][fid_A]
            feat_B = config_info['feature'][fid_B]
            if fid_A not in feature_id_to_nodes_map:
                _register_feature_nodes(fid_A, layer.m_c, int(feat_A['level']))
            if fid_B not in feature_id_to_nodes_map:
                _register_feature_nodes(fid_B, layer.m_c if layer.is_kqt else layer.n_c, int(feat_B['level']))

            data_source = CustomDataNode(type='pdmccmm_data_source', id=f'{layer_id}')
            input_args.append(Argument(f'{layer_id}', [data_source]))
            layer_output_nodes = layer.call_custom_compute(
                feature_id_to_nodes_map[fid_A],
                feature_id_to_nodes_map[fid_B],
                data_source,
            )
            feature_id_to_nodes_map[layer_output_feature_ids[0]] = layer_output_nodes
            _remember_pdm_output_head_shape(layer_output_feature_ids[0])

        elif layer_config['type'] == 'parcpmm':
            n_heads = task_config_info.get('n_heads', 1)
            feat_in = config_info['feature'][layer_input_feature_ids[0]]
            feat_out = config_info['feature'][layer_output_feature_ids[0]]
            shape_A_full = tuple(feat_in['shape'])
            block_size = _matmul_block_size()
            shape_A = _par_input_shape(feat_in, n_heads, split_rows=False, feature_id=layer_input_feature_ids[0])
            if shape_A[1] > block_size:
                raise ValueError(f"parcpmm layer '{layer_id}' input per-head width exceeds matmul_block_size")
            W_shape = (shape_A_full[1], feat_out['shape'][1])

            has_bias = 'bias_path' in layer_config
            parcpmm_layer = ParBlockColMajorCPMM(shape_A, W_shape, block_size, n_heads, n // 2, has_bias=has_bias)

            # Compute input CT count based on mode
            if parcpmm_layer.mode == 'REDUCE':
                n_cts_in = parcpmm_layer.K * parcpmm_layer.num_block_rows_A * parcpmm_layer.G
            else:
                n_cts_in = parcpmm_layer.num_block_rows_A * parcpmm_layer.G

            # Register par-type input (deferred from the top of the loop)
            input_fid = layer_input_feature_ids[0]
            if input_fid not in feature_id_to_nodes_map:
                x = _make_feature_nodes(input_fid, n_cts_in, level)
                feature_id_to_nodes_map[input_fid] = x
                input_args.append(Argument(input_fid, x))

            data_source = CustomDataNode(type='parcpmm_data_source', id=f'{layer_id}')
            input_args.append(Argument(f'{layer_id}', [data_source]))
            layer_output_nodes = parcpmm_layer.call_custom_compute(
                feature_id_to_nodes_map[input_fid],
                data_source,
            )
            feature_id_to_nodes_map[layer_output_feature_ids[0]] = layer_output_nodes

        elif layer_config['type'] == 'parccmm':
            n_heads = task_config_info.get('n_heads', 1)
            fid_A = layer_input_feature_ids[0]
            fid_B = layer_input_feature_ids[1]
            feat_A = config_info['feature'][fid_A]
            feat_B = config_info['feature'][fid_B]

            shape_A = par_feature_shapes.get(fid_A)
            if shape_A is None:
                shape_A = _par_input_shape(feat_A, n_heads, split_rows=False, feature_id=fid_A)
                par_feature_shapes[fid_A] = shape_A
            block_size = _matmul_block_size()

            shape_B = par_feature_shapes.get(fid_B)
            if shape_B is None:
                # Raw parccmm RHS is stored as [H*N, H*P]; if it is produced by
                # partranspose, par_feature_shapes already contains [N, P].
                shape_B = _par_input_shape(feat_B, n_heads, split_rows=True, feature_id=fid_B)
                par_feature_shapes[fid_B] = shape_B

            parccmm_layer = ParBlockColMajorCCMM(shape_A, shape_B, block_size, n_heads, n // 2)
            G = parccmm_layer.G

            # Register par-type inputs (deferred from the top of the loop)
            for idx, input_fid in enumerate(layer_input_feature_ids):
                if input_fid not in feature_id_to_nodes_map:
                    feat = config_info['feature'][input_fid]
                    shape_per_head = par_feature_shapes.get(input_fid)
                    if shape_per_head is None:
                        shape_per_head = _par_input_shape(feat, n_heads, split_rows=(idx == 1), feature_id=input_fid)
                        par_feature_shapes[input_fid] = shape_per_head
                    n_cts = _par_ct_count(shape_per_head, block_size, G)
                    x = _make_feature_nodes(input_fid, n_cts, int(feat['level']))
                    feature_id_to_nodes_map[input_fid] = x
                    input_args.append(Argument(input_fid, x))

            data_source = CustomDataNode(type='parccmm_data_source', id=f'{layer_id}')
            input_args.append(Argument(f'{layer_id}', [data_source]))
            layer_output_nodes = parccmm_layer.call_custom_compute(
                feature_id_to_nodes_map[fid_A],
                feature_id_to_nodes_map[fid_B],
                data_source,
            )
            feature_id_to_nodes_map[layer_output_feature_ids[0]] = layer_output_nodes
            par_feature_shapes[layer_output_feature_ids[0]] = (shape_A[0], shape_B[1])

        elif layer_config['type'] == 'pcmstats':
            n_heads = task_config_info.get('n_heads', 1)
            input_fid = layer_input_feature_ids[0]
            feat_in = config_info['feature'][input_fid]
            shape = tuple(feat_in['shape'])
            block_size = _matmul_block_size()
            layer = ParBlockColMajorLNStats(shape=shape, block_size=block_size, n_heads=n_heads, n_slot=n // 2)

            if input_fid not in feature_id_to_nodes_map:
                x = _make_feature_nodes(input_fid, layer.total_cts, level)
                feature_id_to_nodes_map[input_fid] = x
                input_args.append(Argument(input_fid, x))

            data_source = CustomDataNode(type='layernorm_data_source', id=f'{layer_id}')
            input_args.append(Argument(f'{layer_id}', [data_source]))
            layer_output_nodes = layer.call_custom_compute(feature_id_to_nodes_map[input_fid], data_source)
            feature_id_to_nodes_map[layer_output_feature_ids[0]] = layer_output_nodes
            par_feature_shapes[layer_output_feature_ids[0]] = (shape[0], 1)

        elif layer_config['type'] == 'pcmcenter':
            n_heads = task_config_info.get('n_heads', 1)
            input_fid = layer_input_feature_ids[0]
            feat_in = config_info['feature'][input_fid]
            shape = tuple(feat_in['shape'])
            block_size = _matmul_block_size()
            layer = ParBlockColMajorLNXCentered(shape=shape, block_size=block_size, n_heads=n_heads, n_slot=n // 2)

            if input_fid not in feature_id_to_nodes_map:
                x = _make_feature_nodes(input_fid, layer.total_cts, level)
                feature_id_to_nodes_map[input_fid] = x
                input_args.append(Argument(input_fid, x))

            data_source = CustomDataNode(type='layernorm_data_source', id=f'{layer_id}')
            input_args.append(Argument(f'{layer_id}', [data_source]))
            layer_output_nodes = layer.call_custom_compute(feature_id_to_nodes_map[input_fid], data_source)
            feature_id_to_nodes_map[layer_output_feature_ids[0]] = layer_output_nodes
            par_feature_shapes[layer_output_feature_ids[0]] = (shape[0], shape[1] // n_heads)

        elif layer_config['type'] == 'pcminit':
            n_heads = task_config_info.get('n_heads', 1)
            feat_in = config_info['feature'][layer_input_feature_ids[0]]
            block_size = _matmul_block_size()
            layer = ParBlockColMajorLNMinimaxInit(block_size=block_size, n_slot=n // 2)

            data_source = CustomDataNode(type='layernorm_data_source', id=f'{layer_id}')
            input_args.append(Argument(f'{layer_id}', [data_source]))
            layer_output_nodes = layer.call_custom_compute(
                feature_id_to_nodes_map[layer_input_feature_ids[0]], data_source
            )
            feature_id_to_nodes_map[layer_output_feature_ids[0]] = layer_output_nodes
            par_feature_shapes[layer_output_feature_ids[0]] = par_feature_shapes.get(layer_input_feature_ids[0])

        elif layer_config['type'] == 'pcmgs':
            n_heads = task_config_info.get('n_heads', 1)
            feat_in = config_info['feature'][layer_input_feature_ids[0]]
            block_size = _matmul_block_size()
            layer = ParBlockColMajorLNGoldschmidt(block_size=block_size, n_slot=n // 2)

            data_source = CustomDataNode(type='layernorm_data_source', id=f'{layer_id}')
            input_args.append(Argument(f'{layer_id}', [data_source]))
            layer_output_nodes = layer.call_custom_compute(
                feature_id_to_nodes_map[layer_input_feature_ids[0]],
                feature_id_to_nodes_map[layer_input_feature_ids[1]],
                data_source,
            )
            feature_id_to_nodes_map[layer_output_feature_ids[0]] = layer_output_nodes
            par_feature_shapes[layer_output_feature_ids[0]] = par_feature_shapes.get(layer_input_feature_ids[0])

        elif layer_config['type'] == 'pcmaffine':
            n_heads = task_config_info.get('n_heads', 1)
            feat_in = config_info['feature'][layer_input_feature_ids[0]]
            shape = tuple(feat_in['shape'])
            block_size = _matmul_block_size()
            layer = ParBlockColMajorLNAffine(shape=shape, block_size=block_size, n_heads=n_heads, n_slot=n // 2)

            data_source = CustomDataNode(type='layernorm_data_source', id=f'{layer_id}')
            input_args.append(Argument(f'{layer_id}', [data_source]))
            layer_output_nodes = layer.call_custom_compute(
                feature_id_to_nodes_map[layer_input_feature_ids[0]],
                feature_id_to_nodes_map[layer_input_feature_ids[1]],
                data_source,
            )
            feature_id_to_nodes_map[layer_output_feature_ids[0]] = layer_output_nodes
            par_feature_shapes[layer_output_feature_ids[0]] = (shape[0], shape[1] // n_heads)

        elif layer_config['type'] == 'pcmgamma':
            n_heads = _require_positive_task_config_int('n_heads', layer_id, layer_config['type'])
            input_fid = layer_input_feature_ids[0]
            feat_in = config_info['feature'][input_fid]
            shape = tuple(feat_in['shape'])
            K = int(layer_config.get('K', layer_config.get('k', 1)))
            block_size = _matmul_block_size()
            if shape[1] % (K * n_heads) != 0:
                raise ValueError(
                    f"pcmgamma layer '{layer_id}' expects full feature shape with cols divisible by K * n_heads"
                )
            layer = ParBlockColMajorPolyActRNGamma(
                shape=shape,
                block_size=block_size,
                n_heads=n_heads,
                n_slot=n // 2,
                K=K,
            )

            if input_fid not in feature_id_to_nodes_map:
                x = _make_feature_nodes(input_fid, layer.total_cts, level)
                feature_id_to_nodes_map[input_fid] = x
                input_args.append(Argument(input_fid, x))

            data_source = CustomDataNode(type='polyactrn_gamma_data_source', id=f'{layer_id}')
            input_args.append(Argument(f'{layer_id}', [data_source]))
            layer_output_nodes = layer.call_custom_compute(feature_id_to_nodes_map[input_fid], data_source)
            feature_id_to_nodes_map[layer_output_feature_ids[0]] = layer_output_nodes
            par_feature_shapes[layer_output_feature_ids[0]] = par_feature_shapes.get(
                input_fid, (shape[0], shape[1] // (K * n_heads))
            )

        elif layer_config['type'] == 'pcmpoly':
            n_heads = _require_positive_task_config_int('n_heads', layer_id, layer_config['type'])
            input_fid = layer_input_feature_ids[0]
            feat_in = config_info['feature'][input_fid]
            shape = tuple(feat_in['shape'])
            K = int(layer_config.get('K', layer_config.get('k', 1)))
            degree = layer_config.get('degree', layer_config.get('order', 2))
            block_size = _matmul_block_size()
            if shape[1] % (K * n_heads) != 0:
                raise ValueError(
                    f"pcmpoly layer '{layer_id}' expects full feature shape with cols divisible by K * n_heads"
                )
            layer = ParBlockColMajorPolyActRNPoly(
                shape=shape,
                block_size=block_size,
                n_heads=n_heads,
                n_slot=n // 2,
                degree=degree,
                K=K,
            )

            if input_fid not in feature_id_to_nodes_map:
                x = _make_feature_nodes(input_fid, layer.total_cts, level)
                feature_id_to_nodes_map[input_fid] = x
                input_args.append(Argument(input_fid, x))

            data_source = CustomDataNode(type='polyactrn_poly_data_source', id=f'{layer_id}')
            input_args.append(Argument(f'{layer_id}', [data_source]))
            layer_output_nodes = layer.call_custom_compute(feature_id_to_nodes_map[input_fid], data_source)
            feature_id_to_nodes_map[layer_output_feature_ids[0]] = layer_output_nodes
            par_feature_shapes[layer_output_feature_ids[0]] = par_feature_shapes.get(
                input_fid, (shape[0], shape[1] // (K * n_heads))
            )

        elif layer_config['type'] in {
            'pdmstats',
            'pdmcenter',
            'pdminit',
            'pdmgs',
            'pdmaffine',
            'pdmgamma',
            'pdmpoly',
            'pdmupperaddpt',
            'pdmupperpoly',
            'pdmmulsquare',
            'pdmheadcolsum',
            'pdminvinit',
            'pdminviter',
            'pdmctmul',
            'pdmupperpolymultct',
        }:
            layer_type = layer_config['type']
            n_heads = _require_positive_task_config_int('n_heads', layer_id, layer_type)
            input_fid = layer_input_feature_ids[0]
            feat_in = config_info['feature'][input_fid]
            head_shape = _pdm_head_shape(feat_in, input_fid, layer_id, layer_type)
            shape = _pdm_upper_shape(feat_in, head_shape)
            layer = _pdm_upper_layer(layer_type, shape, head_shape, n_heads, layer_id, layer_config)

            for fid in layer_input_feature_ids:
                feat = config_info['feature'][fid]
                if fid not in feature_id_to_nodes_map:
                    _register_feature_nodes(fid, layer.total_cts, int(feat['level']))

            if layer_type in {'pdmstats', 'pdmcenter', 'pdminit', 'pdmgs', 'pdmaffine'}:
                data_source_type = 'layernorm_data_source'
            elif layer_type == 'pdmgamma':
                data_source_type = 'polyactrn_gamma_data_source'
            elif layer_type == 'pdmpoly':
                data_source_type = 'polyactrn_poly_data_source'
            else:
                data_source_type = 'pdmupper_data_source'

            data_source = CustomDataNode(type=data_source_type, id=f'{layer_id}')
            input_args.append(Argument(f'{layer_id}', [data_source]))
            if layer_type in {'pdmgs', 'pdmaffine', 'pdminviter', 'pdmctmul', 'pdmupperpolymultct'}:
                layer_output_nodes = layer.call_custom_compute(
                    feature_id_to_nodes_map[layer_input_feature_ids[0]],
                    feature_id_to_nodes_map[layer_input_feature_ids[1]],
                    data_source,
                )
            else:
                layer_output_nodes = layer.call_custom_compute(feature_id_to_nodes_map[input_fid], data_source)
            feature_id_to_nodes_map[layer_output_feature_ids[0]] = layer_output_nodes
            _remember_pdm_output_head_shape(layer_output_feature_ids[0])

        else:
            raise ValueError(f'Unsupported layer type: {layer_config["type"]}')

    output_args = [Argument(output_id, feature_id_to_nodes_map[output_id]) for output_id in task_output_feature_ids]

    process_custom_task(
        input_args=input_args,
        output_args=output_args,
        offline_input_args=offline_input_args,
        output_instruction_path=output_dir,
        fpga_acc=False,
    )


if __name__ == '__main__':
    if hasattr(sys, 'frozen'):
        os.chdir(os.path.dirname(sys.executable))
    else:
        os.chdir(os.path.dirname(__file__))

    parser = argparse.ArgumentParser(description='FPGA config generator.')
    parser.add_argument('task_path', type=str, help='Path of the server directory')
    parser.add_argument(
        '--lazy', action='store_true', help='Use lazy weight generation (encode_pt custom compute nodes)'
    )
    parser.add_argument(
        '--parameter-mode',
        choices=sorted(PARAMETER_MODES),
        default=None,
        help='Parameter handling mode. Defaults to plaintext_lazy when --lazy is set, otherwise plaintext_eager.',
    )
    parser.add_argument(
        '--input-mode',
        choices=sorted(INPUT_MODES),
        default='ciphertext',
        help='Graph input mode for generated task signature.',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Directory for mega_ag.json and task_signature.json. Defaults to task_path.',
    )
    args = parser.parse_args()

    task_path = args.task_path
    with open(os.path.join(task_path, 'task_config.json'), 'r', encoding='utf-8') as file:
        config = json.load(file)

    for _, is_fpga in config['server_task'].items():
        if is_fpga['enable_fpga']:
            gen_custom_task(
                task_path,
                use_gpu=True,
                lazy=args.lazy,
                parameter_mode=args.parameter_mode,
                input_mode=args.input_mode,
                output_dir=args.output_dir,
            )
