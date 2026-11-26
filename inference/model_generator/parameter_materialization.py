# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Signature-driven numerical parameters for the browser Provisioner.

The slot formulas follow DensePackedLayer, MultiplexedConv2DPackedLayer,
PolyRelu0D/2D, MultScalarLayer and interleaved_seed.h. Cryptography belongs to
Go/WASM; this module reads bounded H5 ranges for one slot vector.
"""

import json
import math
from pathlib import Path
import re

import h5py
import numpy as np


def parameter_identity(signature, layers):
    source = signature.get('source_id', re.sub(r'__L\d+.*$', '', signature['id']))
    for prefix, kind in (
        ('convw_', 'weight'),
        ('convb_', 'bias'),
        ('convm_', 'mask'),
        ('densew_', 'weight'),
        ('denseb_', 'bias'),
        ('mult_scalar_', 'mult_scalar_weight'),
    ):
        if source.startswith(prefix) and source[len(prefix) :] in layers:
            return source, source[len(prefix) :], kind, None
    for layer in sorted(layers, key=len, reverse=True):
        prefix = 'poly_reluw_' + layer + '_'
        if source.startswith(prefix) and source[len(prefix) :].isdigit():
            return source, layer, 'poly_coeff', int(source[len(prefix) :])
    raise ValueError(f'Unsupported parameter signature: {signature["id"]}')


def parameter_layout(graph, layer_id, signature):
    layer = graph['layer'][layer_id]
    if signature['id'].startswith('dense'):
        return layer.get('dense_seed_tiles')
    if signature['id'].startswith('poly_reluw_'):
        return layer.get('poly_seed_layout')
    if signature['id'].startswith('convw_') and 'seed_weight_subgrid_stride' in layer:
        stride = layer['seed_weight_subgrid_stride']
        return dict(
            kind=layer.get('seed_weight_layout', 'kernel'),
            stride=stride,
            group_size=math.prod(stride),
            kernel_shape=layer['kernel_shape'],
            storage_shape=signature['size'],
            input_geometry=graph['feature'][layer['feature_input'][0]],
            output_geometry=graph['feature'][layer['feature_output'][0]],
        )
    return None


class ParameterMaterializer:
    def __init__(self, task_directory):
        task = Path(task_directory)
        config = json.loads((task / 'provisioner/task_config.json').read_text())
        if config.get('deployment_mode') != 'server_provisioned_runner' or config['pack_style'] != 'multiplexed':
            raise ValueError('Provisioner requires server_provisioned_runner with multiplexed packing')
        self.config = config
        self.graph = json.loads((task / 'provisioner/nn_layers_ct_0.json').read_text())
        self.signature = json.loads((task / 'runner/task_signature.json').read_text())
        if (
            config.get('mat_pack_style', '') not in ('', 'par_block_col_major')
            or self.signature.get('algorithm') != 'CKKS'
        ):
            raise ValueError('Unsupported parameter profile/layout')
        self.parameters = json.loads((task / 'provisioner/ckks_parameter.json').read_text())
        locked_profile = json.loads(
            (Path(__file__).resolve().parents[1] / 'lattisense/frontend/parameter.json').read_text()
        )['CKKS']['15']
        for profile in self.parameters.values():
            if (
                profile.get('ring_type') != 1
                or profile.get('param_name') != 'PN15Default'
                or profile.get('poly_modulus_degree') != 32768
                or profile.get('n_slots') != 32768
                or any(profile.get(key) != locked_profile[key] for key in ('q', 'p', 'log_default_scale'))
            ):
                raise ValueError('Unsupported parameter CKKS profile')
        self.path = task / 'provisioner/model_parameters.h5'
        self.arguments = {}
        signatures = self.signature.get('encrypted_parameters', self.signature.get('offline', []))
        with h5py.File(self.path) as weights:
            for signature in signatures:
                if signature['type'] != 'ct':
                    raise ValueError('Unsupported offline parameter type')
                if 'generator' in signature:
                    from .pcm_parameter_materialization import matrix_argument

                    item = matrix_argument(signature, self.graph, self.config, self.parameters, weights)
                    if item['id'] in self.arguments:
                        raise ValueError('Duplicate parameter argument')
                    self.arguments[item['id']] = item
                    continue
                source, layer_id, kind, degree = parameter_identity(signature, self.graph['layer'])
                layer = self.graph['layer'][layer_id]
                feature = self.graph['feature'][layer['feature_input'][0]]
                profile = self.parameters[feature['ckks_parameter_id']]
                if layer['type'] not in ('fc0', 'fc1', 'conv2d', 'polyact', 'mult_scalar'):
                    raise ValueError('Unsupported parameter layer: ' + layer['type'])
                depthwise = layer.get('groups') == layer.get('channel_input') == layer.get('channel_output')
                if layer['type'] == 'conv2d' and (
                    (layer['groups'] != 1 and not depthwise)
                    or kind == 'mask'
                    or layer.get('upsample_factor_in', [1, 1]) != [1, 1]
                ):
                    raise ValueError('Unsupported convolution parameter geometry')
                if layer['type'] in ('fc0', 'fc1') and len(feature.get('special_info', {}).get('shape', [])) == 1:
                    raise ValueError('Unsupported 1D dense parameter geometry')
                if layer['type'] == 'polyact' and (feature['dim'] == 1 or feature.get('special_info')):
                    raise ValueError('Unsupported polynomial parameter geometry')
                size = signature['size']
                if not size or any(type(n) is not int or n <= 0 for n in size):
                    raise ValueError('Invalid parameter dimensions')
                scale = signature.get('scale')
                if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
                    raise ValueError('Invalid parameter scale')
                if not 0 <= signature['level'] < len(profile['q']):
                    raise ValueError('Invalid parameter level')
                path = layer.get('bias_path') if kind == 'bias' else layer.get('weight_path')
                if path not in weights:
                    raise ValueError(f'Missing parameter H5 dataset: {path}')
                if weights[path].size * 8 > 32 * 1024**2:
                    raise ValueError('Parameter dataset exceeds the validated browser profile')
                layout = parameter_layout(self.graph, layer_id, signature)
                if 'layout' in signature and signature['layout'] != layout:
                    raise ValueError('Signature parameter layout differs from graph')
                item = dict(
                    signature,
                    source_id=source,
                    layer_id=layer_id,
                    param_kind=kind,
                    count=math.prod(size),
                    slots=profile['n_slots'],
                    use_site_level=signature.get('use_site_level', signature['level']),
                )
                if layout is not None:
                    item['layout'] = layout
                if degree is not None:
                    item['coeff_idx'] = degree
                if source.startswith('convw_') and 'seed_weight_subgrid_stride' in layer:
                    item['param_kind'] = 'seed_weight'
                if 'dense_seed_tiles' in layer and layer['dense_seed_tiles'][kind]:
                    item['use_site_level'] = layer['dense_seed_tiles']['use_site_level'] - int(kind == 'bias')
                if 'poly_seed_layout' in layer:
                    item['degree'] = degree
                    item['use_site_level'] = layer['poly_seed_layout']['use_site_levels'][degree]
                if item['id'] in self.arguments:
                    raise ValueError('Duplicate parameter argument')
                self.arguments[item['id']] = item
        if not self.arguments:
            raise ValueError('Task has no encrypted parameters')
        if sum(item['count'] for item in self.arguments.values()) > 16384:
            raise ValueError('Parameter index exceeds the validated 16384-ciphertext browser profile')
        del self.signature

    def plan(self):
        return [
            {key: value for key, value in item.items() if not key.startswith('_')} for item in self.arguments.values()
        ]

    def values(self, argument, flat_index):
        item = self.arguments[argument]
        if type(flat_index) is not int or not 0 <= flat_index < item['count']:
            raise ValueError('Parameter flat index out of bounds')
        if item['param_kind'] == 'matrix':
            from .pcm_parameter_materialization import matrix_values

            result = matrix_values(item, self.graph, self.config, self.path)
            if result.shape != (item['slots'],) or not np.isfinite(result).all():
                raise ValueError('Invalid PCM materialized parameter vector')
            return np.ascontiguousarray(result, dtype='<f8')
        layer = self.graph['layer'][item['layer_id']]
        feature = self.graph['feature'][layer['feature_input'][0]]
        path = layer.get('bias_path') if item['param_kind'] == 'bias' else layer['weight_path']
        from .h5_parameter_values import H5ParameterValues

        with h5py.File(self.path) as weights:
            values = H5ParameterValues(weights[path])
            if layer['type'] in ('fc0', 'fc1'):
                output = self.graph['feature'][layer['feature_output'][0]]
                values = (
                    values.reshape(output['channel'], feature['channel'])
                    if item['param_kind'] == 'weight'
                    else values.reshape(-1)
                )
                result = self._dense(item, feature, values, flat_index)
            elif layer['type'] == 'conv2d':
                values = (
                    values.reshape(
                        layer['channel_output'], layer['channel_input'] // layer['groups'], *layer['kernel_shape']
                    )
                    if item['param_kind'] != 'bias'
                    else values.reshape(-1)
                )
                result = self._convolution(item, layer, feature, values, flat_index)
            else:
                if item['param_kind'] == 'poly_coeff':
                    values = values.reshape(layer['order'] + 1, feature['channel'])[item['coeff_idx']]
                result = self._channel_values(item, feature, values.reshape(-1), flat_index)
        if result.shape != (item['slots'],) or not np.isfinite(result).all():
            raise ValueError('Invalid materialized parameter vector')
        return np.ascontiguousarray(result, dtype='<f8')

    def _channel_values(self, item, feature, values, index):
        slots, pack = item['slots'], feature['pack_num']
        positions = np.arange(slots)
        if feature['dim'] == 0:
            skip = feature['skip']
            channels = index * pack + positions // skip
            valid = (positions % skip == 0) & (positions // skip < pack)
        else:
            shape, skip = feature['shape'], feature['skip']
            expansion = [math.ceil(s / b) for s, b in zip(shape, self.config['block_shape'])]
            physical = [s // e * k for s, e, k in zip(shape, expansion, skip)]
            block = positions // math.prod(physical)
            row = positions % math.prod(physical) // physical[1]
            col = positions % physical[1]
            channels = (
                index * pack // math.prod(expansion) + block * math.prod(skip) + row % skip[0] * skip[1] + col % skip[1]
            )
            valid = block < pack // math.prod(skip)
        valid &= channels < len(values)
        result = np.zeros(slots)
        result[valid] = values[channels[valid]]
        layout = item.get('layout')
        if item['param_kind'] == 'poly_coeff' and layout and layout['packed_degrees'][item['coeff_idx']]:
            result.fill(0)
            stride = layout['stride']
            group = math.prod(stride)
            for sub in range(group):
                request = index * group + sub
                if request >= layout['channel_groups']:
                    break
                plain = dict(item)
                plain.pop('layout')
                source = self._channel_values(plain, feature, values, request)
                select = (row // skip[0] % stride[0] == sub // stride[1]) & (
                    col // skip[1] % stride[1] == sub % stride[1]
                )
                result[select] = source[select]
        return result

    def _dense(self, item, feature, values, index):
        slots, pack = item['slots'], feature['pack_num']
        bias = item['param_kind'] == 'bias'
        special = feature.get('special_info')
        layout = item.get('layout')
        if not special and layout and (layout.get('weight') or layout.get('bias')):
            special = {'shape': layout['input_shape'], 'skip': layout['skip'], 'invalid_fill': layout['invalid_fill']}
        if not special:
            result = np.zeros(slots)
            if bias:
                channels = index * pack + np.arange(pack)
                valid = channels < values.size
                result[np.arange(pack)[valid] * feature['skip']] = values[channels[valid]]
            else:
                packed_in, diagonal = divmod(index, item['size'][1])
                output_group = packed_in
                input_group, diagonal = divmod(diagonal, pack)
                baby = math.ceil(math.sqrt(pack))
                giant, small = divmod(diagonal, baby)
                local = np.arange(pack)
                out_channels = output_group * pack + (local - giant * baby + pack) % pack
                in_channels = input_group * pack + (local + small) % pack
                valid = (out_channels < values.shape[0]) & (in_channels < values.shape[1])
                result[local[valid] * feature['skip']] = values[out_channels[valid], in_channels[valid]]
            return result
        shape, skip, fill = special['shape'], special['skip'], special['invalid_fill']
        physical = np.multiply(shape, skip)
        block_size = int(np.prod(physical))
        blocks = slots // block_size
        valid_skip = np.array(skip) // fill
        per_block = int(np.prod(valid_skip))
        input_blocks = math.ceil(feature['channel'] // math.prod(shape) / (per_block * blocks)) * blocks

        def generate(request):
            output_index, input_index = (request, 0) if bias else divmod(request, input_blocks)
            positions = np.arange(slots)
            block = positions // block_size
            row = positions % block_size // physical[1]
            col = positions % physical[1]
            out_channel = output_index * blocks + block
            result = np.zeros(slots)
            if bias:
                valid = (row == 0) & (col == 0) & (out_channel < values.size)
                result[valid] = values[out_channel[valid]]
            else:
                rotated = (input_index + block) % min(input_blocks, blocks) + input_index // min(
                    input_blocks, blocks
                ) * min(input_blocks, blocks)
                channel = rotated * per_block + row % skip[0] * valid_skip[1] + col % skip[1]
                source = channel * math.prod(shape) + row // skip[0] * shape[1] + col // skip[1]
                valid = (
                    (row % skip[0] < valid_skip[0])
                    & (col % skip[1] < valid_skip[1])
                    & (out_channel < values.shape[0])
                    & (source < values.shape[1])
                )
                result[valid] = values[out_channel[valid], source[valid]]
            return result

        layout = item.get('layout')
        if not layout or not layout[item['param_kind']]:
            return generate(index)
        result = np.zeros(slots)
        requests = math.ceil((values.size if bias else values.shape[0]) / blocks) * (1 if bias else input_blocks)
        group = math.prod(fill)
        for sub in range(group):
            request = index * group + sub
            if request >= requests:
                break
            delta = sub // fill[1] * valid_skip[0] * shape[1] * skip[1] + sub % fill[1] * valid_skip[1]
            result += np.roll(generate(request), delta)
        return result

    def _convolution(self, item, layer, feature, values, index):
        slots, pack = item['slots'], feature['pack_num']
        shape, skip, stride = feature['shape'], feature['skip'], layer['stride']
        if layer.get('is_big_size'):
            return self._inverse_convolution(item, layer, feature, values, index)
        physical = np.multiply(shape, skip)
        block_size = int(np.prod(physical))
        blocks = math.ceil(pack / math.prod(skip))
        positions = np.arange(slots)
        block = positions // block_size
        row = positions % block_size // physical[1]
        col = positions % physical[1]
        if item['param_kind'] == 'bias':
            output_skip = np.multiply(skip, stride)
            channels = (
                index * pack * math.prod(stride)
                + block * int(np.prod(output_skip))
                + row % output_skip[0] * output_skip[1]
                + col % output_skip[1]
            )
            valid = (channels < len(values)) & (block < blocks)
            result = np.zeros(slots)
            result[valid] = values[channels[valid]]
            return result
        if layer['groups'] != 1:
            return self._depthwise(item, layer, feature, values, index, block, row, col, blocks)
        periodic_channels = max(math.prod(skip), 1 << (layer['channel_input'] - 1).bit_length())
        periodic = (
            layer['channel_input'] <= pack and periodic_channels <= pack and periodic_channels % math.prod(skip) == 0
        )
        input_blocks = periodic_channels // math.prod(skip) if periodic else blocks
        inputs = math.ceil(layer['channel_input'] / pack) * input_blocks
        kernel_count = math.prod(layer['kernel_shape'])

        def generate(output, input_index, kernel, boundary):
            kernel_row, kernel_col = divmod(kernel, layer['kernel_shape'][1])
            offset = input_index * math.prod(skip) + block * math.prod(skip) + col % skip[1] + row % skip[0] * skip[1]
            channels_in = offset % periodic_channels if periodic else input_index // blocks * pack + offset % pack
            channels_out = output * blocks + block
            valid = (channels_in < layer['channel_input']) & (channels_out < layer['channel_output']) & (block < blocks)
            if boundary:
                shifted_row = row + (kernel_row - layer['kernel_shape'][0] // 2) * skip[0]
                shifted_col = col + (kernel_col - layer['kernel_shape'][1] // 2) * skip[1]
                valid &= (
                    (shifted_row >= 0) & (shifted_row < physical[0]) & (shifted_col >= 0) & (shifted_col < physical[1])
                )
            result = np.zeros(slots)
            result[valid] = values[channels_out[valid], channels_in[valid], kernel_row, kernel_col]
            return result

        layout = item.get('layout')
        if layout is None:
            output, rest = divmod(index, item['size'][1] * item['size'][2])
            input_index, kernel = divmod(rest, item['size'][2])
            return generate(output, input_index, kernel, True)
        result = np.zeros(slots)
        group, seed_stride = layout['group_size'], layout['stride']
        if layout['kind'] == 'joint':
            requests = math.ceil(layer['channel_output'] / blocks) * inputs * kernel_count
            for sub in range(group):
                request = index * group + sub
                if request >= requests:
                    break
                output, rest = divmod(request, inputs * kernel_count)
                input_index, kernel = divmod(rest, kernel_count)
                select = (row // skip[0] % seed_stride[0] == sub // seed_stride[1]) & (
                    col // skip[1] % seed_stride[1] == sub % seed_stride[1]
                )
                source = generate(output, input_index, kernel, False)
                result[select] = source[select]
        else:
            output, rest = divmod(index, item['size'][1] * item['size'][2])
            seed, kernel = divmod(rest, item['size'][2])
            for sub in range(group):
                request = seed * group + sub
                if request >= inputs:
                    break
                select = (row // skip[0] % seed_stride[0] == sub // seed_stride[1]) & (
                    col // skip[1] % seed_stride[1] == sub % seed_stride[1]
                )
                source = generate(output, request, kernel, False)
                result[select] = source[select]
        return result

    def _inverse_convolution(self, item, layer, feature, values, index):
        from itertools import product

        block_shape = self.config['block_shape']
        positions = np.arange(item['slots'])
        row, col = positions // block_shape[1], positions % block_shape[1]
        valid = positions < math.prod(block_shape)
        result = np.zeros(item['slots'])
        if item['param_kind'] == 'bias':
            result[valid] = values[index]
            return result
        depthwise = layer['groups'] != 1
        if depthwise:
            output, kernel_index = divmod(index, item['size'][1])
            input_channel = 0
        else:
            output, rest = divmod(index, item['size'][1] * item['size'][2])
            input_channel, kernel_index = divmod(rest, item['size'][2])
        kernel = layer['kernel_shape']
        stride = layer['stride']
        shape = feature['shape']
        if any(s // t < b for s, t, b in zip(shape, stride, block_shape)):
            stride = [s // b for s, b in zip(shape, block_shape)]
        output_step = [s // (b * t) for s, b, t in zip(shape, block_shape, stride)]
        input_step = np.multiply(stride, output_step)
        pad = [(k - 1) // 2 for k in kernel]

        def kernel_position(index):
            for r in product(*(range(n) for n in output_step)):
                for segment in product(*(range(n) for n in stride)):
                    if any(seg >= k for seg, k in zip(segment, kernel)):
                        continue
                    split = [(k + t - 1 - seg) // t for k, t, seg in zip(kernel, stride, segment)]
                    for uv in product(*(range(n) for n in split)):
                        if index == 0:
                            original = np.multiply(uv, stride) + segment
                            val = np.array(segment) - pad + np.multiply(stride, np.array(uv) + r)
                            step = (val - val % input_step) // input_step
                            return original, step
                        index -= 1
            raise ValueError('Inverse convolution kernel index out of bounds')

        layout = item.get('layout')
        if layout:
            a, b = layout['stride']
            count = math.prod(kernel) if layout['kind'] == 'canonical' else math.prod(kernel) * math.prod(output_step)
            for sub in range(layout['group_size']):
                request = kernel_index * layout['group_size'] + sub
                if request >= count:
                    break
                original = divmod(request, kernel[1]) if layout['kind'] == 'canonical' else kernel_position(request)[0]
                selected = valid & (row % a == sub // b) & (col % b == sub % b)
                result[selected] = values[output, input_channel, original[0], original[1]]
        else:
            original, step = kernel_position(kernel_index)
            selected = valid & (row + step[0] >= 0) & (row + step[0] < block_shape[0])
            selected &= (col + step[1] >= 0) & (col + step[1] < block_shape[1])
            result[selected] = values[output, input_channel, original[0], original[1]]
        return result

    def _depthwise(self, item, layer, feature, values, index, block, row, col, blocks):
        skip, pack = feature['skip'], feature['pack_num']
        physical = np.multiply(feature['shape'], skip)
        kernel_count = math.prod(layer['kernel_shape'])

        def generate(group, kernel, boundary):
            kr, kc = divmod(kernel, layer['kernel_shape'][1])
            channels = group * pack + block * math.prod(skip) + row % skip[0] * skip[1] + col % skip[1]
            valid = (channels < layer['channel_output']) & (block < blocks)
            if boundary:
                shifted_row = row + (kr - layer['kernel_shape'][0] // 2) * skip[0]
                shifted_col = col + (kc - layer['kernel_shape'][1] // 2) * skip[1]
                valid &= (
                    (shifted_row >= 0) & (shifted_row < physical[0]) & (shifted_col >= 0) & (shifted_col < physical[1])
                )
            result = np.zeros(item['slots'])
            result[valid] = values[channels[valid], 0, kr, kc]
            return result

        layout = item.get('layout')
        if layout is None:
            group, kernel = divmod(index, item['size'][1])
            return generate(group, kernel, True)
        joint = layout['kind'] == 'joint'
        requests = math.ceil(layer['channel_input'] / pack) * kernel_count if joint else kernel_count
        group, seed = (0, index) if joint else divmod(index, item['size'][1])
        result = np.zeros(item['slots'])
        stride = layout['stride']
        for sub in range(layout['group_size']):
            request = seed * layout['group_size'] + sub
            if request >= requests:
                break
            channel_group, kernel = divmod(request, kernel_count) if joint else (group, request)
            source = generate(channel_group, kernel, False)
            selected = (row // skip[0] % stride[0] == sub // stride[1]) & (
                col // skip[1] % stride[1] == sub % stride[1]
            )
            result[selected] = source[selected]
        return result
