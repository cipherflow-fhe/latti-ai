# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Lower exported transformer operators to ordinary float64 ONNX operations.

One original node is lowered at a time. This retains the existing reference's
arithmetic and allows browser ORT to release each node's weights and arena.
"""

import math

import numpy as np
import onnx
from onnx import helper, numpy_helper


class _Builder:
    def __init__(self, feeds):
        self.feeds = {name: np.asarray(value, dtype=np.float64) for name, value in feeds.items()}
        self.used = set(feeds)
        self.nodes, self.initializers = [], []
        self.index = 0

    def name(self):
        while True:
            self.index += 1
            name = f'__latti_reference_{self.index}'
            if name not in self.used:
                self.used.add(name)
                return name

    def constant(self, value, dtype=np.float64):
        name = self.name()
        self.initializers.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def op(self, kind, *inputs, **attributes):
        output = self.name()
        self.nodes.append(helper.make_node(kind, list(inputs), [output], **attributes))
        return output

    def scalar(self, kind, source, value):
        return self.op(kind, source, self.constant(value))

    def sum(self, source):
        return self.op('ReduceSum', source, self.constant([-1], np.int64), keepdims=1)

    def mean(self, source):
        return self.op('ReduceMean', source, axes=[-1], keepdims=1)

    def polynomial(self, source, coefficients):
        if isinstance(coefficients, bytes):
            coefficients = coefficients.decode()
        if isinstance(coefficients, str):
            coefficients = [float(value) for value in coefficients.split(',')]
        result = self.scalar('Add', self.scalar('Mul', source, 0), coefficients[-1])
        for coefficient in coefficients[-2::-1]:
            result = self.scalar('Add', self.op('Mul', result, source), coefficient)
        return result

    def inverse(self, source, epsilon, alpha, attrs):
        a, b, lower = self.constant(1), source, epsilon
        for _ in range(attrs.get('max_inverse_iterations', 15)):
            if not attrs.get('use_asor', 1):
                factor = self.op('Sub', self.constant(2), self.op('Mul', b, a))
                a = self.op('Mul', a, factor)
            else:
                if lower >= 1 - alpha:
                    break
                k = 2 / (1 + lower)
                factor = self.scalar('Mul', self.op('Sub', self.constant(2), self.scalar('Mul', b, k)), k)
                a, b = self.op('Mul', a, factor), self.op('Mul', b, factor)
                lower = k * lower * (2 - k * lower)
        return a

    def layernorm(self, x, weight, bias, attrs):
        eps = attrs.get('eps', attrs.get('epsilon', 1e-5))
        centered = self.op('Sub', x, self.mean(x))
        variance = self.scalar('Add', self.mean(self.op('Mul', centered, centered)), eps)
        if 'max_var' in attrs:
            iv = 1 / ((attrs['max_var'] * attrs.get('w_buffer', 1.05) + eps) * attrs.get('input_scale', 1.0) ** 2)
            iterations = attrs.get('max_inverse_sqrt_iterations', 7)
        else:
            iv, iterations = attrs.get('inv_var_scale', 1.0), attrs.get('num_iters', 2)
        a = self.scalar('Mul', variance, iv)
        y = self.scalar('Mul', a, attrs.get('minimax_c2', attrs.get('c2', 11.52830778)))
        y = self.scalar('Add', y, attrs.get('minimax_c1', attrs.get('c1', -16.15885111)))
        y = self.scalar('Add', self.op('Mul', a, y), attrs.get('minimax_c0', attrs.get('c0', 6.19067182)))
        for _ in range(iterations):
            correction = self.op('Sub', self.constant(3), self.op('Mul', self.op('Mul', a, y), y))
            y = self.op('Mul', self.scalar('Mul', y, 0.5), correction)
        result = self.op('Mul', centered, y)
        result = self.scalar('Mul', result, math.sqrt(iv))
        return self.op('Add', self.op('Mul', result, weight), bias)

    def range_polynomial(self, x, running_max, coefficients, attrs):
        scale = self.scalar(
            'Add', self.scalar('Div', running_max, attrs.get('upper_bound', 3.0)), attrs.get('eps', 1e-3)
        )
        z = self.op('Div', x, scale)
        z2 = self.op('Mul', z, z)
        a0, a1, a2, a4 = coefficients
        result = self.op('Add', a0, self.op('Mul', a1, z))
        result = self.op('Add', result, self.op('Mul', a2, self.scalar('Sub', z2, 1)))
        hermite4 = self.scalar('Add', self.op('Sub', self.op('Mul', z2, z2), self.scalar('Mul', z2, 6)), 3)
        return self.op('Mul', scale, self.op('Add', result, self.op('Mul', a4, hermite4)))

    def finish(self, result, output):
        self.nodes.append(helper.make_node('Identity', [result], [output]))
        inputs = [
            helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(array.dtype), list(array.shape))
            for name, array in self.feeds.items()
        ]
        model = helper.make_model(
            helper.make_graph(
                self.nodes,
                'float64-reference-node',
                inputs,
                [helper.make_tensor_value_info(output, onnx.TensorProto.DOUBLE, None)],
                self.initializers,
            ),
            opset_imports=[helper.make_opsetid('', 17)],
            ir_version=8,
        )
        return onnx.shape_inference.infer_shapes(model), self.feeds


def lower_node(node, feeds, initializers, model_kind, weight_loader=None):
    """Return a small standard ONNX graph and named float64 feeds."""
    builder = _Builder(feeds)
    builder.used.update(node.output)
    attrs = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
    args = []
    for name in node.input:
        if name not in initializers:
            args.append(name)
        elif weight_loader is None:
            args.append(builder.constant(numpy_helper.to_array(initializers[name])))
        else:
            # Large weights travel as one tensor feed, not as repeated protobuf
            # raw_data, external-data and session-initializer copies.
            key = builder.name()
            builder.feeds[key] = np.asarray(weight_loader(initializers[name]), dtype=np.float64)
            args.append(key)
    bert = model_kind == 'bert'
    op = node.op_type
    if op == 'Constant':
        result = builder.constant(numpy_helper.to_array(attrs['value']))
    elif op == 'Identity':
        result = args[0]
    elif op == 'Add':
        result = builder.op('Add', *args)
    elif op == 'Linear':
        x, weight, bias = args
        if attrs.get('select_first_token'):
            x = builder.op('Gather', x, builder.constant(0, np.int64), axis=1)
        if bert:
            weight = builder.op('Transpose', weight, perm=[1, 0])
        result = builder.op('Add', builder.op('MatMul', x, weight), bias)
    elif op == 'CustomLayerNorm':
        result = builder.layernorm(*args, attrs)
    elif op == 'PolyActRN':
        result = builder.range_polynomial(args[0], args[1], args[2:], attrs)
    elif op == 'CustomGELU':
        z = builder.polynomial(builder.scalar('Div', args[0], attrs['scale']), attrs['p1_coefficients'])
        z = builder.polynomial(builder.scalar('Div', z, attrs['f2_input_scale']), attrs['p2_coefficients'])
        z = builder.polynomial(builder.scalar('Div', z, attrs['f3_input_scale']), attrs['p3_coefficients'])
        result = builder.op('Mul', args[0], builder.scalar('Add', z, 0.5))
    elif op == 'CustomTanh':
        z = builder.polynomial(
            builder.scalar('Div', args[0], attrs['scale'] * attrs['f1_input_scale']), attrs['p1_coefficients']
        )
        result = builder.polynomial(builder.scalar('Div', z, attrs['f2_input_scale']), attrs['p2_coefficients'])
    elif op == 'CustomMultiHeadAttention':
        x = args[0]
        batch, tokens, _ = builder.feeds[node.input[0]].shape
        heads, dim = attrs['num_heads'], attrs['head_dim']
        if bert:
            q, k, v = [
                builder.op(
                    'Add', builder.op('MatMul', x, builder.op('Transpose', args[index], perm=[1, 0])), args[index + 1]
                )
                for index in (1, 3, 5)
            ]
            projection, bias = args[7:9]
        else:
            qkv = builder.op('Add', builder.op('MatMul', x, args[1]), args[2])
            names = [builder.name() for _ in range(3)]
            builder.nodes.append(
                helper.make_node('Split', [qkv, builder.constant([heads * dim] * 3, np.int64)], names, axis=-1)
            )
            q, k, v = names
            projection, bias = args[3:5]
        shape = builder.constant([batch, tokens, heads, dim], np.int64)
        q, k, v = [builder.op('Transpose', builder.op('Reshape', item, shape), perm=[0, 2, 1, 3]) for item in (q, k, v)]
        scores = builder.scalar(
            'Mul',
            builder.op('MatMul', q, builder.op('Transpose', k, perm=[0, 1, 3, 2])),
            attrs.get('scaling', dim**-0.5),
        )
        if bert:
            mid = (attrs['range_min'] + attrs['range_max']) / 2
            a = builder.polynomial(
                builder.scalar('Div', builder.scalar('Sub', scores, mid), attrs['exp_divisor']),
                attrs['exp_coefficients'],
            )
            a = builder.scalar('Pow', a, attrs.get('delta_1', 2))
            epsilon = attrs['inverse_epsilon']
            a = builder.op('Mul', a, builder.inverse(builder.sum(a), epsilon, attrs['inverse_alpha'], attrs))
            delta = attrs.get('delta_2', 2)
            while delta > 1:
                a = builder.op('Mul', a, a)
                delta >>= 1
                key = 'final_inverse_alpha' if delta == 1 else 'inverse_alpha'
                a = builder.op('Mul', a, builder.inverse(builder.sum(a), epsilon / tokens, attrs[key], attrs))
        else:
            scale = builder.op('Reshape', args[5], builder.constant([1, heads, 1, tokens], np.int64))
            a = builder.scalar('Div', builder.range_polynomial(scores, scale, args[6:10], attrs), tokens)
        out = builder.op(
            'Reshape',
            builder.op('Transpose', builder.op('MatMul', a, v), perm=[0, 2, 1, 3]),
            builder.constant([batch, tokens, heads * dim], np.int64),
        )
        if bert:
            projection = builder.op('Transpose', projection, perm=[1, 0])
        result = builder.op('Add', builder.op('MatMul', out, projection), bias)
    else:
        raise ValueError('Unsupported transformer ONNX operator: ' + op)
    return builder.finish(result, node.output[0])
