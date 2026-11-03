# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Resolve matrix parameters at their use-site level in encrypted parameter mode."""

from inference.lattisense.frontend import custom_task as fhe
from inference.model_generator.layers.encrypted_param_ops import (
    add_encrypted_param_bias,
    multiply_with_encrypted_param,
)


def custom_compute(*, inputs, output, type, attributes):
    if type == 'encode_pt' and len(inputs) == 1 and hasattr(inputs[0], 'matrix_parameter_builder'):
        # The original generator defines the packing; the arithmetic use-site
        # supplies the level needed to encrypt that packed plaintext.
        output.matrix_parameter_source = inputs[0]
        output.matrix_parameter_attributes = attributes
        output.matrix_parameter_uses = {}
        return output
    return fhe.custom_compute(inputs=inputs, output=output, type=type, attributes=attributes)


def _resolve(parameter, trigger):
    source = getattr(parameter, 'matrix_parameter_source', None)
    if source is None:
        return parameter
    level = trigger.level
    if level not in parameter.matrix_parameter_uses:
        parameter.matrix_parameter_uses[level] = source.matrix_parameter_builder(
            parameter.matrix_parameter_attributes, parameter.id, level, trigger
        )
    return parameter.matrix_parameter_uses[level]


def mult(x, parameter):
    deferred = hasattr(parameter, 'matrix_parameter_source')
    parameter = _resolve(parameter, x)
    if deferred and isinstance(parameter, fhe.CkksCiphertextNode):
        return multiply_with_encrypted_param(x, parameter)
    return fhe.mult(x, parameter)


def add(x, parameter):
    deferred = hasattr(parameter, 'matrix_parameter_source')
    parameter = _resolve(parameter, x)
    if deferred and isinstance(parameter, fhe.CkksCiphertextNode):
        return add_encrypted_param_bias(x, parameter)
    return fhe.add(x, parameter)


def ct_pt_mult_accumulate(xs, parameters):
    parameters = [_resolve(parameter, x) for x, parameter in zip(xs, parameters)]
    return fhe.ct_pt_mult_accumulate(xs, parameters)
