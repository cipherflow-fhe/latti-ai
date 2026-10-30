# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import math

from inference.lattisense.frontend.custom_task import (
    CkksPlaintextRingtNode, add, custom_compute, mult, rescale, rotate_cols,
)
from inference.model_generator.layers.encrypted_param_ops import materialize_encrypted_param


def inverse_kernel_positions(kernel_shape, stride, output_step, padding):
    """Yield (kernel position, block rotation) in the inverse layer's kernel order."""
    input_step = [s * o for s, o in zip(stride, output_step)]
    for r in range(output_step[0]):
        for c in range(output_step[1]):
            for sh in range(min(stride[0], kernel_shape[0])):
                for sw in range(min(stride[1], kernel_shape[1])):
                    for u in range((kernel_shape[0] - 1 - sh) // stride[0] + 1):
                        for v in range((kernel_shape[1] - 1 - sw) // stride[1] + 1):
                            val = [sh - padding[0] + stride[0] * (u + r),
                                   sw - padding[1] + stride[1] * (v + c)]
                            yield ([u * stride[0] + sh, v * stride[1] + sw],
                                   [x // step for x, step in zip(val, input_step)])


class KernelSeedPacking:
    """Pack each row of kernel weights into spatial subgrids, preserving outer axes."""

    def _init_kernel_seed_packing(self, stride, shape, skip, blocks, kernel_steps, op_class):
        self.seed_weight_subgrid_stride = list(stride)
        self.seed_shape = list(shape)
        self.seed_skip = list(skip)
        self.seed_blocks = blocks
        self.seed_kernel_steps = kernel_steps
        self.seed_op_class = op_class
        a, b = stride
        self.use_seed_weight_pack = (
            a > 0 and b > 0 and not (a & (a - 1)) and not (b & (b - 1))
            and (a > 1 or b > 1) and shape[0] % a == 0 and shape[1] % b == 0
        )
        self.seed_requires_boundary_mask = self.kernel_shape != [1, 1] or any(
            step != [0, 0] for step in kernel_steps)

    @property
    def kernel_weight_count(self):
        return len(self.seed_kernel_steps)

    @property
    def seed_weight_group_size(self):
        return math.prod(self.seed_weight_subgrid_stride)

    @property
    def seed_weight_kernel_count(self):
        return (math.ceil(self.kernel_weight_count / self.seed_weight_group_size)
                if self.use_seed_weight_pack else self.kernel_weight_count)

    @property
    def seed_weight_level_cost(self):
        return (2 if self.seed_requires_boundary_mask else 1) if self.use_seed_weight_pack else 0

    def make_seed_mask_pt_nodes(self, layer_id, level):
        if not self.use_seed_weight_pack:
            return []
        masks = []
        for k, step in enumerate(self.seed_kernel_steps):
            attrs = {
                'op_class': self.seed_op_class, 'type': 'seed_mask_pt', 'mask_kind': 'subgrid',
                'seed_shape': self.seed_shape, 'seed_skip': self.seed_skip, 'seed_blocks': self.seed_blocks,
                'seed_a': self.seed_weight_subgrid_stride[0], 'seed_b': self.seed_weight_subgrid_stride[1],
                'seed_step': step, 'subgrid': k % self.seed_weight_group_size,
                'kernel_index': k, 'level': level + self.seed_weight_level_cost,
            }
            first = CkksPlaintextRingtNode(f'convseedm_{layer_id}_{k}_0')
            custom_compute(inputs=[], output=first, type='encode_pt', attributes=attrs)
            second = first
            if self.seed_requires_boundary_mask:
                second = CkksPlaintextRingtNode(f'convseedm_{layer_id}_{k}_1')
                custom_compute(inputs=[], output=second, type='encode_pt',
                               attributes=dict(attrs, mask_kind='kernel', level=attrs['level'] - 1))
            masks.append([first, second])
        return masks

    def kernel_weight(self, row, kernel_index, seed_masks=None, trigger=None):
        if not self.use_seed_weight_pack:
            return row[kernel_index]
        weight = row[kernel_index // self.seed_weight_group_size]
        if trigger is not None:
            weight = materialize_encrypted_param(weight, trigger)
        weight = rescale(mult(weight, seed_masks[kernel_index][0]))
        a, b = self.seed_weight_subgrid_stride
        r, c = divmod(kernel_index % (a * b), b)
        units = [self.seed_skip[0] * self.seed_shape[1] * self.seed_skip[1], self.seed_skip[1]]
        for size, residue, unit in zip((a, b), (r, c), units):
            for bit in range(size.bit_length() - 1):
                step = (1 << bit) * unit
                weight = add(weight, rotate_cols(weight, [step if (residue >> bit) & 1 else -step])[0])
        if self.seed_requires_boundary_mask:
            weight = rescale(mult(weight, seed_masks[kernel_index][1]))
        return weight

    def add_seed_recovery_ops(self, ops, level, rows):
        if not self.use_seed_weight_pack:
            return
        terms = rows * self.kernel_weight_count
        seed_level = level + self.seed_weight_level_cost
        copies = sum(x.bit_length() - 1 for x in self.seed_weight_subgrid_stride)
        ops[seed_level]['mult_plain'] += terms
        ops[seed_level]['rescale'] += terms
        ops[seed_level - 1]['rotate'] += terms * copies
        ops[seed_level - 1]['add'] += terms * copies
        if self.seed_requires_boundary_mask:
            ops[seed_level - 1]['mult_plain'] += terms
            ops[seed_level - 1]['rescale'] += terms
