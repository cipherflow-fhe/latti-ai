# Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from inference.lattisense.frontend.custom_task import *
from inference.model_generator.layers.complex_pack_layer import ComplexPackLayer
from inference.model_generator.layers.complex_unpack_layer import ComplexUnpackLayer


op_class = 'ComplexBtpLayer'


class ComplexBtpLayer:
    """Mega-ag graph builder for the complex CKKS BTP split.

    Input ciphertexts encode z = a + i*b.  The generated graph computes:

        z_half = z * 1/2       (one mult-plain + rescale)
        r       = BTP(z_half)
        r_bar   = conj(r)
        a       = r + r_bar
        b       = (r - r_bar) / i

    The two returned lists are the real and imaginary outputs respectively.
    """

    def __init__(self, complex_bootstrapping: bool = True):
        self.complex_bootstrapping = complex_bootstrapping

    def call_custom_compute(self, x: list[CkksCiphertextNode], data_source, node_prefix: str = ''):
        return self._refresh_nodes(x, data_source, node_prefix=node_prefix)

    def _refresh_nodes(
        self,
        x: list[CkksCiphertextNode],
        data_source,
        output_a=None,
        output_b=None,
        node_prefix: str = '',
    ):
        refreshed_nodes = []
        refreshed_conjugates = []
        for index, x_node in enumerate(x):
            half_id = f'encode_pt_{node_prefix}_half_{index}' if node_prefix else f'encode_pt_complex_btp_half_{index}'
            half_pt = CkksPlaintextRingtNode(half_id)
            custom_compute(
                inputs=[data_source],
                output=half_pt,
                type='encode_pt',
                attributes={
                    'op_class': op_class,
                    'type': 'half_pt',
                    'level': x_node.level,
                    'i': index,
                },
            )
            if x_node.level <= 0:
                raise ValueError('complex_bootstrapping requires input level >= 1')

            z_half = rescale(mult(x_node, half_pt))
            if z_half.level != 0:
                raise ValueError('complex_bootstrapping must enter BTP at level 0')

            refreshed = bootstrap(z_half)
            if self.complex_bootstrapping:
                refreshed_conjugate = rotate_rows(refreshed)
            else:
                # Baseline path: bootstrap z/2 and conj(z/2) independently.
                refreshed_conjugate = bootstrap(rotate_rows(z_half))
            refreshed_nodes.append(refreshed)
            refreshed_conjugates.append(refreshed_conjugate)
        return ComplexUnpackLayer().call(refreshed_nodes, refreshed_conjugates, output_a, output_b)

    def call_paired_custom_compute(
        self,
        a: list[CkksCiphertextNode],
        b: list[CkksCiphertextNode],
        data_source,
        output_a=None,
        output_b=None,
        node_prefix: str = '',
    ):
        if len(a) != len(b):
            raise ValueError('ComplexBtpLayer requires equally sized ciphertext vectors')
        packed = ComplexPackLayer().call(a, b)
        return self._refresh_nodes(packed, data_source, output_a, output_b, node_prefix=node_prefix)

    def get_fhe_op_count(self, n_ct: int, level: int):
        if level <= 0:
            raise ValueError('complex_bootstrapping requires input level >= 1')
        ops = defaultdict(
            lambda: {
                'rotate': 0,
                'mult_plain': 0,
                'mult': 0,
                'add': 0,
                'sub': 0,
                'div_by_i': 0,
                'rescale': 0,
                'bootstrap': 0,
            }
        )
        ops[level]['mult_plain'] += n_ct
        ops[level]['rescale'] += n_ct
        ops[0]['bootstrap'] += n_ct * (1 if self.complex_bootstrapping else 2)
        ops[0]['rotate'] += n_ct  # conjugation is represented by rotate_rows in the graph.
        ops[0]['add'] += n_ct
        ops[0]['sub'] += n_ct
        ops[0]['div_by_i'] += n_ct
        return dict(ops)
