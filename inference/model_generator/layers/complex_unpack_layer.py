# Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from inference.lattisense.frontend.custom_task import *


class ComplexUnpackLayer:
    """Split z = (a + i*b)/2 into a and b without another multiplication."""

    @staticmethod
    def _connect_output(op_type, inputs, output):
        op = FheComputeNode(op_type)
        g_dag.add_edges_from((node, op) for node in inputs)
        g_dag.add_edge(op, output)
        output.is_ntt = inputs[0].is_ntt
        return output

    def call(
        self,
        packed: list[CkksCiphertextNode],
        conjugate: list[CkksCiphertextNode] | None = None,
        output_a: list[CkksCiphertextNode] | None = None,
        output_b: list[CkksCiphertextNode] | None = None,
    ) -> tuple[list[CkksCiphertextNode], list[CkksCiphertextNode]]:
        if conjugate is not None and len(packed) != len(conjugate):
            raise ValueError('ComplexUnpackLayer requires equally sized ciphertext lists')
        if output_a is not None and len(output_a) != len(packed):
            raise ValueError('ComplexUnpackLayer output_a has an invalid ciphertext count')
        if output_b is not None and len(output_b) != len(packed):
            raise ValueError('ComplexUnpackLayer output_b has an invalid ciphertext count')

        real = []
        imag = []
        for index, packed_node in enumerate(packed):
            conjugate_node = rotate_rows(packed_node) if conjugate is None else conjugate[index]
            if output_a is None:
                real.append(add(packed_node, conjugate_node))
            else:
                real.append(self._connect_output(OperationType.Add, [packed_node, conjugate_node], output_a[index]))
            difference = sub(packed_node, conjugate_node)
            if output_b is None:
                imag.append(div_by_i(difference))
            else:
                imag.append(self._connect_output(OperationType.DivByi, [difference], output_b[index]))
        return real, imag
