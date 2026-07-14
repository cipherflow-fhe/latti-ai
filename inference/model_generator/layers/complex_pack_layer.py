# Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from inference.lattisense.frontend.custom_task import *


class ComplexPackLayer:
    """Pack two aligned ciphertext lists into z = a + i*b."""

    def call(self, a: list[CkksCiphertextNode], b: list[CkksCiphertextNode]) -> list[CkksCiphertextNode]:
        if len(a) != len(b):
            raise ValueError('ComplexPackLayer requires equally sized ciphertext lists')
        return [add(a_node, mult_by_i(b_node)) for a_node, b_node in zip(a, b)]
