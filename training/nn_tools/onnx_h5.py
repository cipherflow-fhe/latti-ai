# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Bound ONNX-to-H5 conversion to the datasets used by the current layer."""

from collections.abc import Mapping, MutableMapping
from pathlib import Path

import numpy as np
from onnx import numpy_helper


class InitializerValues(Mapping):
    def __init__(self, model, onnx_path):
        self.tensors = {tensor.name: tensor for tensor in model.graph.initializer}
        self.base_dir = str(Path(onnx_path).parent)

    def __getitem__(self, name):
        return numpy_helper.to_array(self.tensors[name], base_dir=self.base_dir).astype('float64')

    def __iter__(self):
        return iter(self.tensors)

    def __contains__(self, name):
        return name in self.tensors

    def __len__(self):
        return len(self.tensors)


class H5Values(MutableMapping):
    """Keep logical shapes for shared-path checks, and persist values immediately."""

    def __init__(self, file):
        self.file = file
        self.shapes = {}

    def __getitem__(self, name):
        if name not in self.shapes:
            raise KeyError(name)
        return self.file[name][...].reshape(self.shapes[name])

    def __setitem__(self, name, value):
        value = np.asarray(value, dtype='float64')
        if name in self.shapes:
            del self.file[name]
        self.file.create_dataset(name, data=value.reshape(-1))
        self.shapes[name] = value.shape

    def __delitem__(self, name):
        del self.file[name]
        del self.shapes[name]

    def __iter__(self):
        return iter(self.shapes)

    def __contains__(self, name):
        return name in self.shapes

    def __len__(self):
        return len(self.shapes)

    @property
    def total_values(self):
        return sum(self.file[name].size for name in self.shapes)
