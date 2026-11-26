# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Bound ONNX-to-H5 conversion to the datasets used by the current layer."""

from collections.abc import Mapping, MutableMapping
from pathlib import Path
import math
import hashlib

import numpy as np
import onnx
from onnx import numpy_helper

CHUNK_VALUES = 128 * 1024


class InitializerValues(Mapping):
    def __init__(self, model, onnx_path):
        self.tensors = {tensor.name: tensor for tensor in model.graph.initializer}
        self.base_dir = str(Path(onnx_path).parent)

    def __getitem__(self, name):
        return numpy_helper.to_array(self.tensors[name], base_dir=self.base_dir).astype('float64')

    def take(self, name, indices):
        """Gather from bounded raw-data ranges without populating TensorProto."""
        tensor = self.tensors[name]
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            return np.asarray(numpy_helper.to_array(tensor).reshape(-1)[indices], dtype='float64')
        info = onnx.external_data_helper.ExternalDataInfo(tensor)
        dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)).newbyteorder('<')
        count = math.prod(tensor.dims)
        result = np.empty(indices.shape, dtype='float64')
        blocks = indices // CHUNK_VALUES
        with (Path(self.base_dir) / info.location).open('rb') as stream:
            for block in np.unique(blocks):
                start = int(block) * CHUNK_VALUES
                stream.seek((info.offset or 0) + start * dtype.itemsize)
                values = np.fromfile(stream, dtype=dtype, count=min(CHUNK_VALUES, count - start))
                selected = blocks == block
                result[selected] = values[indices[selected] - start]
        return result

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

    def write_initializer(
        self, name, source, source_name, shape, *, transpose=False, columns=None, multipliers=(), divisors=()
    ):
        """Convert and optionally transpose/scale one MiB of output at a time."""
        source_shape = tuple(source.tensors[source_name].dims)
        size = math.prod(source_shape)
        if columns is not None:
            size = source_shape[0] * (columns[1] - columns[0])
        if size != math.prod(shape) or transpose and len(source_shape) != 2:
            raise ValueError(f'{name}: initializer shape differs from compiled graph')
        exists = name in self.shapes
        if exists and self.shapes[name] != tuple(shape):
            raise ValueError(f'H5 path conflict for {name}')
        dataset = self.file[name] if exists else self.file.create_dataset(name, (size,), dtype='float64')
        width = (columns[1] - columns[0]) if columns else (source_shape[1] if len(source_shape) == 2 else 0)
        for start in range(0, size, CHUNK_VALUES):
            positions = np.arange(start, min(start + CHUNK_VALUES, size))
            indices = positions
            if transpose:
                indices = positions % source_shape[0] * source_shape[1] + positions // source_shape[0]
            if columns is not None:
                indices = (
                    (positions % source_shape[0] if transpose else positions // width) * source_shape[1]
                    + (positions // source_shape[0] if transpose else positions % width)
                    + columns[0]
                )
            values = source.take(source_name, indices)
            for multiplier in multipliers:
                values *= multiplier
            for divisor in divisors:
                values /= divisor[positions // math.prod(shape[1:])]
            if exists and not np.allclose(dataset[start : start + len(values)], values):
                raise ValueError(f'H5 path conflict for {name}')
            dataset[start : start + len(values)] = values
        self.shapes[name] = tuple(shape)

    def value_hash(self, name):
        digest = hashlib.sha256()
        dataset = self.file[name]
        for start in range(0, dataset.size, CHUNK_VALUES):
            digest.update(np.asarray(dataset[start : start + CHUNK_VALUES], dtype='<f8').tobytes())
        return digest.hexdigest()

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
