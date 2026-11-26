# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Reshaped H5 views that gather only bounded ranges for one slot vector."""

import math

import numpy as np

CHUNK_VALUES = 32768


class H5ParameterValues:
    def __init__(self, dataset, shape=None, offset=0):
        self.dataset = dataset
        self.shape = tuple(dataset.shape if shape is None else shape)
        self.size = math.prod(self.shape)
        self.offset = offset

    def __len__(self):
        return self.shape[0]

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        if -1 in shape:
            missing = self.size // -math.prod(shape)
            shape = tuple(missing if n == -1 else n for n in shape)
        if math.prod(shape) != self.size:
            raise ValueError('H5 parameter reshape changes size')
        return H5ParameterValues(self.dataset, shape, self.offset)

    def __getitem__(self, indices):
        indices = indices if isinstance(indices, tuple) else (indices,)
        if len(indices) < len(self.shape) and all(np.isscalar(i) for i in indices):
            offset = self.offset
            for axis, index in enumerate(indices):
                offset += int(index) * math.prod(self.shape[axis + 1 :])
            return H5ParameterValues(self.dataset, self.shape[len(indices) :], offset)
        if len(indices) != len(self.shape):
            raise IndexError('H5 parameter needs one index per dimension')
        arrays = np.broadcast_arrays(*[np.asarray(i, dtype=np.int64) for i in indices])
        positions = np.full(arrays[0].shape, self.offset, dtype=np.int64)
        for axis, index in enumerate(arrays):
            if np.any(index < 0) or np.any(index >= self.shape[axis]):
                raise IndexError('H5 parameter index outside shape')
            positions += index * math.prod(self.shape[axis + 1 :])
        blocks = positions // CHUNK_VALUES
        result = np.empty(positions.shape, dtype=np.float64)
        for block in np.unique(blocks):
            start = int(block) * CHUNK_VALUES
            count = min(CHUNK_VALUES, self.dataset.size - start)
            if self.dataset.ndim == 1:
                values = self.dataset[start : start + count]
            else:
                # Point selection also supports legacy multidimensional H5 datasets.
                space = self.dataset.id.get_space()
                coordinates = np.stack(np.unravel_index(np.arange(start, start + count), self.dataset.shape), axis=1)
                space.select_elements(coordinates)
                import h5py

                values = np.empty(count, dtype=np.float64)
                self.dataset.id.read(h5py.h5s.create_simple((count,)), space, values)
            selected = blocks == block
            result[selected] = values[positions[selected] - start]
        return result.item() if result.ndim == 0 else result
