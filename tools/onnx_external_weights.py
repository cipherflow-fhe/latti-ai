# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Bounded ONNX externalization and graph-only extraction for browser jobs."""

import math
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

CHUNK = 1024 * 1024
MAX_TENSOR = 32 * CHUNK
MAX_METADATA = 32 * CHUNK


def _varint(value):
    out = bytearray()
    while value > 127:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)


def _read_varint(stream, end):
    value = 0
    for shift in range(0, 70, 7):
        if stream.tell() >= end:
            raise ValueError('Truncated ONNX protobuf varint')
        byte = stream.read(1)[0]
        value |= (byte & 127) << shift
        if byte < 128:
            return value
    raise ValueError('Invalid ONNX protobuf varint')


def _fields(stream, end):
    while stream.tell() < end:
        start = stream.tell()
        tag = _read_varint(stream, end)
        field, wire = tag >> 3, tag & 7
        if not field:
            raise ValueError('Invalid ONNX protobuf field')
        if wire == 2:
            length = _read_varint(stream, end)
            offset = stream.tell()
        elif wire == 0:
            offset = stream.tell()
            _read_varint(stream, end)
            length = stream.tell() - offset
        elif wire in (1, 5):
            offset, length = stream.tell(), 8 if wire == 1 else 4
        else:
            raise ValueError('Unsupported ONNX protobuf wire type')
        stop = offset + length
        if stop > end:
            raise ValueError('Truncated ONNX protobuf field')
        yield field, wire, start, offset, length, stop
        stream.seek(stop)


def _copy_range(stream, offset, length, target):
    stream.seek(offset)
    while length:
        block = stream.read(min(length, CHUNK))
        if not block:
            raise ValueError('Truncated ONNX weight range')
        target.write(block)
        length -= len(block)


def _safe_external(model, location):
    if (
        not location
        or any(c in location for c in ('\\', '\0', ':'))
        or any(part in ('', '.', '..') for part in location.split('/'))
    ):
        raise ValueError('External data requires a safe relative path')
    path = Path(location)
    if path.is_absolute():
        raise ValueError('External data requires a safe relative path')
    return model.parent / path


def _tensor(stream, end, source, weights, location, attribute=False):
    metadata = bytearray()
    raw = None
    for field, wire, start, offset, length, stop in _fields(stream, end):
        if not attribute and field == 9 and wire == 2 and length > 1024:
            if raw is not None or length > MAX_TENSOR:
                raise ValueError('Invalid or oversized ONNX initializer')
            raw = offset, length
        else:
            if stop - start + len(metadata) > MAX_TENSOR + MAX_METADATA:
                raise ValueError('ONNX tensor metadata exceeds budget')
            stream.seek(start)
            metadata.extend(stream.read(stop - start))
    tensor = onnx.TensorProto.FromString(metadata)
    if weights is None or attribute and tensor.data_location != onnx.TensorProto.EXTERNAL:
        return tensor.SerializeToString()
    external = tensor.data_location == onnx.TensorProto.EXTERNAL
    if raw is not None and external:
        raise ValueError('ONNX tensor mixes inline and external weights')
    if external:
        info = onnx.external_data_helper.ExternalDataInfo(tensor)
        path = _safe_external(source, info.location)
        length = math.prod(tensor.dims) * np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)).itemsize
        offset = info.offset or 0
        if offset < 0 or length > MAX_TENSOR or (info.length is not None and info.length != length):
            raise ValueError('Invalid external weight offset/length')
        if offset + length > path.stat().st_size:
            raise ValueError('External weight range exceeds its file')
        if length <= 1024:
            with path.open('rb') as data:
                data.seek(offset)
                tensor.raw_data = data.read(length)
            tensor.ClearField('external_data')
            tensor.data_location = onnx.TensorProto.DEFAULT
            return tensor.SerializeToString()
    elif raw is not None:
        offset, length = raw
    else:
        # Keep shape/control constants inline so shape inference retains values.
        array = numpy_helper.to_array(tensor)
        if array.nbytes <= 1024:
            return tensor.SerializeToString()
        if array.nbytes > MAX_TENSOR:
            raise ValueError('ONNX initializer exceeds budget')
        raw = array.tobytes()
        length = len(raw)
    expected = math.prod(tensor.dims) * np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)).itemsize
    if length != expected:
        raise ValueError('ONNX weight length differs from shape/dtype')
    start = weights.tell()
    if external:
        with path.open('rb') as data:
            _copy_range(data, offset, length, weights)
    elif isinstance(raw, tuple):
        _copy_range(stream, offset, length, weights)
    else:
        weights.write(raw)
    for field in ('raw_data', 'float_data', 'int32_data', 'int64_data', 'double_data', 'uint64_data'):
        tensor.ClearField(field)
    tensor.ClearField('external_data')
    for key, value in [('location', location), ('offset', str(start)), ('length', str(length))]:
        entry = tensor.external_data.add()
        entry.key, entry.value = key, value
    tensor.data_location = onnx.TensorProto.EXTERNAL
    return tensor.SerializeToString()


def _rewrite(stream, end, kind, source, weights=None, location=''):
    children = {
        'model': {7: 'graph'},
        'graph': {1: 'node', 5: 'tensor'},
        'node': {5: 'attribute'},
        'attribute': {5: 'tensor', 10: 'tensor'},
    }
    result = bytearray()
    for field, wire, start, offset, length, stop in _fields(stream, end):
        child = children[kind].get(field)
        if child and wire == 2:
            stream.seek(offset)
            value = (
                _tensor(stream, stop, source, weights, location, attribute=kind == 'attribute')
                if child == 'tensor'
                else _rewrite(stream, stop, child, source, weights, location)
            )
            result.extend(_varint((field << 3) | 2) + _varint(len(value)) + value)
        else:
            if stop - start > MAX_METADATA:
                raise ValueError('ONNX metadata exceeds budget')
            stream.seek(start)
            result.extend(stream.read(stop - start))
        if len(result) > MAX_METADATA:
            raise ValueError('ONNX graph metadata exceeds budget')
    return bytes(result)


def model_metadata(source):
    """Read graph metadata without ever buffering inline raw weight fields."""
    source = Path(source)
    with source.open('rb') as stream:
        return onnx.ModelProto.FromString(_rewrite(stream, source.stat().st_size, 'model', source))


def externalize(source, destination):
    """Preserve the source; stream weights into one job-local OPFS data file."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    location = destination.name + '.data'
    with source.open('rb') as stream, (destination.parent / location).open('wb') as weights:
        graph = _rewrite(stream, source.stat().st_size, 'model', source, weights, location)
    destination.write_bytes(graph)
    onnx.checker.check_model(str(destination))
    return destination


def save_references(model, base_dir, destination):
    """Serialize only metadata, rebasing shared weight ranges without reading them."""
    destination = Path(destination)
    for tensor in onnx.external_data_helper._get_all_tensors(model):
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            info = onnx.external_data_helper.ExternalDataInfo(tensor)
            path = (Path(base_dir) / info.location).resolve()
            next(item for item in tensor.external_data if item.key == 'location').value = os.path.relpath(
                path, destination.parent
            )
    onnx.save_model(model, str(destination))


def tensor_array(tensor, base_dir):
    """Read one weight range directly into NumPy; never populate TensorProto.raw_data."""
    if tensor.data_location != onnx.TensorProto.EXTERNAL:
        return numpy_helper.to_array(tensor)
    info = onnx.external_data_helper.ExternalDataInfo(tensor)
    dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)).newbyteorder('<')
    count = math.prod(tensor.dims)
    with (Path(base_dir) / info.location).open('rb') as stream:
        stream.seek(info.offset or 0)
        array = np.fromfile(stream, dtype=dtype, count=count)
    if array.size != count:
        raise ValueError('Truncated external weight')
    return array.reshape(tensor.dims)
