# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Validate logical split inputs against the compiler's actual packing contract."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path, suffix_path=None, mat_pack_style=None):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest['schema_version'] != 'onnx_split_v1':
        raise ValueError('Unknown split manifest schema')
    for group in [manifest['source']['files'], *manifest['model_files'].values()]:
        for entry in group:
            if file_hash(entry['path']) != entry['sha256']:
                raise ValueError(f'Model/external data hash mismatch: {entry["path"]}')
    for entry in [*manifest['original_inputs'], *manifest.get('artifacts', [])]:
        artifact = path.parent / entry.get('csv', entry.get('path', ''))
        if file_hash(artifact) != entry['sha256']:
            raise ValueError(f'Input/reference artifact hash mismatch: {artifact}')
    suffix = (path.parent / manifest['models']['suffix']).resolve()
    if suffix_path is not None and Path(suffix_path).resolve() != suffix:
        raise ValueError('Compilation input is not the suffix declared in split manifest')
    names = []
    for entry in manifest['inputs']:
        names.append(entry['name'])
        csv = path.parent / entry['csv']
        if file_hash(csv) != entry['sha256']:
            raise ValueError(f'Boundary CSV hash mismatch: {csv}')
        values = np.loadtxt(csv, delimiter=',', dtype=entry['dtype'], ndmin=1)
        if values.size != entry['element_count'] or values.size != int(np.prod(entry['shape'])):
            raise ValueError(f'Boundary CSV shape/count mismatch: {entry["name"]}')
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite boundary CSV: {entry["name"]}')
        if np.dtype(entry['dtype']).kind != 'f':
            raise ValueError(f'Nonfloating boundary requires a supported control-tensor adapter: {entry["name"]}')
        representation = entry['representation']
        if mat_pack_style is not None and (representation == 'FeatureMat') != bool(mat_pack_style):
            raise ValueError(f'Compiler packing disagrees with logical representation: {entry["name"]}')
        if entry['axes'][0] == 'N' and entry['shape'][0] != 1:
            raise ValueError('FHE input batching currently requires N=1; split preparation preserves full shape')
    if len(names) != len(set(names)):
        raise ValueError('Duplicate logical suffix input')
    return manifest


def write_contract(manifest_path, task_dir):
    from training.model_export.operations import format_id

    manifest_path, task_dir = Path(manifest_path).resolve(), Path(task_dir)
    manifest = load_manifest(manifest_path)
    config_path = task_dir / 'server' / 'task_config.json'
    config = json.loads(config_path.read_text())
    ckks_path = task_dir / 'server' / 'ckks_parameter.json'
    ckks = json.loads(ckks_path.read_text())
    if any(param['param_name'] != 'PN15Default' for param in ckks.values()):
        raise ValueError('Split acceptance requires PN15Default')
    expected = {format_id(entry['name']) for entry in manifest['inputs']}
    if len(expected) != len(manifest['inputs']) or expected != set(config['task_input_param']):
        raise ValueError('Compiler input IDs differ or collide after normalization')
    inputs = []
    for entry in manifest['inputs']:
        fid = format_id(entry['name'])
        param = config['task_input_param'][fid]
        representation = entry['representation']
        expected_dim = {'Feature0D': 0, 'Feature1D': 1, 'Feature2D': 2, 'FeatureMat': 2}[representation]
        if param['dim'] != expected_dim or (param.get('data_type') == 'feature_mat') != (
            representation == 'FeatureMat'
        ):
            raise ValueError(f'Compiled Feature representation differs: {entry["name"]}')
        shape = entry['shape']
        expected_shape = shape[-2:] if representation == 'FeatureMat' else shape[2:]
        if expected_dim and param['shape'] != expected_shape:
            raise ValueError(f'Compiled input geometry differs: {entry["name"]}: {param["shape"]} vs {shape}')
        expected_channels = 1 if representation == 'FeatureMat' else shape[1]
        if param['channel'] != expected_channels:
            raise ValueError(f'Compiled input channels differ: {entry["name"]}')
        inputs.append({**entry, 'feature_id': fid, 'csv': str(manifest_path.parent / entry['csv']), 'packing': param})
    contract = {
        'schema_version': 'onnx_split_input_contract_v1',
        'manifest': str(manifest_path),
        'manifest_sha256': file_hash(manifest_path),
        'suffix_sha256': file_hash(manifest_path.parent / manifest['models']['suffix']),
        'deployment_mode': config.get('deployment_mode', 'client_encrypted_input'),
        'pack_style': config['pack_style'],
        'mat_pack_style': config.get('mat_pack_style', ''),
        'block_shape': config['block_shape'],
        'n_heads': config.get('n_heads'),
        'head_dim': config.get('head_dim'),
        'matmul_block_size': config.get('matmul_block_size'),
        'ckks': ckks,
        'ckks_sha256': file_hash(ckks_path),
        'inputs': inputs,
        'config_sha256': file_hash(config_path),
        'outputs': config['task_output_param'],
        'output_map': {
            format_id(route['name']): route['name'] for route in manifest['output_routes'] if route['route'] == 'suffix'
        },
    }
    encoded = json.dumps(contract, indent=2) + '\n'
    (task_dir / 'input_contract.json').write_text(encoded)
    for role in ('client', 'server', 'runner', 'provisioner'):
        directory = task_dir / role
        if directory.is_dir():
            (directory / 'input_contract.json').write_text(encoded)
    return contract


def validate_signature_size(signature, packing, contract):
    size = signature['size']
    if len(size) != 1 or not isinstance(size[0], int) or size[0] <= 0:
        raise ValueError(f'Invalid graph signature size: {signature["id"]}')
    if packing.get('data_type') == 'feature_mat':
        if contract['mat_pack_style'] != 'par_block_col_major':
            return  # Existing PDM encoder validates its consumer-specific count.
        block, heads = contract['matmul_block_size'], contract['n_heads']
        slots = contract['ckks'][packing['ckks_parameter_id']]['n_slots']
        padded_heads = 1 << (heads - 1).bit_length()
        capacity = slots // (block * block)
        groups = 1 if capacity >= padded_heads else (heads if capacity == 1 else padded_heads // capacity)
        count = math.prod(math.ceil(dim / block) for dim in packing['head_shape']) * groups
    else:
        count = math.ceil(packing['channel'] / packing['pack_num'])
        if packing['dim'] == 2:
            count *= math.prod(math.ceil(dim / block) for dim, block in zip(packing['shape'], contract['block_shape']))
    if size != [count]:
        raise ValueError(f'Graph signature count differs from packing: {signature["id"]}: {size} vs {count}')


def validate_contract(task_dir):
    """Gate invocation before CSV encoding, using existing Client/Runner config."""
    task_dir = Path(task_dir)
    contract = json.loads((task_dir / 'input_contract.json').read_text())
    manifest = Path(contract['manifest'])
    if file_hash(manifest) != contract['manifest_sha256']:
        raise ValueError('Split manifest changed after compilation')
    load_manifest(manifest)
    parameter_mode = contract['deployment_mode'] == 'server_provisioned_runner'
    roles = ('runner', 'provisioner') if parameter_mode else ('client', 'server')
    for role_name in roles:
        role = task_dir / role_name
        config = json.loads((role / 'task_config.json').read_text())
        if json.loads((role / 'input_contract.json').read_text()) != contract:
            raise ValueError(f'Role input contract differs: {role_name}')
        if file_hash(role / 'ckks_parameter.json') != contract['ckks_sha256']:
            raise ValueError(f'CKKS context differs from input contract: {role_name}')
        if (
            config['pack_style'] != contract['pack_style']
            or config.get('mat_pack_style', '') != contract['mat_pack_style']
            or config['block_shape'] != contract['block_shape']
            or config['task_output_param'] != contract['outputs']
            or set(config['task_input_param']) != {e['feature_id'] for e in contract['inputs']}
        ):
            raise ValueError(f'Task interfaces differ from contract: {role_name}')
        if any(config.get(key) != contract[key] for key in ('n_heads', 'head_dim', 'matmul_block_size')):
            raise ValueError(f'Matrix packing context differs from contract: {role_name}')
        for entry in contract['inputs']:
            if config['task_input_param'][entry['feature_id']] != entry['packing']:
                raise ValueError(f'Input layout differs from contract: {entry["feature_id"]}')
    signature_role = task_dir / ('runner' if parameter_mode else 'server')
    signature = json.loads((signature_role / 'task_signature.json').read_text())
    inputs = {
        arg['id']: arg
        for arg in signature['online']
        if arg['phase'] == 'in' and arg['type'] == ('pt' if parameter_mode else 'ct')
    }
    if set(inputs) != {entry['feature_id'] for entry in contract['inputs']}:
        raise ValueError('Graph signature input IDs differ from contract')
    for entry in contract['inputs']:
        sig = inputs[entry['feature_id']]
        expected_level = 0 if parameter_mode else entry['packing']['level']
        if 'level' in sig and sig['level'] != expected_level:
            raise ValueError(f'Graph signature level differs: {entry["name"]}')
        validate_signature_size(sig, entry['packing'], contract)
    outputs = {arg['id'] for arg in signature['online'] if arg['phase'] == 'out' and arg['type'] == 'ct'}
    if outputs != set(contract['output_map']):
        raise ValueError('Graph signature output IDs differ from contract')
    for sig in signature['online']:
        if sig['phase'] == 'out' and sig['type'] == 'ct':
            packing = contract['outputs'][sig['id']]
            validate_signature_size(sig, packing, contract)
            if sig.get('level') != packing['level']:
                raise ValueError(f'Graph signature output level differs: {sig["id"]}')
    return contract
