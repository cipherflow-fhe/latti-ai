#!/usr/bin/env python3
"""Test latti_client: setup, encrypt, export_eval_context, export_full_context, decrypt."""

import csv
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
for candidate in [
    os.path.join(_HERE, 'build'),
    os.path.join(_HERE, '..', '..', '..', '..', 'build', 'inference', 'interface', 'client'),
]:
    if os.path.isdir(candidate):
        sys.path.insert(0, candidate)
        break

import latti_client

TEST_TASK_DIR = os.path.join(_HERE, 'testdata')


def test_import():
    assert hasattr(latti_client, 'InferenceClient')
    assert hasattr(latti_client, 'DecryptedOutput')
    methods = [m for m in dir(latti_client.InferenceClient) if not m.startswith('_')]
    for required in ['setup', 'encrypt', 'decrypt', 'export_eval_context',
                     'export_full_context', 'load_full_context']:
        assert required in methods, f'missing method: {required}'
    print('[PASS] import')


def test_setup():
    client = latti_client.InferenceClient(TEST_TASK_DIR)
    client.setup()
    return client


def test_export_eval_context(client):
    eval_ctx = client.export_eval_context()
    assert isinstance(eval_ctx, bytes) and len(eval_ctx) > 0

    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        f.write(eval_ctx)
        path = f.name
    with open(path, 'rb') as f:
        loaded = f.read()
    os.unlink(path)
    assert eval_ctx == loaded, 'roundtrip mismatch'

    print(f'[PASS] export_eval_context ({len(eval_ctx) / 1024 / 1024:.1f} MB)')
    return eval_ctx


def test_encrypt(client):
    csv_path = os.path.join(TEST_TASK_DIR, 'img.csv')
    assert os.path.isfile(csv_path), f'not found: {csv_path}'

    encrypted = client.encrypt({'input': csv_path})
    ct = encrypted['input']
    assert isinstance(ct, bytes) and len(ct) > 0

    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        f.write(ct)
        path = f.name
    with open(path, 'rb') as f:
        loaded = f.read()
    os.unlink(path)
    assert ct == loaded, 'roundtrip mismatch'

    print(f'[PASS] encrypt ({len(ct) / 1024:.1f} KB)')
    return encrypted


def test_decrypt(client, encrypted):
    result = client.decrypt({'output': encrypted['input']})
    output = result['output']
    assert output.num_outputs > 0

    with open(os.path.join(TEST_TASK_DIR, 'img.csv')) as f:
        original = [float(x) for x in next(csv.reader(f))]

    decrypted = output.output[: len(original)]
    max_error = max(abs(a - b) for a, b in zip(original, decrypted))
    assert max_error < 0.01, f'error too large: {max_error}'
    print(f'[PASS] decrypt (num_outputs={output.num_outputs}, max_error={max_error:.6f})')


def test_export_full_context(client, eval_ctx):
    full_ctx = client.export_full_context()
    assert isinstance(full_ctx, bytes) and len(full_ctx) > 0
    # full_ctx includes the secret key; eval_ctx does not. So full_ctx must be strictly larger.
    # This catches regressions where make_public_context() is inadvertently called.
    assert len(full_ctx) > len(eval_ctx), \
        f'full_ctx ({len(full_ctx)}) should be larger than eval_ctx ({len(eval_ctx)})'
    delta_mb = (len(full_ctx) - len(eval_ctx)) / 1024 / 1024
    print(f'[PASS] export_full_context ({len(full_ctx) / 1024 / 1024:.1f} MB, +{delta_mb:.1f} MB vs eval)')
    return full_ctx


def test_load_full_context_roundtrip(full_ctx):
    """Fresh client + load_full_context should support both encrypt and decrypt."""
    fresh = latti_client.InferenceClient(TEST_TASK_DIR)
    fresh.load_full_context(full_ctx)

    csv_path = os.path.join(TEST_TASK_DIR, 'img.csv')
    encrypted = fresh.encrypt({'input': csv_path})
    assert isinstance(encrypted['input'], bytes) and len(encrypted['input']) > 0

    result = fresh.decrypt({'output': encrypted['input']})
    output = result['output']
    assert output.num_outputs > 0

    with open(csv_path) as f:
        original = [float(x) for x in next(csv.reader(f))]
    decrypted = output.output[: len(original)]
    max_error = max(abs(a - b) for a, b in zip(original, decrypted))
    assert max_error < 0.01, f'error too large: {max_error}'
    print(f'[PASS] load_full_context roundtrip (max_error={max_error:.6f})')


def main():
    print('Testing latti_client module\n')
    test_import()

    print('\n--- Key generation ---')
    client = test_setup()
    print('[PASS] setup')

    print('\n--- Export evaluation context ---')
    eval_ctx = test_export_eval_context(client)

    print('\n--- Export full context ---')
    full_ctx = test_export_full_context(client, eval_ctx)

    print('\n--- Encrypt ---')
    encrypted = test_encrypt(client)

    print('\n--- Decrypt (encrypt-decrypt roundtrip) ---')
    test_decrypt(client, encrypted)

    print('\n--- Load full context in fresh client + roundtrip ---')
    test_load_full_context_roundtrip(full_ctx)

    print('\n=== ALL TESTS PASSED ===\n')


if __name__ == '__main__':
    main()
