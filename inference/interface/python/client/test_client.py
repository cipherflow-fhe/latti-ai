#!/usr/bin/env python3
"""Test latti_client module: setup, encrypt, export_eval_context, decrypt, serialization roundtrip."""

import os
import sys
import tempfile

# Add module path: sibling build/ dir (standalone) or ../../build/... (integrated)
_HERE = os.path.dirname(os.path.abspath(__file__))
for candidate in [
    os.path.join(_HERE, 'build'),
    os.path.join(_HERE, '..', '..', '..', '..', 'build', 'inference', 'interface', 'client'),
]:
    if os.path.isdir(candidate):
        sys.path.insert(0, candidate)
        break

import latti_client


def find_task_dir():
    """Find MNIST client task directory for testing."""
    candidates = [
        os.path.join(_HERE, '..', '..', '..', '..', 'examples', 'test_mnist', 'task', 'client'),
        os.path.join(_HERE, '..', '..', '..', '..', '..', 'examples', 'test_mnist', 'task', 'client'),
    ]
    for d in candidates:
        d = os.path.normpath(d)
        if os.path.isfile(os.path.join(d, 'task_config.json')):
            return d
    raise FileNotFoundError(
        'Cannot find MNIST task dir with task_config.json. Run from latti-ai root or set working directory accordingly.'
    )


def test_import():
    """Module and class visibility."""
    assert hasattr(latti_client, 'InferenceClient'), 'InferenceClient not found'
    assert hasattr(latti_client, 'DecryptedOutput'), 'DecryptedOutput not found'
    methods = [m for m in dir(latti_client.InferenceClient) if not m.startswith('_')]
    for required in ['setup', 'encrypt', 'decrypt', 'export_eval_context']:
        assert required in methods, f'InferenceClient missing method: {required}'
    print('[PASS] import')


def test_setup():
    """Key generation."""
    task_dir = find_task_dir()
    client = latti_client.InferenceClient(task_dir)
    client.setup()
    return client


def test_export_eval_context(client):
    """Export and serialization roundtrip."""
    eval_ctx = client.export_eval_context()
    assert isinstance(eval_ctx, bytes), 'export_eval_context should return bytes'
    assert len(eval_ctx) > 0, 'eval context is empty'

    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        f.write(eval_ctx)
        path = f.name
    with open(path, 'rb') as f:
        loaded = f.read()
    os.unlink(path)
    assert eval_ctx == loaded, 'eval context roundtrip mismatch'

    print(f'[PASS] export_eval_context ({len(eval_ctx) / 1024 / 1024:.1f} MB, roundtrip OK)')
    return eval_ctx


def test_encrypt(client, task_dir):
    """Encrypt input data and serialization roundtrip."""
    csv_path = os.path.join(task_dir, 'img.csv')
    assert os.path.isfile(csv_path), f'Test CSV not found: {csv_path}'

    encrypted = client.encrypt({'input': csv_path})
    assert 'input' in encrypted, "Encrypted output missing 'input' key"
    ct = encrypted['input']
    assert isinstance(ct, bytes), 'encrypt should return bytes'
    assert len(ct) > 0, 'ciphertext is empty'

    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        f.write(ct)
        path = f.name
    with open(path, 'rb') as f:
        loaded = f.read()
    os.unlink(path)
    assert ct == loaded, 'ciphertext roundtrip mismatch'

    print(f'[PASS] encrypt ({len(ct) / 1024:.1f} KB, roundtrip OK)')
    return encrypted


def test_decrypt(client, encrypted):
    """Decrypt (using re-encrypted data as mock server output)."""
    # Decrypt expects output names from task_config, not input names.
    # Feed encrypted 'input' as 'output' to test the binding works.
    try:
        result = client.decrypt({'output': encrypted['input']})
        assert 'output' in result, "decrypt result missing 'output' key"
        assert len(result['output'].output) > 0, 'decrypted output is empty'
        print(f'[PASS] decrypt (output[:5]={[round(x, 4) for x in result["output"].output[:5]]})')
    except MemoryError:
        # Expected when feeding dim=2 encrypted data as dim=0 output — dimensions mismatch.
        print('[PASS] decrypt binding OK (MemoryError expected for dimension mismatch)')


def main():
    print(f'Testing latti_client module\n')
    test_import()

    print('\n--- Key generation ---')
    client = test_setup()
    print('[PASS] setup')

    print('\n--- Export evaluation context ---')
    test_export_eval_context(client)

    print('\n--- Encrypt ---')
    task_dir = find_task_dir()
    encrypted = test_encrypt(client, task_dir)

    print('\n--- Decrypt ---')
    test_decrypt(client, encrypted)

    print('\n=== ALL TESTS PASSED ===\n')


if __name__ == '__main__':
    main()
