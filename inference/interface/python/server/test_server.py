#!/usr/bin/env python3
# Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Test latti_server: full client→server→client roundtrip."""

import ctypes
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for candidate in [
    os.path.join(_HERE, 'build'),
]:
    if os.path.isdir(candidate):
        sys.path.insert(0, candidate)
        break

for lib_dir in [
    os.path.join(_HERE, '..', '..', '..', '..', 'build', 'inference', 'lattisense'),
    os.path.join(_HERE, '..', '..', '..', '..', '..', 'build', 'inference', 'lattisense'),
]:
    lib_dir = os.path.normpath(lib_dir)
    for ext in ['.so', '.dll', '.dylib']:
        lib_path = os.path.join(lib_dir, 'liblattisense' + ext)
        if os.path.isfile(lib_path):
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            break
    else:
        continue
    break

import latti_server


def find_task_dirs():
    root_candidates = [
        os.path.join(_HERE, '..', '..', '..', '..'),
        os.path.join(_HERE, '..', '..', '..', '..', '..'),
    ]
    for root in root_candidates:
        root = os.path.normpath(root)
        client_dir = os.path.join(root, 'examples', 'test_mnist', 'task', 'client')
        server_dir = os.path.join(root, 'examples', 'test_mnist', 'task', 'server')
        if os.path.isfile(os.path.join(client_dir, 'task_config.json')) and os.path.isdir(server_dir):
            return client_dir, server_dir
    raise FileNotFoundError('Cannot find MNIST task directories.')


def test_import():
    assert hasattr(latti_server, 'InferenceServer')
    methods = [m for m in dir(latti_server.InferenceServer) if not m.startswith('_')]
    for required in ['import_eval_context', 'load_model', 'evaluate', 'evaluate_plaintext']:
        assert required in methods, f'missing method: {required}'
    print('[PASS] import')


def test_full_flow():
    client_dir, server_dir = find_task_dirs()

    client_build = os.path.join(_HERE, '..', 'client', 'build')
    if not os.path.isdir(client_build):
        print('[SKIP] latti_client build not found')
        return
    sys.path.insert(0, client_build)
    import latti_client

    print('\n--- Client: setup + encrypt ---')
    client = latti_client.InferenceClient(client_dir)
    client.setup()
    eval_ctx = client.export_eval_context()
    encrypted = client.encrypt({'input': os.path.join(client_dir, 'img.csv')})
    print(f'  eval_ctx={len(eval_ctx) / 1024 / 1024:.1f}MB, ciphertext={len(encrypted["input"]) / 1024:.1f}KB')

    print('\n--- Server: import + load + evaluate ---')
    server = latti_server.InferenceServer(server_dir, use_gpu=False)
    server.import_eval_context(eval_ctx)
    print('[PASS] import_eval_context')

    server.load_model()
    print('[PASS] load_model')

    result = server.evaluate(encrypted)
    assert 'output' in result
    print(f'[PASS] evaluate (result={len(result["output"]) / 1024:.1f}KB)')

    print('\n--- Client: decrypt ---')
    decrypted = client.decrypt(result)
    output = decrypted['output'].output
    predicted = output.index(max(output))
    print(f'[PASS] decrypt (predicted={predicted}, top5={[round(x, 4) for x in output[:5]]})')


def main():
    print('Testing latti_server module\n')
    test_import()
    test_full_flow()
    print('\n=== ALL TESTS PASSED ===\n')


if __name__ == '__main__':
    main()
