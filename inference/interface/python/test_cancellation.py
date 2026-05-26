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

"""Test latti_inference cancellation during a real encrypted evaluation."""

import os
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
for candidate in [
    os.path.join(_HERE, 'build'),
]:
    if os.path.isdir(candidate):
        sys.path.insert(0, candidate)
        break

import latti_inference


def find_task_dirs():
    root_candidates = [
        os.path.join(_HERE, '..', '..', '..'),
        os.path.join(_HERE, '..', '..', '..', '..'),
    ]
    for root in root_candidates:
        root = os.path.normpath(root)
        client_dir = os.path.join(root, 'examples', 'test_mnist', 'task', 'client')
        server_dir = os.path.join(root, 'examples', 'test_mnist', 'task', 'server')
        if os.path.isfile(os.path.join(client_dir, 'task_config.json')) and os.path.isdir(server_dir):
            return client_dir, server_dir
    raise FileNotFoundError('Cannot find MNIST task directories.')


def test_cancel_during_evaluate():
    client_dir, server_dir = find_task_dirs()

    client = latti_inference.InferenceClient(client_dir)
    client.setup()
    eval_ctx = client.export_eval_context()
    encrypted = client.encrypt({'input': os.path.join(client_dir, 'img.csv')})

    server = latti_inference.InferenceServer(server_dir, use_gpu=False)
    server.import_eval_context(eval_ctx)
    server.load_model()

    progress_events = []
    progress_seen = threading.Event()
    outcome = {}

    def progress_callback(completed, total):
        progress_events.append((completed, total))
        progress_seen.set()

    def run_evaluate():
        try:
            server.evaluate(encrypted, progress_callback)
        except RuntimeError as exc:
            outcome['exception'] = exc
        else:
            outcome['completed'] = True

    worker = threading.Thread(target=run_evaluate)
    worker.start()

    assert progress_seen.wait(timeout=30), 'progress callback was not called before timeout'
    server.request_cancel()
    worker.join(timeout=30)

    assert not worker.is_alive(), 'evaluate() did not return after request_cancel()'
    assert 'completed' not in outcome, 'evaluate() completed despite request_cancel()'
    exc = outcome.get('exception')
    assert exc is not None, 'evaluate() returned without an exception'
    assert 'FHE task was cancelled' in str(exc), f'unexpected exception: {exc}'

    assert progress_events, 'progress callback was not called'
    print(f'[PASS] cancellation (first_progress={progress_events[0]})')


def main():
    print('Testing latti_inference cancellation\n')
    test_cancel_during_evaluate()
    print('\n=== ALL TESTS PASSED ===\n')


if __name__ == '__main__':
    main()
