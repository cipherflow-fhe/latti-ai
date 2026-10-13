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

"""Run the inference example with Linux peak RSS recorded immediately after setup."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--task-dir', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build, task, output = args.build_dir.resolve(), args.task_dir.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = root / 'examples/inference.cpp'
    original = source.read_text()
    if original.count('    client.setup();') != 1:
        raise RuntimeError('Example setup measurement anchor changed')
    measured = '#include <chrono>\n#include <sys/resource.h>\n' + original.replace(
        '    client.setup();',
        """    struct rusage key_usage{};
    if (getrusage(RUSAGE_SELF, &key_usage) != 0) return 1;
    const auto before_key_peak = key_usage.ru_maxrss;
    const auto key_start = std::chrono::steady_clock::now();
    client.setup();
    if (getrusage(RUSAGE_SELF, &key_usage) != 0) return 1;
    const auto key_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - key_start).count();
    cout << "KEYGEN_MEMORY {\\"peak_rss_kib\\":" << key_usage.ru_maxrss
         << ",\\"before_setup_peak_rss_kib\\":" << before_key_peak
         << ",\\"elapsed_seconds\\":" << key_seconds << "}" << endl;""",
        1,
    )
    commands = json.loads((build / 'compile_commands.json').read_text())
    entry = next(command for command in commands if Path(command['file']).resolve() == source)
    compiler = entry.get('arguments') or shlex.split(entry['command'])
    linker = shlex.split((build / 'examples/CMakeFiles/inference.dir/link.txt').read_text())
    env = dict(os.environ, GOMAXPROCS='16', GOGC='100', GOMEMLIMIT='off')
    repos = {'latti_ai': root, 'lattisense': root / 'inference/lattisense'}
    repos['lattigo'] = repos['lattisense'] / 'fhe_ops_lib/lattigo'
    environment = {
        'measured_at_utc': datetime.now(timezone.utc).isoformat(),
        'platform': platform.platform(),
        'cpu_count': os.cpu_count(),
        'gomaxprocs': 16,
        'gogc': '100',
        'gomemlimit': 'off',
        'gpu': args.gpu,
        'gpu_info': subprocess.check_output(['nvidia-smi'], text=True) if args.gpu else None,
        'go_version': subprocess.check_output(['go', 'version'], cwd=repos['lattigo'], text=True).strip(),
        'commits': {
            name: subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip()
            for name, path in repos.items()
        },
        'source_sha256': hashlib.sha256(original.encode()).hexdigest(),
        'measured_source_sha256': hashlib.sha256(measured.encode()).hexdigest(),
        'lattisense_library_sha256': hashlib.sha256(
            (build / 'inference/lattisense/liblattisense.so').read_bytes()
        ).hexdigest(),
        'compiler': compiler,
        'linker': linker,
        'task_parameters': json.loads((task / 'client/ckks_parameter.json').read_text()),
        'task_config': json.loads((task / 'client/task_config.json').read_text()),
    }
    (output / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    with tempfile.TemporaryDirectory(prefix='latti-example-memory-') as temp_dir:
        temp = Path(temp_dir)
        copy, obj, binary = temp / 'inference.cpp', temp / 'inference.o', temp / 'inference'
        copy.write_text(measured)
        compiler[compiler.index('-o') + 1] = str(obj)
        compiler[compiler.index(str(source))] = str(copy)
        subprocess.run(compiler, cwd=entry['directory'], check=True)
        linker[linker.index('-o') + 1] = str(binary)
        linker = [str(obj) if arg.endswith('CMakeFiles/inference.dir/inference.cpp.o') else arg for arg in linker]
        linker = [arg for arg in linker if not arg.startswith('-Wl,--dependency-file=')]
        subprocess.run(linker, cwd=build / 'examples', check=True)
        command = [str(binary), '--task-dir', str(task), '--input', str(args.input.resolve()), '--verify']
        if args.gpu:
            command.append('--gpu')
        with (output / 'inference.txt').open('w') as log:
            subprocess.run(
                ['/usr/bin/time', '-v', '-o', str(output / 'process.time.txt'), *command],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    markers = [
        line for line in (output / 'inference.txt').read_text().splitlines() if line.startswith('KEYGEN_MEMORY ')
    ]
    if len(markers) != 1:
        raise RuntimeError('Expected exactly one setup memory measurement')
    result = json.loads(markers[0].removeprefix('KEYGEN_MEMORY '))
    result['eval_context_bytes'] = (task / 'client/eval_context.bin').stat().st_size
    (output / 'keygen-memory.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    print(f'Inference and memory logs: {output}')


if __name__ == '__main__':
    main()
