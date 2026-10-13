#!/usr/bin/env python3
"""Replay the actual Go SDK client paths in separate processes, without server work."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f'SDK measurement anchor changed: {old!r}')
    return source.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--ckks-n', type=int, default=65536)
    parser.add_argument('--gomaxprocs', type=int, default=16)
    parser.add_argument('--schemes', nargs='+', choices=['ckks', 'btp'], default=['ckks', 'btp'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    lattigo = root / 'inference/lattisense/fhe_ops_lib/lattigo'
    sdk = lattigo / 'go_sdk'
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GOMAXPROCS=str(args.gomaxprocs), GOGC='100', GOMEMLIMIT='off')
    environment = {
        'measured_at_utc': datetime.now(timezone.utc).isoformat(),
        'platform': platform.platform(),
        'cpu_count': os.cpu_count(),
        'gomaxprocs': args.gomaxprocs,
        'gogc': '100',
        'gomemlimit': 'off',
        'sampling_interval_ms': 250,
        'lattigo_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=lattigo, text=True).strip(),
        'latti_ai_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'probe_sha256': hashlib.sha256((root / 'examples/streaming_key_memory.go').read_bytes()).hexdigest(),
        'go_version': subprocess.check_output(['go', 'version'], cwd=lattigo, text=True).strip(),
        'cpu': Path('/proc/cpuinfo').read_text(),
        'memory': Path('/proc/meminfo').read_text(),
    }
    (output / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    with tempfile.TemporaryDirectory(prefix='latti-key-memory-') as temp_dir:
        temp = Path(temp_dir)
        main_source = replace_once((sdk / 'main.go').read_text(), 'func main() {}', '')
        for generation, label in [
            ('key := kgen.GenRelinearizationKey(context.sk, 1)', 'rlk'),
            ('key := kgen.GenSwitchingKeyForGalois(element, context.sk)', 'rtk'),
        ]:
            main_source = replace_once(
                main_source, generation, generation + f'\n\t\t\tmemorySample("{label}-generated", buffer)'
            )
        # The two payload writes in streamEvalKeys are the RLK and individual RTKs.
        for indent in ['\t\t', '\t\t\t']:
            anchor = f'_, err := buffer.WriteTo(writer)\n{indent}return err'
            main_source = replace_once(
                main_source,
                anchor,
                f'_, err := buffer.WriteTo(writer)\n{indent}memorySample("key-written", buffer)\n{indent}return err',
            )
        btp_source = (sdk / 'bootstrap.go').read_text()
        anchor = 'dts, std := context.btp_parameter.GenEncapsulationSwitchingKeys(*context.parameter, context.sk)'
        btp_source = replace_once(btp_source, anchor, anchor + '\n\t\tmemorySample("encapsulation-generated", &buffer)')
        (temp / 'main.go').write_text(main_source)
        (temp / 'bootstrap.go').write_text(btp_source)
        overlay = {
            'Replace': {
                str(sdk / 'main.go'): str(temp / 'main.go'),
                str(sdk / 'bootstrap.go'): str(temp / 'bootstrap.go'),
                str(sdk / 'streaming_key_memory.go'): str(root / 'examples/streaming_key_memory.go'),
            }
        }
        (temp / 'overlay.json').write_text(json.dumps(overlay))
        subprocess.run(
            ['cc', '-c', str(root / 'inference/lattisense/fhe_ops_lib/bridge.c'), '-o', str(temp / 'bridge.o')],
            check=True,
        )
        subprocess.run(['ar', 'rcs', str(temp / 'libmemory_bridge.a'), str(temp / 'bridge.o')], check=True)
        build_env = dict(env, CGO_LDFLAGS=f'{env.get("CGO_LDFLAGS", "")} -L{temp} -lmemory_bridge')
        subprocess.run(
            ['go', 'build', f'-overlay={temp / "overlay.json"}', '-o', str(temp / 'probe'), './go_sdk'],
            cwd=lattigo,
            env=build_env,
            check=True,
        )
        for scheme in args.schemes:
            size = args.ckks_n if scheme == 'ckks' else 65536
            for mode in ['batch-generate', 'batch', 'stream']:
                name = f'{scheme}-{mode}'
                destination = temp / 'eval_context.bin'
                print(f'Measuring {name} (N={size})', flush=True)
                with (output / f'{name}.jsonl').open('w') as log:
                    subprocess.run(
                        [
                            '/usr/bin/time',
                            '-v',
                            '-o',
                            str(output / f'{name}.time.txt'),
                            str(temp / 'probe'),
                            mode,
                            scheme,
                            str(size),
                            str(destination),
                        ],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                destination.unlink(missing_ok=True)
    print(f'Raw measurements: {output}', flush=True)


if __name__ == '__main__':
    main()
