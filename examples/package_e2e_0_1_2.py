#!/usr/bin/env python3
"""Build the combined ResNet20/Retina browser delivery archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


VERSION = '0.1.2'
WASM_NAME = f'WASM-{VERSION}.zip'
RESNET_TASK_NAME = f'resnet20-task-{VERSION}.zip'
RESNET_DATA_NAME = f'resnet20-data-{VERSION}.zip'
RETINA_TASK_NAME = f'retina-task-degree8-02-{VERSION}.zip'
RETINA_DATA_NAME = f'retina-data-degree8-02-{VERSION}.zip'
PACKAGE_NAME = f'latti-e2e-{VERSION}.zip'


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def add_path(archive: zipfile.ZipFile, source: Path, name: str) -> None:
    if source.is_dir():
        for child in sorted(source.rglob('*')):
            if child.is_file():
                archive.write(child, f'{name}/{child.relative_to(source).as_posix()}')
    else:
        archive.write(source, name)


def write_archive(output: Path, paths: list[tuple[Path, str]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source, name in paths:
            if not source.exists():
                raise SystemExit(f'missing package input: {source}')
            add_path(archive, source, name)


def task_paths(source: Path, root: str) -> list[tuple[Path, str]]:
    names = ['task_config.json', 'ckks_parameter.json', 'preprocess.js', 'postprocess.js']
    return [(source / name, f'tasks/{root}/client/{name}') for name in names]


def data_paths(source: Path, retina: bool = False) -> list[tuple[Path, str]]:
    names = ['label.csv']
    if retina:
        names = ['images', 'disk', 'lesion', 'risk_factor_5.xlsx', 'label.csv']
    else:
        names = ['images', 'label.csv']
    return [(source / name, f'dataset/{name}') for name in names]


def make_readme(path: Path) -> None:
    path.write_text(
        f"""# Latti browser end-to-end delivery {VERSION}

This archive contains the reusable WASM runtime and independent task/data
archives for ResNet20/CIFAR10 and Retina-DKD degree8_02.

| Archive | Contents |
| --- | --- |
| `{WASM_NAME}` | Browser key generation, restore, CSV encryption/decryption and server ZIP mode |
| `{RESNET_TASK_NAME}` | ResNet20 task config, preprocessing and postprocessing |
| `{RESNET_DATA_NAME}` | Ten CIFAR-10 images and `label.csv` |
| `{RETINA_TASK_NAME}` | Retina task config, preprocessing and postprocessing |
| `{RETINA_DATA_NAME}` | Ten original Retina images, disk/lesion masks, XLSX and `label.csv` |

Both tasks use the same `N16QP1546H192H32` BTP profile. Generate one
`eval_context.bin`/`full_context.bin` pair with `setup()`, then reuse it with
`restore()` when switching task directories. The full context stays local.

The task modules export the same functions:
`preprocess_batch({{inputDirectory, outputDirectory}})` and
`postprocess_batch({{inputDirectory, labelFile, outputDirectory}})`.
The WASM server mode writes `input.zip` and reads `output.zip` directly, so a
runner does not rename ciphertext files or create ZIP archives.
""",
        encoding='utf-8',
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--wasm-zip', type=Path, required=True)
    parser.add_argument('--resnet-task', type=Path, default=Path('examples/test_cifar10/task/client'))
    parser.add_argument('--resnet-data', type=Path, default=Path('../datasets/CIFAR-10/dataset'))
    parser.add_argument('--retina-task', type=Path, default=Path('examples/test_retina_dkd/task/client'))
    parser.add_argument(
        '--retina-data',
        type=Path,
        default=Path('/LattiAI-Task-Resources/retina_degree8_02/task/client/demo_10/dataset'),
    )
    parser.add_argument('--output-dir', type=Path, default=Path('/LattiAI-Task-Resources'))
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    wasm_zip = args.wasm_zip.resolve()
    resnet_task = (
        (root / args.resnet_task).resolve() if not args.resnet_task.is_absolute() else args.resnet_task.resolve()
    )
    resnet_data = (
        (root / args.resnet_data).resolve() if not args.resnet_data.is_absolute() else args.resnet_data.resolve()
    )
    retina_task = (
        (root / args.retina_task).resolve() if not args.retina_task.is_absolute() else args.retina_task.resolve()
    )
    retina_data = args.retina_data.resolve()
    output_dir = args.output_dir.resolve()
    for path in [wasm_zip, resnet_task, resnet_data, retina_task, retina_data]:
        if not path.exists():
            raise SystemExit(f'missing input: {path}')

    with tempfile.TemporaryDirectory(prefix='latti-e2e-package.') as temporary:
        stage = Path(temporary) / f'latti-e2e-{VERSION}'
        stage.mkdir(parents=True)
        wasm = stage / WASM_NAME
        shutil.copy2(wasm_zip, wasm)
        write_archive(stage / RESNET_TASK_NAME, task_paths(resnet_task, 'resnet20'))
        write_archive(stage / RESNET_DATA_NAME, data_paths(resnet_data))
        write_archive(stage / RETINA_TASK_NAME, task_paths(retina_task, 'retina_degree8_02'))
        write_archive(stage / RETINA_DATA_NAME, data_paths(retina_data, retina=True))
        make_readme(stage / 'README.md')

        archives = [
            wasm,
            stage / RESNET_TASK_NAME,
            stage / RESNET_DATA_NAME,
            stage / RETINA_TASK_NAME,
            stage / RETINA_DATA_NAME,
        ]
        manifest = {
            'package': f'latti-e2e-{VERSION}',
            'wasm': WASM_NAME,
            'tasks': {'resnet20': RESNET_TASK_NAME, 'retina_degree8_02': RETINA_TASK_NAME},
            'datasets': {'resnet20': RESNET_DATA_NAME, 'retina_degree8_02': RETINA_DATA_NAME},
            'ckks_profile': 'N16QP1546H192H32',
            'contexts': 'Generate once with setup(); reuse with restore(); contexts are not included.',
            'server_contract': {
                'input': 'input.zip with <case_id>.ct or <case_id>__<input_name>.ct',
                'output': 'output.zip with <case_id>.result.bin',
            },
        }
        (stage / 'manifest.json').write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
        )
        checksum_files = [stage / 'README.md', stage / 'manifest.json', *archives]
        (stage / 'SHA256SUMS').write_text(
            ''.join(f'{digest(path)}  {path.name}\n' for path in checksum_files), encoding='utf-8'
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / PACKAGE_NAME
        if output.exists():
            output.unlink()
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(stage.rglob('*')):
                if path.is_file():
                    archive.write(path, path.relative_to(stage.parent).as_posix())
        (output.with_suffix(output.suffix + '.sha256')).write_text(
            f'{digest(output)}  {output.name}\n', encoding='utf-8'
        )
        print(json.dumps({'package': str(output), 'bytes': output.stat().st_size, 'sha256': digest(output)}, indent=2))


if __name__ == '__main__':
    main()
