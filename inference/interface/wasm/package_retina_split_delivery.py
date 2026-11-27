#!/usr/bin/env python3
"""Package the accepted raw-image Retina batch and its disposable replay keys."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('wasm', 'acceptance', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument(
        '--raw', type=Path, default=Path('/LattiAI-Task-Resources/retina_degree8_02/task/client/demo_10/dataset')
    )
    parser.add_argument('--model', type=Path, default=ROOT.parents[2] / 'runs/retina/model/epoch041.onnx')
    parser.add_argument('--name', default='retina-batch10-split09-encrypted-input-parameter-wasm-delivery')
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in ('', '.', '..'):
        raise ValueError('Delivery name must be a single directory name')
    accepted = args.acceptance.resolve()
    summary = json.loads((accepted / 'summary.json').read_text())
    comparison = json.loads((accepted / 'comparison.json').read_text())
    if (
        summary['status'] != 'passed'
        or comparison['status'] != 'passed'
        or comparison['max_abs_error'] >= 0.1
        or comparison['count'] != 10
    ):
        raise ValueError('A complete ten-sample acceptance with error <0.1 is required')
    if (
        summary['inputMode'] != 'client_ciphertext'
        or summary['loadingMode'] != 'runtime_lazy'
        or summary['ratio'] != '0.9'
    ):
        raise ValueError('Acceptance mode differs')
    keys = json.loads((accepted / 'keys.json').read_text())
    profile = keys['profileId']
    key_source = accepted / 'keys' / profile
    key_manifest = json.loads((key_source / 'manifest.json').read_text())
    if key_manifest['keyIdentity'] != keys['keyIdentity']:
        raise ValueError('Replay key identity differs')
    task_artifact = json.loads((accepted / 'task/artifact-manifest.json').read_text())
    task = accepted / 'task' / task_artifact['entry']
    config = json.loads((accepted / 'bundle/task_config.json').read_text())
    if config['input_mode'] != 'client_ciphertext' or config['parameter_loading_mode'] != 'runtime_lazy':
        raise ValueError('Native runner configuration differs')
    args.output.mkdir(parents=True, exist_ok=True)
    base = args.output.resolve() / args.name
    base.mkdir(exist_ok=False)

    def copy(source, name):
        target = base / name
        target.parent.mkdir(parents=True, exist_ok=True)
        # Accepted artifacts are immutable. Hard links avoid duplicating the
        # large evaluation context on the same filesystem during packaging.
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)
        return target

    with zipfile.ZipFile(args.wasm) as zipped:
        entry = next(name for name in zipped.namelist() if name.endswith('/dist/manifest.json'))
        runtime = json.loads(zipped.read(entry))
        prefix = entry.removesuffix('manifest.json')
        for name, ref in runtime['files'].items():
            sha = hashlib.sha256()
            with zipped.open(prefix + name) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    sha.update(chunk)
            if sha.hexdigest() != ref['sha256']:
                raise ValueError('WASM asset hash mismatch: ' + name)
    copy(args.wasm, 'WASM-0.2.0.zip')
    copy(args.model, 'retina/model/epoch041.onnx')
    for file in sorted(args.raw.rglob('*')):
        if file.is_file():
            copy(file, 'retina/dataset/' + file.relative_to(args.raw).as_posix())
    for file in sorted((accepted / 'prepared').rglob('*')):
        if file.is_file():
            copy(file, 'retina/prepared/' + file.relative_to(accepted / 'prepared').as_posix())
    copy(accepted / 'task/compile-config.json', 'retina/compile-config.json')
    for name in ('output.zip', 'reference.json', 'comparison.json', 'crossings.json', 'preprocessing-check.json'):
        copy(accepted / name, 'fixture/' + name)
    copy(accepted / 'keys/public_context.bin', 'fixture/public_context.bin')
    copy(key_source / 'manifest.json', f'fixture/test-keys/{profile}/manifest.json')
    for ref in key_manifest['artifacts']:
        if ref['path'] not in ('public/eval_context.bin', 'secret/full_context.bin'):
            raise ValueError('Unexpected test key artifact')
        file = key_source / ref['path']
        if str(file.stat().st_size) != ref['bytes'] or digest(file) != ref['sha256']:
            raise ValueError('Test key content changed')
        copy(file, f'fixture/test-keys/{profile}/' + ref['path'])
    for name in ('index.html', 'workflow-check.mjs', 'batch-workflow.mjs', 'serve.py'):
        # Code files use copies so later development cannot mutate a delivery.
        target = base / name
        shutil.copyfile(ROOT / 'delivery' / name, target)
    (base / 'processing').mkdir()
    shutil.copyfile(ROOT / 'src/tasks/retina/raw-preprocess.js', base / 'processing/preprocess.js')
    shutil.copyfile(ROOT / 'src/tasks/retina/raw-postprocess.js', base / 'processing/postprocess.js')
    shutil.copyfile(ROOT / 'README.md', base / 'README.md')
    bindings = {}
    for file in sorted(base.rglob('*')):
        if (
            file.is_file()
            and not str(file.relative_to(base)).startswith('fixture/test-keys/')
            and file.name != 'WASM-0.2.0.zip'
        ):
            bindings[file.relative_to(base).as_posix()] = dict(bytes=file.stat().st_size, sha256=digest(file))
    delivery = dict(
        schema='latti-ai.onnx-batch-delivery.v1',
        testOnly=True,
        mode='server_provisioned_runner',
        ratio='0.9',
        inputMode='client_ciphertext',
        parameterMode='encrypted_offline',
        loadingMode='runtime_lazy',
        count=10,
        modelPath='model/epoch041.onnx',
        profileId=profile,
        keyIdentity=keys['keyIdentity'],
        runtimeBuild=runtime['build'],
        fixtureRuntimeBuild=task_artifact['build'],
        keyProfile=dict(
            parameters=(task / 'provisioner/ckks_parameter.json').read_text(),
            bootstrap=key_manifest['profile']['keyPlan']['bootstrapping'],
        ),
        crossings=json.loads((accepted / 'crossings.json').read_text()),
        bindings=bindings,
        provenance=dict(
            kind='Real native GPU runtime_lazy ten-sample ciphertexts; replay bypasses only FHE execution',
            comparison=comparison,
            nativeOutputSHA256=digest(accepted / 'output.zip'),
            sourceTaskManifestSHA256=digest(accepted / 'task/artifact-manifest.json'),
        ),
    )
    (base / 'delivery-manifest.json').write_text(json.dumps(delivery, ensure_ascii=False, indent=2) + '\n')
    (base / 'TEST-KEYS.md').write_text(
        '# 公开演示测试密钥\n\n本包包含匹配固定结果密文的测试私钥，仅供交付校验，禁止用于业务数据。'
        '数据方生产交接只包含独立 prefix、公共任务元数据和 public_context.bin，不包含整个交付包。\n'
    )
    files = sorted(file for file in base.rglob('*') if file.is_file())
    (base / 'SHA256SUMS').write_text(
        ''.join(digest(file) + '  ' + file.relative_to(base).as_posix() + '\n' for file in files)
    )
    archive = args.output / (args.name + '.zip')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as zipped:
        for file in sorted(base.rglob('*')):
            if file.is_file():
                compression = zipfile.ZIP_STORED if file.suffix in ('.zip', '.bin', '.ct') else zipfile.ZIP_DEFLATED
                zipped.write(file, str(file.relative_to(args.output.resolve())), compress_type=compression)
    (args.output / (archive.name + '.sha256')).write_text(digest(archive) + '  ' + archive.name + '\n')
    print(archive)


if __name__ == '__main__':
    main()
