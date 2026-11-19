#!/usr/bin/env python3
"""打包 retina split 参数加密集成夹具；仅包含明确指定的验收测试密钥。"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
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
    parser.add_argument('--wasm', type=Path, required=True)
    parser.add_argument('--acceptance', type=Path, required=True, help='已通过的六任务验收根目录')
    parser.add_argument('--test-keys', type=Path, required=True, help='该次验收导出的专用测试密钥父目录')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to('/tmp'):
        raise ValueError('交付产物必须位于 /tmp')
    accepted = args.acceptance.resolve()
    unit = accepted / 'retina/parameter'
    keys = json.loads((unit / 'keys.json').read_text())
    comparison = json.loads((unit / 'comparison.json').read_text())
    if not comparison['pass'] or comparison['max_abs_error'] >= 0.1:
        raise ValueError('retina 参数模式验收未通过')
    task_manifest = json.loads((unit / 'task/artifact-manifest.json').read_text())
    task = unit / 'task' / task_manifest['entry']
    split_directory = accepted / 'retina/split'
    split = json.loads((split_directory / 'split/split_manifest.json').read_text())
    if split['split']['ratio_text'] != '0.9':
        raise ValueError('交付只支持验收固定比例 0.9')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = output / 'retina-split-delivery'
    if base.exists():
        raise FileExistsError('请使用新的输出目录，或先清理旧暂存目录：' + str(base))
    base.mkdir()

    def copy(source, name):
        destination = base / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return destination

    # Read and verify runtime assets directly from the nested ZIP.
    with zipfile.ZipFile(args.wasm) as zipped:
        name = next(name for name in zipped.namelist() if name.endswith('/dist/manifest.json'))
        runtime = json.loads(zipped.read(name))
        prefix = name.removesuffix('manifest.json')
        for name, entry in runtime['files'].items():
            sha = hashlib.sha256()
            with zipped.open(prefix + name) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    sha.update(chunk)
            if sha.hexdigest() != entry['sha256']:
                raise ValueError('WASM 文件 hash 不匹配：' + name)
    copy(args.wasm, 'WASM-0.2.0.zip')
    for name in ['model/epoch041.onnx', 'img.csv', 'non_inv.csv', 'inv.csv', 'inputs.json']:
        copy(split_directory / 'source' / name, 'retina/' + name)
    copy(unit / 'task/compile-config.json', 'retina/compile-config.json')
    copy(unit / 'output.ct', 'fixture/output.ct')
    copy(unit / 'comparison.json', 'fixture/original-comparison.json')
    # Split working data also includes suffix weights. Repack only the prefix's
    # references before handing the graph to the data owner.
    sys.path[:0] = [str(ROOT.parents[2] / 'tools'), str(ROOT.parents[2])]
    import onnx
    from evaluate_onnx_csv import expand_polynomials
    from onnx_external_weights import externalize

    prefix_directory = base / 'handoff/prefix'
    prefix_path = externalize(split_directory / 'split/prefix.onnx', prefix_directory / 'model.onnx')
    prefix_model = expand_polynomials(onnx.load(prefix_path, load_external_data=False))
    onnx.save_model(prefix_model, str(prefix_path))
    onnx.checker.check_model(str(prefix_path))
    contract = json.loads((task / 'input_contract.json').read_text())
    input_manifest = json.loads((base / 'retina/inputs.json').read_text())
    specification = dict(
        schema='latti-ai.retina-prefix.v1',
        model='model.onnx',
        externalData=['model.onnx.data'],
        inputs=input_manifest['inputs'],
        outputs=[{key: entry[key] for key in ('name', 'shape', 'dtype', 'feature_id')} for entry in contract['inputs']],
        sourceModelSHA256=split['source']['sha256'],
    )
    (prefix_directory / 'prefix.json').write_text(json.dumps(specification, ensure_ascii=False, indent=2) + '\n')
    for name in ['index.html', 'retina-check.mjs', 'serve.py']:
        copy(ROOT / 'delivery' / name, name)
    copy(ROOT / 'README.md', 'README.md')
    # The complete test profile is necessary: restore verifies public context
    # length/hash as well as the secret context and actual public-key identity.
    profile = keys['profileId']
    key_source = args.test_keys / profile
    manifest = json.loads((key_source / 'manifest.json').read_text())
    if manifest['keyIdentity'] != keys['keyIdentity'] or manifest['profile']['profileId'] != profile:
        raise ValueError('测试密钥身份与结果密文不匹配')
    copy(key_source / 'manifest.json', f'fixture/test-keys/{profile}/manifest.json')
    for ref in manifest['artifacts']:
        if ref['path'] not in ('public/eval_context.bin', 'secret/full_context.bin'):
            raise ValueError('测试密钥 manifest 包含非预期文件')
        source = key_source / ref['path']
        if str(source.stat().st_size) != ref['bytes'] or digest(source) != ref['sha256']:
            raise ValueError('测试密钥 hash 不匹配：' + ref['path'])
        copy(source, f'fixture/test-keys/{profile}/' + ref['path'])
    if {ref['role'] for ref in manifest['artifacts']} != {'public-eval', 'private-full'}:
        raise ValueError('测试密钥不完整')
    bindings = {}
    for name in [
        'retina/model/epoch041.onnx',
        'retina/img.csv',
        'retina/non_inv.csv',
        'retina/inv.csv',
        'retina/inputs.json',
        'retina/compile-config.json',
        'fixture/output.ct',
        'handoff/prefix/model.onnx',
        'handoff/prefix/model.onnx.data',
        'handoff/prefix/prefix.json',
    ]:
        file = base / name
        bindings[name] = dict(bytes=file.stat().st_size, sha256=digest(file))
    original_signature = json.loads((task / 'runner/task_signature.json').read_text())
    delivery = dict(
        schema='latti-ai.retina-split-delivery.v1',
        testOnly=True,
        mode='server_provisioned_runner',
        ratio='0.9',
        runtimeBuild=runtime['build'],
        fixtureRuntimeBuild=task_manifest['build'],
        profileId=profile,
        keyIdentity=keys['keyIdentity'],
        bindings=bindings,
        split=split['split'],
        crossings=[{k: entry[k] for k in ('name', 'shape', 'dtype', 'sha256')} for entry in split['inputs']],
        outputSignature=[entry for entry in original_signature['online'] if entry['phase'] == 'out'],
        provenance=dict(
            kind='真实验收 native Runner 结果；仅重放，不能证明本次新参数包已被原生执行',
            sourceTaskManifestSHA256=digest(unit / 'task/artifact-manifest.json'),
            nativeOutputSHA256=digest(unit / 'output.ct'),
            originalComparison=comparison,
            privateTestKeys='本包包含公开测试夹具的 full context；不得用于业务数据或生产密钥',
        ),
    )
    (base / 'delivery-manifest.json').write_text(json.dumps(delivery, ensure_ascii=False, indent=2) + '\n')
    (base / 'TEST-KEYS.md').write_text(
        '# 仅用于固定测试样本\n\n本包的 fixture/test-keys 含本地验收生成的测试私钥。'
        '它只用于解密本包固定样本的预生成结果、复测浏览器参数加密接口。'
        '任何收到交付包的人均可使用此密钥，禁止用于真实业务数据。'
        '生产流程必须在用户浏览器中重新 setup 并保管私钥；新密钥不能解密本包旧密文。'
        '演示静态服务器不公开 fixture/retina 文件，均由用户目录授权在本地读取。\n',
        encoding='utf8',
    )
    files = sorted(file for file in base.rglob('*') if file.is_file())
    (base / 'SHA256SUMS').write_text(
        ''.join(digest(file) + '  ' + file.relative_to(base).as_posix() + '\n' for file in files)
    )
    archive = output / 'retina-split-delivery.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as zipped:
        for file in sorted(base.rglob('*')):
            if file.is_file():
                # Nested ZIP and ciphertext/key entropy makes compression wasteful.
                compression = zipfile.ZIP_STORED if file.suffix in ('.zip', '.bin', '.ct') else zipfile.ZIP_DEFLATED
                zipped.write(file, str(file.relative_to(output)), compress_type=compression)
    (output / 'retina-split-delivery.zip.sha256').write_text(digest(archive) + '  ' + archive.name + '\n')
    print(archive)


if __name__ == '__main__':
    main()
