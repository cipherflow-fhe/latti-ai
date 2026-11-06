"""Build a local delivery from the runtime archive and explicit task/sample directories.

Task and data versions are their content hashes, independent of the WASM version.
Only pass sample directories selected for this delivery; private keys are rejected.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parent
PRIVATE = {'full_context.bin', 'secret_key.bin', 'key_manifest.json', 'key-manifest.json'}


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def archive(output, name, files, generated=None):
    generated = generated or {}
    records = {relative: digest(source) for relative, source in files.items()}
    records.update({relative: hashlib.sha256(data).hexdigest() for relative, data in generated.items()})
    version = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    path = output / (name + '-' + version[:12] + '.zip')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zipped:
        for relative, source in sorted(files.items()):
            zipped.write(source, relative)
        for relative, data in sorted(generated.items()):
            zipped.writestr(relative, data)
        zipped.writestr('content-manifest.json', json.dumps({'version': 'sha256:' + version, 'files': records}, indent=2) + '\n')
    return {'file': path.name, 'version': 'sha256:' + version, 'sha256': digest(path), 'bytes': str(path.stat().st_size)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wasm', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--retina-reference', type=Path, required=True,
                        help='Original gen_client_inputs_seg.py used for the prepared samples')
    for task in ('cifar10', 'retina'):
        parser.add_argument('--' + task + '-task', type=Path, required=True)
        parser.add_argument('--' + task + '-data', type=Path, required=True)
    args = parser.parse_args()
    version = json.loads((ROOT / 'package.json').read_text())['version']
    with zipfile.ZipFile(args.wasm) as zipped:
        candidates = [name for name in zipped.namelist() if name.endswith('/dist/manifest.json')]
        if len(candidates) != 1:
            raise ValueError('Expected one runtime manifest')
        runtime = json.loads(zipped.read(candidates[0]))
        if runtime['runtimeVersion'] != version or runtime['protocolVersion'] != 2:
            raise ValueError('Runtime package version/protocol mismatch')
        prefix = candidates[0][:-len('manifest.json')]
        for name, info in runtime['files'].items():
            with zipped.open(prefix + name) as stream:
                sha = hashlib.sha256()
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    sha.update(chunk)
            if sha.hexdigest() != info['sha256']:
                raise ValueError('Runtime file hash mismatch: ' + name)
        scripts = {name: zipped.read(prefix + name) for name in runtime['files']
                   if name.startswith('tasks/') or name in ('csv.js', 'io.js', 'zip.js')}
    args.output.mkdir(parents=True, exist_ok=True)
    wasm = args.output / ('WASM-' + version + '.zip')
    if args.wasm.resolve() != wasm.resolve():
        shutil.copyfile(args.wasm, wasm)
    manifest = {'runtimeVersion': version, 'protocolVersion': 2, 'build': runtime['build'],
                'repositories': runtime['repositories'], 'wasm': {'file': wasm.name, 'sha256': digest(wasm)}, 'tasks': {}}
    for name in ('cifar10', 'retina'):
        task, data = getattr(args, name + '_task').resolve(), getattr(args, name + '_data').resolve()
        config = json.loads((task / 'client/task_config.json').read_text())
        if not isinstance(config.get('evaluation_keys'), list):
            raise ValueError('Regenerate task key requirements: ' + name)
        task_files = {'client/' + file: task / 'client' / file for file in ('task_config.json', 'ckks_parameter.json')}
        task_scripts = {'client/runtime/' + file: scripts[file] for file in
                        ('csv.js', 'io.js', 'zip.js', 'tasks/classification.js')}
        for script in ('preprocess.js', 'postprocess.js'):
            source = scripts['tasks/' + name + '/' + script].decode()
            source = source.replace("'../../io.js'", "'./runtime/io.js'").replace("'../../csv.js'", "'./runtime/csv.js'")
            source = source.replace("'../classification.js'", "'./runtime/tasks/classification.js'")
            task_scripts['client/' + script] = source.encode()
        if name == 'retina':
            task_scripts['client/prepare.py'] = scripts['tasks/retina/prepare.py']
            task_scripts['client/README.md'] = scripts['tasks/retina/README.md']
            task_files['tools/gen_client_inputs_seg.py'] = args.retina_reference
        for file in (task / 'server').iterdir():
            if file.is_file() and file.suffix in ('.json', '.h5') and file.name not in PRIVATE:
                task_files['server/' + file.name] = file
        cases = json.loads((data / 'cases.json').read_text())
        if not cases or len({case['case_id'] for case in cases}) != len(cases):
            raise ValueError('Empty or duplicate cases: ' + name)
        data_files = {'cases.json': data / 'cases.json'}
        if name == 'retina':
            provenance = data / 'preprocessing.json'
            if json.loads(provenance.read_text())['referenceSHA256'] != digest(args.retina_reference):
                raise ValueError('Retina preprocessing reference does not match prepared samples')
            data_files['preprocessing.json'] = provenance
        for case in cases:
            if set(case['inputs']) != set(config['task_input_id']) or type(case['label']) is not int or not 0 <= case['label'] < (10 if name == 'cifar10' else 2):
                raise ValueError('Invalid case/label/input set: ' + name)
            for relative in [*case['inputs'].values(), *([case['image']] if 'image' in case else [])]:
                source = (data / relative).resolve()
                if not source.is_relative_to(data) or not source.is_file() or source.name in PRIVATE:
                    raise ValueError('Invalid sample file: ' + relative)
                data_files[source.relative_to(data).as_posix()] = source
        labels = ''.join(json.dumps({'case_id': case['case_id'], 'label': case['label']}) + '\n' for case in cases).encode()
        manifest['tasks'][name] = {'task': archive(args.output, name + '-task', task_files, task_scripts),
                                   'data': archive(args.output, name + '-data', data_files, {'labels.jsonl': labels}), 'cases': len(cases)}
    manifest_path = args.output / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    names = [wasm.name, 'manifest.json'] + [entry[kind]['file'] for entry in manifest['tasks'].values() for kind in ('task', 'data')]
    sums = args.output / 'SHA256SUMS'
    sums.write_text(''.join(digest(args.output / name) + '  ' + name + '\n' for name in sorted(names)))
    delivery = args.output / ('latti-delivery-' + version + '.zip')
    with zipfile.ZipFile(delivery, 'w', allowZip64=True) as zipped:
        for name in [*names, 'SHA256SUMS']:
            zipped.write(args.output / name, name)
    print(delivery)


if __name__ == '__main__':
    main()
