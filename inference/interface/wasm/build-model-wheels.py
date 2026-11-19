#!/usr/bin/env python3
"""Build the locked ONNX/ml_dtypes Pyodide wheels outside the source tree."""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tarfile
import urllib.request
import zipfile


def run(command, directory, environment):
    subprocess.run(command, cwd=directory, env=environment, check=True, timeout=1800)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root', type=Path, default=Path('/tmp/latti-model-wheel-build'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-cache', type=Path)
    parser.add_argument(
        '--record-hashes', action='store_true', help='Print candidate hashes instead of checking the release lock'
    )
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    lock = json.loads((package / 'model-resources.lock.json').read_text())
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Build wheels with a Python 3.12 virtual environment')
    for dependency, expected in [
        ('pyodide-build', '0.30.7'),
        ('pybind11', '2.13.6'),
        ('setuptools', '84.2.1'),
        ('protobuf', '5.29.2'),
        ('numpy', '2.0.2'),
    ]:
        if version(dependency) != expected:
            raise RuntimeError(f'Expected {dependency}=={expected}')
    if subprocess.check_output(['protoc', '--version'], text=True).strip() != 'libprotoc 3.21.12':
        raise RuntimeError('Expected protoc 3.21.12')
    if not shutil.which('emcc'):
        raise RuntimeError('Activate Emscripten 3.1.58 before running this script')
    root, output = args.build_root.resolve(), args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(
        os.environ,
        SOURCE_DATE_EPOCH=lock['buildTools']['sourceDateEpoch'],
        PYODIDE_XBUILDENV_PATH=str(root / 'xbuildenv'),
        MAX_JOBS='8',
        PYTHONUNBUFFERED='1',
    )
    environment['PATH'] = str(Path(sys.executable).parent) + os.pathsep + environment['PATH']
    for name, source in lock['sources'].items():
        archive = root / source['url'].rsplit('/', 1)[1]
        if not archive.exists() and args.source_cache and (args.source_cache / archive.name).exists():
            shutil.copyfile(args.source_cache / archive.name, archive)
        if not archive.exists():
            with urllib.request.urlopen(source['url'], timeout=120) as response, archive.open('wb') as file:
                shutil.copyfileobj(response, file)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != source['sha256']:
            raise RuntimeError(f'Source checksum mismatch: {name}')
        with tarfile.open(archive) as file:
            file.extractall(root, filter='data')
    run(
        ['pyodide', 'xbuildenv', 'install', '0.27.7', '--path', environment['PYODIDE_XBUILDENV_PATH']],
        root,
        environment,
    )
    prefix = root / 'protobuf-wasm'
    flags = f'-fPIC -fexceptions -O2 -ffile-prefix-map={root}=/build'
    run(
        [
            'emcmake',
            'cmake',
            '-S',
            str(root / 'protobuf-21.12/cmake'),
            '-B',
            str(root / 'protobuf-build'),
            '-DCMAKE_INSTALL_PREFIX=' + str(prefix),
            '-DCMAKE_BUILD_TYPE=Release',
            '-DCMAKE_CXX_FLAGS=' + flags,
            '-Dprotobuf_BUILD_TESTS=OFF',
            '-Dprotobuf_BUILD_PROTOC_BINARIES=OFF',
            '-Dprotobuf_BUILD_LIBPROTOC=OFF',
            '-Dprotobuf_WITH_ZLIB=OFF',
        ],
        root,
        environment,
    )
    run(['cmake', '--build', str(root / 'protobuf-build'), '-j8'], root, environment)
    run(['cmake', '--install', str(root / 'protobuf-build')], root, environment)
    source = root / 'onnx-1.19.0'
    cmake = source / 'CMakeLists.txt'
    text = cmake.read_text().replace(
        '  find_package(pybind11)',
        '  if(CMAKE_CROSSCOMPILING)\n    set(Python_FOUND FALSE)\n  endif()\n  find_package(pybind11)',
    )
    text = text.replace(
        '  pybind11_add_module(onnx_cpp2py_export MODULE "${ONNX_ROOT}/onnx/cpp2py_export.cc")',
        '  pybind11_add_module(onnx_cpp2py_export MODULE "${ONNX_ROOT}/onnx/cpp2py_export.cc")\n'
        '  if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")\n'
        '    set_target_properties(onnx_cpp2py_export PROPERTIES SUFFIX ".cpython-312-wasm32-emscripten.so")\n  endif()',
    )
    cmake.write_text(text)
    import pybind11

    include = root / 'xbuildenv/0.27.7/xbuildenv/pyodide-root/cpython/installs/python-3.12.7/include/python3.12'
    environment['CXXFLAGS'] = flags + f' -ffile-prefix-map={pybind11.get_include()}=/pybind11'
    environment['CFLAGS'] = environment['CXXFLAGS']
    environment['CMAKE_ARGS'] = shlex.join(
        [
            '-DPython3_INCLUDE_DIR=' + str(include),
            '-DONNX_USE_LITE_PROTO=ON',
            '-DONNX_GEN_PB_TYPE_STUBS=ON',
            '-DONNX_CUSTOM_PROTOC_EXECUTABLE=' + shutil.which('protoc'),
            '-DProtobuf_PROTOC_EXECUTABLE=' + shutil.which('protoc'),
            '-DProtobuf_INCLUDE_DIR=' + str(prefix / 'include'),
            '-DProtobuf_LITE_LIBRARY=' + str(prefix / 'lib/libprotobuf-lite.a'),
            '-DProtobuf_LIBRARY=' + str(prefix / 'lib/libprotobuf.a'),
            '-Dpybind11_DIR=' + pybind11.get_cmake_dir(),
        ]
    )
    wheels = root / 'wheels'
    run(
        [
            'pyodide',
            'build',
            str(source),
            '--outdir',
            str(wheels),
            '--exports',
            'PyInit_onnx_cpp2py_export,',
            '--no-isolation',
        ],
        root,
        environment,
    )
    environment.pop('CMAKE_ARGS')
    run(
        [
            'pyodide',
            'build',
            str(root / 'ml_dtypes-0.5.1'),
            '--outdir',
            str(wheels),
            '--exports',
            'pyinit',
            '--no-isolation',
            '--skip-dependency-check',
        ],
        root,
        environment,
    )
    hashes = {}
    for file in sorted(wheels.glob('*.whl')):
        destination = output / file.name
        with (
            zipfile.ZipFile(file) as original,
            zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as normalized,
        ):
            for name in sorted(original.namelist()):
                info = zipfile.ZipInfo(name, (2025, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = original.getinfo(name).external_attr
                normalized.writestr(info, original.read(name))
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        hashes[file.name] = digest
        expected = next(asset['sha256'] for asset in lock['assets'] if asset['path'].endswith('/' + file.name))
        if not args.record_hashes and digest != expected:
            raise RuntimeError(f'Rebuilt wheel differs from release lock: {file.name}: {digest}')
    (output / 'wheel-hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps(hashes, indent=2))


if __name__ == '__main__':
    main()
