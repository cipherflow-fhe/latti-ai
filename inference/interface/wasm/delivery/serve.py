#!/usr/bin/env python3
"""仅提供演示网页及 WASM 静态资源；不公开模型、测试密钥或密文目录。"""

import argparse
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    runtime = root / 'runtime'
    archive = root / 'WASM-0.2.0.zip'
    with zipfile.ZipFile(archive) as zipped:
        members = [name for name in zipped.namelist() if '/dist/' in name and not name.endswith('/')]
        for name in members:
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('非法 WASM ZIP 路径')
            relative = name.split('/dist/', 1)[1]
            destination = runtime / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(name) as source, destination.open('wb') as target:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    target.write(chunk)
    manifest = json.loads((runtime / 'manifest.json').read_text())
    for name, entry in manifest['files'].items():
        file = runtime / name
        digest = hashlib.sha256()
        with file.open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(chunk)
        if str(file.stat().st_size) != entry['bytes'] or digest.hexdigest() != entry['sha256']:
            raise ValueError('WASM 资源 hash 不匹配：' + name)

    class Handler(SimpleHTTPRequestHandler):
        def translate_path(self, path):
            path = super().translate_path(path)
            resolved = Path(path).resolve()
            if resolved in (root / 'index.html', root / 'retina-check.mjs'):
                return str(resolved)
            if resolved.is_relative_to(runtime.resolve()):
                return str(resolved)
            return str(root / '.not-served')

        def do_GET(self):
            if self.path in ('/', '/index.html'):
                self.path = '/index.html'
            return super().do_GET()

        def __init__(self, *arguments, **kwargs):
            super().__init__(*arguments, directory=str(root), **kwargs)

    print(f'打开 http://localhost:{args.port}/ ，选择本交付根目录。Ctrl+C 退出。', flush=True)
    with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
        server.serve_forever()


if __name__ == '__main__':
    main()
