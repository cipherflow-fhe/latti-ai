import {createHash} from 'node:crypto';
import {readFile, readdir, writeFile} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import path from 'node:path';

const root = import.meta.dirname;
const pkg = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8'));
const hash = createHash('sha256');
async function include(directory, extensions) {
  for (const entry of (await readdir(directory, {withFileTypes:true})).sort((a,b)=>a.name < b.name ? -1 : 1)) {
    if (entry.name.startsWith('.') || entry.name === 'build-info.js' || entry.name.endsWith('_test.go')) continue;
    const file=path.join(directory,entry.name);
    if (entry.isDirectory()) await include(file, extensions);
    else if (extensions.some(ext=>entry.name.endsWith(ext))) {
      hash.update(path.relative(root,file)); hash.update('\0'); hash.update(await readFile(file));
    }
  }
}
await include(path.join(root,'src'),['.js','.ts','.json','.py','.md','.html']);
await include(path.join(root,'core'),['.go']);
await include(path.join(root,'cmd'),['.go']);
await include(path.join(root,'../../lattisense/fhe_ops_lib/lattigo'),['.go']);
for (const file of ['package.json','go.mod','go.sum']) hash.update(await readFile(path.join(root,file)));
const goroot=execFileSync('go',['env','GOROOT'],{encoding:'utf8'}).trim();
hash.update(await readFile(path.join(goroot,'lib/wasm/wasm_exec.js')));
const info={runtimeVersion:pkg.version,protocolVersion:2,build:hash.digest('hex')};
await writeFile(path.join(root,'src/build-info.js'), `// Generated from package.json and the runtime sources by build-info.mjs.\nexport const BUILD_INFO = Object.freeze(${JSON.stringify(info)});\n`);
process.stdout.write(JSON.stringify(info)+'\n');
