import {createHash} from 'node:crypto';
import {readFile,readdir,writeFile} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
import {BUILD_INFO} from './src/build-info.js';

const root=import.meta.dirname;
const git=(directory,...args)=>execFileSync('git',['-C',directory,...args],{encoding:'utf8'}).trim();
const repositories={};
for(const [name,relative]of Object.entries({lattiAI:'../../..',lattisense:'../../lattisense',lattigo:'../../lattisense/fhe_ops_lib/lattigo',gpuCore:'../../lattisense/backends/lattisense-gpu-core'})) {
  const directory=path.resolve(root,relative);
  repositories[name]={revision:git(directory,'rev-parse','HEAD'),branch:git(directory,'branch','--show-current'),dirty:!!git(directory,'status','--porcelain')};
}
const files={};
async function collect(relative='') {
  for(const entry of(await readdir(path.join(root,'dist',relative),{withFileTypes:true})).sort((a,b)=>a.name<b.name?-1:a.name>b.name?1:0)) {
    const name=path.posix.join(relative,entry.name);
    if(entry.isDirectory()) await collect(name);
    else if(name!=='manifest.json') {
      const data=await readFile(path.join(root,'dist',name));files[name]={bytes:String(data.byteLength),sha256:createHash('sha256').update(data).digest('hex')};
    }
  }
}
await collect();
const manifest={...BUILD_INFO,repositories,toolchain:{go:execFileSync('go',['version'],{encoding:'utf8'}).trim(),node:process.version},defaultProfile:JSON.parse(await readFile(path.join(root,'src/default-profile.json'),'utf8')),files};
await writeFile(path.join(root,'dist/manifest.json'),JSON.stringify(manifest,null,2)+'\n');
const sums=[];
for(const name of[...Object.keys(files).map(name=>'dist/'+name),'dist/manifest.json','README.md','package.json'].sort()) {
  const bytes=await readFile(path.join(root,name));sums.push(createHash('sha256').update(bytes).digest('hex')+'  '+name);
}
await writeFile(path.join(root,'SHA256SUMS'),sums.join('\n')+'\n');
