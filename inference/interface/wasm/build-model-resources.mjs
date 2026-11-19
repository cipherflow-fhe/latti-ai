import {createHash} from 'node:crypto';
import {readFile,mkdir,copyFile,writeFile,rename} from 'node:fs/promises';
import path from 'node:path';
import {modelSourceFiles,repositoryRoot} from './model-sources.mjs';

const root=import.meta.dirname;
const lock=JSON.parse(await readFile(path.join(root,'model-resources.lock.json'),'utf8'));
const cache=process.env.LATTI_MODEL_RESOURCE_CACHE||path.join(root,'node_modules/.cache/latti-model');
const destination=path.join(root,'dist/model');
const timeout=Number(process.env.LATTI_MODEL_RESOURCE_TIMEOUT_MS||300000);
if(!Number.isSafeInteger(timeout)||timeout<=0)throw new Error('Invalid model resource timeout');
await mkdir(cache,{recursive:true});
await mkdir(destination,{recursive:true});
for(const asset of lock.assets) {
  let bytes;
  if(asset.npm)bytes=await readFile(path.join(root,'node_modules',asset.npm));
  else {
    const cached=path.join(cache,path.basename(asset.path));
    try {bytes=await readFile(cached);} catch(error) {
      if(error.code!=='ENOENT')throw error;
      if(!asset.url)throw new Error(`Build the locked ONNX wheel first: ${cached}`);
      const response=await fetch(asset.url,{signal:AbortSignal.timeout(timeout)});
      if(!response.ok)throw new Error(`Model resource HTTP ${response.status}: ${asset.path}`);
      bytes=Buffer.from(await response.arrayBuffer());
      if(createHash('sha256').update(bytes).digest('hex')!==asset.sha256)throw new Error(`Model resource hash mismatch: ${asset.path}`);
      await writeFile(cached+'.part',bytes);await rename(cached+'.part',cached);
    }
  }
  if(bytes.byteLength!==asset.bytes||createHash('sha256').update(bytes).digest('hex')!==asset.sha256)
    throw new Error(`Model resource hash/size mismatch: ${asset.path}`);
  const output=path.join(destination,asset.path);
  await mkdir(path.dirname(output),{recursive:true});await writeFile(output,bytes);
}
const sources={};
for(const name of await modelSourceFiles()) {
  const input=path.join(repositoryRoot,name),output=path.join(destination,'python',name);
  const bytes=await readFile(input);
  sources[name]={bytes:bytes.byteLength,sha256:createHash('sha256').update(bytes).digest('hex')};
  await mkdir(path.dirname(output),{recursive:true});await copyFile(input,output);
}
await writeFile(path.join(destination,'sources.json'),JSON.stringify(sources,null,2)+'\n');
await writeFile(path.join(destination,'pyodide/python-lock.json'),JSON.stringify(lock.pythonLock,null,2)+'\n');
await copyFile(path.join(root,'model-resources.lock.json'),path.join(destination,'resources.json'));
