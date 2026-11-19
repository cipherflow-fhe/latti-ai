import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
import {browserHarness} from './browser-harness.mjs';
import {newSampleOutputRoutes} from './model-regressions.mjs';
const root=path.resolve(process.env.LATTI_MODEL_FAILURE_ROOT||'/tmp/latti-browser-model-failures');
await mkdir(root,{recursive:true});
execFileSync(process.env.LATTI_MODEL_NATIVE_PYTHON||'python3',[path.join(import.meta.dirname,'model-failure-fixtures.py'),path.join(root,'fixtures')]);
const harness=await browserHarness(root,{'/fixtures/':path.join(root,'fixtures'),'/cifar/':path.resolve(import.meta.dirname,'../../../../runs/cifar10')});
const checks=[];
async function check(name,action){
 console.log('[model-failures]',name);await harness.beginPhase(name);
 try {
  const result=await action();
  const workspaces=await harness.page.evaluate(async()=>{
   try{const directory=await(await navigator.storage.getDirectory()).getDirectoryHandle('latti-python-workspaces');return await Array.fromAsync(directory.keys());}
   catch(error){if(error.name==='NotFoundError')return [];throw error;}
  });
  checks.push({name,status:workspaces.length?'failed':'passed',result,workspaces});
  await writeFile(path.join(root,'checks.json'),JSON.stringify(checks,null,2));
  if(workspaces.length)throw new Error(name+': Python workspace retained: '+workspaces);
 }catch(error){await writeFile(path.join(root,'failed-check.json'),JSON.stringify({name,error:error.message},null,2));throw error;}
}
const runtime=path.resolve(process.env.LATTI_RUNTIME_DIRECTORY||path.join(import.meta.dirname,'../dist'));
try {
 await harness.page.evaluate(async()=>{
  globalThis.api=await import('/dist/index.js');globalThis.opfs=await navigator.storage.getDirectory();
  globalThis.write=async(directory,name,data)=>{const parts=name.split('/');for(const part of parts.slice(0,-1))directory=await directory.getDirectoryHandle(part,{create:true});const file=await directory.getFileHandle(parts.at(-1),{create:true}),writer=await file.createWritable();await writer.write(data);await writer.close();return file;};
  globalThis.assertEmpty=async directory=>{if((await Array.fromAsync(directory.values())).length)throw new Error('Uncommitted output retained');};
  globalThis.sourceDirectory=await opfs.getDirectoryHandle('source',{create:true});
  for(const name of ['model.onnx','weights.data','input.csv','inputs.json','cycle.onnx','forwarded.onnx','identity.onnx','reshape.onnx','transpose.onnx','control.onnx'])await write(sourceDirectory,name,await(await fetch('/fixtures/'+name)).arrayBuffer());
  globalThis.source={directory:sourceDirectory,modelPath:'model.onnx',inputsManifest:await sourceDirectory.getFileHandle('inputs.json')};
  globalThis.output=await opfs.getDirectoryHandle('output',{create:true});
 });
 await check('wheel external data and ORT bridge',()=>harness.page.evaluate(async()=>{
  const tools=await api.openBrowserModelTools();try{globalThis.split=await tools.prepareSplit({source,ratio:'0.5'});return JSON.parse(await(await split.manifest.getFile()).text()).runtimeMetrics;}finally{await tools.close();}
 }));
 for(const name of ['cycle','forwarded','identity','reshape','transpose','control','axes','shape','missing','count','nonfinite'])await check('reject '+name,()=>harness.page.evaluate(async name=>{
  let modelPath=name+'.onnx',inputsManifest=source.inputsManifest;
  if(['axes','shape','missing','count','nonfinite'].includes(name)) {
   modelPath='model.onnx';const manifest=JSON.parse(await(await source.inputsManifest.getFile()).text());
   if(name==='axes')manifest.inputs[0].axes=['N','N'];
   if(name==='shape')manifest.inputs[0].shape=[1,2**24];
   if(name==='missing')manifest.inputs=[];
   if(name==='count'||name==='nonfinite'){manifest.inputs[0].csv='bad.csv';await write(sourceDirectory,'bad.csv',name==='count'?'1,2':'NaN');}
   inputsManifest=new File([JSON.stringify(manifest)],'inputs.json');
  }
  const tools=await api.openBrowserModelTools();let error;
  try{await tools.prepareSplit({source:{...source,modelPath,inputsManifest},ratio:'0.5',outputDirectory:output});}catch(value){error=value;}finally{await tools.close();}
  if(!error)throw new Error('Unsafe source accepted');await assertEmpty(output);return error.message;
 },name));
 await check('compile cancellation and retry',()=>harness.page.evaluate(async()=>{
  const directory=await opfs.getDirectoryHandle('cifar',{create:true});
  for(const [remote,local]of [['model/trained_poly.onnx','model.onnx'],['img.csv','img.csv']])await write(directory,local,await(await fetch('/cifar/'+remote)).arrayBuffer());
  const inputsManifest=new File([JSON.stringify({inputs:[{name:'input',csv:'img.csv',shape:[1,3,32,32],axes:['N','C','H','W']}]})],'inputs.json');
  globalThis.compileSource={directory,modelPath:'model.onnx',inputsManifest};
  let cancellation,terminals=0;
  const tools=await api.openBrowserModelTools({onProgress:p=>{if(p.phase==='compile-high-level')cancellation=tools.cancel(p.requestId);}});
  try {
   let error;try{await tools.compileHighLevel({source:{directory,modelPath:'model.onnx',inputsManifest},deploymentMode:'client_encrypted_input',outputDirectory:output});terminals++;}catch(value){terminals++;error=value;}
   if(error?.code!=='CANCELLED'||!(await cancellation)?.accepted||terminals!==1)throw new Error('Compile cancellation failed: '+error?.message);
   await assertEmpty(output);await tools.prepareSplit({source,ratio:'0.5',outputDirectory:output});
   return {terminals,committed:JSON.parse(await(await split.manifest.getFile()).text()).kind};
  }finally{await tools.close();}
 }));
 await check('new sample prefix and suffix output routes',()=>harness.page.evaluate(newSampleOutputRoutes));
 const modelWorker=await readFile(path.join(runtime,'model-worker.js'),'utf8');
 await harness.page.route('**/model-worker.js',route=>route.fulfill({contentType:'text/javascript',body:modelWorker+`
const createWritable=FileSystemFileHandle.prototype.createWritable;
FileSystemFileHandle.prototype.createWritable=function(...args){if(this.name==='artifact-manifest.json')throw new DOMException('injected model quota','QuotaExceededError');return createWritable.apply(this,args);};`}));
 await check('model quota cleanup and committed protection',()=>harness.page.evaluate(async()=>{
  const outputDirectory=await opfs.getDirectoryHandle('quota-output',{create:true}),tools=await api.openBrowserModelTools();let error;
  const before=await(await split.manifest.getFile()).text();
  try{await tools.compileHighLevel({source:compileSource,deploymentMode:'client_encrypted_input',outputDirectory});}catch(value){error=value;}finally{await tools.close();}
  if(!/injected model quota/.test(error?.message))throw new Error('Model quota fault not reported');
  await assertEmpty(outputDirectory);if(await(await split.manifest.getFile()).text()!==before)throw new Error('Committed split changed');return error.message;
 }));
 await harness.page.unroute('**/model-worker.js');
 await harness.page.route('**/model-worker.js',route=>route.fulfill({contentType:'text/javascript',body:modelWorker+`
FileSystemSyncAccessHandle.prototype.write=function(){throw new DOMException('injected workspace quota','QuotaExceededError');};`}));
 await check('Python workspace I/O cleanup',()=>harness.page.evaluate(async()=>{
  const tools=await api.openBrowserModelTools(),outputDirectory=await opfs.getDirectoryHandle('workspace-fault-output',{create:true});let error;
  try{await tools.prepareSplit({source,ratio:'0.5',outputDirectory});}catch(value){error=value;}finally{await tools.close();}
  if(!/injected workspace quota/.test(error?.message))throw new Error('Workspace I/O fault not reported: '+error?.message);
  await assertEmpty(outputDirectory);await assertEmpty(await opfs.getDirectoryHandle('latti-python-workspaces'));return error.message;
 }));
 await harness.page.unroute('**/model-worker.js');
 await harness.page.route('**/model-worker.js',route=>route.fulfill({contentType:'text/javascript',body:modelWorker+`
const syncHandle=FileSystemFileHandle.prototype.createSyncAccessHandle;
FileSystemFileHandle.prototype.createSyncAccessHandle=async function(...args){
 const handle=await syncHandle.apply(this,args);if(this.name==='file-127')progress('workspace-mount');return handle;
};`}));
 await check('cancel during Python workspace mount',()=>harness.page.evaluate(async()=>{
  let cancellation;const outputDirectory=await opfs.getDirectoryHandle('workspace-cancel-output',{create:true});
  const tools=await api.openBrowserModelTools({onProgress:p=>{if(p.phase==='workspace-mount')cancellation=tools.cancel(p.requestId);}});let error;
  try{await tools.prepareSplit({source,ratio:'0.5',outputDirectory});}catch(value){error=value;}finally{await tools.close();}
  if(error?.code!=='CANCELLED'||!(await cancellation)?.accepted)throw new Error('Workspace cancellation failed: '+error?.message);
  await assertEmpty(outputDirectory);await assertEmpty(await opfs.getDirectoryHandle('latti-python-workspaces'));return error.code;
 }));
 await harness.page.unroute('**/model-worker.js');
 await check('provision cancellation and retry',()=>harness.page.evaluate(async()=>{
  const directory=await opfs.getDirectoryHandle('parameter-task',{create:true}),manifest=await(await fetch('/fixtures/parameter-task/artifact-manifest.json')).json();
  for(const name of Object.keys(manifest.files))await write(directory,name,await(await fetch('/fixtures/parameter-task/'+name)).arrayBuffer());
  globalThis.parameterTask={directory,manifest:await write(directory,'artifact-manifest.json',JSON.stringify(manifest))};
  globalThis.parameterDirectory=await(await directory.getDirectoryHandle('task')).getDirectoryHandle('provisioner');
  globalThis.parameterKeys=await opfs.getDirectoryHandle('parameter-keys',{create:true});
  let cancellation,armed=false;
  globalThis.parameterClient=await api.openBrowserClient({clientDirectory:parameterDirectory,keyDirectory:parameterKeys,onProgress:p=>{if(armed&&p.phase==='provision'&&p.chunkBytes>0){armed=false;cancellation=parameterClient.cancel(p.requestId);}}});
  await parameterClient.setup();armed=true;
  const outputDirectory=await opfs.getDirectoryHandle('provision-cancel-output',{create:true});let error;
  try{await parameterClient.provisionRunner({task:parameterTask,parameterDirectory,outputDirectory});}catch(value){error=value;}
  if(error?.code!=='CANCELLED'||!(await cancellation)?.accepted)throw new Error('Provision cancellation failed: '+error?.message);
  await assertEmpty(outputDirectory);globalThis.keptBundle=await parameterClient.provisionRunner({task:parameterTask,parameterDirectory,outputDirectory});
  await parameterClient.close();return JSON.parse(await(await keptBundle.manifest.getFile()).text());
 }));
 const worker=await readFile(path.join(runtime,'worker.js'),'utf8');
 await harness.page.route('**/worker.js',route=>route.fulfill({contentType:'text/javascript',body:worker+`
self.addEventListener('message',event=>{if(event.data.type==='provisionRunner')globalThis.failProvisionWrite=true;});
const createWritable=FileSystemFileHandle.prototype.createWritable;
FileSystemFileHandle.prototype.createWritable=function(...args){if(globalThis.failProvisionWrite&&this.name.endsWith('.part'))throw new DOMException('injected provision quota','QuotaExceededError');return createWritable.apply(this,args);};`}));
 await check('provision I/O fault and committed protection',()=>harness.page.evaluate(async()=>{
  const client=await api.openBrowserClient({clientDirectory:parameterDirectory,keyDirectory:parameterKeys});await client.restore();
  const outputDirectory=await opfs.getDirectoryHandle('provision-fault-output',{create:true}),before=await(await keptBundle.manifest.getFile()).text();let error;
  try{await client.provisionRunner({task:parameterTask,parameterDirectory,outputDirectory});}catch(value){error=value;}finally{await client.close();}
  if(!/injected provision quota/.test(error?.message))throw new Error('Provision I/O fault not reported: '+error?.message);
  await assertEmpty(outputDirectory);if(await(await keptBundle.manifest.getFile()).text()!==before)throw new Error('Committed Runner changed');return error.message;
 }));
 await harness.page.unroute('**/worker.js');
 await check('provision retry after I/O fault',()=>harness.page.evaluate(async()=>{
  const client=await api.openBrowserClient({clientDirectory:parameterDirectory,keyDirectory:parameterKeys});try{await client.restore();return JSON.parse(await(await(await client.provisionRunner({task:parameterTask,parameterDirectory})).manifest.getFile()).text()).state;}finally{await client.close();}
 }));
 await check('parameter signature and layout fail before encryption',()=>harness.page.evaluate(async()=>{
  const get=async name=>{let directory=parameterTask.directory;const parts=name.split('/');for(const part of parts.slice(0,-1))directory=await directory.getDirectoryHandle(part);return directory.getFileHandle(parts.at(-1));};
  const signaturePath='task/runner/task_signature.json',signatureFile=await get(signaturePath),original=new Uint8Array(await(await signatureFile.getFile()).arrayBuffer());
  const manifest=JSON.parse(await(await parameterTask.manifest.getFile()).text());
  const digest=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
  const errors=[];
  for(const fault of ['duplicate','layout','missing']) {
   const signature=JSON.parse(new TextDecoder().decode(original));
   if(fault==='duplicate')signature.encrypted_parameters.push({...signature.encrypted_parameters[0]});
   if(fault==='layout')signature.encrypted_parameters[0].layout={kind:'unsupported'};
   if(fault==='missing')signature.encrypted_parameters=[];
   const bytes=new TextEncoder().encode(JSON.stringify(signature));await write(parameterTask.directory,signaturePath,bytes);
   manifest.files[signaturePath]={bytes:bytes.byteLength,sha256:await digest(bytes)};await write(parameterTask.directory,'artifact-manifest.json',JSON.stringify(manifest));
   const progress=[],client=await api.openBrowserClient({clientDirectory:parameterDirectory,keyDirectory:parameterKeys,onProgress:p=>progress.push(p)});let error;
   try{await client.restore();await client.provisionRunner({task:parameterTask,parameterDirectory});}catch(value){error=value;}finally{await client.close();}
   if(!error||progress.some(p=>p.phase==='provision'&&p.chunkBytes>0))throw new Error('Invalid parameters reached encryption: '+fault);
   if(!/Duplicate parameter argument|layout differs|no encrypted parameters/.test(error.message))throw error;
   errors.push({fault,error:error.message});
  }
  await write(parameterTask.directory,signaturePath,original);manifest.files[signaturePath]={bytes:original.byteLength,sha256:await digest(original)};
  await write(parameterTask.directory,'artifact-manifest.json',JSON.stringify(manifest));return errors;
 }));
}finally{
 await harness.page.evaluate(async()=>{await globalThis.parameterClient?.close();}).catch(()=>{});
 const workspaces=await harness.page.evaluate(async()=>Array.fromAsync((await opfs.getDirectoryHandle('latti-python-workspaces')).keys()));
 const terminals=await harness.page.evaluate(()=>({requests:Object.keys(lattiTerminalCounts).length,duplicates:Object.entries(lattiTerminalCounts).filter(([,count])=>count>1)}));
 await writeFile(path.join(root,'terminals.json'),JSON.stringify(terminals,null,2));
 await harness.close();await writeFile(path.join(root,'memory.json'),JSON.stringify(harness.rss,null,2));
 await writeFile(path.join(root,'workspace-cleanup.json'),JSON.stringify({remaining:workspaces},null,2));
 if(workspaces.length)throw new Error('Python workspaces retained: '+workspaces.join(', '));
 if(terminals.duplicates.length)throw new Error('Duplicate Worker terminal');
}
