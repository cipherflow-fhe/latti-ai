import {mkdir,writeFile,readFile} from 'node:fs/promises';
import path from 'node:path';
import {browserHarness} from './browser-harness.mjs';

const delivery=process.env.LATTI_RETINA_DELIVERY_DIRECTORY;
if(!delivery)throw new Error('Set LATTI_RETINA_DELIVERY_DIRECTORY to the extracted retina-split-delivery folder');
const root=path.resolve(process.env.LATTI_RETINA_DELIVERY_EVIDENCE||'/tmp/latti-retina-delivery-check');
await mkdir(root,{recursive:true});
const runtime=path.resolve(process.env.LATTI_RUNTIME_DIRECTORY||new URL('../dist/',import.meta.url).pathname);
const assets=path.resolve(delivery);
const harness=await browserHarness(root,{'/check/runtime/':runtime,'/check/':assets});
try {
  const origin=new URL(harness.page.url()).origin;
  await harness.page.goto(origin+'/check/index.html');
  await harness.beginPhase('fixture-import');
  await harness.page.evaluate(()=>{
    const input=document.createElement('input');input.type='file';input.webkitdirectory=true;input.id='fixture-upload';document.body.append(input);
  });
  await harness.page.locator('#fixture-upload').setInputFiles(assets);
  await harness.page.evaluate(async()=>{
    const directory=await(await navigator.storage.getDirectory()).getDirectoryHandle('delivery-fixture',{create:true});
    const files=Array.from(document.getElementById('fixture-upload').files);
    for(const file of files) {
      const name=file.webkitRelativePath.split('/').slice(1).join('/');
      if(name==='WASM-0.2.0.zip'||name.startsWith('runtime/'))continue;
      let target=directory;const parts=name.split('/');
      for(const part of parts.slice(0,-1))target=await target.getDirectoryHandle(part,{create:true});
      const writer=await(await target.getFileHandle(parts.at(-1),{create:true})).createWritable();
      await file.stream().pipeTo(writer);
    }
    globalThis.showDirectoryPicker=async()=>directory;
    const log=document.getElementById('log');new MutationObserver(()=>console.log('[retina-delivery]',log.textContent.split('\n').filter(Boolean).at(-1))).observe(log,{childList:true,characterData:true,subtree:true});
  });
  await harness.page.click('#select');
  await harness.beginPhase('retina-split-parameter-replay');
  await harness.page.click('#run');
  await harness.page.waitForFunction(()=>document.getElementById('status').textContent.startsWith('校验'),{},{timeout:1800000});
  const result=await harness.page.evaluate(async()=>{
    if(!globalThis.deliveryCheckResult?.report)throw new Error(document.getElementById('status').textContent);
    const {report,bundle,prefix,prefixRun,inputBatch}=deliveryCheckResult;
    const prefixMetadata=JSON.parse(await(await prefix.manifest.getFile()).text());
    const files=Object.keys(prefixMetadata.files);
    if(prefixMetadata.kind!=='prefix'||files.some(name=>!/^(model\.onnx|compact\.onnx\.data|prefix\.json)$/.test(name)))throw new Error('Independent prefix includes private files');
    const runMetadata=JSON.parse(await(await prefixRun.manifest.getFile()).text());
    let directory=prefixRun.directory;const parts=runMetadata.entry.split('/');
    for(const part of parts.slice(0,-1))directory=await directory.getDirectoryHandle(part);
    const binding=JSON.parse(await(await(await directory.getFileHandle(parts.at(-1))).getFile()).text());
    if(binding.count!==10||binding.weightLoading!=='on_demand'||!(binding.weightReads.reads>0))throw new Error('Batch prefix coverage/weight reads differ');
    if(Object.keys(runMetadata.files).some(name=>!name.startsWith('run/')))throw new Error('Prefix results include private inputs/models');
    const parameters=JSON.parse(await(await(await(await bundle.directory.getDirectoryHandle('encrypted_model_parameters')).getFileHandle('manifest.json')).getFile()).text());
    const argumentsList=Object.values(parameters.arguments),ciphertexts=argumentsList.reduce((sum,arg)=>sum+arg.count,0);
    if(!ciphertexts||argumentsList.some(arg=>arg.verified_count!==arg.count||!(arg.encoding_max_abs_error<1e-5)))throw new Error('Encrypted parameter slot verification failed');
    const batch=JSON.parse(await(await inputBatch.manifest.getFile()).text());
    if(batch.keyIdentity!==report.keyIdentity||report.count!==10||report.samples.length!==10)throw new Error('Batch identity/coverage differs');
    return {...report,ciphertexts,allParameterSlotsVerified:true,dataProviderPrefix:{files,weightReads:binding.weightReads,batchCount:binding.count}};
  });
  const expected=JSON.parse(await readFile(path.join(delivery,'delivery-manifest.json'),'utf8'));
  if(result.status!=='passed'||result.keyIdentity!==expected.keyIdentity||result.maxAbsError>=0.1)throw new Error('Replay report rejected');
  await writeFile(path.join(root,'report.json'),JSON.stringify(result,null,2)+'\n');
  console.log(JSON.stringify(result,null,2));
}finally{
  await harness.close();await writeFile(path.join(root,'memory.json'),JSON.stringify({...harness.rss,phases:harness.phaseMetrics},null,2));
}
