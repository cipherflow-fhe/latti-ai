import {createWriteStream} from 'node:fs';
import {mkdir,readFile,writeFile,rename,link,stat,readdir} from 'node:fs/promises';
import {spawn,execFileSync} from 'node:child_process';
import path from 'node:path';
import {browserHarness} from './browser-harness.mjs';

const root=path.resolve(process.env.LATTI_BATCH_ROOT||'/tmp/latti-retina-raw-batch-20261127');
const repo=path.resolve(import.meta.dirname,'../../../..');
const raw=process.env.LATTI_BATCH_RAW||'/LattiAI-Task-Resources/retina_degree8_02/task/client/demo_10/dataset';
const model=path.join(repo,'runs/retina/model/epoch041.onnx');
const python=process.env.PYTHON||'python3';
await mkdir(root,{recursive:true});
const summary={status:'running',ratio:'0.9',inputMode:'client_ciphertext',parameterMode:'encrypted_offline',backend:'native GPU',loadingMode:'runtime_lazy',raw,model,stages:[],startedAt:new Date().toISOString()};
const save=async(name,value)=>writeFile(path.join(root,name),JSON.stringify(value,null,2)+'\n');
const run=async(command,args,log)=>{
  const stream=createWriteStream(path.join(root,log));
  const child=spawn(command,args,{cwd:root,env:{...process.env,OMP_NUM_THREADS:'8',OPENBLAS_NUM_THREADS:'8'}});
  child.stdout.pipe(stream,{end:false});child.stderr.pipe(stream,{end:false});
  try{await new Promise((resolve,reject)=>{child.once('error',reject);child.once('close',code=>code===0?resolve():reject(new Error(`${command} failed (${code}); see ${log}`)));});}
  finally{await new Promise(resolve=>stream.end(resolve));}
};
const native=async(operation,args=[])=>run(python,[path.join(import.meta.dirname,'raw-batch-native.py'),operation,'--root',root,...args],operation+'.log');
const store=async(data,destination)=>{
  const directory=path.join(root,destination);await mkdir(directory,{recursive:true});
  for(const [name,transfer]of data.transfers){const file=path.resolve(directory,name);if(!file.startsWith(directory+path.sep))throw new Error('Unsafe transfer');await mkdir(path.dirname(file),{recursive:true});await rename(path.join(root,transfer),file);}
};
const dataFiles=[];
for(const name of ['images','disk','lesion'])for(const file of await readdir(path.join(raw,name)))dataFiles.push(name+'/'+file);
dataFiles.push('risk_factor_5.xlsx','label.csv');
const harness=await browserHarness(root,{'/raw/':raw,'/model/':path.dirname(model),'/config/':path.join(repo,'training/config'),'/workflow/':path.join(repo,'inference/interface/wasm/delivery')});
const browserLog=createWriteStream(path.join(root,'browser.jsonl'));
harness.page.on('console',message=>browserLog.write(JSON.stringify({at:Date.now(),stage:summary.stages.at(-1)?.name,text:message.text()})+'\n'));
let gpuSampler,gpuPeak=0;
try {
  await harness.page.exposeFunction('batchStage',async name=>{
    const previous=summary.stages.at(-1);if(previous)previous.durationMs=Date.now()-previous.startedMs;
    summary.stages.push({name,startedMs:Date.now()});console.log('[batch]',name);
    await harness.beginPhase(name);await save('summary.json',summary);
  });
  await harness.page.exposeFunction('batchArtifact',async(name,data)=>{
    if(name==='keys'){await store(data,'keys');summary.keys=data.keys;await save('keys.json',data.keys);}
    else if(name==='runner'){
      await store(data,'bundle');
      const profile=path.join(root,'keys',summary.keys.profileId,'public');await mkdir(profile,{recursive:true});
      await link(path.join(root,'bundle/eval_context.bin'),path.join(profile,'eval_context.bin'));
    }else if(name==='input-batch'){
      await rename(path.join(root,data.transfer),path.join(root,'input.zip'));await save('input-batch.json',data.metadata);
    }else await store(data,name==='prefix-run'?'prefix-run':name);
  });
  await harness.page.exposeFunction('batchNative',async metadata=>{
    await save('crossings.json',metadata.crossings);
    await native('inputs',['--raw',raw]);
    await native('prepare-runner');
    gpuPeak=0;
    gpuSampler=setInterval(()=>{try{const usage=execFileSync('nvidia-smi',['--query-compute-apps=used_gpu_memory','--format=csv,noheader,nounits'],{encoding:'utf8',timeout:2000});gpuPeak=Math.max(gpuPeak,...usage.trim().split('\n').map(Number).filter(Number.isFinite));}catch{}},250);
    try{await run('/usr/bin/time',['-v','-o',path.join(root,'native-memory.txt'),path.join(repo,'build/tools/evaluate_browser_task'),
      '--mode','server_provisioned_runner','--task-dir',path.join(root,'bundle'),'--batch-manifest',path.join(root,'native-inputs/batch.json'),
      '--output-dir',path.join(root,'ciphertexts'),'--gpu'],'native.log');}
    finally{clearInterval(gpuSampler);gpuSampler=undefined;}
    await save('native-gpu.json',{peakMiB:gpuPeak,samplingIntervalMs:250});
    await native('zip-results');return 'output.zip';
  });
  const report=await harness.page.evaluate(async({files})=>{
    const api=await import('/dist/index.js');
    const {runParameterBatchWorkflow,fileAt,readJSON}=await import('/workflow/batch-workflow.mjs');
    const opfs=await navigator.storage.getDirectory();
    const raw=await opfs.getDirectoryHandle('raw',{create:true}),model=await opfs.getDirectoryHandle('model',{create:true}),keyDirectory=await opfs.getDirectoryHandle('model-keys',{create:true});
    const download=async(directory,name,url)=>{const response=await fetch(url);if(!response.ok)throw new Error('Missing fixture: '+url);const file=await fileAt(directory,name,true);await response.body.pipeTo(await file.createWritable());return file;};
    for(const name of files)await download(raw,name,'/raw/'+name);
    await download(model,'epoch041.onnx','/model/epoch041.onnx');
    const transfer=async handle=>{
      const name=crypto.randomUUID()+'.bin',file=await handle.getFile(),chunk=64*1024*1024;
      for(let at=0;at<file.size;at+=chunk){const response=await fetch('/artifact/'+name,{method:'POST',headers:{'X-Latti-Append':at?'1':'0'},body:file.slice(at,at+chunk)});if(!response.ok)throw new Error('Transfer failed');}
      if(!file.size)await fetch('/artifact/'+name,{method:'POST',body:file});return name;
    };
    const exportTree=async directory=>{
      const transfers=[];
      const walk=async(dir,base='')=>{for await(const entry of dir.values()){if(entry.kind==='directory')await walk(entry,base+entry.name+'/');else transfers.push([base+entry.name,await transfer(entry)]);}};
      await walk(directory);return {transfers};
    };
    const processing=await opfs.getDirectoryHandle('static-processing',{create:true});
    const upload=await api.uploadProcessingScripts({directory:processing,
      preprocessFile:new File([await(await fetch('/dist/tasks/retina/raw-preprocess.js')).arrayBuffer()],'preprocess.js'),
      postprocessFile:new File([await(await fetch('/dist/tasks/retina/raw-postprocess.js')).arrayBuffer()],'postprocess.js')});
    for(const name of ['preprocess.js','postprocess.js']){
      const response=await fetch('/artifact/uploaded-'+name,{method:'POST',body:await(await processing.getFileHandle(name)).getFile()});if(!response.ok)throw new Error('Static module publication failed');
    }
    const config=new File([await(await fetch('/config/storage_aware_config.json')).arrayBuffer()],'compile-config.json');
    const result=await runParameterBatchWorkflow({modelDirectory:model,modelPath:'epoch041.onnx',rawDirectory:raw,labelFile:await raw.getFileHandle('label.csv'),
      preprocessingURL:new URL('/artifact/uploaded-preprocess.js',location.href).href,postprocessingURL:new URL('/artifact/uploaded-postprocess.js',location.href).href,
      runtimeURL:new URL('/dist/index.js',location.href).href,compileConfig:config,keyDirectory,generateKeys:true,
      onStage:batchStage,onProgress:()=>{},
      onArtifact:async(name,artifact)=>{
        if(name==='keys'){
          const profile=await keyDirectory.getDirectoryHandle(artifact.keys.profileId);
          await batchArtifact(name,{transfers:(await exportTree(await profile.getDirectoryHandle('secret'))).transfers.map(([n,t])=>[artifact.keys.profileId+'/secret/'+n,t])
            .concat([[artifact.keys.profileId+'/manifest.json',await transfer(await profile.getFileHandle('manifest.json'))],['public_context.bin',await transfer(artifact.publicContext.file)]]),keys:{profileId:artifact.keys.profileId,keyIdentity:artifact.keys.keyIdentity}});
        }else if(name==='runner'){
          // Owner keeps the live SK/PK; the exported runner contains the exact
          // same evaluation context. Remove this duplicate before host export.
          const runnerManifest=await readJSON(artifact.manifest);
          const profile=await keyDirectory.getDirectoryHandle(runnerManifest.profileId);
          await(await profile.getDirectoryHandle('public')).removeEntry('eval_context.bin');
          await batchArtifact(name,await exportTree(artifact.directory));
          // Native transport now owns the committed runner bundle.
          for await(const entry of artifact.directory.values())await artifact.directory.removeEntry(entry.name,{recursive:entry.kind==='directory'});
        }else if(name==='input-batch')await batchArtifact(name,{transfer:await transfer(artifact.archive.file),metadata:await readJSON(artifact.manifest)});
        else await batchArtifact(name,await exportTree(artifact.directory));
      },
      runNative:async({crossings,keys})=>{
        await batchNative({crossings,keyIdentity:keys.keyIdentity});
        return download(opfs,'native-output.zip','/artifact/output.zip');
      }});
    globalThis.batchWorkflowResult=result;
    const {predictionsFile,...statistics}=result.statistics;
    await fetch('/artifact/predictions.csv',{method:'POST',body:await predictionsFile.getFile()});
    return {count:result.count,stages:result.stages,statistics,keys:{profileId:result.keys.profileId,keyIdentity:result.keys.keyIdentity},crossings:result.crossings,
      processing:await readJSON(upload.manifest)};
  },{files:dataFiles});
  await save('workflow.json',report);
  await native('compare',['--model',model,'--raw',raw,'--workflow',path.join(root,'workflow.json')]);
  summary.status='passed';summary.workflow=report;summary.comparison=JSON.parse(await readFile(path.join(root,'comparison.json'),'utf8'));
}catch(error){summary.status='failed';summary.error=error.message;process.exitCode=1;console.error(error);}
finally{
  clearInterval(gpuSampler);const last=summary.stages.at(-1);if(last)last.durationMs=Date.now()-last.startedMs;
  await harness.close();browserLog.end();summary.finishedAt=new Date().toISOString();
  await save('memory.json',{rss:harness.rss,phases:harness.phaseMetrics});await save('summary.json',summary);console.log(JSON.stringify({status:summary.status,error:summary.error,root,maxAbsError:summary.comparison?.max_abs_error}));
}
