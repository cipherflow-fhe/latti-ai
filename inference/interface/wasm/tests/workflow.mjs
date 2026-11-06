import assert from 'node:assert/strict';
import {copyFile,mkdir,mkdtemp,readFile,writeFile} from 'node:fs/promises';
import {spawn} from 'node:child_process';
import path from 'node:path';
import os from 'node:os';
import {browserHarness} from './browser-harness.mjs';

const repo=path.resolve(import.meta.dirname,'../../../..');
const artifacts=process.env.LATTI_BROWSER_ARTIFACTS||await mkdtemp(path.join(os.tmpdir(),'taa-workflow-'));
await mkdir(artifacts,{recursive:true});
assert(process.env.LATTI_WORKFLOW_TASK&&process.env.LATTI_WORKFLOW_KEYS,'Set fresh LATTI_WORKFLOW_TASK and Worker-generated LATTI_WORKFLOW_KEYS');
const task=path.resolve(process.env.LATTI_WORKFLOW_TASK),keys=path.resolve(process.env.LATTI_WORKFLOW_KEYS);
const model=process.env.LATTI_WORKFLOW_MODEL||'cifar10',data=path.resolve(process.env.LATTI_WORKFLOW_DATA||path.join(artifacts,'data'));
async function native(mode) {
  // Optional client-only measurement rerun against this acceptance run's
  // unchanged native outputs. The report always identifies their origin.
  if(mode==='evaluate'&&process.env.LATTI_WORKFLOW_NATIVE_ARTIFACTS) {
    for(const name of ['output.zip','native.json','reload.json'])await copyFile(path.join(process.env.LATTI_WORKFLOW_NATIVE_ARTIFACTS,name),path.join(artifacts,name));
    return;
  }
  const child=spawn(process.env.PYTHON||'python3',[path.join(import.meta.dirname,'workflow-native.py'),mode,artifacts,repo],{env:process.env,stdio:'inherit',cwd:artifacts});
  assert.equal(await new Promise((resolve,reject)=>{child.once('error',reject);child.once('close',resolve);}),0,'native '+mode+' failed');
}
await native('prepare');
const harness=await browserHarness(artifacts,{'/task/':path.join(task,'client'),'/keys/':keys,'/data/':data});
await harness.page.exposeFunction('lattiTestPhase',name=>harness.beginPhase(name));
try {
  const encrypted=await harness.page.evaluate(async model=>{
    const {openBrowserClient}=await import('/dist/index.js');
    const root=await navigator.storage.getDirectory();
    const importFile=async(directory,name,url)=>{
      const response=await fetch(url);if(!response.ok)throw new Error('Test input unavailable: '+url);
      const file=await directory.getFileHandle(name,{create:true});await response.body.pipeTo(await file.createWritable());return file;
    };
    const task=await root.getDirectoryHandle('task',{create:true});
    for(const name of ['task_config.json','ckks_parameter.json'])await importFile(task,name,'/task/'+name);
    const manifest=await(await fetch('/keys/key-manifest.json')).json();
    const keys=await root.getDirectoryHandle('keys',{create:true}),profile=await keys.getDirectoryHandle(manifest.profile.profileId,{create:true});
    const secret=await profile.getDirectoryHandle('secret',{create:true});
    await importFile(secret,'full_context.bin','/keys/full_context.bin');await importFile(profile,'manifest.json','/keys/key-manifest.json');
    const cases=await(await fetch('/artifact/cases.json')).json(),inputDirectories={};
    for(const name of Object.keys(cases[0].inputs))inputDirectories[name]=await root.getDirectoryHandle('csv-'+name,{create:true});
    if(model==='cifar10') {
      const {preprocessDirectory}=await import('/dist/tasks/cifar10/preprocess.js'),images=await root.getDirectoryHandle('images',{create:true});
      for(const sample of cases)await importFile(images,sample.case_id+'.png','/data/'+sample.image);
      await preprocessDirectory({inputDirectory:images,outputDirectory:inputDirectories.input});
      for(const sample of cases) {
        const actual=(await(await(await inputDirectories.input.getFileHandle(sample.case_id+'.csv')).getFile()).text()).trim().split(',').map(Number);
        const expected=(await(await fetch('/data/'+sample.inputs.input)).text()).trim().split(',').map(Number);
        if(actual.length!==expected.length||actual.some((value,index)=>Math.abs(value-expected[index])>1e-6))throw new Error('Current training transform differs');
      }
    } else {
      const prepared=await root.getDirectoryHandle('prepared',{create:true});
      await importFile(prepared,'cases.json','/artifact/cases.json');
      for(const name of Object.keys(inputDirectories))inputDirectories[name]=await prepared.getDirectoryHandle(name,{create:true});
      for(const sample of cases)for(const [name,relative]of Object.entries(sample.inputs))await importFile(inputDirectories[name],sample.case_id+'.csv','/data/'+relative);
      const {preprocessDirectory}=await import('/dist/tasks/retina/preprocess.js');
      const checked=await preprocessDirectory({inputDirectory:prepared,task:JSON.parse(await(await(await task.getFileHandle('task_config.json')).getFile()).text())});
      if(checked.count!==String(cases.length))throw new Error('Retina preprocessing count differs');
    }
    const metrics={wasmBytes:0,heapBytes:0,chunkBytes:0,inFlightChunks:0};
    const client=await openBrowserClient({keyDirectory:keys,onProgress:p=>{for(const name of Object.keys(metrics))metrics[name]=Math.max(metrics[name],p[name]||0);}});
    try {
      await globalThis.lattiTestPhase('restore-and-bind');
      await client.restore();await client.bindTask(task);
      const directoryRuns=[];
      for(const [inputName,inputDirectory]of Object.entries(inputDirectories)) {
        await globalThis.lattiTestPhase('encrypt-directory-'+inputName);
        for(const name of Object.keys(metrics))metrics[name]=0;
        const diskBefore=await navigator.storage.estimate(),start=performance.now();
        const batch=await client.encryptBatch({inputName,inputDirectory});
        directoryRuns.push({inputName,count:batch.count,metrics:{...metrics},elapsedMs:performance.now()-start,diskBefore,diskAfter:await navigator.storage.estimate()});
        const location=await root.resolve(batch.directory);
        if(location.length!==3||location[0]!=='latti-jobs'||location[2]!=='output')throw new Error('Unexpected test batch location');
        await(await root.getDirectoryHandle('latti-jobs')).removeEntry(location[1],{recursive:true});
      }
      for(const name of Object.keys(metrics))metrics[name]=0;
      await globalThis.lattiTestPhase('encrypt-server-zip');
      const diskBefore=await navigator.storage.estimate(),start=performance.now();
      const batch=await client.encryptBatch({mode:'server',inputDirectories,forceZip64:true});
      if(!(await fetch('/artifact/input.zip',{method:'POST',body:await batch.archive.file.getFile()})).ok)throw new Error('Input upload failed');
      const pointer=await root.getFileHandle('input-batch.json',{create:true}),writer=await pointer.createWritable();
      await writer.write(JSON.stringify({directory:batch.directory.name,job:batch.manifest.name,count:batch.count,profileId:manifest.profile.profileId}));await writer.close();
      globalThis.inputBatch=batch;
      return {metrics,directoryRuns,count:batch.count,archiveBytes:batch.archive.bytes,elapsedMs:performance.now()-start,diskBefore,diskAfter:await navigator.storage.estimate()};
    } finally {await client.close();}
  },model);
  await harness.beginPhase('native-server');
  await native('evaluate');
  await harness.beginPhase('decrypt-server-zip');
  const decrypted=await harness.page.evaluate(async model=>{
    const {openBrowserClient}=await import('/dist/index.js');
    const {postprocess}=await import('/dist/tasks/'+model+'/postprocess.js');
    const root=await navigator.storage.getDirectory(),keys=await root.getDirectoryHandle('keys'),task=await root.getDirectoryHandle('task');
    const client=await openBrowserClient({keyDirectory:keys});
    try {
      await client.restore();await client.bindTask(task);
      const incoming=await root.getFileHandle('output.zip',{create:true}),response=await fetch('/artifact/output.zip');
      await response.body.pipeTo(await incoming.createWritable());
      const result=await client.decryptBatch({mode:'server',outputName:JSON.parse(await(await(await task.getFileHandle('task_config.json')).getFile()).text()).task_output_id[0],inputFile:incoming,inputBatch:globalThis.inputBatch});
      const cases=await(await fetch('/artifact/cases.json')).json(),labels=new File([cases.map(sample=>JSON.stringify({case_id:sample.case_id,label:sample.label})+'\n').reverse().join('')],'labels.jsonl');
      const report=await root.getFileHandle('predictions.jsonl',{create:true});
      const statistics=await postprocess({results:result,labels,report});
      if(!(await fetch('/artifact/predictions.jsonl',{method:'POST',body:await report.getFile()})).ok)throw new Error('Report export failed');
      return {...statistics,diskAfter:await navigator.storage.estimate()};
    } finally {await client.close();}
  },model);
  await native('check');
  const checked=JSON.parse(await readFile(path.join(artifacts,'checked.json')));
  assert.equal(decrypted.total,checked.total);assert.equal(decrypted.correct,checked.correct);
  const report={artifacts,model,task,nativeArtifacts:process.env.LATTI_WORKFLOW_NATIVE_ARTIFACTS||artifacts,encrypted,decrypted,checked,rss:harness.rss,phaseMetrics:harness.phaseMetrics};
  await writeFile(path.join(artifacts,'workflow.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
} finally {await harness.close();}
