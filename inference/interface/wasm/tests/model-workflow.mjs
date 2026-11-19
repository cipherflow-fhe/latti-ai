import {createHash} from 'node:crypto';
import {createReadStream,createWriteStream,existsSync} from 'node:fs';
import {mkdir,readFile,writeFile,rename,stat,link,unlink} from 'node:fs/promises';
import {spawn,execFileSync} from 'node:child_process';
import os from 'node:os';
import path from 'node:path';
import {browserHarness} from './browser-harness.mjs';
import {sampleAndBindingRegressions,outputFramingRegressions} from './model-regressions.mjs';

const packageRoot=path.resolve(import.meta.dirname,'..'),repo=path.resolve(packageRoot,'../../..');
const root=path.resolve(process.env.LATTI_MODEL_ACCEPTANCE_ROOT||'/tmp/latti-browser-model-acceptance/final');
const options={models:'retina',modes:'parameter'};
for(let i=2;i<process.argv.length;i+=2) {
  const key=process.argv[i].replace(/^--/,'');
  if(!Object.hasOwn(options,key)||!process.argv[i+1])throw new Error('Usage: --models cifar10,retina,bert,yolov5,vit,imagenet --modes input,parameter');
  options[key]=process.argv[i+1];
}
const models=options.models.split(','),modes=options.modes.split(',');
if(models.some(name=>!['cifar10','retina','bert','yolov5','vit','imagenet'].includes(name))||modes.some(name=>!['input','parameter'].includes(name)))throw new Error('Unknown model/mode diagnostic selection');
const timeout=Number(process.env.LATTI_MODEL_TIMEOUT_MS||1800000);
if(!Number.isSafeInteger(timeout)||timeout<=0)throw new Error('Invalid finite model timeout');
const python=process.env.LATTI_MODEL_NATIVE_PYTHON||'python3';
const runner=process.env.LATTI_MODEL_NATIVE_RUNNER||path.join(repo,'build/tools/evaluate_browser_task');
const backend=process.env.LATTI_MODEL_BACKEND||'gpu',device=process.env.LATTI_GPU_DEVICE||'0';
if(!['cpu','gpu'].includes(backend))throw new Error('Invalid backend');
const fullMode=mode=>mode==='input'?'client_encrypted_input':'server_provisioned_runner';
const fixtures={
  bert:{directory:'BERT',modelKind:'bert',config:'BERT_pcm_config.json',model:'model/trained_poly.onnx',sha256:'850684974c16b0a6afe513ed347d7cbd4ab790dbc7672f2d704ec730a211073b',
    inputs:[{name:'inputs_embeds',csv:'input.csv',shape:[1,44,768],axes:['N','T','D']}]},
  vit:{directory:'ViT',modelKind:'vit',config:'ViT_pcm_config.json',model:'model/trained_poly.onnx',sha256:'96cdfaa5c03c55171ff280f37be438cbe2621f5ba0c3f571f1f9d4b78848529c',
    inputs:[{name:'input',csv:'patch_input.csv',shape:[1,197,768],axes:['N','T','D']}]},
  imagenet:{model:'model/trained_poly.onnx',sha256:'fa59a8d129c11ecd58d43f651228a02f90e49139944131de34ac36d31f051cc7',
    inputs:[{name:'input',csv:'img.csv',shape:[1,3,256,256],axes:['N','C','H','W']}]},
  yolov5:{model:'model/best_conv_outputs.onnx',sha256:'7f924c75ed124391b6847f8d037ec6eb7c60da0032f39292fa7dd558c43b9a3d',
    inputs:[{name:'images',csv:'img.csv',shape:[1,3,512,512],axes:['N','C','H','W']}]},
  cifar10:{model:'model/trained_poly.onnx',sha256:'4e4857a7af897707aa977ee824e1d0d425098cfd912a081f3c922ee546d59218',
    inputs:[{name:'input',csv:'img.csv',shape:[1,3,32,32],axes:['N','C','H','W']}]},
  retina:{model:'model/epoch041.onnx',sha256:'cf91f21f43d903ad5a4102eb899201290b35082f583aae22af48ec6cab94a898',
    inputs:[{name:'input_0',csv:'img.csv',shape:[1,5,512,512],axes:['N','C','H','W']},
      {name:'input_1',csv:'non_inv.csv',shape:[1,2],axes:['N','F']},{name:'input_2',csv:'inv.csv',shape:[1,3],axes:['N','F']}]},
};
await mkdir(root,{recursive:true});
const summary={status:'running',units:[],unfinished:[],stages:[]};let harness,current='environment';
const evalContexts=new Map();
const hash=async file=>{const digest=createHash('sha256');for await(const chunk of createReadStream(file))digest.update(chunk);return digest.digest('hex');};
const json=async(file,value)=>{await mkdir(path.dirname(file),{recursive:true});await writeFile(file,JSON.stringify(value,null,2)+'\n');};
async function run(command,args,log) {
  await mkdir(path.dirname(log),{recursive:true});
  const output=createWriteStream(log,{mode:0o600}),child=spawn(command,args,{cwd:path.dirname(log),env:{...process.env,OMP_NUM_THREADS:'8',OPENBLAS_NUM_THREADS:'8'}});
  child.stdout.pipe(output,{end:false});child.stderr.pipe(output,{end:false});
  let force;
  const timer=setTimeout(()=>{child.kill('SIGTERM');force=setTimeout(()=>child.kill('SIGKILL'),10000);},timeout);
  try {await new Promise((resolve,reject)=>{child.once('error',reject);child.once('exit',(code,signal)=>code===0?resolve():reject(new Error(`${command} failed (${code??signal}); see ${log}`)));});}
  finally{clearTimeout(timer);clearTimeout(force);await new Promise(resolve=>output.end(resolve));}
}
async function stage(name,action) {
  current=name;console.log('[model-workflow]',name);await harness?.beginPhase(name);
  const start=Date.now(),record={stage:name,startedAt:new Date(start).toISOString()};summary.stages.push(record);
  let timer;
  try {
    const result=await Promise.race([action(),new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('Stage timed out: '+name)),timeout);})]);
    record.status='complete';return result;
  }
  catch(error){record.status='failed';record.error=error.message;throw error;}
  finally{
    record.durationMs=Date.now()-start;
    clearTimeout(timer);
    if(harness)record.storage=await harness.page.evaluate(()=>navigator.storage.estimate()).catch(()=>null);
    await json(path.join(root,'summary.json'),summary);
  }
}
async function saveArtifact(result,destination) {
  await mkdir(destination,{recursive:true});
  for(const [name,transfer]of result.transfers) {
    const file=path.resolve(destination,name);
    if(!file.startsWith(destination+path.sep))throw new Error('Unsafe artifact transfer path');
    await mkdir(path.dirname(file),{recursive:true});await rename(path.join(root,transfer),file);
  }
  await writeFile(path.join(destination,result.manifestName),result.manifest);
  return JSON.parse(result.manifest);
}
const nativeCheck=(operation,args,log)=>run(python,[path.join(import.meta.dirname,'model-workflow-native.py'),operation,...args],log);
try {
  if(!existsSync(runner))throw new Error('Build evaluate_browser_task or set LATTI_MODEL_NATIVE_RUNNER');
  if(!process.env.CHROME_PATH&&existsSync('/opt/google/chrome/chrome'))process.env.CHROME_PATH='/opt/google/chrome/chrome';
  const sources={};
  for(const model of models) {
    const fixture=fixtures[model],files={};
    for(const name of [fixture.model,...fixture.inputs.map(input=>input.csv)]) {
      const file=path.join(repo,'runs',fixture.directory||model,name);files[name]={bytes:(await stat(file)).size,sha256:await hash(file)};
    }
    if(files[fixture.model].sha256!==fixture.sha256)throw new Error('Fixed model SHA256 differs: '+model);
    sources[model]={...fixture,files};
  }
  await json(path.join(root,'source-manifest.json'),{models:sources,compileConfig:{file:'training/config/storage_aware_config.json',sha256:await hash(path.join(repo,'training/config/storage_aware_config.json'))}});
  const build=JSON.parse(await readFile(path.join(packageRoot,'dist/manifest.json'),'utf8'));
  await json(path.join(root,'environment.json'),{createdAt:new Date().toISOString(),os:{platform:os.platform(),release:os.release(),architecture:os.arch()},
    node:process.version,build,backend,device,nativeRunner:runner,nativeRunnerSHA256:await hash(runner),
    nativePython:execFileSync(python,['-c','import sys,onnx,onnxruntime,numpy,scipy,h5py; print(dict(python=sys.version,onnx=onnx.__version__,ort=onnxruntime.__version__,numpy=numpy.__version__,scipy=scipy.__version__,h5py=h5py.__version__))'],{encoding:'utf8'}).trim(),
    chromium:process.env.CHROME_PATH?execFileSync(process.env.CHROME_PATH,['--version'],{encoding:'utf8'}).trim():'Playwright Chromium',
    devices:backend==='gpu'?execFileSync('nvidia-smi',['--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],{encoding:'utf8'}).trim():os.cpus()[0]?.model});
  harness=await browserHarness(root,{'/models/':path.join(repo,'runs'),'/config/':path.join(repo,'training/config')});
  const browserLog=createWriteStream(path.join(root,'browser.jsonl'),{mode:0o600});
  harness.page.on('console',message=>browserLog.write(JSON.stringify({at:Date.now(),stage:current,type:message.type(),text:message.text()})+'\n'));
  await harness.page.evaluate(async timeout=>{
    const api=await import('/dist/index.js');globalThis.api=api;
    globalThis.opfs=await navigator.storage.getDirectory();globalThis.modelTools=await api.openBrowserModelTools({timeoutMs:timeout,onProgress:p=>console.log(JSON.stringify(p))});
    globalThis.sources={};globalThis.splits={};globalThis.tasks={};
    globalThis.fileAt=async(directory,name,create=false)=>{const parts=name.split('/');for(const part of parts.slice(0,-1))directory=await directory.getDirectoryHandle(part,{create});return directory.getFileHandle(parts.at(-1),{create});};
    globalThis.dirAt=async(directory,name)=>{for(const part of name.split('/'))directory=await directory.getDirectoryHandle(part);return directory;};
    globalThis.transfer=async handle=>{const name=crypto.randomUUID()+'.bin';const response=await fetch('/artifact/'+name,{method:'POST',body:await handle.getFile()});if(!response.ok)throw new Error('Artifact transfer failed');return name;};
    globalThis.exportArtifact=async artifact=>{
      const manifestFile=await artifact.manifest.getFile(),manifest=await manifestFile.text(),metadata=JSON.parse(manifest),transfers=[];
      for(const name of Object.keys(metadata.files))transfers.push([name,await transfer(await fileAt(artifact.directory,name))]);
      return {manifestName:manifestFile.name,manifest,transfers};
    };
  },timeout);
  await harness.page.evaluate(fixtures=>{globalThis.fixtures=fixtures;},fixtures);
  for(const model of models) {
    const modelRoot=path.join(root,model);
    await stage(model+'/source',()=>harness.page.evaluate(async({model,fixture})=>{
      const directory=await opfs.getDirectoryHandle('source-'+model,{create:true});
      for(const name of [fixture.model,...fixture.inputs.map(input=>input.csv)]) {
        const response=await fetch('/models/'+(fixture.directory||model)+'/'+name);if(!response.ok)throw new Error('Missing fixed source');
        const writer=await(await fileAt(directory,name,true)).createWritable();await writer.write(await response.arrayBuffer());await writer.close();
      }
      const manifest=await directory.getFileHandle('inputs.json',{create:true}),writer=await manifest.createWritable();await writer.write(JSON.stringify({inputs:fixture.inputs,model_kind:fixture.modelKind||'standard'}));await writer.close();
      sources[model]={directory,modelPath:fixture.model,inputsManifest:manifest};
    },{model,fixture:fixtures[model]}));
    await stage(model+'/split',async()=>saveArtifact(await harness.page.evaluate(async model=>{
      splits[model]=await modelTools.prepareSplit({source:sources[model],ratio:'0.9'});return exportArtifact(splits[model]);
    },model),path.join(modelRoot,'split')));
    await stage(model+'/oracle',()=>nativeCheck('oracle',['--browser',path.join(modelRoot,'split'),'--output',path.join(root,'oracle',model),'--modes',modes.map(fullMode).join(',')],path.join(root,'oracle',model,'native.log')));
    await stage(model+'/kernel-audit',async()=>{
      const result=await harness.page.evaluate(async model=>{
        const directory=await opfs.getDirectoryHandle('kernel-audit-'+model,{create:true});
        const {cleanupPythonWorkspace}=await import('/dist/python-workspace.js'),jobId=crypto.randomUUID();
        const worker=new Worker('/dist/model-worker.js',{type:'module'});
        try {
          const artifact=await new Promise((resolve,reject)=>{
            worker.onerror=()=>reject(new Error('Kernel audit Worker failed'));
            worker.onmessage=({data})=>{if(data.type==='result')resolve(data.result);else if(data.type==='error')reject(new Error(data.message));};
            worker.postMessage({protocolVersion:2,type:'diagnoseORT',requestId:1,jobId,
              source:sources[model],outputDirectory:directory});
          });
          return exportArtifact(artifact);
        }finally{worker.terminate();await cleanupPythonWorkspace(jobId);}
      },model);
      await saveArtifact(result,path.join(root,'oracle',model,'browser-kernels'));
      await nativeCheck('kernel-audit',['--browser',path.join(root,'oracle',model,'browser-kernels'),
        '--output',path.join(root,'oracle',model,'kernel-comparison.json')],path.join(root,'oracle',model,'kernel-comparison.log'));
    });
    for(const mode of modes) {
      const unit=path.join(modelRoot,mode),deploymentMode=fullMode(mode),key=model+'-'+mode;
      const record={model,mode,deploymentMode,status:'running'};summary.units.push(record);
      await stage(key+'/high',async()=>saveArtifact(await harness.page.evaluate(async({model,mode,deploymentMode,key})=>{
        const config=fixtures[model].config||(mode==='parameter'?'storage_aware_config.json':null);
        const compileConfig=config?new File([await(await fetch('/config/'+config)).arrayBuffer()],'config.json'):undefined;
        tasks[key]=await modelTools.compileHighLevel({split:splits[model],deploymentMode,compileConfig});return exportArtifact(tasks[key]);
      },{model,mode,deploymentMode,key}),path.join(unit,'high')));
      const metadata=await stage(key+'/low',async()=>saveArtifact(await harness.page.evaluate(async key=>{
        tasks[key]=await modelTools.generateLowLevel({task:tasks[key]});return exportArtifact(tasks[key]);
      },key),path.join(unit,'task')));
      await stage(key+'/compiler-comparison',()=>nativeCheck('verify-task',['--browser',path.join(unit,'task'),'--oracle',path.join(root,'oracle',model),'--output',path.join(unit,'compiler-comparison.json')],path.join(unit,'compiler-comparison.log')));
      const setup=await stage(key+'/setup',()=>harness.page.evaluate(async({key,mode})=>{
        const manifest=JSON.parse(await(await tasks[key].manifest.getFile()).text()),role=mode==='input'?'client':'provisioner';
        globalThis.clientDirectory=await dirAt(tasks[key].directory,manifest.entry+'/'+role);
        globalThis.keyDirectory=await opfs.getDirectoryHandle('shared-profile-keys',{create:true});
        globalThis.client=await api.openBrowserClient({clientDirectory,keyDirectory,onProgress:p=>console.log(JSON.stringify(p))});
        let setup,generated=true;
        try{setup=await client.setup();globalThis.evalFile=setup.evalContext.file;}
        catch(error) {
          if(!error.message.startsWith('manifest.json exists;'))throw error;
          generated=false;const restored=await client.restore();
          const directory=await keyDirectory.getDirectoryHandle(restored.profileId);
          const keys=JSON.parse(await(await(await directory.getFileHandle('manifest.json')).getFile()).text());
          globalThis.evalFile=await fileAt(directory,'public/eval_context.bin');
          setup={...restored,evalContext:{bytes:keys.artifacts.find(ref=>ref.role==='public-eval').bytes},
            fullContext:{bytes:keys.artifacts.find(ref=>ref.role==='private-full').bytes}};
        }
        const result={profileId:setup.profileId,keyIdentity:setup.keyIdentity,evalBytes:setup.evalContext.bytes,fullBytes:setup.fullContext.bytes,
          generated,source:generated?'browser setup':'compatible browser profile restored from this acceptance run'};
        await client.close();return result;
      },{key,mode}));
      await json(path.join(unit,'keys.json'),setup);
      const restored=await stage(key+'/restore',()=>harness.page.evaluate(async()=>{
        globalThis.client=await api.openBrowserClient({clientDirectory,keyDirectory,onProgress:p=>console.log(JSON.stringify(p))});return client.restore();
      }));
      if(restored.profileId!==setup.profileId||restored.keyIdentity!==setup.keyIdentity)throw new Error('Key identity changed after restore');
      if(model==='cifar10'&&mode==='input')await stage(key+'/sample-and-binding-regressions',async()=>
        json(path.join(unit,'binding-regressions.json'),await harness.page.evaluate(sampleAndBindingRegressions,key)));
      const contract=JSON.parse(await readFile(path.join(unit,'task',metadata.entry,'input_contract.json'),'utf8'));
      const nativeArgs=['--mode',deploymentMode,'--output',path.join(unit,'output.ct'),'--device',device];
      if(backend==='gpu')nativeArgs.push('--gpu');
      if(mode==='input') {
        const contextKey=setup.profileId+'-'+setup.keyIdentity;
        const encrypted=await stage(key+'/encrypt',()=>harness.page.evaluate(async({key,transferEvaluation})=>{
          const metadata=JSON.parse(await(await tasks[key].manifest.getFile()).text());
          const contract=JSON.parse(await(await(await fileAt(tasks[key].directory,metadata.entry+'/input_contract.json')).getFile()).text()),files=[];
          for(const input of contract.inputs) {
            const file=await fileAt(tasks[key].directory,input.csv.slice(metadata.virtualRoot.length+1));
            const result=await client.encrypt({inputName:input.feature_id,inputFile:file});files.push([input.feature_id,await transfer(result.file)]);
          }
          return {files,evaluation:transferEvaluation?await transfer(evalFile):null};
        },{key,transferEvaluation:!evalContexts.has(contextKey)}));
        if(encrypted.evaluation) {
          await rename(path.join(root,encrypted.evaluation),path.join(unit,'eval_context.bin'));
          evalContexts.set(contextKey,path.join(unit,'eval_context.bin'));
        }else await link(evalContexts.get(contextKey),path.join(unit,'eval_context.bin'));
        nativeArgs.push('--task-dir',path.join(unit,'task',metadata.entry,'server'),'--eval-context',path.join(unit,'eval_context.bin'));
        for(const [index,[id,transfer]]of encrypted.files.entries()) {
          const file=path.join(unit,'input-'+index+'.ct');await rename(path.join(root,transfer),file);nativeArgs.push('--input',id+'='+file);
        }
      } else {
        await stage(key+'/provision',async()=>saveArtifact(await harness.page.evaluate(async key=>{
          globalThis.provisionedBundle=await client.provisionRunner({task:tasks[key],parameterDirectory:clientDirectory});return exportArtifact(provisionedBundle);
        },key),path.join(unit,'bundle')));
        const contextKey=setup.profileId+'-'+setup.keyIdentity,context=path.join(unit,'bundle/eval_context.bin');
        if(evalContexts.has(contextKey)) {
          if(await hash(context)!==await hash(evalContexts.get(contextKey)))throw new Error('Reused public evaluation context differs');
          await unlink(context);await link(evalContexts.get(contextKey),context);
        }else evalContexts.set(contextKey,context);
        await harness.page.evaluate(async()=>{
          // The exact committed Runner has been exported; retain its host evidence.
          for await(const entry of provisionedBundle.directory.values())await provisionedBundle.directory.removeEntry(entry.name,{recursive:entry.kind==='directory'});
        });
        await stage(key+'/parameter-check',()=>nativeCheck('verify-bundle',['--browser',path.join(unit,'task'),'--bundle',path.join(unit,'bundle'),'--output',path.join(unit,'parameter-check.json')],path.join(unit,'parameter-check.log')));
        nativeArgs.push('--task-dir',path.join(unit,'bundle'));
        for(const input of contract.inputs)nativeArgs.push('--input',input.feature_id+'='+path.join(unit,'task',input.csv.slice(metadata.virtualRoot.length+1)));
      }
      await stage(key+'/native',()=>run('/usr/bin/time',['-v','-o',path.join(unit,'native-memory.txt'),runner,...nativeArgs],path.join(unit,'native.log')));
      await stage(key+'/output-framing-regressions',async()=>json(path.join(unit,'output-framing-regressions.json'),
        await harness.page.evaluate(outputFramingRegressions,{key,model,mode})));
      await stage(key+'/decrypt',async()=>{
        const result=await harness.page.evaluate(async({key,model,mode})=>{
          const response=await fetch('/artifact/'+model+'/'+mode+'/output.ct');if(!response.ok)throw new Error('Native result missing');
          const result=await client.decryptOutputs({task:tasks[key],outputFile:new File([await response.arrayBuffer()],'output.ct')});
          const file=await result.manifest.getFile(),manifest=await file.text(),metadata=JSON.parse(manifest),transfers=[];
          for(const output of metadata.outputs)transfers.push([output.file,await transfer(await result.directory.getFileHandle(output.file))]);
          await client.close();return {manifestName:file.name,manifest,transfers};
        },{key,model,mode});
        await saveArtifact(result,path.join(unit,'results'));
      });
      await stage(key+'/compare',()=>nativeCheck('compare',['--browser',path.join(unit,'task'),'--results',path.join(unit,'results'),'--output',path.join(unit,'comparison.json')],path.join(unit,'comparison.log')));
      record.status='complete';record.comparison=JSON.parse(await readFile(path.join(unit,'comparison.json'),'utf8'));
    }
  }
  if(models.includes('cifar10')&&models.includes('retina'))await stage('state-isolation/cifar10-repeat',async()=>{
    const result=await harness.page.evaluate(async()=>{
      const repeated=await modelTools.prepareSplit({source:sources.cifar10,ratio:'0.9'});
      if(await repeated.directory.isSameEntry(splits.cifar10.directory))throw new Error('Repeated split overwrote committed directory');
      return exportArtifact(repeated);
    });
    await saveArtifact(result,path.join(root,'state-isolation/cifar10-repeat'));
    const first=JSON.parse(await readFile(path.join(root,'cifar10/split/artifact-manifest.json'),'utf8'));
    const repeated=JSON.parse(result.manifest);
    for(const [name,entry]of Object.entries(first.files)) {
      if(name===first.entry) {
        const original=JSON.parse((await readFile(path.join(root,'cifar10/split',name),'utf8')).replaceAll(first.virtualRoot,'/ROOT'));
        const current=JSON.parse((await readFile(path.join(root,'state-isolation/cifar10-repeat',name),'utf8')).replaceAll(repeated.virtualRoot,'/ROOT'));
        delete original.validation.prepare_wall_s;delete current.validation.prepare_wall_s;
        if(JSON.stringify(original)!==JSON.stringify(current))throw new Error('Repeated split manifest changed');
      }else if(repeated.files[name]?.sha256!==entry.sha256)throw new Error('Repeated split content changed: '+name);
    }
  });
  summary.status='complete';browserLog.end();
}catch(error) {
  summary.status='failed';summary.unfinished.push({stage:current,error:error.message});process.exitCode=1;console.error(error);
}finally {
  if(harness) {
    await harness.page.evaluate(async()=>{await globalThis.client?.close();await globalThis.modelTools?.close();}).catch(()=>{});
    summary.terminals=await harness.page.evaluate(()=>({requests:Object.keys(lattiTerminalCounts).length,
      duplicates:Object.entries(lattiTerminalCounts).filter(([,count])=>count>1)})).catch(()=>null);
    if(summary.terminals?.duplicates.length){summary.status='failed';summary.unfinished.push({stage:'protocol',error:'Duplicate Worker terminal response'});process.exitCode=1;}
    await harness.close();
    const memory={...harness.rss,phases:harness.phaseMetrics,rendererPeakBytes:harness.rss.rendererPeakKiB*1024,
      marginBytes:4294967296-harness.rss.rendererPeakKiB*1024,
      accounting:'Renderer RSS sums include Workers and WASM. JS/linear-memory capacities are diagnostics and are not added to RSS.'};
    await json(path.join(root,'memory.json'),memory);
    if(!(memory.rendererPeakBytes<4294967296)){summary.status='failed';summary.unfinished.push({stage:'memory',error:'Renderer RSS must be strictly below 4 GiB'});process.exitCode=1;}
  }
  await json(path.join(root,'summary.json'),summary);
}
