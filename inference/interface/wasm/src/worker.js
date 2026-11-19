import './wasm_exec.js';
import {BUILD_INFO} from './build-info.js';
import {runnerOutputs} from './runner-output.js';
import {csvChunks} from './csv.js';
import {openParameterModel} from './parameter-model.js';
import {stageCSV,numericToCSV} from './csv.js';
import {writeZip,openZip,checkedZipName} from './zip.js';
import {fileOf,fileSource,safeName,setErrorMessage,writeFile,copyFile,pump,jobDirectory,cleanupOrphanJobs,cleanupAfter,artifact,commit} from './io.js';

const {protocolVersion:PROTOCOL_VERSION,runtimeVersion:RUNTIME_VERSION,build:BUILD_ID}=BUILD_INFO;
const sessionJobId=crypto.randomUUID();
const send=(type,requestId=0,jobId=sessionJobId,fields={})=>postMessage({...fields,protocolVersion:PROTOCOL_VERSION,requestId,jobId,type});
let go;
let runtimeFailure, runtimeInitialized=false;
const runtimeFailed=error=>{
  runtimeFailure=error;
  if(runtimeInitialized) send('error',0,sessionJobId,{code:'RUNTIME_EXITED',phase:'runtime',message:error.message,retryable:false});
};
const wasmReady = (async () => {
  go = new Go();
  const response = await fetch(new URL('./latti_client.wasm', import.meta.url));
  if (!response.ok) throw new Error(`WASM_FETCH_FAILED_${response.status}`);
  const fallback=response.clone();
  let instance;
  try {
    ({instance} = await WebAssembly.instantiateStreaming(response, go.importObject));
  } catch (error) {
    ({instance} = await WebAssembly.instantiate(await fallback.arrayBuffer(), go.importObject));
  }
  const run = go.run(instance);
  run.then(()=>runtimeFailed(new Error('WASM runtime exited unexpectedly')),runtimeFailed);
  const deadline = performance.now() + 30000;
  while (typeof globalThis.lattiCore !== 'function') {
    if (runtimeFailure) throw runtimeFailure;
    if (performance.now() > deadline) throw new Error('WASM_RUNTIME_TIMEOUT');
    await new Promise(resolve => setTimeout(resolve, 0));
  }
})();
const runtimeReady=wasmReady.then(async () => {
  if (!navigator.storage?.getDirectory || !navigator.locks?.request) throw new Error('OPFS and Web Locks are required');
  const opfs=await navigator.storage.getDirectory();
  const probeName=`probe-${sessionJobId}`;
  try {
    const file=await opfs.getFileHandle(probeName,{create:true});
    const writer=await file.createWritable();
    try {await writer.write(new Uint8Array([1]));await writer.close();} catch(error) {await writer.abort().catch(()=>{});throw error;}
    if((await file.getFile()).size!==1) throw new Error('OPFS probe failed');
  } finally {await opfs.removeEntry(probeName).catch(()=>{});}
  const actual=core('info').data;
  if(actual.runtimeVersion!==RUNTIME_VERSION || actual.protocolVersion!==PROTOCOL_VERSION || actual.build!==BUILD_ID) throw new Error('WASM_BUILD_MISMATCH');
  let rawDeflate=false;try {new DecompressionStream('deflate-raw');rawDeflate=true;} catch {}
  await navigator.locks.request('latti-shared-export',{mode:'exclusive'},cleanupOrphanJobs);
  runtimeInitialized=true;
  send('ready',0,sessionJobId,{...actual,capabilities:{protocolVersion:PROTOCOL_VERSION,opfs:true,zip64:true,rawDeflate}});
});
runtimeReady.catch(error => {
  runtimeFailure=error;
  send('error',0,sessionJobId,{code:'RUNTIME_INIT_FAILED',phase:'initialization',message:error.message,retryable:false});
});
const core = (operation,...args) => {
  if(runtimeFailure) throw runtimeFailure;
  const result = globalThis.lattiCore(operation,...args);
  if (result.error) throw new Error(result.error);
  return result;
};

let state='initializing',keyReady=false,task,profile,keyDirectory,taskDirectory,initialParameters,keyIdentity,inputContract;
let active,queue=Promise.resolve(),closing=false,lastProgress=0;
const cancelled=()=>Object.assign(new Error('Operation cancelled'),{code:'CANCELLED'});
const check=()=>{if(active?.controller.signal.aborted) throw cancelled();};
const read=async(directory,name,maxBytes=64*1024*1024)=>{
  const file=await(await directory.getFileHandle(name)).getFile();
  if(file.size>maxBytes) throw Object.assign(new Error(name+': METADATA_BUDGET_EXCEEDED'),{code:'METADATA_BUDGET_EXCEEDED'});
  return file.text();
};
const progress=(phase,bytes,result={},final=false)=>{
  active.progress={phase,bytes:String(bytes),elapsedMs:performance.now()-active.start,wasmBytes:go.mem.buffer.byteLength,heapBytes:result.heapBytes||0,chunkBytes:result.data?.chunk?.byteLength||0,inFlightChunks:result.data?.chunk?1:0,...(active.entry?{entry:active.entry}:{}),final};
  const now=performance.now();if(!final&&now-lastProgress<100) return;lastProgress=now;
  send('progress',active.requestId,active.jobId,{progress:active.progress});
};
const assertAbsent=async(directory,name)=>{
  try {await directory.getFileHandle(name);} catch(error) {if(error.name==='NotFoundError') return;throw error;}
  throw new Error(name+' exists; restore it or select a new directory');
};
const bindTask=async directory=>{
  const [config,parameters]=await Promise.all([read(directory,'task_config.json'),read(directory,'ckks_parameter.json')]);
  if(!config.trim()) throw new Error('Task configuration is required');
  const parsed=JSON.parse(config);
  let ownerManifest;
  try{ownerManifest=JSON.parse(await read(directory,'task_manifest.json'));}
  catch(error){if(error.name!=='NotFoundError'||parsed.task_manifest_required)throw error;}
  if(ownerManifest) {
    const manifest=ownerManifest;
    if(manifest.schema!=='latti-ai.browser-client-task.v1'||manifest.deployment_mode!==parsed.deployment_mode)throw new Error('Task manifest mode differs');
    for(const name of ['task_config.json','ckks_parameter.json','task_signature.json','key_requirements.json'])
      if(!manifest.files?.[name])throw new Error('Task manifest omits '+name);
    for(const [name,entry]of Object.entries(manifest.files)) {
      const file=await(await directory.getFileHandle(safeName(name))).getFile();
      const bytes=await file.arrayBuffer(),sha256=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
      if(file.size!==entry.bytes||sha256!==entry.sha256)throw new Error('Task file hash mismatch: '+name);
    }
  }
  let contract;
  try {contract=JSON.parse(await read(directory,'input_contract.json'));}
  catch(error){if(error.name!=='NotFoundError')throw error;}
  if(contract) {
    if((parsed.deployment_mode||'client_encrypted_input')!==contract.deployment_mode||
      JSON.stringify(parsed.task_output_param)!==JSON.stringify(contract.outputs))throw new Error('Task mode/output contract differs');
    const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(parameters))),b=>b.toString(16).padStart(2,'0')).join('');
    if(digest!==contract.ckks_sha256)throw new Error('Task CKKS contract hash differs');
  }
  if(ownerManifest) {
    const signature=JSON.parse(await read(directory,'task_signature.json'));
    const outputs=signature.online.filter(arg=>arg.phase==='out');
    if(outputs.length!==Object.keys(parsed.task_output_param).length||new Set(outputs.map(arg=>arg.id)).size!==outputs.length)
      throw new Error('Task output signature IDs differ');
    for(const arg of outputs) {
      const packing=parsed.task_output_param[arg.id];
      if(!packing||arg.type!=='ct'||arg.level!==packing.level||!Array.isArray(arg.size)||arg.size.length!==1||!Number.isSafeInteger(arg.size[0])||arg.size[0]<=0)
        throw new Error('Task output signature differs: '+arg.id);
      if(packing.data_type!=='feature_mat') {
        let count=Math.ceil(packing.channel/packing.pack_num);
        if(packing.dim===2)count*=packing.shape.reduce((n,dim,i)=>n*Math.ceil(dim/parsed.block_shape[i]),1);
        if(arg.size[0]!==count)throw new Error('Task output signature packing differs: '+arg.id);
      }
    }
  }
  core('bind-task',config,parameters);task=parsed;taskDirectory=directory;inputContract=contract;
  return {profileId:profile.profileId};
};
const inputNames=()=>{
  const names=task.task_input_id||Object.keys(task.task_input_param);
  if(!Array.isArray(names)||names.some(name=>typeof name!=='string'||!Object.hasOwn(task.task_input_param,name))||new Set(names).size!==names.length) throw new Error('Invalid task input IDs');
  return [...names].sort();
};
const requireFeature=(name,encrypting)=>{
  safeName(name);
  const table=encrypting?task?.task_input_param:task?.task_output_param;
  if(!table||!Object.hasOwn(table,name)) throw new Error('Unknown task feature: '+name);
};
const files=async(directory,pattern)=>{
  const result=[],names=new Set();
  for await(const file of directory.values()) {
    check();if(file.kind!=='file'||!pattern.test(file.name)) continue;
    safeName(file.name);const key=file.name.normalize('NFC').toLowerCase();
    if(names.has(key)) throw new Error('Duplicate input name: '+file.name);names.add(key);result.push(file);
    if(result.length>100000) throw new Error('METADATA_BUDGET_EXCEEDED');
  }
  return result.sort((a,b)=>a.name<b.name?-1:a.name>b.name?1:0);
};
const caseId=file=>{const name=safeName(file.name.replace(/\.csv$/i,''));checkedZipName(name);if(name.includes('__')) throw new Error('Case ID contains reserved separator');return name;};

async function hashFile(file,directory,name) {
  const work=await directory.getDirectoryHandle(name,{create:true}),snapshot=await fileOf(file);
  const result=await pump(core,'sha256',['',String(snapshot.size)],work,{bytes:fileSource(snapshot)},check,progress);
  return (await(await result.get('sha256').handle.getFile()).text()).trim();
}

async function setup() {
  const publicDir=await keyDirectory.getDirectoryHandle('public',{create:true});
  const privateDir=await keyDirectory.getDirectoryHandle('secret',{create:true});
  await assertAbsent(keyDirectory,'manifest.json');
  for(const [dir,name] of [[publicDir,'eval_context.bin'],[privateDir,'full_context.bin']]) await assertAbsent(dir,name);
  const job=await jobDirectory(active.jobId),created=[];let failure,committed=false;
  keyReady=false;
  try {
    const pending=await keyDirectory.getFileHandle('pending.json',{create:true});created.push([keyDirectory,'pending.json']);
    await writeFile(pending,JSON.stringify({profileId:profile.profileId,jobId:active.jobId}));
    const outputs=await pump(core,'setup',[],job.directory,{},check,progress);
    const refs=[];
    for(const [id,dir,name,role] of [['eval',publicDir,'eval_context.bin','public-eval'],['full',privateDir,'full_context.bin','private-full']]) {
      check();const file=await dir.getFileHandle(name,{create:true});created.push([dir,name]);
      const bytes=await copyFile(outputs.get(id).handle,file,check);
      if(bytes!==outputs.get(id).bytes) throw new Error('Export size mismatch');
      const ref=await artifact(dir,name,role,file,profile.profileId);
      ref.sha256=await hashFile(file,job.directory,'hash-'+id);refs.push(ref);
    }
    check();
    const manifest=await keyDirectory.getFileHandle('manifest.json',{create:true});created.push([keyDirectory,'manifest.json']);
    keyIdentity=core('key-identity').data;
    await writeFile(manifest,JSON.stringify({state:'committed',runtimeVersion:RUNTIME_VERSION,protocolVersion:PROTOCOL_VERSION,profile,keyIdentity,artifacts:refs.map(ref=>({artifactId:ref.artifactId,role:ref.role,path:(ref.role==='private-full'?'secret/':'public/')+ref.file.name,bytes:ref.bytes,sha256:ref.sha256}))})+'\n');
    committed=true;
    await keyDirectory.removeEntry('pending.json');
    keyReady=true;
    return {evalContext:refs[0],fullContext:refs[1],manifest,profileId:profile.profileId,keyIdentity};
  } catch(error) {
    failure=error;
    await cleanupAfter(error,(committed?[]:created).map(([dir,name])=>async()=>{
      try {await dir.removeEntry(name);} catch(e) {throw setErrorMessage(e,e.message+'; residual key directory/'+name);}
    }));
    try {core('close');core('init-keys',initialParameters,profile.keyPlan.bootstrapping);}
    catch(resetError) {setErrorMessage(error,error.message+'; client reset: '+resetError.message);runtimeFailure=resetError;}
    task=undefined;
    throw error;
  } finally {await cleanupAfter(failure,[()=>job.cleanup()]);}
}

async function openRun(run,contract) {
  const manifestFile=await run.manifest.getFile();
  if(manifestFile.size>1024*1024)throw new Error('Run manifest exceeds metadata budget');
  const manifest=JSON.parse(await manifestFile.text());
  if(manifest.kind!=='run'||manifest.schema!=='latti-ai.browser-model-artifact.v1')throw new Error('A runPrefix artifact is required');
  const get=async name=>{
    if(typeof name!=='string'||!name||/[\\\0:]/.test(name)||name.split('/').some(p=>!p||p==='.'||p==='..'))throw new Error('Invalid run artifact path');
    let directory=run.directory;const parts=name.split('/');
    for(const part of parts.slice(0,-1))directory=await directory.getDirectoryHandle(part);
    const file=await(await directory.getFileHandle(parts.at(-1))).getFile(),expected=manifest.files[name];
    if(!expected||file.size!==expected.bytes||file.size>128*1024*1024)throw new Error('Run artifact size differs');
    const sha=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',await file.arrayBuffer())),b=>b.toString(16).padStart(2,'0')).join('');
    if(sha!==expected.sha256)throw new Error('Run artifact hash mismatch');
    return file;
  };
  const binding=JSON.parse(await(await get(manifest.entry)).text());
  if(binding.split_manifest_sha256!==contract.manifest_sha256)throw new Error('Run is bound to a different split');
  const samplePath=manifest.entry.slice(0,manifest.entry.lastIndexOf('/')+1)+binding.sample_manifest;
  if(manifest.files[samplePath]?.sha256!==binding.sample_manifest_sha256)throw new Error('Run sample manifest binding differs');
  const sample=JSON.parse(await(await get(samplePath)).text());
  return {sample,samplePath,get};
}

async function transform(encrypting,options,work) {
  requireFeature(encrypting?options.inputName:options.outputName,encrypting);check();
  const source=await fileOf(encrypting?options.inputFile:options.ciphertextFile);
  if(encrypting&&inputContract) {
    const entry=inputContract.inputs.find(input=>input.feature_id===options.inputName);
    if(!entry)throw new Error('Input feature is absent from split contract');
    let expected=entry.sha256;
    if(options.run) {
      const {sample}=await openRun(options.run,inputContract);
      const actual=sample.inputs.find(input=>input.name===entry.name);
      if(!actual||JSON.stringify(actual.shape)!==JSON.stringify(entry.shape)||actual.dtype!==entry.dtype||actual.representation!==entry.representation)throw new Error('Run crossing interface differs');
      expected=actual.sha256;
    }
    const job=await jobDirectory(active.jobId+'-binding');let failure;
    try {if(await hashFile(source,job.directory,'input')!==expected)throw new Error('Crossing CSV content differs from bound sample');}
    catch(error){failure=error;throw error;}finally{await cleanupAfter(failure,[()=>job.cleanup()]);}
  }
  const name=safeName(source.name.replace(encrypting?/\.csv$/i:/\.ct\.bin$/i,'')+(encrypting?'.ct.bin':'.csv'));
  const numeric=await work.getFileHandle('numeric.tmp',{create:true});
  if(encrypting) {
    const bytes=await stageCSV(source,numeric,check);
    const outputs=await pump(core,'encrypt',[options.inputName,String(bytes)],work,{values:fileSource(numeric)},check,progress);
    return {name,file:outputs.get('ciphertext').handle,role:'ciphertext'};
  }
  const incoming=await work.getFileHandle('incoming.ct',{create:true});await copyFile(source,incoming,check);
  const outputs=await pump(core,'decrypt',[options.outputName,String(source.size)],work,{ciphertext:fileSource(incoming)},check,progress);
  const csv=await work.getFileHandle('plaintext.part',{create:true});
  await numericToCSV(await outputs.get('values').handle.getFile(),csv,check);
  return {name,file:csv,role:'plaintext'};
}

async function single(encrypting,options) {
  const job=await jobDirectory(active.jobId);let retained=false,destination,failure;
  try {
    const work=await job.directory.getDirectoryHandle('work',{create:true});
    const converted=await transform(encrypting,options,work);
    destination=options.outputDirectory?await options.outputDirectory.getDirectoryHandle(active.jobId,{create:true}):await job.directory.getDirectoryHandle('output',{create:true});
    const file=await destination.getFileHandle(converted.name,{create:true});
    await copyFile(converted.file,file,check);check();
    const ref=await artifact(destination,converted.name,converted.role,file,profile.profileId);
    const manifest=await commit(destination,[ref],profile.profileId);
    retained=true;return {...ref,manifest};
  } catch(error) {failure=error;throw error;} finally {
    const actions=[];
    if(!retained&&options.outputDirectory&&destination) actions.push(()=>options.outputDirectory.removeEntry(active.jobId,{recursive:true}));
    actions.push(()=>!retained||options.outputDirectory?job.cleanup():job.directory.removeEntry('work',{recursive:true}));
    await cleanupAfter(failure,actions);
  }
}

async function decryptOutputs(options) {
  const manifest=JSON.parse(await(await options.task.manifest.getFile()).text());
  if(manifest.schema!=='latti-ai.browser-model-artifact.v1'||manifest.kind!=='task')throw new Error('Compiled task artifact required');
  const get=async name=>{
    if(typeof name!=='string'||!name||/[\\\0:]/.test(name)||name.split('/').some(part=>!part||part==='.'||part==='..'))throw new Error('Invalid artifact path');
    let directory=options.task.directory;const parts=name.split('/');
    for(const part of parts.slice(0,-1))directory=await directory.getDirectoryHandle(part);
    const file=await(await directory.getFileHandle(parts.at(-1))).getFile(),metadata=manifest.files[name];
    if(!metadata||file.size!==metadata.bytes)throw new Error('Missing output artifact binding');
    const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',await file.arrayBuffer())),b=>b.toString(16).padStart(2,'0')).join('');
    if(hash!==metadata.sha256)throw new Error('Output artifact hash mismatch');
    return file;
  };
  const contract=JSON.parse(await(await get(manifest.entry+'/input_contract.json')).text());
  const prefix=manifest.virtualRoot+'/';
  if(!contract.manifest.startsWith(prefix))throw new Error('Split reference escapes artifact');
  const splitPath=contract.manifest.slice(prefix.length);
  const run=options.run?await openRun(options.run,contract):null;
  const split=run?run.sample:JSON.parse(await(await get(splitPath)).text());
  const outputs=await runnerOutputs(options.outputFile,Object.keys(contract.output_map));
  if(JSON.stringify(task.task_output_param)!==JSON.stringify(contract.outputs)||
    (task.deployment_mode||'client_encrypted_input')!==contract.deployment_mode)throw new Error('Bound output task differs from contract');
  const job=await jobDirectory(active.jobId);let destination,retained=false,failure;
  try {
    destination=options.outputDirectory?await options.outputDirectory.getDirectoryHandle(active.jobId,{create:true}):await job.directory.getDirectoryHandle('output',{create:true});
    const entries=[];
    for(const [index,route]of split.output_routes.entries()) {
      check();
      const expected=route.shape.reduce((a,b)=>a*b,1),name=`output-${index}.csv`;
      const file=await destination.getFileHandle(name,{create:true});
      if(route.route==='suffix') {
        const ids=Object.keys(contract.output_map).filter(id=>contract.output_map[id]===route.name);
        if(ids.length!==1)throw new Error('Ambiguous output route');
        const work=await job.directory.getDirectoryHandle('work-'+index,{create:true});
        const converted=await transform(false,{outputName:ids[0],ciphertextFile:new File([outputs.get(ids[0])],ids[0]+'.ct.bin')},work);
        await copyFile(converted.file,file,check);
        await job.directory.removeEntry('work-'+index,{recursive:true});
      } else {
        const reference=route.plaintext;
        if(!reference)throw new Error('Split lacks portable plaintext output routes; prepare it again');
        const referencePath=run?run.samplePath:splitPath;
        const source=await(run?run.get:get)(referencePath.slice(0,referencePath.lastIndexOf('/')+1)+reference.csv);
        await copyFile(source,file,check);
      }
      let count=0;
      for await(const values of csvChunks(await file.getFile())){check();count+=values.length;}
      if(count!==expected)throw new Error('Output count differs from original shape');
      entries.push({name:route.name,shape:route.shape,dtype:route.dtype,count,file:name,
        numericDtype:route.route==='suffix'?'float64':route.dtype,route:route.route});
    }
    const outputManifest=await commit(destination,[],profile.profileId,{outputs:entries,
      deploymentMode:contract.deployment_mode,sourceManifest:contract.manifest_sha256});
    retained=true;return {directory:destination,manifest:outputManifest};
  }catch(error){failure=error;throw error;}finally {
    const actions=[];
    if(!retained&&destination&&options.outputDirectory)actions.push(()=>options.outputDirectory.removeEntry(active.jobId,{recursive:true}));
    if(!retained||options.outputDirectory)actions.push(()=>job.cleanup());
    await cleanupAfter(failure,actions);
  }
}

async function provisionRunner(options) {
  const job=await jobDirectory(active.jobId);let producer,destination,retained=false,failure;
  try {
    producer=await openParameterModel(options.task,options.parameterDirectory,active.controller.signal,
      event=>progress('parameters-'+event.phase,0));
    check();
    const {metadata}=producer,plan=JSON.parse(metadata.plan);
    core('bind-task',metadata.config,metadata.parameters);task=JSON.parse(metadata.config);
    destination=options.outputDirectory?await options.outputDirectory.getDirectoryHandle(active.jobId,{create:true}):await job.directory.getDirectoryHandle('runner',{create:true});
    const work=await job.directory.getDirectoryHandle('work',{create:true});
    const sources=new Proxy({}, {get(_,id) {
      const match=typeof id==='string'&&/^parameter:(\d+):(\d+)$/.exec(id);
      if(!match)return undefined;
      const item=plan[Number(match[1])],index=Number(match[2]);
      if(!item||!Number.isSafeInteger(index)||index<0||index>=item.count)throw new Error('Invalid parameter request');
      return {size:BigInt(item.slots*8),async readAt(offset,length) {
        check();
        if(offset!==0n||length!==BigInt(item.slots*8))throw new Error('Parameter source must request one full slot vector');
        const bytes=await producer.value(item.id,index);check();
        if(!(bytes instanceof Uint8Array)||bytes.byteLength!==item.slots*8)throw new Error('Invalid parameter materialization length');
        return bytes;
      }};
    }});
    const outputs=await pump(core,'provision',[metadata.plan],work,sources,check,progress);
    await producer.close();producer=undefined;
    const encrypted=await destination.getDirectoryHandle('encrypted_model_parameters',{create:true});
    const publicFiles=[];
    for(const [name,bytes]of Object.entries(metadata.files)) {
      const file=await destination.getFileHandle(safeName(name),{create:true});await writeFile(file,bytes);publicFiles.push([name,file]);
    }
    const graph=await destination.getFileHandle('nn_layers_ct_0.json',{create:true});
    await writeFile(graph,metadata.publicGraph);publicFiles.push(['nn_layers_ct_0.json',graph]);
    for(let index=0;index<plan.length;index++) {
      const name=`parameter-${index}.bin`,file=await encrypted.getFileHandle(name,{create:true});
      await copyFile(outputs.get('parameter-'+index).handle,file,check);publicFiles.push(['encrypted_model_parameters/'+name,file]);
    }
    const storeManifest=await encrypted.getFileHandle('manifest.json',{create:true});
    await copyFile(outputs.get('parameter-manifest').handle,storeManifest,check);
    publicFiles.push(['encrypted_model_parameters/manifest.json',storeManifest]);
    const publicKeys=await keyDirectory.getDirectoryHandle('public'),evaluation=await destination.getFileHandle('eval_context.bin',{create:true});
    await copyFile(await publicKeys.getFileHandle('eval_context.bin'),evaluation,check);publicFiles.push(['eval_context.bin',evaluation]);
    const files={};let index=0;
    for(const [name,file]of publicFiles) {
      const snapshot=await file.getFile(),hashWork=await work.getDirectoryHandle('hash-'+index++,{create:true});
      const hash=await pump(core,'sha256',['',String(snapshot.size)],hashWork,{bytes:fileSource(file)},check,progress);
      files[name]={bytes:String(snapshot.size),sha256:await(await hash.get('sha256').handle.getFile()).text()};
    }
    const manifest=await destination.getFileHandle('manifest.json',{create:true});check();
    await writeFile(manifest,JSON.stringify({schema:'latti-ai.browser-runner.v1',state:'committed',
      deploymentMode:'server_provisioned_runner',parameterLoadingMode:'runtime_lazy',profileId:profile.profileId,keyIdentity,
      runtimeMetrics:metadata.runtimeMetrics,files})+'\n');
    retained=true;return {directory:destination,manifest};
  }catch(error){failure=error;throw error;}finally {
    const actions=[];
    if(producer)actions.push(()=>producer.close());
    if(!retained&&destination&&options.outputDirectory)actions.push(()=>options.outputDirectory.removeEntry(active.jobId,{recursive:true}));
    if(!retained||options.outputDirectory)actions.push(()=>job.cleanup());
    else actions.push(()=>job.directory.removeEntry('work',{recursive:true}));
    await cleanupAfter(failure,actions);
  }
}

async function preflightServer(options) {
  const expected=inputNames(),provided=Object.keys(options.inputDirectories||{}).sort();
  if(JSON.stringify(expected)!==JSON.stringify(provided)) throw new Error('Input directories must match the exact task input IDs');
  const maps=new Map();let cases;
  for(const name of expected) {
    const list=await files(options.inputDirectories[name],/\.csv$/i),byCase=new Map();
    if(!list.length) throw new Error(name+': empty CSV directory');
    for(const file of list) {const id=caseId(file),key=id.normalize('NFC').toLowerCase();if(byCase.has(key)) throw new Error('Duplicate case: '+id);byCase.set(key,{id,file});}
    if(cases&&(cases.size!==byCase.size||[...cases.keys()].some(key=>byCase.get(key)?.id!==cases.get(key).id))) throw new Error('Input directories contain different case IDs');
    cases??=byCase;maps.set(name,byCase);
  }
  return {names:expected,maps,cases:[...cases.values()].map(value=>value.id).sort()};
}

async function expectedCases(batch) {
  if(!batch?.directory||!batch?.manifest) throw new Error('Server results require the input BatchResult');
  const metadata=await batch.manifest.getFile();
  if(metadata.size>1024*1024) throw new Error('METADATA_BUDGET_EXCEEDED');
  const manifest=JSON.parse(await metadata.text());
  if(manifest.state!=='committed'||manifest.profileId!==profile.profileId) throw new Error('Invalid input batch manifest');
  const cases=new Set(),file=await(await batch.directory.getFileHandle(safeName(manifest.entries))).getFile();
  const reader=file.stream().pipeThrough(new TextDecoderStream('utf-8',{fatal:true})).getReader();let partial='',bytes=0;
  try {
    for(;;) {
      check();const {value,done}=await reader.read();if(done) break;
      bytes+=value.length*2;if(bytes>64*1024*1024) throw new Error('METADATA_BUDGET_EXCEEDED');
      partial+=value;let end;
      while((end=partial.indexOf('\n'))>=0) {
        const line=partial.slice(0,end);partial=partial.slice(end+1);
        const record=JSON.parse(line);cases.add(safeName(record.case_id));
        if(cases.size>100000) throw new Error('METADATA_BUDGET_EXCEEDED');
      }
      if(partial.length>128*1024) throw new Error('Invalid case manifest line');
    }
    if(partial||!cases.size) throw new Error('Incomplete or empty case manifest');
    return cases;
  } finally {await reader.cancel().catch(()=>{});reader.releaseLock();}
}

async function batch(encrypting,options) {
  const server=options.mode==='server';
  if(server&&!encrypting) {
    requireFeature(options.outputName,false);
    if(Object.keys(task.task_output_param).length!==1) throw new Error('Server naming requires one selected task output');
  }
  const preflight=server&&encrypting?await preflightServer(options):undefined;
  const list=!server?await files(options.inputDirectory,encrypting?/\.csv$/i:/\.ct\.bin$/i):undefined;
  if(list&&!list.length) throw new Error('Empty input directory');
  if(!server) requireFeature(encrypting?options.inputName:options.outputName,encrypting);
  const job=await jobDirectory(active.jobId);let retained=false,writer,failure,count=0n;
  try {
    const output=options.outputDirectory?await options.outputDirectory.getDirectoryHandle(active.jobId,{create:true}):await job.directory.getDirectoryHandle('output',{create:true});
    const work=await job.directory.getDirectoryHandle('work',{create:true});
    const entries=await output.getFileHandle('entries.jsonl',{create:true});writer=await entries.createWritable();
    let archive;
    const record=async(case_id,converted,name=converted.name)=>{
      check();safeName(name);const file=await output.getFileHandle(name,{create:true});await copyFile(converted.file,file,check);
      const ref=await artifact(output,name,converted.role,file,profile.profileId);
      await writer.write(JSON.stringify({case_id,artifactId:ref.artifactId,role:ref.role,path:name,bytes:ref.bytes})+'\n');count++;
      return ref;
    };
    if(server&&encrypting) {
      const archiveFile=await output.getFileHandle('input.zip',{create:true}),spoolHandle=await work.getFileHandle('central.tmp',{create:true});
      async function* zipEntries() {
        for(const id of preflight.cases) for(const inputName of preflight.names) {
          active.entry=id+' / '+inputName;
          check();const source=preflight.maps.get(inputName).get(id.normalize('NFC').toLowerCase()).file;
          const converted=await transform(true,{inputName,inputFile:source},work);
          const name=preflight.names.length===1?id+'.ct':id+'__'+inputName+'.ct';
          yield {name,source:fileSource(converted.file)};
          await writer.write(JSON.stringify({case_id:id,inputName,path:name})+'\n');count++;
        }
      }
      await writeZip(archiveFile,zipEntries(),{spoolHandle,forceZip64:!!options.forceZip64,check});
      archive=await artifact(output,'input.zip','ciphertext',archiveFile,profile.profileId);
    } else if(server) {
      const cases=await expectedCases(options.inputBatch);
      const incoming=await work.getFileHandle('incoming.zip',{create:true});await copyFile(options.inputFile,incoming,check);
      const zip=await openZip(fileSource(incoming)),seen=new Set();
      // Validate the complete case set before decrypting any result.
      for await(const entry of zip.entries()) {
        active.entry=entry.name;
        if(!entry.name.endsWith('.result.bin')) throw new Error('Invalid server result entry: '+entry.name);
        const id=entry.name.slice(0,-'.result.bin'.length);safeName(id);
        if(!cases.has(id)||seen.has(id)) throw new Error('Unexpected or duplicate server result: '+id);seen.add(id);
      }
      if(seen.size!==cases.size) throw new Error('Server result archive is missing cases');
      for await(const entry of zip.entries()) {
        check();const id=entry.name.slice(0,-'.result.bin'.length),file=await work.getFileHandle('result.ct.bin',{create:true});
        await zip.readEntry(entry,file);
        await record(id,await transform(false,{outputName:options.outputName,ciphertextFile:file},work),id+'.csv');
      }
    } else {
      const names=new Set();
      for(const file of list) {
        const name=safeName(file.name.replace(encrypting?/\.csv$/i:/\.ct\.bin$/i,'')+(encrypting?'.ct.bin':'.csv')),key=name.normalize('NFC').toLowerCase();
        if(names.has(key)) throw new Error('Duplicate output name: '+name);names.add(key);
      }
      for(const file of list) {
        active.entry=file.name;
        try {await record(file.name.replace(encrypting?/\.csv$/i:/\.ct\.bin$/i,''),await transform(encrypting,{...options,[encrypting?'inputFile':'ciphertextFile']:file},work));}
        catch(error) {throw setErrorMessage(error,file.name+': '+error.message);}
      }
    }
    await writer.close();writer=null;check();
    const manifest=await commit(output,archive?[archive]:[],profile.profileId,{entries:'entries.jsonl',count:String(count)});
    retained=true;
    return {directory:output,manifest,count:String(count),...(archive?{archive}:{})};
  } catch(error) {failure=error;throw error;} finally {
    const actions=[];
    if(writer) actions.push(()=>writer.abort());
    if(!retained) {
      if(options.outputDirectory) actions.push(()=>options.outputDirectory.removeEntry(active.jobId,{recursive:true}));
      actions.push(()=>job.cleanup());
    } else if(options.outputDirectory) actions.push(()=>job.cleanup());
    else actions.push(()=>job.directory.removeEntry('work',{recursive:true}));
    await cleanupAfter(failure,actions);
  }
}

async function operate(type,options) {
  if(type==='open') {
    taskDirectory=options.clientDirectory;
    let initial=options.profile;
    if(!initial&&!taskDirectory) {
      const response=await fetch(new URL('./default-profile.json',import.meta.url));if(!response.ok) throw new Error('Default profile unavailable');initial=await response.json();
    }
    if(initial) {initialParameters=initial.parameters;core('init-keys',initialParameters,!!initial.bootstrap);}
    else {
      const [config,parameters]=await Promise.all([read(taskDirectory,'task_config.json'),read(taskDirectory,'ckks_parameter.json')]);
      initialParameters=parameters;core('init',config,parameters);task=JSON.parse(config);
    }
    profile=core('profile').data;
    const parent=options.keyDirectory||taskDirectory;
    keyDirectory=await parent.getDirectoryHandle(profile.profileId,{create:true});
    let pending;
    try {pending=JSON.parse(await read(keyDirectory,'pending.json',1024*1024));}
    catch(error){if(error.name!=='NotFoundError')throw error;}
    if(pending) {
      if(pending.profileId!==profile.profileId)throw new Error('Interrupted setup profile differs');
      let published=false;
      try {published=JSON.parse(await read(keyDirectory,'manifest.json',1024*1024)).state==='committed';}
      catch(error){if(error.name!=='NotFoundError'&&!(error instanceof SyntaxError))throw error;}
      if(!published) {
        for(const [sub,name]of [['public','eval_context.bin'],['secret','full_context.bin']]) {
          try {await(await keyDirectory.getDirectoryHandle(sub)).removeEntry(name);}
          catch(error){if(error.name!=='NotFoundError')throw error;}
        }
        try {await keyDirectory.removeEntry('manifest.json');}catch(error){if(error.name!=='NotFoundError')throw error;}
      }
      await keyDirectory.removeEntry('pending.json');
    }
    if(taskDirectory) await bindTask(taskDirectory);
    return {profileId:profile.profileId};
  }
  if(type==='bindTask') return bindTask(options.clientDirectory);
  if(type==='setup') return setup();
  if(type==='restore') {
    keyReady=false;
    const manifest=JSON.parse(await read(keyDirectory,'manifest.json',1024*1024));
    if(manifest.state!=='committed'||manifest.profile?.profileId!==profile.profileId||JSON.stringify(manifest.profile)!==JSON.stringify(profile)) throw new Error('Key manifest profile mismatch');
    const full=manifest.artifacts.find(ref=>ref.role==='private-full');
    if(full?.path!=='secret/full_context.bin') throw new Error('Invalid full context reference');
    const dir=await keyDirectory.getDirectoryHandle('secret'),file=await(await dir.getFileHandle('full_context.bin')).getFile();
    if(String(file.size)!==full.bytes||file.size>64*1024*1024) throw new Error('Full context size exceeds profile bound');
    const job=await jobDirectory(active.jobId);
    let failure;
    try {
      if(full.sha256&&await hashFile(file,job.directory,'hash-full')!==full.sha256)throw new Error('Full key context hash mismatch');
      await pump(core,'restore',['',String(file.size)],job.directory,{full:fileSource(file)},check,progress);
      keyIdentity=core('key-identity').data;
      if(manifest.keyIdentity&&manifest.keyIdentity!==keyIdentity)throw new Error('Restored key identity differs');
      const publicRef=manifest.artifacts.find(ref=>ref.role==='public-eval');
      const publicDirectory=await keyDirectory.getDirectoryHandle('public'),publicFile=await publicDirectory.getFileHandle('eval_context.bin');
      if(String((await publicFile.getFile()).size)!==publicRef?.bytes)throw new Error('Public evaluation context length differs');
      if(publicRef.sha256&&await hashFile(publicFile,job.directory,'hash-eval')!==publicRef.sha256)throw new Error('Public evaluation context hash mismatch');
    }
    catch(error) {failure=error;throw error;} finally {await cleanupAfter(failure,[()=>job.cleanup()]);}
    keyReady=true;
    return {profileId:profile.profileId,keyIdentity};
  }
  if(type==='close') {core('close');keyReady=false;task=undefined;return;}
  if(type==='provisionRunner') {
    if(!keyReady)throw new Error('Client keys must be setup or restored');
    return provisionRunner(options);
  }
  if(!keyReady||!task) throw new Error('Client keys and a bound task are required');
  if(type==='encrypt'||type==='decrypt') return single(type==='encrypt',options);
  if(type==='decryptOutputs') return decryptOutputs(options);
  if(type==='encryptBatch'||type==='decryptBatch') return batch(type==='encryptBatch',options);
  throw new Error('Unknown command');
}

const commands=new Set(['open','bindTask','setup','restore','encrypt','decrypt','decryptOutputs','provisionRunner','encryptBatch','decryptBatch','close']);
onmessage=({data})=>{
  if(!data||data.protocolVersion!==PROTOCOL_VERSION||!Number.isSafeInteger(data.requestId)||data.requestId<=0||typeof data.jobId!=='string'||typeof data.type!=='string') {
    send('error',data?.requestId||0,data?.jobId||sessionJobId,{code:'PROTOCOL_ERROR',phase:'protocol',message:'Invalid protocol v2 envelope',retryable:false});return;
  }
  if(data.type==='cancel') {
    const accepted=active?.requestId===data.targetRequestId;
    if(accepted) active.controller.abort();
    send('result',data.requestId,data.jobId,{result:{accepted}});return;
  }
  if(!commands.has(data.type)) {send('error',data.requestId,data.jobId,{code:'PROTOCOL_ERROR',phase:'protocol',message:'Unknown command',retryable:false});return;}
  if(data.type==='close') {closing=true;active?.controller.abort();}
  queue=queue.then(async()=>{
    if((closing&&data.type!=='close')||state==='closed'||state==='failed') {
      send('error',data.requestId,data.jobId,{code:'SESSION_CLOSED',phase:'session',message:'Browser client is closed',retryable:false});return;
    }
    active={requestId:data.requestId,jobId:data.jobId,start:performance.now(),controller:new AbortController()};state='busy';
    try {
      await runtimeReady;
      const result=await navigator.locks.request('latti-shared-export',{mode:'exclusive',signal:active.controller.signal},()=>operate(data.type,data));
      state=data.type==='close'?'closed':'idle';
      progress(data.type,active.progress?.bytes||'0',{},true);
      if(data.type==='setup') for(const artifact of [result.evalContext,result.fullContext]) send('artifact',data.requestId,data.jobId,{artifact});
      if(data.type==='encrypt'||data.type==='decrypt') send('artifact',data.requestId,data.jobId,{artifact:result});
      send('result',data.requestId,data.jobId,{result});
    } catch(error) {
      state=runtimeFailure?'failed':'idle';
      if((error.code==='CANCELLED'||error.name==='AbortError')&&!error.cleanupFailed) send('cancelled',data.requestId,data.jobId,{reason:'Operation stopped; uncommitted files removed'});
      else send('error',data.requestId,data.jobId,{code:error.cleanupFailed&&(error.code==='CANCELLED'||error.name==='AbortError')?'OPFS_CLEANUP_FAILED':typeof error.code==='string'?error.code:({QuotaExceededError:'OPFS_QUOTA_EXCEEDED',NotAllowedError:'PERMISSION_DENIED',NotFoundError:'FILE_NOT_FOUND'}[error.name]||'WORKER_OPERATION_FAILED'),phase:active.progress?.phase||data.type,...(active.entry?{entry:active.entry}:{}),message:error.message||String(error),retryable:false});
    } finally {active=undefined;}
  });
};
