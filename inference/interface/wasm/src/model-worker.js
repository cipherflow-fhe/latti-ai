import {BUILD_INFO} from './build-info.js';
import {fileOf,writeFile} from './io.js';
import {mountPythonWorkspace} from './python-workspace.js';

const resources=new URL('./model/',import.meta.url);
const CHUNK=1024*1024;
let python,ort,lock,operation,phase,workspace;
let runtime='pyodide',largestTensorBytes=0;
const memories=[];
const NativeMemory=WebAssembly.Memory;
WebAssembly.Memory=new Proxy(NativeMemory,{construct(target,args){
  const memory=Reflect.construct(target,args);memories.push({runtime,memory});return memory;
}});
const currentMetrics=()=>({pyodideMemoryBytes:python?python._module.HEAPU8.buffer.byteLength:0,
  ortMemoryBytes:Math.max(0,...memories.filter(entry=>entry.runtime==='ort').map(entry=>entry.memory.buffer.byteLength)),largestTensorBytes});
const runtimeMetrics=()=>{
  const current=currentMetrics(),stages=operation.stageMetrics||[];
  return {...Object.fromEntries(Object.keys(current).map(key=>[key,Math.max(current[key],...stages.map(s=>s[key]||0))])),
    stages,workerStages:stages.length+1};
};
const checkpoint=async stage=>{
  progress('checkpoint-'+stage);
  const metrics={phase:stage,...currentMetrics()};
  await workspace.checkpoint();
  return {resume:true,metrics};
};
const digest=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
const relative=name=>{
  if(typeof name!=='string'||!name||/[\\\0:]/.test(name)||name.split('/').some(part=>!part||part==='.'||part==='..'))
    throw new Error('Invalid relative model path');
  return name;
};
async function fileAt(directory,name,create=false) {
  const parts=relative(name).split('/');
  for(const part of parts.slice(0,-1))directory=await directory.getDirectoryHandle(part,{create});
  return directory.getFileHandle(parts.at(-1),{create});
}
function progress(name,extra={}) {
  phase=name;
  postMessage({protocolVersion:BUILD_INFO.protocolVersion,requestId:operation.requestId,jobId:operation.jobId,
    type:'progress',phase,bytes:'0',elapsedMs:performance.now()-operation.started,
    wasmBytes:python?python._module.HEAPU8.buffer.byteLength:0,heapBytes:0,chunkBytes:CHUNK,inFlightChunks:1,final:false,...extra});
}
async function fetchVerified(name,metadata) {
  const response=await fetch(new URL(name,resources));
  if(!response.ok)throw new Error(`Missing model resource: ${name}`);
  const bytes=new Uint8Array(await response.arrayBuffer());
  if(bytes.byteLength!==metadata.bytes||await digest(bytes)!==metadata.sha256)throw new Error(`Resource hash mismatch: ${name}`);
  return bytes;
}
async function initialize() {
  if(python)return;
  progress('dependencies');
  lock=await(await fetch(new URL('resources.json',resources))).json();
  const {loadPyodide}=await import(new URL('pyodide/pyodide.mjs',resources));
  python=await loadPyodide({indexURL:new URL('pyodide/',resources).href,
    lockFileURL:new URL('pyodide/python-lock.json',resources).href});
  await python.loadPackage(Object.keys(lock.pythonLock.packages));
  const sources=await(await fetch(new URL('sources.json',resources))).json();
  for(const [name,metadata]of Object.entries(sources)) {
    const bytes=await fetchVerified('python/'+relative(name),metadata),destination='/latti/repo/'+name;
    python.FS.mkdirTree(destination.slice(0,destination.lastIndexOf('/')));python.FS.writeFile(destination,bytes);
  }
  await python.runPythonAsync("import sys, json\nfrom pathlib import Path\nsys.path.insert(0, '/latti/repo')\nfrom inference.interface.wasm.python import browser_pipeline as pipeline\n");
  ort=await import(new URL('ort/ort.wasm.min.mjs',resources));
  ort.env.wasm.wasmPaths=new URL('ort/',resources).href;ort.env.wasm.numThreads=1;
}
async function copyIn(file,path) {
  file=await fileOf(file);
  python.FS.mkdirTree(path.slice(0,path.lastIndexOf('/')));
  const stream=python.FS.open(path,'w');
  try {
    for(let offset=0;offset<file.size;offset+=CHUNK) {
      const bytes=new Uint8Array(await file.slice(offset,offset+CHUNK).arrayBuffer());
      python.FS.write(stream,bytes,0,bytes.byteLength,offset);
    }
  }finally{python.FS.close(stream);}
}
async function copyOut(path,handle) {
  const size=python.FS.stat(path).size,stream=python.FS.open(path,'r'),writer=await handle.createWritable();
  const buffer=new Uint8Array(CHUNK);
  try {
    for(let offset=0;offset<size;offset+=CHUNK) {
      const count=python.FS.read(stream,buffer,0,Math.min(CHUNK,size-offset),offset);
      if(count!==Math.min(CHUNK,size-offset))throw new Error('Short model artifact read');
      await writer.write(buffer.subarray(0,count));
    }
    await writer.close();
  }catch(error){await writer.abort().catch(()=>{});throw error;}
  finally{python.FS.close(stream);}
  return size;
}
async function withGlobals(values,code) {
  for(const [name,value]of Object.entries(values))python.globals.set(name,value);
  try {return await python.runPythonAsync(code);}
  finally{for(const name of Object.keys(values))python.globals.delete(name);}
}
const hashIn=path=>withGlobals({hash_path:path},"from onnx_split_contract import file_hash\nfile_hash(hash_path)");
async function importSource(source,root) {
  if(source?.directory?.kind!=='directory')throw new Error('ModelSources requires a source directory');
  const model=relative(source.modelPath),directory=root+'/source';
  let sourceBytes=0;
  const copySource=async name=>{
    const file=await(await fileAt(source.directory,name)).getFile();sourceBytes+=file.size;
    if(sourceBytes>512*1024*1024)throw new Error('Source files exceed the 512 MiB browser profile');
    await copyIn(file,directory+'/'+relative(name));
  };
  await copySource(model);
  if((await fileOf(source.inputsManifest)).size>1024*1024)throw new Error('Input manifest exceeds metadata budget');
  await copyIn(source.inputsManifest,directory+'/inputs.json');
  const references=JSON.parse(await withGlobals({source_root:directory,model_name:model,input_manifest:directory+'/inputs.json'},
    'pipeline.source_references(source_root, model_name, input_manifest)'));
  for(const name of references)await copySource(name);
  return {directory,model:directory+'/'+model,manifest:directory+'/inputs.json',modelPath:model};
}
async function importArtifact(artifact,kind) {
  if(artifact?.directory?.kind!=='directory'||artifact?.manifest?.kind!=='file')throw new Error('ArtifactDirectory required');
  const manifestFile=await artifact.manifest.getFile();
  if(manifestFile.size>1024*1024)throw new Error('Artifact manifest exceeds metadata budget');
  const text=await manifestFile.text(),manifest=JSON.parse(text);
  if(manifest.schema!=='latti-ai.browser-model-artifact.v1'||manifest.kind!==kind||
    !/^\/latti\/jobs\/[0-9a-f-]+$/.test(manifest.virtualRoot))throw new Error('Invalid model artifact manifest');
  relative(manifest.entry);
  if(!await artifact.manifest.isSameEntry(await artifact.directory.getFileHandle('artifact-manifest.json')))
    throw new Error('Artifact manifest belongs to a different directory');
  let total=0;
  for(const [name,metadata]of Object.entries(manifest.files)) {
    const file=await(await fileAt(artifact.directory,name)).getFile();
    const streamedGraph=/\/(?:compiled_)?mega_ag\.json$|\/model_parameters\.h5$/.test(name);
    if(!streamedGraph)total+=file.size;
    if((!streamedGraph&&file.size>512*1024*1024)||total>2*1024*1024*1024)throw new Error('Model artifact exceeds the browser memory profile');
    if(file.size!==metadata.bytes)throw new Error(`Artifact hash mismatch: ${name}`);
    await copyIn(file,manifest.virtualRoot+'/'+name);
    if(await hashIn(manifest.virtualRoot+'/'+name)!==metadata.sha256)throw new Error(`Artifact hash mismatch: ${name}`);
  }
  return manifest;
}
const types={float32:['float32',Float32Array],float64:['float64',Float64Array],
  int32:['int32',Int32Array],int64:['int64',BigInt64Array],uint32:['uint32',Uint32Array],
  uint64:['uint64',BigUint64Array],int8:['int8',Int8Array],uint8:['uint8',Uint8Array],bool:['bool',Uint8Array]};
async function ortRun(text) {
  const request=JSON.parse(text);progress('ort-'+request.stage);
  largestTensorBytes=Math.max(largestTensorBytes,request.maximumTensorBytes||0);
  runtime='ort';
  let session,result;const feeds={};
  try {
    let model=python.FS.readFile(request.model);
    const externalData=request.externalData.map(entry=>({path:entry.path,data:python.FS.readFile(entry.file)}));
    // Extended MatmulTransposeFusion emits a float-only FusedMatMul kernel in
    // this ORT Web build. Keep standard float64 kernels for transformer feeds.
    session=await ort.InferenceSession.create(model,{executionProviders:['wasm'],externalData,
      ...(request.feeds.some(feed=>feed.dtype==='float64')?{graphOptimizationLevel:'basic'}:{})});
    // Sessions own parsed models. Drop JS initialization buffers before feeds.
    model=null;externalData.length=0;
    for(const entry of request.feeds) {
      const type=types[entry.dtype];if(!type)throw new Error(`Unsupported ORT input dtype: ${entry.dtype}`);
      const bytes=python.FS.readFile(entry.file);
      largestTensorBytes=Math.max(largestTensorBytes,bytes.byteLength);
      feeds[entry.name]=new ort.Tensor(type[0],new type[1](bytes.buffer,bytes.byteOffset,bytes.byteLength/type[1].BYTES_PER_ELEMENT),entry.shape);
    }
    result=await session.run(feeds,request.outputs||undefined);const outputs=[];
    for(const [index,[name,tensor]]of Object.entries(result).entries()) {
      const file=`output-${index}.bin`;
      largestTensorBytes=Math.max(largestTensorBytes,tensor.data.byteLength);
      python.FS.writeFile(request.directory+'/'+file,new Uint8Array(tensor.data.buffer,tensor.data.byteOffset,tensor.data.byteLength));
      outputs.push({name,file,shape:tensor.dims,dtype:tensor.type==='bool'?'bool':tensor.type});
      tensor.dispose();delete result[name];
    }
    return JSON.stringify({outputs});
  }finally{
    for(const tensor of Object.values(result||{}))tensor.dispose();
    for(const tensor of Object.values(feeds))tensor.dispose();
    await session?.release();runtime='pyodide';
  }
}
async function exportArtifact(root,kind,entry,extra={},directory=root) {
  progress('persist');
  workspace.flush();
  const names=JSON.parse(await withGlobals({artifact_root:root,export_directory:directory},`
json.dumps(sorted(str(p.relative_to(artifact_root)) for p in Path(export_directory).rglob('*') if p.is_file()))
`));
  const files={};
  for(const name of names) {
    const path=root+'/'+name,file=await fileAt(operation.outputDirectory,name,true);
    const bytes=await copyOut(path,file);files[name]={bytes,sha256:await hashIn(path)};
  }
  const manifest={schema:'latti-ai.browser-model-artifact.v1',build:BUILD_INFO.build,kind,
    virtualRoot:root,entry,files,runtimeMetrics:runtimeMetrics(),...extra};
  const file=await operation.outputDirectory.getFileHandle('artifact-manifest.json',{create:true});
  await writeFile(file,JSON.stringify(manifest,null,2)+'\n');
  return {directory:operation.outputDirectory,manifest:file};
}
async function execute(message) {
  operation={...message,started:performance.now()};
  await initialize();
  workspace ||= await mountPythonWorkspace(python.FS,message.jobId);
  if(message.type==='parameterValue') {
    await withGlobals({argument:message.argument,flat_index:message.flatIndex},
      "parameter_materializer.values(argument, flat_index).tofile('/tmp/parameter-values.bin')");
    return {bytes:python.FS.readFile('/tmp/parameter-values.bin'),runtimeMetrics:currentMetrics()};
  }
  if(message.type==='parameterPlan') {
    const manifest=await importArtifact(message.task,'task');
    if(manifest.deploymentMode!=='server_provisioned_runner')throw new Error('Parameter deployment mode required');
    const task=manifest.virtualRoot+'/'+manifest.entry;
    if(message.parameterDirectory) {
      const source=await(await message.parameterDirectory.getFileHandle('model_parameters.h5')).getFile();
      const expected=manifest.files[manifest.entry+'/provisioner/model_parameters.h5'];
      if(!expected||source.size!==expected.bytes)throw new Error('Parameter H5 content differs from compiled task');
      await copyIn(source,task+'/provisioner/model_parameters.h5');
      if(await hashIn(task+'/provisioner/model_parameters.h5')!==expected.sha256)throw new Error('Parameter H5 content differs from compiled task');
    }
    const plan=await withGlobals({task_directory:task},`
from inference.model_generator.parameter_materialization import ParameterMaterializer
parameter_materializer = ParameterMaterializer(task_directory)
json.dumps(parameter_materializer.plan())
`);
    const publicGraph=await withGlobals({task_directory:task},`
graph = json.loads(Path(task_directory, 'runner/nn_layers_ct_0.json').read_text())
for layer in graph['layer'].values():
    for key in ('weight_path', 'bias_path', 'absorb_path'):
        layer.pop(key, None)
json.dumps(graph)
`);
    return {plan,runtimeMetrics:runtimeMetrics(),config:python.FS.readFile(task+'/provisioner/task_config.json',{encoding:'utf8'}),
      parameters:python.FS.readFile(task+'/provisioner/ckks_parameter.json',{encoding:'utf8'}),
      files:Object.fromEntries(await Promise.all(['task_config.json','ckks_parameter.json','mega_ag.json','compiled_mega_ag.json',
        'fhe_parameter.json','task_signature.json','key_requirements.json'].map(async name=>[name,await fileAt(message.task.directory,manifest.entry+'/runner/'+name)]))),
      publicGraph};
  }
  let root='/latti/jobs/'+message.jobId;python.FS.mkdirTree(root);
  if(message.type==='diagnoseORT') {
    const source=await importSource(message.source,root);
    await withGlobals({source_root:source.directory,model_path:source.modelPath,inputs_manifest:source.manifest,
      diagnostic_output:root+'/diagnostic',ort_run:ortRun},
      'await pipeline.diagnose_ort(source_root, model_path, inputs_manifest, diagnostic_output, ort_run)');
    return exportArtifact(root,'diagnostic','diagnostic/observations.npz',{sourceModel:source.model.slice(root.length+1)});
  }
  if(message.type==='prepareSplit') {
    if(!message.resume) {
      progress('source');const source=await importSource(message.source,root);
      progress('externalize');
      await withGlobals({state_root:root,source_root:source.directory,model_name:source.modelPath,input_manifest:source.manifest,
        split_output:root+'/split',ratio:message.ratio},
        'pipeline.begin_split(state_root, source_root, model_name, input_manifest, split_output, ratio)');
      return checkpoint('externalize');
    }
    progress('split-resume');
    const step=JSON.parse(await withGlobals({state_root:root,ort_run:ortRun},'await pipeline.split_step(state_root, ort_run)'));
    if(!step.complete)return checkpoint(step.phase);
    const sourceModel=JSON.parse(python.FS.readFile(root+'/split/split_manifest.json',{encoding:'utf8'})).source.path.slice(root.length+1);
    return exportArtifact(root,'split','split/split_manifest.json',{sourceModel});
  }
  if(message.type==='compileHighLevel') {
    progress('source');let source,splitManifest;
    if(Boolean(message.source)===Boolean(message.split))throw new Error('Choose exactly one of source or split');
    if(message.source) {
      const original=await importSource(message.source,root);
      source=await withGlobals({source_directory:original.directory,model_path:original.modelPath,
        inputs_manifest:original.manifest,destination:root+'/compile-source/model.onnx'},
      'pipeline.prepare_compile_source(source_directory, model_path, inputs_manifest, destination)');
    }
    else {
      const manifest=await importArtifact(message.split,'split');
      root=manifest.virtualRoot;
      splitManifest=manifest.virtualRoot+'/'+manifest.entry;
      source=manifest.virtualRoot+'/split/suffix.onnx';
    }
    let config='';
    if(message.compileConfig){
      if((await fileOf(message.compileConfig)).size>1024*1024)throw new Error('Compile config exceeds metadata budget');
      config=root+'/compile-config.json';await copyIn(message.compileConfig,config);
    }
    progress('compile-high-level');
    const task=await withGlobals({model_path:source,compile_output:root+'/compile/'+message.jobId,mode:message.deploymentMode,
      config_path:config,split_manifest:splitManifest||null,profile_memory:message.profileMemory===true,
      compile_progress:text=>{const {phase:step,...record}=JSON.parse(text);progress('high-'+step,record);}},
    'pipeline.compile_high_level(model_path, compile_output, mode, config_path, split_manifest, compile_progress, profile_memory)');
    return exportArtifact(root,'task',task.slice(root.length+1),{deploymentMode:message.deploymentMode});
  }
  if(message.type==='exportPrefix') {
    const manifest=await importArtifact(message.split,'split');
    progress('export-prefix');
    await withGlobals({split_manifest:manifest.virtualRoot+'/'+manifest.entry,prefix_output:root},
      'pipeline.export_prefix(split_manifest, prefix_output)');
    return exportArtifact(root,'prefix','prefix.json');
  }
  if(message.type==='runPrefix') {
    if(message.split||!message.prefix)throw new Error('runPrefix requires an independent prefix artifact, not a split');
    const manifest=await importArtifact(message.prefix,'prefix');
    const inputDirectory=root+'/inputs';
    if((await fileOf(message.inputsManifest)).size>1024*1024)throw new Error('Run input manifest exceeds metadata budget');
    await copyIn(message.inputsManifest,inputDirectory+'/inputs.json');
    const inputs=JSON.parse(python.FS.readFile(inputDirectory+'/inputs.json',{encoding:'utf8'}));
    const samples=inputs.samples||[{inputs:inputs.inputs}];
    if(!Array.isArray(samples)||!samples.length)throw new Error('Prefix input manifest is empty');
    const copied=new Set();
    for(const sample of samples) {
      if(!Array.isArray(sample.inputs)||!sample.inputs.length)throw new Error('Invalid prefix input manifest');
      let sampleBytes=0;
      for(const entry of sample.inputs) {
        if(!entry||typeof entry.csv!=='string')throw new Error('Invalid prefix input manifest entry');
        const file=await(await fileAt(message.inputDirectory,relative(entry.csv))).getFile();sampleBytes+=file.size;
        if(sampleBytes>128*1024*1024)throw new Error('Sample files exceed the validated 128 MiB browser profile');
        if(copied.has(entry.csv))continue;
        copied.add(entry.csv);
        await copyIn(file,inputDirectory+'/'+relative(entry.csv));
      }
    }
    progress('run-prefix');
    await withGlobals({prefix_manifest:manifest.virtualRoot+'/'+manifest.entry,inputs_directory:inputDirectory,
      inputs_manifest:inputDirectory+'/inputs.json',run_output:root+'/run',ort_run:ortRun},
      'await pipeline.run_exported_prefix(prefix_manifest, inputs_directory, inputs_manifest, run_output, ort_run)');
    return exportArtifact(root,'run','run/run_manifest.json',{},root+'/run');
  }
  if(message.type==='generateLowLevel') {
    const manifest=await importArtifact(message.task,'task');
    progress('compile-low-level');
    const directory=await withGlobals({task_directory:manifest.virtualRoot+'/'+manifest.entry,mode:manifest.deploymentMode},
      'pipeline.generate_low_level(task_directory, mode)');
    return exportArtifact(manifest.virtualRoot,'task',manifest.entry,{deploymentMode:manifest.deploymentMode,lowLevel:directory.slice(manifest.virtualRoot.length+1)});
  }
  throw new Error('Unsupported model operation: '+message.type);
}
self.onmessage=async({data})=>{
  if(data.protocolVersion!==BUILD_INFO.protocolVersion)return;
  try {const result=await execute(data);postMessage({type:result?.resume?'checkpoint':'result',protocolVersion:BUILD_INFO.protocolVersion,requestId:data.requestId,jobId:data.jobId,result},result instanceof Uint8Array?[result.buffer]:[]);}
  catch(error){postMessage({type:'error',protocolVersion:BUILD_INFO.protocolVersion,requestId:data.requestId,jobId:data.jobId,
    message:error.message||String(error),code:'MODEL_FAILED',phase});}
};
