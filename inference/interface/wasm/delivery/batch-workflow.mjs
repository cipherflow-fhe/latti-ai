// Shared Chromium batch flow. Only runNative crosses into the native GPU service.
export const fileAt=async(directory,name,create=false)=>{
  if(typeof name!=='string'||!name||/[\\\0:]/.test(name)||name.split('/').some(p=>!p||p==='.'||p==='..'))throw new Error('Invalid artifact path');
  const parts=name.split('/');
  for(const part of parts.slice(0,-1))directory=await directory.getDirectoryHandle(part,{create});
  return directory.getFileHandle(parts.at(-1),{create});
};
export const directoryAt=async(directory,name)=>{
  for(const part of name.split('/')){if(!part||part==='.'||part==='..')throw new Error('Invalid directory path');directory=await directory.getDirectoryHandle(part);}
  return directory;
};
export const readJSON=async file=>JSON.parse(await(await file.getFile()).text());
const copy=async(source,target)=>{await(await source.getFile()).stream().pipeTo(await target.createWritable());};
const writeJSON=async(directory,name,value)=>{
  const handle=await directory.getFileHandle(name,{create:true}),writer=await handle.createWritable();
  try{await writer.write(JSON.stringify(value,null,2)+'\n');await writer.close();}catch(error){await writer.abort().catch(()=>{});throw error;}
  return handle;
};

/** Model sources stay with the model owner. The provider uses prefix + public
 * task metadata + public context, and runs samples serially. Processing modules
 * must be supplied by the application's trusted operator.
 */
export async function runParameterBatchWorkflow({modelDirectory,modelPath,rawDirectory,labelFile,
  preprocessingURL,postprocessingURL,compileConfig,keyDirectory,keyProfile,generateKeys=false,
  runtimeURL=new URL('./runtime/index.js',import.meta.url).href,
  onProgress=()=>{},onStage=async()=>{},onArtifact=async()=>{},runNative}) {
  const {openBrowserModelTools,openBrowserClient}=await import(runtimeURL);
  const {preprocess_batch}=await import(preprocessingURL),{postprocess_batch}=await import(postprocessingURL);
  const root=await(await navigator.storage.getDirectory()).getDirectoryHandle('batch-'+crypto.randomUUID(),{create:true});
  const stages=[];
  const stage=async(name,action)=>{
    onProgress({type:'stage',name});await onStage(name);
    const start=performance.now(),value=await action();
    stages.push({name,durationMs:performance.now()-start});return value;
  };
  const models=await openBrowserModelTools({onProgress});let owner,provider;
  try {
    const prepared=await root.getDirectoryHandle('prepared',{create:true});
    const preprocessing=await stage('preprocess',()=>preprocess_batch({inputDirectory:rawDirectory,outputDirectory:prepared}));
    const batch=await readJSON(preprocessing.inputsManifest);
    if(!batch.samples?.length)throw new Error('No raw samples');
    await onArtifact('prepared',{directory:prepared,manifest:preprocessing.inputsManifest});
    const source=await root.getDirectoryHandle('model-source',{create:true});
    await copy(await fileAt(modelDirectory,modelPath),await fileAt(source,modelPath,true));
    // The model's first sample selects shapes and split boundaries. All samples
    // subsequently run through the exported prefix with their own bindings.
    for(const entry of batch.samples[0].inputs)await copy(await fileAt(prepared,entry.csv),await fileAt(source,entry.csv,true));
    const inputsManifest=await writeJSON(source,'inputs.json',{inputs:batch.samples[0].inputs});
    const split=await stage('onnx-split',()=>models.prepareSplit({source:{directory:source,modelPath,inputsManifest},ratio:'0.9'}));
    const prefix=await stage('export-prefix',()=>models.exportPrefix({split}));
    await onArtifact('split',split);await onArtifact('prefix',prefix);
    let task=await stage('high-level',()=>models.compileHighLevel({split,deploymentMode:'server_provisioned_runner',compileConfig}));
    task=await stage('low-level',()=>models.generateLowLevel({task}));
    await onArtifact('task',task);
    const metadata=await readJSON(task.manifest),provisioner=await directoryAt(task.directory,metadata.entry+'/provisioner');
    owner=await openBrowserClient({clientDirectory:provisioner,keyDirectory,profile:keyProfile,onProgress});
    const keys=await stage(generateKeys?'key-generation':'key-restore',()=>generateKeys?owner.setup():owner.restore());
    const publicContext=await stage('export-public-context',()=>owner.exportPublicContext());
    await onArtifact('keys',{keys,keyDirectory,publicContext});
    const bundle=await stage('provision-parameters',()=>owner.provisionRunner({task,parameterDirectory:provisioner}));
    await onArtifact('runner',bundle);
    // Give the data provider only the independent prefix and public metadata.
    const dataTask=await root.getDirectoryHandle('data-task',{create:true});
    for(const name of ['task_config.json','ckks_parameter.json','input_contract.json','task_manifest.json','task_signature.json','key_requirements.json'])await copy(await provisioner.getFileHandle(name),await dataTask.getFileHandle(name,{create:true}));
    const publicSpec=await readJSON(await prefix.directory.getFileHandle('prefix.json'));
    const prefixInputs={samples:batch.samples.map(sample=>({...sample,inputs:sample.inputs.filter(input=>publicSpec.inputs.some(v=>v.name===input.name))}))};
    const prefixRun=await stage('prefix-batch',()=>models.runPrefix({prefix,inputDirectory:prepared,inputsManifest:new File([JSON.stringify(prefixInputs)],'inputs.json')}));
    await onArtifact('prefix-run',prefixRun);
    const runMeta=await readJSON(prefixRun.manifest),binding=await readJSON(await fileAt(prefixRun.directory,runMeta.entry));
    const runBase=runMeta.entry.slice(0,runMeta.entry.lastIndexOf('/')+1),inputDirectories={},crossings=[];
    for(const record of binding.samples) {
      const path=runBase+record.manifest,sample=await readJSON(await fileAt(prefixRun.directory,path)),base=path.slice(0,path.lastIndexOf('/')+1);
      const hashes={};
      for(const input of sample.inputs) {
        inputDirectories[input.feature_id]??=await root.getDirectoryHandle('crossing-'+input.feature_id,{create:true});
        await copy(await fileAt(prefixRun.directory,base+input.csv),await inputDirectories[input.feature_id].getFileHandle(record.case_id+'.csv',{create:true}));
        hashes[input.feature_id]=input.sha256;
      }
      crossings.push({case_id:record.case_id,sha256:hashes});
    }
    const providerKeys=await root.getDirectoryHandle('provider-public-only',{create:true});
    provider=await openBrowserClient({clientDirectory:dataTask,keyDirectory:providerKeys,profile:keyProfile,onProgress});
    const imported=await stage('import-public-context',()=>provider.importPublicContext({inputFile:publicContext.file}));
    if(imported.keyIdentity!==keys.keyIdentity)throw new Error('Provider public key identity differs');
    const inputBatch=await stage('encrypt-input-batch',()=>provider.encryptBatch({mode:'server',inputDirectories,run:prefixRun}));
    await onArtifact('input-batch',inputBatch);
    await provider.close();provider=undefined;
    if(typeof runNative!=='function')throw new Error('A native GPU runner or bound replay ciphertext archive is required');
    const resultArchive=await stage('native-runner',()=>runNative({bundle,inputBatch,crossings,keys}));
    const config=await readJSON(await provisioner.getFileHandle('task_config.json'));
    const outputNames=Object.keys(config.task_output_param);
    if(outputNames.length!==1)throw new Error('Batch archive requires one selected output feature');
    const results=await stage('decrypt-batch',()=>owner.decryptBatch({mode:'server',outputName:outputNames[0],inputFile:resultArchive,inputBatch}));
    await onArtifact('results',results);
    const post=await root.getDirectoryHandle('postprocessed',{create:true});
    const statistics=await stage('postprocess',()=>postprocess_batch({inputDirectory:results.directory,labelFile,outputDirectory:post}));
    return {root,split,prefix,task,bundle,prefixRun,inputBatch,results,keys,publicContext,crossings,statistics,stages,count:batch.samples.length};
  }finally{await provider?.close();await owner?.close();await models.close();}
}
