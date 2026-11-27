// Executed inside the workflow's real Chromium page, using its current task/keys.
export async function sampleAndBindingRegressions(key) {
  const checks=[];
  const reject=async(name,action,pattern)=>{
    let error;try{await action();}catch(value){error=value;}
    if(!error||!pattern.test(error.message))throw new Error(name+' did not reject as expected: '+error?.message);
    checks.push({name,status:'passed',error:error.message});
  };
  const write=async(directory,name,text)=>{
    const file=await fileAt(directory,name,true),writer=await file.createWritable();await writer.write(text);await writer.close();return file;
  };
  const hash=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
  const metadata=JSON.parse(await(await tasks[key].manifest.getFile()).text());
  const owner=clientDirectory,contract=JSON.parse(await(await(await fileAt(owner,'input_contract.json')).getFile()).text());
  const entry=contract.inputs[0],csv=await fileAt(tasks[key].directory,entry.csv.slice(metadata.virtualRoot.length+1));
  await reject('crossing CSV substitution',()=>client.encrypt({inputName:entry.feature_id,inputFile:new File(['0'],'changed.csv')}),/content differs/);
  const clone=await opfs.getDirectoryHandle('binding-regressions',{create:true});
  const originalManifest=JSON.parse(await(await(await fileAt(owner,'task_manifest.json')).getFile()).text());
  const originals={};for(const name of [...Object.keys(originalManifest.files),'task_manifest.json'])originals[name]=await(await(await fileAt(owner,name)).getFile()).text();
  const restore=async()=>{for(const [name,text]of Object.entries(originals))await write(clone,name,text);};
  const mutate=async(name,change,rehash=true)=>{
    await restore();const document=JSON.parse(originals[name]);change(document);await write(clone,name,JSON.stringify(document));
    if(rehash){const manifest=structuredClone(originalManifest),bytes=await(await(await fileAt(clone,name)).getFile()).arrayBuffer();manifest.files[name]={bytes:bytes.byteLength,sha256:await hash(bytes)};await write(clone,'task_manifest.json',JSON.stringify(manifest));}
  };
  await mutate('input_contract.json',document=>{document.deployment_mode='server_provisioned_runner';});
  await reject('mode contract exchange',()=>client.bindTask(clone),/mode\/output contract differs/);
  await mutate('input_contract.json',document=>{document.ckks_sha256='0'.repeat(64);});
  await reject('CKKS contract substitution',()=>client.bindTask(clone),/CKKS contract hash differs/);
  await mutate('task_config.json',document=>{delete document.evaluation_keys;});
  await reject('missing evaluation key requirements',()=>client.bindTask(clone),/evaluation_keys/);
  for(const field of ['id','level','size']) {
    await mutate('task_signature.json',document=>{const output=document.online.find(arg=>arg.phase==='out');if(field==='id')output.id='unknown';else if(field==='level')output.level++;else output.size=[99999];});
    await reject('output signature '+field,()=>client.bindTask(clone),/output signature/);
  }
  await mutate('task_config.json',document=>{document.task_output_param[Object.keys(document.task_output_param)[0]].level++;},false);
  await reject('task file hash substitution',()=>client.bindTask(clone),/hash mismatch/);
  await mutate('task_config.json',document=>{delete document.task_manifest_required;},false);
  await reject('task manifest flag removal',()=>client.bindTask(clone),/hash mismatch/);
  await client.bindTask(owner);
  const split=splits.cifar10,splitMetadata=JSON.parse(await(await split.manifest.getFile()).text());
  for(const path of [splitMetadata.sourceModel,'source/img.csv']) {
    const file=await fileAt(split.directory,path),original=new Uint8Array(await(await file.getFile()).arrayBuffer());
    try {
      await write(split.directory,path,new Uint8Array([0]));
      await reject('split hash '+path,()=>modelTools.compileHighLevel({split,deploymentMode:'client_encrypted_input'}),/hash mismatch/);
    }finally{await write(split.directory,path,original);}
  }
  const sample=await(await(await fileAt(sources.cifar10.directory,'img.csv')).getFile()).text();
  const changed=sample.replace(/[-+0-9.eE]+/,value=>String(Number(value)+0.01));
  if(changed===sample)throw new Error('New sample did not change');
  await write(split.directory,'new-sample.csv',changed);
  const prefix=await modelTools.exportPrefix({split});
  const run=await modelTools.runPrefix({prefix,inputDirectory:split.directory,inputsManifest:new File([JSON.stringify({inputs:[{name:'input',csv:'new-sample.csv',shape:[1,3,32,32],axes:['N','C','H','W']}]})],'inputs.json')});
  const runMetadata=JSON.parse(await(await run.manifest.getFile()).text()),binding=JSON.parse(await(await(await fileAt(run.directory,runMetadata.entry)).getFile()).text());
  const samplePath=runMetadata.entry.slice(0,runMetadata.entry.lastIndexOf('/')+1)+binding.sample_manifest;
  const prepared=JSON.parse(await(await(await fileAt(run.directory,samplePath)).getFile()).text());
  for(const crossing of prepared.inputs) {
    const compiled=contract.inputs.find(input=>input.name===crossing.name);
    const file=await fileAt(run.directory,samplePath.slice(0,samplePath.lastIndexOf('/')+1)+crossing.csv);
    if(crossing.sha256!==compiled.sha256)await reject('unbound new sample '+crossing.name,()=>client.encrypt({inputName:compiled.feature_id,inputFile:file}),/content differs/);
    const encrypted=await client.encrypt({inputName:compiled.feature_id,inputFile:file,run});
    if((await encrypted.file.getFile()).size===0)throw new Error('New sample encryption is empty');
  }
  checks.push({name:'runPrefix new sample and both crossing encryptions',status:'passed',binding});
  return checks;
}

export async function outputFramingRegressions({key,model,mode}) {
  const file=new File([await(await fetch('/artifact/'+model+'/'+mode+'/output.ct')).arrayBuffer()],'output.ct');
  const checks=[];
  const reject=async(name,invalid)=>{let error;try{await client.decryptOutputs({task:tasks[key],outputFile:invalid});}catch(value){error=value;}if(!error)throw new Error('Accepted '+name);checks.push({name,error:error.message,status:'passed'});};
  await reject('truncated output',file.slice(0,file.size-1));
  await reject('trailing output',new File([file,new Uint8Array([0])],'tail.ct'));
  const u64=value=>{const bytes=new Uint8Array(8);new DataView(bytes.buffer).setBigUint64(0,BigInt(value),true);return bytes;};
  const string=value=>{const bytes=new TextEncoder().encode(value);return [u64(bytes.length),bytes];};
  await reject('unknown output ID',new File([...string('latti-ai.runner-output.v1'),u64(1),...string('unknown'),u64(1),new Uint8Array([1])],'unknown.ct'));
  await reject('invalid output length',new File([...string('latti-ai.runner-output.v1'),u64(1),...string('output'),u64(2n**63n)],'length.ct'));
  return checks;
}

// This is a routing/cryptographic echo fixture, not a native model execution.
export async function newSampleOutputRoutes() {
  const get=async(directory,name)=>{const parts=name.split('/');for(const part of parts.slice(0,-1))directory=await directory.getDirectoryHandle(part);return directory.getFileHandle(parts.at(-1));};
  const digest=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
  await write(split.directory,'new.csv',Array.from({length:512},(_,i)=>(i+1)/512).join('\n'));
  const tools=await api.openBrowserModelTools();let run;
  try{const prefix=await tools.exportPrefix({split});run=await tools.runPrefix({prefix,inputDirectory:split.directory,inputsManifest:new File([JSON.stringify({inputs:[{name:'input',csv:'new.csv',shape:[1,512],axes:['N','F']}]})],'inputs.json')});}finally{await tools.close();}
  const runMetadata=JSON.parse(await(await run.manifest.getFile()).text()),binding=JSON.parse(await(await(await get(run.directory,runMetadata.entry)).getFile()).text());
  const samplePath=runMetadata.entry.slice(0,runMetadata.entry.lastIndexOf('/')+1)+binding.sample_manifest;
  const sample=JSON.parse(await(await(await get(run.directory,samplePath)).getFile()).text());
  const metadata=JSON.parse(await(await split.manifest.getFile()).text());
  const profile=await(await fetch('/fixtures/echo-parameters.json')).text();
  const feature={dim:0,channel:512,pack_num:512,skip:8,level:1,ckks_parameter_id:'p'};
  const config={evaluation_keys:[],use_btp:false,pack_style:'ordinary',task_input_param:{input:feature},task_output_param:{output:feature}};
  const role=await opfs.getDirectoryHandle('routing-echo-role',{create:true});
  await write(role,'task_config.json',JSON.stringify(config));await write(role,'ckks_parameter.json',profile);
  const contract={manifest:metadata.virtualRoot+'/'+metadata.entry,manifest_sha256:await digest(await(await(await get(split.directory,metadata.entry)).getFile()).arrayBuffer()),outputs:config.task_output_param,output_map:{output:'output'},deployment_mode:'client_encrypted_input'};
  const file=await write(split.directory,'echo/input_contract.json',JSON.stringify(contract)),bytes=await(await file.getFile()).arrayBuffer();
  const taskManifest={...metadata,kind:'task',entry:'echo',files:{...metadata.files,'echo/input_contract.json':{bytes:bytes.byteLength,sha256:await digest(bytes)}}};
  const task={directory:split.directory,manifest:await write(split.directory,'echo-task-manifest.json',JSON.stringify(taskManifest))};
  const client=await api.openBrowserClient({clientDirectory:role,keyDirectory:await opfs.getDirectoryHandle('routing-echo-keys',{create:true})});
  try {
    await client.setup();
    const expectedSuffix=Array.from({length:512},(_,i)=>((i+1)/512+1)**2);
    const source=new File([expectedSuffix.join(',')],'suffix.csv');
    const encrypted=await client.encrypt({inputName:'input',inputFile:source}),wire=await encrypted.file.getFile();
    const u64=value=>{const bytes=new Uint8Array(8);new DataView(bytes.buffer).setBigUint64(0,BigInt(value),true);return bytes;};
    const string=value=>{const bytes=new TextEncoder().encode(value);return [u64(bytes.length),bytes];};
    const bundle=new File([...string('latti-ai.runner-output.v1'),u64(1),...string('output'),u64(wire.size),wire],'echo.ct');
    const result=await client.decryptOutputs({task,outputFile:bundle,run});
    const outputs=JSON.parse(await(await result.manifest.getFile()).text()).outputs;let maxError=0;
    for(const output of outputs) {
      const actual=(await(await(await get(result.directory,output.file)).getFile()).text()).trim().split(/[\s,]+/).map(Number);
      const route=sample.output_routes.find(route=>route.name===output.name);
      const expected=route.route==='suffix'?expectedSuffix:(await(await(await get(run.directory,samplePath.slice(0,samplePath.lastIndexOf('/')+1)+route.plaintext.csv)).getFile()).text()).trim().split(/[\s,]+/).map(Number);
      if(actual.length!==expected.length)throw new Error('Route count differs');
      for(let i=0;i<actual.length;i++)maxError=Math.max(maxError,Math.abs(actual[i]-expected[i]));
    }
    if(outputs.length!==2||!(maxError<1e-5))throw new Error('New sample routing differs: '+maxError);
    return {fixture:'cryptographic echo; native inference is tested by model-workflow',outputs:outputs.map(output=>({name:output.name,route:output.route})),maxError};
  }finally{await client.close();}
}
