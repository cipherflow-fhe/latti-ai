// Retina integration fixture: execute the real browser workflow, then replay
// a native result for the exact sample and the bundled disposable test keys.
const relative=name=>{
  if(typeof name!=='string'||!name||/[\\\0:]/.test(name)||name.split('/').some(p=>!p||p==='.'||p==='..'))throw new Error('非法交付文件路径');
  return name;
};
async function fileAt(directory,name) {
  const parts=relative(name).split('/');
  for(const part of parts.slice(0,-1))directory=await directory.getDirectoryHandle(part);
  return directory.getFileHandle(parts.at(-1));
}
async function directoryAt(directory,name) {
  for(const part of relative(name).split('/'))directory=await directory.getDirectoryHandle(part);
  return directory;
}
const json=async file=>JSON.parse(await(await file.getFile()).text());
const canonical=value=>JSON.stringify(value,(_,v)=>v&&typeof v==='object'&&!Array.isArray(v)?Object.fromEntries(Object.entries(v).sort(([a],[b])=>a.localeCompare(b))):v);
const digest=async file=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',await file.arrayBuffer())),b=>b.toString(16).padStart(2,'0')).join('');

/** Public APIs only; keyDirectory is the parent of profile-ID directories. */
export async function runRetinaParameterWorkflow({sourceDirectory,compileConfig,keyDirectory,resultFile,
  runtimeURL=new URL('./runtime/index.js',import.meta.url).href,onProgress=()=>{},beforeDecrypt=async()=>{},dataProvider}) {
  const {openBrowserModelTools,openBrowserClient}=await import(runtimeURL);
  const stages=[],models=await openBrowserModelTools({onProgress:event=>onProgress({type:'runtime',...event})});
  let client;
  const stage=async(name,action)=>{
    onProgress({type:'stage',name});const start=performance.now();
    const value=await action();stages.push({name,durationMs:performance.now()-start});return value;
  };
  try {
    const inputsManifest=await sourceDirectory.getFileHandle('inputs.json');
    const split=await stage('ONNX split（ratio=0.9）',()=>models.prepareSplit({source:{directory:sourceDirectory,
      modelPath:'model/epoch041.onnx',inputsManifest},ratio:'0.9'}));
    let task=await stage('High-Level 编译 suffix',()=>models.compileHighLevel({split,
      deploymentMode:'server_provisioned_runner',compileConfig}));
    task=await stage('Low-Level 编译 suffix',()=>models.generateLowLevel({task}));
    const metadata=await json(task.manifest),provisioner=await directoryAt(task.directory,metadata.entry+'/provisioner');
    client=await openBrowserClient({clientDirectory:provisioner,keyDirectory,onProgress:event=>onProgress({type:'runtime',...event})});
    await stage('恢复固定结果配套的测试密钥',()=>client.restore());
    await client.close();
    client=await openBrowserClient({clientDirectory:provisioner,keyDirectory,onProgress:event=>onProgress({type:'runtime',...event})});
    const restored=await stage('关闭后重新恢复密钥',()=>client.restore());
    const bundle=await stage('参数物化、加密及自解密校验',()=>client.provisionRunner({task,parameterDirectory:provisioner}));
    const prefixRun=dataProvider?await stage('数据提供方：仅用接收的 prefix 与本地 CSV 推理',()=>dataProvider({task,bundle})):null;
    await beforeDecrypt({task,restored,bundle});
    const results=await stage('读取固定 native 密文并解密全部输出',()=>client.decryptOutputs({task,outputFile:resultFile}));
    return {split,task,bundle,results,prefixRun,stages};
  }finally{await client?.close();await models.close();}
}

/** Select the extracted outer delivery directory; no native Runner is invoked. */
export async function verifyRetinaDelivery(deliveryDirectory,{runtimeURL,onProgress=()=>{}}={}) {
  const manifest=await json(await fileAt(deliveryDirectory,'delivery-manifest.json'));
  if(manifest.schema!=='latti-ai.retina-split-delivery.v1'||manifest.mode!=='server_provisioned_runner'||manifest.ratio!=='0.9'||manifest.testOnly!==true)
    throw new Error('交付包不是固定 retina 参数加密测试夹具');
  for(const [name,entry]of Object.entries(manifest.bindings)) {
    const file=await(await fileAt(deliveryDirectory,name)).getFile();
    if(file.size!==entry.bytes||file.size>64*1024*1024||await digest(file)!==entry.sha256)throw new Error('固定样本/配置/结果被修改：'+name);
  }
  const sourceDirectory=await directoryAt(deliveryDirectory,'retina');
  const keyDirectory=await directoryAt(deliveryDirectory,'fixture/test-keys');
  const resultFile=await fileAt(deliveryDirectory,'fixture/output.ct');
  const result=await runRetinaParameterWorkflow({sourceDirectory,compileConfig:await fileAt(deliveryDirectory,'retina/compile-config.json'),
    keyDirectory,resultFile,runtimeURL,onProgress,dataProvider:async({task})=>{
      const prefixRun=await runRetinaPrefix({prefixDirectory:await directoryAt(deliveryDirectory,'handoff/prefix'),csvDirectory:sourceDirectory,runtimeURL});
      const split=await json(await fileAt(task.directory,'split/split_manifest.json'));
      for(const [index,entry]of split.inputs.entries()) {
        const expected=(await(await(await fileAt(task.directory,'split/'+entry.csv.split('/').at(-1))).getFile()).text()).trim().split(/[\s,]+/).map(Number);
        const actual=prefixRun.outputs[index];
        if(actual.name!==entry.name||actual.values.length!==expected.length)throw new Error('数据方 prefix 输出接口不匹配');
        for(let i=0;i<expected.length;i++)if(Math.abs(actual.values[i]-expected[i])>1e-7+1e-6*Math.abs(expected[i]))throw new Error('数据方 prefix 与已验证边界不等价');
      }
      return prefixRun;
    },beforeDecrypt:async({task,restored,bundle})=>{
      if(restored.profileId!==manifest.profileId||restored.keyIdentity!==manifest.keyIdentity)throw new Error('结果密文与测试密钥身份不匹配');
      const metadata=await json(task.manifest);
      if(metadata.build!==manifest.runtimeBuild)throw new Error('WASM 构建与交付清单不匹配');
      const signature=await json(await fileAt(task.directory,metadata.entry+'/runner/task_signature.json'));
      const outputs=signature.online.filter(arg=>arg.phase==='out');
      if(canonical(outputs)!==canonical(manifest.outputSignature))throw new Error('新编译任务的输出签名不匹配');
      const split=await json(await fileAt(task.directory,'split/split_manifest.json'));
      if(canonical(split.split)!==canonical(manifest.split)||canonical(split.inputs.map(x=>({name:x.name,shape:x.shape,dtype:x.dtype,sha256:x.sha256})))!==canonical(manifest.crossings))
        throw new Error('新切分任务的分区或固定样本边界不匹配');
      const runner=await json(bundle.manifest);
      if(runner.state!=='committed'||runner.profileId!==manifest.profileId||runner.keyIdentity!==manifest.keyIdentity)throw new Error('新参数包身份不匹配');
    }});
  const metadata=await json(result.results.manifest),taskMetadata=await json(result.task.manifest);
  const split=await json(await fileAt(result.task.directory,'split/split_manifest.json'));
  if(metadata.outputs.length!==split.output_routes.length)throw new Error('输出数量不匹配');
  const outputs=[];let maximum=0;
  for(const [index,entry]of metadata.outputs.entries()) {
    const route=split.output_routes[index];
    if(entry.name!==route.name||canonical(entry.shape)!==canonical(route.shape)||entry.dtype!==route.dtype)throw new Error('输出名称、shape 或 dtype 不匹配');
    const parse=text=>text.trim().split(/[\s,]+/).filter(Boolean).map(Number);
    const actual=parse(await(await(await fileAt(result.results.directory,entry.file)).getFile()).text());
    const expected=parse(await(await(await fileAt(result.task.directory,'split/'+route.plaintext.csv)).getFile()).text());
    if(actual.length!==expected.length||actual.length!==entry.count||!actual.every(Number.isFinite)||!expected.every(Number.isFinite))throw new Error('输出元素数或有限值检查失败');
    let error=0;for(let i=0;i<actual.length;i++)error=Math.max(error,Math.abs(actual[i]-expected[i]));
    maximum=Math.max(maximum,error);outputs.push({name:entry.name,shape:entry.shape,count:entry.count,maxAbsError:error,values:actual});
  }
  if(!(maximum<0.1))throw new Error('全部输出最大误差必须严格小于 0.1：'+maximum);
  return {...result,report:{status:'passed',mode:manifest.mode,ratio:manifest.ratio,testOnly:true,
    inference:'数据方实际运行仅含 prefix 的交付图；native Runner 步骤由固定密文回放替代，新参数包未在本次被原生执行',
    keyIdentity:manifest.keyIdentity,runtimeBuild:taskMetadata.build,outputs,maxAbsError:maximum,stages:result.stages}};
}

/** Data owner's operation: no suffix, original full model or key access. */
export async function runRetinaPrefix({prefixDirectory,csvDirectory,runtimeURL=new URL('./runtime/index.js',import.meta.url).href}) {
  const specification=await json(await prefixDirectory.getFileHandle('prefix.json'));
  if(specification.schema!=='latti-ai.retina-prefix.v1')throw new Error('非法 prefix 交付格式');
  const ort=await import(new URL('./model/ort/ort.wasm.min.mjs',runtimeURL));
  const {csvChunks}=await import(new URL('./csv.js',runtimeURL));
  ort.env.wasm.wasmPaths=new URL('./model/ort/',runtimeURL).href;ort.env.wasm.numThreads=1;
  const start=performance.now(),feeds={};let session,result;
  try {
    const model=new Uint8Array(await(await(await fileAt(prefixDirectory,specification.model)).getFile()).arrayBuffer());
    const externalData=[];
    for(const name of specification.externalData)externalData.push({path:relative(name),data:new Uint8Array(await(await(await fileAt(prefixDirectory,name)).getFile()).arrayBuffer())});
    session=await ort.InferenceSession.create(model,{executionProviders:['wasm'],externalData,graphOptimizationLevel:'disabled'});
    for(const input of specification.inputs) {
      const count=input.shape.reduce((a,b)=>a*b,1);if(!Number.isSafeInteger(count)||count<1||count>2**22)throw new Error('prefix 输入超出预算');
      const values=new Float32Array(count);let offset=0;
      for await(const chunk of csvChunks(await(await fileAt(csvDirectory,input.csv)).getFile())) {
        if(offset+chunk.length>count)throw new Error('prefix CSV 元素过多');
        if(!chunk.every(Number.isFinite))throw new Error('prefix CSV 非有限值');
        values.set(chunk,offset);offset+=chunk.length;
      }
      if(offset!==count||!values.every(Number.isFinite))throw new Error('prefix CSV 元素数或范围不匹配');
      feeds[input.name]=new ort.Tensor('float32',values,input.shape);
    }
    result=await session.run(feeds,specification.outputs.map(x=>x.name));
    const directory=await(await navigator.storage.getDirectory()).getDirectoryHandle('retina-prefix-'+crypto.randomUUID(),{create:true});
    const outputs=[];
    for(const output of specification.outputs) {
      const tensor=result[output.name];
      if(canonical(tensor.dims)!==canonical(output.shape)||tensor.type!==output.dtype||!tensor.data.every(Number.isFinite))throw new Error('prefix 输出 shape/dtype/finite 不匹配');
      const file=await directory.getFileHandle(output.feature_id+'.csv',{create:true}),writer=await file.createWritable();
      try{await writer.write(Array.from(tensor.data).join(',')+'\n');await writer.close();}catch(error){await writer.abort().catch(()=>{});throw error;}
      outputs.push({...output,file:output.feature_id+'.csv',values:Array.from(tensor.data)});
    }
    return {directory,outputs,durationMs:performance.now()-start};
  }finally{for(const tensor of Object.values(result||{}))tensor.dispose();for(const tensor of Object.values(feeds))tensor.dispose();await session?.release();}
}
