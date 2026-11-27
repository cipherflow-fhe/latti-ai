import {runParameterBatchWorkflow,fileAt,directoryAt,readJSON} from './batch-workflow.mjs';
export {runParameterBatchWorkflow} from './batch-workflow.mjs';
const canonical=value=>JSON.stringify(value,(_,v)=>v&&typeof v==='object'&&!Array.isArray(v)?Object.fromEntries(Object.entries(v).sort(([a],[b])=>a.localeCompare(b))):v);
const digest=async file=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',await file.arrayBuffer())),b=>b.toString(16).padStart(2,'0')).join('');

/** Default delivery acceptance: raw ten-sample Retina batch, encrypted inputs
 * and parameters. The bound ZIP replays only the native GPU FHE stage.
 */
export async function verifyDelivery(deliveryDirectory,{runtimeURL,onProgress=()=>{},onStage=async()=>{}}={}) {
  const manifest=await readJSON(await fileAt(deliveryDirectory,'delivery-manifest.json'));
  if(manifest.schema!=='latti-ai.onnx-batch-delivery.v1'||manifest.mode!=='server_provisioned_runner'||manifest.testOnly!==true||manifest.ratio!=='0.9'||manifest.inputMode!=='client_ciphertext')throw new Error('Expected encrypted-input parameter batch delivery');
  for(const [name,entry]of Object.entries(manifest.bindings)) {
    const file=await(await fileAt(deliveryDirectory,name)).getFile();
    // The large evaluation key is verified by restore's streaming hash gate.
    if(file.size!==entry.bytes||file.size>512*1024*1024||await digest(file)!==entry.sha256)throw new Error('Delivery file binding differs: '+name);
  }
  const runtime=runtimeURL||new URL('./runtime/index.js',import.meta.url).href;
  const preprocessingURL=new URL('./processing/preprocess.js',import.meta.url).href;
  const postprocessingURL=new URL('./processing/postprocess.js',import.meta.url).href;
  const result=await runParameterBatchWorkflow({modelDirectory:await directoryAt(deliveryDirectory,'retina'),modelPath:manifest.modelPath,
    rawDirectory:await directoryAt(deliveryDirectory,'retina/dataset'),labelFile:await fileAt(deliveryDirectory,'retina/dataset/label.csv'),
    preprocessingURL,postprocessingURL,compileConfig:await fileAt(deliveryDirectory,'retina/compile-config.json'),
    keyDirectory:await directoryAt(deliveryDirectory,'fixture/test-keys'),keyProfile:manifest.keyProfile,runtimeURL:runtime,onProgress,onStage,
    runNative:async({crossings,keys,bundle})=>{
      if(keys.keyIdentity!==manifest.keyIdentity||canonical(crossings)!==canonical(manifest.crossings))throw new Error('Replay key or per-sample crossing binding differs');
      const runner=await readJSON(await bundle.directory.getFileHandle('task_config.json'));
      if(runner.input_mode!=='client_ciphertext'||runner.parameter_loading_mode!=='runtime_lazy')throw new Error('Runner mode differs');
      return fileAt(deliveryDirectory,'fixture/output.zip');
    }});
  const reference=await readJSON(await fileAt(deliveryDirectory,'fixture/reference.json'));
  const cases=Object.keys(reference).sort(),actual=[];
  for await(const entry of result.results.directory.values())if(entry.kind==='file'&&entry.name.endsWith('.csv'))actual.push(entry.name.slice(0,-4));
  if(canonical(cases)!==canonical(actual.sort())||cases.length!==manifest.count)throw new Error('Decrypted sample coverage differs');
  let maxAbsError=0;const samples=[];
  for(const id of cases) {
    const values=(await(await(await result.results.directory.getFileHandle(id+'.csv')).getFile()).text()).trim().split(/[\s,]+/).map(Number);
    if(values.length!==reference[id].length||values.some(x=>!Number.isFinite(x)))throw new Error('Output count/finite check failed: '+id);
    const error=Math.max(...values.map((value,i)=>Math.abs(value-reference[id][i])));maxAbsError=Math.max(maxAbsError,error);
    samples.push({caseId:id,maxAbsError:error});
  }
  if(!(maxAbsError<0.1))throw new Error('Maximum absolute error must be below 0.1');
  const {predictionsFile,...statistics}=result.statistics;
  const report={status:'passed',count:cases.length,ratio:'0.9',inputMode:'client_ciphertext',parameterMode:'encrypted_offline',
    runner:'bound native GPU runtime_lazy ciphertext replay',keyIdentity:result.keys.keyIdentity,maxAbsError,tolerance:0.1,samples,statistics,stages:result.stages};
  return {...result,report};
}
