import {BUILD_INFO} from './build-info.js';
import {cleanupPythonWorkspace} from './python-workspace.js';
import {cleanupAfter} from './io.js';

/** Internal parameter producer, owned and cancelled by the FHE operation. */
export async function openParameterModel(task,parameterDirectory,signal,onProgress) {
  const worker=new Worker(new URL('./model-worker.js',import.meta.url),{type:'module'});
  const pending=new Map(),jobId=crypto.randomUUID();let nextID=0,closed=false,closing;
  const fail=error=>{for(const waiter of pending.values())waiter.reject(error);pending.clear();};
  const close=()=>{
    if(closed)return closing;closed=true;clearTimeout(timeout);worker.terminate();signal.removeEventListener('abort',abort);
    fail(Object.assign(new Error('Parameter materializer closed'),{code:'CANCELLED'}));
    closing=cleanupPythonWorkspace(jobId);return closing;
  };
  const abort=()=>{close().catch(()=>{});};
  const timeout=setTimeout(()=>{fail(new Error('Parameter materializer timed out'));abort();},1800000);
  signal.addEventListener('abort',abort,{once:true});
  if(signal.aborted)abort();
  worker.onerror=()=>fail(new Error('Parameter model Worker failed'));
  worker.onmessage=({data})=>{
    if(data.jobId!==jobId||data.protocolVersion!==BUILD_INFO.protocolVersion)return;
    if(data.type==='progress'){onProgress?.(data);return;}
    const waiter=pending.get(data.requestId);if(!waiter)return;pending.delete(data.requestId);
    if(data.type==='error')waiter.reject(new Error(data.message));else waiter.resolve(data.result);
  };
  const request=(type,options={})=>new Promise((resolve,reject)=>{
    if(closed){reject(new Error('Parameter materializer is closed'));return;}
    const requestId=++nextID;pending.set(requestId,{resolve,reject});
    try{worker.postMessage({type,...options,requestId,jobId,protocolVersion:BUILD_INFO.protocolVersion});}
    catch(error){pending.delete(requestId);reject(error);}
  });
  try {
    const metadata=await request('parameterPlan',{task,parameterDirectory});
    return {metadata,value:(argument,flatIndex)=>request('parameterValue',{argument,flatIndex}),close};
  }catch(error){await cleanupAfter(error,[close]);throw error;}
}
