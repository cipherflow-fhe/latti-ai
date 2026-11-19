import {BUILD_INFO} from './build-info.js';
import {cleanupPythonWorkspace} from './python-workspace.js';
import {cleanupAfter} from './io.js';

/** One operation runs at a time; later calls queue in invocation order. */
export async function openBrowserModelTools({onProgress,timeoutMs=1800000}={}) {
  if(!globalThis.isSecureContext)throw new Error('A secure browser context is required');
  if(!Number.isSafeInteger(timeoutMs)||timeoutMs<=0)throw new Error('Invalid model operation timeout');
  let worker,active,closed=false,nextID=0,queue=Promise.resolve();
  const start=()=>{
    if(worker)return;
    worker=new Worker(new URL('./model-worker.js',import.meta.url),{type:'module'});
    worker.onmessage=({data})=>{
      if(!active||data.requestId!==active.requestId||data.jobId!==active.jobId)return;
      if(data.protocolVersion!==BUILD_INFO.protocolVersion)return;
      if(data.type==='progress'){try{onProgress?.(data);}catch{}return;}
      if(data.type==='result')finish(null,data.result);
      else if(data.type==='error')finish(Object.assign(new Error(data.message),{code:data.code,phase:data.phase}));
    };
    worker.onerror=()=>finish(new Error('Model Worker failed'));
    worker.onmessageerror=()=>finish(new Error('Model Worker message failed'));
  };
  const cleanup=async operation=>{
    if(!operation.output)return;
    // A manifest is the commit marker, including cancellation races at commit.
    try {
      const file=await(await operation.output.getFileHandle('artifact-manifest.json')).getFile();
      const manifest=JSON.parse(await file.text());
      if(manifest.schema==='latti-ai.browser-model-artifact.v1'&&manifest.files&&manifest.entry)return;
    } catch(error) {
      if(error.name!=='NotFoundError'&&!(error instanceof SyntaxError))throw error;
    }
    await operation.parent.removeEntry(operation.jobId,{recursive:true});
  };
  const finish=async(error,result)=>{
    const operation=active;
    if(!operation||operation.finishing)return;
    operation.finishing=true;clearTimeout(operation.timer);
    worker?.terminate();worker=undefined;
    try {await cleanupAfter(error,[()=>cleanupPythonWorkspace(operation.jobId),...(error?[()=>cleanup(operation)]:[])]);}
    catch(cleanupError){error=cleanupError;}
    active=undefined;
    if(error)operation.reject(error);else operation.resolve(result);
  };
  const request=(type,options)=>{
    if(closed)return Promise.reject(new Error('Model tools are closed'));
    const run=async()=>{
      if(closed)throw new Error('Model tools are closed');
      const parent=options.outputDirectory||await(await navigator.storage.getDirectory()).getDirectoryHandle('latti-model',{create:true});
      const requestId=++nextID,jobId=crypto.randomUUID();
      const output=await parent.getDirectoryHandle(jobId,{create:true});
      if(closed){await parent.removeEntry(jobId,{recursive:true});throw new Error('Model tools are closed');}
      return new Promise((resolve,reject)=>{
        active={requestId,jobId,parent,output,resolve,reject};
        active.timer=setTimeout(()=>finish(Object.assign(new Error('Model operation timed out'),{code:'TIMEOUT'})),timeoutMs);
        try {
          start();worker.postMessage({...options,type,requestId,jobId,outputDirectory:output,
            protocolVersion:BUILD_INFO.protocolVersion});
        }catch(error){finish(error);}
      });
    };
    const result=queue.then(run);queue=result.catch(()=>{});return result;
  };
  return {
    prepareSplit: options=>request('prepareSplit',options),
    runPrefix: options=>request('runPrefix',options),
    compileHighLevel: options=>request('compileHighLevel',options),
    generateLowLevel: options=>request('generateLowLevel',options),
    async cancel(requestId) {
      if(!active||active.finishing||(requestId!==undefined&&requestId!==active.requestId))return {accepted:false};
      await finish(Object.assign(new Error('Model operation cancelled'),{code:'CANCELLED'}));
      return {accepted:true};
    },
    async close() {
      closed=true;
      await finish(Object.assign(new Error('Model tools closed'),{code:'CANCELLED'}));
      worker?.terminate();worker=undefined;
      await queue;
    },
  };
}
