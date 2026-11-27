import {BUILD_INFO} from './build-info.js';
export {openBrowserModelTools} from './model.js';

export async function openBrowserClient({clientDirectory,keyDirectory,profile,onProgress,onArtifact}) {
  if (!globalThis.isSecureContext) throw new Error('A secure browser context is required');
  if (!clientDirectory && !keyDirectory) throw new Error('Select a client or key directory');
  for (const directory of [clientDirectory,keyDirectory]) {
    if (directory && directory.kind !== 'directory') throw new Error('Select a directory');
  }
  const worker = new Worker(new URL('./worker.js',import.meta.url),{type:'module'});
  const {protocolVersion,runtimeVersion,build} = BUILD_INFO;
  const sessionJobId = globalThis.crypto?.randomUUID?.() || `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  let nextID = 0, closed = false, closing = false, closePromise, queue = Promise.resolve(), readyResolve, readyReject;
  const pending = new Map();
  const ready = new Promise((resolve,reject) => { readyResolve=resolve; readyReject=reject; });
  const startupTimer=setTimeout(()=>settleWorker(new Error('Browser client initialization timed out')),30000);
  const settleWorker = error => {
    clearTimeout(startupTimer);
    closed = true;
    readyReject?.(error);
    for (const waiter of pending.values()) waiter.reject(error);
    pending.clear(); worker.terminate();
  };
  worker.onmessage = ({data}) => {
    if (!data || data.protocolVersion !== protocolVersion || !data.jobId || data.requestId === undefined || typeof data.type !== 'string') {
      settleWorker(new Error('Invalid Worker protocol envelope')); return;
    }
    if (data.requestId===0 && data.type==='error') {settleWorker(new Error(data.message || 'Worker initialization failed'));return;}
    if (data.type === 'ready' && data.requestId === 0) {
      if (data.runtimeVersion !== runtimeVersion || data.build !== build || data.capabilities?.protocolVersion !== protocolVersion) {
        settleWorker(new Error('WASM_PROTOCOL_VERSION_MISMATCH')); return;
      }
      clearTimeout(startupTimer); readyResolve(data); return;
    }
    const waiter = pending.get(data.requestId);
    if (!waiter) return;
    if (data.jobId!==waiter.jobId) {settleWorker(new Error('Worker job ID mismatch'));return;}
    if (data.type === 'artifact') {onArtifact?.(data.artifact);return;}
    if (data.type === 'progress') { onProgress?.({...data.progress,requestId:data.requestId,jobId:data.jobId}); return; }
    if (!['result','error','cancelled'].includes(data.type)) {settleWorker(new Error('Unexpected Worker message'));return;}
    pending.delete(data.requestId);
    if (data.type === 'error') {
      const error = new Error(data.message || 'Browser client Worker failed');
      error.code = data.code; error.phase = data.phase; error.entry = data.entry; error.retryable = data.retryable;
      waiter.reject(error);
    } else if (data.type === 'cancelled') waiter.reject(Object.assign(new Error(data.reason || 'Operation cancelled'), {code:'CANCELLED'}));
    else waiter.resolve(data.result);
  };
  worker.onerror = () => settleWorker(new Error('Browser client Worker failed'));
  worker.onmessageerror = () => settleWorker(new Error('Browser client Worker message failed'));
  const request = (type,options={},immediate=false) => {
    if (closed || (closing && type!=='close' && type!=='cancel')) return Promise.reject(new Error('Browser client is closed'));
    const run = () => new Promise((resolve,reject) => {
      if (closed || (closing && type!=='close' && type!=='cancel')) {reject(new Error('Browser client is closed'));return;}
      const requestId = ++nextID, jobId = globalThis.crypto?.randomUUID?.() || `${sessionJobId}-${requestId}`;
      pending.set(requestId,{resolve,reject,jobId});
      try { worker.postMessage({...options,protocolVersion,requestId,jobId,type}); }
      catch (error) { pending.delete(requestId); reject(error); }
    });
    const result = immediate ? run() : queue.then(run);
    if(!immediate) queue = result.catch(()=>{});
    return result;
  };
  try { await ready; await request('open',{clientDirectory,keyDirectory,profile}); }
  catch (error) { settleWorker(error); throw error; }
  const client = {
    setup: () => request('setup'),
    restore: () => request('restore'),
    exportPublicContext: options => request('exportPublicContext',options),
    importPublicContext: options => request('importPublicContext',options),
    encrypt: options => request('encrypt',options),
    decrypt: options => request('decrypt',options),
    decryptOutputs: options => request('decryptOutputs',options),
    provisionRunner: options => request('provisionRunner',options),
    bindTask: clientDirectory => request('bindTask',{clientDirectory}),
    encryptBatch: options => request('encryptBatch',options),
    decryptBatch: options => request('decryptBatch',options),
    cancel: requestId => request('cancel',{targetRequestId:requestId??[...pending.keys()][0]},true),
    close() {
      if(closePromise) return closePromise;
      if(closed) return Promise.resolve();
      closing=true;
      closePromise=request('close',{},true).finally(()=>{closed=true;worker.terminate();});
      return closePromise;
    },
  };
  return client;
}

/** Save operator-supplied processing modules in a selected static resource directory.
 * Modules are executable code: the caller reviews them before importing their URL.
 */
export async function uploadProcessingScripts({directory,preprocessFile,postprocessFile}) {
  if(directory?.kind!=='directory')throw new Error('Select the static resource directory');
  const modules=[['preprocess.js',preprocessFile],['postprocess.js',postprocessFile]],files={};
  // Validate both files before publishing either.
  const sources=await Promise.all(modules.map(async([name,source])=>{
    const file=source?.kind==='file'?await source.getFile():source;
    if(!file||!file.name?.endsWith('.js')||file.size===0||file.size>1024*1024)throw new Error('Select a JavaScript module up to 1 MiB');
    const bytes=await file.arrayBuffer();
    const sha256=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
    return {name,bytes,sha256};
  }));
  for(const {name,bytes,sha256}of sources) {
    const handle=await directory.getFileHandle(name,{create:true}),writer=await handle.createWritable();
    try{await writer.write(bytes);await writer.close();}catch(error){await writer.abort().catch(()=>{});throw error;}
    files[name]={bytes:bytes.byteLength,sha256};
  }
  const manifest=await directory.getFileHandle('processing-manifest.json',{create:true}),writer=await manifest.createWritable();
  try{await writer.write(JSON.stringify({schema:'latti-ai.processing-scripts.v1',files},null,2)+'\n');await writer.close();}
  catch(error){await writer.abort().catch(()=>{});throw error;}
  return {directory,manifest};
}
