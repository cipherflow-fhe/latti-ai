import {fileSource, MAX_CHUNK_BYTES} from './zip.js';

export const fileOf=async value=>value.kind==='file'?value.getFile():value;
export const safeName=name=>{
  if(typeof name!=='string'||!name||/[\/\\\0]/.test(name)||name==='.'||name==='..') throw new Error('Invalid file name');
  return name;
};
// DOMException exposes message through a read-only prototype getter.
export const setErrorMessage=(error,message)=>Object.defineProperty(error,'message',{value:message,writable:true,configurable:true});
// Try every cleanup action and retain the operation's original error.
export async function cleanupAfter(error,actions) {
  const failures=[];
  for(const action of actions) try {await action();} catch(failure) {failures.push(failure.message||String(failure));}
  if(!failures.length)return;
  const detail='; cleanup: '+failures.join('; ');
  if(error) {setErrorMessage(error,error.message+detail);error.cleanupFailed=true;return;}
  throw Object.assign(new Error(detail.slice(2)),{code:'OPFS_CLEANUP_FAILED',cleanupFailed:true});
}
export async function writeFile(handle,data) {
  const writer=await handle.createWritable();
  try {await writer.write(data);await writer.close();}
  catch(error) {await writer.abort().catch(()=>{});throw error;}
}
export async function copyFile(source,target,check=()=>{}) {
  const reader=(await fileOf(source)).stream().getReader(),writer=await target.createWritable();let bytes=0n;
  try {
    for(;;) {
      check();const {value,done}=await reader.read();if(done) break;
      for(let at=0;at<value.byteLength;at+=Number(MAX_CHUNK_BYTES)) {
        const chunk=value.subarray(at,at+Number(MAX_CHUNK_BYTES));await writer.write(chunk);bytes+=BigInt(chunk.byteLength);
      }
    }
    await writer.close();return bytes;
  } catch(error) {await writer.abort().catch(()=>{});throw error;}
  finally {await reader.cancel().catch(()=>{});reader.releaseLock();}
}

// Only the current chunk crosses the Go/JS boundary. Disk completion is the
// acknowledgement, including patch writes and source reads.
export async function pump(core,operation,args,directory,sources,check,progress) {
  const outputs=new Map(),operationId=core('start',operation,...args).data;
  let active=true,ack='',input,eof;
  try {
    for(;;) {
      check();
      const result=core('next',operationId,ack,input,eof),event=result.data;
      input=undefined;eof=undefined;ack='';
      if(event.type==='done') {active=false;return outputs;}
      if(event.type==='error'||event.type==='cancelled') {active=false;throw Object.assign(new Error(event.message||event.type),{code:event.type==='cancelled'?'CANCELLED':'CORE_FAILED'});}
      try {
        if(event.type==='need-input') {
          const source=sources[event.sourceId];
          if(!source||event.maxBytes<1||event.maxBytes>Number(MAX_CHUNK_BYTES)) throw new Error('Invalid input request');
          const offset=BigInt(event.offset);
          input=await source.readAt(offset,BigInt(event.maxBytes));
          eof=offset+BigInt(input.byteLength)===BigInt(await source.size);
        } else if(event.type==='begin-output') {
          if(outputs.has(event.outputId)) throw new Error('Duplicate output');
          const handle=await directory.getFileHandle(safeName(event.outputId)+'.part',{create:true});
          outputs.set(event.outputId,{handle,writer:await handle.createWritable(),bytes:0n});
        } else if(event.type==='write-chunk'||event.type==='write-at') {
          const out=outputs.get(event.outputId),offset=BigInt(event.offset),chunk=event.chunk;
          if(!out?.writer||!chunk||chunk.byteLength>Number(MAX_CHUNK_BYTES)||offset<0n||offset>BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('Invalid output write');
          if(event.type==='write-chunk') {
            if(offset!==out.bytes) throw new Error('Output append offset mismatch');
            await out.writer.write(chunk);out.bytes+=BigInt(chunk.byteLength);
          } else {
            if(offset+BigInt(chunk.byteLength)>out.bytes) throw new Error('Output patch outside reserved range');
            await out.writer.seek(Number(offset));await out.writer.write(chunk);await out.writer.seek(Number(out.bytes));
          }
          progress(operation,out.bytes,result);
        } else if(event.type==='end-output') {
          const out=outputs.get(event.outputId);
          if(!out?.writer||BigInt(event.bytes)!==out.bytes) throw new Error('Final output length mismatch');
          await out.writer.close();out.writer=null;
          if(BigInt((await out.handle.getFile()).size)!==out.bytes) throw new Error('Closed output length mismatch');
        } else throw new Error('Unknown streaming event');
      } catch(error) {
        // Deliver the I/O error before cancellation so the producer cannot
        // continue generating a subsequent key/ciphertext.
        const terminal=core('next',operationId,error.message).data;
        if(!['error','cancelled'].includes(terminal.type)) throw new Error('Core did not stop after I/O failure');
        active=false;throw error;
      }
    }
  } finally {
    if(active) {try {core('cancel',operationId);} catch {}}
    for(const out of outputs.values()) if(out.writer) await out.writer.abort().catch(()=>{});
  }
}

export async function jobDirectory(jobId) {
  const root=await navigator.storage.getDirectory();
  const jobs=await root.getDirectoryHandle('latti-jobs',{create:true});
  const directory=await jobs.getDirectoryHandle(jobId,{create:true});
  return {directory,async cleanup(){
    try {await jobs.removeEntry(jobId,{recursive:true});}
    catch(error) {throw Object.assign(new Error(error.message+'; residual latti-jobs/'+jobId,{cause:error}),{code:'OPFS_CLEANUP_FAILED'});}
  }};
}

// Caller holds the same origin-wide lock used throughout every operation.
// Acquiring it proves no other Worker is currently writing these job folders;
// browser termination releases the lock without needing a stale-owner timeout.
export async function cleanupOrphanJobs() {
  const root=await navigator.storage.getDirectory();let jobs;
  try {jobs=await root.getDirectoryHandle('latti-jobs');} catch(error) {if(error.name==='NotFoundError')return;throw error;}
  for await(const directory of jobs.values()) {
    if(directory.kind!=='directory') continue;
    let committed=false;
    try {
      const output=await directory.getDirectoryHandle('output');
      const manifest=await(await output.getFileHandle('manifest.json')).getFile();
      if(manifest.size<=1024*1024)committed=JSON.parse(await manifest.text()).state==='committed';
    } catch(error) {if(error.name!=='NotFoundError'&&!(error instanceof SyntaxError))throw error;}
    if(!committed) try {await jobs.removeEntry(directory.name,{recursive:true});}
    catch(error) {throw Object.assign(new Error(error.message+'; residual latti-jobs/'+directory.name,{cause:error}),{code:'OPFS_CLEANUP_FAILED'});}
  }
}
export async function artifact(directory,name,role,file,profileId) {
  const snapshot=await file.getFile();
  return {artifactId:crypto.randomUUID(),role,file,bytes:String(snapshot.size),profileId};
}
export async function commit(directory,entries,profileId,extra={}) {
  const manifest=await directory.getFileHandle('manifest.json',{create:true});
  await writeFile(manifest,JSON.stringify({state:'committed',profileId,...extra,artifacts:entries.map(e=>({artifactId:e.artifactId,role:e.role,path:e.file.name,bytes:e.bytes}))})+'\n');
  return manifest;
}
export {fileSource};
