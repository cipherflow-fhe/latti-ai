import assert from 'node:assert/strict';
import {mkdtemp,readFile,writeFile} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {browserHarness} from './browser-harness.mjs';

const artifacts=process.env.LATTI_BROWSER_ARTIFACTS||await mkdtemp(path.join(os.tmpdir(),'taa-pregeneration-'));
console.log('Artifacts:',artifacts);
const harness=await browserHarness(artifacts);
try {
  const setup=await harness.page.evaluate(async()=>{
    const {openBrowserClient}=await import('/dist/index.js');
    const root=await navigator.storage.getDirectory(),keys=await root.getDirectoryHandle('keys',{create:true});
    const metrics={wasmBytes:0,heapBytes:0,chunkBytes:0,inFlightChunks:0};let lastLog=0;
    let diskPeak=0;const diskBefore=await navigator.storage.estimate();
    const timer=setInterval(async()=>{diskPeak=Math.max(diskPeak,(await navigator.storage.estimate()).usage);},1000);
    const client=await openBrowserClient({keyDirectory:keys,onProgress:p=>{
      for(const key of Object.keys(metrics))metrics[key]=Math.max(metrics[key],p[key]||0);
      if(p.elapsedMs-lastLog>30000){lastLog=p.elapsedMs;console.log(JSON.stringify(p));}
    }});
    try {
      const start=performance.now(),result=await client.setup();
      for(const [name,handle]of [['eval_context.bin',result.evalContext.file],['full_context.bin',result.fullContext.file],['key-manifest.json',result.manifest]]) {
        if(!(await fetch('/artifact/'+name,{method:'POST',body:await handle.getFile()})).ok)throw new Error('Artifact upload failed');
      }
      const diskAfter=await navigator.storage.estimate();diskPeak=Math.max(diskPeak,diskAfter.usage);
      return {profileId:result.profileId,evalBytes:result.evalContext.bytes,fullBytes:result.fullContext.bytes,elapsedMs:performance.now()-start,metrics,diskBefore,diskAfter,diskPeak};
    } finally {clearInterval(timer);await client.close();}
  });
  assert(BigInt(setup.evalBytes)>0n);assert(BigInt(setup.fullBytes)>0n);assert(setup.metrics.chunkBytes<=4*1024**2);assert.equal(setup.metrics.inFlightChunks,1);
  await harness.page.reload();
  const restored=await harness.page.evaluate(async()=>{
    const {openBrowserClient}=await import('/dist/index.js');
    const client=await openBrowserClient({keyDirectory:await(await navigator.storage.getDirectory()).getDirectoryHandle('keys')});
    try{return await client.restore();}finally{await client.close();}
  });
  assert.equal(restored.profileId,setup.profileId);
  const manifest=JSON.parse(await readFile(path.join(artifacts,'key-manifest.json')));
  assert.equal(manifest.profile.profileId,setup.profileId);
  await writeFile(path.join(artifacts,'pregeneration.json'),JSON.stringify({setup,restored,rss:harness.rss},null,2)+'\n');
  console.log(JSON.stringify({artifacts,setup,restored,rss:harness.rss},null,2));
} finally {await harness.close();}
