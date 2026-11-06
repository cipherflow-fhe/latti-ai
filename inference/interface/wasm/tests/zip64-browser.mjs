import assert from 'node:assert/strict';
import {mkdtemp} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
import os from 'node:os';
import {browserHarness} from './browser-harness.mjs';
const artifacts=await mkdtemp(path.join(os.tmpdir(),'taa-zip64-'));
const harness=await browserHarness(artifacts),{page}=harness;
await page.exposeFunction('lattiTestPhase',name=>harness.beginPhase(name));
try{
 const result=await page.evaluate(async()=>{
  const {writeZip,openZip,fileSource}=await import('/dist/zip.js');
  const root=await navigator.storage.getDirectory(),directory=await root.getDirectoryHandle('zip64-boundaries',{create:true});
  const start=performance.now(),baseline=performance.memory?.usedJSHeapSize||null,usageBefore=await navigator.storage.estimate();console.log('Storage quota '+JSON.stringify(usageBefore));
  const block=new Uint8Array(4*1024*1024).fill(0x5a),size=(1n<<32n)+17n;
  const spoolHandle=await directory.getFileHandle('central.tmp',{create:true});
  const smallSize=64n*1024n*1024n,small=await directory.getFileHandle('small.zip',{create:true});
  await globalThis.lattiTestPhase('zip-write-64MiB');
  await writeZip(small,[{name:'small.ct',source:{size:smallSize,async readAt(offset,maxBytes){return block.subarray(0,Number(smallSize-offset<maxBytes?smallSize-offset:maxBytes));}}}],{spoolHandle});
  await globalThis.lattiTestPhase('zip-read-64MiB');
  const smallReader=await openZip(small),smallOutput=await directory.getFileHandle('small-extracted.ct',{create:true});
  for await(const record of smallReader.entries())await smallReader.readEntry(record,smallOutput);
  if(BigInt((await smallOutput.getFile()).size)!==smallSize)throw new Error('Small extraction size');
  if(!(await fetch('/artifact/small.zip',{method:'POST',body:await small.getFile()})).ok)throw new Error('Small transfer');
  await directory.removeEntry('small-extracted.ct');await directory.removeEntry('small.zip');
  await globalThis.lattiTestPhase('zip-write-4GiB');
  let bytes=0n,next=512n*1024n*1024n,reads=0;
  const source={size,async readAt(offset,maxBytes){const length=Number(size-offset<maxBytes?size-offset:maxBytes);bytes+=BigInt(length);reads++;if(bytes>=next){console.log('ZIP64 write bytes '+bytes);next+=512n*1024n*1024n;}return block.subarray(0,length);}};
  const large=await directory.getFileHandle('large.zip',{create:true});
  await writeZip(large,[{name:'large.ct',source},{name:'after.ct',file:new Blob(['tail'])}],{spoolHandle});
  const zip=await openZip(large),records=[];for await(const record of zip.entries())records.push(record);
  if(records[0].size!==size||records[1].offset<=0xffffffffn)throw new Error('Real size/offset boundary not crossed');
  await globalThis.lattiTestPhase('zip-read-4GiB');
  const extracted=await directory.getFileHandle('large-extracted.ct',{create:true});await zip.readEntry(records[0],extracted);
  const file=await extracted.getFile();if(BigInt(file.size)!==size)throw new Error('Large extraction size');
  const last=await fileSource(file).readAt(size-17n,17n);if(last.some(value=>value!==0x5a))throw new Error('Large extraction bytes');
  if(!(await fetch('/artifact/large.zip',{method:'POST',body:await large.getFile()})).ok)throw new Error('Large transfer');
  const opfsWithExtracted=(await navigator.storage.estimate()).usage;
  await directory.removeEntry('large-extracted.ct');
  await globalThis.lattiTestPhase('zip-write-65536entries');
  const countFile=await directory.getFileHandle('count.zip',{create:true});
  async function* emptyEntries(){for(let index=0;index<65536;index++)yield{name:'empty-'+index,file:new Blob()};}
  await writeZip(countFile,emptyEntries(),{spoolHandle});console.log('ZIP64 65536 entries written');
  const counted=await openZip(countFile);if(counted.count!==65536n)throw new Error('ZIP64 entry count');
  if(!(await fetch('/artifact/count.zip',{method:'POST',body:await countFile.getFile()})).ok)throw new Error('Count transfer');
  const usageAfter=await navigator.storage.estimate();
  return{smallEntryBytes:String(smallSize),largeBytes:String((await large.getFile()).size),entryBytes:String(size),secondOffset:String(records[1].offset),count:String(counted.count),sourceReads:reads,maxSourceChunk:block.byteLength,elapsedMs:performance.now()-start,jsHeapBefore:baseline,jsHeapAfter:performance.memory?.usedJSHeapSize||null,opfsBefore:usageBefore.usage,opfsWithExtracted,opfsAfter:usageAfter.usage};
 });
 const independent=JSON.parse(execFileSync('python3',['-c',`import json,sys,zipfile
from pathlib import Path
root=Path(sys.argv[1])
with zipfile.ZipFile(root/'small.zip') as z:
 assert z.getinfo('small.ct').file_size==64*1024*1024
 assert z.testzip() is None
with zipfile.ZipFile(root/'large.zip') as z:
 assert z.getinfo('large.ct').file_size>2**32
 assert z.getinfo('after.ct').header_offset>2**32
 assert z.read('after.ct')==b'tail'
 assert z.testzip() is None
with zipfile.ZipFile(root/'count.zip') as z:
 assert len(z.infolist())==65536
 assert z.testzip() is None
print(json.dumps({'smallCRC':'pass','largeCRC':'pass','countCRC':'pass'}))`,artifacts],{encoding:'utf8',maxBuffer:1024*1024}));
 assert.equal(result.count,'65536');console.log(JSON.stringify({artifacts,result,independent,rss:harness.rss,phaseMetrics:harness.phaseMetrics},null,2));
}finally{await harness.close();}
