import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile,mkdtemp,mkdir} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {createWriteStream} from 'node:fs';
import {pipeline} from 'node:stream/promises';
import path from 'node:path';
import os from 'node:os';
import {chromium} from 'playwright-core';

const root=path.resolve(import.meta.dirname,'..');
const runtime=process.env.LATTI_RUNTIME_DIRECTORY?path.resolve(process.env.LATTI_RUNTIME_DIRECTORY):path.join(root,'dist');
const artifacts=process.env.LATTI_BROWSER_ARTIFACTS||await mkdtemp(path.join(os.tmpdir(),'taa-browser-'));
await mkdir(artifacts,{recursive:true});
const parameters=execFileSync('python3',['-c',`import json,sys
v=json.load(open(sys.argv[1]))['CKKS']['12']
v.update(poly_modulus_degree=4096,ring_type=1,n_slots=4096)
print(json.dumps({'p':v},separators=(',',':')))` ,path.resolve(root,'../../lattisense/frontend/parameter.json')],{encoding:'utf8'}).trim();
const server=createServer(async(req,res)=>{
  try {
    if(req.method==='POST'&&/^\/artifact\/[A-Za-z0-9._-]+$/.test(req.url)) {
      await pipeline(req,createWriteStream(path.join(artifacts,path.basename(req.url)),{mode:0o600}));res.end('saved');return;
    }
    if(req.url==='/') {res.setHeader('Content-Type','text/html');res.end('<!doctype html><title>WASM 0.2.0 OPFS acceptance</title>');return;}
    const name=req.url.split('?')[0],base=name.startsWith('/dist/')?runtime:root;
    const file=path.resolve(base,'.'+(name.startsWith('/dist/')?name.slice(5):name));
    if(!file.startsWith(base+path.sep)) throw new Error('Invalid path');
    const data=await readFile(file);res.setHeader('Content-Type',name.endsWith('.wasm')?'application/wasm':name.endsWith('.json')?'application/json':'text/javascript');res.end(data);
  } catch {res.statusCode=404;res.end('missing');}
});
server.listen(0,'127.0.0.1');await new Promise(resolve=>server.once('listening',resolve));
const browser=await chromium.launch({...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{}),headless:true,args:['--no-sandbox']});
const page=await browser.newPage();page.on('console',message=>console.log('[browser]',message.type(),message.text()));page.on('pageerror',error=>console.error(error));
try {
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const taskChecks=await page.evaluate(async()=>{
    const {preprocessDirectory,decodeRGB,resizeRGB,normalizeRGB}=await import('/dist/tasks/cifar10/preprocess.js');
    const {postprocess}=await import('/dist/tasks/cifar10/postprocess.js');
    const {fixtures}=await(await fetch('/tests/fixtures/preprocessing.json')).json();
    const root=await navigator.storage.getDirectory();
    const write=async(directory,name,data)=>{const handle=await directory.getFileHandle(name,{create:true}),writer=await handle.createWritable();await writer.write(data);await writer.close();return handle;};
    const images=await root.getDirectoryHandle('images',{create:true}),csv=await root.getDirectoryHandle('image-csv',{create:true});
    const decode=text=>Uint8Array.from(atob(text),c=>c.charCodeAt(0));
    for(const fixture of fixtures)await write(images,fixture.name,decode(fixture.image));
    const original=createImageBitmap;let live=0,peak=0;
    globalThis.createImageBitmap=async(...args)=>{const image=await original(...args),close=image.close.bind(image);live++;peak=Math.max(peak,live);image.close=()=>{live--;close();};return image;};
    try {if((await preprocessDirectory({inputDirectory:images,outputDirectory:csv})).count!=='6')throw new Error('Preprocessing count');}
    finally {globalThis.createImageBitmap=original;}
    if(live||peak!==1)throw new Error('Decoded bitmaps accumulated');
    const errors={};
    for(const fixture of fixtures) {
      const file=await(await images.getFileHandle(fixture.name)).getFile();
      const {pixels,width,height}=await decodeRGB(file),actual=resizeRGB(pixels,width,height),expected=decode(fixture.rgb);
      const reference=new DataView(decode(fixture.values).buffer),values=normalizeRGB(actual);
      const csvValues=(await(await(await csv.getFileHandle(fixture.name.replace(/\.(png|jpg)$/i,'.csv'))).getFile()).text()).trim().split(',').map(Number);
      let pixelError=0,maxError=0;
      for(let i=0;i<actual.length;i++)pixelError=Math.max(pixelError,Math.abs(actual[i]-expected[i]));
      for(let i=0;i<values.length;i++) {
        maxError=Math.max(maxError,Math.abs(values[i]-reference.getFloat32(i*4,true)));
        if(values[i]!==csvValues[i])throw new Error('CSV did not roundtrip float32');
      }
      if(fixture.name.endsWith('.png')&&(pixelError>1||maxError>(width===32&&height===32?1e-6:1/(255*0.224)+1e-6)))throw new Error('Pillow reference mismatch: '+fixture.name);
      errors[fixture.name]={pixelError,maxError};
    }
    const expectError=async(action,message)=>{let rejected=false;try{await action();}catch{rejected=true;}if(!rejected)throw new Error(message);};
    await expectError(()=>preprocessDirectory({inputDirectory:images,outputDirectory:csv}),'Existing CSV overwritten');
    const duplicates=await root.getDirectoryHandle('duplicate-images',{create:true}),empty=await root.getDirectoryHandle('empty-images',{create:true});
    await write(duplicates,'same.png',decode(fixtures[0].image));await write(duplicates,'same.JPG',decode(fixtures[0].image));
    await expectError(()=>preprocessDirectory({inputDirectory:duplicates,outputDirectory:csv}),'Duplicate image stems accepted');
    await expectError(()=>preprocessDirectory({inputDirectory:empty,outputDirectory:csv}),'Empty image directory accepted');
    await write(empty,'broken.png','broken');await expectError(()=>preprocessDirectory({inputDirectory:empty,outputDirectory:csv}),'Corrupt image accepted');
    await expectError(()=>preprocessDirectory({inputDirectory:images,outputDirectory:csv,check(){throw new Error('cancelled');}}),'Preprocessing cancellation ignored');
    const logits=await root.getDirectoryHandle('classification',{create:true});
    const records=[];
    for(const [id,values]of [['002','0,0,7,0,0,6,0,0,0,0'],['001','0,8,0,0,0,0,0,0,0,8']]) {
      const file=await write(logits,id+'.csv',values);records.push({case_id:id,path:file.name,role:'plaintext',bytes:String((await file.getFile()).size)});
    }
    await write(logits,'entries.jsonl',records.map(record=>JSON.stringify(record)+'\n').join(''));
    const manifest=await write(logits,'manifest.json',JSON.stringify({state:'committed',count:'2',entries:'entries.jsonl'}));
    const report=await root.getFileHandle('predictions.jsonl',{create:true});
    const statistics=await postprocess({results:{directory:logits,manifest},labels:new File(['{"case_id":"001","label":1}\n{"case_id":"002","label":5}\n'],'labels.jsonl'),report});
    if(statistics.total!==2||statistics.correct!==1||statistics.accuracy!==0.5)throw new Error('Browser classification differs');
    return {errors,peak,statistics};
  });
  await page.evaluate(async parameters=>{
    const {openBrowserClient}=await import('/dist/index.js');
    const root=await navigator.storage.getDirectory(),keys=await root.getDirectoryHandle('keys',{create:true});
    const jobs=await root.getDirectoryHandle('latti-jobs',{create:true}),orphan=await jobs.getDirectoryHandle('orphan',{create:true});
    const orphanFile=await orphan.getFileHandle('incomplete.part',{create:true}),orphanWriter=await orphanFile.createWritable();await orphanWriter.write('partial');await orphanWriter.close();
    const committed=await(await jobs.getDirectoryHandle('committed-keep',{create:true})).getDirectoryHandle('output',{create:true});
    const marker=await committed.getFileHandle('manifest.json',{create:true}),markerWriter=await marker.createWritable();await markerWriter.write('{"state":"committed"}');await markerWriter.close();
    const progress=[],profile={parameters,bootstrap:false};
    const client=await openBrowserClient({keyDirectory:keys,profile,onProgress:p=>progress.push(p)});
    let removed=false;try{await jobs.getDirectoryHandle('orphan');}catch(error){removed=error.name==='NotFoundError';}
    if(!removed)throw new Error('Startup retained an abandoned job');await jobs.getDirectoryHandle('committed-keep');
    const result=await client.setup();
    if(!result.fullContext.bytes||result.fullContext.role!=='private-full'||result.evalContext.role!=='public-eval') throw new Error('Invalid committed key artifacts');
    for(const [name,ref]of [['eval_context.bin',result.evalContext],['full_context.bin',result.fullContext]]) {
      if(!(await fetch('/artifact/'+name,{method:'POST',body:await ref.file.getFile()})).ok) throw new Error('Artifact export failed');
    }
    if(!(await fetch('/artifact/ckks_parameter.json',{method:'POST',body:parameters})).ok) throw new Error('Parameter export failed');
    if(!progress.some(p=>p.chunkBytes>0&&p.inFlightChunks===1&&typeof p.bytes==='string')) throw new Error('Missing bounded key progress');
    await client.close();
  },parameters);
  await page.reload();
  const result=await page.evaluate(async parameters=>{
    const {openBrowserClient}=await import('/dist/index.js');
    const {writeZip,openZip}=await import('/dist/zip.js');
    const root=await navigator.storage.getDirectory(),keys=await root.getDirectoryHandle('keys');
    const taskDirectory=await root.getDirectoryHandle('task',{create:true});
    const write=async(directory,name,data)=>{const handle=await directory.getFileHandle(name,{create:true}),writer=await handle.createWritable();await writer.write(data);await writer.close();return handle;};
    const feature={dim:0,channel:3,pack_num:8,skip:512,level:1,ckks_parameter_id:'p'};
    const task={evaluation_keys:[],use_btp:false,pack_style:'ordinary',task_input_id:['input'],task_output_id:['output'],task_input_param:{input:feature},task_output_param:{output:feature}};
    await write(taskDirectory,'task_config.json',JSON.stringify(task));await write(taskDirectory,'ckks_parameter.json',parameters);
    const profile={parameters,bootstrap:false},progress=[];
    let cancelNext=false,cancelledJob,cancelAck,closeNext=false,closing;
    const client=await openBrowserClient({keyDirectory:keys,profile,onProgress:p=>{
      progress.push(p);
      if(cancelNext){cancelNext=false;cancelledJob=p.jobId;cancelAck=client.cancel(p.requestId);}
      if(closeNext){closeNext=false;closing=client.close();}
    }});
    await client.restore();await client.bindTask(taskDirectory);
    const outputs=await root.getDirectoryHandle('results',{create:true});
    const encrypted=await client.encrypt({inputName:'input',inputFile:new File(['1.25,-2,0.03'],'input.csv'),outputDirectory:outputs});
    if(encrypted.file.name!=='input.ct.bin'||encrypted.role!=='ciphertext') throw new Error('Single-file artifact contract');
    const decrypted=await client.decrypt({outputName:'output',ciphertextFile:encrypted.file,outputDirectory:outputs});
    const values=(await(await decrypted.file.getFile()).text()).trim().split(',').map(Number);
    if(values.some((v,i)=>Math.abs(v-[1.25,-2,0.03][i])>0.001)) throw new Error('Cross-session streaming roundtrip');
    let rejected=false;try {await client.setup();} catch {rejected=true;}if(!rejected) throw new Error('Existing key overwrite accepted');
    const csv=await root.getDirectoryHandle('csv',{create:true});
    for(const [name,data]of [['1.csv','1,2,3'],['10.CSV','10,11,12'],['图片 空格.csv','-1,0,1']]) await write(csv,name,data);
    const encryptedBatch=await client.encryptBatch({inputName:'input',inputDirectory:csv});
    const decryptedBatch=await client.decryptBatch({outputName:'output',inputDirectory:encryptedBatch.directory});
    if(encryptedBatch.count!=='3'||decryptedBatch.count!=='3') throw new Error('Disk batch summary');
    const manifest=JSON.parse(await(await decryptedBatch.manifest.getFile()).text());
    if(manifest.state!=='committed') throw new Error('Uncommitted batch published');
    const duplicates=await root.getDirectoryHandle('duplicates',{create:true});await write(duplicates,'same.csv','1');await write(duplicates,'same.CSV','2');
    rejected=false;try {await client.encryptBatch({inputName:'input',inputDirectory:duplicates});} catch {rejected=true;}if(!rejected) throw new Error('Duplicate accepted');
    const invalid=await root.getDirectoryHandle('invalid',{create:true});await write(invalid,'broken.csv','NaN');
    rejected=false;try {await client.encryptBatch({inputName:'input',inputDirectory:invalid});} catch(error) {rejected=error.message.includes('broken.csv');}if(!rejected) throw new Error('Invalid CSV lost source context');
    const serverCSV=await root.getDirectoryHandle('server-csv',{create:true});await write(serverCSV,'001.csv','1.25,-2,0.03');
    const input=await client.encryptBatch({mode:'server',inputDirectories:{input:serverCSV},forceZip64:true});
    const zip=await openZip(input.archive.file),entries=[];
    for await(const entry of zip.entries()) entries.push(entry);
    if(entries.length!==1||entries[0].name!=='001.ct') throw new Error('Server ZIP entry contract');
    const ct=await root.getFileHandle('server.ct',{create:true});await zip.readEntry(entries[0],ct);
    const archive=await root.getFileHandle('output.zip',{create:true}),spoolHandle=await root.getFileHandle('spool.tmp',{create:true});
    // Echo is a wire/OPFS fixture only; actual server acceptance is separate.
    await writeZip(archive,[{name:'001.result.bin',file:await ct.getFile()}],{spoolHandle,forceZip64:true});
    const output=await client.decryptBatch({mode:'server',outputName:'output',inputFile:archive,inputBatch:input});
    if(output.count!=='1') throw new Error('Server result count');
    const serverValues=(await(await(await output.directory.getFileHandle('001.csv')).getFile()).text()).trim().split(',').map(Number);
    if(serverValues.some((v,i)=>Math.abs(v-[1.25,-2,0.03][i])>0.001)) throw new Error('Server ZIP fixture differs');
    rejected=false;try {await client.encryptBatch({mode:'server',inputDirectories:{wrong:serverCSV}});} catch {rejected=true;}if(!rejected) throw new Error('Wrong task input accepted');
    cancelNext=true;
    rejected=false;
    try {await client.encrypt({inputName:'input',inputFile:new File([Array(32768).fill('0.25').join(',')],'cancel.csv')});}
    catch(error){rejected=error.code==='CANCELLED';}
    if(!rejected||!(await cancelAck).accepted)throw new Error('Cancellation did not settle the target request');
    const jobs=await root.getDirectoryHandle('latti-jobs');
    let absent=false;try{await jobs.getDirectoryHandle(cancelledJob);}catch(error){absent=error.name==='NotFoundError';}
    if(!absent)throw new Error('Cancelled job retained uncommitted files');
    closeNext=true;
    const active=client.encrypt({inputName:'input',inputFile:new File([Array(32768).fill('0.25').join(',')],'close.csv')});
    const queued=client.restore();
    const terminals=await Promise.allSettled([active,queued]);
    if(terminals.some(result=>result.status!=='rejected')||!closing)throw new Error('Close did not settle active/queued work');
    await closing;
    rejected=false;try {await client.restore();} catch {rejected=true;}if(!rejected) throw new Error('Closed client accepted work');
    return {values,serverValues,progress};
  },parameters);
  assert(result.progress.some(p=>p.phase==='encrypt'));
  await page.evaluate(async parameters=>{
    const {openBrowserClient}=await import('/dist/index.js');
    const keyDirectory=await(await navigator.storage.getDirectory()).getDirectoryHandle('race-keys',{create:true});
    const options={keyDirectory,profile:{parameters,bootstrap:false}};
    const first=await openBrowserClient(options),second=await openBrowserClient(options);
    const results=await Promise.allSettled([first.setup(),second.setup()]);
    if(results.filter(result=>result.status==='fulfilled').length!==1)throw new Error('Concurrent key setup did not protect the common target');
    await first.close();await second.close();
  },parameters);
  // Runtime fetch failure must settle open instead of leaving it pending.
  await page.route('**/latti_client.wasm',route=>route.fulfill({status:404,body:'missing'}));
  const startupFailure=await page.evaluate(async()=>{
    const {openBrowserClient}=await import('/dist/index.js');
    try {await openBrowserClient({keyDirectory:await navigator.storage.getDirectory()});return false;} catch(error) {return error.message.includes('404');}
  });assert(startupFailure);await page.unroute('**/latti_client.wasm');
  const buildInfo=await readFile(path.join(runtime,'build-info.js'),'utf8');
  await page.route('**/build-info.js',route=>route.fulfill({contentType:'text/javascript',body:buildInfo.replace(/"build":"[^"]+"/,'"build":"mixed-package"')}));
  assert(await page.evaluate(async()=>{
    const {openBrowserClient}=await import('/dist/index.js');
    try {await openBrowserClient({keyDirectory:await navigator.storage.getDirectory()});return false;}
    catch(error) {return error.message.includes('WASM_BUILD_MISMATCH');}
  }),'Mixed actual Worker/WASM build accepted');
  await page.unroute('**/build-info.js');
  // Faults execute inside a real Dedicated Worker against actual OPFS handles.
  for(const fault of ['write','seek','close','cleanup']) {
    const original=await readFile(path.join(runtime,'io.js'),'utf8');
    const hook=`
const originalCreateWritable=FileSystemFileHandle.prototype.createWritable;
FileSystemFileHandle.prototype.createWritable=async function(...args) {
 const writer=await originalCreateWritable.apply(this,args);
 if(!this.name.endsWith('.part')) return writer;
 return new Proxy(writer,{get(target,key){
  if(key===${JSON.stringify(fault==='cleanup'?'write':fault)})return async()=>{globalThis.injectedFailure=true;throw new DOMException('injected ${fault} failure','QuotaExceededError');};
  const value=Reflect.get(target,key,target);return typeof value==='function'?value.bind(target):value;
 }});
};
${fault==='cleanup'?`const originalRemove=FileSystemDirectoryHandle.prototype.removeEntry;
FileSystemDirectoryHandle.prototype.removeEntry=function(name,...args){
 if(globalThis.injectedFailure)throw new Error('injected removal failure: '+name);
 return originalRemove.call(this,name,...args);
};`:''}`;
    await page.route('**/io.js',route=>route.fulfill({contentType:'text/javascript',body:original+hook}));
    await page.evaluate(async({parameters,fault})=>{
      const {openBrowserClient}=await import('/dist/index.js');
      const keyDirectory=await(await navigator.storage.getDirectory()).getDirectoryHandle('fault-'+fault,{create:true});
      const client=await openBrowserClient({keyDirectory,profile:{parameters,bootstrap:false}});
      let failed=false;
      try{await client.setup();}catch(error){failed=error.message.includes('injected '+fault+' failure')&&(fault!=='cleanup'||error.message.includes('injected removal failure'));}
      if(!failed)throw new Error('OPFS '+fault+' failure was not reported');
      for await(const directory of keyDirectory.values()) if(directory.kind==='directory') {
        for(const name of ['manifest.json','public/eval_context.bin','secret/full_context.bin']) {
          const parts=name.split('/');let dir=directory,exists=false;
          try{for(const part of parts.slice(0,-1))dir=await dir.getDirectoryHandle(part);await dir.getFileHandle(parts.at(-1));exists=true;}
          catch(error){if(error.name!=='NotFoundError')throw error;}
          if(exists)throw new Error('Failed setup published '+name);
        }
      }
      await client.close();
    },{parameters,fault});
    await page.unroute('**/io.js');
  }
  console.log(JSON.stringify({artifacts,taskChecks,result},null,2));
} finally {await browser.close();await new Promise(resolve=>server.close(resolve));}
