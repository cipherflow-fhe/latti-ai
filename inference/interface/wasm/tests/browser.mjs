import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, mkdtemp, mkdir } from 'node:fs/promises';
import { createWriteStream } from 'node:fs';
import { pipeline } from 'node:stream/promises';
import path from 'node:path';
import os from 'node:os';
import { chromium } from 'playwright-core';

const root = path.resolve(import.meta.dirname, '..');
const artifacts = process.env.LATTI_BROWSER_ARTIFACTS || await mkdtemp(path.join(os.tmpdir(), 'latti-browser-'));
await mkdir(artifacts, {recursive:true});
const server = createServer(async (req, res) => {
  try {
    if (req.method === 'POST' && /^\/artifact\/[A-Za-z0-9._-]+$/.test(req.url)) {
      await pipeline(req, createWriteStream(path.join(artifacts, path.basename(req.url)), {mode:0o600}));
      res.end('saved'); return;
    }
    if (req.url === '/') { res.setHeader('Content-Type','text/html'); res.end('<!doctype html><title>WASM persistence test</title>'); return; }
    const name = req.url.split('?')[0];
    const file = path.resolve(root, '.'+name);
    if (!file.startsWith(root+path.sep)) throw new Error('invalid file');
    const data = await readFile(file);
    res.setHeader('Content-Type', name.endsWith('.wasm') ? 'application/wasm' : 'text/javascript');
    res.end(data);
  } catch { res.statusCode=404; res.end('missing'); }
});
server.listen(0,'127.0.0.1');
await new Promise(resolve=>server.once('listening',resolve));
const browser = await chromium.launch({executablePath:process.env.CHROME_PATH || '/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
const page = await browser.newPage();
page.on('console',msg=>console.log('[browser]',msg.text()));
page.on('pageerror',error=>console.error('[browser error]',error.message));
try {
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.evaluate(async () => {
    const {openBrowserClient}=await import('/dist/index.js');
    const root=await navigator.storage.getDirectory();
    const keys=await root.getDirectoryHandle('N16QP1546H192H32',{create:true});
    const config={param0:{poly_modulus_degree:65536,n_mult_level:9,log_default_scale:40,param_name:'N16QP1546H192H32'}};
    const write=async (dir,name,text)=>{
      const writer=await (await dir.getFileHandle(name,{create:true})).createWritable();
      await writer.write(text); await writer.close();
    };
    // An empty key directory uses the built-in preset without JSON files.
    const client=await openBrowserClient({keyDirectory:keys});
    for (const [folder,name] of [['public','eval_context.bin'],['secret','full_context.bin']]) {
      const dir=await keys.getDirectoryHandle(folder,{create:true});
      await write(dir,name,'existing');
      let rejected=false; try { await client.setup(); } catch { rejected=true; }
      if (!rejected || await (await (await dir.getFileHandle(name)).getFile()).text()!=='existing') throw new Error('existing parameter keys overwritten');
      await dir.removeEntry(name);
    }
    await client.close();
    const task=await root.getDirectoryHandle('pregenerated-task',{create:true});
    const p={dim:0,channel:1,pack_num:1,skip:1,level:1,ckks_parameter_id:'param0'};
    await write(task,'task_config.json',JSON.stringify({use_btp:true,pack_style:'ordinary',task_input_param:{input:p},task_output_param:{output:p}}));
    // The task resolves param0 from the built-in preset.
    await (await openBrowserClient({clientDirectory:task,keyDirectory:keys})).close();
    await write(task,'task_config.json','');
    await write(task,'ckks_parameter.json',JSON.stringify(config));
    for (const options of [{clientDirectory:task},{clientDirectory:task,keyDirectory:keys}]) {
      let rejected=false;try { await openBrowserClient(options); } catch { rejected=true; }
      if (!rejected) throw new Error('empty task configuration accepted');
    }
    await task.removeEntry('ckks_parameter.json');
    await write(task,'task_config.json',JSON.stringify({use_btp:true,pack_style:'ordinary',task_input_param:{input:p},task_output_param:{output:p}}));
    const parent=await root.getDirectoryHandle('user-keys',{create:true});
    await (await openBrowserClient({keyDirectory:parent})).close();
    const nested=await parent.getDirectoryHandle('N16QP1546H192H32');
    const publicDir=await nested.getDirectoryHandle('public',{create:true});
    await write(publicDir,'eval_context.bin','existing');
    const rebound=await openBrowserClient({clientDirectory:task,keyDirectory:parent});
    let rejected=false;try { await rebound.setup(); } catch { rejected=true; }
    if (!rejected || await (await (await publicDir.getFileHandle('eval_context.bin')).getFile()).text()!=='existing') throw new Error('parent directory key protection failed');
    await rebound.close();
    // External parameter files cannot override the fixed-profile path.
    await write(keys,'ckks_parameter.json','{invalid');
    await (await openBrowserClient({clientDirectory:task,keyDirectory:keys})).close();
    for (const {useBTP,feature} of [{useBTP:true,feature:{...p,ckks_parameter_id:'other'}},{useBTP:false,feature:p}]) {
      await write(task,'task_config.json',JSON.stringify({use_btp:useBTP,pack_style:'ordinary',task_input_param:{input:feature},task_output_param:{output:feature}}));
      rejected=false;try { await openBrowserClient({clientDirectory:task,keyDirectory:keys}); } catch { rejected=true; }
      if (!rejected) throw new Error('incompatible task accepted with built-in preset');
    }
  });
  await page.evaluate(async () => {
    const {openBrowserClient} = await import('/dist/index.js');
    const directory=await (await navigator.storage.getDirectory()).getDirectoryHandle('client',{create:true});
    const param={dim:0,channel:3,pack_num:8,skip:256,level:1,ckks_parameter_id:'p'};
    for (const [name,text] of Object.entries({
      'task_config.json':JSON.stringify({use_btp:false,pack_style:'ordinary',task_input_param:{input:param},task_output_param:{output:param}}),
      'ckks_parameter.json':JSON.stringify({p:{poly_modulus_degree:4096}}),
    })) {
      const w=await (await directory.getFileHandle(name,{create:true})).createWritable(); await w.write(text); await w.close();
    }
    globalThis.directory=directory;
    globalThis.progress=[];
    globalThis.client=await openBrowserClient({clientDirectory:directory,onProgress:p=>progress.push(p)});
    const keys=await client.setup();
    if ((await keys.fullContext.getFile()).size===0 || (await keys.evalContext.getFile()).size===0) throw new Error('empty keys');
    globalThis.encrypted=await client.encrypt({inputName:'input',inputFile:new File(['1.25,-2,0.03'],'input.csv'),outputDirectory:directory});
    for (const name of ['eval_context.bin','input.ct.bin']) {
      const file=await (await directory.getFileHandle(name)).getFile();
      if (!(await fetch('/artifact/'+name,{method:'POST',body:file})).ok) throw new Error('artifact transfer failed');
    }
    await client.close();
  });
  await page.reload();
  const result=await page.evaluate(async () => {
    const {openBrowserClient}=await import('/dist/index.js');
    const directory=await (await navigator.storage.getDirectory()).getDirectoryHandle('client');
    const progress=[];
    const client=await openBrowserClient({clientDirectory:directory,onProgress:p=>progress.push(p)});
    await client.restore();
    const output=await client.decrypt({outputName:'output',ciphertextFile:await directory.getFileHandle('input.ct.bin'),outputDirectory:directory});
    const csv=await (await output.getFile()).text();
    const before=await (await directory.getFileHandle('eval_context.bin')).getFile();
    let rejected=false; try {await client.setup()} catch {rejected=true}
    if (!rejected || (await (await directory.getFileHandle('eval_context.bin')).getFile()).lastModified!==before.lastModified) throw new Error('existing keys overwritten');
    const named=await client.encrypt({inputName:'input',inputFile:new File(['4,5,6'],'new.csv'),outputDirectory:directory});
    if(named.name!=='new.ct.bin'||output.name!=='input.csv')throw new Error('source file names were not preserved');
    const csvDirectory=await directory.getDirectoryHandle('csv',{create:true});
    const encryptedDirectory=await directory.getDirectoryHandle('encrypted',{create:true});
    const decryptedDirectory=await directory.getDirectoryHandle('decrypted',{create:true});
    const write=async(dir,name,text)=>{
      const writer=await(await dir.getFileHandle(name,{create:true})).createWritable();
      await writer.write(text);await writer.close();
    };
    const samples={'2.csv':'2,3,4','10.CSV':'10,11,12','1.csv':'1,2,3','图片 空格.csv':'-1,0,1'};
    for(const[name,text]of Object.entries(samples))await write(csvDirectory,name,text);
    await write(csvDirectory,'notes.txt','not a CSV');
    await write(await csvDirectory.getDirectoryHandle('nested',{create:true}),'hidden.csv','9,9,9');
    const encrypted=await client.encrypt_batch({inputName:'input',inputDirectory:csvDirectory,outputDirectory:encryptedDirectory});
    if(JSON.stringify(encrypted.map(f=>f.name))!==JSON.stringify(['1.ct.bin','10.ct.bin','2.ct.bin','图片 空格.ct.bin']))throw new Error('batch encryption names or order differ');
    const decrypted=await client.decrypt_batch({outputName:'output',inputDirectory:encryptedDirectory,outputDirectory:decryptedDirectory});
    if(JSON.stringify(decrypted.map(f=>f.name))!==JSON.stringify(['1.csv','10.csv','2.csv','图片 空格.csv']))throw new Error('batch decryption names or order differ');
    for(const file of decrypted){
      const values=(await(await file.getFile()).text()).trim().split(',').map(Number);
      const expected=samples[file.name]||samples['10.CSV'];
      if(values.some((v,i)=>Math.abs(v-Number(expected.split(',')[i]))>0.001))throw new Error('batch roundtrip differs: '+file.name);
    }
    const empty=await directory.getDirectoryHandle('empty',{create:true});
    if((await client.encrypt_batch({inputName:'input',inputDirectory:empty,outputDirectory:encryptedDirectory})).length ||
       (await client.decrypt_batch({outputName:'output',inputDirectory:empty,outputDirectory:decryptedDirectory})).length)throw new Error('nonempty result for empty directory');
    const duplicates=await directory.getDirectoryHandle('duplicates',{create:true});
    await write(duplicates,'same.csv','1,2,3');await write(duplicates,'same.CSV','4,5,6');
    let duplicateRejected=false;
    try{await client.encrypt_batch({inputName:'input',inputDirectory:duplicates,outputDirectory:empty});}catch{duplicateRejected=true;}
    if(!duplicateRejected)throw new Error('duplicate output names accepted');
    for await(const entry of empty.values())throw new Error('duplicate preflight wrote '+entry.name);
    const invalid=await directory.getDirectoryHandle('invalid',{create:true});
    await write(invalid,'broken.csv','NaN');
    let sourceReported=false;
    try{await client.encrypt_batch({inputName:'input',inputDirectory:invalid,outputDirectory:encryptedDirectory});}catch(e){sourceReported=e.message.includes('broken.csv');}
    if(!sourceReported)throw new Error('batch error omitted source file');
    let unsafeRejected=false;
    try{await client.encrypt({inputName:'input',inputFile:new File(['1,2,3'],'bad\\name.csv'),outputDirectory:directory});}catch{unsafeRejected=true;}
    if(!unsafeRejected)throw new Error('unsafe output file name accepted');
    rejected=false;try { await client.encrypt({inputName:'../input',inputFile:new File(['4'],'invalid.csv'),outputDirectory:directory}); } catch {rejected=true}
    if (!rejected) throw new Error('invalid input accepted');
    await client.close();
    rejected=false;try {await client.restore()}catch{rejected=true}
    if (!rejected) throw new Error('closed client accepted operation');
    return {values:csv.trim().split(/[\s,]+/).map(Number),progress};
  });
  for (const [i,value] of [1.25,-2,0.03].entries()) assert(Math.abs(result.values[i]-value)<0.001);
  assert(result.progress.some(p=>p.phase==='restore'));
  console.log(JSON.stringify({artifacts,result},null,2));
} finally { await browser.close(); await new Promise(resolve=>server.close(resolve)); }
