import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { copyFile, mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { chromium } from 'playwright-core';

const root=path.resolve(import.meta.dirname,'..');
const repo=path.resolve(root,'../../..');
const artifacts=process.env.LATTI_BROWSER_ARTIFACTS;
assert(artifacts,'Run via npm run test:pregeneration (isolated native directory chooser)');
const preset='N16QP1546H192H32';
const taskRoot=path.join(repo,'examples/test_cifar10/task');
await mkdir(path.join(artifacts,'resnet20'));
await copyFile(path.join(taskRoot,'client/task_config.json'),path.join(artifacts,'resnet20/task_config.json'));
await copyFile(path.join(taskRoot,'client/img.csv'),path.join(artifacts,'resnet20/img.csv'));
await mkdir(path.join(artifacts,'scalar'));
const scalar={dim:0,channel:3,pack_num:32,skip:1024,level:1,ckks_parameter_id:'param0'};
await writeFile(path.join(artifacts,'scalar/task_config.json'),JSON.stringify({use_btp:true,pack_style:'ordinary',task_input_param:{input:scalar},task_output_param:{output:scalar}}));
const server=createServer(async(req,res)=>{
  try {
    if(req.url==='/') {
      res.setHeader('Content-Type','text/html');
      res.end('<!doctype html><title>Parameter pregeneration acceptance</title><button id="pick">Select test directory</button><script>pick.onclick=async()=>{try{window.directory=await showDirectoryPicker({mode:"readwrite"})}catch(e){window.pickError=e.message}}</script>');return;
    }
    if(req.url==='/favicon.ico'){res.writeHead(204).end();return;}
    const file=path.resolve(root,'.'+req.url);
    if(!file.startsWith(root+path.sep))throw Error('invalid path');
    res.setHeader('Content-Type',file.endsWith('.wasm')?'application/wasm':'text/javascript');
    res.end(await readFile(file));
  }catch{res.writeHead(404).end();}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/usr/bin/google-chrome',headless:false,args:['--no-sandbox','--disable-gpu','--lang=en-US']});
const page=await browser.newPage();
page.on('console',message=>console.log('[browser]',message.text()));
page.on('pageerror',error=>console.error('[browser error]',error.message));
const cdp=await browser.newBrowserCDPSession();
const rss={sampleIntervalMs:250,samples:0,rendererPeakKiB:0,chromeProcessesPeakKiB:0};
let sampling=false;
async function sampleRSS(){
  if(sampling)return;sampling=true;
  try {
    const {processInfo}=await cdp.send('SystemInfo.getProcessInfo');
    let renderers=0,total=0;
    for(const p of processInfo){
      try {
        const status=await readFile(`/proc/${p.id}/status`,'utf8');
        const value=Number(status.match(/^VmRSS:\s+(\d+)/m)?.[1]||0);
        total+=value;if(p.type==='renderer')renderers+=value;
      }catch{}
    }
    rss.samples++;rss.rendererPeakKiB=Math.max(rss.rendererPeakKiB,renderers);rss.chromeProcessesPeakKiB=Math.max(rss.chromeProcessesPeakKiB,total);
  }finally{sampling=false;}
}
async function selectDirectory(){
  await page.locator('#pick').click();
  await page.waitForFunction(()=>window.directory||window.pickError,{},{timeout:15000});
  assert.equal(await page.evaluate(()=>window.pickError),undefined);
}
let timer;
try {
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await selectDirectory();
  await sampleRSS();
  const baseline={...rss};
  timer=setInterval(()=>sampleRSS().catch(()=>{}),rss.sampleIntervalMs);
  const setup=await page.evaluate(async()=>{
    const {openBrowserClient}=await import('/dist/index.js');
    const keyDirectory=directory;
    const metrics={wasmBytes:0,heapBytes:0,payloadBytes:0,chunkBytes:0,inFlightChunks:0,chunks:0};
    let lastLog=0;
    const client=await openBrowserClient({keyDirectory,onProgress:p=>{
      for(const key of ['wasmBytes','heapBytes','payloadBytes','chunkBytes','inFlightChunks'])metrics[key]=Math.max(metrics[key],p[key]);
      if(p.chunkBytes)metrics.chunks++;
      if(p.elapsedMs-lastLog>30000){lastLog=p.elapsedMs;console.log(JSON.stringify(p));}
    }});
    const start=performance.now();
    const keys=await client.setup();
    const result={metrics,elapsedMs:performance.now()-start,evalBytes:(await keys.evalContext.getFile()).size,fullBytes:(await keys.fullContext.getFile()).size};
    let rejected=false;
    try{await client.encrypt({inputName:'input',inputFile:new File(['1'],'input.csv'),outputDirectory:directory});}catch(e){rejected=/task configuration/.test(e.message);}
    if(!rejected)throw Error('key-only encryption accepted without task');
    await client.close();
    return result;
  });
  clearInterval(timer);timer=undefined;
  await sampleRSS();
  assert(setup.evalBytes>3*1024**3);
  assert(setup.fullBytes>0&&setup.fullBytes<30*1024**2);
  assert(setup.metrics.chunkBytes<=4*1024**2);
  assert.equal(setup.metrics.inFlightChunks,1);
  assert.equal((await stat(path.join(artifacts,preset,'public/eval_context.bin'))).size,setup.evalBytes);
  await assert.rejects(stat(path.join(artifacts,preset,'ckks_parameter.json')),{code:'ENOENT'});
  await writeFile(path.join(artifacts,'setup.json'),JSON.stringify({setup,rss,baseline},null,2));
  console.log(JSON.stringify({artifacts,setup,rss,baseline}));
  // Save a ciphertext, then cross a real page reload before decrypting it.
  await page.evaluate(async()=>{
    const {openBrowserClient}=await import('/dist/index.js');
    const client=await openBrowserClient({keyDirectory:await directory.getDirectoryHandle('N16QP1546H192H32'),clientDirectory:await directory.getDirectoryHandle('scalar')});
    await client.restore();
    await client.encrypt({inputName:'input',inputFile:new File(['1.25,-2,0.03'],'old.csv'),outputDirectory:directory});
    await client.close();
  });
  await page.reload();await selectDirectory();
  const scalarValues=await page.evaluate(async()=>{
    const {openBrowserClient}=await import('/dist/index.js');
    const client=await openBrowserClient({keyDirectory:await directory.getDirectoryHandle('N16QP1546H192H32'),clientDirectory:await directory.getDirectoryHandle('scalar')});
    await client.restore();
    const output=await client.decrypt({outputName:'output',ciphertextFile:await directory.getFileHandle('old.ct.bin'),outputDirectory:directory});
    const values=(await(await output.getFile()).text()).trim().split(',').map(Number);
    await client.encrypt({inputName:'input',inputFile:new File(['4,5,6'],'new.csv'),outputDirectory:directory});
    await client.close();return values;
  });
  for(const [i,want]of [1.25,-2,0.03].entries())assert(Math.abs(scalarValues[i]-want)<0.001);
  await page.evaluate(async()=>{
    const {openBrowserClient}=await import('/dist/index.js');
    const task=await directory.getDirectoryHandle('resnet20');
    const client=await openBrowserClient({keyDirectory:await directory.getDirectoryHandle('N16QP1546H192H32'),clientDirectory:task});
    await client.restore();
    await client.encrypt({inputName:'input',inputFile:await task.getFileHandle('img.csv'),outputDirectory:directory});
    await client.close();
  });
  const native=spawn(process.env.PYTHON||'python3',[path.join(root,'tests/pregeneration-server.py'),artifacts,repo],{cwd:repo,env:{...process.env,PYTHONPATH:path.join(repo,'inference/interface/python/build'),LD_LIBRARY_PATH:path.join(repo,'build/inference/lattisense')},stdio:'inherit'});
  assert.equal(await new Promise((resolve,reject)=>{native.once('error',reject);native.once('close',resolve);}),0,'native ResNet-20 server failed');
  await page.reload();await selectDirectory();
  const decrypted=await page.evaluate(async()=>{
    const {openBrowserClient}=await import('/dist/index.js');
    const client=await openBrowserClient({keyDirectory:await directory.getDirectoryHandle('N16QP1546H192H32'),clientDirectory:await directory.getDirectoryHandle('resnet20')});
    await client.restore();
    const output=await client.decrypt({outputName:'output',ciphertextFile:await directory.getFileHandle('server-output.bin'),outputDirectory:directory});
    const values=(await(await output.getFile()).text()).trim().split(',').map(Number);
    await client.close();return values;
  });
  const expected=JSON.parse(await readFile(path.join(artifacts,'server.json'),'utf8'));
  assert.equal(decrypted.length,10);
  const maxAbsoluteError=Math.max(...decrypted.map((v,i)=>Math.abs(v-expected.plaintext[i])));
  assert(maxAbsoluteError<0.1,`ResNet-20 error ${maxAbsoluteError}`);
  const report={passed:true,artifacts,setup,rss,baseline,scalarValues,decrypted,...expected,maxAbsoluteError};
  await writeFile(path.join(artifacts,'report.json'),JSON.stringify(report,null,2));
  console.log(JSON.stringify(report,null,2));
}finally{
  clearInterval(timer);await browser.close();await new Promise(resolve=>server.close(resolve));
}
