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
    await client.encrypt({inputName:'input',inputFile:new File(['4,5,6'],'new.csv'),outputDirectory:directory});
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
