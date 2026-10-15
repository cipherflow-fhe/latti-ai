import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { chromium } from 'playwright-core';

const root=path.resolve(import.meta.dirname,'..');
const repo=path.resolve(root,'../../..');
const artifacts=process.env.LATTI_BROWSER_ARTIFACTS;
assert(artifacts,'Run via npm run test:workflow');
const keyDirectory=process.env.LATTI_WORKFLOW_KEYS;
async function native(mode){
  const processHandle=spawn(process.env.PYTHON||'python3',[path.join(root,'tests/workflow-native.py'),mode,artifacts,repo,keyDirectory],{
    cwd:repo,env:{...process.env,PYTHONPATH:path.join(repo,'inference/interface/python/build'),LD_LIBRARY_PATH:path.join(repo,'build/inference/lattisense')},stdio:'inherit',
  });
  assert.equal(await new Promise((resolve,reject)=>{processHandle.once('error',reject);processHandle.once('close',resolve);}),0,'native workflow '+mode+' failed');
}
if(keyDirectory)await native('prepare');
const server=createServer(async(req,res)=>{
  try{
    if(req.url==='/'){
      res.setHeader('Content-Type','text/html');
      res.end('<!doctype html><button id="pick">Select test directory</button><script>pick.onclick=async()=>{try{window.directory=await showDirectoryPicker({mode:"readwrite"})}catch(e){window.pickError=e.message}}</script>');return;
    }
    if(req.url==='/favicon.ico'){res.writeHead(204).end();return;}
    const taskRequest=req.url.startsWith('/task/'),runtimeRequest=req.url.startsWith('/dist/');
    const base=taskRequest?(process.env.LATTI_WORKFLOW_TASK||path.join(repo,'examples/test_cifar10/task/client')):
      runtimeRequest&&process.env.LATTI_WORKFLOW_RUNTIME?process.env.LATTI_WORKFLOW_RUNTIME:root;
    const relative=taskRequest?req.url.slice(5):runtimeRequest&&process.env.LATTI_WORKFLOW_RUNTIME?req.url.slice(5):req.url;
    const file=path.resolve(base,'.'+relative);
    if(!file.startsWith(base+path.sep))throw Error('invalid path');
    res.setHeader('Content-Type',file.endsWith('.wasm')?'application/wasm':file.endsWith('.json')?'application/json':'text/javascript');
    res.end(await readFile(file));
  }catch{res.writeHead(404).end();}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/usr/bin/google-chrome',headless:false,args:['--no-sandbox','--disable-gpu','--lang=en-US']});
const page=await browser.newPage();
page.on('pageerror',error=>console.error('[browser]',error.message));
try{
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.locator('#pick').click();
  await page.waitForFunction(()=>window.directory||window.pickError,{},{timeout:15000});
  assert.equal(await page.evaluate(()=>window.pickError),undefined);
  const preprocessing=await page.evaluate(async()=>{
    const module=await import('/task/preprocess.js').catch(()=>({}));
    if(typeof module.preprocess_batch!=='function')throw Error('preprocess_batch is not implemented');
    const{fixtures}=await(await fetch('/tests/fixtures/preprocessing.json')).json();
    const decode=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
    const write=async(dir,name,data)=>{
      const writer=await(await dir.getFileHandle(name,{create:true})).createWritable();await writer.write(data);await writer.close();
    };
    const images=await directory.getDirectoryHandle('images',{create:true});
    const csv=await directory.getDirectoryHandle('csv',{create:true});
    for(const f of fixtures.toReversed())await write(images,f.name,decode(f.image));
    await write(images,'notes.txt','ignored');
    await write(await images.getDirectoryHandle('nested',{create:true}),'hidden.png',decode(fixtures[0].image));
    const events=[];
    const getFile=FileSystemFileHandle.prototype.getFile;
    const createWritable=FileSystemFileHandle.prototype.createWritable;
    FileSystemFileHandle.prototype.getFile=async function(){events.push('read:'+this.name);return getFile.call(this);};
    FileSystemFileHandle.prototype.createWritable=async function(...args){
      const writer=await createWritable.apply(this,args),close=writer.close.bind(writer),name=this.name;
      writer.close=async()=>{await close();events.push('closed:'+name);};return writer;
    };
    const bitmap=globalThis.createImageBitmap;
    let live=0,peak=0;
    globalThis.createImageBitmap=async(...args)=>{
      const image=await bitmap(...args),close=image.close.bind(image);live++;peak=Math.max(peak,live);
      image.close=()=>{live--;close();};return image;
    };
    let outputs;
    try{outputs=await module.preprocess_batch({inputDirectory:images,outputDirectory:csv});}
    finally{FileSystemFileHandle.prototype.getFile=getFile;FileSystemFileHandle.prototype.createWritable=createWritable;globalThis.createImageBitmap=bitmap;}
    const expectedNames=['down.csv','jpeg.csv','rgb.csv','solid.csv','up.csv','wide.csv'];
    if(JSON.stringify(outputs.map(f=>f.name))!==JSON.stringify(expectedNames))throw Error('preprocessing names or order differ');
    if(peak!==1||live!==0)throw Error('decoded images accumulated');
    const expectedEvents=fixtures.map(f=>f.name).sort().flatMap(name=>['read:'+name,'closed:'+name.replace(/\.(png|jpg)$/i,'.csv')]);
    if(JSON.stringify(events)!==JSON.stringify(expectedEvents))throw Error('files read before the previous CSV closed');
    const errors={};
    for(const f of fixtures){
      const text=await(await(await csv.getFileHandle(f.name.replace(/\.(png|jpg)$/i,'.csv'))).getFile()).text();
      const values=text.trim().split(',').map(Number),expected=new Float32Array(decode(f.values).buffer),rgb=decode(f.rgb);
      if(values.length!==3072||!values.every(Number.isFinite))throw Error('invalid model input: '+f.name);
      let maxError=0,pixelError=0;
      for(let c=0;c<3;c++)for(let k=0;k<1024;k++){
        const i=c*1024+k;
        maxError=Math.max(maxError,Math.abs(values[i]-expected[i]));
        pixelError=Math.max(pixelError,Math.abs(Math.round((values[i]*[0.229,0.224,0.225][c]+[0.485,0.456,0.406][c])*255)-rgb[k*3+c]));
      }
      if(f.name.endsWith('.png')&&(pixelError>1||maxError>(f.width===32&&f.height===32?1e-6:1/(255*0.224)+1e-6)))throw Error('Pillow reference differs: '+f.name+' '+maxError+' '+pixelError);
      errors[f.name]={maxError,pixelError};
    }
    const duplicates=await directory.getDirectoryHandle('duplicate-images',{create:true});
    const duplicateOutput=await directory.getDirectoryHandle('duplicate-csv',{create:true});
    for(const name of ['same.png','same.jpg'])await write(duplicates,name,decode(fixtures[0].image));
    let rejected=false;try{await module.preprocess_batch({inputDirectory:duplicates,outputDirectory:duplicateOutput});}catch{rejected=true;}
    if(!rejected)throw Error('duplicate image stems accepted');
    for await(const file of duplicateOutput.values())throw Error('duplicate preflight wrote '+file.name);
    const empty=await directory.getDirectoryHandle('empty-images',{create:true});
    if((await module.preprocess_batch({inputDirectory:empty,outputDirectory:csv})).length)throw Error('empty image directory produced output');
    await write(empty,'broken.png','not an image');
    rejected=false;try{await module.preprocess_batch({inputDirectory:empty,outputDirectory:csv});}catch(e){rejected=e.message.includes('broken.png');}
    if(!rejected)throw Error('image error omitted source name');
    return{outputs:outputs.map(f=>f.name),events,peak,live,errors};
  });
  assert(preprocessing.outputs.length,'no preprocessed files');
  const postprocessing=await page.evaluate(async()=>{
    const module=await import('/task/postprocess.js').catch(()=>({}));
    if(typeof module.postprocess_batch!=='function')throw Error('postprocess_batch is not implemented');
    const write=async(dir,name,text)=>{
      let file;
      try{file=await dir.getFileHandle(name,{create:true});}catch(error){throw new Error(name+': '+error.message);}
      const writer=await file.createWritable();await writer.write(text);await writer.close();return file;
    };
    const logits=await directory.getDirectoryHandle('logits',{create:true});
    const reports=await directory.getDirectoryHandle('reports',{create:true});
    await write(logits,'002.csv','0,0,7,0,0,6,0,0,0,0\n');
    await write(logits,'图 像,a.csv','0,8,0,0,0,0,0,0,0,8\n');
    await write(logits,'001.csv','0,0,0,6,0,0,0,0,0,0\n');
    const labelFile=await write(directory,'label.csv','\uFEFFfilename,label\r\n"图 像,a.png",1\r\n002.jpg,5\r\n001.png,3\r\n');
    const result=await module.postprocess_batch({inputDirectory:logits,labelFile,outputDirectory:reports});
    const report=await(await result.predictionsFile.getFile()).text();
    if(result.total!==3||result.correct!==2||result.accuracy!==2/3)throw Error('incorrect argmax or accuracy');
    const expected='filename,label,prediction,correct\n001.png,3,3,1\n002.jpg,5,2,0\n"图 像,a.png",1,1,1\n';
    if(report!==expected)throw Error('prediction report or sample order differs: '+report);
    const reordered=new File(['filename,label\n001.png,3\n002.jpg,5\n"图 像,a.png",1\n'],'labels.csv');
    const again=await module.postprocess_batch({inputDirectory:logits,labelFile:reordered,outputDirectory:reports});
    if(again.accuracy!==result.accuracy||(await(await again.predictionsFile.getFile()).text())!==report)throw Error('label row order affected results');
    const failures=[];
    const cases=[
      ['missing result','filename,label\n001.png,3\n002.png,0\n','0,0,0,6,0,0,0,0,0,0'],
      ['wrong sample','filename,label\n002.png,3\n','0,0,0,6,0,0,0,0,0,0'],
      ['duplicate label','filename,label\n001.png,3\n001.png,3\n','0,0,0,6,0,0,0,0,0,0'],
      ['duplicate stem','filename,label\n001.png,3\n001.jpg,3\n','0,0,0,6,0,0,0,0,0,0'],
      ['invalid class','filename,label\n001.png,10\n','0,0,0,6,0,0,0,0,0,0'],
      ['noninteger class','filename,label\n001.png,3.5\n','0,0,0,6,0,0,0,0,0,0'],
      ['nonfinite logits','filename,label\n001.png,3\n','NaN,0,0,6,0,0,0,0,0,0'],
      ['short logits','filename,label\n001.png,3\n','0,0,0,6,0,0,0,0,0'],
      ['empty field','filename,label\n001.png,3\n','0,,0,6,0,0,0,0,0,0'],
      ['invalid quote','filename,label\n"001.png,3\n','0,0,0,6,0,0,0,0,0,0'],
      ['unsafe filename','filename,label\n../001.png,3\n','0,0,0,6,0,0,0,0,0,0'],
    ];
    for(const[name,labels,values]of cases){
      const input=await directory.getDirectoryHandle('invalid-'+failures.length,{create:true});
      await write(input,'001.csv',values);
      let rejected=false;try{await module.postprocess_batch({inputDirectory:input,labelFile:new File([labels],'label.csv'),outputDirectory:reports});}catch{rejected=true;}
      if(!rejected)throw Error('accepted '+name);failures.push(name);
    }
    const empty=await directory.getDirectoryHandle('empty-logits',{create:true});
    let rejected=false;try{await module.postprocess_batch({inputDirectory:empty,labelFile:new File(['filename,label\n'],'label.csv'),outputDirectory:reports});}catch{rejected=true;}
    if(!rejected)throw Error('empty accuracy accepted');
    const extra=await directory.getDirectoryHandle('extra-logits',{create:true});
    await write(extra,'001.csv','0,0,0,6,0,0,0,0,0,0');
    await write(extra,'002.csv','0,0,0,6,0,0,0,0,0,0');
    rejected=false;try{await module.postprocess_batch({inputDirectory:extra,labelFile:new File(['filename,label\n001.png,3\n'],'label.csv'),outputDirectory:reports});}catch{rejected=true;}
    if(!rejected)throw Error('extra result accepted');
    failures.push('extra result');
    return{total:result.total,correct:result.correct,accuracy:result.accuracy,report,failures};
  });
  let endToEnd;
  if(keyDirectory){
    const encrypted=await page.evaluate(async()=>{
      const{preprocess_batch}=await import('/task/preprocess.js');
      const{openBrowserClient}=await import('/dist/index.js');
      const work=await directory.getDirectoryHandle('cifar10');
      const csv=await work.getDirectoryHandle('csv',{create:true});
      await preprocess_batch({inputDirectory:await work.getDirectoryHandle('images'),outputDirectory:csv});
      const metrics={wasmBytes:0,heapBytes:0},events=[];
      const client=await openBrowserClient({clientDirectory:await work.getDirectoryHandle('client'),keyDirectory:await work.getDirectoryHandle('N16QP1546H192H32'),onProgress:p=>{
        for(const name of Object.keys(metrics))metrics[name]=Math.max(metrics[name],p[name]);
      }});
      await client.restore();
      let active=0,peak=0;
      const encrypt=client.encrypt;
      client.encrypt=async options=>{
        events.push({start:options.inputFile.name,time:performance.now()});peak=Math.max(peak,++active);
        const result=await encrypt(options);active--;
        events.push({closed:result.name,time:performance.now(),bytes:(await result.getFile()).size});return result;
      };
      try{
        const files=await client.encrypt_batch({inputName:'input',inputDirectory:csv,outputDirectory:await work.getDirectoryHandle('encrypted-inputs',{create:true})});
        if(files.length!==3||peak!==1||active!==0)throw Error('batch encryption did not process one sample at a time');
        return{files:files.map(f=>f.name),metrics,events,peak};
      }finally{await client.close();}
    });
    await native('evaluate');
    await page.reload();
    await page.locator('#pick').click();
    await page.waitForFunction(()=>window.directory||window.pickError,{},{timeout:15000});
    const decrypted=await page.evaluate(async()=>{
      const{openBrowserClient}=await import('/dist/index.js');
      const{postprocess_batch}=await import('/task/postprocess.js');
      const work=await directory.getDirectoryHandle('cifar10');
      const logits=await work.getDirectoryHandle('logits',{create:true});
      const metrics={wasmBytes:0,heapBytes:0},events=[];
      const client=await openBrowserClient({clientDirectory:await work.getDirectoryHandle('client'),keyDirectory:await work.getDirectoryHandle('N16QP1546H192H32'),onProgress:p=>{
        for(const name of Object.keys(metrics))metrics[name]=Math.max(metrics[name],p[name]);
      }});
      await client.restore();
      const decrypt=client.decrypt;
      client.decrypt=async options=>{
        events.push({start:options.ciphertextFile.name,time:performance.now()});
        const result=await decrypt(options);events.push({closed:result.name,time:performance.now()});return result;
      };
      let files;
      try{files=await client.decrypt_batch({outputName:'output',inputDirectory:await work.getDirectoryHandle('encrypted-results'),outputDirectory:logits});}
      finally{await client.close();}
      const result=await postprocess_batch({inputDirectory:logits,labelFile:await work.getFileHandle('label.csv'),outputDirectory:await work.getDirectoryHandle('reports',{create:true})});
      return{files:files.map(f=>f.name),metrics,events,total:result.total,correct:result.correct,accuracy:result.accuracy};
    });
    await native('check');
    const checked=JSON.parse(await readFile(path.join(artifacts,'cifar10/checked.json'),'utf8'));
    assert.equal(decrypted.total,checked.total);assert.equal(decrypted.correct,checked.correct);assert.equal(decrypted.accuracy,checked.accuracy);
    endToEnd={encrypted,decrypted,...checked};
  }
  await writeFile(path.join(artifacts,'report.json'),JSON.stringify({passed:true,preprocessing,postprocessing,endToEnd},null,2));
  console.log(JSON.stringify({artifacts,preprocessing,postprocessing,endToEnd},null,2));
}finally{await browser.close();await new Promise(resolve=>server.close(resolve));}
