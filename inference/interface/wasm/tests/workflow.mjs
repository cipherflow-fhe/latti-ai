import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright-core';

const root=path.resolve(import.meta.dirname,'..');
const repo=path.resolve(root,'../../..');
const artifacts=process.env.LATTI_BROWSER_ARTIFACTS;
assert(artifacts,'Run via npm run test:workflow');
const server=createServer(async(req,res)=>{
  try{
    if(req.url==='/'){
      res.setHeader('Content-Type','text/html');
      res.end('<!doctype html><button id="pick">Select test directory</button><script>pick.onclick=async()=>{try{window.directory=await showDirectoryPicker({mode:"readwrite"})}catch(e){window.pickError=e.message}}</script>');return;
    }
    if(req.url==='/favicon.ico'){res.writeHead(204).end();return;}
    const base=req.url.startsWith('/task/')?path.join(repo,'examples/test_cifar10/task/client'):root;
    const relative=req.url.startsWith('/task/')?req.url.slice(5):req.url;
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
  await writeFile(path.join(artifacts,'report.json'),JSON.stringify({preprocessing},null,2));
  console.log(JSON.stringify({artifacts,preprocessing},null,2));
}finally{await browser.close();await new Promise(resolve=>server.close(resolve));}
