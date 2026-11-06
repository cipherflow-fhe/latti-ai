import assert from 'node:assert/strict';
import {mkdtemp,writeFile} from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {browserHarness} from './browser-harness.mjs';

// Run against the extracted delivery, including task modules' local imports.
assert(process.env.LATTI_DELIVERY_DIRECTORY,'Set LATTI_DELIVERY_DIRECTORY to the extracted delivery');
const delivery=path.resolve(process.env.LATTI_DELIVERY_DIRECTORY);
const artifacts=await mkdtemp(path.join(os.tmpdir(),'taa-delivery-browser-'));
const harness=await browserHarness(artifacts,{'/delivery/':delivery});
const errors=[];
harness.page.on('pageerror',error=>errors.push(error.message));
try {
  const result=await harness.page.evaluate(async()=>{
    const cifar=await import('/delivery/cifar10/task/client/preprocess.js');
    const cifarPost=await import('/delivery/cifar10/task/client/postprocess.js');
    const retina=await import('/delivery/retina/task/client/preprocess.js');
    const retinaPost=await import('/delivery/retina/task/client/postprocess.js');
    if([cifarPost.postprocess,retina.preprocessDirectory,retinaPost.postprocess].some(value=>typeof value!=='function'))throw new Error('Missing task exports');
    const root=await navigator.storage.getDirectory();
    const images=await root.getDirectoryHandle('images',{create:true}),csv=await root.getDirectoryHandle('csv',{create:true});
    const cases=await(await fetch('/delivery/cifar10/data/cases.json')).json();
    for(const sample of cases) {
      const image=await images.getFileHandle(sample.case_id+'.png',{create:true});
      await(await fetch('/delivery/cifar10/data/'+sample.image)).body.pipeTo(await image.createWritable());
    }
    const processed=await cifar.preprocessDirectory({inputDirectory:images,outputDirectory:csv});
    let maxAbsoluteError=0,values=0;
    for(const sample of cases) {
      const actual=(await(await(await csv.getFileHandle(sample.case_id+'.csv')).getFile()).text()).trim().split(',').map(Number);
      const expected=(await(await fetch('/delivery/cifar10/data/'+sample.inputs.input)).text()).trim().split(',').map(Number);
      if(actual.length!==3072||actual.length!==expected.length)throw new Error('Invalid CIFAR shape');
      for(let index=0;index<actual.length;index++){maxAbsoluteError=Math.max(maxAbsoluteError,Math.abs(actual[index]-expected[index]));values++;}
    }
    if(maxAbsoluteError>1e-6)throw new Error('Packaged preprocessing differs from reference');
    return {count:processed.count,values,maxAbsoluteError,taskModules:4};
  });
  // Exercise the shipped HTML and its real Worker initialization. Native
  // picker interaction is represented by a real OPFS directory handle.
  await harness.page.addInitScript(()=>{
    window.showDirectoryPicker=async()=>{
      const root=await navigator.storage.getDirectory();
      return root.getDirectoryHandle('example-keys',{create:true});
    };
  });
  await harness.page.goto(new URL('/dist/example.html',harness.page.url()).href);
  await harness.page.locator('#keys').click();
  await harness.page.locator('#status').filter({hasText:'Generate new keys or restore this directory.'}).waitFor({timeout:60000});
  await harness.page.locator('#close').click();
  await harness.page.locator('#status').filter({hasText:'Client closed.'}).waitFor();
  assert.deepEqual(errors,[]);
  const report={artifacts,...result,exampleWorkerStartAndClose:true,rss:harness.rss};
  await writeFile(path.join(artifacts,'delivery-browser.json'),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify(report,null,2));
} finally {await harness.close();}
