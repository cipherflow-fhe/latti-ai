import test from 'node:test';
import assert from 'node:assert/strict';
import {postprocess_batch} from '../src/tasks/retina/raw-postprocess.js';

function directory(entries={}) {
  const data=new Map(Object.entries(entries));
  const handle=name=>({kind:'file',name,getFile:async()=>new File([data.get(name)],name),createWritable:async()=>{
    let pending='';return {write:async text=>{pending+=text;},close:async()=>data.set(name,pending),abort:async()=>{}};
  }});
  return {async *values(){for(const name of data.keys())yield handle(name);},getFileHandle:async(name,{create=false}={})=>{if(!data.has(name)){if(!create)throw new Error('Missing file');data.set(name,'');}return handle(name);}};
}
test('raw batch postprocessing matches filenames and computes tied AUC and confusion metrics',async()=>{
  const inputDirectory=directory({'a.csv':'0,0','b.csv':'0,0','c.csv':'0,2','d.csv':'2,0'});
  const result=await postprocess_batch({inputDirectory,labelFile:new File(['filename,label\nd.jpg,1\nc.jpg,1\nb.jpg,0\na.jpg,0\n'],'label.csv'),outputDirectory:directory()});
  const {predictionsFile,...metrics}=result;
  assert.deepEqual(metrics,{total:4,correct:3,accuracy:0.75,specificity:1,sensitivity:0.5,auc:0.5});
  assert.match(await(await predictionsFile.getFile()).text(),/a.jpg,0,0,1/);
});
test('raw batch rejects missing labels/nonfinite logits and reports undefined single-class AUC',async()=>{
  const run=(entries,labels)=>postprocess_batch({inputDirectory:directory(entries),labelFile:new File([labels],'label.csv'),outputDirectory:directory()});
  await assert.rejects(run({'a.csv':'0,1'},'filename,label\nb.jpg,1\n'),/same samples/);
  await assert.rejects(run({'a.csv':'NaN,1'},'filename,label\na.jpg,1\n'),/finite logits/);
  const result=await run({'a.csv':'0,1'},'filename,label\na.jpg,1\n');
  assert.equal(result.auc,null);assert.equal(result.specificity,null);assert.equal(result.sensitivity,1);
});
