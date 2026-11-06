import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {resizeRGB,normalizeRGB} from '../src/tasks/cifar10/preprocess.js';
import {argmax,classify} from '../src/tasks/classification.js';
import {postprocess as cifar} from '../src/tasks/cifar10/postprocess.js';
import {postprocess as retina} from '../src/tasks/retina/postprocess.js';

const fixtures=JSON.parse(await readFile(new URL('./fixtures/preprocessing.json',import.meta.url)));
for(const fixture of fixtures.fixtures) test('current torchvision reference: '+fixture.name,()=>{
  const pixels=resizeRGB(Buffer.from(fixture.decoded,'base64'),fixture.width,fixture.height);
  const expected=Buffer.from(fixture.rgb,'base64');
  assert.deepEqual(new Uint8Array(pixels),new Uint8Array(expected),'Pillow RGB8 mismatch');
  const values=normalizeRGB(pixels),bytes=Buffer.from(fixture.values,'base64');
  let maximum=0;
  for(let i=0;i<3072;i++) maximum=Math.max(maximum,Math.abs(values[i]-bytes.readFloatLE(i*4)));
  assert(maximum<=1e-6,'float32 CHW/normalization mismatch: '+maximum);
});

function handle(text='') {
  let committed=new Blob([text]);
  return {getFile:async()=>committed,createWritable:async()=>{
    const pending=[];
    return {write:async value=>pending.push(value),close:async()=>{committed=new Blob(pending);},abort:async()=>{}};
  }};
}
function input(records,labels) {
  const files=new Map(),entries=[];
  for(const [id,logits]of records) {
    const file=handle(logits),path=id+'.csv';files.set(path,file);
    entries.push(JSON.stringify({case_id:id,path,role:'plaintext',bytes:String(new Blob([logits]).size)}));
  }
  files.set('entries.jsonl',handle(entries.join('\n')+'\n'));
  return {
    results:{directory:{getFileHandle:async name=>{if(!files.has(name))throw new Error('Missing file');return files.get(name);}},manifest:handle(JSON.stringify({state:'committed',count:String(records.length),entries:'entries.jsonl'}))},
    labels:new Blob([labels.map(([case_id,label])=>JSON.stringify({case_id,label})).join('\n')]),report:handle('old report'),
  };
}

test('argmax uses first index on ties and validates every logit',()=>{
  assert.equal(argmax([3,3],2),0);
  for(const values of [[],[1],[1,2,3],[NaN,1],[1,Infinity]])assert.throws(()=>argmax(values,2));
});
test('Retina matches shuffled labels by exact case ID and streams independent predictions',async()=>{
  const options=input([['002','0,2'],['001','3,3']],[['001',0],['002',1]]);
  assert.deepEqual(await retina(options),{total:2,correct:2,accuracy:1});
  const records=(await(await options.report.getFile()).text()).trim().split('\n').map(JSON.parse);
  assert.deepEqual(records.map(r=>[r.case_id,r.prediction,r.label]),[['002',1,1],['001',0,0]]);
});
test('CIFAR requires ten logits and computes correct/total',async()=>{
  const options=input([['a','0,0,0,0,0,0,0,0,9,1'],['b','1,0,0,0,0,0,0,0,0,0']],[['b',1],['a',8]]);
  assert.deepEqual(await cifar(options),{total:2,correct:1,accuracy:0.5});
});
test('missing, extra, duplicate, invalid or empty labels never commit a report',async()=>{
  for(const labels of [[],[['other',1]],[['a',1],['b',0]],[['a',0],['a',1]],[['a',2]],[['a',-1]],[['a',0.5]],[['A',0]]]) {
    const options=input([['a','0,1']],labels);
    await assert.rejects(retina(options));assert.equal(await(await options.report.getFile()).text(),'old report');
  }
});
test('malformed logits, duplicate results, cancellation and uncommitted batches fail',async()=>{
  for(const logits of ['NaN,0','0,Inf','0','0,1,2','']) await assert.rejects(retina(input([['a',logits]],[['a',0]])));
  await assert.rejects(retina(input([['a','0,1'],['a','0,1']],[['a',1]])));
  await assert.rejects(retina(input([],[['a',0]])));
  const options=input([['a','0,1']],[['a',1]]);
  await assert.rejects(retina({...options,check(){throw new Error('cancelled');}}),/cancelled/);
  options.results.manifest=handle('{"state":"pending"}');await assert.rejects(retina(options),/not committed/);
  await assert.rejects(classify({...options,classes:3}),/Unsupported/);
});
