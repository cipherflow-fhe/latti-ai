import test from 'node:test';
import assert from 'node:assert/strict';
import {runnerOutputs} from '../src/runner-output.js';

const u64=value=>{const bytes=new Uint8Array(8);new DataView(bytes.buffer).setBigUint64(0,BigInt(value),true);return bytes;};
const string=value=>{const bytes=new TextEncoder().encode(value);return [u64(bytes.byteLength),bytes];};
const bundle=(entries,tail=[])=>new File([
  ...string('latti-ai.runner-output.v1'),u64(entries.length),
  ...entries.flatMap(([name,data])=>[...string(name),u64(data.byteLength),data]),...tail,
],'output.ct');

test('runner outputs preserve raw Feature wire including values across chunks',async()=>{
  const wire=new Uint8Array(4*1024*1024+17);wire[0]=19;wire[wire.length-1]=247;
  const other=new Uint8Array([4,5,6]);
  const result=await runnerOutputs(bundle([['second',wire],['first',other]]),['first','second']);
  assert.deepEqual(new Uint8Array(await result.get('second').arrayBuffer()),wire);
  assert.deepEqual(new Uint8Array(await result.get('first').arrayBuffer()),other);
});
test('runner outputs reject truncation, invalid lengths, duplicate and unknown IDs and tails',async()=>{
  const data=new Uint8Array([1,2,3]),valid=bundle([['output',data]]);
  for(let end=0;end<valid.size;end++)await assert.rejects(()=>runnerOutputs(valid.slice(0,end),['output']));
  await assert.rejects(()=>runnerOutputs(bundle([['output',data]], [u64(0)]),['output']),/Trailing/);
  await assert.rejects(()=>runnerOutputs(bundle([['unknown',data]]),['output']),/Unknown/);
  await assert.rejects(()=>runnerOutputs(bundle([['output',data],['output',data]]),['output','other']),/duplicate/);
  await assert.rejects(()=>runnerOutputs(bundle([['output',new Uint8Array()]]),['output']),/Empty/);
  await assert.rejects(()=>runnerOutputs(new File([...string('latti-ai.runner-output.v1'),u64(1),...string('output'),u64(2n**63n)],'invalid'),['output']),/Truncated/);
});
