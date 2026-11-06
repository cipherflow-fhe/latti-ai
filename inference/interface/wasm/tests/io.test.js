import test from 'node:test';
import assert from 'node:assert/strict';
import {cleanupAfter} from '../src/io.js';

test('cleanup drains every action while preserving the first operation error',async()=>{
  const original=Object.assign(new Error('write failed'),{code:'OPFS_QUOTA_EXCEEDED'}),visited=[];
  await cleanupAfter(original,[
    async()=>{visited.push(1);throw new Error('residual latti-jobs/job-a');},
    async()=>{visited.push(2);},
    async()=>{visited.push(3);throw new Error('residual external/job-a');},
  ]);
  assert.deepEqual(visited,[1,2,3]);
  assert.equal(original.code,'OPFS_QUOTA_EXCEEDED');
  assert.match(original.message,/^write failed; cleanup: .*latti-jobs\/job-a.*external\/job-a/);
  assert.equal(original.cleanupFailed,true);
});

test('cleanup failure after successful work is observable',async()=>{
  await assert.rejects(cleanupAfter(undefined,[async()=>{throw new Error('residual job-b');}]),
    error=>error.code==='OPFS_CLEANUP_FAILED'&&error.message.includes('residual job-b'));
});

test('native DOMException retains its identity and quota error when cleanup fails',async()=>{
  const original=new DOMException('disk full','QuotaExceededError');
  await cleanupAfter(original,[async()=>{throw new DOMException('directory locked','InvalidStateError');}]);
  assert.equal(original.name,'QuotaExceededError');
  assert.equal(original.code,22);
  assert.equal(original.cleanupFailed,true);
  assert.match(original.message,/^disk full; cleanup: directory locked$/);
});
