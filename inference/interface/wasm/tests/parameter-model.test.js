import test from 'node:test';
import assert from 'node:assert/strict';
import {BUILD_INFO} from '../src/build-info.js';
import {openParameterModel} from '../src/parameter-model.js';

test('active parameter requests can span thirty minutes; an idle producer still closes',async t=>{
  t.mock.timers.enable({apis:['setTimeout']});
  const originalWorker=globalThis.Worker;
  let terminated=false;
  globalThis.Worker=class {
    postMessage(message) {
      const result=message.type==='parameterPlan'?{runtimeMetrics:{pyodideMemoryBytes:1}}:
        {bytes:new Uint8Array([7]),runtimeMetrics:{pyodideMemoryBytes:2}};
      queueMicrotask(()=>this.onmessage({data:{protocolVersion:BUILD_INFO.protocolVersion,
        jobId:message.jobId,requestId:message.requestId,type:'result',result}}));
    }
    terminate(){terminated=true;}
  };
  t.after(()=>{if(originalWorker===undefined)delete globalThis.Worker;else globalThis.Worker=originalWorker;});
  const controller=new AbortController();
  const producer=await openParameterModel({},null,controller.signal);
  for(let index=0;index<4;index++) {
    t.mock.timers.tick(10*60*1000);
    assert.deepEqual(await producer.value('parameter',index),new Uint8Array([7]));
    assert.equal(terminated,false);
  }
  assert.equal(producer.metadata.runtimeMetrics.pyodideMemoryBytes,2);
  t.mock.timers.tick(30*60*1000);
  assert.equal(terminated,true);
  await assert.rejects(producer.value('parameter',4),/closed/);
  await producer.close();
});
