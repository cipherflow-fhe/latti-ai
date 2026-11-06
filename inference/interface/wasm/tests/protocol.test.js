import {test} from 'node:test';
import assert from 'node:assert/strict';
import {openBrowserClient} from '../src/index.js';
import {BUILD_INFO} from '../src/build-info.js';

class FakeWorker {
  static ready = () => ({...BUILD_INFO,type:'ready',requestId:0,jobId:'session',capabilities:{protocolVersion:2}});
  static handle = (worker,request) => worker.reply(request);
  static instance;
  constructor() {
    FakeWorker.instance=this; this.sent=[]; this.terminated=false;
    queueMicrotask(()=>this.onmessage?.({data:FakeWorker.ready()}));
  }
  postMessage(request) {this.sent.push(request);queueMicrotask(()=>FakeWorker.handle(this,request));}
  reply(request, fields={}) {this.onmessage({data:{protocolVersion:2,requestId:request.requestId,jobId:request.jobId,type:'result',...fields}});}
  terminate() {this.terminated=true;}
}
const directory={kind:'directory'};
const open=()=>openBrowserClient({keyDirectory:directory});

test('protocol v2 validates startup and settles pending requests', async t=>{
  const originalWorker=globalThis.Worker, secure=globalThis.isSecureContext;
  globalThis.Worker=FakeWorker;globalThis.isSecureContext=true;
  t.after(()=>{globalThis.Worker=originalWorker;globalThis.isSecureContext=secure;});
  const reset=()=>{
    FakeWorker.ready=()=>({...BUILD_INFO,type:'ready',requestId:0,jobId:'session',capabilities:{protocolVersion:2}});
    FakeWorker.handle=(worker,request)=>worker.reply(request);
  };
  await t.test('requests use protected envelopes and close is idempotent',async()=>{
    reset();const client=await open();
    await client.encrypt({type:'bad',requestId:'bad',jobId:'bad',protocolVersion:1});
    const worker=FakeWorker.instance, request=worker.sent.at(-1);
    assert.equal(request.type,'encrypt');assert.equal(request.protocolVersion,2);
    assert.notEqual(request.jobId,'bad');assert.notEqual(request.requestId,'bad');
    await Promise.all([client.close(),client.close()]);
    assert.equal(worker.sent.filter(r=>r.type==='close').length,1);assert.equal(worker.terminated,true);
  });
  for(const override of [{runtimeVersion:'0.1.2'},{build:'old-build'},{protocolVersion:1}]) {
    await t.test(`rejects mixed runtime ${JSON.stringify(override)}`,async()=>{
      reset();const ready=FakeWorker.ready;FakeWorker.ready=()=>({...ready(),...override});
      await assert.rejects(open(),/MISMATCH|protocol/);assert.equal(FakeWorker.instance.terminated,true);
    });
  }
  await t.test('startup error is terminal for open',async()=>{
    reset();FakeWorker.ready=()=>({protocolVersion:2,requestId:0,jobId:'session',type:'error',message:'WASM_FETCH_FAILED_404'});
    await assert.rejects(open(),/WASM_FETCH_FAILED_404/);assert.equal(FakeWorker.instance.terminated,true);
  });
  for(const trigger of ['onerror','onmessageerror','runtime-exit','wrong-job']) {
    await t.test(`${trigger} settles active and queued requests`,async()=>{
      reset();const client=await open();
      FakeWorker.handle=(worker,request)=>{
        if(trigger==='runtime-exit') worker.onmessage({data:{protocolVersion:2,requestId:0,jobId:'session',type:'error',message:'runtime exited'}});
        else if(trigger==='wrong-job') worker.reply(request,{jobId:'unmatched'});
        else worker[trigger]({});
      };
      const active=client.setup(), queued=client.restore();
      const result=await Promise.allSettled([active,queued]);
      assert.deepEqual(result.map(r=>r.status),['rejected','rejected']);
      assert.equal(FakeWorker.instance.terminated,true);
      await client.close();
    });
  }
  await t.test('a duplicate terminal cannot settle a later request',async()=>{
    reset();const client=await open();
    FakeWorker.handle=(worker,request)=>{worker.reply(request,{result:request.type});worker.reply(request,{result:'duplicate'});};
    assert.equal(await client.setup(),'setup');assert.equal(await client.restore(),'restore');await client.close();
  });
  await t.test('close rejects work which has not started',async()=>{
    reset();const client=await open();
    const queued=client.setup();const closing=client.close();
    await assert.rejects(queued,/closed/);await closing;
    assert.equal(FakeWorker.instance.sent.some(r=>r.type==='setup'),false);
  });
});
