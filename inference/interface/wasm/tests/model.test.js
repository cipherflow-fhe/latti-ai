import test from 'node:test';
import assert from 'node:assert/strict';
import {openBrowserModelTools} from '../src/model.js';

test('model operations serialize; cancellation rebuilds the Worker and settles once',async()=>{
  const original={Worker:globalThis.Worker,secure:globalThis.isSecureContext};
  const workers=[],removed=[];
  globalThis.isSecureContext=true;
  globalThis.Worker=class {
    constructor(){workers.push(this);}
    postMessage(message){this.message=message;}
    terminate(){this.terminated=true;}
    result(result){this.onmessage({data:{...this.message,type:'result',result}});}
  };
  const directory={async getDirectoryHandle(){return {async getFileHandle(){throw Object.assign(new Error('missing'),{name:'NotFoundError'});}};},async removeEntry(name){removed.push(name);}};
  const tick=()=>new Promise(resolve=>setImmediate(resolve));
  try {
    const tools=await openBrowserModelTools();
    const first=tools.prepareSplit({outputDirectory:directory});
    const rejected=assert.rejects(first,error=>error.code==='CANCELLED');
    const second=tools.compileHighLevel({outputDirectory:directory});
    await tick();assert.equal(workers.length,1);
    assert.deepEqual(await tools.cancel(999),{accepted:false});
    assert.deepEqual(await tools.cancel(),{accepted:true});await rejected;await tick();
    assert.equal(workers[0].terminated,true);assert.equal(workers.length,2);
    workers[0].result('stale');workers[1].result('compiled');
    assert.equal(await second,'compiled');assert.equal(removed.length,1);
    assert.deepEqual(await tools.cancel(),{accepted:false});await tools.close();
    await assert.rejects(tools.prepareSplit({outputDirectory:directory}),/closed/);
  }finally{globalThis.Worker=original.Worker;globalThis.isSecureContext=original.secure;}
});

test('closing while a destination opens cleans only the uncommitted operation',async()=>{
  const original=globalThis.isSecureContext;globalThis.isSecureContext=true;
  let release,removed=false;
  const directory={getDirectoryHandle:()=>new Promise(resolve=>{release=resolve;}),async removeEntry(){removed=true;}};
  try {
    const tools=await openBrowserModelTools();
    const pending=tools.prepareSplit({outputDirectory:directory});
    const rejected=assert.rejects(pending,/closed/);
    await new Promise(resolve=>setImmediate(resolve));
    const closed=tools.close();release({});await closed;await rejected;assert.equal(removed,true);
  }finally{globalThis.isSecureContext=original;}
});
