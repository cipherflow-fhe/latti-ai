import {browserHarness} from './browser-harness.mjs';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
const root=process.env.LATTI_WORKSPACE_TEST_ROOT||'/tmp/latti-python-workspace';await mkdir(root,{recursive:true});
const harness=await browserHarness(root);
const code=`
import os, json, h5py, numpy as np, onnx
from pathlib import Path
root=Path('/latti/jobs/check');root.mkdir()
f=root/'data'
with f.open('w+b') as stream:
    stream.write(b'abcdef');stream.seek(2);assert stream.read(2)==b'cd'
    stream.seek(1024*1024);stream.write(b'z');assert stream.tell()==1024*1024+1
    stream.seek(6);assert stream.read(32)==bytes(32)
    stream.truncate(3)
assert f.stat().st_size==3 and f.read_bytes()==b'abc'
with f.open('ab') as stream:stream.write(b'd')
assert f.read_bytes()==b'abcd'
with f.open('rb') as stream:
    f.unlink();assert stream.read()==b'abcd'
for index in range(300):
    f.write_bytes(b'temporary');f.unlink()
    np.arange(8,dtype=np.float64).tofile(f)
    np.testing.assert_array_equal(np.fromfile(f,dtype=np.float64),np.arange(8))
    f.unlink()
f.write_bytes(b'duplicate')
with f.open('rb') as stream:
    descriptor=os.dup(stream.fileno());f.unlink()
    assert os.read(descriptor,9)==b'duplicate';os.close(descriptor)
f.write_bytes(b'replaced')
g=root/'other';g.write_bytes(b'new')
with f.open('rb') as stream:
    g.replace(f);assert stream.read()==b'replaced'
assert f.read_bytes()==b'new'
with h5py.File(root/'weights.h5','w') as h:
    h['matrix']=np.arange(32768,dtype=np.float64).reshape(128,256)
with h5py.File(root/'weights.h5','r') as h:
    np.testing.assert_array_equal(h['matrix'][3:5],np.arange(32768).reshape(128,256)[3:5])
m=onnx.helper.make_model(onnx.helper.make_graph([], 'test', [], []))
onnx.save(m,str(root/'model.onnx'));assert onnx.load(str(root/'model.onnx'))==m
json.dumps({'status':'passed','checks':['seek','sparse','truncate','append','unlink-open','dup','rename-open','recycle','numpy-tofile','h5','onnx']})
`;
try {
 const worker=await readFile(new URL('../dist/model-worker.js',import.meta.url),'utf8');
 await harness.page.route('**/model-worker.js',route=>route.fulfill({contentType:'text/javascript',body:worker.replace("  if(message.type==='parameterValue')",`  if(message.type==='prepareSplit') {
 if(!message.resume) {
  const result=JSON.parse(await python.runPythonAsync(${JSON.stringify(code)}));
  python.FS.writeFile('/latti/jobs/check/result.json',JSON.stringify(result));
  return checkpoint('workspace-test');
 }
 await python.runPythonAsync("import h5py, onnx\\nwith h5py.File('/latti/jobs/check/weights.h5','r') as h: assert h['matrix'][3,0] == 768\\nassert Path('/latti/jobs/check/data').read_bytes() == b'new'\\nassert onnx.load('/latti/jobs/check/model.onnx').graph.name == 'test'");
 const result=JSON.parse(python.FS.readFile('/latti/jobs/check/result.json',{encoding:'utf8'}));
 result.checks.push('worker-restart','restored-h5','restored-onnx');return result;
 }\n  if(message.type==='parameterValue')`)}));
 const result=await harness.page.evaluate(async()=>{
  const {openBrowserModelTools}=await import('/dist/index.js');const tools=await openBrowserModelTools();
  try {
   const result=await tools.prepareSplit({});
   const root=await navigator.storage.getDirectory(),cache=await root.getDirectoryHandle('latti-python-workspaces');
   const entries=await Array.fromAsync(cache.keys());if(entries.length)throw new Error('Workspace retained: '+entries);
   return {...result,workspaceEntries:entries};
  }finally{await tools.close();}
 });
 await writeFile(root+'/checks.json',JSON.stringify(result,null,2));console.log(result);
}finally{await harness.close();}
