import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
import {browserHarness} from './browser-harness.mjs';
const root=path.resolve(process.env.LATTI_MODEL_NUMERIC_ROOT||'/tmp/latti-browser-model-numerics');
await mkdir(root,{recursive:true});
execFileSync(process.env.LATTI_MODEL_TRAINING_PYTHON||'python3',[path.join(import.meta.dirname,'model-numerics.py'),root],{stdio:['ignore','pipe','pipe']});
const harness=await browserHarness(root,{'/numeric/':root});
try {
 const runtime=path.resolve(process.env.LATTI_RUNTIME_DIRECTORY||path.join(import.meta.dirname,'../dist'));
 const original=await readFile(path.join(runtime,'model-worker.js'),'utf8');
 const code=`
import sys, json, h5py, numpy as np
from pathlib import Path
from training.nn_tools.export import export_h5_from_onnx
from training.nn_tools.feature_mat_h5 import export_feature_mat_h5_from_onnx
report = {}
for activation in ('relu', 'relu6', 'silu', 'gelu'):
    for kind in ('ordinary', 'feature_mat'):
        if activation == 'relu6' and kind == 'feature_mat':
            continue
        root = Path('/tmp/numeric', activation, kind)
        exporter = export_h5_from_onnx if kind == 'ordinary' else export_feature_mat_h5_from_onnx
        exporter(str(root/'source.onnx'), str(root/'ct.json'), str(root/'pyodide.h5'), verbose=False)
        with h5py.File(root/'old.h5') as expected, h5py.File(root/'pyodide.h5') as actual:
            assert set(expected) == set(actual)
            maximum = 0.0
            for name in actual:
                a, b = actual[name], expected[name]
                assert a.shape == b.shape and a.dtype == b.dtype == np.float64
                np.testing.assert_allclose(a[...], b[...], atol=1e-12, rtol=1e-12)
                maximum = max(maximum, float(np.max(np.abs(a[...] - b[...]))))
            report[activation + '/' + kind] = dict(max_abs_error=maximum, datasets=list(actual))
assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
json.dumps(dict(execution='actual Pyodide', torch_modules=[], atol=1e-12, rtol=1e-12, h5=report))
`;
 const branch=`if(message.type==='numericRegression') {
 for(const activation of ['relu','relu6','silu','gelu'])for(const kind of ['ordinary','feature_mat']) {
  if(activation==='relu6'&&kind==='feature_mat')continue;
  const root='/tmp/numeric/'+activation+'/'+kind;python.FS.mkdirTree(root);
  for(const name of ['source.onnx','ct.json','old.h5']) {
   const response=await fetch('/numeric/numeric-fixtures/'+activation+'/'+kind+'/'+name);
   if(!response.ok)throw new Error('Missing independent torch fixture');
   python.FS.writeFile(root+'/'+name,new Uint8Array(await response.arrayBuffer()));
  }
 }
 return await python.runPythonAsync(${JSON.stringify(code)});
}
`;
 await harness.page.route('**/model-worker.js',route=>route.fulfill({contentType:'text/javascript',body:original.replace("  if(message.type==='parameterValue')",branch+"  if(message.type==='parameterValue')")}));
 const result=await harness.page.evaluate(async()=>{
  const {BUILD_INFO}=await import('/dist/build-info.js'),worker=new Worker('/dist/model-worker.js',{type:'module'});
  const {cleanupPythonWorkspace}=await import('/dist/python-workspace.js'),jobId=crypto.randomUUID();
  try{return await new Promise((resolve,reject)=>{
   const timer=setTimeout(()=>reject(new Error('Numerical regression timeout')),120000);
   worker.onmessage=({data})=>{if(data.type==='result'){clearTimeout(timer);resolve(JSON.parse(data.result));}if(data.type==='error'){clearTimeout(timer);reject(new Error(data.message));}};
   worker.onerror=()=>reject(new Error('Numerical Worker failed'));
   worker.postMessage({type:'numericRegression',requestId:1,jobId,protocolVersion:BUILD_INFO.protocolVersion});
  });}finally{worker.terminate();await cleanupPythonWorkspace(jobId);}
 });
 await writeFile(path.join(root,'pyodide-comparison.json'),JSON.stringify(result,null,2));
 console.log(JSON.stringify(result,null,2));
}finally{await harness.close();await writeFile(path.join(root,'memory.json'),JSON.stringify(harness.rss,null,2));}
