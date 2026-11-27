import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
import {browserHarness} from './browser-harness.mjs';

const root=path.resolve(process.env.LATTI_PREFIX_EVIDENCE||'/tmp/latti-prefix-browser');
await mkdir(root,{recursive:true});
execFileSync(process.env.LATTI_MODEL_NATIVE_PYTHON||'python3',[new URL('./prefix-export.py',import.meta.url).pathname,path.join(root,'fixtures')]);
const harness=await browserHarness(root,{'/fixtures/':path.join(root,'fixtures/private-source')});
const runtime=process.env.LATTI_RUNTIME_DIRECTORY||new URL('../dist/',import.meta.url).pathname;
try {
  const worker=await readFile(path.join(runtime,'model-worker.js'),'utf8');
  // Node weights must be fed from bounded reads; prohibit the previous whole-
  // external-file route even when a small fixture would fit in memory.
  await harness.page.route('**/model-worker.js',route=>route.fulfill({contentType:'text/javascript',body:worker.replace(
    'workspace ||= await mountPythonWorkspace(python.FS,message.jobId);',
    `workspace ||= await mountPythonWorkspace(python.FS,message.jobId);
    const readFile=python.FS.readFile;
    python.FS.readFile=(name,...args)=>{if(String(name).endsWith('.data'))throw new Error('Whole weight file read forbidden');return readFile(name,...args);};`)}));
  const result=await harness.page.evaluate(async()=>{
    const {openBrowserModelTools}=await import('/dist/index.js');
    const opfs=await navigator.storage.getDirectory();
    const write=async(directory,name,data)=>{const file=await directory.getFileHandle(name,{create:true}),writer=await file.createWritable();await writer.write(data);await writer.close();return file;};
    const source=await opfs.getDirectoryHandle('private-source',{create:true});
    for(const name of ['model.onnx','weights.data','input.csv','inputs.json'])await write(source,name,await(await fetch('/fixtures/'+name)).arrayBuffer());
    const tools=await openBrowserModelTools();
    const reject=async(action,pattern)=>{try{await action();}catch(error){if(pattern.test(error.message))return error.message;throw error;}throw new Error('Invalid prefix operation accepted');};
    try {
      const split=await tools.prepareSplit({source:{directory:source,modelPath:'model.onnx',inputsManifest:await source.getFileHandle('inputs.json')},ratio:'0.5'});
      await reject(()=>tools.runPrefix({split}),/independent prefix/);
      const prefix=await tools.exportPrefix({split});
      const metadata=JSON.parse(await(await prefix.manifest.getFile()).text());
      const files=Object.keys(metadata.files);
      if(files.length!==3||files.some(name=>!['model.onnx','compact.onnx.data','prefix.json'].includes(name)))throw new Error('Prefix includes private material');
      const modelParent=await opfs.getDirectoryHandle('latti-model');
      await modelParent.removeEntry(split.directory.name,{recursive:true});
      await opfs.removeEntry('private-source',{recursive:true});
      const inputs=await opfs.getDirectoryHandle('csv-only',{create:true});
      await write(inputs,'input.csv',Array.from({length:512},(_,i)=>i+0.25).join(','));
      const binding={inputs:[{name:'input',csv:'input.csv',shape:[1,512],axes:['N','F']}]};
      const run=await tools.runPrefix({prefix,inputDirectory:inputs,inputsManifest:new File([JSON.stringify(binding)],'inputs.json')});
      const runMeta=JSON.parse(await(await run.manifest.getFile()).text());
      if(Object.keys(runMeta.files).some(name=>!name.startsWith('run/')))throw new Error('Run includes private inputs or models');
      const output=await run.directory.getDirectoryHandle('run');
      const sample=JSON.parse(await(await(await output.getFileHandle('sample_manifest.json')).getFile()).text());
      const actual=(await(await(await output.getFileHandle(sample.inputs[0].csv)).getFile()).text()).trim().split(',').map(Number);
      if(actual.some((value,index)=>value!==index+1.25))throw new Error('Independent prefix numerics differ');
      const report=JSON.parse(await(await(await output.getFileHandle('run_manifest.json')).getFile()).text());
      if(report.weightLoading!=='on_demand'||report.weightReads.largestTensorBytes!==2048)throw new Error('Unbounded weight read');
      const batchBinding={samples:['case-a','case-b'].map(case_id=>({case_id,inputs:binding.inputs}))};
      const batchRun=await tools.runPrefix({prefix,inputDirectory:inputs,inputsManifest:new File([JSON.stringify(batchBinding)],'inputs.json')});
      const batchDirectory=await batchRun.directory.getDirectoryHandle('run');
      const batchReport=JSON.parse(await(await(await batchDirectory.getFileHandle('run_manifest.json')).getFile()).text());
      if(batchReport.count!==2||batchReport.samples.map(sample=>sample.case_id).join(',')!=='case-a,case-b')throw new Error('Batch sample coverage differs');
      for(const record of batchReport.samples) {
        const current=await(await batchDirectory.getDirectoryHandle('cases')).getDirectoryHandle(record.case_id);
        const sample=JSON.parse(await(await(await current.getFileHandle('sample_manifest.json')).getFile()).text());
        const values=(await(await(await current.getFileHandle(sample.inputs[0].csv)).getFile()).text()).trim().split(',').map(Number);
        if(values.some((value,index)=>value!==index+1.25))throw new Error('Batch prefix output differs');
      }
      const bad=structuredClone(binding);bad.inputs[0].axes=['C','N'];
      await reject(()=>tools.runPrefix({prefix,inputDirectory:inputs,inputsManifest:new File([JSON.stringify(bad)],'inputs.json')}),/shape\/axes/);
      const data=await prefix.directory.getFileHandle('compact.onnx.data'),original=await data.getFile();
      await write(prefix.directory,data.name,new Uint8Array(original.size));
      await reject(()=>tools.runPrefix({prefix,inputDirectory:inputs,inputsManifest:new File([JSON.stringify(binding)],'inputs.json')}),/hash mismatch/);
      return {status:'passed',files,originalModelAndSplitDeleted:true,wholeWeightFileReadsForbidden:true,weightReads:report.weightReads,outputElements:actual.length,batchCount:batchReport.count};
    }finally{await tools.close();}
  });
  await writeFile(path.join(root,'report.json'),JSON.stringify(result,null,2)+'\n');
  console.log(JSON.stringify(result));
}finally{await harness.close();}
