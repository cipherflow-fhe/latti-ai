import {csvChunks} from '../../csv.js';
import {safeName} from '../../io.js';

// Segmentation and clinical features are prepared offline by prepare.py.
// Validate those outputs without loading an entire image tensor into memory.
export async function preprocessDirectory({inputDirectory,task,check=()=>{}}) {
  const ids=['input_0','input_1','input_2'];
  if(task.task_input_id.length!==3||ids.some(id=>!task.task_input_id.includes(id))) throw new Error('Expected Retina three-input task');
  const manifest=await(await inputDirectory.getFileHandle('cases.json')).getFile();
  if(manifest.size>16*1024*1024) throw new Error('METADATA_BUDGET_EXCEEDED');
  const cases=JSON.parse(await manifest.text()),seen=new Set(),inputDirectories={};
  if(!cases.length||cases.length>100000) throw new Error('Invalid Retina case count');
  for(const id of ids)inputDirectories[id]=await inputDirectory.getDirectoryHandle(id);
  for(const record of cases) {
    check();const name=safeName(record.case_id),key=name.normalize('NFC').toLowerCase();
    if(seen.has(key)) throw new Error('Duplicate Retina case: '+name);seen.add(key);
    for(const id of ids) {
      if(record.inputs[id]!==id+'/'+name+'.csv') throw new Error('Retina case/input mismatch: '+name+'/'+id);
      const feature=task.task_input_param[id],expected=feature.channel*(feature.shape||[]).reduce((a,b)=>a*b,1);
      const file=await(await inputDirectories[id].getFileHandle(name+'.csv')).getFile();let count=0;
      for await(const block of csvChunks(file)) {check();count+=block.length;if(count>expected)break;}
      if(count!==expected) throw new Error('Retina shape mismatch: '+name+'/'+id);
    }
  }
  return {count:String(cases.length),inputDirectories};
}
