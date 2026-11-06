import {csvChunks} from '../csv.js';
import {safeName} from '../io.js';

async function* lines(file) {
  const reader=file.stream().pipeThrough(new TextDecoderStream('utf-8',{fatal:true})).getReader();let pending='',bytes=0;
  try {
    for(;;) {
      const {value,done}=await reader.read();if(done)break;
      bytes+=value.length*2;if(bytes>64*1024*1024) throw new Error('METADATA_BUDGET_EXCEEDED');
      pending+=value;let end;
      while((end=pending.indexOf('\n'))>=0) {yield pending.slice(0,end).replace(/\r$/,'');pending=pending.slice(end+1);}
      if(pending.length>128*1024) throw new Error('Metadata line too long');
    }
    if(pending)yield pending.replace(/\r$/,'');
  } finally {await reader.cancel().catch(()=>{});reader.releaseLock();}
}

export function argmax(logits,classes) {
  if(logits.length!==classes||logits.some(value=>!Number.isFinite(value))) throw new Error(`Expected ${classes} finite logits`);
  let prediction=0;
  for(let i=1;i<classes;i++) if(logits[i]>logits[prediction]) prediction=i;
  return prediction;
}

// Labels are an independent UTF-8 JSONL table: {"case_id":"001","label":0}.
// The logits BatchResult's manifest is the only source of result file names.
export async function classify({results,labels,report,classes,check=()=>{}}) {
  if(classes!==2&&classes!==10) throw new Error('Unsupported classification task');
  const table=new Map(),names=new Set();
  for await(const line of lines(labels)) {
    check();const {case_id:id,label}=JSON.parse(line);safeName(id);
    const key=id.normalize('NFC').toLowerCase();
    if(names.has(key)||!Number.isInteger(label)||label<0||label>=classes) throw new Error('Invalid or duplicate label: '+id);
    names.add(key);table.set(id,label);
    if(table.size>100000) throw new Error('METADATA_BUDGET_EXCEEDED');
  }
  if(!table.size) throw new Error('Empty labels');
  const metadata=await results.manifest.getFile();
  if(metadata.size>1024*1024) throw new Error('METADATA_BUDGET_EXCEEDED');
  const manifest=JSON.parse(await metadata.text());
  if(manifest.state!=='committed') throw new Error('Results are not committed');
  const entries=await(await results.directory.getFileHandle(safeName(manifest.entries))).getFile(),seen=new Set();
  const writer=await report.createWritable();let total=0,correct=0;
  try {
    for await(const line of lines(entries)) {
      check();const record=JSON.parse(line),id=safeName(record.case_id),name=safeName(record.path);
      if(seen.has(id)||!table.has(id)||record.role!=='plaintext') throw new Error('Missing label or duplicate result: '+id);
      seen.add(id);
      const file=await(await results.directory.getFileHandle(name)).getFile(),logits=[];
      if(String(file.size)!==record.bytes) throw new Error('Result size mismatch: '+id);
      for await(const block of csvChunks(file)) {logits.push(...block);if(logits.length>classes) throw new Error('Too many logits: '+id);}
      const prediction=argmax(logits,classes),label=table.get(id);total++;if(prediction===label)correct++;
      await writer.write(JSON.stringify({case_id:id,logits,prediction,label,correct:prediction===label})+'\n');
    }
    if(!total||seen.size!==table.size||String(total)!==manifest.count) throw new Error('Label/result case sets differ');
    check();await writer.close();
    return {total,correct,accuracy:correct/total};
  } catch(error) {await writer.abort().catch(()=>{});throw error;}
}
