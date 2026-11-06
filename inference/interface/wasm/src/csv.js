const NUMBER = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;

// Parse one bounded numeric block at a time, including fields split across
// input chunks. A trailing comma matches the compiler's example CSV files.
export async function* csvChunks(file) {
  const reader=file.stream().getReader(), decoder=new TextDecoder('utf-8',{fatal:true});
  let field='', values=[], count=0;
  const flush=()=>{
    const text=field.trim();
    if(!NUMBER.test(text)) throw new Error('Invalid CSV numeric field');
    const value=Number(text);
    if(!Number.isFinite(value)) throw new Error('Non-finite CSV value');
    values.push(value);field='';count++;
  };
  try {
    let done=false;
    while(!done) {
      const result=await reader.read();done=result.done;
      const text=decoder.decode(result.value,{stream:!done});
      for(const character of text) {
        if(character===',') flush();
        else if(character==='\r'||character==='\n') {if(field.trim()) flush();}
        else {
          field+=character;
          if(field.length>128) throw new Error('CSV numeric field is too long');
        }
        if(values.length===8192) {yield values;values=[];}
      }
    }
    if(field.trim()) flush();
    if(!count) throw new Error('Empty CSV input');
    if(values.length) yield values;
  } finally { await reader.cancel().catch(()=>{});reader.releaseLock(); }
}

export async function stageCSV(file,handle,check=()=>{}) {
  const writer=await handle.createWritable();let bytes=0n;
  try {
    for await(const values of csvChunks(file)) {
      check();
      const block=new Uint8Array(values.length*8), view=new DataView(block.buffer);
      values.forEach((value,index)=>view.setFloat64(index*8,value,true));
      await writer.write(block);bytes+=BigInt(block.byteLength);
    }
    await writer.close();return bytes;
  } catch(error) {await writer.abort().catch(()=>{});throw error;}
}

export async function numericToCSV(file,handle,check=()=>{}) {
  if(file.size%8) throw new Error('Invalid numeric file size');
  const writer=await handle.createWritable();
  try {
    for(let offset=0;offset<file.size;offset+=65536) {
      check();
      const bytes=await file.slice(offset,offset+65536).arrayBuffer(),view=new DataView(bytes),fields=[];
      for(let at=0;at<bytes.byteLength;at+=8) {
        const value=view.getFloat64(at,true);
        if(!Number.isFinite(value)) throw new Error('Non-finite plaintext output');
        fields.push(String(value));
      }
      await writer.write((offset?',':'')+fields.join(','));
    }
    await writer.write('\n');await writer.close();
  } catch(error) {await writer.abort().catch(()=>{});throw error;}
}
