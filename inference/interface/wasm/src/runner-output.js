import {fileOf} from './io.js';

/** Parse only framing; Feature payloads remain exact bounded File slices. */
export async function runnerOutputs(source,expectedIDs) {
  const file=await fileOf(source),size=BigInt(file.size),expected=new Set(expectedIDs),outputs=new Map();
  if(expected.size!==expectedIDs.length)throw new Error('Duplicate expected output ID');
  let offset=0n;
  const bytes=async length=>{
    if(length<0n||offset>size||length>size-offset)throw new Error('Truncated runner output');
    const end=offset+length;
    if(end>BigInt(Number.MAX_SAFE_INTEGER))throw new Error('Runner output offset exceeds File range');
    const result=file.slice(Number(offset),Number(end));offset=end;return result;
  };
  const uint64=async()=>new DataView(await(await bytes(8n)).arrayBuffer()).getBigUint64(0,true);
  const string=async()=>{
    const length=await uint64();
    if(length>65536n)throw new Error('Runner output name is too long');
    return new TextDecoder('utf-8',{fatal:true}).decode(await(await bytes(length)).arrayBuffer());
  };
  if(await string()!=='latti-ai.runner-output.v1')throw new Error('Invalid runner output format');
  const count=await uint64();
  if(count!==BigInt(expected.size))throw new Error('Runner output ID count differs');
  for(let i=0n;i<count;i++) {
    const id=await string();
    if(!expected.has(id)||outputs.has(id))throw new Error('Unknown or duplicate runner output ID');
    const length=await uint64();
    if(length===0n)throw new Error('Empty runner Feature output');
    outputs.set(id,await bytes(length));
  }
  if(offset!==size)throw new Error('Trailing runner output data');
  return outputs;
}
