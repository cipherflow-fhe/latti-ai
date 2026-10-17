import './wasm_exec.js';
import { readCSV } from './csv.js';
import { makeZip, readZip } from './zip.js';

const presetName='N16QP1546H192H32';
const fixedParameters=JSON.stringify({param0:{poly_modulus_degree:65536,n_mult_level:9,log_default_scale:40,param_name:presetName}});
let directory, keyDirectory, taskText, parameterText, ready=false, go;
const runtimeReady = (async () => {
  go = new Go();
  const {instance} = await WebAssembly.instantiateStreaming(fetch(new URL('./latti_client.wasm',import.meta.url)),go.importObject);
  go.run(instance).catch(()=>{});
  while (typeof globalThis.lattiCore !== 'function') await new Promise(resolve => setTimeout(resolve, 0));
})();
const core = (operation,...args) => {
  const result = globalThis.lattiCore(operation,...args);
  if (result.error) throw new Error(result.error);
  return result;
};
const progress = (phase,start,bytes,result={}) => postMessage({type:'progress',progress:{
  phase,bytes,elapsedMs:performance.now()-start,wasmBytes:go.mem.buffer.byteLength,
  heapBytes:result.heapBytes || 0,payloadBytes:result.payloadBytes || 0,
  chunkBytes:result.data?.chunk?.byteLength || 0,inFlightChunks:result.data?.chunk ? 1 : 0,
}});
const fileOf = async value => value.kind === 'file' ? value.getFile() : value;
const read = async (dir,name) => (await (await dir.getFileHandle(name)).getFile()).text();
const initialize = () => taskText ? core('init',taskText,parameterText) : core('init-keys',parameterText);
const contextDirectory = (name,create=false) => keyDirectory ? keyDirectory.getDirectoryHandle(name,{create}) : directory;
const validName = name => {
  if (typeof name !== 'string' || !/^[A-Za-z0-9._-]+$/.test(name) || name.includes('..')) throw new Error('Invalid task name');
  return name;
};
const outputName = (source,extension,suffix) => {
  const name=source.replace(extension,'')+suffix;
  if (!name || /[\/\\\0]/.test(name) || name==='.' || name==='..') throw new Error('Invalid output file name');
  return name;
};
async function writeFile(handle,data) {
  const writer = await handle.createWritable();
  try { await writer.write(data); await writer.close(); }
  catch (error) { await writer.abort().catch(()=>{}); throw error; }
}
const csvFiles = async directory => {
  const files = [];
  for await (const file of directory.values()) if (file.kind === 'file' && /\.csv$/i.test(file.name)) files.push(file);
  files.sort((a,b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0);
  return files;
};
const caseID = file => {
  const name = file.name.replace(/\.csv$/i, '');
  if (!/^[A-Za-z0-9._-]+$/.test(name) || name === '.' || name === '..') throw new Error('Invalid case name: ' + file.name);
  return name;
};
const outputFile = async options => {
  if (options.outputFile) return options.outputFile;
  if (!options.outputDirectory) throw new Error('Server mode requires outputFile or outputDirectory');
  return options.outputDirectory.getFileHandle(options.outputName || 'input.zip', {create: true});
};
async function encryptServer(options, start) {
  const directories = options.inputDirectories;
  if (!directories || typeof directories !== 'object') throw new Error('Server mode requires inputDirectories');
  const inputNames = Object.keys(directories).sort();
  if (!inputNames.length) throw new Error('Server mode requires at least one input');
  const filesByInput = new Map(), cases = new Map();
  for (const inputName of inputNames) {
    validName(inputName);
    const files = await csvFiles(directories[inputName]);
    if (!files.length) throw new Error(inputName + ': no CSV inputs found');
    const byCase = new Map();
    for (const file of files) {
      const name = caseID(file), key = name.toLowerCase();
      if (byCase.has(key)) throw new Error(inputName + ': duplicate case ' + name);
      byCase.set(key, {name, file});
      if (!cases.has(key)) cases.set(key, name);
    }
    filesByInput.set(inputName, byCase);
  }
  for (const [key, name] of cases) for (const inputName of inputNames)
    if (!filesByInput.get(inputName).has(key)) throw new Error('Case ' + name + ' is missing input ' + inputName);
  const entries = [], orderedCases = [...cases.values()].sort((a,b) => a < b ? -1 : a > b ? 1 : 0);
  for (const name of orderedCases) {
    const key = name.toLowerCase();
    for (const inputName of inputNames) {
      const file = filesByInput.get(inputName).get(key).file;
      const result = core('encrypt', inputName, await readCSV(await fileOf(file)));
      const entryName = inputNames.length === 1 ? name + '.ct' : name + '__' + inputName + '.ct';
      entries.push({name: entryName, data: result.data});
      progress('encrypt', start, result.data.byteLength, result);
    }
  }
  const target = await outputFile(options);
  await writeFile(target, makeZip(entries));
  return target;
}
async function decryptServer(options, start) {
  const name = validName(options.outputName);
  if (!options.inputFile) throw new Error('Server mode requires inputFile');
  const file = await fileOf(options.inputFile), entries = await readZip(await file.arrayBuffer());
  const results = [];
  for (const [entryName, data] of entries) {
    if (!entryName.endsWith('.result.bin') || entryName.includes('/')) throw new Error('Invalid server result entry: ' + entryName);
    const caseName = entryName.slice(0, -'.result.bin'.length);
    if (!/^[A-Za-z0-9._-]+$/.test(caseName) || caseName === '.' || caseName === '..') throw new Error('Invalid server result case: ' + entryName);
    results.push({caseName, data});
  }
  if (!results.length) throw new Error('Server result archive contains no result entries');
  results.sort((a,b) => a.caseName < b.caseName ? -1 : a.caseName > b.caseName ? 1 : 0);
  const seen = new Set(), outputs = [];
  for (const resultEntry of results) {
    const key = resultEntry.caseName.toLowerCase();
    if (seen.has(key)) throw new Error('Duplicate server result case: ' + resultEntry.caseName);
    seen.add(key);
    const result = core('decrypt', name, resultEntry.data);
    const output = await options.outputDirectory.getFileHandle(resultEntry.caseName + '.csv', {create: true});
    await writeFile(output, Array.from(result.data).join(',') + '\n');
    outputs.push(output);
    progress('decrypt', start, resultEntry.data.byteLength, result);
  }
  return outputs;
}
async function operate({operation,...options}) {
  await runtimeReady;
  const start=performance.now();
  if (operation === 'open') {
    directory=options.clientDirectory;
    keyDirectory=options.keyDirectory;
    if (keyDirectory && keyDirectory.name !== presetName) keyDirectory=await keyDirectory.getDirectoryHandle(presetName,{create:true});
    [taskText,parameterText]=await Promise.all([
      directory ? read(directory,'task_config.json') : undefined,
      keyDirectory ? fixedParameters : read(directory,'ckks_parameter.json'),
    ]);
    if (directory && !taskText) throw new Error('Task configuration is required');
    initialize();
    return;
  }
  if (operation === 'setup') {
    const secret=await contextDirectory('secret',true);
    const publicDir=await contextDirectory('public',true);
    for (const [dir,name] of [[secret,'full_context.bin'],...(keyDirectory ? [[publicDir,'eval_context.bin']] : [])]) {
      let exists=false;
      try { await dir.getFileHandle(name); exists=true; }
      catch (error) { if (error.name !== 'NotFoundError') throw error; }
      if (exists) throw new Error(name+' exists; restore it or select a new directory');
    }
    ready=false;
    const evalContext=await publicDir.getFileHandle('eval_context.bin',{create:true});
    const writer=await evalContext.createWritable();
    let bytes=0;
    try {
      core('begin');
      for (;;) {
        const result=core('next');
        if (result.data.done) { progress('generation',start,bytes,result); break; }
        await writer.write(result.data.chunk);
        bytes+=result.data.chunk.byteLength;
        progress('generation',start,bytes,result);
      }
      await writer.close();
    } catch (error) {
      await writer.abort().catch(()=>{});
      initialize(); throw error;
    }
    const result=core('full');
    const fullContext=await secret.getFileHandle('full_context.bin',{create:true});
    await writeFile(fullContext,result.data);
    ready=true;
    progress('full-export',start,result.data.byteLength,result);
    return {evalContext,fullContext};
  }
  if (operation === 'restore') {
    const secret=await contextDirectory('secret');
    const file=await (await secret.getFileHandle('full_context.bin')).getFile();
    const result=core('restore',new Uint8Array(await file.arrayBuffer()));
    ready=true;progress('restore',start,file.size,result);return;
  }
  if (operation === 'close') { core('close');ready=false;return; }
  if (!ready) throw new Error('setup or restore is required');
  if (operation === 'encrypt') {
    const name=validName(options.inputName);
    const file=await fileOf(options.inputFile);
    const filename=outputName(file.name,/\.csv$/i,'.ct.bin');
    const values=await readCSV(file);
    const result=core('encrypt',name,values);
    const output=await options.outputDirectory.getFileHandle(filename,{create:true});
    await writeFile(output,result.data);
    progress('encrypt',start,result.data.byteLength,result);return output;
  }
  if (operation === 'decrypt') {
    const name=validName(options.outputName);
    const file=await fileOf(options.ciphertextFile);
    const filename=outputName(file.name,/\.ct\.bin$/i,'.csv');
    const result=core('decrypt',name,new Uint8Array(await file.arrayBuffer()));
    const output=await options.outputDirectory.getFileHandle(filename,{create:true});
    await writeFile(output,Array.from(result.data).join(',')+'\n');
    progress('decrypt',start,file.size,result);return output;
  }
  if (operation === 'encrypt-server') return encryptServer(options, start);
  if (operation === 'decrypt-server') return decryptServer(options, start);
  throw new Error('Unknown client operation');
}
let queue=Promise.resolve();
onmessage=({data})=>{
  queue=queue.then(async()=>{
    try { postMessage({id:data.id,result:await operate(data)}); }
    catch (error) { postMessage({id:data.id,error:error.message}); }
  });
};
