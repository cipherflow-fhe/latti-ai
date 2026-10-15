export async function openBrowserClient({clientDirectory,keyDirectory,onProgress}) {
  if (!globalThis.isSecureContext) throw new Error('A secure browser context is required');
  if (!clientDirectory && !keyDirectory) throw new Error('Select a client or key directory');
  for (const directory of [clientDirectory,keyDirectory]) {
    if (directory && directory.kind !== 'directory') throw new Error('Select a directory');
  }
  const worker = new Worker(new URL('./worker.js',import.meta.url),{type:'module'});
  let nextID = 0, closed = false, queue = Promise.resolve();
  const pending = new Map();
  worker.onmessage = ({data}) => {
    if (data.type === 'progress') { onProgress?.(data.progress); return; }
    const waiter = pending.get(data.id);
    if (!waiter) return;
    pending.delete(data.id);
    if (data.error) waiter.reject(new Error(data.error));
    else waiter.resolve(data.result);
  };
  worker.onerror = () => {
    closed = true;
    for (const waiter of pending.values()) waiter.reject(new Error('Browser client Worker failed'));
    pending.clear(); worker.terminate();
  };
  const request = (operation,options={}) => {
    if (closed) return Promise.reject(new Error('Browser client is closed'));
    const run = () => new Promise((resolve,reject) => {
      const id = ++nextID;
      pending.set(id,{resolve,reject});
      try { worker.postMessage({id,operation,...options}); }
      catch (error) { pending.delete(id); reject(error); }
    });
    const result = queue.then(run);
    queue = result.catch(()=>{});
    return result;
  };
  try { await request('open',{clientDirectory,keyDirectory}); }
  catch (error) { worker.terminate(); throw error; }
  const batch = async (operation,{inputDirectory,outputDirectory,...options}) => {
    if (closed) throw new Error('Browser client is closed');
    const encrypting=operation==='encrypt';
    const extension=encrypting ? /\.csv$/i : /\.ct\.bin$/i;
    const suffix=encrypting ? '.ct.bin' : '.csv';
    const files=[];
    for await (const file of inputDirectory.values()) {
      if (file.kind==='file' && extension.test(file.name)) files.push(file);
    }
    files.sort((a,b)=>a.name<b.name ? -1 : a.name>b.name ? 1 : 0);
    const names=new Set();
    for (const file of files) {
      const name=file.name.replace(extension,suffix);
      if (/[\/\\\0]/.test(name) || name==='.' || name==='..') throw new Error('Invalid output file name: '+file.name);
      if (names.has(name.toLowerCase())) throw new Error('Duplicate output file name: '+name);
      names.add(name.toLowerCase());
    }
    const outputs=[];
    for (const file of files) {
      try {
        outputs.push(await client[operation]({...options,outputDirectory,[encrypting ? 'inputFile' : 'ciphertextFile']:file}));
      } catch (error) { throw new Error(file.name+': '+error.message,{cause:error}); }
    }
    return outputs;
  };
  const client = {
    setup: () => request('setup'),
    restore: () => request('restore'),
    encrypt: options => request('encrypt',options),
    decrypt: options => request('decrypt',options),
    encrypt_batch: options => batch('encrypt',options),
    decrypt_batch: options => batch('decrypt',options),
    async close() {
      if (closed) return;
      const result = request('close'); closed = true;
      try { await result; } finally { worker.terminate(); }
    },
  };
  return client;
}
