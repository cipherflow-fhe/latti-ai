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
  return {
    setup: () => request('setup'),
    restore: () => request('restore'),
    encrypt: options => request('encrypt',options),
    decrypt: options => request('decrypt',options),
    async close() {
      if (closed) return;
      const result = request('close'); closed = true;
      try { await result; } finally { worker.terminate(); }
    },
  };
}
