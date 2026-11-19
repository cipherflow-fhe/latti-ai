// A private OPFS file pool backs Python's model workspace. Directory metadata
// stays in Emscripten; large file contents use synchronous, bounded disk I/O.
const CACHE='latti-python-workspaces';
export async function cleanupPythonWorkspace(jobId) {
  if(!globalThis.navigator?.storage?.getDirectory)return;
  try {
    const root=await(await navigator.storage.getDirectory()).getDirectoryHandle(CACHE);
    // terminate() returns before Chromium has released the Worker's file locks.
    // A busy WASM Worker can take several seconds to release its handles.
    // Retry only that transient lock error, for at most ten seconds.
    const deadline=performance.now()+10000;
    for(;;) {
      try {await root.removeEntry(jobId,{recursive:true});break;}
      catch(error) {
        if(error.name!=='NoModificationAllowedError'||performance.now()>=deadline)throw error;
        await new Promise(resolve=>setTimeout(resolve,10));
      }
    }
  }catch(error){if(error.name!=='NotFoundError')throw error;}
}

export async function mountPythonWorkspace(FS,jobId) {
  const root=await(await(await navigator.storage.getDirectory()).getDirectoryHandle(CACHE,{create:true})).getDirectoryHandle(jobId,{create:true});
  let saved;
  try{saved=JSON.parse(await(await(await root.getFileHandle('workspace.json')).getFile()).text());}
  catch(error){if(error.name!=='NotFoundError')throw error;}
  const handles=[],free=[];let restoring;
  try {
    for(let index=0;index<256;index++) {
      const file=await root.getFileHandle('file-'+index,{create:true}),handle=await file.createSyncAccessHandle();
      handles.push(handle);free.push(handle);
    }
  }catch(error){for(const handle of handles)handle.close();throw error;}
  const memfs=FS.filesystems.MEMFS;
  const fail=code=>{throw new FS.ErrnoError(code);};
  const release=node=>{
    if(node.disk&&node.removed&&!node.openCount){node.disk.truncate(0);free.push(node.disk);node.disk=null;}
  };
  const create=(parent,name,mode,device)=>{
    if(FS.isFile(mode)&&!free.length)fail(33); // EMFILE: bounded workspace file count.
    const node=memfs.createNode(parent,name,mode,device);node.node_ops=nodeOps;
    if(FS.isFile(mode)) {
      if(restoring){
        node.disk=handles[restoring.slot];
        const index=free.indexOf(node.disk);if(index<0)throw new Error('Invalid workspace checkpoint slot');
        free.splice(index,1);
      }else{node.disk=free.pop();node.disk.truncate(0);}
      node.openCount=0;node.stream_ops=streamOps;
    }
    return node;
  };
  const nodeOps={...memfs.node_ops,
    mknod:create,
    getattr(node) {
      const result=memfs.node_ops.getattr(node);
      if(node.disk){result.size=node.disk.getSize();result.blocks=Math.ceil(result.size/512);}
      return result;
    },
    setattr(node,attributes) {
      const {size,...other}=attributes;
      if(node.disk&&size!==undefined)node.disk.truncate(size);
      memfs.node_ops.setattr(node,node.disk?other:attributes);
    },
    unlink(parent,name) {
      const node=parent.contents[name];memfs.node_ops.unlink(parent,name);
      node.removed=true;release(node);
    },
    rename(node,parent,name) {
      const replaced=parent.contents[name];
      memfs.node_ops.rename(node,parent,name);
      if(replaced&&replaced!==node){replaced.removed=true;release(replaced);}
    },
  };
  const streamOps={
    open(stream){stream.node.openCount++;},
    dup(stream){stream.node.openCount++;},
    close(stream){stream.node.openCount--;release(stream.node);},
    read(stream,buffer,offset,length,position) {
      return stream.node.disk.read(buffer.subarray(offset,offset+length),{at:position});
    },
    write(stream,buffer,offset,length,position) {
      const count=stream.node.disk.write(buffer.subarray(offset,offset+length),{at:position});
      if(count!==length)fail(29); // EIO; quota and permissions also propagate.
      stream.node.timestamp=Date.now();
      return count;
    },
    llseek(stream,offset,whence) {
      const position=offset+(whence===1?stream.position:whence===2?stream.node.disk.getSize():0);
      if(!Number.isSafeInteger(position)||position<0||![0,1,2].includes(whence))fail(28);
      return position;
    },
    allocate(stream,offset,length) {
      const disk=stream.node.disk;disk.truncate(Math.max(disk.getSize(),offset+length));
    },
    mmap(){fail(43);},
  };
  FS.mkdirTree('/latti/jobs');
  const mounted=FS.mount({mount:()=>create(null,'/',0o40777,0)}, {}, '/latti/jobs');
  for(const entry of saved||[]) {
    if(typeof entry.path!=='string'||entry.path.split('/').some(p=>!p||p==='.'||p==='..'))throw new Error('Invalid workspace checkpoint path');
    if(FS.isFile(entry.mode)&&(!Number.isSafeInteger(entry.slot)||entry.slot<0||entry.slot>=handles.length))throw new Error('Invalid workspace checkpoint slot');
    restoring=entry;try{FS.mknod('/latti/jobs/'+entry.path,entry.mode,0);}finally{restoring=undefined;}
  }
  const flush=()=>{for(const handle of handles)handle.flush();};
  return {flush,async checkpoint(){
    const entries=[];
    const walk=(node,path='')=>{
      for(const [name,child]of Object.entries(node.contents||{})) {
        const relative=path?path+'/'+name:name;
        if(child.disk&&child.openCount)throw new Error('Open Python file at stage checkpoint');
        entries.push({path:relative,mode:child.mode,...(child.disk?{slot:handles.indexOf(child.disk)}:{})});
        if(FS.isDir(child.mode))walk(child,relative);
      }
    };
    walk(mounted);flush();
    const file=await root.getFileHandle('workspace.json',{create:true}),writer=await file.createWritable();
    try{await writer.write(JSON.stringify(entries));await writer.close();}
    catch(error){await writer.abort().catch(()=>{});throw error;}
    for(const handle of handles)handle.close();
  }};
}
