import {createServer} from 'node:http';
import {createReadStream,createWriteStream} from 'node:fs';
import {mkdir,readFile,stat} from 'node:fs/promises';
import {pipeline} from 'node:stream/promises';
import path from 'node:path';
import {chromium} from 'playwright-core';

// Test transport and measurements only. Runtime owns Feature/ZIP operations.
export async function browserHarness(artifacts,mounts={}) {
  const runtime=path.resolve(process.env.LATTI_RUNTIME_DIRECTORY||path.join(import.meta.dirname,'../dist'));
  await mkdir(artifacts,{recursive:true});
  const server=createServer(async(req,res)=>{
    try {
      if(req.url==='/'){res.setHeader('Content-Type','text/html');res.end('<!doctype html><title>WASM acceptance</title>');return;}
      if(req.url==='/favicon.ico'){res.writeHead(204).end();return;}
      if(req.method==='POST'&&/^\/artifact\/[A-Za-z0-9._-]+$/.test(req.url)) {
        await pipeline(req,createWriteStream(path.join(artifacts,path.basename(req.url)),{mode:0o600,flags:req.headers['x-latti-append']==='1'?'a':'w'}));res.end('saved');return;
      }
      const directories={'/dist/':runtime,'/artifact/':artifacts,...mounts};
      const prefix=Object.keys(directories).find(prefix=>req.url.startsWith(prefix));
      if(!prefix)throw new Error('Unknown route');
      const base=path.resolve(directories[prefix]),file=path.resolve(base,decodeURIComponent(req.url.slice(prefix.length)));
      if(!file.startsWith(base+path.sep))throw new Error('Invalid path');
      const info=await stat(file);res.setHeader('Content-Length',info.size);
      res.setHeader('Content-Type',file.endsWith('.wasm')?'application/wasm':/\.m?js$/.test(file)?'text/javascript':file.endsWith('.json')?'application/json':file.endsWith('.html')?'text/html':'application/octet-stream');
      await pipeline(createReadStream(file),res);
    } catch(error) {if(!res.headersSent)res.writeHead(404);res.end();console.error('Test HTTP:',req.url,error.message);}
  });
  await new Promise(resolve=>server.listen(Number(process.env.LATTI_BROWSER_PORT||0),'127.0.0.1',resolve));
  const context=await chromium.launchPersistentContext(path.join(artifacts,'browser-profile'),{
    ...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{}),headless:true,args:['--no-sandbox'],
  });
  const page=await context.newPage();
  await page.addInitScript(()=>{
    globalThis.lattiTerminalCounts={};let workerId=0;
    const NativeWorker=globalThis.Worker;
    globalThis.Worker=new Proxy(NativeWorker,{construct(target,args){
      const worker=Reflect.construct(target,args),id=++workerId;
      worker.addEventListener('message',({data})=>{
        if(data.requestId>0&&['result','error','cancelled'].includes(data.type)) {
          const key=id+':'+data.requestId;
          lattiTerminalCounts[key]=(lattiTerminalCounts[key]||0)+1;
        }
      });return worker;
    }});
  });
  page.on('console',message=>console.log('[browser]',message.text()));page.on('pageerror',error=>console.error(error));
  const cdp=await context.browser().newBrowserCDPSession();
  const heapSessions=new Map(),pending=new Map();let nextMessage=0;
  cdp.on('Target.receivedMessageFromTarget',event=>{
    const response=JSON.parse(event.message),request=pending.get(response.id);
    if(!request)return;pending.delete(response.id);
    if(response.error)request.reject(new Error(response.error.message));else request.resolve(response.result);
  });
  const heapUsage=sessionId=>new Promise((resolve,reject)=>{
    const id=++nextMessage;
    const timeout=setTimeout(()=>{pending.delete(id);reject(new Error('Heap sample timed out'));},2000);
    pending.set(id,{resolve:value=>{clearTimeout(timeout);resolve(value);},reject:error=>{clearTimeout(timeout);reject(error);}});
    cdp.send('Target.sendMessageToTarget',{sessionId,message:JSON.stringify({id,method:'Runtime.getHeapUsage'})}).catch(error=>{
      const request=pending.get(id);pending.delete(id);request?.reject(error);
    });
  });
  const rss={sampleIntervalMs:250,samples:0,rendererPeakKiB:0,chromeProcessesPeakKiB:0,
    jsHeapPeakBytes:0,workerJSHeapPeakBytes:0,externalArrayBufferPeakBytes:0,embedderHeapPeakBytes:0,jsHeapSamples:0,heapSampleErrors:0,
    limitBytes:4294967296,baselineRendererKiB:null,baselineChromeKiB:null,
    maxSampleGapMs:0,processReadErrors:0,observations:[]};let sampling=false,heapSampling=false,lastSample,heapPending;
  const phaseMetrics={};let currentPhase;
  const sample=async()=>{
    if(sampling)return;sampling=true;
    const phase=currentPhase&&phaseMetrics[currentPhase];
    try {
      const timestamp=Date.now();
      if(lastSample!==undefined)rss.maxSampleGapMs=Math.max(rss.maxSampleGapMs,timestamp-lastSample);
      lastSample=timestamp;
      const {processInfo}=await cdp.send('SystemInfo.getProcessInfo');let renderers=0,total=0;
      const processes=[];
      for(const process of processInfo) {
        try {const status=await readFile(`/proc/${process.id}/status`,'utf8'),bytes=Number(status.match(/^VmRSS:\s+(\d+)/m)?.[1]||0);total+=bytes;if(process.type==='renderer')renderers+=bytes;processes.push({pid:process.id,type:process.type,rssKiB:bytes});} catch {rss.processReadErrors++;}
      }
      if(rss.baselineRendererKiB===null){rss.baselineRendererKiB=renderers;rss.baselineChromeKiB=total;}
      rss.observations.push({timestamp,phase:currentPhase||'baseline',rendererKiB:renderers,chromeKiB:total,processes});
      rss.samples++;rss.rendererPeakKiB=Math.max(rss.rendererPeakKiB,renderers);rss.chromeProcessesPeakKiB=Math.max(rss.chromeProcessesPeakKiB,total);
      if(phase) {
        phase.samples++;
        phase.rendererPeakKiB=Math.max(phase.rendererPeakKiB,renderers);
        phase.chromeProcessesPeakKiB=Math.max(phase.chromeProcessesPeakKiB,total);
      }
      // A busy Python Worker can delay CDP heap replies. Keep RSS sampling independent.
      if(!heapSampling) {
        heapSampling=true;
        heapPending=(async()=>{
          const {targetInfos}=await cdp.send('Target.getTargets');let js=0,workerJS=0,backing=0,embedder=0;
          for(const target of targetInfos.filter(target=>target.type==='page'||target.type==='worker')) {
            try {
              if(!heapSessions.has(target.targetId))heapSessions.set(target.targetId,(await cdp.send('Target.attachToTarget',{targetId:target.targetId,flatten:false})).sessionId);
              const usage=await heapUsage(heapSessions.get(target.targetId));js+=usage.usedSize;backing+=usage.backingStorageSize||0;embedder+=usage.embedderHeapUsedSize||0;
              if(target.type==='worker')workerJS+=usage.usedSize;
              rss.jsHeapSamples++;
            } catch {rss.heapSampleErrors++;heapSessions.delete(target.targetId);}
          }
          rss.externalArrayBufferPeakBytes=Math.max(rss.externalArrayBufferPeakBytes,backing);rss.embedderHeapPeakBytes=Math.max(rss.embedderHeapPeakBytes,embedder);
          rss.jsHeapPeakBytes=Math.max(rss.jsHeapPeakBytes,js);rss.workerJSHeapPeakBytes=Math.max(rss.workerJSHeapPeakBytes,workerJS);
          if(phase) {
            phase.externalArrayBufferPeakBytes=Math.max(phase.externalArrayBufferPeakBytes,backing);
            phase.embedderHeapPeakBytes=Math.max(phase.embedderHeapPeakBytes,embedder);
            phase.jsHeapPeakBytes=Math.max(phase.jsHeapPeakBytes,js);
            phase.workerJSHeapPeakBytes=Math.max(phase.workerJSHeapPeakBytes,workerJS);
          }
        })().catch(()=>{rss.heapSampleErrors++;}).finally(()=>{heapSampling=false;});
      }
    } finally {sampling=false;}
  };
  const timer=setInterval(()=>sample().catch(()=>{}),rss.sampleIntervalMs);
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  return {page,rss,phaseMetrics,async beginPhase(name){
    await sample();currentPhase=name;
    phaseMetrics[name]={sampleIntervalMs:rss.sampleIntervalMs,samples:0,rendererPeakKiB:0,chromeProcessesPeakKiB:0,jsHeapPeakBytes:0,workerJSHeapPeakBytes:0,externalArrayBufferPeakBytes:0,embedderHeapPeakBytes:0};
    await sample();
  },async close(){clearInterval(timer);await sample();await heapPending;await context.close();await new Promise(resolve=>server.close(resolve));}};
}
