import {safeName,setErrorMessage} from '../../io.js';

const MEAN=[0.485,0.456,0.406],STD=[0.229,0.224,0.225];
const PRECISION=1<<22;

// Pillow's separable bilinear filter widens its support when downsampling and
// rounds RGB8 after each pass. Canvas resizing does not provide that contract.
function coefficients(input,output) {
  const scale=input/output,support=Math.max(1,scale),table=[];
  for(let i=0;i<output;i++) {
    const center=(i+0.5)*scale;
    const start=Math.max(0,Math.trunc(center-support+0.5));
    const end=Math.min(input,Math.trunc(center+support+0.5));
    const weights=[];let sum=0;
    for(let x=start;x<end;x++) {const w=Math.max(0,1-Math.abs((x-center+0.5)/support));weights.push(w);sum+=w;}
    table.push({start,weights:weights.map(w=>Math.floor(w/sum*PRECISION+0.5))});
  }
  return table;
}

export function resizeRGB(pixels,width,height,targetWidth=32,targetHeight=32) {
  if(!Number.isSafeInteger(width)||!Number.isSafeInteger(height)||width<1||height<1||pixels.length!==width*height*3) throw new Error('Invalid RGB8 image');
  let horizontal=pixels;
  if(width!==targetWidth) {
    const table=coefficients(width,targetWidth);horizontal=new Uint8Array(targetWidth*height*3);
    for(let y=0;y<height;y++) for(let x=0;x<targetWidth;x++) for(let c=0;c<3;c++) {
      const {start,weights}=table[x];let sum=PRECISION/2;
      for(let k=0;k<weights.length;k++) sum+=pixels[(y*width+start+k)*3+c]*weights[k];
      horizontal[(y*targetWidth+x)*3+c]=Math.min(255,Math.max(0,Math.floor(sum/PRECISION)));
    }
  }
  if(height===targetHeight) return horizontal;
  const table=coefficients(height,targetHeight),output=new Uint8Array(targetWidth*targetHeight*3);
  for(let y=0;y<targetHeight;y++) for(let x=0;x<targetWidth;x++) for(let c=0;c<3;c++) {
    const {start,weights}=table[y];let sum=PRECISION/2;
    for(let k=0;k<weights.length;k++) sum+=horizontal[((start+k)*targetWidth+x)*3+c]*weights[k];
    output[(y*targetWidth+x)*3+c]=Math.min(255,Math.max(0,Math.floor(sum/PRECISION)));
  }
  return output;
}

export function normalizeRGB(pixels) {
  if(pixels.length!==3072) throw new Error('CIFAR input must be 32 x 32 RGB8');
  const output=new Float32Array(3072),f=Math.fround;
  for(let c=0;c<3;c++) for(let i=0;i<1024;i++) output[c*1024+i]=f(f(f(pixels[i*3+c]/255)-f(MEAN[c]))/f(STD[c]));
  return output;
}

export async function decodeRGB(file) {
  // Reject animated PNGs instead of silently selecting a frame. Read chunks
  // by header so an image's compressed payload is never copied just to scan it.
  const signature=new Uint8Array(await file.slice(0,8).arrayBuffer());
  if(signature.join(',')==='137,80,78,71,13,10,26,10') {
    for(let offset=8;offset<file.size;) {
      const bytes=await file.slice(offset,offset+8).arrayBuffer();
      if(bytes.byteLength!==8) throw new Error('Truncated PNG');
      const view=new DataView(bytes),length=view.getUint32(0),type=String.fromCharCode(...new Uint8Array(bytes,4));
      if(type==='acTL') throw new Error('Animated PNG is unsupported');
      offset+=length+12;
      if(offset>file.size) throw new Error('Truncated PNG chunk');
      if(type==='IEND') break;
    }
  } else if(signature[0]!==255||signature[1]!==216) throw new Error('Expected PNG or JPEG image');
  const bitmap=await createImageBitmap(file,{premultiplyAlpha:'none',colorSpaceConversion:'none',imageOrientation:'none'});
  try {
    const {width,height}=bitmap,canvas=new OffscreenCanvas(width,height),context=canvas.getContext('2d',{willReadFrequently:true});
    context.drawImage(bitmap,0,0);
    const rgba=context.getImageData(0,0,width,height).data,rgb=new Uint8Array(width*height*3);
    for(let i=0;i<width*height;i++) {
      if(rgba[i*4+3]!==255) throw new Error('Transparent images are unsupported');
      rgb.set(rgba.subarray(i*4,i*4+3),i*3);
    }
    return {pixels:rgb,width,height};
  } finally {bitmap.close();}
}

export async function preprocessDirectory({inputDirectory,outputDirectory,check=()=>{}}) {
  const files=[],names=new Set();
  for await(const file of inputDirectory.values()) {
    check();if(file.kind!=='file'||! /\.(png|jpe?g)$/i.test(file.name)) continue;
    const name=safeName(file.name.replace(/\.(png|jpe?g)$/i,'.csv')),key=name.normalize('NFC').toLowerCase();
    if(names.has(key)) throw new Error('Duplicate image basename: '+file.name);names.add(key);
    if(files.length>=100000) throw new Error('METADATA_BUDGET_EXCEEDED');
    files.push({file,name});
  }
  if(!files.length) throw new Error('Empty image directory');
  for await(const entry of outputDirectory.values()) if(names.has(entry.name.normalize('NFC').toLowerCase())) throw new Error('Output already exists: '+entry.name);
  files.sort((a,b)=>a.file.name<b.file.name?-1:a.file.name>b.file.name?1:0);
  const created=[];
  try {
    for(const {file,name}of files) {
      check();let writer;
      try {
        const {pixels,width,height}=await decodeRGB(await file.getFile());
        const values=normalizeRGB(resizeRGB(pixels,width,height));check();
        const output=await outputDirectory.getFileHandle(name,{create:true});created.push(name);writer=await output.createWritable();
        for(let i=0;i<values.length;i+=256) {check();await writer.write((i?',':'')+values.subarray(i,i+256).join(','));}
        await writer.write('\n');await writer.close();writer=undefined;
      } catch(error) {if(writer)await writer.abort().catch(()=>{});throw setErrorMessage(error,file.name+': '+error.message);}
    }
    return {count:String(files.length),directory:outputDirectory};
  } catch(error) {
    for(const name of created) try {await outputDirectory.removeEntry(name);} catch(failure) {setErrorMessage(error,error.message+'; residual '+name+': '+failure.message);}
    throw error;
  }
}
