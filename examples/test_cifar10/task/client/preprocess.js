// CIFAR-10 evaluation transform from train.py: Resize, ToTensor, Normalize.
const imageExtension=/\.(png|jpe?g)$/i;
const mean=new Float32Array([0.485,0.456,0.406]);
const std=new Float32Array([0.229,0.224,0.225]);
const precision=2**22;

function coefficients(size) {
  const scale=size/32,support=Math.max(1,scale);
  return Array.from({length:32},(_,i)=>{
    const center=(i+0.5)*scale;
    const start=Math.max(0,Math.floor(center-support+0.5));
    const end=Math.min(size,Math.floor(center+support+0.5));
    const weights=Array.from({length:end-start},(_,j)=>Math.max(0,1-Math.abs((start+j+0.5-center)/support)));
    const total=weights.reduce((a,b)=>a+b,0);
    return {start,weights:weights.map(w=>Math.floor(w/total*precision+0.5))};
  });
}

function resizeRGB(pixels,width,height) {
  const horizontal=new Uint8Array(32*height*3),output=new Uint8Array(32*32*3);
  const xs=coefficients(width),ys=coefficients(height);
  for(let y=0;y<height;y++)for(let x=0;x<32;x++)for(let c=0;c<3;c++){
    let value=precision/2;
    const {start,weights}=xs[x];
    for(let j=0;j<weights.length;j++)value+=pixels[(y*width+start+j)*4+c]*weights[j];
    horizontal[(y*32+x)*3+c]=Math.max(0,Math.min(255,Math.floor(value/precision)));
  }
  for(let y=0;y<32;y++)for(let x=0;x<32;x++)for(let c=0;c<3;c++){
    let value=precision/2;
    const {start,weights}=ys[y];
    for(let j=0;j<weights.length;j++)value+=horizontal[((start+j)*32+x)*3+c]*weights[j];
    output[(y*32+x)*3+c]=Math.max(0,Math.min(255,Math.floor(value/precision)));
  }
  return output;
}

export async function preprocess_batch({inputDirectory,outputDirectory}) {
  // Accept either the dataset root (dataset/images/) or the images directory
  // itself, so CIFAR and Retina task runners can pass the package dataset path
  // without a task-specific path rewrite.
  let imageDirectory=inputDirectory;
  try{imageDirectory=await inputDirectory.getDirectoryHandle('images');}
  catch(error){if(error.name!=='NotFoundError')throw error;}
  const files=[];
  for await(const file of imageDirectory.values())if(file.kind==='file'&&imageExtension.test(file.name))files.push(file);
  files.sort((a,b)=>a.name<b.name?-1:a.name>b.name?1:0);
  const names=new Set();
  for(const file of files){
    const name=file.name.replace(imageExtension,'.csv');
    if(/[\/\\\0]/.test(name))throw Error('Invalid output file name: '+file.name);
    if(names.has(name.toLowerCase()))throw Error('Duplicate output file name: '+name);
    names.add(name.toLowerCase());
  }
  const outputs=[];
  for(const file of files){
    let image,canvas,writer;
    try{
      image=await createImageBitmap(await file.getFile());
      canvas=new OffscreenCanvas(image.width,image.height);
      const context=canvas.getContext('2d',{willReadFrequently:true});
      context.drawImage(image,0,0);
      const pixels=context.getImageData(0,0,image.width,image.height).data;
      for(let i=3;i<pixels.length;i+=4)if(pixels[i]!==255)throw Error('An opaque RGB image is required');
      const rgb=resizeRGB(pixels,image.width,image.height),values=new Float32Array(3072);
      for(let c=0;c<3;c++)for(let i=0;i<1024;i++)values[c*1024+i]=Math.fround(Math.fround(Math.fround(rgb[i*3+c]/255)-mean[c])/std[c]);
      const output=await outputDirectory.getFileHandle(file.name.replace(imageExtension,'.csv'),{create:true});
      writer=await output.createWritable();
      await writer.write(Array.from(values).join(',')+'\n');
      await writer.close();
      outputs.push(output);
    }catch(error){
      if(writer)await writer.abort().catch(()=>{});
      throw new Error(file.name+': '+error.message,{cause:error});
    }finally{
      image?.close();
      if(canvas)canvas.width=canvas.height=0;
    }
  }
  return outputs;
}
