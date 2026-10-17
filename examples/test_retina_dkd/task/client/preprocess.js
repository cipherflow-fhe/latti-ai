// gen_client_inputs_seg.py, configured for retina_degree8_02: 512px + --normalize.
// Standalone browser ES module; XLSX uses native ZIP/XML APIs (no CDN dependency).
const imageExtension = /\.(png|jpe?g)$/i;
const size = 512, area = size * size, precision = 2 ** 22;
const mean = new Float32Array([0.485, 0.456, 0.406]);
const std = new Float32Array([0.229, 0.224, 0.225]);

function coefficients(length) {
  const scale = length / size, support = Math.max(1, scale);
  return Array.from({length: size}, (_, i) => {
    const center = (i + 0.5) * scale;
    const start = Math.max(0, Math.floor(center - support + 0.5));
    const end = Math.min(length, Math.floor(center + support + 0.5));
    const weights = Array.from({length: end - start}, (_, j) =>
      Math.max(0, 1 - Math.abs((start + j + 0.5 - center) / support)));
    const total = weights.reduce((a, b) => a + b, 0);
    return {start, weights: weights.map(w => Math.floor(w / total * precision + 0.5))};
  });
}

// Pillow RGB8 BILINEAR: widened kernel on downsampling, two rounded passes.
function resizeRGB(pixels, width, height) {
  const horizontal = new Uint8Array(size * height * 3), output = new Uint8Array(area * 3);
  const xs = coefficients(width), ys = coefficients(height);
  for (let y = 0; y < height; y++) for (let x = 0; x < size; x++) for (let c = 0; c < 3; c++) {
    let value = precision / 2;
    const {start, weights} = xs[x];
    for (let j = 0; j < weights.length; j++) value += pixels[(y * width + start + j) * 4 + c] * weights[j];
    horizontal[(y * size + x) * 3 + c] = Math.max(0, Math.min(255, Math.floor(value / precision)));
  }
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) for (let c = 0; c < 3; c++) {
    let value = precision / 2;
    const {start, weights} = ys[y];
    for (let j = 0; j < weights.length; j++) value += horizontal[((start + j) * size + x) * 3 + c] * weights[j];
    output[(y * size + x) * 3 + c] = Math.max(0, Math.min(255, Math.floor(value / precision)));
  }
  return output;
}

async function readImage(handle, mask = false) {
  let image, canvas;
  try {
    image = await createImageBitmap(await handle.getFile(), {imageOrientation: 'none', colorSpaceConversion: 'none'});
    canvas = new OffscreenCanvas(image.width, image.height);
    const context = canvas.getContext('2d', {willReadFrequently: true});
    context.drawImage(image, 0, 0);
    const pixels = context.getImageData(0, 0, image.width, image.height).data;
    for (let i = 0; i < pixels.length; i += 4) {
      if (pixels[i + 3] !== 255) throw Error('An opaque image is required: ' + handle.name);
      // Python thresholds the source mask BEFORE resizing, independently per channel.
      if (mask) for (let c = 0; c < 3; c++) pixels[i + c] = pixels[i + c] > 80 ? 255 : 0;
    }
    return resizeRGB(pixels, image.width, image.height);
  } finally {
    image?.close();
    if (canvas) canvas.width = canvas.height = 0;
  }
}

function parseXML(text) {
  const document = new DOMParser().parseFromString(text, 'application/xml');
  if (document.getElementsByTagName('parsererror').length) throw Error('Invalid XLSX XML');
  return document;
}
const elements = (node, name) => Array.from(node.getElementsByTagNameNS('*', name));
const richText = node => elements(node, 't').map(part => part.textContent).join('');

// Read the central directory, so ZIP entries with a trailing data descriptor work.
async function openXLSX(file) {
  const buffer = await file.arrayBuffer(), view = new DataView(buffer), bytes = new Uint8Array(buffer);
  const decoder = new TextDecoder(), entries = new Map();
  let end = bytes.length - 22;
  for (; end >= Math.max(0, bytes.length - 65557); end--)
    if (view.getUint32(end, true) === 0x06054b50 && end + 22 + view.getUint16(end + 20, true) === bytes.length) break;
  if (end < 0 || view.getUint32(end, true) !== 0x06054b50) throw Error('Expected an unencrypted .xlsx ZIP workbook');
  const count = view.getUint16(end + 10, true);
  let offset = view.getUint32(end + 16, true);
  for (let i = 0; i < count; i++) {
    if (view.getUint32(offset, true) !== 0x02014b50) throw Error('Invalid XLSX ZIP directory');
    const length = view.getUint16(offset + 28, true);
    const name = decoder.decode(bytes.subarray(offset + 46, offset + 46 + length));
    entries.set(name, {flags: view.getUint16(offset + 8, true), method: view.getUint16(offset + 10, true),
      size: view.getUint32(offset + 20, true), local: view.getUint32(offset + 42, true)});
    offset += 46 + length + view.getUint16(offset + 30, true) + view.getUint16(offset + 32, true);
  }
  return async name => {
    const entry = entries.get(name);
    if (!entry) return null;
    if (entry.flags & 1) throw Error('Encrypted XLSX is unsupported');
    if (view.getUint32(entry.local, true) !== 0x04034b50) throw Error('Invalid XLSX ZIP entry');
    const start = entry.local + 30 + view.getUint16(entry.local + 26, true) + view.getUint16(entry.local + 28, true);
    const compressed = bytes.subarray(start, start + entry.size);
    if (entry.method === 0) return parseXML(decoder.decode(compressed));
    if (entry.method !== 8) throw Error('Unsupported XLSX ZIP compression');
    const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
    return parseXML(await new Response(stream).text());
  };
}

async function readClinicalFactors(file) {
  const read = await openXLSX(file);
  const workbook = await read('xl/workbook.xml'), relationships = await read('xl/_rels/workbook.xml.rels');
  if (!workbook || !relationships) throw Error('Missing XLSX workbook');
  const sheet = elements(workbook, 'sheet')[0];
  const id = sheet?.getAttributeNS('http://schemas.openxmlformats.org/officeDocument/2006/relationships', 'id');
  const relation = elements(relationships, 'Relationship').find(item => item.getAttribute('Id') === id);
  if (!relation || relation.getAttribute('TargetMode') === 'External') throw Error('Missing first XLSX worksheet');
  const sheetPath = new URL(relation.getAttribute('Target'), 'https://xlsx.invalid/xl/workbook.xml').pathname.slice(1);
  const worksheet = await read(sheetPath);
  if (!worksheet) throw Error('Missing XLSX worksheet: ' + sheetPath);
  const shared = await read('xl/sharedStrings.xml');
  const strings = shared ? elements(shared, 'si').map(richText) : [];
  const factors = new Map();
  for (const row of elements(worksheet, 'row').slice(1)) {
    const cells = Array(8).fill('');
    for (const cell of elements(row, 'c')) {
      const letters = cell.getAttribute('r')?.match(/^[A-Z]+/)?.[0];
      if (!letters) throw Error('Missing XLSX cell address');
      let column = 0;
      for (const letter of letters) column = column * 26 + letter.charCodeAt(0) - 64;
      if (column > 8) continue;
      const type = cell.getAttribute('t'), value = elements(cell, 'v')[0]?.textContent ?? '';
      cells[column - 1] = type === 'inlineStr' ? richText(cell) : type === 's' ? strings[Number(value)] :
        type === 'str' ? value : value === '' ? '' : Number(value);
    }
    const patient = typeof cells[1] === 'string' ? cells[1] : String(Math.trunc(cells[1]));
    let [, , , press, course, mashing, protein, eGFR] = cells;
    if (course === '') course = 0;
    if (mashing === '') { mashing = 0; protein = 0; }
    if (!patient || [press, course, mashing, protein, eGFR].some(value => typeof value !== 'number' || !Number.isFinite(value)))
      throw Error('Invalid clinical factors in XLSX row ' + row.getAttribute('r'));
    factors.set(patient, {nonInv: [press / 202, course / 564], inv: [mashing / 13.8, protein / 207, eGFR / 132]});
  }
  return factors;
}

function patientID(filename) {
  const parts = filename.split(/-|_/);
  let name = parts.length > 3 ? (parts[0].length < 4 ? parts[1] : parts[0]) :
    (parts[0].length < 4 ? parts.at(-2) : parts[0]);
  if (!name) throw Error('Cannot extract patient ID: ' + filename);
  name = name.replace(/^[Nn]/, '');
  return name.charAt(0).toUpperCase() + name.slice(1).toLowerCase();
}

async function writeCSV(directory, name, rows) {
  const handle = await directory.getFileHandle(name, {create: true});
  const writer = await handle.createWritable();
  try {
    for (const row of rows) await writer.write(Array.from(row).join(',') + '\n');
    await writer.close();
    return handle;
  } catch (error) { await writer.abort().catch(() => {}); throw error; }
}

export async function preprocess_batch({inputDirectory, outputDirectory, normalize = true}) {
  const files = [];
  const names = new Set();
  const images = await inputDirectory.getDirectoryHandle('images');
  for await (const file of images.values()) if (file.kind === 'file' && imageExtension.test(file.name)) {
    const name = file.name.replace(imageExtension, '.csv');
    if (/[\/\\\0]/.test(name) || names.has(name.toLowerCase())) throw Error('Invalid or duplicate output name: ' + file.name);
    names.add(name.toLowerCase()); files.push(file);
  }
  files.sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0);
  if (!files.length) return [];
  const factors = await readClinicalFactors(await (await inputDirectory.getFileHandle('risk_factor_5.xlsx')).getFile());
  const diskDirectory = await inputDirectory.getDirectoryHandle('disk');
  const lesionDirectory = await inputDirectory.getDirectoryHandle('lesion');
  const samples = [];
  // Validate all associations before writing this batch.
  for (const image of files) {
    const stem = image.name.replace(imageExtension, ''), pid = patientID(image.name);
    if (!factors.has(pid)) throw Error(image.name + ': patient ' + pid + ' missing from risk_factor_5.xlsx');
    samples.push({image, name: stem + '.csv', clinical: factors.get(pid),
      disk: await diskDirectory.getFileHandle(stem + '_disk.jpg'),
      lesion: await lesionDirectory.getFileHandle(stem + '_lesion.jpg')});
  }
  const destinations = {};
  for (const name of ['input_0', 'input_1', 'input_2'])
    destinations[name] = await outputDirectory.getDirectoryHandle(name, {create: true});
  const outputs = [];
  for (const sample of samples) {
    try {
      const rgb = await readImage(sample.image), disk = await readImage(sample.disk, true), lesion = await readImage(sample.lesion, true);
      const rows = Array.from({length: 5}, () => new Float32Array(area));
      for (let i = 0; i < area; i++) {
        for (let c = 0; c < 3; c++) {
          const value = Math.fround(rgb[i * 3 + c] / 255);
          rows[c][i] = normalize ? Math.fround(Math.fround(value - mean[c]) / std[c]) : value;
        }
        for (const [channel, mask] of [[3, disk], [4, lesion]]) {
          const sum = Math.fround(Math.fround(Math.fround(mask[i * 3] / 255) + Math.fround(mask[i * 3 + 1] / 255)) + Math.fround(mask[i * 3 + 2] / 255));
          const binary = Number(Math.fround(sum / 3) > 0.5);
          rows[channel][i] = channel === 3 ? 1 - binary : binary;
        }
      }
      outputs.push({
        input_0: await writeCSV(destinations.input_0, sample.name, rows),
        input_1: await writeCSV(destinations.input_1, sample.name, [sample.clinical.nonInv]),
        input_2: await writeCSV(destinations.input_2, sample.name, [sample.clinical.inv]),
      });
    } catch (error) { throw new Error(sample.image.name + ': ' + error.message, {cause: error}); }
  }
  return outputs;
}
