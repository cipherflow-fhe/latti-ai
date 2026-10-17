// Retina-DKD: two logits in [DN, NDRD] order; ties select class 0, like torch.argmax.
function parseCSV(text) {
  const rows = [];
  let row = [], field = '', quoted = false, closed = false;
  const endField = () => { row.push(field); field = ''; closed = false; };
  const endRow = () => { endField(); if (row.some(value => value !== '')) rows.push(row); row = []; };
  text = text.replace(/^\uFEFF/, '');
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; } else { quoted = false; closed = true; }
      } else field += c;
    } else if (c === ',') endField();
    else if (c === '\n' || c === '\r') { endRow(); if (c === '\r' && text[i + 1] === '\n') i++; }
    else if (c === '"' && !field && !closed) quoted = true;
    else {
      if (closed || c === '"') throw Error('Invalid CSV quoting');
      field += c;
    }
  }
  if (quoted) throw Error('Unclosed CSV quote');
  if (field || row.length || closed) endRow();
  return rows;
}

const escapeCSV = value => /[",\r\n]/.test(value) ? '"' + value.replaceAll('"', '""') + '"' : value;
const numeric = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;

export async function postprocess_batch({inputDirectory, labelFile, outputDirectory}) {
  const file = labelFile.kind === 'file' ? await labelFile.getFile() : labelFile;
  const [header, ...rows] = parseCSV(await file.text());
  if (header?.length !== 2 || header[0] !== 'filename' || header[1] !== 'label') throw Error('Expected filename,label header');
  const labels = new Map(), names = new Set();
  for (const row of rows) {
    const [filename, value] = row, label = Number(value);
    if (row.length !== 2 || !filename || /[\/\\\0]/.test(filename) || !/\.(png|jpe?g)$/i.test(filename)) throw Error('Invalid image filename in label table');
    if (!/^[01]$/.test(value.trim())) throw Error('Expected label 0 (DN) or 1 (NDRD): ' + filename);
    const name = filename.replace(/\.(png|jpe?g)$/i, '.csv');
    if (names.has(name.toLowerCase())) throw Error('Duplicate labeled sample: ' + filename);
    names.add(name.toLowerCase()); labels.set(name, {filename, label});
  }
  if (!labels.size) throw Error('At least one labeled sample is required');
  const files = [];
  for await (const entry of inputDirectory.values()) if (entry.kind === 'file' && /\.csv$/i.test(entry.name)) files.push(entry);
  files.sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0);
  if (files.length !== labels.size || files.some(entry => !labels.has(entry.name))) throw Error('Labels and logits must contain the same samples');
  const predictionsFile = await outputDirectory.getFileHandle('predictions.csv', {create: true});
  const writer = await predictionsFile.createWritable();
  let correct = 0;
  try {
    await writer.write('filename,label,prediction,correct\n');
    for (const entry of files) {
      try {
        const fields = parseCSV(await (await entry.getFile()).text()).flat();
        if (fields.length !== 2 || fields.some(value => !numeric.test(value.trim()) || !Number.isFinite(Number(value)))) throw Error('Expected 2 finite logits');
        const prediction = Number(Number(fields[1]) > Number(fields[0]));
        const {filename, label} = labels.get(entry.name), matched = Number(prediction === label);
        correct += matched;
        await writer.write(`${escapeCSV(filename)},${label},${prediction},${matched}\n`);
      } catch (error) { throw new Error(entry.name + ': ' + error.message, {cause: error}); }
    }
    await writer.close();
  } catch (error) { await writer.abort().catch(() => {}); throw error; }
  return {predictionsFile, total: files.length, correct, accuracy: correct / files.length};
}
