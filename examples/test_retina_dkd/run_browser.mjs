// Generate the deliverable with the actual browser modules and OPFS handles.
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile, readdir, mkdir, writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import path from 'node:path';

const require = createRequire(new URL('../../inference/interface/wasm/package.json', import.meta.url));
const {chromium} = require('playwright-core');
const demo = path.resolve(process.argv[2] || '/LattiAI-Task-Resources/retina_degree8_02/task/client/demo_10');
const routes = new Map([
  ['/modules/preprocess.js', path.join(import.meta.dirname, 'task/client/preprocess.js')],
  ['/modules/postprocess.js', path.join(import.meta.dirname, 'task/client/postprocess.js')],
]);
const inputs = [];
async function collect(relative) {
  for (const entry of await readdir(path.join(demo, relative), {withFileTypes: true})) {
    const name = relative + '/' + entry.name;
    if (entry.isDirectory()) await collect(name);
    else if (entry.isFile()) { inputs.push(name); routes.set('/files/' + name, path.join(demo, name)); }
  }
}
await collect('dataset');
await collect('reference_logits');
try { await collect('validation/fixtures'); } catch (error) { if (error.code !== 'ENOENT') throw error; }

// Only exact, pre-enumerated files are served on loopback; there is no upload route.
const server = createServer(async (request, response) => {
  try {
    if (request.method !== 'GET' || request.headers['sec-fetch-site'] === 'cross-site') {
      response.writeHead(403).end(); return;
    }
    const name = decodeURIComponent(new URL(request.url, 'http://127.0.0.1').pathname);
    if (name === '/') { response.setHeader('Content-Type', 'text/html'); response.end('<!doctype html><title>Retina preprocessing</title>'); return; }
    const file = routes.get(name);
    if (!file) { response.writeHead(404).end(); return; }
    response.setHeader('Content-Type', name.endsWith('.js') ? 'text/javascript' : 'application/octet-stream');
    response.end(await readFile(file));
  } catch { response.writeHead(404).end(); }
});
server.listen(0, '127.0.0.1');
await new Promise(resolve => server.once('listening', resolve));
let browser;
try {
  browser = await chromium.launch({executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome',
    headless: true, args: ['--no-sandbox']});
  const page = await browser.newPage();
  page.on('console', message => console.log('[browser]', message.text()));
  // Playwright writes only known artifact subdirectories, without a network writer.
  await page.exposeFunction('saveArtifact', async (relative, text) => {
    if (typeof relative !== 'string' || typeof text !== 'string' ||
        !/^(?:csv\/input_[012]|reports|validation\/(?:normalized|unnormalized)\/input_[012])\/[^/\\\0]+\.csv$/.test(relative))
      throw Error('Invalid artifact path');
    const destination = path.join(demo, relative);
    await mkdir(path.dirname(destination), {recursive: true});
    await writeFile(destination, text);
    console.log('wrote', relative);
  });
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const report = await page.evaluate(async inputs => {
    const {preprocess_batch} = await import('/modules/preprocess.js');
    const {postprocess_batch} = await import('/modules/postprocess.js');
    const root = await navigator.storage.getDirectory();
    const directory = async relative => {
      let current = root;
      for (const part of relative.split('/')) current = await current.getDirectoryHandle(part, {create: true});
      return current;
    };
    const write = async (dir, name, content) => {
      const writer = await (await dir.getFileHandle(name, {create: true})).createWritable();
      await writer.write(content); await writer.close();
    };
    for (const relative of inputs) {
      const parts = relative.split('/'), name = parts.pop();
      const response = await fetch('/files/' + relative.split('/').map(encodeURIComponent).join('/'));
      if (!response.ok) throw Error('Cannot read ' + relative);
      await write(await directory(parts.join('/')), name, await response.blob());
    }
    const exportCSV = async (dir, relative) => {
      for await (const entry of dir.values()) {
        if (entry.kind === 'directory') await exportCSV(entry, relative + '/' + entry.name);
        else if (entry.name.endsWith('.csv')) await window.saveArtifact(relative + '/' + entry.name, await (await entry.getFile()).text());
      }
    };
    const start = performance.now();
    const batches = {};
    for (const [input, output, normalize] of [
      ['dataset', 'csv', true],
      ...(inputs.some(name => name.startsWith('validation/fixtures/')) ? [
        ['validation/fixtures', 'validation/normalized', true],
        ['validation/fixtures', 'validation/unnormalized', false],
      ] : []),
    ]) {
      const result = await preprocess_batch({inputDirectory: await directory(input),
        outputDirectory: await directory(output), normalize});
      batches[output] = result.map(item => Object.fromEntries(Object.entries(item).map(([key, file]) => [key, file.name])));
      await exportCSV(await directory(output), output);
    }
    const assessment = await postprocess_batch({inputDirectory: await directory('reference_logits'),
      labelFile: await (await directory('dataset')).getFileHandle('label.csv'), outputDirectory: await directory('reports')});
    await exportCSV(await directory('reports'), 'reports');

    // Exercise argmax and failure paths on native file handles, independently of history.
    const checkInput = await directory('checks/logits'), checkOutput = await directory('checks/output');
    const checkLabels = text => new File(['filename,label\n' + text], 'label.csv');
    const assess = labelFile => postprocess_batch({inputDirectory: checkInput, labelFile, outputDirectory: checkOutput});
    const checks = [];
    const rejects = async (name, operation, pattern) => {
      try { await operation(); } catch (error) {
        if (!pattern.test(error.message)) throw error;
        checks.push(name); return;
      }
      throw Error('Expected rejection: ' + name);
    };
    await write(checkInput, 'test.csv', '3,3\n');
    if ((await assess(checkLabels('test.jpg,0\n'))).correct !== 1) throw Error('Argmax tie must select 0');
    checks.push('argmax tie selects class 0');
    await write(checkInput, 'test.csv', '-2,1e-3\n');
    if ((await assess(checkLabels('test.jpg,1\n'))).correct !== 1) throw Error('Argmax must select class 1');
    checks.push('argmax handles negative logits and scientific notation');
    for (const value of ['NaN,1', 'Infinity,1', '1,,2', '1,2,3', '']) {
      await write(checkInput, 'test.csv', value);
      await rejects('invalid logits: ' + JSON.stringify(value), () => assess(checkLabels('test.jpg,0\n')), /2 finite logits/);
    }
    await write(checkInput, 'test.csv', '0,1');
    await rejects('out of range label', () => assess(checkLabels('test.jpg,2\n')), /Expected label/);
    await rejects('missing labeled result', () => assess(checkLabels('missing.jpg,0\n')), /same samples/);
    await rejects('duplicate labeled basename', () => assess(checkLabels('test.jpg,0\ntest.png,1\n')), /Duplicate/);
    await rejects('empty label table', () => assess(checkLabels('')), /At least one/);
    await rejects('unsafe filename', () => assess(checkLabels('../test.jpg,0\n')), /Invalid image filename/);
    await rejects('unclosed CSV quote', () => assess(checkLabels('"test.jpg,0\n')), /Unclosed/);

    const dataset = await directory('dataset'), images = await dataset.getDirectoryHandle('images');
    await write(images, '18-F687365-22631.png', await (await images.getFileHandle('18-F687365-22631.jpg')).getFile());
    await rejects('preprocess duplicate basename', () => preprocess_batch({inputDirectory: dataset,
      outputDirectory: checkOutput}), /duplicate output name/);
    await images.removeEntry('18-F687365-22631.png');
    return {browser: navigator.userAgent, batches, checks, seconds: (performance.now() - start) / 1000,
      accuracy_basis: 'Provided historical logits; this run does not execute classification inference',
      total: assessment.total, correct: assessment.correct, accuracy: assessment.accuracy};
  }, inputs);
  assert.equal(report.total, 10); assert.equal(report.correct, 9); assert.equal(report.accuracy, 0.9);
  assert.equal(report.batches.csv.length, 10);
  await mkdir(path.join(demo, 'validation'), {recursive: true});
  await writeFile(path.join(demo, 'validation/browser.json'), JSON.stringify(report, null, 2) + '\n');
  await writeFile(path.join(demo, 'reports/accuracy.json'), JSON.stringify({total: report.total, correct: report.correct,
    accuracy: report.accuracy, accuracy_basis: report.accuracy_basis}, null, 2) + '\n');
  console.log(`Browser finished: ${report.correct}/${report.total}, ${report.accuracy * 100}%, ${report.checks.length} checks`);
} finally {
  if (browser) await browser.close();
  await new Promise(resolve => server.close(resolve));
}
