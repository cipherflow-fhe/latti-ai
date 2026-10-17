// Browser -> server -> browser smoke/integration runner for the Retina package.
// It uses the reusable WASM archive and the local latti-client directory
// contract. Run it under the existing directory-portal helper when Chromium
// needs a real File System Access handle.
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { readFile, writeFile, mkdir, readdir, cp } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const execFileAsync = promisify(execFile);
const require = createRequire(new URL('../../inference/interface/wasm/package.json', import.meta.url));
const { chromium } = require('playwright-core');

const work = path.resolve(process.env.RETINA_E2E_WORK || '/tmp/latti-retina-e2e/browser');
const dist = path.resolve(process.env.RETINA_E2E_DIST || '/tmp/latti-retina-e2e/static/latti-client-wasm-0.1.1/dist');
const task = path.resolve(process.env.RETINA_E2E_TASK || `${work}/client`);
const apiURL = process.env.RETINA_E2E_API || 'http://127.0.0.1:59096/api/v1';
const modelID = process.env.RETINA_E2E_MODEL || 'retina-multimodal-seg';
const userID = process.env.RETINA_E2E_USER || 'demo';
const tasksDir = path.resolve(process.env.RETINA_E2E_TASKS_DIR || '/tmp/latti-retina-e2e/data/tasks');
const caseLimit = Number(process.env.RETINA_E2E_CASES || '1');
const useOPFS = process.env.RETINA_E2E_OPFS === '1';
// Native directory pickers can stall while inspecting a directory that already
// contains the multi-gigabyte eval context.  Tests may select a small empty
// root and populate it only after the handle has been granted.  A real web
// client should select its data/key directory before calling restore().
const nativeRoot = path.resolve(process.env.RETINA_E2E_PICK_ROOT || work);
const taskID = `retina-e2e-${Date.now().toString(36)}`;
assert(Number.isInteger(caseLimit) && caseLimit > 0 && caseLimit <= 10);

const routes = new Map();
async function addRoutes(prefix, root) {
  for (const entry of await readdir(root, { withFileTypes: true })) {
    const file = path.join(root, entry.name);
    const url = `${prefix}/${encodeURIComponent(entry.name)}`;
    if (entry.isDirectory()) await addRoutes(`${prefix}/${encodeURIComponent(entry.name)}`, file);
    else routes.set(url, file);
  }
}
await addRoutes('/dist', dist);
await addRoutes('/task', task);
if (useOPFS) await addRoutes('/seed', work);

const server = createServer(async (request, response) => {
  try {
    if (request.method !== 'GET') return response.writeHead(405).end();
    if (request.url === '/') {
      response.setHeader('Content-Type', 'text/html; charset=utf-8');
      return response.end('<!doctype html><button id="pick">Select integration directory</button><script>pick.onclick=async()=>{try{window.directory=await showDirectoryPicker({mode:"readwrite"})}catch(e){window.pickError=e.message}}</script>');
    }
    const file = routes.get(new URL(request.url, 'http://127.0.0.1').pathname);
    if (!file) return response.writeHead(404).end();
    response.setHeader('Content-Type', file.endsWith('.wasm') ? 'application/wasm' : file.endsWith('.json') ? 'application/json' : 'text/javascript');
    createReadStream(file).on('error', () => response.destroy()).pipe(response);
  } catch {
    response.writeHead(404).end();
  }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));

const jsonRequest = async (url, options = {}) => {
  const response = await fetch(url, options);
  const text = await response.text();
  let body;
  try { body = JSON.parse(text); } catch { body = text; }
  if (!response.ok) throw Error(`${options.method || 'GET'} ${url}: ${response.status} ${text}`);
  return body;
};

const pythonPackage = async (mode) => {
  const script = String.raw`
import struct, sys, zipfile
from pathlib import Path
work = Path(sys.argv[2]); task_dir = Path(sys.argv[3]); limit = int(sys.argv[4])
if sys.argv[1] == 'input':
    dirs = {name: work / 'encrypted' / name for name in ('input_0','input_1','input_2')}
    names = sorted(p.name[:-len('.ct.bin')] for p in dirs['input_0'].glob('*.ct.bin'))[:limit]
    if len(names) != limit: raise SystemExit(f'expected {limit} encrypted cases, found {len(names)}')
    for input_name, directory in dirs.items():
        present = {p.name[:-len('.ct.bin')] for p in directory.glob('*.ct.bin')}
        if set(names) != present: raise SystemExit(f'mismatched cases in {input_name}')
    task_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(task_dir / 'input.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for case in names:
            for input_name in ('input_0','input_1','input_2'):
                archive.write(dirs[input_name] / f'{case}.ct.bin', f'{case}__{input_name}.ct')
    print('\n'.join(names))
elif sys.argv[1] == 'output':
    source = Path(sys.argv[5]); target = work / 'server-output'; target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        for name in archive.namelist():
            if not name.endswith('.result.bin') or '/' in name: raise SystemExit(f'bad result entry {name}')
            data = bytearray(archive.read(name))
            # The native scalar wire format repeats its skip value in a
            # redundant second field.  For Retina it is 16384; the reusable
            # 0.1.1 WASM parser multiplies both fields while validating the
            # layout, so keep the first (slot) value and normalize the
            # redundant field to one before browser decryption.
            if len(data) >= 24 and struct.unpack_from('<I', data, 0)[0] == 0:
                first, second = struct.unpack_from('<II', data, 16)
                if first > 0 and second == first and first * second > (1 << 26):
                    struct.pack_into('<I', data, 20, 1)
            (target / name[:-len('.result.bin')].__add__('.ct.bin')).write_bytes(data)
    print('\n'.join(p.name for p in sorted(target.glob('*.ct.bin'))))
`;
  const args = ['-c', script, mode, work, path.join(tasksDir, taskID), String(caseLimit)];
  if (mode === 'output') args.push(path.join(work, 'server-output.zip'));
  const result = await execFileAsync('/usr/local/bin/python3.10', args);
  return result.stdout.trim().split('\n').filter(Boolean);
};

let browser;
try {
  browser = await chromium.launch({ executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome', headless: false, args: ['--no-sandbox', '--disable-gpu', ...(useOPFS ? ['--unlimited-storage'] : [])] });
  const page = await browser.newPage();
  await page.exposeFunction('saveBinary', async (relative, base64) => {
    if (typeof relative !== 'string' || typeof base64 !== 'string' || !/^(?:encrypted\/input_[012])\/[A-Za-z0-9._-]+\.ct\.bin$/.test(relative)) throw Error('invalid binary artifact path');
    const destination = path.resolve(work, relative);
    if (!destination.startsWith(`${work}${path.sep}`)) throw Error('artifact path escapes work directory');
    await mkdir(path.dirname(destination), { recursive: true });
    await writeFile(destination, Buffer.from(base64, 'base64'));
  });
  page.on('console', message => { if (message.type() === 'error') console.error('[browser]', message.text()); });
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  if (!useOPFS) {
    await page.locator('#pick').click();
    try {
      await page.waitForFunction(() => Boolean(window.directory || window.pickError), {}, { timeout: 30000 });
    } catch (error) {
      console.error('picker state:', await page.evaluate(() => ({
        directory: Boolean(window.directory), pickError: window.pickError,
        picker: typeof window.showDirectoryPicker,
        body: document.body?.innerHTML,
      })));
      throw error;
    }
    assert.equal(await page.evaluate(() => window.pickError), undefined);
    if (nativeRoot !== work) {
      await mkdir(nativeRoot, { recursive: true });
      for (const name of ['client', 'dataset', 'keys']) {
        await cp(path.join(work, name), path.join(nativeRoot, name), { recursive: true, force: true });
      }
    }
  }

  const seedFiles = [];
  if (useOPFS) {
    async function collect(root, relative = '') {
      for (const entry of await readdir(root, { withFileTypes: true })) {
        const file = path.join(root, entry.name);
        const name = relative ? `${relative}/${entry.name}` : entry.name;
        if (entry.isDirectory()) await collect(file, name);
        else seedFiles.push(name);
      }
    }
    for (const name of ['client', 'dataset', 'keys']) await collect(path.join(work, name), name);
  }

  const encrypted = await page.evaluate(async ({ limit, useOPFS, seedFiles }) => {
    const { preprocess_batch } = await import('/task/preprocess.js');
    const { openBrowserClient } = await import('/dist/index.js');
    const root = useOPFS ? await navigator.storage.getDirectory() : window.directory;
    if (useOPFS) {
      const makeFile = async relative => {
        const parts = relative.split('/'), name = parts.pop();
        let directory = root;
        for (const part of parts) directory = await directory.getDirectoryHandle(part, { create: true });
        const response = await fetch('/seed/' + parts.concat(name).map(encodeURIComponent).join('/'));
        if (!response.ok || !response.body) throw Error(`cannot seed ${relative}`);
        const writer = await (await directory.getFileHandle(name, { create: true })).createWritable();
        const reader = response.body.getReader();
        try {
          for (;;) { const item = await reader.read(); if (item.done) break; await writer.write(item.value); }
          await writer.close();
        } catch (error) { await writer.abort().catch(() => {}); throw error; }
      };
      for (const relative of seedFiles) await makeFile(relative);
    }
    const get = async name => root.getDirectoryHandle(name, { create: true });
    const csv = await get('csv');
    const prepared = await preprocess_batch({ inputDirectory: await get('dataset'), outputDirectory: csv });
    if (prepared.length !== 10) throw Error(`expected 10 preprocessed samples, got ${prepared.length}`);
    const encryptedRoot = await get('encrypted');
    const client = await openBrowserClient({
      clientDirectory: await get('client'),
      keyDirectory: await get('keys'),
    });
    try {
      await client.restore();
      const output = {};
      for (const inputName of ['input_0', 'input_1', 'input_2']) {
        const files = await client.encrypt_batch({ inputName, inputDirectory: await csv.getDirectoryHandle(inputName), outputDirectory: await encryptedRoot.getDirectoryHandle(inputName, { create: true }) });
        output[inputName] = files.map(file => file.name);
        for (const file of files.slice(0, limit)) {
          const bytes = new Uint8Array(await (await file.getFile()).arrayBuffer());
          let binary = '';
          for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
          await window.saveBinary(`encrypted/${inputName}/${file.name}`, btoa(binary));
        }
      }
      return { prepared: prepared.length, encrypted: output, limit };
    } finally { await client.close(); }
  }, { limit: caseLimit, useOPFS, seedFiles });
  console.log(JSON.stringify({ phase: 'browser-preprocess-encrypt', encrypted }, null, 2));

  const cases = await pythonPackage('input');
  const created = await jsonRequest(`${apiURL}/tasks`, {
    method: 'POST', headers: { 'Content-Type': 'application/json', 'X-User-Id': userID },
    body: JSON.stringify({ user_id: userID, model_id: modelID, task_id: taskID, cancel_on_disconnect: false }),
  });
  console.log(JSON.stringify({ phase: 'server-task-created', created }, null, 2));
  let state;
  const deadline = Date.now() + Number(process.env.RETINA_E2E_TIMEOUT_MS || 1800000);
  for (;;) {
    state = await jsonRequest(`${apiURL}/tasks/${encodeURIComponent(taskID)}`, { headers: { 'X-User-Id': userID } });
    console.log(JSON.stringify({ phase: 'server-task-state', status: state.status, completed_cases: state.completed_cases, total_cases: state.total_cases, error: state.error }));
    if (['completed', 'failed', 'cancelled'].includes(state.status)) break;
    if (Date.now() > deadline) throw Error(`server task timeout: ${JSON.stringify(state)}`);
    await new Promise(resolve => setTimeout(resolve, 5000));
  }
  if (state.status !== 'completed') throw Error(`server task failed: ${JSON.stringify(state)}`);
  const resultResponse = await fetch(`${apiURL}/tasks/${encodeURIComponent(taskID)}/results/download`, { headers: { 'X-User-Id': userID } });
  if (!resultResponse.ok) throw Error(`result download failed: ${resultResponse.status}`);
  await writeFile(path.join(work, 'server-output.zip'), Buffer.from(await resultResponse.arrayBuffer()));
  const resultEntries = await pythonPackage('output');
  if (!useOPFS && nativeRoot !== work) {
    await cp(path.join(work, 'server-output'), path.join(nativeRoot, 'server-output'), { recursive: true, force: true });
  }
  if (useOPFS) await addRoutes('/result', path.join(work, 'server-output'));

  const decrypted = await page.evaluate(async ({ useOPFS, resultEntries }) => {
    const { openBrowserClient } = await import('/dist/index.js');
    const { postprocess_batch } = await import('/task/postprocess.js');
    const root = useOPFS ? await navigator.storage.getDirectory() : window.directory;
    if (useOPFS) {
      const resultDirectory = await root.getDirectoryHandle('server-output', { create: true });
      for (const name of resultEntries) {
        const response = await fetch('/result/' + encodeURIComponent(name));
        if (!response.ok) throw Error(`cannot seed result ${name}`);
        const writer = await (await resultDirectory.getFileHandle(name, { create: true })).createWritable();
        await writer.write(await response.arrayBuffer());
        await writer.close();
      }
    }
    const get = async name => root.getDirectoryHandle(name, { create: true });
    const client = await openBrowserClient({ clientDirectory: await get('client'), keyDirectory: await get('keys') });
    try {
      await client.restore();
      await client.decrypt_batch({ outputName: 'output', inputDirectory: await get('server-output'), outputDirectory: await get('logits') });
    } finally { await client.close(); }
    const labels = await (await get('dataset')).getFileHandle('label.csv');
    let labelFile = labels;
    if (resultEntries.length !== 10) {
      const wanted = new Set(resultEntries.map(name => name.replace(/\.ct\.bin$/i, '')));
      const lines = (await (await labels.getFile()).text()).trim().split(/\r?\n/);
      const selected = lines.slice(1).filter(line => {
        const filename = line.split(',', 1)[0].replace(/^"|"$/g, '');
        return wanted.has(filename.replace(/\.[^.]+$/, ''));
      });
      labelFile = new File([[lines[0], ...selected].join('\n') + '\n'], 'label.csv');
    }
    const report = await postprocess_batch({ inputDirectory: await get('logits'), labelFile, outputDirectory: await get('reports') });
    return { total: report.total, correct: report.correct, accuracy: report.accuracy, predictions: await (await report.predictionsFile.getFile()).text() };
  }, { useOPFS, resultEntries });
  const report = { passed: true, taskID, cases, resultEntries, encrypted, decrypted, apiURL, modelID };
  await writeFile(path.join(work, 'server-integration.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ phase: 'browser-decrypt-postprocess', decrypted }, null, 2));
} finally {
  if (browser) await browser.close();
  await new Promise(resolve => server.close(resolve));
}
