#!/usr/bin/env node
'use strict';

const path = require('path');
const fs = require('fs');
const os = require('os');
const assert = require('assert');

// latti_client.node is dynamically linked against libfhe_ops_lib.so.
// Ensure LD_LIBRARY_PATH includes the directory containing it, e.g.:
//   LD_LIBRARY_PATH=.../build/inference/lattisense/fhe_ops_lib node test_client.js
const { InferenceClient } = require('./build/Release/latti_client.node');

const TEST_TASK_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'latti-streaming-node-'));
fs.cpSync(path.join(__dirname, '..', 'python', 'client', 'testdata'), TEST_TASK_DIR, {recursive: true});
if (process.argv.includes('--btp')) {
    const configPath = path.join(TEST_TASK_DIR, 'task_config.json');
    const config = JSON.parse(fs.readFileSync(configPath));
    config.use_btp = true;
    fs.writeFileSync(configPath, JSON.stringify(config));
    const parametersPath = path.join(TEST_TASK_DIR, 'ckks_parameter.json');
    const parameters = JSON.parse(fs.readFileSync(parametersPath));
    parameters.param0.poly_modulus_degree = 65536;
    fs.writeFileSync(parametersPath, JSON.stringify(parameters));
}

async function testImport() {
    console.log('[TEST] import');
    const client = new InferenceClient(TEST_TASK_DIR);
    console.log('[PASS] InferenceClient constructor');
    return client;
}

async function testSetup(client) {
    console.log('\n[TEST] setup (key generation)');
    const start = Date.now();
    const promise = client.setup();
    assert(promise instanceof Promise);
    await promise;
    assert(fs.statSync(path.join(TEST_TASK_DIR, 'eval_context.bin')).size > 0);
    console.log(`[PASS] setup (${((Date.now() - start) / 1000).toFixed(1)}s)`);
}

async function testEncrypt(client) {
    console.log('\n[TEST] encrypt');
    const csvPath = path.join(TEST_TASK_DIR, 'img.csv');
    assert(fs.existsSync(csvPath), `missing test input: ${csvPath}`);
    const start = Date.now();
    const encrypted = await client.encrypt({input: csvPath});
    const ct = encrypted.input;
    console.log(`[PASS] encrypt (${(ct.length / 1024).toFixed(1)} KB, ${((Date.now() - start) / 1000).toFixed(1)}s)`);
    return encrypted;
}

async function testDecrypt(client, encrypted) {
    console.log('\n[TEST] decrypt (encrypt-decrypt roundtrip)');
    const start = Date.now();
    const result = await client.decrypt({output: encrypted.input});
    const output = result.output;
    console.log(`[PASS] decrypt (numOutputs=${output.numOutputs}, ${((Date.now() - start) / 1000).toFixed(1)}s)`);

    // Verify against original CSV
    const csv = fs.readFileSync(path.join(TEST_TASK_DIR, 'img.csv'), 'utf8');
    const original = csv.trim().split(',').map(Number);
    const decrypted = Array.from(output.output).slice(0, original.length);
    const maxError = Math.max(...original.map((v, i) => Math.abs(v - decrypted[i])));
    if (!Number.isFinite(maxError) || maxError >= 0.01) {
        throw new Error(`max error too large: ${maxError}`);
    }
    console.log(`[PASS] roundtrip verified (maxError=${maxError.toFixed(6)})`);
}

async function testExportFullContext(client) {
    console.log('\n[TEST] exportFullContext (client SK/PK persistence)');
    const fullCtx = await client.exportFullContext();
    assert(fullCtx.length > 0);
    console.log(`[PASS] exportFullContext (${(fullCtx.length / 1024 / 1024).toFixed(1)} MB)`);
    return fullCtx;
}

async function testSetupFailure() {
    console.log('\n[TEST] setup publication failure rejects Promise');
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'latti-streaming-failure-'));
    try {
        fs.cpSync(TEST_TASK_DIR, directory, {recursive: true, filter: p => path.basename(p) !== 'eval_context.bin'});
        // Rename cannot publish a regular file over a directory; the old target must survive.
        fs.mkdirSync(path.join(directory, 'eval_context.bin'));
        const client = new InferenceClient(directory);
        await assert.rejects(client.setup());
        assert(fs.statSync(path.join(directory, 'eval_context.bin')).isDirectory());
        assert(!fs.readdirSync(directory).some(name => name.startsWith('.eval-context-')));
        console.log('[PASS] rejected; old target retained; temporary file removed');
    } finally {
        fs.rmSync(directory, {recursive: true, force: true});
    }
}

async function testLoadFullContextRoundtrip(taskDir, fullCtx) {
    console.log('\n[TEST] loadFullContext + encrypt + decrypt roundtrip (fresh client)');
    const { InferenceClient } = require('./build/Release/latti_client.node');
    const fresh = new InferenceClient(taskDir);
    await fresh.loadFullContext(fullCtx);

    const csvPath = path.join(taskDir, 'img.csv');
    const encrypted = await fresh.encrypt({input: csvPath});
    if (!encrypted.input || encrypted.input.length === 0) {
        throw new Error('empty ciphertext');
    }

    const result = await fresh.decrypt({output: encrypted.input});
    const csv = fs.readFileSync(csvPath, 'utf8');
    const original = csv.trim().split(',').map(Number);
    const decrypted = Array.from(result.output.output).slice(0, original.length);
    const maxError = Math.max(...original.map((v, i) => Math.abs(v - decrypted[i])));
    if (!Number.isFinite(maxError) || maxError >= 0.01) {
        throw new Error(`max error too large: ${maxError}`);
    }
    console.log(`[PASS] loadFullContext roundtrip (maxError=${maxError.toFixed(6)})`);
}

async function testReleaseThenReloadRoundtrip(c, taskDir, fullCtx) {
    console.log('\n[TEST] release + reload roundtrip');
    await c.release();                  // frees freshly-generated keys
    await c.loadFullContext(fullCtx);   // restore from saved bytes

    const csvPath = path.join(taskDir, 'img.csv');
    const encrypted = await c.encrypt({input: csvPath});
    const result = await c.decrypt({output: encrypted.input});

    const csv = fs.readFileSync(csvPath, 'utf8');
    const original = csv.trim().split(',').map(Number);
    const decrypted = Array.from(result.output.output).slice(0, original.length);
    const maxError = Math.max(...original.map((v, i) => Math.abs(v - decrypted[i])));
    if (!Number.isFinite(maxError) || maxError >= 0.01) {
        throw new Error(`max error too large: ${maxError}`);
    }
    console.log(`[PASS] release + reload roundtrip (maxError=${maxError.toFixed(6)})`);
}

async function main() {
    console.log('Testing latti_client N-API addon\n');
    const client = await testImport();
    await testSetup(client);
    const fullCtx = await testExportFullContext(client);
    const encrypted = await testEncrypt(client);
    await testDecrypt(client, encrypted);
    await testLoadFullContextRoundtrip(TEST_TASK_DIR, fullCtx);
    await testReleaseThenReloadRoundtrip(client, TEST_TASK_DIR, fullCtx);
    await testSetupFailure();
    console.log('\n=== ALL TESTS PASSED ===\n');
}

main().finally(() => fs.rmSync(TEST_TASK_DIR, {recursive: true, force: true})).catch(err => {
    console.error('[FATAL]', err);
    process.exit(1);
});
