#!/usr/bin/env node
'use strict';

const path = require('path');
const fs = require('fs');

// latti_client.node is dynamically linked against libfhe_ops_lib.so.
// Ensure LD_LIBRARY_PATH includes the directory containing it, e.g.:
//   LD_LIBRARY_PATH=.../build/inference/lattisense/fhe_ops_lib node test_client.js
const { InferenceClient } = require('./build/Release/latti_client.node');

const TEST_TASK_DIR = path.join(__dirname, '..', 'python', 'client', 'testdata');

async function testImport() {
    console.log('[TEST] import');
    const client = new InferenceClient(TEST_TASK_DIR);
    console.log('[PASS] InferenceClient constructor');
    return client;
}

async function testSetup(client) {
    console.log('\n[TEST] setup (key generation)');
    const start = Date.now();
    await client.setup();
    console.log(`[PASS] setup (${((Date.now() - start) / 1000).toFixed(1)}s)`);
}

async function testExportEvalContext(client) {
    console.log('\n[TEST] exportEvalContext');
    const start = Date.now();
    const evalCtx = await client.exportEvalContext();
    console.log(`[PASS] exportEvalContext (${(evalCtx.length / 1024 / 1024).toFixed(1)} MB, ${((Date.now() - start) / 1000).toFixed(1)}s)`);
    return evalCtx;
}

async function testEncrypt(client) {
    console.log('\n[TEST] encrypt');
    const csvPath = path.join(TEST_TASK_DIR, 'img.csv');
    if (!fs.existsSync(csvPath)) {
        console.log(`[SKIP] ${csvPath} not found`);
        return null;
    }
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
    if (maxError >= 0.01) {
        console.log(`[FAIL] max error too large: ${maxError}`);
        process.exit(1);
    }
    console.log(`[PASS] roundtrip verified (maxError=${maxError.toFixed(6)})`);
}

async function testExportFullContext(client, evalCtx) {
    console.log('\n[TEST] exportFullContext');
    const fullCtx = await client.exportFullContext();
    // full_ctx includes the secret key; eval_ctx does not. So full_ctx must be strictly larger.
    if (fullCtx.length <= evalCtx.length) {
        throw new Error(`fullCtx (${fullCtx.length}) should be larger than evalCtx (${evalCtx.length})`);
    }
    const deltaMb = (fullCtx.length - evalCtx.length) / 1024 / 1024;
    console.log(`[PASS] exportFullContext (${(fullCtx.length / 1024 / 1024).toFixed(1)} MB, +${deltaMb.toFixed(1)} MB vs eval)`);
    return fullCtx;
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
    if (maxError >= 0.01) {
        throw new Error(`max error too large: ${maxError}`);
    }
    console.log(`[PASS] loadFullContext roundtrip (maxError=${maxError.toFixed(6)})`);
}

async function testReleaseThenReloadRoundtrip(taskDir, fullCtx) {
    console.log('\n[TEST] release + reload roundtrip');
    const { InferenceClient } = require('./build/Release/latti_client.node');
    const c = new InferenceClient(taskDir);
    await c.setup();
    await c.release();                  // frees freshly-generated keys
    await c.loadFullContext(fullCtx);   // restore from saved bytes

    const csvPath = path.join(taskDir, 'img.csv');
    const encrypted = await c.encrypt({input: csvPath});
    const result = await c.decrypt({output: encrypted.input});

    const csv = fs.readFileSync(csvPath, 'utf8');
    const original = csv.trim().split(',').map(Number);
    const decrypted = Array.from(result.output.output).slice(0, original.length);
    const maxError = Math.max(...original.map((v, i) => Math.abs(v - decrypted[i])));
    if (maxError >= 0.01) {
        throw new Error(`max error too large: ${maxError}`);
    }
    console.log(`[PASS] release + reload roundtrip (maxError=${maxError.toFixed(6)})`);
}

async function main() {
    console.log('Testing latti_client N-API addon\n');
    const client = await testImport();
    await testSetup(client);
    const evalCtx = await testExportEvalContext(client);
    const fullCtx = await testExportFullContext(client, evalCtx);
    const encrypted = await testEncrypt(client);
    if (encrypted) {
        await testDecrypt(client, encrypted);
        await testLoadFullContextRoundtrip(TEST_TASK_DIR, fullCtx);
        await testReleaseThenReloadRoundtrip(TEST_TASK_DIR, fullCtx);
    }
    console.log('\n=== ALL TESTS PASSED ===\n');
}

main().catch(err => {
    console.error('[FATAL]', err);
    process.exit(1);
});
