// Run a Go wasm test binary while exposing its actual linear-memory size.
const fs = require('fs');
const path = require('path');
const {execFileSync} = require('child_process');
globalThis.require = require;
globalThis.fs = fs;
globalThis.TextEncoder = require('util').TextEncoder;
globalThis.TextDecoder = require('util').TextDecoder;
globalThis.performance ??= require('performance');
globalThis.crypto ??= require('crypto');
const goRoot = execFileSync('go', ['env', 'GOROOT'], {encoding: 'utf8'}).trim();
require(path.join(goRoot, 'lib', 'wasm', 'wasm_exec.js'));
const go = new Go();
go.argv = process.argv.slice(2);
go.env = {...process.env};
go.exit = process.exit;
WebAssembly.instantiate(fs.readFileSync(process.argv[2]), go.importObject).then(({instance}) => {
  globalThis.taaWasmMemoryBytes = () => instance.exports.mem.buffer.byteLength;
  return go.run(instance);
}).catch(error => { console.error(error); process.exitCode = 1; });
