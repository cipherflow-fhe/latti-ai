# Latti Browser WASM Client

This package provides the browser client for local key generation, context persistence, CSV encryption, and result decryption. It is intended for a desktop Chromium web application running from `https://` or `http://localhost`.

## Package contents

```text
latti-client-wasm-<version>/
  dist/
    index.js           # browser API entry point
    index.d.ts         # TypeScript declarations
    worker.js          # Dedicated Worker adapter
    csv.js             # streamed CSV parser
    wasm_exec.js       # Go WASM runtime
    latti_client.wasm  # FHE client core
  package.json
  SHA256SUMS
  README.md
```

The package does not need a Node.js runtime or a locally installed native proxy. The web application imports `dist/index.js`; that module starts the Worker and the Worker loads the WASM runtime.

## Browser requirements

- Desktop Chromium with File System Access API support.
- A secure context: HTTPS or `http://localhost`.
- The static server must serve `.wasm` with `application/wasm` and allow module Workers.
- The selected client directory must contain `task_config.json` and `ckks_parameter.json` before `openBrowserClient()` is called.

The task files determine CKKS/BTP mode, parameters, packing, and input layout. The page API does not override those values.

## Main call flow

```js
import { openBrowserClient } from './vendor/latti-client-wasm/dist/index.js';

async function hasFile(directory, name) {
  try {
    await directory.getFileHandle(name);
    return true;
  } catch (error) {
    if (error.name === 'NotFoundError') return false;
    throw error;
  }
}

const clientDirectory = await window.showDirectoryPicker({ mode: 'readwrite' });
// Put task_config.json and ckks_parameter.json in clientDirectory first.

const client = await openBrowserClient({
  clientDirectory,
  onProgress: progress => console.log(progress.phase, progress),
});

try {
  if (await hasFile(clientDirectory, 'full_context.bin')) {
    await client.restore();
  } else {
    const { evalContext, fullContext } = await client.setup();
    // Upload evalContext to the service when the task needs it.
    // Keep fullContext.bin in the user's local directory.
    console.log('created', evalContext.name, fullContext.name);
  }

  const [inputFile] = await window.showOpenFilePicker({
    types: [{ description: 'CSV', accept: { 'text/csv': ['.csv'] } }],
  });
  const outputDirectory = await window.showDirectoryPicker({ mode: 'readwrite' });
  const ciphertextFile = await client.encrypt({
    inputName: 'input', // must match task_config.json
    inputFile,
    outputDirectory,
  });

  // Send eval_context.bin and input.ct.bin to the service as needed.
  // Never upload full_context.bin: it contains the local secret key.
  console.log('created', ciphertextFile.name);

  // After the service writes its output to a local file, decrypt it:
  // const outputFile = await client.decrypt({
  //   outputName: 'output',
  //   ciphertextFile: serverOutputFile,
  //   outputDirectory,
  // });
} finally {
  await client.close();
}
```

`setup()` writes `eval_context.bin` and `full_context.bin` into the client directory. `restore()` loads `full_context.bin` after a page refresh or a later session. `encrypt()` writes `<inputName>.ct.bin`; `decrypt()` writes `<outputName>.csv`. These methods return `FileSystemFileHandle` values and do not return complete file contents to the page.

The file arguments may be `FileSystemFileHandle` values or `File` values. The client serializes operations through one Worker session, so calls on one client should be awaited in order.

## Using the ZIP in another web project

Unzip the archive into a static asset directory and import `dist/index.js` from the application. With a bundler, keep the relative `dist/worker.js`, `dist/wasm_exec.js`, and `dist/latti_client.wasm` assets available; with a plain ES-module application, serve the extracted directory unchanged and use the relative import shown above.

The archive's `SHA256SUMS` file covers the runtime files in `dist/`.
