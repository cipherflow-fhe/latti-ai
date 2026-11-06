# Latti browser WASM 0.2.0

The browser client uses Lattigo v6, Worker protocol 2, and OPFS for its working
files. Serve `dist/` over HTTPS or localhost in desktop Chromium with Dedicated
Workers, OPFS and Web Locks. User directories require File System Access handles.

```sh
npm ci
npm test
npm run build
npm run test:browser
npm run package -- /tmp/latti-packages
```

`package.json` supplies the runtime version. `dist/manifest.json` records the
actual build identity, source revisions, toolchain, default profile provenance
and file hashes. Verify an extracted package with `sha256sum -c SHA256SUMS`.
Packaging creates `latti-client-wasm-0.2.0.zip` and `WASM-0.2.0.zip`.
Task configurations, preprocessing scripts and authorized data are separate
artifacts with their own versions.

Open `dist/example.html` to select keys, bind a task, encrypt CSV directories,
save the input ZIP, decrypt returned results and export predictions with labels.
Directory and file pickers run from explicit button clicks.

Build the combined delivery after producing current tasks and prepared samples:

```sh
python3 package_delivery.py --wasm /tmp/latti-packages/WASM-0.2.0.zip \
  --cifar10-task /tmp/current-cifar/task --cifar10-data /tmp/prepared-cifar \
  --retina-task /tmp/current-retina/task --retina-data /tmp/prepared-retina \
  --retina-reference /task/gen_csv/gen_client_inputs_seg.py \
  --output /tmp/latti-delivery
```

The builder verifies runtime file hashes and Retina preprocessing provenance,
includes standalone task client modules and the original Retina reference,
and selects only files referenced by `cases.json`. It writes independently
content-versioned task/data ZIPs, a total manifest and `SHA256SUMS`, then
`latti-delivery-0.2.0.zip`. Prepared data directories must contain only samples
authorized for delivery. Private full contexts are never delivery inputs.

## Keys and task binding

```js
import {openBrowserClient} from '/dist/index.js';

// Call directory pickers from a user gesture.
const keyDirectory = await showDirectoryPicker({mode: 'readwrite'});
const client = await openBrowserClient({keyDirectory, onProgress: console.log});
const keys = await client.setup();
// Upload only keys.evalContext.file. keys.fullContext contains the private key.
await client.close();

const restored = await openBrowserClient({keyDirectory});
await restored.restore();
await restored.bindTask(taskClientDirectory);
```

Taskless generation defaults to the CI PN15Default profile derived at build time
from the current native parameter table. A `profile` option selects explicit
parameters and a bootstrapping plan. `profile.parameters` is the original task
parameter JSON text: pass it directly to Go; parsing and serializing its uint64
moduli through JavaScript numbers loses precision. Task-only opening reads
`task_config.json` and `ckks_parameter.json` directly from `clientDirectory`.

Keys reside beneath a profile-ID directory in the selected key directory:
`public/eval_context.bin`, `secret/full_context.bin`, and `manifest.json`.
Setup refuses to overwrite existing keys. Restore verifies the manifest and
reads the client-only full container. Binding compares actual parameters and
key capabilities; task-local IDs and layout names do not establish equality.
Run the current `inference/interface/gen_mega_ag.py` after compiling a task. It
exports the computation graph's `evaluation_keys` into the client configuration.
Binding requires this list and checks key kind, ring role, Galois element,
Q/P levels, base two decomposition and compression against the expanded profile.
Regenerate tasks missing this field before opening them in the browser.

## Files, batches and server ZIPs

```js
const encrypted = await restored.encrypt({
  inputName: 'input', inputFile: csvFile,
});
const plaintext = await restored.decrypt({
  outputName: 'output', ciphertextFile: resultFile,
});
console.log(encrypted.file, plaintext.file); // committed ArtifactRef handles

const inputBatch = await restored.encryptBatch({
  mode: 'server', inputDirectories: {input: csvDirectory},
});
// Upload inputBatch.archive.file using file.stream() or a File snapshot.
// Submit keys.evalContext.file separately to the actual deployment.
const results = await restored.decryptBatch({
  mode: 'server', outputName: 'output', inputFile: serverOutputZip,
  inputBatch,
});
```

Server inputs use `<case_id>.ct` for one input, or
`<case_id>__<input_name>.ct` for multiple inputs. The directory keys must match
the task input IDs exactly, with identical case sets. Results use
`<case_id>.result.bin`, checked against the committed input batch. Server mode
requires one selected task output; HTTP endpoints and authentication belong to
the actual deployment.

For directory batches, pass `inputDirectory` and the task `inputName` or
`outputName`. The result contains `directory`, `manifest`, decimal-string `count`
and an optional archive reference. Enumerate `entries.jsonl` as a stream for
individual files. Large batches return handles and summaries. An optional
`outputDirectory` receives a unique job subdirectory.

## Task preprocessing and classification

`dist/tasks/cifar10/preprocess.js` exports `preprocessDirectory({inputDirectory,
outputDirectory, check?})` and returns `{directory,count}`. It writes one CSV at
a time with Pillow bilinear resize semantics and the current training script's
float32 CHW/ImageNet normalization. The output directory must have no conflicting
CSV names. Empty directories, duplicate basenames and corrupt images fail.

`dist/tasks/{cifar10,retina}/postprocess.js` exports
`postprocess({results,labels,report,check?})`. Supply the decrypted BatchResult,
an independent UTF-8 JSONL File (`{"case_id":"001","label":0}` per line), and
a destination file handle. It validates exact case/label sets and finite logits,
writes one JSON prediction record per case, and returns `{total,correct,accuracy}`.
Ties select the first class. Classification consumes 10 CIFAR or 2 Retina logits.
Retina's existing offline preprocessing and three input directories are described
in `dist/tasks/retina/README.md`; the runtime does not replace segmentation tools.

CSV parsing, numeric staging, Feature serialization and ZIP32/ZIP64 operations
use bounded disk I/O. A single ciphertext/key and encoder workspace can exceed
the 4 MiB I/O chunk limit. ZIP DEFLATE input requires raw-DEFLATE support; output
uses STORE. Metadata and decompression budgets reject oversized or malformed
archives. Reader limits can be set explicitly through `openZip`.

`cancel(requestId)` cooperatively stops the selected operation at an I/O or
cryptographic boundary; progress reports its request ID. Omitting the ID selects
the active operation. `close()` cancels active work, settles pending requests and
terminates the Worker. Committed files persist; callers own their deletion.

## Validation scope

`npm test` covers Go streaming/profile/wire behavior and Node ZIP/protocol tests.
`test:browser` exercises real Worker/OPFS setup, reload/restore, task binding,
single files, directory batches, forced ZIP64 fixtures and startup failures.
Its ZIP echo fixture does not perform model inference. `node tests/zip64-browser.mjs`
checks actual >4 GiB data/offsets and >65,535 entries against Python zipfile.
`LATTI_DELIVERY_DIRECTORY=/extracted-delivery node tests/delivery-browser.mjs`
checks standalone task imports, packaged CIFAR preprocessing against the
delivered reference CSVs, and the shipped example's Worker startup/close.
`test:pregeneration` records full default-profile Worker setup, reload/restore,
linear memory, Go HeapAlloc, sampled JS heap via CDP (including workers),
renderer RSS and OPFS storage. Samples are observed maxima, not allocator high
water marks. These large tests use a
persistent browser context; site quota still depends on the actual browser.

`test:workflow` consumes fresh generator outputs and Worker-generated keys:

```sh
LATTI_WORKFLOW_TASK=/tmp/current-cifar/task \
LATTI_WORKFLOW_KEYS=/tmp/worker-pregeneration \
LATTI_CIFAR10_DATA=/path/to/cifar10 \
PYTHON=/path/to/python PYTHONPATH=/path/to/rebuilt/python-bindings \
npm run test:workflow
```

For Retina, set `LATTI_WORKFLOW_MODEL=retina` and `LATTI_WORKFLOW_DATA` to the
offline prepared directory containing `cases.json` and the three input folders.
The runner uses actual native GPU inference and unmodified server ZIP bytes,
then independently checks decrypted logits and accuracy. Set
`LATTI_RUNTIME_DIRECTORY=/extracted-package/dist` to exercise a release archive.
Record these results and the complete regressions against the six specifications
in `docs/spec/` before release.

The workflow report separates observed JS/RSS maxima by client phase and records
Go/WASM progress and OPFS usage for directory and ZIP encryption. For an
additional client measurement run, `LATTI_WORKFLOW_NATIVE_ARTIFACTS` may point
to this acceptance run's completed native artifacts. This reuses its unchanged
`output.zip`, `native.json` and `reload.json`; the report names their origin and
does not count this as a new native inference run.
