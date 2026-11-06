import {test} from 'node:test';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {openZip,writeZip} from '../src/zip.js';
import {makeZip} from './zip-fixture.js';

class MemoryFile {
  constructor(bytes = new Uint8Array()) { this.bytes = bytes; this.kind = 'file'; this.aborted = false; }
  async getFile() { return new Blob([this.bytes]); }
  async createWritable() {
    let bytes = new Uint8Array(), position = 0;
    return {
      write: async data => { const next = new Uint8Array(Math.max(bytes.length, position + data.length)); next.set(bytes); next.set(data, position); bytes = next; position += data.length; },
      seek: async offset => { position = offset; },
      close: async () => { this.bytes = bytes; },
      abort: async () => { this.aborted = true; },
    };
  }
}

test('forced ZIP64 output is readable by Python and the streaming reader', async () => {
  const archive = new MemoryFile(), spool = new MemoryFile();
  const content = new TextEncoder().encode('streamed ZIP64');
  const result = await writeZip(archive, [{name: '001.ct', file: new Blob([content])}], {spoolHandle: spool, forceZip64: true});
  assert.equal(result.zip64, true);
  const python = spawnSync('python3', ['-c', 'import sys,io,zipfile; z=zipfile.ZipFile(io.BytesIO(sys.stdin.buffer.read())); sys.stdout.buffer.write(z.read("001.ct"))'], {input: archive.bytes});
  assert.equal(python.status, 0, python.stderr.toString());
  assert.deepEqual(python.stdout, Buffer.from(content));
  const reader = await openZip(archive), entries = [];
  for await (const entry of reader.entries()) entries.push(entry);
  const output = new MemoryFile();
  await reader.readEntry(entries[0], output);
  assert.deepEqual(output.bytes, content);
});

test('CRC failure aborts the entry output', async () => {
  const bytes = makeZip([{name: '001.ct', data: new Uint8Array([1, 2, 3])}]);
  bytes[36] ^= 1;
  const reader = await openZip(new MemoryFile(bytes));
  const entry = (await Array.fromAsync(reader.entries()))[0], output = new MemoryFile();
  await assert.rejects(reader.readEntry(entry, output), /ZIP_ENTRY_CHECKSUM/);
  assert.equal(output.aborted, true);
  assert.equal(output.bytes.length, 0);
});

test('seekable Python ZIP64 accepts consistent populated 32-bit local sizes',async()=>{
  const python=spawnSync('python3',['-c',`import io,sys,zipfile
out=io.BytesIO()
with zipfile.ZipFile(out,'w') as z:
    with z.open('case.result.bin','w',force_zip64=True) as f: f.write(b'actual server bytes')
sys.stdout.buffer.write(out.getvalue())`]);
  assert.equal(python.status,0,python.stderr.toString());
  const bytes=new Uint8Array(python.stdout),view=new DataView(bytes.buffer);
  // Python 3.10 fills these fields after seeking back; later versions retain
  // sentinels. Both encodings keep the same two sizes in the local ZIP64 extra.
  view.setUint32(18,19,true);view.setUint32(22,19,true);
  const reader=await openZip(new MemoryFile(bytes)),output=new MemoryFile();
  await reader.readEntry((await Array.fromAsync(reader.entries()))[0],output);
  assert.equal(new TextDecoder().decode(output.bytes),'actual server bytes');
  const extraStart=30+view.getUint16(26,true);
  view.setBigUint64(extraStart+4,20n,true);
  await assert.rejects(openZip(new MemoryFile(bytes)),/ZIP_LOCAL_MISMATCH/);
});

test('local name mismatch is rejected before opening output', async () => {
  const bytes = makeZip([{name: '001.ct', data: new Uint8Array([1])}]);
  bytes[30] = 'x'.charCodeAt(0);
  await assert.rejects(openZip(new MemoryFile(bytes)), /ZIP_LOCAL_MISMATCH/);
});

test('writer aborts an open output when opening the spool fails', async () => {
  const output = new MemoryFile();
  await assert.rejects(writeZip(output, [], {spoolHandle: {createWritable: async () => { throw new Error('spool failure'); }}}), /spool failure/);
  assert.equal(output.aborted, true);
});

test('writer rejects an unaddressable entry before reading its source', async () => {
  const output = new MemoryFile(), spool = new MemoryFile();
  const source = {size: BigInt(Number.MAX_SAFE_INTEGER), async readAt() { throw new Error('unexpected read'); }};
  await assert.rejects(writeZip(output, [{name: 'a', source}], {spoolHandle: spool}), /IO_RANGE_UNSUPPORTED/);
  assert.equal(output.aborted, true);
  assert.equal(spool.aborted, true);
});

test('writer uses one source size throughout the entry copy', async () => {
  const output = new MemoryFile(), spool = new MemoryFile();
  let reads = 0;
  const source = {get size() { return ++reads === 1 ? 1n : 2n; }, async readAt() { return new Uint8Array([42]); }};
  await writeZip(output, [{name: 'a', source}], {spoolHandle: spool});
  assert.equal(reads, 1);
  const reader = await openZip(output), entry = (await Array.fromAsync(reader.entries()))[0], decoded = new MemoryFile();
  await reader.readEntry(entry, decoded);
  assert.deepEqual(decoded.bytes, new Uint8Array([42]));
});

test('ZIP64 central sizes are decoded independently', async () => {
  const base = makeZip([{name: 'a', data: new Uint8Array([42])}]);
  const view = new DataView(base.buffer), eocd = base.length - 22, central = view.getUint32(eocd + 16, true);
  for (const sentinelField of [20, 24]) {
    const extra = new Uint8Array(12), x = new DataView(extra.buffer);
    x.setUint16(0, 1, true); x.setUint16(2, 8, true); x.setBigUint64(4, 1n, true);
    const insert = central + 47, bytes = new Uint8Array(base.length + extra.length);
    bytes.set(base.subarray(0, insert)); bytes.set(extra, insert); bytes.set(base.subarray(insert), insert + extra.length);
    const v = new DataView(bytes.buffer);
    v.setUint32(central + sentinelField, 0xffffffff, true);
    v.setUint16(central + 30, extra.length, true);
    v.setUint32(eocd + extra.length + 12, view.getUint32(eocd + 12, true) + extra.length, true);
    const reader = await openZip(new MemoryFile(bytes));
    const entry = (await Array.fromAsync(reader.entries()))[0];
    assert.equal(entry.size, 1n); assert.equal(entry.compressedSize, 1n);
  }
});

function pythonDescriptorZip(zip64) {
  const code = `import io, sys, zipfile
class NonSeekable(io.BytesIO):
    def seekable(self): return False
    def seek(self, *args): raise io.UnsupportedOperation()
out = NonSeekable()
with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    with z.open('data.ct', 'w', force_zip64=${zip64 ? 'True' : 'False'}) as entry:
        entry.write(b'hello streaming ' * 10000)
sys.stdout.buffer.write(out.getvalue())`;
  const p = spawnSync('python3', ['-c', code]);
  assert.equal(p.status, 0, p.stderr.toString());
  return new Uint8Array(p.stdout);
}

test('Python DEFLATE with signed and unsigned ZIP32/ZIP64 descriptors streams correctly', async () => {
  for (const zip64 of [false, true]) for (const signed of [false, true]) {
    let bytes = pythonDescriptorZip(zip64);
    if (!signed) {
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), end = bytes.length - 22;
      const central = view.getUint32(end + 16, true), descriptor = central - (zip64 ? 24 : 16);
      assert.equal(view.getUint32(descriptor, true), 0x08074b50);
      const next = new Uint8Array(bytes.length - 4);
      next.set(bytes.subarray(0, descriptor)); next.set(bytes.subarray(descriptor + 4), descriptor);
      new DataView(next.buffer).setUint32(end - 4 + 16, central - 4, true);
      bytes = next;
    }
    const reader = await openZip(new MemoryFile(bytes));
    const entry = (await Array.fromAsync(reader.entries()))[0], output = new MemoryFile();
    try { await reader.readEntry(entry, output); }
    catch (error) { throw new Error(`zip64=${zip64} signed=${signed}: ${error.message}`, {cause: error}); }
    assert.equal(new TextDecoder().decode(output.bytes), 'hello streaming '.repeat(10000));
  }
});

test('DEFLATE enforces actual output budget and aborts', async () => {
  const reader = await openZip(new MemoryFile(pythonDescriptorZip(false)), {maxEntryOutputBytes: 100n});
  const entry = (await Array.fromAsync(reader.entries()))[0], output = new MemoryFile();
  await assert.rejects(reader.readEntry(entry, output), /ZIP_OUTPUT_BUDGET_EXCEEDED/);
  assert.equal(output.aborted, true);
});

test('invalid descriptor is rejected', async () => {
  const bytes = pythonDescriptorZip(false), view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const central = view.getUint32(bytes.length - 22 + 16, true);
  bytes[central - 12] ^= 1;
  await assert.rejects(openZip(new MemoryFile(bytes)), /ZIP_BAD_DESCRIPTOR/);
});

test('raw DEFLATE trailing bytes fail before output is committed', async () => {
  const original = pythonDescriptorZip(false), v = new DataView(original.buffer, original.byteOffset, original.byteLength);
  const end = original.length - 22, central = v.getUint32(end + 16, true), descriptor = central - 16;
  const compressed = v.getUint32(central + 20, true);
  const bytes = new Uint8Array(original.length + 1);
  bytes.set(original.subarray(0, descriptor)); bytes[descriptor] = 0xff; bytes.set(original.subarray(descriptor), descriptor + 1);
  const next = new DataView(bytes.buffer);
  next.setUint32(descriptor + 1 + 8, compressed + 1, true);
  next.setUint32(central + 1 + 20, compressed + 1, true);
  next.setUint32(end + 1 + 16, central + 1, true);
  const reader = await openZip(new MemoryFile(bytes)), entry = (await Array.fromAsync(reader.entries()))[0], output = new MemoryFile();
  await assert.rejects(reader.readEntry(entry, output));
  assert.equal(output.aborted, true);
});

test('rejects overlapping local entries before exposing any entry', async () => {
  const bytes = makeZip([{name: 'a', data: new Uint8Array(40)}, {name: 'b', data: new Uint8Array([1])}]);
  const view = new DataView(bytes.buffer), end = bytes.length - 22, central = view.getUint32(end + 16, true);
  const second = central + 46 + view.getUint16(central + 28, true);
  view.setUint32(second + 42, 35, true);
  await assert.rejects(openZip(new MemoryFile(bytes)), /ZIP_ENTRY_OVERLAP/);
});

test('rejects a descriptor overlapping the next local header', async () => {
  const code = `import io, sys, zipfile
class NonSeekable(io.BytesIO):
    def seekable(self): return False
    def seek(self, *args): raise io.UnsupportedOperation()
out = NonSeekable()
with zipfile.ZipFile(out, 'w') as z:
    z.writestr('a', b'first')
    z.writestr('b', b'second')
sys.stdout.buffer.write(out.getvalue())`;
  const p = spawnSync('python3', ['-c', code]);
  assert.equal(p.status, 0, p.stderr.toString());
  const bytes = new Uint8Array(p.stdout), view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const central = view.getUint32(bytes.length - 22 + 16, true);
  const second = central + 46 + view.getUint16(central + 28, true) + view.getUint16(central + 30, true) + view.getUint16(central + 32, true);
  view.setUint32(second + 42, view.getUint32(second + 42, true) - 4, true);
  await assert.rejects(openZip(new MemoryFile(bytes)), /ZIP_BAD_DESCRIPTOR/);
});

test('validates ZIP64 end record bounds and permits an extension', async () => {
  const archive = new MemoryFile();
  await writeZip(archive, [{name: 'a', file: new Blob(['x'])}], {spoolHandle: new MemoryFile(), forceZip64: true});
  const bytes = archive.bytes, view = new DataView(bytes.buffer), locator = bytes.length - 42;
  const end64 = Number(view.getBigUint64(locator + 8, true));
  const extended = new Uint8Array(bytes.length + 4);
  extended.set(bytes.subarray(0, locator)); extended.set([1, 2, 3, 4], locator); extended.set(bytes.subarray(locator), locator + 4);
  new DataView(extended.buffer).setBigUint64(end64 + 4, 48n, true);
  assert.equal((await openZip(new MemoryFile(extended))).count, 1n);
  const malformed = extended.slice();
  new DataView(malformed.buffer).setBigUint64(end64 + 4, 49n, true);
  await assert.rejects(openZip(new MemoryFile(malformed)), /ZIP64_EOCD_RANGE/);
  const multidisk = bytes.slice();
  new DataView(multidisk.buffer).setUint32(locator + 16, 2, true);
  await assert.rejects(openZip(new MemoryFile(multidisk)), /ZIP_MULTIDISK_UNSUPPORTED/);
});

test('rejects central directory extending into the end record', async () => {
  const bytes = makeZip([{name: 'a', data: new Uint8Array([1])}]), view = new DataView(bytes.buffer);
  const end = bytes.length - 22;
  view.setUint32(end + 12, view.getUint32(end + 12, true) + 1, true);
  await assert.rejects(openZip(new MemoryFile(bytes)), /ZIP_CENTRAL_OUT_OF_RANGE/);
});
