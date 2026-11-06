// Bounded OPFS ZIP32/ZIP64 reader and STORED writer.

export const MAX_CHUNK_BYTES = 4n * 1024n * 1024n;
const U32 = 0xffffffffn, U16 = 0xffffn;
const enc = new TextEncoder(), dec = new TextDecoder('utf-8', {fatal: true});
const u16 = n => { const b = new Uint8Array(2); new DataView(b.buffer).setUint16(0, Number(n), true); return b; };
const u32 = n => { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, Number(n) >>> 0, true); return b; };
const u64 = n => { const b = new Uint8Array(8); new DataView(b.buffer).setBigUint64(0, BigInt(n), true); return b; };
const r16 = (b, o) => new DataView(b.buffer, b.byteOffset, b.byteLength).getUint16(o, true);
const r32 = (b, o) => new DataView(b.buffer, b.byteOffset, b.byteLength).getUint32(o, true);
const r64 = (b, o) => new DataView(b.buffer, b.byteOffset, b.byteLength).getBigUint64(o, true);
const join = parts => { const n = parts.reduce((a, b) => a + b.byteLength, 0), out = new Uint8Array(n); let p = 0; for (const b of parts) { out.set(b, p); p += b.byteLength; } return out; };

export function checkedZipName(name) {
  if (typeof name !== 'string' || !name || /[\\/\0]/.test(name) || name === '.' || name === '..' || name.includes('..')) throw new Error('ZIP_INVALID_NAME');
  const bytes = enc.encode(name); if (bytes.byteLength > 0xffff) throw new Error('ZIP_NAME_TOO_LONG'); return {name, bytes};
}

const crcTable = (() => { const t = new Uint32Array(256); for (let i = 0; i < 256; ++i) { let x = i; for (let j = 0; j < 8; ++j) x = x & 1 ? 0xedb88320 ^ (x >>> 1) : x >>> 1; t[i] = x >>> 0; } return t; })();
export function crc32Update(crc, data) { let x = (crc ^ 0xffffffff) >>> 0; for (const b of data) x = crcTable[(x ^ b) & 255] ^ (x >>> 8); return (x ^ 0xffffffff) >>> 0; }
function checkedRange(offset, length) { offset = BigInt(offset); length = BigInt(length); if (offset < 0n || length < 0n || offset > BigInt(Number.MAX_SAFE_INTEGER) || offset + length > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('IO_RANGE_UNSUPPORTED'); return [Number(offset), Number(length)]; }

export function fileSource(fileOrHandle) {
  const filePromise=fileOrHandle.kind==='file'?fileOrHandle.getFile():Promise.resolve(fileOrHandle);
  return {size:filePromise.then(file=>BigInt(file.size)),async readAt(offset,maxBytes) {
    const file=await filePromise,[at,n]=checkedRange(offset,maxBytes);
    if(n>Number(MAX_CHUNK_BYTES)) throw new Error('IO_CHUNK_TOO_LARGE');
    return new Uint8Array(await file.slice(at,at+n).arrayBuffer());
  }};
}

async function openWritable(handle) {
  const writer=await handle.createWritable(),buffer=new Uint8Array(65536);
  let position=0n,closed=false,used=0;
  const flush=async()=>{if(used){await writer.write(buffer.subarray(0,used));used=0;}};
  return {
    get position(){return position;},
    async append(data) {
      if(closed) throw new Error('ZIP_WRITER_CLOSED');checkedRange(position,data.byteLength);
      if(data.byteLength>=buffer.length) {await flush();await writer.write(data);}
      else {if(used+data.byteLength>buffer.length) await flush();buffer.set(data,used);used+=data.byteLength;}
      position+=BigInt(data.byteLength);
    },
    async patch(offset,data) {
      if(closed||offset<0n||offset+BigInt(data.byteLength)>position||offset>BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('ZIP_INVALID_PATCH');
      await flush();await writer.seek(Number(offset));await writer.write(data);await writer.seek(Number(position));
    },
    async close(){if(!closed){await flush();await writer.close();closed=true;}},
    async abort(){if(!closed){closed=true;await writer.abort().catch(()=>{});}},
  };
}

async function copySource(source, sink, total) {
  let offset = 0n, size = 0n, crc = 0;
  while (offset < total) { const want = total - offset > MAX_CHUNK_BYTES ? MAX_CHUNK_BYTES : total - offset; const b = await source.readAt(offset, want); if (!b.byteLength || BigInt(b.byteLength) > want) throw new Error('ZIP_SHORT_READ'); await sink.append(b); crc = crc32Update(crc, b); offset += BigInt(b.byteLength); size += BigInt(b.byteLength); }
  return {size, crc};
}
function localHeader(name, zip64, size, crc) { const extra = zip64 ? join([u16(1), u16(16), u64(size), u64(size)]) : new Uint8Array(); return join([u32(0x04034b50), u16(zip64 ? 45 : 20), u16(0x800), u16(0), u16(0), u16(0), u32(crc), u32(zip64 ? U32 : size), u32(zip64 ? U32 : size), u16(name.byteLength), u16(extra.byteLength), name, extra]); }
function centralHeader(name, zip64, size, crc, offset) { const fields = []; if (zip64 || size >= U32) fields.push(u64(size), u64(size)); if (zip64 || offset >= U32) fields.push(u64(offset)); const extra = fields.length ? join([u16(1), u16(fields.reduce((n, b) => n + b.byteLength, 0)), ...fields]) : new Uint8Array(); return join([u32(0x02014b50), u16(45), u16(zip64 ? 45 : 20), u16(0x800), u16(0), u16(0), u16(0), u32(crc), u32(zip64 || size >= U32 ? U32 : size), u32(zip64 || size >= U32 ? U32 : size), u16(name.byteLength), u16(extra.byteLength), u16(0), u16(0), u16(0), u32(0), u32(zip64 || offset >= U32 ? U32 : offset), name, extra]); }

/** Write one entry at a time to OPFS; central records are spooled separately. */
export async function writeZip(outputHandle, entries, {spoolHandle, forceZip64 = false, check = () => {}, maxEntries = 100000, maxMetadataBytes = 64 * 1024 * 1024} = {}) {
  if (!spoolHandle) throw new Error('ZIP_SPOOL_REQUIRED');
  const out = await openWritable(outputHandle); let spool, count = 0n, centralSize = 0n, centralOffset = 0n, zip64 = forceZip64;
  const names=new Set();let metadata=0;
  try {
    spool = await openWritable(spoolHandle);
    for await (const entry of entries) { check(); const {bytes: name} = checkedZipName(entry.name); const key=entry.name.normalize('NFC').toLowerCase();metadata+=(entry.name.length+key.length)*2+256;if(names.has(key)) throw new Error('ZIP_DUPLICATE_NAME');if(names.size>=maxEntries||metadata>maxMetadataBytes) throw new Error('METADATA_BUDGET_EXCEEDED');names.add(key);const source = entry.source || fileSource(entry.file), expected = BigInt(await source.size), use64 = forceZip64 || expected >= U32 || out.position >= U32, offset = out.position; const header = localHeader(name, use64, expected, 0); checkedRange(offset, BigInt(header.byteLength) + expected); zip64 ||= use64; await out.append(header); const checkedSource={readAt:(...args)=>{check();return source.readAt(...args);}}; const got = await copySource(checkedSource, out, expected); if (got.size !== expected) throw new Error('ZIP_SIZE_MISMATCH'); if(got.crc!==0) await out.patch(offset + 14n, u32(got.crc)); const central = centralHeader(name, use64, got.size, got.crc, offset); await spool.append(central); centralSize += BigInt(central.byteLength); ++count; }
    centralOffset = out.position; await spool.close(); const ss = fileSource(spoolHandle), total = BigInt(await ss.size); for (let p = 0n; p < total;) { const n = total - p > MAX_CHUNK_BYTES ? MAX_CHUNK_BYTES : total - p, b = await ss.readAt(p, n); if (!b.byteLength) throw new Error('ZIP_SPOOL_READ'); await out.append(b); p += BigInt(b.byteLength); }
    const need64 = zip64 || count >= U16 || centralSize >= U32 || centralOffset >= U32; if (need64) { const eo = out.position; await out.append(join([u32(0x06064b50), u64(44), u16(45), u16(45), u32(0), u32(0), u64(count), u64(count), u64(centralSize), u64(centralOffset)])); await out.append(join([u32(0x07064b50), u32(0), u64(eo), u32(1)])); }
    await out.append(join([u32(0x06054b50), u16(0), u16(0), u16(need64 ? U16 : count), u16(need64 ? U16 : count), u32(need64 ? U32 : centralSize), u32(need64 ? U32 : centralOffset), u16(0)])); await out.close(); return {bytes: out.position, count, centralOffset, centralSize, zip64: need64};
  } catch (e) { await out.abort(); if (spool) await spool.abort(); throw e; }
}

function parseZip64(extra, required) {
  let p = 0, found = false; const values = {};
  while (p < extra.byteLength) {
    if (p + 4 > extra.byteLength) throw new Error('ZIP_BAD_EXTRA');
    const id = r16(extra, p), n = r16(extra, p + 2); p += 4;
    if (p + n > extra.byteLength) throw new Error('ZIP_BAD_EXTRA');
    if (id === 1) {
      if (found) throw new Error('ZIP_DUPLICATE_ZIP64_EXTRA');
      found = true; let q = p;
      for (const [field, width] of [['size', 8], ['compressedSize', 8], ['offset', 8], ['disk', 4]]) {
        if (!required[field]) continue;
        if (q + width > p + n) throw new Error('ZIP_MISSING_ZIP64_FIELD');
        values[field] = width === 8 ? r64(extra, q) : BigInt(r32(extra, q)); q += width;
      }
      if (q !== p + n) throw new Error('ZIP_BAD_ZIP64_EXTRA');
    }
    p += n;
  }
  if (Object.values(required).some(Boolean) && !found) throw new Error('ZIP_MISSING_ZIP64_FIELD');
  return values;
}
function hasZip64Extra(extra) {
  let p = 0, found = false;
  while (p < extra.length) {
    if (p + 4 > extra.length) throw new Error('ZIP_BAD_EXTRA');
    const id = r16(extra, p), size = r16(extra, p + 2); p += 4;
    if (p + size > extra.length) throw new Error('ZIP_BAD_EXTRA');
    if (id === 1) {
      if (found || size !== 16) throw new Error('ZIP_BAD_LOCAL_ZIP64');
      found = true;
    }
    p += size;
  }
  return found;
}
async function findEnd(source) { const size = BigInt(await source.size), window = size > 0x10016n ? 0x10016n : size, b = await source.readAt(size - window, window); for (let i = b.byteLength - 22; i >= 0; --i) if (r32(b, i) === 0x06054b50 && BigInt(i + 22 + r16(b, i + 20)) === window) return {at: size - window + BigInt(i), bytes: b, index: i}; throw new Error('ZIP_EOCD_NOT_FOUND'); }

async function validateLocal(source, record, limit) {
  if (record.method !== 0 && record.method !== 8) throw new Error(`ZIP_METHOD_UNSUPPORTED: ${record.name}`);
  if (record.flags & 1) throw new Error(`ZIP_ENCRYPTION_UNSUPPORTED: ${record.name}`);
  if (record.offset + 30n > limit) throw new Error('ZIP_ENTRY_OVERLAP');
  const h = await source.readAt(record.offset, 30n);
  if (h.length !== 30 || r32(h, 0) !== 0x04034b50) throw new Error('ZIP_BAD_LOCAL');
  const nl = r16(h, 26), xl = r16(h, 28), dataStart = record.offset + 30n + BigInt(nl + xl), dataEnd = dataStart + record.compressedSize;
  if (dataEnd > limit) throw new Error('ZIP_ENTRY_OVERLAP');
  const body = await source.readAt(record.offset + 30n, BigInt(nl + xl));
  if (body.length !== nl + xl || r16(h, 6) !== record.flags || r16(h, 8) !== record.method) throw new Error('ZIP_LOCAL_MISMATCH');
  let localName; try { localName = dec.decode(body.subarray(0, nl)); } catch { throw new Error('ZIP_BAD_UTF8'); }
  if (localName !== record.name) throw new Error('ZIP_LOCAL_MISMATCH');
  const localSize = BigInt(r32(h, 22)), localCompressed = BigInt(r32(h, 18));
  const local64 = localSize === U32 || localCompressed === U32;
  if (local64 && (localSize !== U32 || localCompressed !== U32)) throw new Error('ZIP_BAD_LOCAL_ZIP64');
  if (!(record.flags & 8)) {
    // A seekable Python writer may fill the 32-bit sizes on close while
    // retaining its reserved ZIP64 local extra. Validate both representations.
    const extra64=hasZip64Extra(body.subarray(nl));
    const z = parseZip64(body.subarray(nl), {size: local64||extra64, compressedSize: local64||extra64});
    if(extra64&&!local64&&(z.size!==localSize||z.compressedSize!==localCompressed)) throw new Error('ZIP_LOCAL_MISMATCH');
    if (r32(h, 14) !== record.crc || (local64 ? z.size : localSize) !== record.size || (local64 ? z.compressedSize : localCompressed) !== record.compressedSize) throw new Error('ZIP_LOCAL_MISMATCH');
  } else {
    const zip64 = local64 || hasZip64Extra(body.subarray(nl)) || record.size >= U32 || record.compressedSize >= U32;
    const maxLength = zip64 ? 24 : 16, available = limit - dataEnd;
    const descriptor = await source.readAt(dataEnd, available < BigInt(maxLength) ? available : BigInt(maxLength));
    const matches = start => {
      const needed = start + (zip64 ? 20 : 12);
      if (descriptor.length < needed || r32(descriptor, start) !== record.crc) return false;
      const compressed = zip64 ? r64(descriptor, start + 4) : BigInt(r32(descriptor, start + 4));
      const size = zip64 ? r64(descriptor, start + 12) : BigInt(r32(descriptor, start + 8));
      return compressed === record.compressedSize && size === record.size;
    };
    if (!matches(0) && !(descriptor.length >= 4 && r32(descriptor, 0) === 0x08074b50 && matches(4))) throw new Error('ZIP_BAD_DESCRIPTOR');
  }
  return dataStart;
}

/** Return bounded central records; readEntry streams decoded data to OPFS. */
export async function openZip(sourceLike, {maxEntries = 100000, maxMetadataBytes = 64 * 1024 * 1024, maxEntryOutputBytes = 8n * 1024n ** 3n, maxTotalOutputBytes = 32n * 1024n ** 3n} = {}) {
  maxEntryOutputBytes = BigInt(maxEntryOutputBytes); maxTotalOutputBytes = BigInt(maxTotalOutputBytes);
  if (maxEntryOutputBytes < 0n || maxTotalOutputBytes < 0n) throw new Error('ZIP_INVALID_BUDGET');
  const source = sourceLike.readAt ? sourceLike : fileSource(sourceLike), end = await findEnd(source), b = end.bytes, i = end.index;
  const disk = r16(b, i + 4), centralDisk = r16(b, i + 6), diskCount = BigInt(r16(b, i + 8));
  let count = BigInt(r16(b, i + 10)), centralSize = BigInt(r32(b, i + 12)), centralOffset = BigInt(r32(b, i + 16)), centralLimit = end.at;
  if (disk || centralDisk || diskCount !== count) throw new Error('ZIP_MULTIDISK_UNSUPPORTED');
  const locator = end.at >= 20n ? await source.readAt(end.at - 20n, 20n) : new Uint8Array();
  const hasLocator = locator.length === 20 && r32(locator, 0) === 0x07064b50;
  if (count === U16 || centralSize === U32 || centralOffset === U32 || hasLocator) {
    if (!hasLocator) throw new Error('ZIP64_LOCATOR_REQUIRED');
    if (r32(locator, 4) !== 0 || r32(locator, 16) !== 1) throw new Error('ZIP_MULTIDISK_UNSUPPORTED');
    const offset = r64(locator, 8);
    if (offset + 56n > end.at - 20n) throw new Error('ZIP64_EOCD_RANGE');
    const h = await source.readAt(offset, 56n);
    if (h.length !== 56 || r32(h, 0) !== 0x06064b50) throw new Error('ZIP64_EOCD_REQUIRED');
    const recordSize = r64(h, 4), count64 = r64(h, 32), size64 = r64(h, 40), offset64 = r64(h, 48);
    if (recordSize < 44n || offset + 12n + recordSize !== end.at - 20n) throw new Error('ZIP64_EOCD_RANGE');
    if (r32(h, 16) || r32(h, 20) || r64(h, 24) !== count64) throw new Error('ZIP_MULTIDISK_UNSUPPORTED');
    if ((count !== U16 && count !== count64) || (centralSize !== U32 && centralSize !== size64) || (centralOffset !== U32 && centralOffset !== offset64)) throw new Error('ZIP64_EOCD_MISMATCH');
    count = count64; centralSize = size64; centralOffset = offset64; centralLimit = offset;
  }
  if (centralOffset + centralSize > centralLimit) throw new Error('ZIP_CENTRAL_OUT_OF_RANGE');
  if (count > BigInt(maxEntries)) throw new Error('METADATA_BUDGET_EXCEEDED'); const total = BigInt(await source.size), records = [], names = new Set(); let p = centralOffset, metadata = 0n;
  for (let n = 0n; n < count; ++n) { const h = await source.readAt(p, 46n); if (h.byteLength !== 46 || r32(h, 0) !== 0x02014b50) throw new Error('ZIP_BAD_CENTRAL'); const nl = r16(h, 28), xl = r16(h, 30), cl = r16(h, 32), body = await source.readAt(p + 46n, BigInt(nl + xl + cl)); if (body.byteLength !== nl + xl + cl) throw new Error('ZIP_BAD_CENTRAL'); let name; try { name = dec.decode(body.subarray(0, nl)); } catch { throw new Error('ZIP_BAD_UTF8'); } checkedZipName(name); const extra = body.subarray(nl, nl + xl), sz0 = BigInt(r32(h, 24)), compressed0 = BigInt(r32(h, 20)), off0 = BigInt(r32(h, 42)), disk0 = BigInt(r16(h, 34)), z = parseZip64(extra, {size: sz0 === U32, compressedSize: compressed0 === U32, offset: off0 === U32, disk: disk0 === U16}), size = sz0 === U32 ? z.size : sz0, compressedSize = compressed0 === U32 ? z.compressedSize : compressed0, offset = off0 === U32 ? z.offset : off0, disk = disk0 === U16 ? z.disk : disk0; if (disk !== 0n) throw new Error('ZIP_MULTIDISK_UNSUPPORTED'); if (offset + 30n > total) throw new Error('ZIP_BAD_ZIP64'); const key = name.normalize('NFC').toLowerCase(); metadata += BigInt((name.length + key.length) * 2 + 256); if (metadata > BigInt(maxMetadataBytes)) throw new Error('METADATA_BUDGET_EXCEEDED'); if (names.has(key)) throw new Error('ZIP_DUPLICATE_NAME'); names.add(key); records.push({name, key, flags: r16(h, 8), method: r16(h, 10), crc: r32(h, 16), size, compressedSize, offset, source}); p += 46n + BigInt(nl + xl + cl); }
  if (p !== centralOffset + centralSize) throw new Error('ZIP_CENTRAL_SIZE_MISMATCH');
  const ordered = [...records].sort((a, b) => a.offset < b.offset ? -1 : a.offset > b.offset ? 1 : 0);
  for (let index = 0; index < ordered.length; index++) {
    const record = ordered[index], limit = index + 1 < ordered.length ? ordered[index + 1].offset : centralOffset;
    record.dataStart = await validateLocal(source, record, limit);
    Object.freeze(record);
  }
  const recordSet = new Set(records); let totalOutput = 0n, busy = false;
  return {count, async *entries() { for (const record of records) yield record; }, async readEntry(record, sinkHandle) {
    if (!recordSet.has(record)) throw new Error('ZIP_UNKNOWN_ENTRY');
    if (busy) throw new Error('ZIP_READER_BUSY');
    busy = true;
    try {
      let pos = record.dataStart, remaining = record.compressedSize;
      const compressed = new ReadableStream({async pull(controller) {
        if (!remaining) { controller.close(); return; }
        const want = remaining > MAX_CHUNK_BYTES ? MAX_CHUNK_BYTES : remaining;
        const chunk = await source.readAt(pos, want);
        if (!chunk.byteLength || BigInt(chunk.byteLength) > want) throw new Error('ZIP_TRUNCATED_ENTRY');
        pos += BigInt(chunk.byteLength); remaining -= BigInt(chunk.byteLength); controller.enqueue(chunk);
      }});
      let stream = compressed;
      if (record.method === 8) {
        let decoder; try { decoder = new DecompressionStream('deflate-raw'); } catch { throw new Error('ZIP_DEFLATE_UNSUPPORTED'); }
        stream = compressed.pipeThrough(decoder);
      }
      const sink = await openWritable(sinkHandle), reader = stream.getReader();
      try {
        let crc = 0, size = 0n;
        for (;;) {
          const {value, done} = await reader.read(); if (done) break;
          size += BigInt(value.byteLength);
          if (size > maxEntryOutputBytes || totalOutput + size > maxTotalOutputBytes) throw new Error('ZIP_OUTPUT_BUDGET_EXCEEDED');
          if (size > record.size) throw new Error('ZIP_ENTRY_SIZE_MISMATCH');
          for (let offset = 0; offset < value.byteLength; offset += Number(MAX_CHUNK_BYTES)) {
            const chunk = value.subarray(offset, offset + Number(MAX_CHUNK_BYTES));
            await sink.append(chunk); crc = crc32Update(crc, chunk);
          }
        }
        if (remaining || size !== record.size || crc !== record.crc) throw new Error('ZIP_ENTRY_CHECKSUM');
        await sink.close(); totalOutput += size;
      } catch (error) { await reader.cancel(error).catch(() => {}); await sink.abort(); throw error; }
      finally { reader.releaseLock(); }
    } finally { busy = false; }
  }};
}

