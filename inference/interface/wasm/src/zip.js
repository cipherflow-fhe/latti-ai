// Minimal ZIP reader/writer for the latti-client server contract.
// Input ciphertexts are stored without compression so their bytes are kept
// unchanged. Server output archives may use raw DEFLATE.

const encoder = new TextEncoder();
const decoder = new TextDecoder();

const crcTable = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < table.length; i++) {
    let value = i;
    for (let bit = 0; bit < 8; bit++) value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
    table[i] = value >>> 0;
  }
  return table;
})();

const crc32 = data => {
  let value = 0xffffffff;
  for (const byte of data) value = crcTable[(value ^ byte) & 0xff] ^ (value >>> 8);
  return (value ^ 0xffffffff) >>> 0;
};

const concat = parts => {
  const total = parts.reduce((size, part) => size + part.length, 0);
  const result = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) { result.set(part, offset); offset += part.length; }
  return result;
};

const u16 = (view, offset, value) => view.setUint16(offset, value, true);
const u32 = (view, offset, value) => view.setUint32(offset, value >>> 0, true);

export const checkedZipName = name => {
  if (typeof name !== 'string' || !name || name.length > 65535 || /[\\/\0]/.test(name) ||
      name.includes('..')) throw new Error('Invalid ZIP entry name');
  return name;
};

export function makeZip(entries) {
  const local = [], central = [];
  let offset = 0;
  for (const entry of entries) {
    const name = checkedZipName(entry.name);
    const data = entry.data instanceof Uint8Array ? entry.data : new Uint8Array(entry.data);
    if (data.length > 0xffffffff) throw new Error('ZIP entry is too large');
    const nameBytes = encoder.encode(name), checksum = crc32(data);
    const localHeader = new Uint8Array(30 + nameBytes.length), localView = new DataView(localHeader.buffer);
    u32(localView, 0, 0x04034b50); u16(localView, 4, 20); u16(localView, 6, 0); u16(localView, 8, 0);
    u16(localView, 10, 0); u16(localView, 12, 0); u32(localView, 14, checksum); u32(localView, 18, data.length);
    u32(localView, 22, data.length); u16(localView, 26, nameBytes.length); u16(localView, 28, 0); localHeader.set(nameBytes, 30);
    local.push(localHeader, data);

    const centralHeader = new Uint8Array(46 + nameBytes.length), centralView = new DataView(centralHeader.buffer);
    u32(centralView, 0, 0x02014b50); u16(centralView, 4, 20); u16(centralView, 6, 20); u16(centralView, 8, 0);
    u16(centralView, 10, 0); u16(centralView, 12, 0); u16(centralView, 14, 0); u32(centralView, 16, checksum);
    u32(centralView, 20, data.length); u32(centralView, 24, data.length); u16(centralView, 28, nameBytes.length);
    u16(centralView, 30, 0); u16(centralView, 32, 0); u16(centralView, 34, 0); u16(centralView, 36, 0);
    u32(centralView, 38, 0); u32(centralView, 42, offset); centralHeader.set(nameBytes, 46); central.push(centralHeader);
    offset += localHeader.length + data.length;
  }
  const localBytes = concat(local), centralBytes = concat(central);
  if (localBytes.length > 0xffffffff || centralBytes.length > 0xffffffff || entries.length > 0xffff)
    throw new Error('ZIP archive is too large');
  const end = new Uint8Array(22), view = new DataView(end.buffer);
  u32(view, 0, 0x06054b50); u16(view, 4, 0); u16(view, 6, 0); u16(view, 8, entries.length); u16(view, 10, entries.length);
  u32(view, 12, centralBytes.length); u32(view, 16, localBytes.length); u16(view, 20, 0);
  return concat([localBytes, centralBytes, end]);
}

const findEnd = bytes => {
  const start = Math.max(0, bytes.length - 65557);
  for (let offset = bytes.length - 22; offset >= start; offset--)
    if (bytes[offset] === 0x50 && bytes[offset + 1] === 0x4b && bytes[offset + 2] === 0x05 && bytes[offset + 3] === 0x06) return offset;
  throw new Error('ZIP end record not found');
};

async function inflate(data) {
  if (typeof DecompressionStream !== 'function') throw new Error('Browser raw-deflate support is required');
  const stream = new Blob([data]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export async function readZip(raw) {
  const bytes = raw instanceof Uint8Array ? raw : new Uint8Array(raw);
  const end = findEnd(bytes), endView = new DataView(bytes.buffer, bytes.byteOffset + end, bytes.length - end);
  const count = endView.getUint16(10, true), centralSize = endView.getUint32(12, true), centralOffset = endView.getUint32(16, true);
  if (end + 22 > bytes.length || centralOffset + centralSize > bytes.length || count === 0) throw new Error('Invalid ZIP archive');
  const entries = new Map(), view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = centralOffset;
  for (let i = 0; i < count; i++) {
    if (view.getUint32(offset, true) !== 0x02014b50) throw new Error('Invalid ZIP central directory');
    const flags = view.getUint16(offset + 8, true), method = view.getUint16(offset + 10, true);
    const compressedSize = view.getUint32(offset + 20, true), size = view.getUint32(offset + 24, true);
    const nameLength = view.getUint16(offset + 28, true), extraLength = view.getUint16(offset + 30, true), commentLength = view.getUint16(offset + 32, true);
    const name = decoder.decode(bytes.subarray(offset + 46, offset + 46 + nameLength));
    checkedZipName(name);
    if (flags & 1 || entries.has(name)) throw new Error('Invalid or duplicate ZIP entry: ' + name);
    const localOffset = view.getUint32(offset + 42, true);
    if (localOffset + 30 > bytes.length || view.getUint32(localOffset, true) !== 0x04034b50) throw new Error('Invalid ZIP local header: ' + name);
    const localNameLength = view.getUint16(localOffset + 26, true), localExtraLength = view.getUint16(localOffset + 28, true);
    const dataStart = localOffset + 30 + localNameLength + localExtraLength;
    if (dataStart + compressedSize > bytes.length) throw new Error('Truncated ZIP entry: ' + name);
    const compressed = bytes.subarray(dataStart, dataStart + compressedSize);
    let data;
    if (method === 0) data = new Uint8Array(compressed);
    else if (method === 8) data = await inflate(compressed);
    else throw new Error('Unsupported ZIP compression for ' + name);
    if (data.length !== size || crc32(data) !== view.getUint32(offset + 16, true)) throw new Error('ZIP entry checksum mismatch: ' + name);
    entries.set(name, data);
    offset += 46 + nameLength + extraLength + commentLength;
  }
  if (offset !== centralOffset + centralSize) throw new Error('Invalid ZIP central directory size');
  return entries;
}
