import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeZip, readZip } from '../src/zip.js';

test('server ZIP writer preserves stored entry bytes', async () => {
  const archive = makeZip([
    {name: '001.ct', data: new Uint8Array([0, 1, 2, 255])},
    {name: '002__input_1.ct', data: new TextEncoder().encode('clinical')},
  ]);
  const entries = await readZip(archive);
  assert.deepEqual([...entries.keys()], ['001.ct', '002__input_1.ct']);
  assert.deepEqual([...entries.get('001.ct')], [0, 1, 2, 255]);
  assert.equal(new TextDecoder().decode(entries.get('002__input_1.ct')), 'clinical');
});

test('server ZIP names reject traversal and nested paths', () => {
  assert.throws(() => makeZip([{name: '../escape.ct', data: new Uint8Array()}]));
  assert.throws(() => makeZip([{name: 'nested/escape.ct', data: new Uint8Array()}]));
});
