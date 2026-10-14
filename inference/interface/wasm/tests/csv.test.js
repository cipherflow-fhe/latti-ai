import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readCSV } from '../src/csv.js';

test('CSV consumes streamed numeric fields across byte boundaries', async () => {
  const file = { stream: () => new ReadableStream({ start(c) {
    for (const text of ['1.2', '5, -2', '\n3e', '-2,4\r', '\n']) c.enqueue(new TextEncoder().encode(text));
    c.close();
  } }) };
  assert.deepEqual(await readCSV(file), [1.25, -2, 0.03, 4]);
});

test('CSV rejects invalid numbers and empty fields', async () => {
  for (const text of ['NaN', 'Infinity', '1,,2', '0x10', '1;2']) {
    await assert.rejects(readCSV(new Blob([text])));
  }
});

test('CSV accepts trailing delimiters used by task examples', async () => {
  assert.deepEqual(await readCSV(new Blob(['1,2,\n3,4,\n'])), [1, 2, 3, 4]);
  assert.deepEqual(await readCSV(new Blob(['1,2,'])), [1, 2]);
});
