// Read numeric CSV without holding an additional full text copy.
export async function readCSV(file) {
  const decoder = new TextDecoder();
  const values = [];
  let field = '', afterComma = false;
  const flush = () => {
    const text = field.trim();
    if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(text)) throw new Error('Invalid CSV numeric field');
    const value = Number(text);
    if (!Number.isFinite(value)) throw new Error('Non-finite CSV value');
    values.push(value); field = '';
  };
  const consume = text => {
    for (const character of text) {
      if (character === ',') { flush(); afterComma = true; }
      else if (character === '\r' || character === '\n') {
        if (field.trim()) { flush(); afterComma = false; }
        else if (afterComma) afterComma = false;
      } else {
        field += character;
        if (field.length > 128) throw new Error('CSV numeric field is too long');
        if (character.trim()) afterComma = false;
      }
    }
  };
  const reader = file.stream().getReader();
  try {
    for (;;) {
      const {value,done} = await reader.read();
      if (done) break;
      consume(decoder.decode(value,{stream:true}));
    }
    consume(decoder.decode());
    if (field.trim()) flush();
    else if (afterComma) afterComma = false;
    if (!values.length) throw new Error('Empty CSV input');
    return values;
  } finally { reader.releaseLock(); }
}
