import {readdir} from 'node:fs/promises';
import path from 'node:path';

export const repositoryRoot=path.resolve(import.meta.dirname,'../../..');

// Keep native and browser algorithms identical, including target cost/profile data.
export async function modelSourceFiles() {
  const files=[
    'training/run_compile.py',
    'training/config/storage_aware_config.json',
    'training/config/BERT_pcm_config.json',
    'training/config/ViT_pcm_config.json',
    'tools/prepare_split_onnx.py',
    'tools/onnx_external_weights.py',
    'tools/onnx_split_contract.py',
    'tools/evaluate_onnx_csv.py',
    'tools/evaluate_transformer_onnx_csv.py',
    'tools/expand_transformer_onnx.py',
    'inference/interface/gen_mega_ag.py',
  ];
  async function collect(relative) {
    for(const entry of await readdir(path.join(repositoryRoot,relative),{withFileTypes:true})) {
      if(entry.name==='__pycache__'||entry.name==='unittests')continue;
      const name=path.posix.join(relative,entry.name);
      if(entry.isDirectory())await collect(name);
      else if(/\.(py|json)$/.test(entry.name))files.push(name);
    }
  }
  for(const directory of [
    'training/model_compiler','training/model_export','training/nn_tools',
    'inference/model_generator','inference/lattisense/frontend','inference/interface/wasm/python',
  ])await collect(directory);
  return files.sort();
}
