export interface Progress {
  phase: 'generation' | 'full-export' | 'restore' | 'encrypt' | 'decrypt';
  bytes: number;
  elapsedMs: number;
  wasmBytes: number;
  heapBytes: number;
  payloadBytes: number;
  chunkBytes: number;
  inFlightChunks: number;
}
export interface BrowserClient {
  setup(): Promise<{evalContext: FileSystemFileHandle; fullContext: FileSystemFileHandle}>;
  restore(): Promise<void>;
  encrypt(options: {inputName:string; inputFile:FileSystemFileHandle | File; outputDirectory:FileSystemDirectoryHandle}): Promise<FileSystemFileHandle>;
  decrypt(options: {outputName:string; ciphertextFile:FileSystemFileHandle | File; outputDirectory:FileSystemDirectoryHandle}): Promise<FileSystemFileHandle>;
  close(): Promise<void>;
}
export function openBrowserClient(options: {clientDirectory:FileSystemDirectoryHandle; onProgress?:(progress:Progress)=>void}): Promise<BrowserClient>;
