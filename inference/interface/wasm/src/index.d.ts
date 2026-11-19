export interface Progress {
  requestId: number;
  jobId: string;
  phase: string;
  bytes: string;
  elapsedMs: number;
  wasmBytes: number;
  heapBytes: number;
  chunkBytes: number;
  inFlightChunks: number;
  entry?: string;
  final: boolean;
}
export interface ArtifactRef {
  artifactId: string;
  role: 'public-eval' | 'private-full' | 'ciphertext' | 'plaintext';
  file: FileSystemFileHandle;
  bytes: string;
  profileId: string;
  manifest?: FileSystemFileHandle;
  sha256?: string;
}
export interface BatchResult {
  directory: FileSystemDirectoryHandle;
  manifest: FileSystemFileHandle;
  count: string;
  archive?: ArtifactRef;
}
export interface KeyProfile {
  /** Exact task-format uint64 JSON, passed to Go without a JS parse/stringify. */
  parameters: string;
  bootstrap: boolean;
}
export type FileSource = FileSystemFileHandle | File;
export interface ArtifactDirectory {
  directory: FileSystemDirectoryHandle;
  manifest: FileSystemFileHandle;
}
export interface ModelSources {
  directory: FileSystemDirectoryHandle;
  modelPath: string;
  inputsManifest: FileSource;
}
export type DeploymentMode = 'client_encrypted_input' | 'server_provisioned_runner';
export interface BrowserModelTools {
  prepareSplit(options: {source: ModelSources; ratio: string; outputDirectory?: FileSystemDirectoryHandle}): Promise<ArtifactDirectory>;
  runPrefix(options: {split: ArtifactDirectory; inputsManifest: FileSource; outputDirectory?: FileSystemDirectoryHandle}): Promise<ArtifactDirectory>;
  compileHighLevel(options: ({source: ModelSources; split?: never} | {split: ArtifactDirectory; source?: never}) & {
    deploymentMode: DeploymentMode; compileConfig?: FileSource; outputDirectory?: FileSystemDirectoryHandle;
  }): Promise<ArtifactDirectory>;
  generateLowLevel(options: {task: ArtifactDirectory; outputDirectory?: FileSystemDirectoryHandle}): Promise<ArtifactDirectory>;
  cancel(requestId?: number): Promise<{accepted: boolean}>;
  close(): Promise<void>;
}
export function openBrowserModelTools(options?: {
  onProgress?: (progress: Progress) => void;
  timeoutMs?: number;
}): Promise<BrowserModelTools>;
export interface EncryptOptions {
  inputName: string;
  inputFile: FileSource;
  outputDirectory?: FileSystemDirectoryHandle;
  /** Bind new crossing CSVs to a runPrefix result for an existing split task. */
  run?: ArtifactDirectory;
}
export interface DecryptOptions {
  outputName: string;
  ciphertextFile: FileSource;
  outputDirectory?: FileSystemDirectoryHandle;
}
export interface BrowserClient {
  setup(): Promise<{evalContext: ArtifactRef; fullContext: ArtifactRef; manifest: FileSystemFileHandle; profileId: string; keyIdentity: string}>;
  restore(): Promise<{profileId: string; keyIdentity: string}>;
  bindTask(clientDirectory: FileSystemDirectoryHandle): Promise<{profileId: string}>;
  encrypt(options: EncryptOptions): Promise<ArtifactRef>;
  decrypt(options: DecryptOptions): Promise<ArtifactRef>;
  decryptOutputs(options: {task: ArtifactDirectory; outputFile: FileSource; run?: ArtifactDirectory; outputDirectory?: FileSystemDirectoryHandle}): Promise<ArtifactDirectory>;
  provisionRunner(options: {task: ArtifactDirectory; parameterDirectory: FileSystemDirectoryHandle; outputDirectory?: FileSystemDirectoryHandle}): Promise<ArtifactDirectory>;
  encryptBatch(options: {inputName: string; inputDirectory: FileSystemDirectoryHandle; outputDirectory?: FileSystemDirectoryHandle}): Promise<BatchResult>;
  encryptBatch(options: {mode: 'server'; inputDirectories: Record<string, FileSystemDirectoryHandle>; outputDirectory?: FileSystemDirectoryHandle; forceZip64?: boolean}): Promise<BatchResult>;
  decryptBatch(options: {outputName: string; inputDirectory: FileSystemDirectoryHandle; outputDirectory?: FileSystemDirectoryHandle}): Promise<BatchResult>;
  decryptBatch(options: {mode: 'server'; outputName: string; inputFile: FileSource; inputBatch: BatchResult; outputDirectory?: FileSystemDirectoryHandle}): Promise<BatchResult>;
  /** Omit requestId to cancel the currently active operation. */
  cancel(requestId?: number): Promise<{accepted: boolean}>;
  close(): Promise<void>;
}
export function openBrowserClient(options: {
  clientDirectory?: FileSystemDirectoryHandle;
  /** Keys are stored in a profile-ID subdirectory of this directory. */
  keyDirectory?: FileSystemDirectoryHandle;
  /** Defaults to the bundled CI PN15 profile when no task is supplied. */
  profile?: KeyProfile;
  onProgress?: (progress: Progress) => void;
  onArtifact?: (artifact: ArtifactRef) => void;
}): Promise<BrowserClient>;
