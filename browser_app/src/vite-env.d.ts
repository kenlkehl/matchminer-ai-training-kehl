/// <reference types="vite/client" />

declare module "pdfjs-dist/build/pdf.worker.mjs?url" {
  const src: string;
  export default src;
}

declare module "tesseract.js/dist/tesseract.esm.min.js" {
  const tesseract: unknown;
  export default tesseract;
}

interface MatchMinerLocalOcrResult {
  text: string;
  engine: string;
}

interface MatchMinerRuntimeProgress {
  status?: string;
  modelId?: string;
  name?: string;
  file?: string;
  current?: number;
  total?: number;
  progress?: number;
  overallPercent?: number;
  loaded?: number;
  totalBytes?: number;
  loadedBytes?: number;
  fileLoadedBytes?: number;
  fileTotalBytes?: number;
  detail?: string;
}

interface MatchMinerElectronApi {
  parsePdfWithLocalOcr: (request: { fileName: string; bytes: ArrayBuffer }) => Promise<MatchMinerLocalOcrResult>;
  stopCurrentJob: () => Promise<unknown>;
  runtimeStatus: () => Promise<{
    llama: { running: boolean; port: number | null; modelPath: string | null; contextTokens: number | null };
    onnx: { workerRunning: boolean; pendingJobs: number };
  }>;
  prepareRuntimeArtifacts: (onProgress?: (progress: MatchMinerRuntimeProgress) => void) => Promise<unknown>;
  warmRuntime: (request: {
    task: "text-generation" | "feature-extraction" | "text-classification";
    modelId?: string;
    contextTokens?: number;
    llamaModelRepo?: string;
    llamaModelFile?: string;
  }, onProgress?: (progress: MatchMinerRuntimeProgress) => void) => Promise<unknown>;
  generateText: (request: {
    prompt: string;
    maxNewTokens: number;
    contextTokens?: number;
    llamaModelRepo?: string;
    llamaModelFile?: string;
    enableThinking?: boolean;
    systemPrompt?: string;
  }) => Promise<string>;
  tokenizeText: (request: { text: string; contextTokens?: number; llamaModelRepo?: string; llamaModelFile?: string }) => Promise<number[]>;
  detokenizeTokens: (request: { tokens: number[]; contextTokens?: number; llamaModelRepo?: string; llamaModelFile?: string }) => Promise<string>;
  embedTrialSpaceTexts: (request: { modelId: string; texts: string[] }) => Promise<number[][]>;
  scoreTrialCheckerTexts: (request: { modelId: string; texts: string[] }) => Promise<number[]>;
  scoreBoilerplateCheckerTexts: (request: { modelId: string; texts: string[] }) => Promise<number[]>;
}

interface Window {
  matchminerElectron?: MatchMinerElectronApi;
}
