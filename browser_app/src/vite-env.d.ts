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

interface MatchMinerElectronApi {
  parsePdfWithLocalOcr: (request: { fileName: string; bytes: ArrayBuffer }) => Promise<MatchMinerLocalOcrResult>;
}

interface Window {
  matchminerElectron?: MatchMinerElectronApi;
}
