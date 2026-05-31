/// <reference types="vite/client" />

declare module "pdfjs-dist/build/pdf.worker.mjs?url" {
  const src: string;
  export default src;
}

declare module "tesseract.js/dist/tesseract.esm.min.js" {
  const tesseract: unknown;
  export default tesseract;
}
