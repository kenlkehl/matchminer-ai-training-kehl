import type { ClinicalNote, ModelSettings, PatientDocument } from "../types";
import { parseClinicalDate } from "../lib/dateParsing";
import { ocrPdfWithGraniteDocling } from "./graniteDoclingOcr";
import pdfWorkerSrc from "pdfjs-dist/build/pdf.worker.mjs?url";
import tesseractWorkerSrc from "tesseract.js/dist/worker.min.js?url";
import tesseractCoreSrc from "tesseract.js-core/tesseract-core-lstm.wasm.js?url";

type TesseractBrowserModule = {
  createWorker: (
    langs?: string | string[],
    oem?: number,
    options?: {
      workerPath?: string;
      workerBlobURL?: boolean;
      corePath?: string;
      logger?: (message: { status?: string; progress?: number; [key: string]: unknown }) => void;
      errorHandler?: (error: unknown) => void;
    }
  ) => Promise<{
    recognize: (image: HTMLCanvasElement) => Promise<{ data: { text: string } }>;
    terminate: () => Promise<unknown>;
  }>;
};

export interface PdfProgress {
  phase: "text" | "ocr" | "docling" | "local-ocr";
  current: number;
  total: number;
  detail?: string;
  percent?: number;
}

export interface PdfParseOptions {
  ocrMode?: ModelSettings["pdfOcrMode"];
}

export async function parsePdfPatientFile(
  file: File,
  onProgress?: (progress: PdfProgress) => void,
  options: PdfParseOptions = {}
): Promise<PatientDocument> {
  logPdfDebug("Starting PDF ingest", {
    fileName: file.name,
    fileSize: file.size,
    fileType: file.type || "(none)",
    pdfWorkerSrc
  });
  try {
    const pdfjs = await import("pdfjs-dist");
    pdfjs.GlobalWorkerOptions.workerSrc = pdfWorkerSrc;

    const sourceBytes = new Uint8Array(await file.arrayBuffer());
    const pdfBytes = sourceBytes.slice();
    logPdfDebug("Read PDF bytes", { byteLength: sourceBytes.byteLength });
    const pdf = await pdfjs.getDocument({ data: pdfBytes }).promise;
    logPdfDebug("PDF.js document loaded", { pages: pdf.numPages });
    const textPages: string[] = [];

    for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
      onProgress?.({ phase: "text", current: pageNumber, total: pdf.numPages });
      const page = await pdf.getPage(pageNumber);
      const content = await page.getTextContent();
      const text = content.items
        .map((item: unknown) => (typeof item === "object" && item && "str" in item ? String((item as { str: string }).str) : ""))
        .join(" ")
        .replace(/\s+/g, " ")
        .trim();
      logPdfDebug("PDF.js page text extracted", {
        pageNumber,
        itemCount: content.items.length,
        textChars: text.length
      });
      textPages.push(text);
    }

    let rawText = textPages.join("\n\n").trim();
    logPdfDebug("PDF.js text extraction complete", {
      totalChars: rawText.length,
      perPageChars: textPages.map((text) => text.length)
    });
    if (!rawText) {
      logPdfDebug("No embedded PDF text found; starting OCR fallback");
      const ocrMode = options.ocrMode ?? "auto";
      if (ocrMode === "auto" || ocrMode === "granite") {
        rawText = await tryGraniteDoclingOcr(pdf, ocrMode, onProgress);
      } else if (ocrMode === "local") {
        rawText = await tryLocalPdfOcr(file.name, sourceBytes, ocrMode, onProgress);
      }
      if (!rawText && (ocrMode === "auto" || ocrMode === "browser")) {
        rawText = await ocrPdfPages(pdf, onProgress);
      }
    }

    if (!rawText.trim()) throw new Error("No text could be extracted from the PDF");

    const today = parseClinicalDate(new Date());
    const note: ClinicalNote = {
      id: "pdf-0",
      dateInput: today.input,
      isoDate: today.isoDate,
      epochMs: today.epochMs,
      text: rawText
    };

    logPdfDebug("PDF ingest complete", { totalChars: rawText.length });
    return {
      id: crypto.randomUUID(),
      source: "pdf",
      fileName: file.name,
      notes: [note],
      rawText,
      createdAt: new Date().toISOString()
    };
  } catch (error) {
    logPdfError("PDF ingest failed", error);
    throw error;
  }
}

async function tryGraniteDoclingOcr(
  pdf: { numPages: number; getPage: (pageNumber: number) => Promise<any> },
  ocrMode: ModelSettings["pdfOcrMode"],
  onProgress?: (progress: PdfProgress) => void
): Promise<string> {
  try {
    const text = (await ocrPdfWithGraniteDocling(pdf, onProgress)).trim();
    if (!text) throw new Error("Granite Docling returned no text");
    return text;
  } catch (error) {
    logPdfError("Granite Docling OCR failed", error);
    if (ocrMode === "granite") throw error;
    return "";
  }
}

async function tryLocalPdfOcr(
  fileName: string,
  pdfBytes: Uint8Array,
  ocrMode: ModelSettings["pdfOcrMode"],
  onProgress?: (progress: PdfProgress) => void
): Promise<string> {
  if (ocrMode === "browser") return "";
  const localOcr = window.matchminerElectron?.parsePdfWithLocalOcr;
  if (!localOcr) {
    const message = "Local PDF OCR is unavailable because the Electron preload API is not present";
    logPdfDebug(message);
    if (ocrMode === "local") throw new Error(message);
    return "";
  }

  onProgress?.({ phase: "local-ocr", current: 0, total: 1, detail: "Docling/OCRmyPDF" });
  try {
    const result = await localOcr({ fileName, bytes: toArrayBuffer(pdfBytes) });
    const text = result.text.trim();
    logPdfDebug("Local OCR complete", { engine: result.engine, textChars: text.length });
    if (!text) throw new Error(`Local OCR returned no text (${result.engine})`);
    onProgress?.({ phase: "local-ocr", current: 1, total: 1, detail: result.engine, percent: 100 });
    return text;
  } catch (error) {
    logPdfError("Local OCR failed", error);
    if (ocrMode === "local") throw error;
    return "";
  }
}

async function ocrPdfPages(pdf: { numPages: number; getPage: (pageNumber: number) => Promise<any> }, onProgress?: (progress: PdfProgress) => void): Promise<string> {
  const tesseractWorkerScriptUrl = toAbsoluteAssetUrl(tesseractWorkerSrc);
  const tesseractCoreUrl = toAbsoluteAssetUrl(tesseractCoreSrc);
  const workerPath = createTesseractWorkerUrl(tesseractWorkerScriptUrl);
  let worker: Awaited<ReturnType<TesseractBrowserModule["createWorker"]>> | null = null;
  let activeOcrPage = 0;
  const reportOcrProgress = (pageNumber: number, pagePercent: number) => {
    const boundedPagePercent = Math.max(0, Math.min(100, pagePercent));
    const totalPercent = Math.round(((pageNumber - 1 + boundedPagePercent / 100) / pdf.numPages) * 100);
    onProgress?.({
      phase: "ocr",
      current: pageNumber,
      total: pdf.numPages,
      detail: `page ${pageNumber} of ${pdf.numPages}`,
      percent: totalPercent
    });
  };
  try {
    logPdfDebug("Loading Tesseract browser bundle");
    const { default: tesseract } = (await import("tesseract.js/dist/tesseract.esm.min.js")) as { default: TesseractBrowserModule };
    logPdfDebug("Creating Tesseract worker", {
      workerPath,
      tesseractWorkerScriptUrl,
      corePath: tesseractCoreUrl,
      pages: pdf.numPages
    });
    worker = await tesseract.createWorker("eng", 1, {
      workerPath,
      workerBlobURL: false,
      corePath: tesseractCoreUrl,
      logger: (message) => {
        logPdfDebug("Tesseract progress", message);
        if (message.status === "recognizing text" && activeOcrPage > 0) {
          reportOcrProgress(activeOcrPage, (message.progress ?? 0) * 100);
        }
      },
      errorHandler: (error) => {
        logPdfError("Tesseract worker errorHandler", error);
      }
    });
    logPdfDebug("Tesseract worker ready");

    const out: string[] = [];
    for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
      activeOcrPage = pageNumber;
      reportOcrProgress(pageNumber, 0);
      const page = await pdf.getPage(pageNumber);
      const viewport = page.getViewport({ scale: 2 });
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d");
      if (!context) throw new Error("Canvas is unavailable for PDF OCR");
      canvas.width = Math.floor(viewport.width);
      canvas.height = Math.floor(viewport.height);
      logPdfDebug("Rendering PDF page for OCR", { pageNumber, width: canvas.width, height: canvas.height });
      await page.render({ canvasContext: context, viewport }).promise;
      const result = await worker.recognize(canvas);
      const text = result.data.text.trim();
      logPdfDebug("OCR page complete", { pageNumber, textChars: text.length });
      reportOcrProgress(pageNumber, 100);
      out.push(text);
    }
    const rawText = out.join("\n\n").trim();
    logPdfDebug("OCR complete", { totalChars: rawText.length });
    return rawText;
  } catch (error) {
    logPdfError("OCR failed", error);
    throw error;
  } finally {
    if (worker) await worker.terminate().catch((error) => logPdfError("Tesseract worker termination failed", error));
    URL.revokeObjectURL(workerPath);
  }
}

function createTesseractWorkerUrl(workerScriptUrl: string): string {
  const source = `var process = undefined, require = undefined, module = undefined, exports = undefined;\nself.process = undefined;\nself.require = undefined;\nself.module = undefined;\nself.exports = undefined;\nimportScripts(${JSON.stringify(workerScriptUrl)});`;
  return URL.createObjectURL(new Blob([source], { type: "application/javascript" }));
}

function toAbsoluteAssetUrl(src: string): string {
  return new URL(src, window.location.href).href;
}

function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  const copy = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(copy).set(bytes);
  return copy;
}

function logPdfDebug(message: string, details?: unknown): void {
  console.info("[MatchMiner PDF]", message, details ?? "");
}

function logPdfError(message: string, error: unknown): void {
  console.error("[MatchMiner PDF]", message, serializeError(error));
}

function serializeError(error: unknown): unknown {
  if (error instanceof Error) {
    return {
      name: error.name,
      message: error.message,
      stack: error.stack
    };
  }
  return error;
}
