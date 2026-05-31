import type { ClinicalNote, PatientDocument } from "../types";
import { parseClinicalDate } from "../lib/dateParsing";
import pdfWorkerSrc from "pdfjs-dist/build/pdf.worker.mjs?url";
import tesseractWorkerSrc from "tesseract.js/dist/worker.min.js?url";
import tesseractCoreSrc from "tesseract.js-core/tesseract-core-lstm.wasm.js?url";

export interface PdfProgress {
  phase: "text" | "ocr";
  current: number;
  total: number;
}

export async function parsePdfPatientFile(
  file: File,
  onProgress?: (progress: PdfProgress) => void
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

    const bytes = new Uint8Array(await file.arrayBuffer());
    logPdfDebug("Read PDF bytes", { byteLength: bytes.byteLength });
    const pdf = await pdfjs.getDocument({ data: bytes }).promise;
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
      rawText = await ocrPdfPages(pdf, onProgress);
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

async function ocrPdfPages(pdf: { numPages: number; getPage: (pageNumber: number) => Promise<any> }, onProgress?: (progress: PdfProgress) => void): Promise<string> {
  const { createWorker } = await import("tesseract.js");
  const workerPath = createTesseractWorkerUrl();
  logPdfDebug("Creating Tesseract worker", {
    workerPath,
    corePath: tesseractCoreSrc,
    pages: pdf.numPages
  });
  let worker: Awaited<ReturnType<typeof createWorker>> | null = null;
  try {
    worker = await createWorker("eng", 1, {
      workerPath,
      workerBlobURL: false,
      corePath: tesseractCoreSrc,
      logger: (message) => {
        logPdfDebug("Tesseract progress", message);
        if (message.status === "recognizing text") {
          onProgress?.({ phase: "ocr", current: Math.round((message.progress ?? 0) * 100), total: 100 });
        }
      },
      errorHandler: (error) => {
        logPdfError("Tesseract worker errorHandler", error);
      }
    });
    logPdfDebug("Tesseract worker ready");

    const out: string[] = [];
    for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
      onProgress?.({ phase: "ocr", current: pageNumber, total: pdf.numPages });
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

function createTesseractWorkerUrl(): string {
  const source = `var process = undefined, require = undefined, module = undefined, exports = undefined;\nself.process = undefined;\nself.require = undefined;\nself.module = undefined;\nself.exports = undefined;\nimportScripts(${JSON.stringify(tesseractWorkerSrc)});`;
  return URL.createObjectURL(new Blob([source], { type: "application/javascript" }));
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
