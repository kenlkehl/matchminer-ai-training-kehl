import type { ClinicalNote, PatientDocument } from "../types";
import { parseClinicalDate } from "../lib/dateParsing";
import pdfWorkerSrc from "pdfjs-dist/build/pdf.worker.mjs?url";

export interface PdfProgress {
  phase: "text" | "ocr";
  current: number;
  total: number;
}

export async function parsePdfPatientFile(
  file: File,
  onProgress?: (progress: PdfProgress) => void
): Promise<PatientDocument> {
  const pdfjs = await import("pdfjs-dist");
  pdfjs.GlobalWorkerOptions.workerSrc = pdfWorkerSrc;

  const bytes = new Uint8Array(await file.arrayBuffer());
  const pdf = await pdfjs.getDocument({ data: bytes }).promise;
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
    textPages.push(text);
  }

  let rawText = textPages.join("\n\n").trim();
  const textDensity = rawText.length / Math.max(1, pdf.numPages);
  if (textDensity < 300) {
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

  return {
    id: crypto.randomUUID(),
    source: "pdf",
    fileName: file.name,
    notes: [note],
    rawText,
    createdAt: new Date().toISOString()
  };
}

async function ocrPdfPages(pdf: { numPages: number; getPage: (pageNumber: number) => Promise<any> }, onProgress?: (progress: PdfProgress) => void): Promise<string> {
  const { createWorker } = await import("tesseract.js");
  const worker = await createWorker("eng", 1, {
    logger: (message) => {
      if (message.status === "recognizing text") {
        onProgress?.({ phase: "ocr", current: Math.round((message.progress ?? 0) * 100), total: 100 });
      }
    }
  });

  const out: string[] = [];
  try {
    for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
      onProgress?.({ phase: "ocr", current: pageNumber, total: pdf.numPages });
      const page = await pdf.getPage(pageNumber);
      const viewport = page.getViewport({ scale: 2 });
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d");
      if (!context) throw new Error("Canvas is unavailable for PDF OCR");
      canvas.width = Math.floor(viewport.width);
      canvas.height = Math.floor(viewport.height);
      await page.render({ canvasContext: context, viewport }).promise;
      const result = await worker.recognize(canvas);
      out.push(result.data.text.trim());
    }
  } finally {
    await worker.terminate();
  }
  return out.join("\n\n").trim();
}
