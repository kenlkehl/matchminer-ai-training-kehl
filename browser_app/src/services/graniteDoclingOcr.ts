import type { PdfProgress } from "./pdfIngest";
import { abortError, assertNotAborted } from "../lib/abort";

const GRANITE_DOCLING_MODEL_ID = "onnx-community/granite-docling-258M-ONNX";
const DOCLING_PROMPT = "Convert this page to docling.";
const DOCLING_RENDER_SCALE = 2;
const DOCLING_MAX_NEW_TOKENS = 3072;

type PdfLike = {
  numPages: number;
  getPage: (pageNumber: number) => Promise<any>;
};

type TransformersModule = {
  AutoProcessor: { from_pretrained: (modelId: string) => Promise<any> };
  AutoModelForVision2Seq: { from_pretrained: (modelId: string, options?: Record<string, unknown>) => Promise<any> };
  RawImage: { fromCanvas: (canvas: HTMLCanvasElement | OffscreenCanvas) => any };
  env?: any;
};

let doclingRuntime: Promise<{ processor: any; model: any; RawImage: TransformersModule["RawImage"] }> | null = null;

export async function ocrPdfWithGraniteDocling(
  pdf: PdfLike,
  onProgress?: (progress: PdfProgress) => void,
  signal?: AbortSignal
): Promise<string> {
  assertNotAborted(signal);
  if (!("gpu" in navigator)) {
    throw new Error("Granite Docling OCR requires WebGPU");
  }

  onProgress?.({
    phase: "docling",
    current: 0,
    total: pdf.numPages,
    detail: "loading Granite Docling WebGPU",
    percent: 0
  });
  const { processor, model, RawImage } = await getDoclingRuntime();
  assertNotAborted(signal);
  const textPages: string[] = [];
  for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
    assertNotAborted(signal);
    reportDoclingProgress(pdf.numPages, pageNumber, 0, onProgress);
    const page = await pdf.getPage(pageNumber);
    const canvas = await renderCanvasForPage(page, signal);
    try {
      assertNotAborted(signal);
      const rawImage = RawImage.fromCanvas(canvas);
      const docTags = await generateDocTags(processor, model, rawImage);
      assertNotAborted(signal);
      const text = docTagsToPlainText(docTags);
      console.info("[MatchMiner PDF]", "Granite Docling page complete", {
        pageNumber,
        docTagChars: docTags.length,
        textChars: text.length
      });
      reportDoclingProgress(pdf.numPages, pageNumber, 100, onProgress);
      textPages.push(text);
    } finally {
      canvas.width = 0;
      canvas.height = 0;
    }
  }

  const rawText = textPages.join("\n\n").trim();
  console.info("[MatchMiner PDF]", "Granite Docling OCR complete", { totalChars: rawText.length });
  return rawText;
}

export function docTagsToPlainText(docTags: string): string {
  return docTags
    .replace(/<\|end_of_text\|>/g, "")
    .replace(/<loc_\d+>/g, "")
    .replace(/<\/(?:text|page_header|section_header_level_\d+|list_item|caption|formula|table_cell|page_footer)>/g, "\n")
    .replace(/<\/(?:chart|picture|table)>/g, "\n")
    .replace(/<[^>]+>/g, "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[ \t]{2,}/g, " ")
    .trim();
}

async function getDoclingRuntime(): Promise<{ processor: any; model: any; RawImage: TransformersModule["RawImage"] }> {
  if (!doclingRuntime) {
    doclingRuntime = import("@huggingface/transformers").then(async (module) => {
      const transformers = module as unknown as TransformersModule;
      if (transformers.env?.backends?.onnx?.wasm) {
        transformers.env.backends.onnx.wasm.numThreads = crossOriginIsolated ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1)) : 1;
      }
      const processor = await transformers.AutoProcessor.from_pretrained(GRANITE_DOCLING_MODEL_ID);
      const model = await transformers.AutoModelForVision2Seq.from_pretrained(GRANITE_DOCLING_MODEL_ID, {
        dtype: "fp32",
        device: "webgpu"
      });
      return { processor, model, RawImage: transformers.RawImage };
    });
  }
  return doclingRuntime;
}

async function generateDocTags(processor: any, model: any, rawImage: any): Promise<string> {
  const messages = [
    {
      role: "user",
      content: [
        { type: "image" },
        { type: "text", text: DOCLING_PROMPT }
      ]
    }
  ];
  const prompt = processor.apply_chat_template(messages, { add_generation_prompt: true });
  const inputs = await processor(prompt, [rawImage], { do_image_splitting: true });
  const generatedIds = await model.generate({
    ...inputs,
    max_new_tokens: DOCLING_MAX_NEW_TOKENS,
    do_sample: false
  });
  const promptLength = inputs.input_ids?.dims?.at(-1) ?? 0;
  const decoded = processor.batch_decode(generatedIds.slice(null, [promptLength, null]), {
    skip_special_tokens: true
  });
  return String(decoded?.[0] ?? "").trim();
}

async function renderCanvasForPage(page: any, signal?: AbortSignal): Promise<HTMLCanvasElement> {
  assertNotAborted(signal);
  const viewport = page.getViewport({ scale: DOCLING_RENDER_SCALE });
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d", { alpha: false });
  if (!context) throw new Error("Canvas is unavailable for Granite Docling OCR");
  canvas.width = Math.floor(viewport.width);
  canvas.height = Math.floor(viewport.height);
  context.fillStyle = "white";
  context.fillRect(0, 0, canvas.width, canvas.height);
  const renderTask = page.render({ canvasContext: context, viewport });
  const abortRender = () => renderTask.cancel?.();
  signal?.addEventListener("abort", abortRender, { once: true });
  try {
    await renderTask.promise;
  } catch (error) {
    if (signal?.aborted) throw abortError();
    throw error;
  } finally {
    signal?.removeEventListener("abort", abortRender);
  }
  assertNotAborted(signal);
  return canvas;
}

function reportDoclingProgress(totalPages: number, pageNumber: number, pagePercent: number, onProgress?: (progress: PdfProgress) => void): void {
  const boundedPagePercent = Math.max(0, Math.min(100, pagePercent));
  const totalPercent = Math.round(((pageNumber - 1 + boundedPagePercent / 100) / totalPages) * 100);
  onProgress?.({
    phase: "docling",
    current: pageNumber,
    total: totalPages,
    detail: "Granite Docling WebGPU",
    percent: totalPercent
  });
}
