import { MODEL_QUERY_PROMPT } from "../data/defaultSettings";
import { chunkClinicalNotesByTokens, type SerialSummaryChunk } from "../lib/text";
import type { ClinicalNote } from "../types";

type AnyPipeline = (...args: any[]) => Promise<any> | any;

const pipelineCache = new Map<string, Promise<AnyPipeline>>();
const tokenizerCache = new Map<string, Promise<any>>();
const classifierCache = new Map<string, Promise<any>>();

export async function isWebGpuAvailable(): Promise<boolean> {
  const gpu = (navigator as Navigator & { gpu?: unknown }).gpu;
  if (!gpu) return false;
  try {
    const adapter = await (gpu as { requestAdapter: () => Promise<unknown> }).requestAdapter();
    return Boolean(adapter);
  } catch {
    return false;
  }
}

export async function warmModel(modelId: string, task: "text-generation" | "feature-extraction" | "text-classification", dtype: string): Promise<void> {
  if (task === "text-classification") {
    await getClassifier(modelId, dtype);
    return;
  }
  await getPipeline(task, modelId, dtype);
}

export async function generateText(modelId: string, prompt: string, options: { dtype: string; maxNewTokens: number }): Promise<string> {
  const generator = await getPipeline("text-generation", modelId, options.dtype);
  const output = await generator(prompt, {
    max_new_tokens: options.maxNewTokens,
    temperature: 0.2,
    do_sample: false,
    return_full_text: false
  });
  const first = Array.isArray(output) ? output[0] : output;
  return String(first?.generated_text ?? first?.text ?? first ?? "").trim();
}

export async function chunkClinicalNotesForSummary(
  modelId: string,
  notes: ClinicalNote[],
  options: { chunkSizeTokens: number; overlapTokens: number }
): Promise<SerialSummaryChunk[]> {
  const tokenizer = await getTokenizer(modelId);
  return chunkClinicalNotesByTokens(notes, {
    encode: async (text) => tokenIdsFromTokenizerOutput(await tokenizer(text, { add_special_tokens: false })),
    decode: async (tokenIds) => String(await tokenizer.decode(tokenIds, { skip_special_tokens: true }))
  }, options);
}

export async function embedText(modelId: string, text: string, dtype = "q8"): Promise<number[]> {
  const extractor = await getPipeline("feature-extraction", modelId, dtype);
  const output = await extractor(MODEL_QUERY_PROMPT + text, {
    pooling: "mean",
    normalize: true
  });
  if (Array.isArray(output)) return flattenNumberArray(output);
  if (typeof output?.tolist === "function") return flattenNumberArray(output.tolist());
  if (output?.data) return Array.from(output.data as Iterable<number>);
  throw new Error("Embedding model returned an unsupported output shape");
}

export async function scoreTrialChecker(modelId: string, texts: string[], dtype = "q8"): Promise<number[]> {
  return scoreClassifier(modelId, texts, dtype, "sigmoid");
}

export async function scoreBoilerplateChecker(modelId: string, texts: string[], dtype = "q8"): Promise<number[]> {
  return scoreClassifier(modelId, texts, dtype, "softmax_positive");
}

async function getPipeline(task: string, modelId: string, dtype: string): Promise<AnyPipeline> {
  const key = `${task}:${modelId}:${dtype}`;
  if (!pipelineCache.has(key)) {
    pipelineCache.set(
      key,
      import("@huggingface/transformers").then(async (module) => {
        const env = (module as any).env;
        if (env?.backends?.onnx?.wasm) {
          env.backends.onnx.wasm.numThreads = crossOriginIsolated ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1)) : 1;
        }
        return loadWithRootOnnxFallback((subfolder) =>
          (module as any).pipeline(task, modelId, modelOptions(dtype, subfolder))
        );
      })
    );
  }
  return pipelineCache.get(key)!;
}

async function getClassifier(modelId: string, dtype: string): Promise<{ tokenizer: any; model: any }> {
  const key = `${modelId}:${dtype}`;
  if (!classifierCache.has(key)) {
    classifierCache.set(
      key,
      import("@huggingface/transformers").then(async (module) => {
        const env = (module as any).env;
        if (env?.backends?.onnx?.wasm) {
          env.backends.onnx.wasm.numThreads = crossOriginIsolated ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1)) : 1;
        }
        const tokenizer = await getTokenizer(modelId);
        const model = await loadWithRootOnnxFallback((subfolder) =>
          (module as any).AutoModelForSequenceClassification.from_pretrained(modelId, modelOptions(dtype, subfolder))
        );
        return { tokenizer, model };
      })
    );
  }
  return classifierCache.get(key)!;
}

function modelOptions(dtype: string, subfolder?: string): Record<string, unknown> {
  const options: Record<string, unknown> = {
    device: "webgpu"
  };
  if (dtype !== "auto") options.dtype = dtype;
  if (subfolder !== undefined) options.subfolder = subfolder;
  return options;
}

async function loadWithRootOnnxFallback<T>(load: (subfolder?: string) => Promise<T>): Promise<T> {
  try {
    return await load();
  } catch (error) {
    if (!shouldRetryRootOnnx(error)) throw error;
    return load("");
  }
}

function shouldRetryRootOnnx(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return message.includes("Could not locate file") && /\/resolve\/[^/]+\/onnx\//.test(message);
}

async function getTokenizer(modelId: string): Promise<any> {
  if (!tokenizerCache.has(modelId)) {
    tokenizerCache.set(
      modelId,
      import("@huggingface/transformers").then((module) => (module as any).AutoTokenizer.from_pretrained(modelId))
    );
  }
  return tokenizerCache.get(modelId)!;
}

async function scoreClassifier(modelId: string, texts: string[], dtype: string, transform: "sigmoid" | "softmax_positive"): Promise<number[]> {
  const { tokenizer, model } = await getClassifier(modelId, dtype);
  const out: number[] = [];
  for (const text of texts) {
    const inputs = await tokenizer(text, {
      truncation: true,
      padding: true,
      max_length: transform === "sigmoid" ? 4096 : 3192
    });
    const result = await model(inputs);
    const logits = Array.from(result.logits?.data ?? []) as number[];
    out.push(transform === "sigmoid" ? sigmoid(logits[0] ?? 0) : softmaxPositive(logits));
  }
  return out;
}

function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

function softmaxPositive(values: number[]): number {
  if (values.length === 0) return 0;
  const max = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - max));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return (exps[1] ?? exps[0] ?? 0) / total;
}

function flattenNumberArray(value: unknown): number[] {
  if (Array.isArray(value)) {
    if (typeof value[0] === "number") return value as number[];
    return flattenNumberArray(value[0]);
  }
  if (ArrayBuffer.isView(value)) return Array.from(value as unknown as ArrayLike<number>);
  return [];
}

function tokenIdsFromTokenizerOutput(output: unknown): number[] {
  const inputIds = (output as { input_ids?: unknown })?.input_ids ?? output;
  return flattenNumberArray(inputIds).map((value) => Number(value));
}
