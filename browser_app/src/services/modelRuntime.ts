import { DEFAULT_LLAMA_GGUF_FILE, DEFAULT_LLAMA_GGUF_REPO, MODEL_QUERY_PROMPT } from "../data/defaultSettings";
import { abortError, assertNotAborted } from "../lib/abort";
import { chunkClinicalNotesByTokens, chunkTextByTokens, type SerialSummaryChunk } from "../lib/text";
import type { ClinicalNote, ModelSettings } from "../types";

type AnyPipeline = (...args: any[]) => Promise<any> | any;
export type WarmModelProgress = MatchMinerRuntimeProgress & {
  status?: string;
  task?: string;
  model?: string;
  file?: string;
  progress?: number;
  loaded?: number;
  total?: number;
  files?: Record<string, { loaded: number; total: number }>;
};
export type WarmModelProgressCallback = (progress: WarmModelProgress) => void;

interface TextGenerationOptions {
  dtype: string;
  maxNewTokens: number;
  contextTokens?: number;
  enableThinking?: boolean;
  systemPrompt?: string;
  signal?: AbortSignal;
}

const pipelineCache = new Map<string, Promise<AnyPipeline>>();
const tokenizerCache = new Map<string, Promise<any>>();
const classifierCache = new Map<string, Promise<any>>();
const patchedLogitSessions = new WeakSet<object>();
const REASONING_SYSTEM_PROMPT = "Reasoning: high";
let webGpuDetailsLogged = false;
let runtimePreferences: Pick<ModelSettings, "llmBackend" | "onnxBackend" | "llmContextTokens" | "llamaModelRepo" | "llamaModelFile"> | null = null;

export function setRuntimePreferences(settings: Pick<ModelSettings, "llmBackend" | "onnxBackend" | "llmContextTokens" | "llamaModelRepo" | "llamaModelFile">): void {
  runtimePreferences = settings;
}

export async function isWebGpuAvailable(): Promise<boolean> {
  const gpu = (navigator as Navigator & { gpu?: unknown }).gpu;
  if (!gpu) return false;
  try {
    const adapter = await (gpu as { requestAdapter: (options?: unknown) => Promise<unknown> }).requestAdapter({ powerPreference: "high-performance" });
    logWebGpuDetails(adapter);
    return Boolean(adapter);
  } catch {
    return false;
  }
}

export async function warmModel(
  modelId: string,
  task: "text-generation" | "feature-extraction" | "text-classification",
  dtype: string,
  onProgress?: WarmModelProgressCallback,
  signal?: AbortSignal
): Promise<void> {
  assertNotAborted(signal);
  const native = nativeApi();
  if (native && task === "text-generation" && shouldUseNativeLlm()) {
    await native.warmRuntime({
      task,
      contextTokens: runtimePreferences?.llmContextTokens,
      llamaModelRepo: runtimePreferences?.llamaModelRepo ?? DEFAULT_LLAMA_GGUF_REPO,
      llamaModelFile: runtimePreferences?.llamaModelFile ?? DEFAULT_LLAMA_GGUF_FILE
    }, onProgress).catch((error) => rethrowIfSignalAborted(error, signal));
    assertNotAborted(signal);
    return;
  }
  if (native && task !== "text-generation" && shouldUseNativeOnnx()) {
    await native.warmRuntime({ task, modelId }, onProgress).catch((error) => rethrowIfSignalAborted(error, signal));
    assertNotAborted(signal);
    return;
  }
  if (task === "text-classification") {
    await getClassifier(modelId, dtype, onProgress).catch((error) => rethrowIfSignalAborted(error, signal));
    assertNotAborted(signal);
    return;
  }
  await getPipeline(task, modelId, dtype, onProgress).catch((error) => rethrowIfSignalAborted(error, signal));
  assertNotAborted(signal);
}

export async function resetTextGenerationPipeline(modelId: string, dtype: string): Promise<void> {
  if (shouldUseNativeLlm()) return;
  const key = `text-generation:${modelId}:${dtype}`;
  const cached = pipelineCache.get(key);
  pipelineCache.delete(key);
  if (!cached) return;
  try {
    const pipe = await cached;
    await (pipe as { dispose?: () => Promise<unknown> }).dispose?.();
  } catch {
    // The session is already failed or still failing; deleting the cache is the important part.
  }
  await delay(250);
}

export async function generateText(modelId: string, prompt: string, options: TextGenerationOptions): Promise<string> {
  assertNotAborted(options.signal);
  const native = nativeApi();
  const enableThinking = options.enableThinking !== false;
  const systemPrompt = options.systemPrompt ?? REASONING_SYSTEM_PROMPT;
  if (native && shouldUseNativeLlm()) {
    const output = await native.generateText({
      prompt,
      maxNewTokens: options.maxNewTokens,
      contextTokens: options.contextTokens,
      llamaModelRepo: runtimePreferences?.llamaModelRepo ?? DEFAULT_LLAMA_GGUF_REPO,
      llamaModelFile: runtimePreferences?.llamaModelFile ?? DEFAULT_LLAMA_GGUF_FILE,
      enableThinking,
      systemPrompt
    }).catch((error) => rethrowIfSignalAborted(error, options.signal));
    assertNotAborted(options.signal);
    return stripThinkingBlocks(output).trim();
  }
  const generator = await getPipeline("text-generation", modelId, options.dtype).catch((error) => rethrowIfSignalAborted(error, options.signal));
  assertNotAborted(options.signal);
  const contextTokens = Number.isFinite(options.contextTokens) ? Math.max(1, Math.floor(options.contextTokens!)) : 16384;
  const formattedPrompt = formatPromptForModel(generator, modelId, prompt, { enableThinking, systemPrompt });
  const output = await generator(formattedPrompt, {
    max_new_tokens: options.maxNewTokens,
    temperature: 0.2,
    do_sample: false,
    return_full_text: false,
    tokenizer_encode_kwargs: {
      max_length: contextTokens,
      truncation: true
    }
  }).catch((error: unknown) => rethrowIfSignalAborted(error, options.signal));
  assertNotAborted(options.signal);
  const first = Array.isArray(output) ? output[0] : output;
  return stripThinkingBlocks(String(first?.generated_text ?? first?.text ?? first ?? "")).trim();
}

export async function chunkClinicalNotesForSummary(
  modelId: string,
  notes: ClinicalNote[],
  options: { chunkSizeTokens: number; overlapTokens: number; signal?: AbortSignal }
): Promise<SerialSummaryChunk[]> {
  assertNotAborted(options.signal);
  if (shouldUseNativeLlm()) {
    return chunkClinicalNotesByTokens(notes, nativeTokenCodec(), options);
  }
  const tokenizer = await getTokenizer(modelId).catch((error) => rethrowIfSignalAborted(error, options.signal));
  assertNotAborted(options.signal);
  return chunkClinicalNotesByTokens(notes, {
    encode: async (text) => encodeTextWithTokenizer(tokenizer, text),
    decode: async (tokenIds) => String(await tokenizer.decode(tokenIds, { skip_special_tokens: true }))
  }, options);
}

export async function splitSummaryChunkForModel(
  modelId: string,
  chunk: SerialSummaryChunk,
  options: { chunkSizeTokens: number; overlapTokens: number; signal?: AbortSignal }
): Promise<SerialSummaryChunk[]> {
  assertNotAborted(options.signal);
  if (shouldUseNativeLlm()) {
    return chunkTextByTokens(chunk.text, nativeTokenCodec(), {
      ...options,
      fallbackFirstDate: chunk.firstDate,
      fallbackLastDate: chunk.lastDate,
      useFallbackDateRangeWhenNoHeaders: true
    });
  }
  const tokenizer = await getTokenizer(modelId).catch((error) => rethrowIfSignalAborted(error, options.signal));
  assertNotAborted(options.signal);
  return chunkTextByTokens(chunk.text, {
    encode: async (text) => encodeTextWithTokenizer(tokenizer, text),
    decode: async (tokenIds) => String(await tokenizer.decode(tokenIds, { skip_special_tokens: true }))
  }, {
    ...options,
    fallbackFirstDate: chunk.firstDate,
    fallbackLastDate: chunk.lastDate,
    useFallbackDateRangeWhenNoHeaders: true
  });
}

export async function countTextTokens(modelId: string, text: string, signal?: AbortSignal): Promise<number> {
  assertNotAborted(signal);
  if (shouldUseNativeLlm()) {
    const tokenIds = await nativeTokenCodec().encode(text).catch((error) => rethrowIfSignalAborted(error, signal));
    assertNotAborted(signal);
    return tokenIds.length;
  }
  const tokenizer = await getTokenizer(modelId).catch((error) => rethrowIfSignalAborted(error, signal));
  assertNotAborted(signal);
  const tokenIds = await encodeTextWithTokenizer(tokenizer, text).catch((error) => rethrowIfSignalAborted(error, signal));
  assertNotAborted(signal);
  return tokenIds.length;
}

export async function embedText(modelId: string, text: string, dtype = "q8", signal?: AbortSignal): Promise<number[]> {
  assertNotAborted(signal);
  const native = nativeApi();
  if (native && shouldUseNativeOnnx()) {
    const embeddings = await native.embedTrialSpaceTexts({ modelId, texts: [text] }).catch((error) => rethrowIfSignalAborted(error, signal));
    assertNotAborted(signal);
    const embedding = embeddings[0];
    if (!embedding?.length) throw new Error("Native TrialSpace model returned no embedding");
    return embedding;
  }
  const extractor = await getPipeline("feature-extraction", modelId, dtype).catch((error) => rethrowIfSignalAborted(error, signal));
  assertNotAborted(signal);
  const output = await extractor(MODEL_QUERY_PROMPT + text, {
    pooling: "mean",
    normalize: true
  }).catch((error: unknown) => rethrowIfSignalAborted(error, signal));
  assertNotAborted(signal);
  if (Array.isArray(output)) return flattenNumberArray(output);
  if (typeof output?.tolist === "function") return flattenNumberArray(output.tolist());
  if (output?.data) return Array.from(output.data as Iterable<number>);
  throw new Error("Embedding model returned an unsupported output shape");
}

export async function scoreTrialChecker(modelId: string, texts: string[], dtype = "q8", signal?: AbortSignal): Promise<number[]> {
  assertNotAborted(signal);
  const native = nativeApi();
  if (native && shouldUseNativeOnnx()) {
    const scores = await native.scoreTrialCheckerTexts({ modelId, texts }).catch((error) => rethrowIfSignalAborted(error, signal));
    assertNotAborted(signal);
    return scores;
  }
  return scoreClassifier(modelId, texts, dtype, "sigmoid", signal);
}

export async function scoreBoilerplateChecker(modelId: string, texts: string[], dtype = "q8", signal?: AbortSignal): Promise<number[]> {
  assertNotAborted(signal);
  const native = nativeApi();
  if (native && shouldUseNativeOnnx()) {
    const scores = await native.scoreBoilerplateCheckerTexts({ modelId, texts }).catch((error) => rethrowIfSignalAborted(error, signal));
    assertNotAborted(signal);
    return scores;
  }
  return scoreClassifier(modelId, texts, dtype, "softmax_positive", signal);
}

function nativeApi(): MatchMinerElectronApi | null {
  return typeof window !== "undefined" ? window.matchminerElectron ?? null : null;
}

function rethrowIfSignalAborted(error: unknown, signal?: AbortSignal): never {
  if (signal?.aborted) throw abortError();
  throw error;
}

function shouldUseNativeLlm(): boolean {
  return Boolean(nativeApi()) && runtimePreferences?.llmBackend !== "browser-webgpu";
}

function shouldUseNativeOnnx(): boolean {
  return Boolean(nativeApi()) && runtimePreferences?.onnxBackend !== "browser-webgpu";
}

function nativeTokenCodec() {
  const native = nativeApi();
  if (!native) throw new Error("Native runtime is unavailable");
  const requestModel = {
    contextTokens: runtimePreferences?.llmContextTokens,
    llamaModelRepo: runtimePreferences?.llamaModelRepo ?? DEFAULT_LLAMA_GGUF_REPO,
    llamaModelFile: runtimePreferences?.llamaModelFile ?? DEFAULT_LLAMA_GGUF_FILE
  };
  return {
    encode: async (text: string) => native.tokenizeText({ text, ...requestModel }),
    decode: async (tokenIds: number[]) => native.detokenizeTokens({ tokens: tokenIds, ...requestModel })
  };
}

async function getPipeline(task: string, modelId: string, dtype: string, onProgress?: WarmModelProgressCallback): Promise<AnyPipeline> {
  const key = `${task}:${modelId}:${dtype}`;
  if (!pipelineCache.has(key)) {
    pipelineCache.set(
      key,
      import("@huggingface/transformers").then(async (module) => {
        const env = (module as any).env;
        if (env?.backends?.onnx?.wasm) {
          env.backends.onnx.wasm.numThreads = crossOriginIsolated ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1)) : 1;
        }
        const pipe = await loadWithRootOnnxFallback<AnyPipeline>((subfolder) =>
          (module as any).pipeline(task, modelId, modelOptions(dtype, subfolder, onProgress)) as Promise<AnyPipeline>
        );
        if (task === "text-generation") patchGenerationLogitSessions(pipe, module as any);
        return pipe;
      })
    );
  }
  return pipelineCache.get(key)!;
}

async function getClassifier(modelId: string, dtype: string, onProgress?: WarmModelProgressCallback): Promise<{ tokenizer: any; model: any }> {
  const key = `${modelId}:${dtype}`;
  if (!classifierCache.has(key)) {
    classifierCache.set(
      key,
      import("@huggingface/transformers").then(async (module) => {
        const env = (module as any).env;
        if (env?.backends?.onnx?.wasm) {
          env.backends.onnx.wasm.numThreads = crossOriginIsolated ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1)) : 1;
        }
        const tokenizer = await getTokenizer(modelId, onProgress);
        const model = await loadWithRootOnnxFallback((subfolder) =>
          (module as any).AutoModelForSequenceClassification.from_pretrained(modelId, modelOptions(dtype, subfolder, onProgress))
        );
        return { tokenizer, model };
      })
    );
  }
  return classifierCache.get(key)!;
}

function modelOptions(dtype: string, subfolder?: string, onProgress?: WarmModelProgressCallback): Record<string, unknown> {
  const options: Record<string, unknown> = {
    device: "webgpu"
  };
  if (dtype !== "auto") options.dtype = dtype;
  if (subfolder !== undefined) options.subfolder = subfolder;
  if (onProgress) options.progress_callback = onProgress;
  return options;
}

function patchGenerationLogitSessions(pipe: unknown, transformers: { Tensor?: new (...args: any[]) => { ort_tensor?: unknown } }): void {
  const Tensor = transformers.Tensor;
  const sessions = (pipe as { model?: { sessions?: Record<string, unknown> } } | null)?.model?.sessions;
  if (!Tensor || !sessions) return;

  for (const session of Object.values(sessions)) {
    const run = (session as { run?: unknown }).run;
    const inputNames = (session as { inputNames?: unknown }).inputNames;
    if (typeof run !== "function" || !Array.isArray(inputNames) || !inputNames.includes("num_logits_to_keep")) continue;
    if (patchedLogitSessions.has(session as object)) continue;

    const originalRun = run.bind(session);
    (session as { run: typeof originalRun }).run = (feeds: Record<string, unknown>, ...args: unknown[]) => {
      if (feeds?.num_logits_to_keep) {
        return originalRun({ ...feeds, num_logits_to_keep: new Tensor("int64", [1n], []).ort_tensor }, ...args);
      }
      return originalRun(feeds, ...args);
    };
    patchedLogitSessions.add(session as object);
  }
}

function formatPromptForModel(generator: AnyPipeline, modelId: string, prompt: string, options: { enableThinking: boolean; systemPrompt: string }): string {
  if (!shouldUseChatTemplate(modelId)) return prompt;
  const tokenizer = (generator as { tokenizer?: { apply_chat_template?: unknown } }).tokenizer;
  if (typeof tokenizer?.apply_chat_template !== "function") return prompt;
  try {
    return String(tokenizer.apply_chat_template([
      { role: "system", content: options.systemPrompt },
      { role: "user", content: prompt }
    ], {
      tokenize: false,
      add_generation_prompt: true,
      enable_thinking: options.enableThinking
    }));
  } catch {
    try {
      return String(tokenizer.apply_chat_template([{ role: "user", content: prompt }], {
        tokenize: false,
        add_generation_prompt: true,
        enable_thinking: options.enableThinking
      }));
    } catch {
      return prompt;
    }
  }
}

function shouldUseChatTemplate(modelId: string): boolean {
  return /(^|\/)LFM2(?:\.5)?-/i.test(modelId) || /LiquidAI\/LFM2/i.test(modelId);
}

export function stripThinkingBlocks(text: string): string {
  const thinking = text.match(/<think>([\s\S]*?)<\/think>/i)?.[1]?.trim();
  if (thinking) {
    console.info("[MatchMiner LLM Thinking]", thinking);
  }
  let cleaned = text.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  cleaned = cleaned.replace(/<\/?think>/gi, "").trim();
  return cleaned;
}

function logWebGpuDetails(adapter: unknown): void {
  if (webGpuDetailsLogged || !adapter) return;
  webGpuDetailsLogged = true;

  const raw = adapter as { info?: unknown; limits?: Record<string, unknown>; features?: Set<string> };
  const limits = raw.limits ?? {};
  console.info("[MatchMiner WebGPU] Adapter", {
    info: raw.info ?? null,
    limits: {
      maxBufferSize: limits.maxBufferSize,
      maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup
    },
    features: raw.features ? Array.from(raw.features).sort() : []
  });
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
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

async function getTokenizer(modelId: string, onProgress?: WarmModelProgressCallback): Promise<any> {
  if (!tokenizerCache.has(modelId)) {
    tokenizerCache.set(
      modelId,
      import("@huggingface/transformers").then((module) =>
        (module as any).AutoTokenizer.from_pretrained(modelId, onProgress ? { progress_callback: onProgress } : undefined)
      )
    );
  }
  return tokenizerCache.get(modelId)!;
}

async function scoreClassifier(modelId: string, texts: string[], dtype: string, transform: "sigmoid" | "softmax_positive", signal?: AbortSignal): Promise<number[]> {
  const { tokenizer, model } = await getClassifier(modelId, dtype).catch((error) => rethrowIfSignalAborted(error, signal));
  const out: number[] = [];
  for (const text of texts) {
    assertNotAborted(signal);
    const inputs = await tokenizer(text, {
      truncation: true,
      padding: true,
      max_length: transform === "sigmoid" ? 4096 : 3192
    }).catch((error: unknown) => rethrowIfSignalAborted(error, signal));
    assertNotAborted(signal);
    const result = await model(inputs).catch((error: unknown) => rethrowIfSignalAborted(error, signal));
    assertNotAborted(signal);
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

async function encodeTextWithTokenizer(tokenizer: any, text: string): Promise<number[]> {
  const tokenIds = tokenIdsFromTokenizerOutput(await tokenizer(text, { add_special_tokens: false }));
  if (text.trim() && tokenIds.length === 0) {
    throw new Error("Tokenizer returned no input_ids for non-empty text");
  }
  return tokenIds;
}

function flattenNumberArray(value: unknown): number[] {
  if (typeof value === "number") return Number.isFinite(value) ? [value] : [];
  if (typeof value === "bigint") return [Number(value)];
  if (Array.isArray(value)) {
    return value.flatMap((item) => flattenNumberArray(item));
  }
  const viewLength = (value as { length?: unknown } | null)?.length;
  if (ArrayBuffer.isView(value) && typeof viewLength === "number") {
    return Array.from(value as unknown as ArrayLike<number | bigint>)
      .map((item) => Number(item))
      .filter(Number.isFinite);
  }
  if (value && typeof value === "object" && "data" in value) {
    return flattenNumberArray((value as { data?: unknown }).data);
  }
  if (value && typeof (value as { tolist?: unknown }).tolist === "function") {
    return flattenNumberArray((value as { tolist: () => unknown }).tolist());
  }
  if (value && typeof value !== "string" && typeof (value as { [Symbol.iterator]?: unknown })[Symbol.iterator] === "function") {
    return Array.from(value as Iterable<unknown>).flatMap((item) => flattenNumberArray(item));
  }
  return [];
}

export function tokenIdsFromTokenizerOutput(output: unknown): number[] {
  const inputIds = (output as { input_ids?: unknown })?.input_ids ?? output;
  return flattenNumberArray(inputIds).map((value) => Number(value));
}
