import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import {
  AlertTriangle,
  Brain,
  CircleStop,
  Database,
  Download,
  ExternalLink,
  FileText,
  Loader2,
  Lock,
  Play,
  RefreshCcw,
  Settings,
  Table2,
  Trash2,
  Upload,
  X
} from "lucide-react";
import type { MatchResult, ModelSettings, PatientDocument, PromptKey, StatusMessage, TrialSpaceRecord } from "./types";
import { DEFAULT_PROMPTS } from "./data/defaultPrompts";
import { DEFAULT_LFM_CONTEXT_TOKENS, DEFAULT_MODEL_SETTINGS } from "./data/defaultSettings";
import { assertNotAborted, isAbortError } from "./lib/abort";
import { ctGovStudyUrl } from "./lib/ctGov";
import { buildExtractiveFallbackSummary, fillPrompt, splitBoilerplate, type SerialSummaryChunk } from "./lib/text";
import { hashTextEmbedding } from "./lib/hashEmbedding";
import { parseCsvPatientFile } from "./services/csvIngest";
import { parsePdfPatientFile, type PdfProgress } from "./services/pdfIngest";
import { clearPatientSideData, loadModelSettings, loadPrompts, resetPrompt, saveModelSettings, savePrompt, saveTrialIndex } from "./services/storage";
import { chunkClinicalNotesForSummary, countTextTokens, embedText, generateText, isWebGpuAvailable, resetTextGenerationPipeline, setRuntimePreferences, splitSummaryChunkForModel, warmModel, type WarmModelProgress } from "./services/modelRuntime";
import { DEFAULT_EMBEDDED_TRIAL_INDEX_URL, ensureTrialEmbeddings, fetchCtGovCancerTrials, loadOrFetchTrialIndex } from "./services/trialIndex";
import { fetchEmbeddedTrialIndex, parseEmbeddedTrialIndexFile, type EmbeddedTrialIndexFetchProgress } from "./services/trialImport";
import { retrieveByEmbedding, scoreAndRankMatches } from "./services/matching";

const promptOrder: PromptKey[] = ["patientSummary", "trialSpaceExtraction", "trialDeepScreen", "boilerplateDeepScreen"];
const DEFAULT_BROWSER_LLM_CONTEXT_TOKENS = DEFAULT_LFM_CONTEXT_TOKENS;
const MIN_BROWSER_LLM_CONTEXT_TOKENS = 4096;
const BROWSER_LLM_CONTEXT_MARGIN_TOKENS = 128;
const MIN_ADAPTIVE_SUMMARY_CHUNK_TOKENS = 16;
const MIN_SUMMARY_RETRY_CHUNK_TOKENS = 256;
const MAX_SERIAL_SUMMARY_ATTEMPTS = 5;
const THINKING_SUMMARY_MAX_TOKENS = DEFAULT_MODEL_SETTINGS.maxSummaryTokens;
const LEGACY_SUMMARY_MAX_TOKENS = 1600;

interface TrialProgress {
  phase: "download" | "extract" | "embed" | "import";
  current: number;
  total?: number;
  detail?: string;
  percent?: number;
  unit?: "count" | "bytes";
}

interface SummaryProgress {
  current: number;
  total?: number;
  detail?: string;
  percent?: number;
}

interface ModelCacheProgress {
  label: string;
  detail?: string;
  current: number;
  total: number;
  percent?: number;
  loadedBytes?: number;
  totalBytes?: number;
}

interface ModelWarmupStep {
  label: string;
  modelId: string;
  task: "text-generation" | "feature-extraction" | "text-classification";
  dtype: string;
}

export default function App() {
  const [webGpu, setWebGpu] = useState<boolean | null>(null);
  const [initialized, setInitialized] = useState(false);
  const [patientDocument, setPatientDocument] = useState<PatientDocument | null>(null);
  const [summary, setSummary] = useState("");
  const [patientBoilerplate, setPatientBoilerplate] = useState("");
  const [trialIndex, setTrialIndex] = useState<TrialSpaceRecord[]>([]);
  const [matches, setMatches] = useState<MatchResult[]>([]);
  const [settings, setSettings] = useState<ModelSettings>(DEFAULT_MODEL_SETTINGS);
  const [prompts, setPrompts] = useState<Record<PromptKey, string>>(
    Object.fromEntries(promptOrder.map((key) => [key, DEFAULT_PROMPTS[key].defaultValue])) as Record<PromptKey, string>
  );
  const [activePrompt, setActivePrompt] = useState<PromptKey>("patientSummary");
  const [showInputs, setShowInputs] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [status, setStatus] = useState<StatusMessage[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [stopRequested, setStopRequested] = useState(false);
  const [pdfProgress, setPdfProgress] = useState<PdfProgress | null>(null);
  const [summaryProgress, setSummaryProgress] = useState<SummaryProgress | null>(null);
  const [trialProgress, setTrialProgress] = useState<TrialProgress | null>(null);
  const [modelCacheProgress, setModelCacheProgress] = useState<ModelCacheProgress | null>(null);
  const [embeddedTrialUrl, setEmbeddedTrialUrl] = useState(DEFAULT_EMBEDDED_TRIAL_INDEX_URL);
  const [showTrialRefresh, setShowTrialRefresh] = useState(false);
  const autoCacheStarted = useRef(false);
  const autoTrialIndexStarted = useRef(false);
  const activeJobController = useRef<AbortController | null>(null);

  useEffect(() => {
    void Promise.all([isWebGpuAvailable(), loadModelSettings(), loadPrompts()]).then(([gpu, savedSettings, savedPrompts]) => {
      const effectiveSettings = window.matchminerElectron ? withThinkingSummaryHeadroom(savedSettings) : {
        ...savedSettings,
        llmBackend: "browser-webgpu" as const,
        onnxBackend: "browser-webgpu" as const
      };
      setWebGpu(gpu);
      setSettings(effectiveSettings);
      setPrompts(savedPrompts);
      const usingNativeRuntime =
        Boolean(window.matchminerElectron) &&
        (effectiveSettings.llmBackend !== "browser-webgpu" || effectiveSettings.onnxBackend !== "browser-webgpu");
      addStatus(
        gpu ? "success" : usingNativeRuntime ? "info" : "warning",
        gpu ? "WebGPU is available." : usingNativeRuntime ? "WebGPU is not available; native runtimes will be used." : "WebGPU is not available in this runtime."
      );
      setInitialized(true);
    });
  }, []);

  useEffect(() => {
    setRuntimePreferences(settings);
  }, [settings]);

  useEffect(() => {
    if (!initialized || autoCacheStarted.current) return;
    if (!window.matchminerElectron && webGpu === false) return;
    const cacheKey = modelCacheStorageKey(settings);
    if (localStorage.getItem(cacheKey)) return;

    autoCacheStarted.current = true;
    void cacheModels({ automatic: true, cacheKey });
  }, [initialized, settings, webGpu]);

  useEffect(() => {
    if (!initialized || busy || autoTrialIndexStarted.current) return;
    void preloadDefaultTrialIndex();
  }, [initialized, busy]);

  const sortedNotes = patientDocument?.notes ?? [];
  const summaryParts = useMemo(() => splitBoilerplate(summary), [summary]);

  function commitSummaryText(nextSummary: string) {
    const summaryText = String(nextSummary ?? "");
    setSummary(summaryText);
    setPatientBoilerplate(splitBoilerplate(summaryText).patientBoilerplate);
  }

  function startJob(label: string, options: { automatic?: boolean } = {}): AbortSignal | null {
    if (activeJobController.current) {
      if (!options.automatic) {
        addStatus("warning", `${busy ?? "Another job"} is already running. Use Stop before starting ${label.toLowerCase()}.`);
      }
      return null;
    }
    const controller = new AbortController();
    activeJobController.current = controller;
    setStopRequested(false);
    setBusy(label);
    return controller.signal;
  }

  function updateJobLabel(signal: AbortSignal, label: string) {
    if (activeJobController.current?.signal === signal) setBusy(label);
  }

  function finishJob(signal: AbortSignal) {
    if (activeJobController.current?.signal !== signal) return;
    activeJobController.current = null;
    setStopRequested(false);
    setBusy(null);
  }

  function stopActiveJob() {
    const controller = activeJobController.current;
    if (!controller || controller.signal.aborted) return;
    controller.abort();
    setStopRequested(true);
    addStatus("warning", `Stop requested for ${busy ?? "current job"}.`);
    void window.matchminerElectron?.stopCurrentJob().catch((error) => {
      console.warn("[MatchMiner] Native stop request failed", error);
    });
  }

  function addStoppedStatus(label: string) {
    addStatus("warning", `${label} stopped.`);
  }

  async function handleFile(file: File) {
    const signal = startJob(file.name.toLowerCase().endsWith(".pdf") ? "Reading PDF" : "Reading records");
    if (!signal) return;
    setPdfProgress(null);
    setSummaryProgress(null);
    setMatches([]);
    commitSummaryText("");
    try {
      const lower = file.name.toLowerCase();
      const doc = lower.endsWith(".csv") ? await parseCsvPatientFile(file) : await parsePdfPatientFile(file, setPdfProgress, { ocrMode: settings.pdfOcrMode, signal });
      assertNotAborted(signal);
      setPatientDocument(doc);
      sessionStorage.setItem("matchminer-current-patient", JSON.stringify({ fileName: doc.fileName, createdAt: doc.createdAt }));
      addStatus("success", `Loaded ${doc.source.toUpperCase()} with ${doc.notes.length} record${doc.notes.length === 1 ? "" : "s"}.`);
    } catch (error) {
      if (isAbortError(error)) addStoppedStatus("Record loading");
      else addStatus("error", errorMessage(error));
    } finally {
      finishJob(signal);
      setPdfProgress(null);
    }
  }

  async function summarize() {
    const loadedPatientDocument = patientDocument;
    if (!loadedPatientDocument) return;
    const startedSignal = startJob("Summarizing");
    if (!startedSignal) return;
    const signal: AbortSignal = startedSignal;
    const activePatientDocument: PatientDocument = loadedPatientDocument;
    setSummaryProgress({ current: 0, detail: "tokenizing clinical record" });
    setMatches([]);
    let usedAutomaticChunkRetry = false;
    try {
      let chunkSizeTokens = normalizeSummaryChunkTokens(settings.summaryChunkTokens);
      let finalResult: { summary: string; splitOversizedSegment: boolean; attempt: number; chunkSizeTokens: number } | null = null;
      for (let attempt = 1; attempt <= MAX_SERIAL_SUMMARY_ATTEMPTS; attempt += 1) {
        assertNotAborted(signal);
        try {
          if (attempt > 1) {
            commitSummaryText("");
            setSummaryProgress({
              current: 0,
              detail: `retrying with ${formatCount(chunkSizeTokens)}-token serial chunks`
            });
          }
          finalResult = await runSerialSummaryAttempt(chunkSizeTokens, attempt);
          break;
        } catch (error) {
          if (isAbortError(error)) throw error;
          const nextChunkSizeTokens = nextSmallerSummaryRetryChunkSize(chunkSizeTokens);
          if (
            attempt >= MAX_SERIAL_SUMMARY_ATTEMPTS ||
            nextChunkSizeTokens >= chunkSizeTokens ||
            !isRetryableSummaryGenerationError(error)
          ) {
            throw error;
          }
          await resetTextGenerationPipeline(settings.llmModelId, settings.llmDtype);
          console.warn("[MatchMiner Summary] Retrying serial summary with smaller base chunks", {
            attempt,
            failedChunkSizeTokens: chunkSizeTokens,
            nextChunkSizeTokens,
            error: errorMessage(error)
          });
          usedAutomaticChunkRetry = true;
          addStatus("warning", `LLM summarization attempt ${attempt} failed; retrying with ${formatCount(nextChunkSizeTokens)}-token chunks. ${errorMessage(error)}`);
          chunkSizeTokens = nextChunkSizeTokens;
        }
      }
      assertNotAborted(signal);
      if (!finalResult) {
        throw new Error("Patient summary generation did not produce a result.");
      }
      commitSummaryText(finalResult.summary);
      const retried = finalResult.attempt > 1;
      const chunkDetail = retried ? ` after retrying with ${formatCount(finalResult.chunkSizeTokens)}-token chunks` : "";
      addStatus("success", finalResult.splitOversizedSegment || retried ? `Patient summary generated locally${chunkDetail}.` : "Patient summary generated locally.");
    } catch (error) {
      if (isAbortError(error)) {
        addStoppedStatus("Patient summarization");
        return;
      }
      const fallback = buildExtractiveFallbackSummary(activePatientDocument.notes);
      commitSummaryText(fallback);
      const retryText = usedAutomaticChunkRetry ? " after automatic chunk-size retries" : "";
      addStatus("warning", `LLM summarization failed${retryText}; local extractive summary was used. ${errorMessage(error)}`);
    } finally {
      finishJob(signal);
      setSummaryProgress(null);
    }

    async function runSerialSummaryAttempt(effectiveSummaryChunkTokens: number, attempt: number): Promise<{ summary: string; splitOversizedSegment: boolean; attempt: number; chunkSizeTokens: number }> {
      let prior = "";
      let activeLlmContextTokens = normalizeLlmContextTokens(settings.llmContextTokens);
      const summaryMaxNewTokens = normalizeSummaryMaxNewTokens(settings.maxSummaryTokens);
      const chunks = await chunkClinicalNotesForSummary(settings.llmModelId, activePatientDocument.notes, {
        chunkSizeTokens: effectiveSummaryChunkTokens,
        overlapTokens: Math.min(settings.summaryChunkOverlapTokens, effectiveSummaryChunkTokens - 1),
        signal
      });
      assertNotAborted(signal);
      console.info("[MatchMiner Summary] Prepared serial summary chunks", {
        attempt,
        chunks: chunks.length,
        chunkSizeTokens: effectiveSummaryChunkTokens,
        configuredChunkSizeTokens: settings.summaryChunkTokens,
        overlapTokens: settings.summaryChunkOverlapTokens,
        summaryMaxNewTokens,
        activeLlmContextTokens,
        recordChars: activePatientDocument.rawText.length
      });
      const pending = [...chunks];
      let completed = 0;
      let totalWork = pending.length;
      let splitOversizedSegment = false;
      while (pending.length > 0) {
        assertNotAborted(signal);
        const chunk = pending.shift()!;
        const progressDetail = `${attempt > 1 ? `attempt ${attempt}, ` : ""}${formatCount(chunk.tokenCount)} source tokens, ${chunk.firstDate} to ${chunk.lastDate}`;
        setSummaryProgress({
          current: completed + 1,
          total: totalWork,
          detail: progressDetail,
          percent: percentComplete(completed, totalWork)
        });
        const prompt = buildSummaryPrompt(prompts.patientSummary, prior, chunk);
        const promptTokens = await countTextTokens(settings.llmModelId, prompt, signal);
        assertNotAborted(signal);
        const promptBudget = summaryPromptTokenBudget(summaryMaxNewTokens, activeLlmContextTokens);
        if (promptTokens > promptBudget) {
          const smallerChunkSize = smallerSummaryChunkSize(chunk.tokenCount, promptTokens, promptBudget);
          if (smallerChunkSize >= chunk.tokenCount) {
            throw new Error(`Serial summary prompt is still too large for the local LLM (${formatCount(promptTokens)} prompt tokens; budget ${formatCount(promptBudget)}). Reduce the Patient summary prompt text or Summary tokens setting.`);
          }
          const smallerChunks = await splitSummaryChunkForModel(settings.llmModelId, chunk, {
            chunkSizeTokens: smallerChunkSize,
            overlapTokens: Math.min(settings.summaryChunkOverlapTokens, smallerChunkSize - 1),
            signal
          });
          assertNotAborted(signal);
          if (smallerChunks.length <= 1) {
            throw new Error(`Serial summary prompt is still too large for the local LLM (${formatCount(promptTokens)} prompt tokens; budget ${formatCount(promptBudget)}).`);
          }

          splitOversizedSegment = true;
          totalWork += smallerChunks.length - 1;
          pending.unshift(...smallerChunks);
          const splitDetail = `split segment before LLM call: ${formatCount(promptTokens)} prompt tokens over ${formatCount(promptBudget)} budget`;
          setSummaryProgress({
            current: completed + 1,
            total: totalWork,
            detail: splitDetail,
            percent: percentComplete(completed, totalWork)
          });
          console.warn("[MatchMiner Summary] Split summary segment before LLM call", {
            attempt,
            originalChunkTokens: chunk.tokenCount,
            smallerChunkSize,
            smallerChunks: smallerChunks.length,
            promptTokens,
            promptBudget
          });
          continue;
        }
        console.info("[MatchMiner Summary] Calling local LLM for serial chunk", {
          attempt,
          chunk: completed + 1,
          chunks: totalWork,
          chunkTokens: chunk.tokenCount,
          promptTokens,
          activeLlmContextTokens,
          promptChars: prompt.length,
          priorSummaryChars: prior.length
        });
        try {
          const nextSummary = await generateText(settings.llmModelId, prompt, {
            dtype: settings.llmDtype,
            maxNewTokens: summaryMaxNewTokens,
            contextTokens: activeLlmContextTokens,
            enableThinking: true,
            signal
          });
          assertNotAborted(signal);
          if (!nextSummary.trim()) {
            throw new Error("Local LLM returned no visible patient summary text. The response may have contained only hidden thinking text.");
          }
          prior = nextSummary;
          commitSummaryText(prior);
          console.info("[MatchMiner Summary] Received serial summary chunk response", {
            attempt,
            chunk: completed + 1,
            chunks: totalWork,
            generatedSummaryChars: prior.length
          });
        } catch (error) {
          if (isAbortError(error)) throw error;
          if (!isOversizedGenerationError(error)) throw error;
          await resetTextGenerationPipeline(settings.llmModelId, settings.llmDtype);
          const lowerContextTokens = nextLowerLlmContextTokens(activeLlmContextTokens);
          if (lowerContextTokens < activeLlmContextTokens) {
            activeLlmContextTokens = lowerContextTokens;
            splitOversizedSegment = true;
            pending.unshift(chunk);
            const splitDetail = `restarting local LLM at ${formatCount(activeLlmContextTokens)} context tokens after allocation failure`;
            setSummaryProgress({
              current: completed + 1,
              total: totalWork,
              detail: splitDetail,
              percent: percentComplete(completed, totalWork)
            });
            console.warn("[MatchMiner Summary] Lowered local LLM context and will retry", {
              attempt,
              activeLlmContextTokens,
              chunkTokens: chunk.tokenCount,
              promptTokens,
              error: errorMessage(error)
            });
            continue;
          }
          const smallerChunkSize = smallerSummaryChunkSize(chunk.tokenCount, promptTokens, Math.max(MIN_ADAPTIVE_SUMMARY_CHUNK_TOKENS, promptBudget - 256));
          if (smallerChunkSize >= chunk.tokenCount) throw error;
          const smallerChunks = await splitSummaryChunkForModel(settings.llmModelId, chunk, {
            chunkSizeTokens: smallerChunkSize,
            overlapTokens: Math.min(settings.summaryChunkOverlapTokens, smallerChunkSize - 1),
            signal
          });
          assertNotAborted(signal);
          if (smallerChunks.length <= 1) throw error;

          splitOversizedSegment = true;
          totalWork += smallerChunks.length - 1;
          pending.unshift(...smallerChunks);
          const splitDetail = `split oversized segment into ${smallerChunks.length} smaller segments after ${formatCount(promptTokens)} prompt tokens`;
          setSummaryProgress({
            current: completed + 1,
            total: totalWork,
            detail: splitDetail,
            percent: percentComplete(completed, totalWork)
          });
          console.warn("[MatchMiner Summary] Split oversized summary segment and will retry serially", {
            attempt,
            originalChunkTokens: chunk.tokenCount,
            smallerChunkSize,
            smallerChunks: smallerChunks.length,
            promptTokens,
            error: errorMessage(error)
          });
          continue;
        }
        completed += 1;
        setSummaryProgress({
          current: completed,
          total: totalWork,
          detail: progressDetail,
          percent: percentComplete(completed, totalWork)
        });
      }
      return { summary: prior, splitOversizedSegment, attempt, chunkSizeTokens: effectiveSummaryChunkTokens };
    }
  }

  async function preloadDefaultTrialIndex() {
    const signal = startJob("Loading trial spaces", { automatic: true });
    if (!signal) return;
    autoTrialIndexStarted.current = true;
    setTrialProgress({ phase: "import", current: 0, detail: "checking local trial spaces" });
    try {
      const records = await loadOrFetchTrialIndex(signal, (progress) => {
        setTrialProgress(trialProgressFromEmbeddedFetch(progress));
      });
      assertNotAborted(signal);
      setTrialIndex(records);
      addStatus("success", `Trial spaces ready: ${formatCount(records.length)} loaded.`);
    } catch (error) {
      if (isAbortError(error)) addStoppedStatus("Trial loading");
      else addStatus("warning", `Default trial-space download unavailable. ${errorMessage(error)}`);
    } finally {
      finishJob(signal);
      setTrialProgress(null);
    }
  }

  async function refreshCtGov() {
    const signal = startJob("Downloading ClinicalTrials.gov");
    if (!signal) return;
    setTrialProgress({ phase: "download", current: 0, detail: "starting" });
    try {
      const records = await fetchCtGovCancerTrials({
        pageSize: 1000,
        signal,
        onProgress: ({ records: downloaded, totalCount, page }) => {
          const total = totalCount ? ` of ${totalCount}` : "";
          setTrialProgress({
            phase: "download",
            current: downloaded,
            total: totalCount,
            detail: `page ${page}`,
            percent: percentComplete(downloaded, totalCount)
          });
          addStatus("info", `Downloaded ${downloaded}${total} open phase I-III interventional cancer trial records from ClinicalTrials.gov across ${page} page${page === 1 ? "" : "s"}.`);
        }
      });
      assertNotAborted(signal);
      addStatus("success", `Downloaded ${records.length} public phase I-III interventional cancer trial records from ClinicalTrials.gov.`);
      updateJobLabel(signal, "Extracting trial spaces");
      setTrialProgress({ phase: "extract", current: 0, total: records.length, detail: "warming local LLM", percent: 0 });
      const spaces = await extractTrialSpaces(records, signal);
      assertNotAborted(signal);
      await saveTrialIndex(spaces);
      assertNotAborted(signal);
      setTrialIndex(spaces);
      addStatus("success", `Extracted ${spaces.length} trial spaces from ${records.length} ClinicalTrials.gov trials.`);
    } catch (error) {
      if (isAbortError(error)) addStoppedStatus("ClinicalTrials.gov refresh");
      else addStatus("error", errorMessage(error));
    } finally {
      finishJob(signal);
      setTrialProgress(null);
    }
  }

  async function loadEmbeddedTrialFile(file: File) {
    const signal = startJob("Loading embedded trial index");
    if (!signal) return;
    setTrialProgress({ phase: "import", current: 0, detail: file.name });
    try {
      const records = await parseEmbeddedTrialIndexFile(file, signal);
      await persistEmbeddedTrialIndex(records, file.name, signal);
    } catch (error) {
      if (isAbortError(error)) addStoppedStatus("Embedded trial import");
      else addStatus("error", errorMessage(error));
    } finally {
      finishJob(signal);
      setTrialProgress(null);
    }
  }

  async function loadEmbeddedTrialUrl() {
    const url = embeddedTrialUrl.trim();
    if (!url) {
      addStatus("warning", "Enter a URL for the embedded trial index.");
      return;
    }
    const signal = startJob("Loading embedded trial index");
    if (!signal) return;
    setTrialProgress({ phase: "import", current: 0, detail: "fetching URL", unit: "bytes" });
    try {
      const records = await fetchEmbeddedTrialIndex(url, signal, (progress) => {
        setTrialProgress(trialProgressFromEmbeddedFetch(progress));
      });
      await persistEmbeddedTrialIndex(records, url, signal);
      setShowTrialRefresh(false);
    } catch (error) {
      if (isAbortError(error)) addStoppedStatus("Embedded trial import");
      else addStatus("error", errorMessage(error));
    } finally {
      finishJob(signal);
      setTrialProgress(null);
    }
  }

  async function persistEmbeddedTrialIndex(records: TrialSpaceRecord[], sourceLabel: string, signal?: AbortSignal) {
    assertNotAborted(signal);
    if (!records.length) throw new Error("Embedded trial index contains no trial spaces");
    const embeddingDim = records[0].embedding?.length ?? 0;
    setTrialProgress({ phase: "import", current: records.length, total: records.length, detail: "saving", percent: 100 });
    await saveTrialIndex(records);
    assertNotAborted(signal);
    setTrialIndex(records);
    setMatches([]);
    addStatus("success", `Loaded ${records.length} pre-embedded trial spaces${embeddingDim ? ` (dim=${embeddingDim})` : ""} from ${sourceLabel}.`);
  }

  async function extractTrialSpaces(records: TrialSpaceRecord[], signal: AbortSignal): Promise<TrialSpaceRecord[]> {
    const spaces: TrialSpaceRecord[] = [];
    let fallbackCount = 0;
    addStatus("info", "Extracting trial spaces with the local LLM. This can take a long time for a full CT.gov refresh.");
    await warmModel(settings.llmModelId, "text-generation", settings.llmDtype, undefined, signal);
    for (let index = 0; index < records.length; index += 1) {
      assertNotAborted(signal);
      const record = records[index];
      setTrialProgress({
        phase: "extract",
        current: index + 1,
        total: records.length,
        detail: `${spaces.length} trial spaces found`,
        percent: percentComplete(index, records.length)
      });
      try {
        const prompt = fillPrompt(prompts.trialSpaceExtraction, {
          trial_text: trialDocumentForExtraction(record)
        });
        const response = await generateText(settings.llmModelId, prompt, {
          dtype: settings.llmDtype,
          maxNewTokens: 1800,
          contextTokens: settings.llmContextTokens,
          signal
        });
        assertNotAborted(signal);
        const extracted = parseExtractedTrialSpaces(record, response);
        if (extracted.length) {
          spaces.push(...extracted);
        } else {
          fallbackCount += 1;
          spaces.push(record);
        }
      } catch (error) {
        if (isAbortError(error)) throw error;
        fallbackCount += 1;
        spaces.push(record);
        if (fallbackCount <= 3) {
          addStatus("warning", `Trial-space extraction fell back to one heuristic space for ${record.nctId}. ${errorMessage(error)}`);
        }
      }
      if ((index + 1) % 25 === 0 || index + 1 === records.length) {
        addStatus("info", `Extracted spaces from ${index + 1} of ${records.length} trials; current spaces=${spaces.length}.`);
      }
    }
    setTrialProgress({
      phase: "extract",
      current: records.length,
      total: records.length,
      detail: `${spaces.length} trial spaces found`,
      percent: 100
    });
    if (fallbackCount) {
      addStatus("warning", `${fallbackCount} trial${fallbackCount === 1 ? "" : "s"} used one heuristic fallback space.`);
    }
    return spaces;
  }

  async function cacheModels(options: { automatic?: boolean; cacheKey?: string } = {}) {
    const signal = startJob("Caching models", { automatic: options.automatic });
    if (!signal) return;
    const cacheKey = options.cacheKey ?? modelCacheStorageKey(settings);
    const steps = modelWarmupSteps(settings);
    setModelCacheProgress({
      label: "Preparing model cache",
      detail: "checking required local models",
      current: 0,
      total: steps.length,
      percent: 0
    });
    try {
      for (let index = 0; index < steps.length; index += 1) {
        assertNotAborted(signal);
        const step = steps[index];
        setModelCacheProgress({
          label: `Caching ${step.label}`,
          detail: "checking cache",
          current: index + 1,
          total: steps.length,
          percent: steppedPercent(index, 0, steps.length)
        });
        await warmModel(step.modelId, step.task, step.dtype, (progress) => {
          setModelCacheProgress(modelCacheProgressFromWarmup(step, index, steps.length, progress));
        }, signal);
        assertNotAborted(signal);
        setModelCacheProgress({
          label: `${step.label} ready`,
          detail: "cached locally",
          current: index + 1,
          total: steps.length,
          percent: steppedPercent(index, 100, steps.length)
        });
      }
      assertNotAborted(signal);
      localStorage.setItem(cacheKey, new Date().toISOString());
      setModelCacheProgress({
        label: "Model cache ready",
        detail: "all required models are available locally",
        current: steps.length,
        total: steps.length,
        percent: 100
      });
      addStatus("success", options.automatic ? "First-run model cache complete." : "Model cache warmup complete.");
    } catch (error) {
      addStatus("warning", isAbortError(error) ? "Model cache warmup stopped." : `Model cache warmup stopped: ${errorMessage(error)}`);
      setModelCacheProgress({
        label: "Model cache stopped",
        detail: errorMessage(error),
        current: 0,
        total: steps.length
      });
    } finally {
      finishJob(signal);
      window.setTimeout(() => {
        setModelCacheProgress((progress) => progress?.label === "Model cache ready" || progress?.label === "Model cache stopped" ? null : progress);
      }, 3000);
    }
  }

  async function runMatching() {
    const trimmedSummary = summary.trim();
    if (!trimmedSummary) {
      addStatus("warning", "A patient summary is required before matching.");
      return;
    }
    const signal = startJob("Matching trials");
    if (!signal) return;
    setTrialProgress(null);
    try {
      let records = trialIndex.length ? trialIndex : await loadOrFetchTrialIndex(signal);
      assertNotAborted(signal);
      let patientEmbedding: number[];
      try {
        setTrialProgress({ phase: "embed", current: 0, total: records.length, detail: "preparing embeddings", percent: 0 });
        records = await ensureTrialEmbeddings(records, settings.trialSpaceModelId, (done, total) => {
          setTrialProgress({
            phase: "embed",
            current: done,
            total,
            percent: percentComplete(done, total)
          });
          if (done === total || done % 10 === 0) addStatus("info", `Embedded ${done} of ${total} trial spaces.`);
        }, signal);
        assertNotAborted(signal);
        setTrialProgress({ phase: "embed", current: records.length, total: records.length, detail: "embedding patient summary", percent: 100 });
        patientEmbedding = await embedText(settings.trialSpaceModelId, trimmedSummary, settings.classifierDtype, signal);
      } catch (error) {
        if (isAbortError(error)) throw error;
        addStatus("warning", `TrialSpace model unavailable; using local lexical fallback. ${errorMessage(error)}`);
        records = records.map((record) => ({
          ...record,
          embedding: record.embedding?.length ? record.embedding : hashTextEmbedding(record.trialSpaceText)
        }));
        patientEmbedding = hashTextEmbedding(trimmedSummary);
      }
      assertNotAborted(signal);
      setTrialIndex(records);
      const candidates = retrieveByEmbedding(patientEmbedding, records, settings.retrievalCount);
      const ranked = await scoreAndRankMatches({
        patientSummary: trimmedSummary,
        patientBoilerplate: patientBoilerplate || summaryParts.patientBoilerplate,
        candidates,
        trialCheckerModelId: settings.trialCheckerModelId,
        boilerplateCheckerModelId: settings.boilerplateCheckerModelId,
        dtype: settings.classifierDtype,
        displayCount: settings.displayCount,
        signal
      });
      assertNotAborted(signal);
      const deepScreened = settings.runDeepScreen ? await runDeepScreen(ranked, trimmedSummary, signal) : ranked;
      assertNotAborted(signal);
      setMatches(deepScreened.map((match, index) => ({ ...match, rank: index + 1 })));
      addStatus("success", `Ranked ${deepScreened.length} trial options.`);
    } catch (error) {
      if (isAbortError(error)) addStoppedStatus("Trial matching");
      else addStatus("error", errorMessage(error));
    } finally {
      finishJob(signal);
      setTrialProgress(null);
    }
  }

  async function runDeepScreen(ranked: MatchResult[], patientSummary: string, signal: AbortSignal): Promise<MatchResult[]> {
    const screened: MatchResult[] = [];
    for (const match of ranked) {
      assertNotAborted(signal);
      try {
        const trialPrompt = fillPrompt(prompts.trialDeepScreen, {
          patient_summary: patientSummary,
          trial_summary: match.trial.trialSpaceText
        });
        const trialResponse = await generateText(settings.llmModelId, trialPrompt, {
          dtype: settings.llmDtype,
          maxNewTokens: 700,
          contextTokens: settings.llmContextTokens,
          signal
        });
        assertNotAborted(signal);
        const boilerplatePrompt = fillPrompt(prompts.boilerplateDeepScreen, {
          patient_boilerplate: patientBoilerplate || summaryParts.patientBoilerplate,
          trial_boilerplate: match.trial.boilerplateText
        });
        const boilerplateResponse = await generateText(settings.llmModelId, boilerplatePrompt, {
          dtype: settings.llmDtype,
          maxNewTokens: 600,
          contextTokens: settings.llmContextTokens,
          signal
        });
        assertNotAborted(signal);
        screened.push({
          ...match,
          llmTrialCheckScore: parseFinalScore(trialResponse),
          llmTrialCheckReasoning: trialResponse,
          llmBoilerplateExcluded: parseYesNo(boilerplateResponse),
          llmBoilerplateReasoning: boilerplateResponse
        });
      } catch (error) {
        if (isAbortError(error)) throw error;
        screened.push({ ...match, warnings: [...match.warnings, `Deep screen unavailable: ${errorMessage(error)}`] });
      }
    }
    return screened;
  }

  async function persistPrompt(key: PromptKey, value: string) {
    const next = { ...prompts, [key]: value };
    setPrompts(next);
    await savePrompt(key, value);
  }

  async function restorePrompt(key: PromptKey) {
    const next = await resetPrompt(key);
    setPrompts(next);
  }

  async function persistSettings(next: ModelSettings) {
    setSettings(next);
    await saveModelSettings(next);
  }

  async function deleteLocalPatientData() {
    setPatientDocument(null);
    commitSummaryText("");
    setMatches([]);
    await clearPatientSideData();
    addStatus("success", "Local patient workspace cleared.");
  }

  function addStatus(kind: StatusMessage["kind"], text: string) {
    setStatus((prior) => [{ kind, text }, ...prior].slice(0, 7));
  }

  const busyNow = Boolean(busy);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><Brain size={22} /></div>
          <div>
            <h1>MatchMiner-AI</h1>
            <p>Local clinical trial matching</p>
          </div>
        </div>
        <div className="top-actions">
          {busy && (
            <button className="text-button danger" disabled={stopRequested} onClick={stopActiveJob} type="button">
              {stopRequested ? <Loader2 className="spin" size={16} /> : <CircleStop size={16} />} {stopRequested ? "Stopping" : "Stop"}
            </button>
          )}
          <button className="icon-button" onClick={() => setShowSettings(true)} title="Settings" type="button">
            <Settings size={18} />
          </button>
          <button className="text-button danger" disabled={busyNow} onClick={deleteLocalPatientData} type="button">
            <Trash2 size={16} /> Clear patient
          </button>
        </div>
      </header>

      <section className="notice">
        <Lock size={18} />
        <span>
          Runs locally on this device. Beta software only. It does not replace medical advice, may be wrong, must not make autonomous decisions, does not guarantee eligibility, and does not show whether a trial has slots.
        </span>
      </section>

      {modelCacheProgress && <ModelCacheProgressView progress={modelCacheProgress} />}

      <main className="workspace">
        <section className="left-column">
          <Panel title="Records" icon={<Upload size={18} />}>
            <label className="dropzone">
              <input
                type="file"
                accept=".csv,application/pdf,.pdf,text/csv"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void handleFile(file);
                  event.currentTarget.value = "";
                }}
              />
              <FileText size={34} />
              <div className="dropzone-copy">
                <strong>{patientDocument ? patientDocument.fileName : "Upload medical records"}</strong>
                <span>PDF with all medical records, or CSV with one row per document and columns date and text.</span>
              </div>
            </label>
            <div className="compact-row">
              <button className="text-button" disabled={!patientDocument} onClick={() => setShowInputs((value) => !value)} type="button">
                <Table2 size={16} /> {showInputs ? "Hide inputs" : "View inputs"}
              </button>
              <span className="muted">{sortedNotes.length ? `${sortedNotes.length} dated record${sortedNotes.length === 1 ? "" : "s"}` : "No records loaded"}</span>
            </div>
            {pdfProgress && <ProgressLine label={formatPdfProgress(pdfProgress)} />}
            {showInputs && patientDocument && <InputViewer document={patientDocument} />}
          </Panel>

          <Panel title="Trial index" icon={<Database size={18} />}>
            <div className="readiness-grid trial-readiness">
              <Readiness label="Trial spaces" value={trialIndex.length ? `${trialIndex.length}` : "not loaded"} />
              <Readiness label="Deep screen" value={settings.runDeepScreen ? "on" : "off"} />
            </div>
            <div className="button-grid">
              <button
                className="text-button"
                disabled={busyNow}
                onClick={() => {
                  setEmbeddedTrialUrl((url) => url.trim() || DEFAULT_EMBEDDED_TRIAL_INDEX_URL);
                  setShowTrialRefresh(true);
                }}
                type="button"
              >
                {busy === "Loading embedded trial index" ? <Loader2 className="spin" size={16} /> : <RefreshCcw size={16} />} Refresh trial spaces
              </button>
              <button className="text-button" disabled={busyNow} onClick={refreshCtGov} type="button">
                {busy === "Downloading ClinicalTrials.gov" || busy === "Extracting trial spaces" ? <Loader2 className="spin" size={16} /> : <RefreshCcw size={16} />} CT.gov refresh
              </button>
            </div>
            <div className="trial-import">
              <label className={`text-button file-loader ${busyNow ? "disabled-control" : ""}`}>
                <Upload size={16} /> Load embedded file
                <input
                  type="file"
                  accept=".json,.jsonl,.ndjson,.csv,.parquet,application/json,text/csv"
                  disabled={busyNow}
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) void loadEmbeddedTrialFile(file);
                    event.currentTarget.value = "";
                  }}
                />
              </label>
            </div>
            {trialProgress && <ProgressLine label={formatTrialProgress(trialProgress)} />}
          </Panel>

        </section>

        <section className="center-column">
          <Panel title="Patient summary" icon={<FileText size={18} />}>
            <div className="toolbar">
              <button className="primary-button" disabled={!patientDocument || busyNow} onClick={summarize} type="button">
                {busy === "Summarizing" ? <Loader2 className="spin" size={17} /> : <Brain size={17} />} Summarize
              </button>
              <button className="primary-button" disabled={!summary.trim() || busyNow} onClick={runMatching} type="button">
                {busy === "Matching trials" ? <Loader2 className="spin" size={17} /> : <Play size={17} />} Match trials
              </button>
            </div>
            {summaryProgress && <ProgressLine label={formatSummaryProgress(summaryProgress)} />}
            <textarea
              className="summary-box"
              value={summary}
              onChange={(event) => {
                commitSummaryText(event.target.value);
              }}
              placeholder="Patient summary"
            />
          </Panel>
        </section>

        <section className="right-column">
          <Panel title="Retrieved trial spaces" icon={<Brain size={18} />}>
            {matches.length === 0 ? (
              <div className="empty-state">
                <AlertTriangle size={22} />
                <span>No ranked trials yet.</span>
              </div>
            ) : (
              <div className="results-list">
                {matches.map((match) => (
                  <ResultCard key={match.id} match={match} />
                ))}
              </div>
            )}
          </Panel>
        </section>
      </main>

      {showTrialRefresh && (
        <TrialRefreshDialog
          url={embeddedTrialUrl}
          busy={busy}
          progress={trialProgress}
          onUrlChange={setEmbeddedTrialUrl}
          onClose={() => setShowTrialRefresh(false)}
          onRefresh={() => void loadEmbeddedTrialUrl()}
        />
      )}

      {showSettings && (
        <SettingsDialog
          settings={settings}
          prompts={prompts}
          activePrompt={activePrompt}
          status={status}
          busy={busy}
          webGpu={webGpu}
          summaryProgress={summaryProgress}
          modelCacheProgress={modelCacheProgress}
          onClose={() => setShowSettings(false)}
          onChange={(next) => void persistSettings(next)}
          onActivePromptChange={setActivePrompt}
          onPromptChange={(key, value) => void persistPrompt(key, value)}
          onPromptReset={(key) => void restorePrompt(key)}
          onCacheModels={() => void cacheModels()}
        />
      )}
    </div>
  );
}

function Panel({ title, icon, children }: { title: string; icon: ReactNode; children: ReactNode }) {
  return (
    <section className="panel">
      <div className="panel-heading">
        {icon}
        <h2>{title}</h2>
      </div>
      {children}
    </section>
  );
}

function InputViewer({ document }: { document: PatientDocument }) {
  if (document.source === "pdf") {
    return <textarea className="input-text" value={document.rawText} readOnly />;
  }
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Text</th>
          </tr>
        </thead>
        <tbody>
          {document.notes.map((note) => (
            <tr key={note.id}>
              <td>{note.isoDate}</td>
              <td>{note.text}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Readiness({ label, value }: { label: string; value: string }) {
  return (
    <div className="readiness">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ProgressLine({ label }: { label: string }) {
  return (
    <div className="progress-line">
      <Loader2 className="spin" size={15} />
      <span>{label}</span>
    </div>
  );
}

function ModelCacheProgressView({ progress, compact = false }: { progress: ModelCacheProgress; compact?: boolean }) {
  const percent = typeof progress.percent === "number" ? Math.round(progress.percent) : undefined;
  const bytes = formatProgressBytes(progress);
  const step = progress.total > 0 && progress.current > 0 ? `${progress.current}/${progress.total}` : "";
  return (
    <section className={compact ? "model-cache-panel compact" : "model-cache-panel"}>
      <div className="model-cache-head">
        <div>
          <strong>{progress.label}</strong>
          <span>{[step, progress.detail, bytes].filter(Boolean).join(" - ")}</span>
        </div>
        {typeof percent === "number" && <span className="progress-percent">{percent}%</span>}
      </div>
      <ProgressBar percent={percent} />
    </section>
  );
}

function TrialRefreshDialog({
  url,
  busy,
  progress,
  onUrlChange,
  onClose,
  onRefresh
}: {
  url: string;
  busy: string | null;
  progress: TrialProgress | null;
  onUrlChange: (url: string) => void;
  onClose: () => void;
  onRefresh: () => void;
}) {
  const busyNow = Boolean(busy);
  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="trial-refresh-title">
      <div className="modal trial-refresh-modal">
        <div className="modal-head">
          <h2 id="trial-refresh-title">Refresh trial spaces</h2>
          <button className="icon-button" disabled={busyNow} onClick={onClose} title="Close" type="button">
            <X size={18} />
          </button>
        </div>
        <label>
          Pre-embedded trials URL
          <input
            autoFocus
            disabled={busyNow}
            onChange={(event) => onUrlChange(event.target.value)}
            placeholder={DEFAULT_EMBEDDED_TRIAL_INDEX_URL}
            value={url}
          />
        </label>
        {progress && <ProgressLine label={formatTrialProgress(progress)} />}
        <div className="compact-row end">
          <button className="text-button" disabled={busyNow} onClick={onClose} type="button">
            Cancel
          </button>
          <button className="primary-button" disabled={busyNow || !url.trim()} onClick={onRefresh} type="button">
            {busy === "Loading embedded trial index" ? <Loader2 className="spin" size={16} /> : <Download size={16} />} Refresh
          </button>
        </div>
      </div>
    </div>
  );
}

function ProgressBar({ percent }: { percent?: number }) {
  const safePercent = typeof percent === "number" ? Math.max(0, Math.min(100, percent)) : undefined;
  const ariaProps = safePercent === undefined ? {} : { "aria-valuenow": safePercent };
  return (
    <div
      className={`progress-track ${safePercent === undefined ? "indeterminate" : ""}`}
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      {...ariaProps}
    >
      <div className="progress-fill" style={{ width: safePercent === undefined ? undefined : `${safePercent}%` }} />
    </div>
  );
}

function formatPdfProgress(progress: PdfProgress): string {
  if (progress.phase === "docling") {
    if (progress.current <= 0) return "Loading Granite Docling WebGPU";
    const percent = typeof progress.percent === "number" ? ` (${progress.percent}%)` : "";
    return `Processing PDF page ${progress.current} of ${progress.total} with Granite Docling${percent}`;
  }
  if (progress.phase === "local-ocr") {
    const detail = progress.detail ? ` with ${progress.detail}` : "";
    return `Processing PDF${detail}`;
  }
  if (progress.phase === "ocr") {
    const percent = typeof progress.percent === "number" ? ` (${progress.percent}%)` : "";
    return `Processing PDF page ${progress.current} of ${progress.total}${percent}`;
  }
  const phase = progress.phase.toUpperCase();
  const detail = progress.detail ?? `${progress.current}/${progress.total}`;
  const percent = typeof progress.percent === "number" ? ` (${progress.percent}%)` : "";
  return `${phase} ${detail}${percent}`;
}

function formatSummaryProgress(progress: SummaryProgress): string {
  if (progress.current <= 0) {
    const detail = progress.detail ? ` - ${progress.detail}` : "";
    return `Preparing serial summary chunks${detail}`;
  }
  const current = formatCount(progress.current);
  const total = typeof progress.total === "number" ? ` of ${formatCount(progress.total)}` : "";
  const percent = typeof progress.percent === "number" ? ` (${progress.percent}%)` : "";
  const detail = progress.detail ? ` - ${progress.detail}` : "";
  return `Summarizing record segment ${current}${total}${percent}${detail}`;
}

function formatTrialProgress(progress: TrialProgress): string {
  const current = progress.unit === "bytes" ? formatBytes(progress.current) : formatCount(progress.current);
  const total = typeof progress.total === "number"
    ? ` of ${progress.unit === "bytes" ? formatBytes(progress.total) : formatCount(progress.total)}`
    : "";
  const percent = typeof progress.percent === "number" ? ` (${progress.percent}%)` : "";
  const detail = progress.detail ? ` - ${progress.detail}` : "";
  if (progress.phase === "download") return `Downloading trial records ${current}${total}${percent}${detail}`;
  if (progress.phase === "extract") return `Processing trial ${current}${total}${percent}${detail}`;
  if (progress.phase === "import") return `Loading embedded trial index${total ? ` ${current}${total}${percent}` : ""}${detail}`;
  return `Embedding trial space ${current}${total}${percent}${detail}`;
}

function trialProgressFromEmbeddedFetch(progress: EmbeddedTrialIndexFetchProgress): TrialProgress {
  return {
    phase: "import",
    current: progress.loadedBytes ?? 0,
    total: progress.totalBytes,
    percent: typeof progress.percent === "number" ? Math.round(progress.percent) : undefined,
    detail: progress.detail,
    unit: "bytes"
  };
}

function modelWarmupSteps(settings: ModelSettings): ModelWarmupStep[] {
  return [
    {
      label: settings.llmBackend === "browser-webgpu" ? "Browser LLM" : "Local LLM",
      modelId: settings.llmModelId,
      task: "text-generation",
      dtype: settings.llmDtype
    },
    {
      label: "TrialSpace model",
      modelId: settings.trialSpaceModelId,
      task: "feature-extraction",
      dtype: settings.classifierDtype
    },
    {
      label: "TrialChecker model",
      modelId: settings.trialCheckerModelId,
      task: "text-classification",
      dtype: settings.classifierDtype
    },
    {
      label: "BoilerplateChecker model",
      modelId: settings.boilerplateCheckerModelId,
      task: "text-classification",
      dtype: settings.classifierDtype
    }
  ];
}

function modelCacheStorageKey(settings: ModelSettings): string {
  const signature = JSON.stringify({
    llmBackend: settings.llmBackend,
    onnxBackend: settings.onnxBackend,
    browserLlm: settings.llmModelId,
    nativeLlm: `${settings.llamaModelRepo}/${settings.llamaModelFile}`,
    trialSpace: settings.trialSpaceModelId,
    trialChecker: settings.trialCheckerModelId,
    boilerplateChecker: settings.boilerplateCheckerModelId,
    llmDtype: settings.llmDtype,
    classifierDtype: settings.classifierDtype
  });
  return `matchminer-model-cache-ready:v2:${signature}`;
}

function modelCacheProgressFromWarmup(step: ModelWarmupStep, index: number, total: number, progress: WarmModelProgress): ModelCacheProgress {
  const implicitPercent = progress.status === "ready" ? 100 : undefined;
  const nestedPercent = firstFinite(progress.overallPercent, progress.progress, implicitPercent);
  const loadedBytes = firstFinite(progress.loadedBytes, progress.loaded);
  const totalBytes = firstFinite(progress.totalBytes, progress.loaded !== undefined ? progress.total : undefined);
  return {
    label: `Caching ${step.label}`,
    detail: modelCacheProgressDetail(progress),
    current: index + 1,
    total,
    percent: steppedPercent(index, nestedPercent, total),
    loadedBytes,
    totalBytes
  };
}

function modelCacheProgressDetail(progress: WarmModelProgress): string {
  const file = progress.file ? shortFileName(progress.file) : "";
  if (progress.status === "cached") return progress.detail || (file ? `${file} already cached` : "already cached");
  if (progress.status === "checking") return progress.detail || "checking cache";
  if (progress.status === "download") return file ? `starting ${file}` : "starting download";
  if (progress.status === "progress" || progress.status === "progress_total") return file ? `downloading ${file}` : "downloading model files";
  if (progress.status === "done") return progress.detail || (file ? `${file} cached` : "cached locally");
  if (progress.status === "ready") return "loading runtime";
  if (progress.status === "loading") return progress.detail || "loading runtime";
  return progress.detail || "warming local runtime";
}

function formatProgressBytes(progress: ModelCacheProgress): string {
  if (typeof progress.loadedBytes !== "number") return "";
  if (typeof progress.totalBytes === "number" && progress.totalBytes > 0) {
    return `${formatBytes(progress.loadedBytes)} of ${formatBytes(progress.totalBytes)}`;
  }
  return formatBytes(progress.loadedBytes);
}

function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value < 0) return "";
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size >= 10 || unit === 0 ? size.toFixed(0) : size.toFixed(1)} ${units[unit]}`;
}

function shortFileName(file: string): string {
  const parts = file.split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] ?? file;
}

function firstFinite(...values: Array<number | undefined>): number | undefined {
  return values.find((value) => typeof value === "number" && Number.isFinite(value));
}

function steppedPercent(index: number, nestedPercent: number | undefined, total: number): number | undefined {
  if (!total || total <= 0) return undefined;
  const safeNested = typeof nestedPercent === "number" && Number.isFinite(nestedPercent)
    ? Math.max(0, Math.min(100, nestedPercent))
    : 0;
  return Math.round(Math.max(0, Math.min(100, ((index + safeNested / 100) / total) * 100)));
}

function percentComplete(current: number, total: number | undefined): number | undefined {
  if (!total || total <= 0) return undefined;
  return Math.max(0, Math.min(100, Math.round((current / total) * 100)));
}

function formatCount(value: number): string {
  return value.toLocaleString();
}

function withThinkingSummaryHeadroom(settings: ModelSettings): ModelSettings {
  return {
    ...settings,
    maxSummaryTokens: settings.maxSummaryTokens <= LEGACY_SUMMARY_MAX_TOKENS ? THINKING_SUMMARY_MAX_TOKENS : settings.maxSummaryTokens
  };
}

function ResultCard({ match }: { match: MatchResult }) {
  const boiler = match.boilerplateScore;
  const ctGovUrl = match.trial.url || ctGovStudyUrl(match.trial.nctId);
  return (
    <article className="result-card">
      <div className="result-head">
        <span className="rank">#{match.rank}</span>
        <div>
          <h3>{match.trial.title}</h3>
          <p>{match.trial.nctId} {match.trial.overallStatus ? `- ${match.trial.overallStatus}` : ""}</p>
        </div>
      </div>
      <div className="score-row">
        <Score label="Match" value={match.trialCheckerScore} />
        <Score label="Retrieval" value={match.cosineSimilarity} />
        <Score label="Exclusion risk" value={boiler} invert />
        {typeof match.llmTrialCheckScore === "number" && <Score label="LLM" value={match.llmTrialCheckScore / 5} />}
      </div>
      <details>
        <summary>Trial space</summary>
        <p>{match.trial.trialSpaceText}</p>
      </details>
      <details>
        <summary>Boilerplate exclusions</summary>
        <p>{match.trial.boilerplateText || "None extracted."}</p>
      </details>
      {match.llmTrialCheckReasoning && (
        <details>
          <summary>Deep screen</summary>
          <p>{match.llmTrialCheckReasoning}</p>
          {match.llmBoilerplateReasoning && <p>{match.llmBoilerplateReasoning}</p>}
        </details>
      )}
      <div className="result-foot">
        <span>{match.trial.locations?.slice(0, 3).join(" | ")}</span>
        {ctGovUrl && (
          <a href={ctGovUrl} rel="noreferrer" target="_blank">
            CT.gov <ExternalLink size={14} />
          </a>
        )}
      </div>
      {match.warnings.map((warning) => (
        <div className="status warning" key={warning}>{warning}</div>
      ))}
    </article>
  );
}

function Score({ label, value, invert = false }: { label: string; value: number | null | undefined; invert?: boolean }) {
  const display = typeof value === "number" && Number.isFinite(value) ? value : null;
  const percent = display === null ? null : Math.max(0, Math.min(100, display * 100));
  const tone = percent === null ? "unknown" : invert ? (percent > 50 ? "bad" : "good") : percent > 50 ? "good" : "mixed";
  return (
    <div className={`score ${tone}`}>
      <span>{label}</span>
      <strong>{percent === null ? "n/a" : `${percent.toFixed(0)}%`}</strong>
    </div>
  );
}

interface SettingsDialogProps {
  settings: ModelSettings;
  prompts: Record<PromptKey, string>;
  activePrompt: PromptKey;
  status: StatusMessage[];
  busy: string | null;
  webGpu: boolean | null;
  summaryProgress: SummaryProgress | null;
  modelCacheProgress: ModelCacheProgress | null;
  onChange: (settings: ModelSettings) => void;
  onClose: () => void;
  onActivePromptChange: (key: PromptKey) => void;
  onPromptChange: (key: PromptKey, value: string) => void;
  onPromptReset: (key: PromptKey) => void;
  onCacheModels: () => void;
}

function SettingsDialog({
  settings,
  prompts,
  activePrompt,
  status,
  busy,
  webGpu,
  summaryProgress,
  modelCacheProgress,
  onChange,
  onClose,
  onActivePromptChange,
  onPromptChange,
  onPromptReset,
  onCacheModels
}: SettingsDialogProps) {
  const [activeTab, setActiveTab] = useState<"general" | "advanced">("general");
  const busyNow = Boolean(busy);
  const progressLabel = summaryProgress ? formatSummaryProgress(summaryProgress) : modelCacheProgress ? null : busy;
  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true">
      <div className="modal">
        <div className="modal-head">
          <h2>Settings</h2>
          <button className="icon-button" onClick={onClose} title="Close" type="button">
            <X size={18} />
          </button>
        </div>
        <div className="tabs modal-tabs" role="tablist">
          <button className={activeTab === "general" ? "tab active" : "tab"} onClick={() => setActiveTab("general")} type="button">
            General
          </button>
          <button className={activeTab === "advanced" ? "tab active" : "tab"} onClick={() => setActiveTab("advanced")} type="button">
            Advanced
          </button>
        </div>
        {activeTab === "general" ? (
          <div className="settings-section">
            <div className="settings-subsection">
              <h3>Runtime</h3>
              <div className="readiness-grid">
                <Readiness label="WebGPU" value={webGpu === null ? "checking" : webGpu ? "ready" : "unavailable"} />
                <Readiness label="Active job" value={busy ?? "none"} />
                <Readiness label="Deep screen" value={settings.runDeepScreen ? "on" : "off"} />
              </div>
            </div>
            <label>
              PDF OCR
              <select value={settings.pdfOcrMode} onChange={(event) => onChange({ ...settings, pdfOcrMode: event.target.value as ModelSettings["pdfOcrMode"] })}>
                <option value="auto">Auto: Granite WebGPU, then browser fallback</option>
                <option value="granite">Granite WebGPU only</option>
                <option value="browser">Browser only: Tesseract.js</option>
                <option value="local">Local CLI only: Docling/OCRmyPDF</option>
              </select>
            </label>
            <div className="settings-grid">
              <label>
                Retrieval count
                <input type="number" min={1} max={200} value={settings.retrievalCount} onChange={(event) => onChange({ ...settings, retrievalCount: Number(event.target.value) })} />
              </label>
              <label>
                Display count
                <input type="number" min={1} max={50} value={settings.displayCount} onChange={(event) => onChange({ ...settings, displayCount: Number(event.target.value) })} />
              </label>
              <label className="checkbox-label">
                <input type="checkbox" checked={settings.runDeepScreen} onChange={(event) => onChange({ ...settings, runDeepScreen: event.target.checked })} />
                Deep screen
              </label>
            </div>
          </div>
        ) : (
          <div className="settings-section advanced-settings">
            <div className="settings-actions">
              <button className="text-button" disabled={busyNow} onClick={onCacheModels} type="button">
                <Download size={16} /> Cache models
              </button>
              {modelCacheProgress && <ModelCacheProgressView progress={modelCacheProgress} compact />}
              {progressLabel && <ProgressLine label={progressLabel} />}
            </div>
            <div className="settings-grid">
              <label>
                LLM backend
                <select value={settings.llmBackend} onChange={(event) => onChange({ ...settings, llmBackend: event.target.value as ModelSettings["llmBackend"] })}>
                  <option value="llama.cpp">Native llama.cpp</option>
                  <option value="browser-webgpu">Browser WebGPU</option>
                </select>
              </label>
              <label>
                ONNX backend
                <select value={settings.onnxBackend} onChange={(event) => onChange({ ...settings, onnxBackend: event.target.value as ModelSettings["onnxBackend"] })}>
                  <option value="native-onnx">Native ONNX Runtime</option>
                  <option value="browser-webgpu">Browser WebGPU</option>
                </select>
              </label>
              <label>
                LLM dtype
                <select value={settings.llmDtype} onChange={(event) => onChange({ ...settings, llmDtype: event.target.value as ModelSettings["llmDtype"] })}>
                  <option value="q4">q4</option>
                  <option value="q4f16">q4f16</option>
                  <option value="fp16">fp16</option>
                  <option value="auto">auto</option>
                </select>
              </label>
              <label>
                Classifier dtype
                <select value={settings.classifierDtype} onChange={(event) => onChange({ ...settings, classifierDtype: event.target.value as ModelSettings["classifierDtype"] })}>
                  <option value="auto">auto</option>
                  <option value="q8">q8</option>
                  <option value="fp16">fp16</option>
                </select>
              </label>
              <label>
                LLM context tokens
                <input type="number" min={4096} max={131072} step={1000} value={settings.llmContextTokens} onChange={(event) => onChange({ ...settings, llmContextTokens: Number(event.target.value) })} />
              </label>
              <label>
                Summary tokens
                <input type="number" min={100} max={15000} value={settings.maxSummaryTokens} onChange={(event) => onChange({ ...settings, maxSummaryTokens: Number(event.target.value) })} />
              </label>
              <label>
                Summary target chunk tokens
                <input type="number" min={256} max={120000} step={1024} value={settings.summaryChunkTokens} onChange={(event) => onChange({ ...settings, summaryChunkTokens: Number(event.target.value) })} />
              </label>
              <label>
                Summary overlap tokens
                <input type="number" min={0} max={5000} value={settings.summaryChunkOverlapTokens} onChange={(event) => onChange({ ...settings, summaryChunkOverlapTokens: Number(event.target.value) })} />
              </label>
            </div>
            <div className="settings-stack">
              <label>
                Browser LLM model
                <input value={settings.llmModelId} onChange={(event) => onChange({ ...settings, llmModelId: event.target.value })} />
              </label>
              <label>
                llama.cpp GGUF repo
                <input value={settings.llamaModelRepo} onChange={(event) => onChange({ ...settings, llamaModelRepo: event.target.value })} />
              </label>
              <label>
                llama.cpp GGUF file
                <input value={settings.llamaModelFile} onChange={(event) => onChange({ ...settings, llamaModelFile: event.target.value })} />
              </label>
              <label>
                TrialSpace model
                <input value={settings.trialSpaceModelId} onChange={(event) => onChange({ ...settings, trialSpaceModelId: event.target.value })} />
              </label>
              <label>
                TrialChecker model
                <input value={settings.trialCheckerModelId} onChange={(event) => onChange({ ...settings, trialCheckerModelId: event.target.value })} />
              </label>
              <label>
                BoilerplateChecker model
                <input value={settings.boilerplateCheckerModelId} onChange={(event) => onChange({ ...settings, boilerplateCheckerModelId: event.target.value })} />
              </label>
            </div>
            <div className="advanced-block">
              <div className="tabs">
                {promptOrder.map((key) => (
                  <button key={key} className={activePrompt === key ? "tab active" : "tab"} onClick={() => onActivePromptChange(key)} type="button">
                    {DEFAULT_PROMPTS[key].label}
                  </button>
                ))}
              </div>
              <textarea
                className="prompt-box"
                value={prompts[activePrompt]}
                onChange={(event) => onPromptChange(activePrompt, event.target.value)}
              />
              <div className="compact-row end">
                <span className="muted">{DEFAULT_PROMPTS[activePrompt].variables.map((v) => `{${v}}`).join(" ")}</span>
                <button className="text-button" onClick={() => onPromptReset(activePrompt)} type="button">
                  <RefreshCcw size={15} /> Reset
                </button>
              </div>
            </div>
            <div className="advanced-block">
              <div className="status-list compact-status-list">
                {status.length ? status.map((item, index) => (
                  <div className={`status ${item.kind}`} key={`${item.text}-${index}`}>
                    {item.text}
                  </div>
                )) : <div className="status">No run log entries.</div>}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function trialDocumentForExtraction(record: TrialSpaceRecord): string {
  return [
    record.title,
    record.overallStatus ? `Status: ${record.overallStatus}` : "",
    record.phases?.length ? `Phase: ${record.phases.join(", ")}` : "",
    record.conditions?.length ? `Conditions: ${record.conditions.join(", ")}` : "",
    record.trialSpaceText
  ].filter(Boolean).join("\n\n");
}

function parseExtractedTrialSpaces(record: TrialSpaceRecord, response: string): TrialSpaceRecord[] {
  const boilerplateText = extractTrialBoilerplate(response) || record.boilerplateText;
  const spaceText = response.split(/^\s*Boilerplate exclusions\s*:/im)[0] ?? response;
  const matches = Array.from(spaceText.matchAll(/(?:^|\n)\s*\d+[\.)]\s*(Age range allowed:.*?)(?=\n\s*\d+[\.)]|\n\s*Boilerplate exclusions\s*:|$)/gis));
  return matches.map((match, index) => ({
    ...record,
    spaceId: `${record.nctId}-llm-space-${index + 1}`,
    trialSpaceText: normalizeExtractedSpace(match[1]),
    boilerplateText,
    embedding: undefined
  }));
}

function extractTrialBoilerplate(response: string): string {
  const marker = response.match(/^\s*Boilerplate exclusions\s*:/im);
  if (!marker || marker.index === undefined) return "";
  return response
    .slice(marker.index + marker[0].length)
    .split(/\r?\n/)
    .map((line) => line.replace(/^\s*[-*]\s*/, "").trim())
    .filter(Boolean)
    .join("\n");
}

function normalizeExtractedSpace(text: string): string {
  return text
    .replace(/\s+/g, " ")
    .replace(/^\s*[-*]\s*/, "")
    .trim();
}

function parseFinalScore(text: string): number | null {
  const match = text.match(/final\s+score\s*:\s*(\d)/i);
  return match ? Math.min(5, Number(match[1])) : null;
}

function parseYesNo(text: string): boolean | null {
  const tail = text.trim().slice(-20).toLowerCase();
  if (tail.includes("yes!")) return true;
  if (tail.includes("no!")) return false;
  return null;
}

function buildSummaryPrompt(template: string, prior: string, chunk: SerialSummaryChunk): string {
  return fillPrompt(template, {
    prior_summary: prior || "None - this is the first segment for this patient",
    first_date: chunk.firstDate,
    last_date: chunk.lastDate,
    record_segment: chunk.text
  });
}

function summaryPromptTokenBudget(maxSummaryTokens: number, llmContextTokens = DEFAULT_BROWSER_LLM_CONTEXT_TOKENS): number {
  const contextTokens = normalizeLlmContextTokens(llmContextTokens);
  return Math.max(
    MIN_ADAPTIVE_SUMMARY_CHUNK_TOKENS,
    contextTokens - Math.max(0, Math.floor(maxSummaryTokens)) - BROWSER_LLM_CONTEXT_MARGIN_TOKENS
  );
}

function normalizeLlmContextTokens(value: number): number {
  return Number.isFinite(value) ? Math.max(MIN_BROWSER_LLM_CONTEXT_TOKENS, Math.floor(value)) : DEFAULT_BROWSER_LLM_CONTEXT_TOKENS;
}

function normalizeSummaryMaxNewTokens(value: number): number {
  return Number.isFinite(value) ? Math.max(100, Math.floor(value)) : THINKING_SUMMARY_MAX_TOKENS;
}

function normalizeSummaryChunkTokens(value: number): number {
  return Number.isFinite(value) ? Math.max(MIN_SUMMARY_RETRY_CHUNK_TOKENS, Math.floor(value)) : DEFAULT_MODEL_SETTINGS.summaryChunkTokens;
}

function nextSmallerSummaryRetryChunkSize(value: number): number {
  const current = normalizeSummaryChunkTokens(value);
  if (current <= MIN_SUMMARY_RETRY_CHUNK_TOKENS) return current;
  return Math.max(MIN_SUMMARY_RETRY_CHUNK_TOKENS, Math.floor(current / 2));
}

function nextLowerLlmContextTokens(value: number): number {
  const current = normalizeLlmContextTokens(value);
  if (current <= MIN_BROWSER_LLM_CONTEXT_TOKENS) return current;
  const halved = Math.floor(current / 2 / 1024) * 1024;
  return Math.max(MIN_BROWSER_LLM_CONTEXT_TOKENS, halved);
}

function smallerSummaryChunkSize(chunkTokens: number, promptTokens: number, promptBudget: number): number {
  const promptOverhead = Math.max(0, promptTokens - chunkTokens);
  const sourceBudget = promptBudget - promptOverhead - BROWSER_LLM_CONTEXT_MARGIN_TOKENS;
  const target = sourceBudget > 0 ? Math.min(Math.floor(sourceBudget), Math.ceil(chunkTokens / 2)) : Math.ceil(chunkTokens / 2);
  return Math.max(MIN_ADAPTIVE_SUMMARY_CHUNK_TOKENS, target);
}

function isOversizedGenerationError(error: unknown): boolean {
  const message = errorMessage(error).toLowerCase();
  return (
    message.includes("tensor shape is too large") ||
    message.includes("integer overflow") ||
    message.includes("safeintonoverflow") ||
    message.includes("failed to allocate memory") ||
    message.includes("failed to download data from buffer") ||
    message.includes("invalid buffer") ||
    message.includes("mapasync") ||
    message.includes("webgpu validation failed") ||
    message.includes("bind group layout") ||
    message.includes("createbindgroup") ||
    message.includes("binding index") ||
    message.includes("attentionprobs") ||
    message.includes("out of memory") ||
    message.includes("oom") ||
    (message.includes("ortrun") && (message.includes("invalid_argument") || message.includes("error_code: 1")))
  );
}

function isRetryableSummaryGenerationError(error: unknown): boolean {
  if (isOversizedGenerationError(error)) return true;
  const message = errorMessage(error).toLowerCase();
  const contextSized =
    message.includes("context") &&
    (message.includes("length") || message.includes("token") || message.includes("exceed") || message.includes("window"));
  return (
    contextSized ||
    message.includes("no visible patient summary text") ||
    message.includes("maximum sequence length") ||
    message.includes("prompt is too long") ||
    message.includes("input is too long") ||
    message.includes("truncat")
  );
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
