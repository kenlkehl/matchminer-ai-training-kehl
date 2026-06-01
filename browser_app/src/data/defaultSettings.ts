import type { ModelSettings } from "../types";

export const MODEL_QUERY_PROMPT =
  "Instruct: Given a cancer patient summary, retrieve clinical trial options that are reasonable for that patient; or, given a clinical trial option, retrieve cancer patients who are reasonable candidates for that trial. ";

export const LEGACY_BROWSER_LLM_MODEL_ID = "onnx-community/gemma-4-E2B-it-ONNX";
export const DEFAULT_BROWSER_LLM_MODEL_ID = "LiquidAI/LFM2.5-1.2B-Thinking-ONNX";

export const DEFAULT_MODEL_SETTINGS: ModelSettings = {
  llmModelId: DEFAULT_BROWSER_LLM_MODEL_ID,
  trialSpaceModelId: "ksg-dfci/TrialSpace-0526-ONNX",
  trialCheckerModelId: "ksg-dfci/TrialChecker-0526-ONNX",
  boilerplateCheckerModelId: "ksg-dfci/BoilerplateChecker-0526-ONNX",
  pdfOcrMode: "auto",
  llmDtype: "q4",
  classifierDtype: "auto",
  llmContextTokens: 32768,
  maxSummaryTokens: 1600,
  summaryChunkTokens: 24000,
  summaryChunkOverlapTokens: 250,
  retrievalCount: 30,
  displayCount: 10,
  runDeepScreen: false
};

export const SOURCE_MODEL_IDS = {
  trialSpace: "ksg-dfci/TrialSpace-0526",
  trialChecker: "ksg-dfci/TrialChecker-0526",
  boilerplateChecker: "ksg-dfci/BoilerplateChecker-0526"
};
