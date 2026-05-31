export type SourceKind = "csv" | "pdf";
export type PipelineStage = "idle" | "running" | "complete" | "error";

export interface ClinicalNote {
  id: string;
  dateInput: string;
  isoDate: string;
  epochMs: number;
  text: string;
}

export interface PatientDocument {
  id: string;
  source: SourceKind;
  fileName: string;
  notes: ClinicalNote[];
  rawText: string;
  createdAt: string;
}

export type PromptKey =
  | "patientSummary"
  | "trialSpaceExtraction"
  | "trialDeepScreen"
  | "boilerplateDeepScreen";

export interface PromptTemplate {
  key: PromptKey;
  label: string;
  description: string;
  variables: string[];
  defaultValue: string;
}

export interface ModelSettings {
  llmModelId: string;
  trialSpaceModelId: string;
  trialCheckerModelId: string;
  boilerplateCheckerModelId: string;
  llmDtype: "q4" | "q4f16" | "fp16" | "auto";
  classifierDtype: "q8" | "fp16" | "auto";
  maxSummaryTokens: number;
  retrievalCount: number;
  displayCount: number;
  runDeepScreen: boolean;
}

export interface TrialSpaceRecord {
  spaceId: string;
  nctId: string;
  title: string;
  overallStatus?: string;
  conditions?: string[];
  phases?: string[];
  locations?: string[];
  url?: string;
  trialSpaceText: string;
  boilerplateText: string;
  embedding?: number[];
}

export interface MatchResult {
  id: string;
  trial: TrialSpaceRecord;
  cosineSimilarity: number;
  trialCheckerScore: number | null;
  boilerplateScore: number | null;
  llmTrialCheckScore?: number | null;
  llmTrialCheckReasoning?: string;
  llmBoilerplateExcluded?: boolean | null;
  llmBoilerplateReasoning?: string;
  rank: number;
  warnings: string[];
}

export interface TrialIndexManifest {
  createdAt: string;
  source: string;
  ctgovApiVersion?: string;
  ctgovDataTimestamp?: string;
  embeddingModel: string;
  embeddingDim: number;
  trialSpaces: number;
  indexUrl: string;
}

export interface StatusMessage {
  kind: "info" | "success" | "warning" | "error";
  text: string;
}
