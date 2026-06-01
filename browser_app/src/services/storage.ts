import Dexie, { type Table } from "dexie";
import type { ModelSettings, PromptKey, TrialSpaceRecord } from "../types";
import {
  DEFAULT_BROWSER_LLM_MODEL_ID,
  DEFAULT_LFM_CONTEXT_TOKENS,
  DEFAULT_LLAMA_GGUF_FILE,
  DEFAULT_LLAMA_GGUF_REPO,
  DEFAULT_MODEL_SETTINGS,
  LEGACY_BROWSER_LLM_MODEL_ID
} from "../data/defaultSettings";
import { getDefaultPromptValues } from "../data/defaultPrompts";

const LEGACY_LFM_CONTEXT_TOKENS = 32768;
const LEGACY_ELECTRON_LFM_CONTEXT_TOKENS = 65536;

interface PromptRow {
  key: PromptKey;
  value: string;
  updatedAt: string;
}

interface SettingRow {
  key: "modelSettings";
  value: ModelSettings;
  updatedAt: string;
}

interface TrialRow extends TrialSpaceRecord {
  key: string;
}

class MatchMinerDb extends Dexie {
  prompts!: Table<PromptRow, PromptKey>;
  settings!: Table<SettingRow, string>;
  trials!: Table<TrialRow, string>;

  constructor() {
    super("matchminer-local-browser");
    this.version(1).stores({
      prompts: "key",
      settings: "key",
      trials: "key,nctId"
    });
  }
}

export const db = new MatchMinerDb();

export async function loadPrompts(): Promise<Record<PromptKey, string>> {
  const defaults = getDefaultPromptValues();
  const rows = await db.prompts.toArray();
  for (const row of rows) defaults[row.key] = row.value;
  return defaults;
}

export async function savePrompt(key: PromptKey, value: string): Promise<void> {
  await db.prompts.put({ key, value, updatedAt: new Date().toISOString() });
}

export async function resetPrompt(key: PromptKey): Promise<Record<PromptKey, string>> {
  await db.prompts.delete(key);
  return loadPrompts();
}

export async function loadModelSettings(): Promise<ModelSettings> {
  const row = await db.settings.get("modelSettings");
  const stored: Partial<ModelSettings> = row?.value ?? {};
  const settings = { ...DEFAULT_MODEL_SETTINGS, ...stored };
  let changed = false;
  if (stored.llmModelId === LEGACY_BROWSER_LLM_MODEL_ID) {
    settings.llmModelId = DEFAULT_BROWSER_LLM_MODEL_ID;
    settings.llmContextTokens = DEFAULT_MODEL_SETTINGS.llmContextTokens;
    settings.maxSummaryTokens = DEFAULT_MODEL_SETTINGS.maxSummaryTokens;
    settings.summaryChunkTokens = DEFAULT_MODEL_SETTINGS.summaryChunkTokens;
    changed = true;
  }
  if (shouldMigrateOldLfmContextDefault(stored, settings)) {
    settings.llmContextTokens = DEFAULT_LFM_CONTEXT_TOKENS;
    changed = true;
  }
  if (row && changed) {
    await db.settings.put({ key: "modelSettings", value: settings, updatedAt: new Date().toISOString() });
  }
  return settings;
}

function shouldMigrateOldLfmContextDefault(stored: Partial<ModelSettings>, settings: ModelSettings): boolean {
  if (stored.llmContextTokens === LEGACY_LFM_CONTEXT_TOKENS) return true;
  return stored.llmContextTokens === LEGACY_ELECTRON_LFM_CONTEXT_TOKENS && usesDefaultLfmConfig(settings);
}

function usesDefaultLfmConfig(settings: ModelSettings): boolean {
  return (
    settings.llmModelId === DEFAULT_BROWSER_LLM_MODEL_ID &&
    settings.llamaModelRepo === DEFAULT_LLAMA_GGUF_REPO &&
    settings.llamaModelFile === DEFAULT_LLAMA_GGUF_FILE
  );
}

export async function saveModelSettings(value: ModelSettings): Promise<void> {
  await db.settings.put({ key: "modelSettings", value, updatedAt: new Date().toISOString() });
}

export async function saveTrialIndex(records: TrialSpaceRecord[]): Promise<void> {
  await db.transaction("rw", db.trials, async () => {
    await db.trials.clear();
    await db.trials.bulkPut(records.map((record) => ({ ...record, key: record.spaceId })));
  });
}

export async function loadTrialIndex(): Promise<TrialSpaceRecord[]> {
  const rows = await db.trials.toArray();
  return rows.map(({ key: _key, ...record }) => record);
}

export async function clearPatientSideData(): Promise<void> {
  sessionStorage.removeItem("matchminer-current-patient");
}
