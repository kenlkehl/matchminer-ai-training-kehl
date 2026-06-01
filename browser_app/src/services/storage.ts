import Dexie, { type Table } from "dexie";
import type { ModelSettings, PromptKey, TrialSpaceRecord } from "../types";
import { DEFAULT_BROWSER_LLM_MODEL_ID, DEFAULT_MODEL_SETTINGS, LEGACY_BROWSER_LLM_MODEL_ID } from "../data/defaultSettings";
import { getDefaultPromptValues } from "../data/defaultPrompts";

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
  if (stored.llmModelId === LEGACY_BROWSER_LLM_MODEL_ID) {
    settings.llmModelId = DEFAULT_BROWSER_LLM_MODEL_ID;
    settings.llmContextTokens = DEFAULT_MODEL_SETTINGS.llmContextTokens;
    settings.maxSummaryTokens = DEFAULT_MODEL_SETTINGS.maxSummaryTokens;
    settings.summaryChunkTokens = DEFAULT_MODEL_SETTINGS.summaryChunkTokens;
  }
  return settings;
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
