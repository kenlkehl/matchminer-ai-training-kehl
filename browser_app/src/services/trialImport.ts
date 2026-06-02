import Papa from "papaparse";
import type { TrialSpaceRecord } from "../types";
import { assertNotAborted } from "../lib/abort";

type RawRecord = Record<string, unknown>;

const RECORD_ARRAY_KEYS = ["records", "trialSpaces", "trial_spaces", "data"];

export async function parseEmbeddedTrialIndexFile(file: File, signal?: AbortSignal): Promise<TrialSpaceRecord[]> {
  assertNotAborted(signal);
  if (/\.parquet$/i.test(file.name)) {
    throw new Error("Parquet files cannot be imported directly in the browser. Export JSON, JSONL, or CSV from the pre-embed script.");
  }
  const text = await file.text();
  assertNotAborted(signal);
  return parseEmbeddedTrialIndexText(text, file.name);
}

export async function fetchEmbeddedTrialIndex(url: string, signal?: AbortSignal): Promise<TrialSpaceRecord[]> {
  const trimmed = url.trim();
  if (!trimmed) throw new Error("Enter a URL for the embedded trial index");
  if (/\.parquet(?:$|[?#])/i.test(trimmed)) {
    throw new Error("Parquet URLs cannot be imported directly in the browser. Publish JSON, JSONL, or CSV instead.");
  }
  assertNotAborted(signal);
  const response = await fetch(trimmed, { signal });
  if (!response.ok) throw new Error(`Could not load embedded trial index: ${response.status} ${response.statusText}`);
  const text = await response.text();
  assertNotAborted(signal);
  return parseEmbeddedTrialIndexText(text, trimmed);
}

export function parseEmbeddedTrialIndexText(text: string, sourceName = "embedded trial index"): TrialSpaceRecord[] {
  const trimmed = text.trim();
  if (!trimmed) throw new Error(`${sourceName} is empty`);
  const rows = parseRows(trimmed, sourceName);
  const records = rows.map((row, index) => normalizeTrialRecord(row, index));
  return ensureUniqueSpaceIds(records);
}

function parseRows(text: string, sourceName: string): RawRecord[] {
  const lower = sourceName.toLowerCase();
  if (lower.endsWith(".csv")) return parseCsvRows(text, sourceName);
  if (lower.endsWith(".jsonl") || lower.endsWith(".ndjson")) return parseJsonLines(text, sourceName);
  if (looksLikeJson(text)) return parseJsonRows(text, sourceName);
  return parseCsvRows(text, sourceName);
}

function parseJsonRows(text: string, sourceName: string): RawRecord[] {
  let payload: unknown;
  try {
    payload = JSON.parse(text) as unknown;
  } catch (error) {
    if (text.includes("\n")) return parseJsonLines(text, sourceName);
    throw error;
  }
  const rows = Array.isArray(payload) ? payload : findRecordArray(payload);
  if (!rows) {
    if (isRecord(payload)) {
      throw new Error(`${sourceName} looks like metadata, not embedded trial records`);
    }
    throw new Error(`${sourceName} must contain an array of embedded trial records`);
  }
  return rows.map((row) => {
    if (!isRecord(row)) throw new Error(`${sourceName} contains a non-object trial record`);
    return row;
  });
}

function parseJsonLines(text: string, sourceName: string): RawRecord[] {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      const row = JSON.parse(line) as unknown;
      if (!isRecord(row)) throw new Error(`${sourceName} line ${index + 1} is not an object`);
      return row;
    });
}

function parseCsvRows(text: string, sourceName: string): RawRecord[] {
  const parsed = Papa.parse<RawRecord>(text, {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: false
  });
  if (parsed.errors.length) {
    const first = parsed.errors[0];
    throw new Error(`CSV parse error in ${sourceName} on row ${first.row ?? "unknown"}: ${first.message}`);
  }
  return parsed.data;
}

function normalizeTrialRecord(row: RawRecord, index: number): TrialSpaceRecord {
  const nctId = readString(row, ["nctId", "nct_id", "nct"]);
  const trialSpaceText = readString(row, ["trialSpaceText", "trial_space_text", "this_cohort", "this_space", "cohort", "text"]);
  const embedding = parseEmbedding(row.embedding);
  if (!nctId) throw new Error(`Embedded trial record ${index + 1} is missing nct_id`);
  if (!trialSpaceText) throw new Error(`Embedded trial record ${index + 1} is missing trial-space text`);
  if (!embedding.length) throw new Error(`Embedded trial record ${index + 1} is missing a numeric embedding`);

  const spaceId = readString(row, ["spaceId", "space_id", "id"]) || `${nctId}-space-${index + 1}`;
  const title = readString(row, ["title", "briefTitle", "brief_title", "officialTitle", "official_title"]) || nctId;
  const url = readString(row, ["url", "trialUrl", "trial_url"]) || `https://clinicaltrials.gov/study/${nctId}`;
  return {
    spaceId,
    nctId,
    title,
    overallStatus: readString(row, ["overallStatus", "overall_status"]),
    conditions: readStringArray(row.conditions),
    phases: readStringArray(row.phases),
    locations: readStringArray(row.locations),
    url,
    trialSpaceText,
    boilerplateText: readString(row, ["boilerplateText", "boilerplate_text", "trial_boilerplate"]) || "",
    embedding
  };
}

function parseEmbedding(value: unknown): number[] {
  if (Array.isArray(value)) return cleanEmbedding(value);
  if (typeof value !== "string") return [];
  const trimmed = value.trim();
  if (!trimmed) return [];
  if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
    try {
      const parsed = JSON.parse(trimmed) as unknown;
      if (Array.isArray(parsed)) return cleanEmbedding(parsed);
    } catch {
      // Fall through to delimiter parsing.
    }
  }
  return cleanEmbedding(trimmed.replace(/^\[|\]$/g, "").split(/[,\s]+/));
}

function cleanEmbedding(values: unknown[]): number[] {
  const embedding = values.map((value) => Number(value)).filter((value) => Number.isFinite(value));
  return embedding.length === values.length ? embedding : [];
}

function readString(row: RawRecord, keys: string[]): string {
  for (const key of keys) {
    const value = row[key];
    if (typeof value === "string" && value.trim()) return value.trim();
    if (typeof value === "number" && Number.isFinite(value)) return String(value);
  }
  return "";
}

function readStringArray(value: unknown): string[] | undefined {
  if (Array.isArray(value)) return value.map((item) => String(item).trim()).filter(Boolean);
  if (typeof value !== "string" || !value.trim()) return undefined;
  const trimmed = value.trim();
  if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
    try {
      const parsed = JSON.parse(trimmed) as unknown;
      if (Array.isArray(parsed)) return parsed.map((item) => String(item).trim()).filter(Boolean);
    } catch {
      return undefined;
    }
  }
  return trimmed.split("|").map((item) => item.trim()).filter(Boolean);
}

function findRecordArray(payload: unknown): unknown[] | null {
  if (!isRecord(payload)) return null;
  for (const key of RECORD_ARRAY_KEYS) {
    const value = payload[key];
    if (Array.isArray(value)) return value;
  }
  return null;
}

function ensureUniqueSpaceIds(records: TrialSpaceRecord[]): TrialSpaceRecord[] {
  const seen = new Set<string>();
  return records.map((record, index) => {
    if (!seen.has(record.spaceId)) {
      seen.add(record.spaceId);
      return record;
    }
    const uniqueId = `${record.spaceId}-${index + 1}`;
    seen.add(uniqueId);
    return { ...record, spaceId: uniqueId };
  });
}

function looksLikeJson(text: string): boolean {
  return text.startsWith("[") || text.startsWith("{");
}

function isRecord(value: unknown): value is RawRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
