import type { TrialIndexManifest, TrialSpaceRecord } from "../types";
import { assertNotAborted } from "../lib/abort";
import { embedText } from "./modelRuntime";
import { loadTrialIndex, saveTrialIndex } from "./storage";
import { fetchEmbeddedTrialIndex, type EmbeddedTrialIndexFetchProgress } from "./trialImport";

interface CtGovStudy {
  protocolSection?: {
    identificationModule?: { nctId?: string; briefTitle?: string; officialTitle?: string };
    statusModule?: { overallStatus?: string };
    descriptionModule?: { briefSummary?: string; detailedDescription?: string };
    conditionsModule?: { conditions?: string[] };
    designModule?: { phases?: string[]; studyType?: string };
    eligibilityModule?: { eligibilityCriteria?: string; sex?: string; minimumAge?: string; maximumAge?: string };
    contactsLocationsModule?: { locations?: Array<{ city?: string; state?: string; country?: string; status?: string }> };
  };
}

const CTGOV_CANCER_QUERY = "cancer OR lymphoma OR carcinoma OR leukemia OR sarcoma OR melanoma OR myeloma OR myelodysplastic OR myeloproliferative";
const CTGOV_PHASE_I_TO_III_OPEN_INTERVENTIONAL_FILTERS = "phase:0 1 2 3,status:not rec,studyType:int";
export const DEFAULT_EMBEDDED_TRIAL_INDEX_URL = "https://huggingface.co/datasets/ksg-dfci/mmai-synthetic-0526/blob/main/trial_space_embeddings_6-2-26.parquet";

export async function loadManifest(signal?: AbortSignal): Promise<TrialIndexManifest> {
  assertNotAborted(signal);
  const response = await fetch("/trial_index.manifest.json", { signal });
  if (!response.ok) throw new Error("Could not load trial index manifest");
  assertNotAborted(signal);
  return response.json();
}

export async function loadOrFetchTrialIndex(
  signal?: AbortSignal,
  onProgress?: (progress: EmbeddedTrialIndexFetchProgress) => void
): Promise<TrialSpaceRecord[]> {
  assertNotAborted(signal);
  const cached = await loadTrialIndex();
  if (cached.length) return cached;
  const manifest = await loadManifest(signal);
  const records = await fetchEmbeddedTrialIndex(manifest.indexUrl, signal, onProgress);
  assertNotAborted(signal);
  await saveTrialIndex(records);
  return records;
}

export async function ensureTrialEmbeddings(records: TrialSpaceRecord[], modelId: string, onProgress?: (done: number, total: number) => void, signal?: AbortSignal): Promise<TrialSpaceRecord[]> {
  const updated: TrialSpaceRecord[] = [];
  for (let index = 0; index < records.length; index += 1) {
    assertNotAborted(signal);
    const record = records[index];
    if (record.embedding?.length) {
      updated.push(record);
    } else {
      const embedding = await embedText(modelId, record.trialSpaceText, undefined, signal);
      updated.push({ ...record, embedding });
    }
    assertNotAborted(signal);
    onProgress?.(index + 1, records.length);
  }
  assertNotAborted(signal);
  await saveTrialIndex(updated);
  return updated;
}

export async function fetchCtGovCancerTrials(options: {
  pageSize?: number;
  maxPages?: number | null;
  signal?: AbortSignal;
  onProgress?: (progress: { records: number; totalCount?: number; page: number }) => void;
} = {}): Promise<TrialSpaceRecord[]> {
  const pageSize = options.pageSize ?? 1000;
  const maxPages = options.maxPages ?? null;
  let pageToken = "";
  const records: TrialSpaceRecord[] = [];
  const seenPageTokens = new Set<string>();

  for (let page = 0; maxPages === null || page < maxPages; page += 1) {
    assertNotAborted(options.signal);
    const params = new URLSearchParams({
      "query.cond": CTGOV_CANCER_QUERY,
      aggFilters: CTGOV_PHASE_I_TO_III_OPEN_INTERVENTIONAL_FILTERS,
      countTotal: "true",
      pageSize: String(pageSize),
      format: "json"
    });
    if (pageToken) params.set("pageToken", pageToken);
    const response = await fetch(`https://clinicaltrials.gov/api/v2/studies?${params.toString()}`, { signal: options.signal });
    if (!response.ok) throw new Error(`ClinicalTrials.gov request failed: ${response.status}`);
    assertNotAborted(options.signal);
    const payload = (await response.json()) as { studies?: CtGovStudy[]; nextPageToken?: string; totalCount?: number };
    for (const study of payload.studies ?? []) {
      assertNotAborted(options.signal);
      const record = ctGovStudyToTrialSpace(study);
      if (record) records.push(record);
    }
    assertNotAborted(options.signal);
    options.onProgress?.({ records: records.length, totalCount: payload.totalCount, page: page + 1 });
    if (!payload.nextPageToken) break;
    if (seenPageTokens.has(payload.nextPageToken)) {
      throw new Error("ClinicalTrials.gov returned a repeated page token");
    }
    seenPageTokens.add(payload.nextPageToken);
    pageToken = payload.nextPageToken;
  }
  return records;
}

export function ctGovStudyToTrialSpace(study: CtGovStudy): TrialSpaceRecord | null {
  const protocol = study.protocolSection;
  const id = protocol?.identificationModule?.nctId;
  if (!id) return null;
  const title = protocol?.identificationModule?.briefTitle || protocol?.identificationModule?.officialTitle || id;
  const summary = protocol?.descriptionModule?.briefSummary ?? "";
  const eligibility = protocol?.eligibilityModule?.eligibilityCriteria ?? "";
  const sex = protocol?.eligibilityModule?.sex ?? "ALL";
  const minAge = protocol?.eligibilityModule?.minimumAge ?? "NA";
  const maxAge = protocol?.eligibilityModule?.maximumAge ?? "NA";
  const conditions = protocol?.conditionsModule?.conditions ?? [];
  const phases = protocol?.designModule?.phases ?? [];
  const locationLabels = (protocol?.contactsLocationsModule?.locations ?? []).slice(0, 8).map((loc) =>
    [loc.city, loc.state, loc.country].filter(Boolean).join(", ")
  );
  const trialSpaceText = `Age range allowed: ${minAge} to ${maxAge}. Sex allowed: ${sex}. Cancer type allowed: ${conditions.join(" or ") || "cancer"}. Trial phase: ${phases.join(" or ") || "See trial criteria"}. Histology allowed: See trial criteria. Cancer burden allowed: See trial criteria. Prior treatment required: See trial criteria. Prior treatment excluded: See trial criteria. Biomarkers required: See trial criteria. Biomarkers excluded: See trial criteria.\n\nTrial summary:\n${summary}\n\nEligibility criteria:\n${eligibility}`.trim();
  return {
    spaceId: `${id}-heuristic-1`,
    nctId: id,
    title,
    overallStatus: protocol?.statusModule?.overallStatus,
    conditions,
    phases,
    locations: locationLabels.filter(Boolean),
    url: `https://clinicaltrials.gov/study/${id}`,
    trialSpaceText,
    boilerplateText: extractBoilerplateFromCriteria(eligibility)
  };
}

function extractBoilerplateFromCriteria(criteria: string): string {
  const lines = criteria
    .split(/\r?\n/)
    .map((line) => line.replace(/^\s*[-*]\s*/, "").trim())
    .filter(Boolean);
  const relevant = lines.filter((line) =>
    /ecog|performance|brain|cardiac|heart|renal|kidney|hepatic|liver|infection|hiv|hepatitis|pregnan|autoimmune|pneumonitis|interstitial|uncontrolled|organ dysfunction/i.test(line)
  );
  return relevant.slice(0, 20).join("\n");
}
