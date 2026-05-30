import type { TrialIndexManifest, TrialSpaceRecord } from "../types";
import { embedText } from "./modelRuntime";
import { loadTrialIndex, saveTrialIndex } from "./storage";

interface CtGovStudy {
  protocolSection?: {
    identificationModule?: { nctId?: string; briefTitle?: string; officialTitle?: string };
    statusModule?: { overallStatus?: string };
    descriptionModule?: { briefSummary?: string; detailedDescription?: string };
    conditionsModule?: { conditions?: string[] };
    eligibilityModule?: { eligibilityCriteria?: string; sex?: string; minimumAge?: string; maximumAge?: string };
    contactsLocationsModule?: { locations?: Array<{ city?: string; state?: string; country?: string; status?: string }> };
  };
}

export async function loadManifest(): Promise<TrialIndexManifest> {
  const response = await fetch("/trial_index.manifest.json");
  if (!response.ok) throw new Error("Could not load trial index manifest");
  return response.json();
}

export async function loadOrFetchTrialIndex(): Promise<TrialSpaceRecord[]> {
  const cached = await loadTrialIndex();
  if (cached.length) return cached;
  const manifest = await loadManifest();
  const response = await fetch(manifest.indexUrl);
  if (!response.ok) throw new Error(`Could not load trial index at ${manifest.indexUrl}`);
  const records = (await response.json()) as TrialSpaceRecord[];
  await saveTrialIndex(records);
  return records;
}

export async function ensureTrialEmbeddings(records: TrialSpaceRecord[], modelId: string, onProgress?: (done: number, total: number) => void): Promise<TrialSpaceRecord[]> {
  const updated: TrialSpaceRecord[] = [];
  for (let index = 0; index < records.length; index += 1) {
    const record = records[index];
    if (record.embedding?.length) {
      updated.push(record);
    } else {
      const embedding = await embedText(modelId, record.trialSpaceText);
      updated.push({ ...record, embedding });
    }
    onProgress?.(index + 1, records.length);
  }
  await saveTrialIndex(updated);
  return updated;
}

export async function fetchCtGovCancerTrials(options: { pageSize?: number; maxPages?: number } = {}): Promise<TrialSpaceRecord[]> {
  const pageSize = options.pageSize ?? 100;
  const maxPages = options.maxPages ?? 2;
  let pageToken = "";
  const records: TrialSpaceRecord[] = [];

  for (let page = 0; page < maxPages; page += 1) {
    const params = new URLSearchParams({
      "query.cond": "cancer",
      "filter.overallStatus": "RECRUITING",
      countTotal: "true",
      pageSize: String(pageSize),
      format: "json"
    });
    if (pageToken) params.set("pageToken", pageToken);
    const response = await fetch(`https://clinicaltrials.gov/api/v2/studies?${params.toString()}`);
    if (!response.ok) throw new Error(`ClinicalTrials.gov request failed: ${response.status}`);
    const payload = (await response.json()) as { studies?: CtGovStudy[]; nextPageToken?: string };
    for (const study of payload.studies ?? []) {
      const record = ctGovStudyToTrialSpace(study);
      if (record) records.push(record);
    }
    if (!payload.nextPageToken) break;
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
  const locationLabels = (protocol?.contactsLocationsModule?.locations ?? []).slice(0, 8).map((loc) =>
    [loc.city, loc.state, loc.country].filter(Boolean).join(", ")
  );
  const trialSpaceText = `Age range allowed: ${minAge} to ${maxAge}. Sex allowed: ${sex}. Cancer type allowed: ${conditions.join(" or ") || "cancer"}. Histology allowed: See trial criteria. Cancer burden allowed: See trial criteria. Prior treatment required: See trial criteria. Prior treatment excluded: See trial criteria. Biomarkers required: See trial criteria. Biomarkers excluded: See trial criteria.\n\nTrial summary:\n${summary}\n\nEligibility criteria:\n${eligibility}`.trim();
  return {
    spaceId: `${id}-heuristic-1`,
    nctId: id,
    title,
    overallStatus: protocol?.statusModule?.overallStatus,
    conditions,
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
