#!/usr/bin/env tsx
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

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

interface TrialSpaceRecord {
  spaceId: string;
  nctId: string;
  title: string;
  overallStatus?: string;
  conditions?: string[];
  locations?: string[];
  url?: string;
  trialSpaceText: string;
  boilerplateText: string;
}

interface VersionPayload {
  apiVersion?: string;
  dataTimestamp?: string;
}

const args = new Map<string, string>();
for (let i = 2; i < process.argv.length; i += 2) {
  args.set(process.argv[i], process.argv[i + 1]);
}

const output = resolve(args.get("--output") ?? "public/ctgov_trial_index.json");
const pageSize = Number(args.get("--page-size") ?? 100);
const maxPages = Number(args.get("--max-pages") ?? 10);

async function main() {
  const version = (await fetch("https://clinicaltrials.gov/api/v2/version").then((r) => r.json()).catch(() => ({}))) as VersionPayload;
  const records: TrialSpaceRecord[] = [];
  let pageToken = "";
  for (let page = 0; page < maxPages; page += 1) {
    const params = new URLSearchParams({
      "query.cond": "cancer",
      "filter.overallStatus": "RECRUITING",
      countTotal: "true",
      pageSize: String(pageSize),
      format: "json"
    });
    if (pageToken) params.set("pageToken", pageToken);
    const response = await fetch(`https://clinicaltrials.gov/api/v2/studies?${params}`);
    if (!response.ok) throw new Error(`ClinicalTrials.gov ${response.status}`);
    const payload = (await response.json()) as { studies?: CtGovStudy[]; nextPageToken?: string };
    for (const study of payload.studies ?? []) {
      const record = toRecord(study);
      if (record) records.push(record);
    }
    console.log(`Fetched page ${page + 1}, records=${records.length}`);
    if (!payload.nextPageToken) break;
    pageToken = payload.nextPageToken;
  }

  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, JSON.stringify(records, null, 2) + "\n");
  await writeFile(
    output.replace(/\.json$/, ".manifest.json"),
    JSON.stringify(
      {
        createdAt: new Date().toISOString(),
        source: "clinicaltrials.gov/api/v2/studies",
        ctgovApiVersion: version.apiVersion,
        ctgovDataTimestamp: version.dataTimestamp,
        embeddingModel: "ksg-dfci/TrialSpace-0526-ONNX",
        embeddingDim: 0,
        trialSpaces: records.length,
        indexUrl: `/${output.split("/").at(-1)}`
      },
      null,
      2
    ) + "\n"
  );
  console.log(`Wrote ${output}`);
}

function toRecord(study: CtGovStudy): TrialSpaceRecord | null {
  const protocol = study.protocolSection;
  const id = protocol?.identificationModule?.nctId;
  if (!id) return null;
  const title = protocol?.identificationModule?.briefTitle || protocol?.identificationModule?.officialTitle || id;
  const summary = protocol?.descriptionModule?.briefSummary ?? "";
  const eligibility = protocol?.eligibilityModule?.eligibilityCriteria ?? "";
  const conditions = protocol?.conditionsModule?.conditions ?? [];
  const locations = (protocol?.contactsLocationsModule?.locations ?? [])
    .slice(0, 12)
    .map((loc) => [loc.city, loc.state, loc.country].filter(Boolean).join(", "))
    .filter(Boolean);
  const minAge = protocol?.eligibilityModule?.minimumAge ?? "NA";
  const maxAge = protocol?.eligibilityModule?.maximumAge ?? "NA";
  const sex = protocol?.eligibilityModule?.sex ?? "ALL";
  const trialSpaceText = `Age range allowed: ${minAge} to ${maxAge}. Sex allowed: ${sex}. Cancer type allowed: ${conditions.join(" or ") || "cancer"}. Histology allowed: See trial criteria. Cancer burden allowed: See trial criteria. Prior treatment required: See trial criteria. Prior treatment excluded: See trial criteria. Biomarkers required: See trial criteria. Biomarkers excluded: See trial criteria.\n\nTrial summary:\n${summary}\n\nEligibility criteria:\n${eligibility}`.trim();
  return {
    spaceId: `${id}-heuristic-1`,
    nctId: id,
    title,
    overallStatus: protocol?.statusModule?.overallStatus,
    conditions,
    locations,
    url: `https://clinicaltrials.gov/study/${id}`,
    trialSpaceText,
    boilerplateText: boilerplate(eligibility)
  };
}

function boilerplate(criteria: string): string {
  return criteria
    .split(/\r?\n/)
    .map((line) => line.replace(/^\s*[-*]\s*/, "").trim())
    .filter((line) => /ecog|performance|brain|cardiac|heart|renal|kidney|hepatic|liver|infection|hiv|hepatitis|pregnan|autoimmune|pneumonitis|interstitial|uncontrolled|organ dysfunction/i.test(line))
    .slice(0, 30)
    .join("\n");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
