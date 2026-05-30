import type { MatchResult, TrialSpaceRecord } from "../types";
import { scoreBoilerplateChecker, scoreTrialChecker } from "./modelRuntime";

export function cosineSimilarity(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  if (n === 0) return 0;
  let dot = 0;
  let aMag = 0;
  let bMag = 0;
  for (let i = 0; i < n; i += 1) {
    dot += a[i] * b[i];
    aMag += a[i] * a[i];
    bMag += b[i] * b[i];
  }
  if (!aMag || !bMag) return 0;
  return dot / (Math.sqrt(aMag) * Math.sqrt(bMag));
}

export function retrieveByEmbedding(patientEmbedding: number[], trials: TrialSpaceRecord[], count: number): Array<{ trial: TrialSpaceRecord; cosineSimilarity: number }> {
  return trials
    .filter((trial) => trial.embedding?.length)
    .map((trial) => ({
      trial,
      cosineSimilarity: cosineSimilarity(patientEmbedding, trial.embedding ?? [])
    }))
    .sort((a, b) => b.cosineSimilarity - a.cosineSimilarity)
    .slice(0, count);
}

export async function scoreAndRankMatches(input: {
  patientSummary: string;
  patientBoilerplate: string;
  candidates: Array<{ trial: TrialSpaceRecord; cosineSimilarity: number }>;
  trialCheckerModelId: string;
  boilerplateCheckerModelId: string;
  dtype: string;
  displayCount: number;
}): Promise<MatchResult[]> {
  const warnings: string[] = [];
  let trialScores: number[] = [];
  try {
    trialScores = await scoreTrialChecker(
      input.trialCheckerModelId,
      input.candidates.map((candidate) => formatTrialCheckerPair(input.patientSummary, candidate.trial.trialSpaceText)),
      input.dtype
    );
  } catch (error) {
    warnings.push(`TrialChecker unavailable: ${error instanceof Error ? error.message : String(error)}`);
    trialScores = input.candidates.map((candidate) => candidate.cosineSimilarity);
  }

  const ordered = input.candidates
    .map((candidate, index) => ({
      ...candidate,
      trialCheckerScore: trialScores[index] ?? null
    }))
    .sort((a, b) => (b.trialCheckerScore ?? b.cosineSimilarity) - (a.trialCheckerScore ?? a.cosineSimilarity));

  const deduped: typeof ordered = [];
  const seenNcts = new Set<string>();
  for (const candidate of ordered) {
    if (seenNcts.has(candidate.trial.nctId)) continue;
    seenNcts.add(candidate.trial.nctId);
    deduped.push(candidate);
    if (deduped.length >= input.displayCount) break;
  }

  let boilerplateScores: number[] = [];
  try {
    boilerplateScores = await scoreBoilerplateChecker(
      input.boilerplateCheckerModelId,
      deduped.map((candidate) => formatBoilerplatePair(input.patientBoilerplate, candidate.trial.boilerplateText)),
      input.dtype
    );
  } catch (error) {
    warnings.push(`BoilerplateChecker unavailable: ${error instanceof Error ? error.message : String(error)}`);
    boilerplateScores = deduped.map(() => Number.NaN);
  }

  return deduped.map((candidate, index) => ({
    id: `${candidate.trial.spaceId}-${index}`,
    trial: candidate.trial,
    cosineSimilarity: candidate.cosineSimilarity,
    trialCheckerScore: candidate.trialCheckerScore,
    boilerplateScore: Number.isFinite(boilerplateScores[index]) ? boilerplateScores[index] : null,
    rank: index + 1,
    warnings
  }));
}

export function formatTrialCheckerPair(patientSummary: string, trialSpace: string): string {
  return `${trialSpace}\nNow here is the patient summary:${patientSummary}`;
}

export function formatBoilerplatePair(patientBoilerplate: string, trialBoilerplate: string): string {
  return `Patient history: ${patientBoilerplate || ""}\nTrial exclusions:${trialBoilerplate || ""}`;
}
