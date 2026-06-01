export function ctGovStudyUrl(nctId: string | null | undefined): string {
  const id = String(nctId ?? "").trim();
  return id ? `https://clinicaltrials.gov/study/${encodeURIComponent(id)}` : "";
}
