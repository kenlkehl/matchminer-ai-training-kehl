import type { ClinicalNote } from "../types";

export function buildRecordSegment(notes: ClinicalNote[]): string {
  return notes
    .map((note) => `=== Clinical Note dated ${note.isoDate} ===\n${note.text.trim()}`)
    .join("\n\n");
}

export function splitBoilerplate(summary: string): { patientSummary: string; patientBoilerplate: string } {
  const marker = /(?:^|\n)\s*Boilerplate conditions:\s*/i;
  const match = summary.match(marker);
  if (!match || match.index === undefined) {
    return { patientSummary: summary.trim(), patientBoilerplate: "" };
  }
  const before = summary.slice(0, match.index).trim();
  const after = summary.slice(match.index + match[0].length).trim();
  return { patientSummary: before, patientBoilerplate: after };
}

export function fillPrompt(template: string, values: Record<string, string>): string {
  return Object.entries(values).reduce(
    (prompt, [key, value]) => prompt.replaceAll(`{${key}}`, value),
    template
  );
}

export function chunkTextByCharacters(text: string, maxChars = 18000, overlapChars = 1200): string[] {
  const clean = text.trim();
  if (clean.length <= maxChars) return [clean];
  const chunks: string[] = [];
  let start = 0;
  while (start < clean.length) {
    const hardEnd = Math.min(clean.length, start + maxChars);
    let end = hardEnd;
    const nextBreak = clean.lastIndexOf("\n\n", hardEnd);
    if (nextBreak > start + maxChars * 0.55) end = nextBreak;
    chunks.push(clean.slice(start, end).trim());
    if (end >= clean.length) break;
    start = Math.max(0, end - overlapChars);
  }
  return chunks;
}

export function buildExtractiveFallbackSummary(notes: ClinicalNote[]): string {
  const all = notes.map((note) => `${note.isoDate}: ${note.text}`).join("\n\n");
  const candidates = all
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean);
  const important = candidates.filter((s) =>
    /cancer|carcinoma|sarcoma|leukemia|lymphoma|metasta|stage|biopsy|mutation|marker|chemo|radiation|surgery|immunotherapy|targeted|trial|ecog|brain|renal|hepatic|heart|pneumonitis/i.test(s)
  );
  const selected = (important.length ? important : candidates).slice(0, 24).join(" ");
  const first = notes[0]?.isoDate ?? "unknown date";
  const last = notes.at(-1)?.isoDate ?? first;
  return `Age: Not clearly documented
Sex: Not clearly documented
Cancer type: See source record text
Histology: See source record text
Current extent: See source record text
Biomarkers: See source record text
Treatment history:
# Source records from ${first} to ${last}. ${selected}

Boilerplate conditions:
Review the source records directly for performance status, organ dysfunction, active infections, brain metastases, pneumonitis, cardiac disease, or prior cancers.`;
}
