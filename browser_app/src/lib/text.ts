import type { ClinicalNote } from "../types";

export interface SerialSummaryChunk {
  text: string;
  firstDate: string;
  lastDate: string;
  tokenStart: number;
  tokenEnd: number;
  tokenCount: number;
}

export interface TokenTextCodec {
  encode(text: string): number[] | Promise<number[]>;
  decode(tokenIds: number[]): string | Promise<string>;
}

export function buildRecordSegment(notes: ClinicalNote[]): string {
  return notes
    .map((note) => `=== Clinical Note dated ${note.isoDate} ===\n${note.text.trim()}\n`)
    .join("\n")
    .trim();
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

export async function chunkClinicalNotesByTokens(
  notes: ClinicalNote[],
  tokenizer: TokenTextCodec,
  options: { chunkSizeTokens: number; overlapTokens: number }
): Promise<SerialSummaryChunk[]> {
  if (notes.length === 0) return [];

  const chunkSize = Math.max(256, Math.floor(options.chunkSizeTokens));
  const overlap = Math.max(0, Math.min(Math.floor(options.overlapTokens), chunkSize - 1));
  const fullText = buildRecordSegment(notes);
  const allDates = notes.map((note) => note.isoDate);
  const allTokens = await tokenizer.encode(fullText);

  if (allTokens.length <= chunkSize) {
    return [{
      text: fullText,
      firstDate: allDates[0],
      lastDate: allDates.at(-1) ?? allDates[0],
      tokenStart: 0,
      tokenEnd: allTokens.length,
      tokenCount: allTokens.length
    }];
  }

  const chunks: SerialSummaryChunk[] = [];
  const stride = chunkSize - overlap;
  const dateHeaderPattern = /=== Clinical Note dated (.+?) ===/g;
  let start = 0;

  while (start < allTokens.length) {
    const end = Math.min(start + chunkSize, allTokens.length);
    const chunkTokens = allTokens.slice(start, end);
    const chunkText = (await tokenizer.decode(chunkTokens)).trim();
    const foundDates = Array.from(chunkText.matchAll(dateHeaderPattern), (match) => match[1]);
    const previousLastDate = chunks.at(-1)?.lastDate ?? allDates[0];

    chunks.push({
      text: chunkText,
      firstDate: foundDates[0] ?? previousLastDate,
      lastDate: foundDates.at(-1) ?? foundDates[0] ?? previousLastDate,
      tokenStart: start,
      tokenEnd: end,
      tokenCount: chunkTokens.length
    });

    if (end >= allTokens.length) break;
    start += stride;
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
