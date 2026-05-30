import Papa from "papaparse";
import type { ClinicalNote, PatientDocument } from "../types";
import { parseClinicalDate, sortByParsedDate } from "../lib/dateParsing";

export async function parseCsvPatientFile(file: File): Promise<PatientDocument> {
  const text = await file.text();
  const parsed = Papa.parse<Record<string, unknown>>(text, {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: false
  });

  if (parsed.errors.length) {
    const first = parsed.errors[0];
    throw new Error(`CSV parse error on row ${first.row ?? "unknown"}: ${first.message}`);
  }

  const fields = parsed.meta.fields ?? [];
  const dateCol = fields.find((field) => field.toLowerCase() === "date");
  const textCol = fields.find((field) => field.toLowerCase() === "text");
  if (!dateCol || !textCol) {
    throw new Error("CSV must contain columns named date and text");
  }

  const notes: ClinicalNote[] = parsed.data
    .map((row, index) => {
      const rawDate = row[dateCol];
      const rawText = row[textCol];
      const parsedDate = parseClinicalDate(rawDate);
      const body = String(rawText ?? "").trim();
      if (!body) return null;
      return {
        id: `csv-${index}`,
        dateInput: parsedDate.input,
        isoDate: parsedDate.isoDate,
        epochMs: parsedDate.epochMs,
        text: body
      } satisfies ClinicalNote;
    })
    .filter((note): note is ClinicalNote => note !== null);

  if (!notes.length) throw new Error("CSV contains no non-empty text rows");

  const sorted = sortByParsedDate(notes);
  return {
    id: crypto.randomUUID(),
    source: "csv",
    fileName: file.name,
    notes: sorted,
    rawText: sorted.map((note) => `${note.isoDate}\n${note.text}`).join("\n\n"),
    createdAt: new Date().toISOString()
  };
}
