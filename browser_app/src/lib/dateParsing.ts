import { isValid, parse, parseISO } from "date-fns";

const DATE_PATTERNS = [
  "yyyy-MM-dd",
  "yyyy/MM/dd",
  "MM/dd/yyyy",
  "M/d/yyyy",
  "MM-dd-yyyy",
  "M-d-yyyy",
  "MMM d, yyyy",
  "MMMM d, yyyy",
  "d MMM yyyy",
  "d MMMM yyyy",
  "yyyy-MM",
  "MM/yyyy",
  "M/yyyy"
];

export interface ParsedDate {
  input: string;
  date: Date;
  epochMs: number;
  isoDate: string;
}

export function parseClinicalDate(value: unknown): ParsedDate {
  if (value instanceof Date && isValid(value)) {
    return toParsedDate(value, value.toISOString());
  }

  if (typeof value === "number" && Number.isFinite(value)) {
    const date = new Date(value);
    if (isValid(date)) return toParsedDate(date, String(value));
  }

  const raw = String(value ?? "").trim();
  if (!raw) throw new Error("Missing date");

  const iso = parseISO(raw);
  if (isValid(iso)) return toParsedDate(iso, raw);

  for (const pattern of DATE_PATTERNS) {
    const parsed = parse(raw, pattern, new Date(2000, 0, 1));
    if (isValid(parsed)) return toParsedDate(parsed, raw);
  }

  const native = new Date(raw);
  if (isValid(native)) return toParsedDate(native, raw);

  throw new Error(`Could not parse date: ${raw}`);
}

export function sortByParsedDate<T extends { epochMs: number }>(records: T[]): T[] {
  return [...records].sort((a, b) => a.epochMs - b.epochMs);
}

function toParsedDate(date: Date, input: string): ParsedDate {
  const normalized = new Date(date);
  if (Number.isNaN(normalized.getTime())) throw new Error(`Could not parse date: ${input}`);
  return {
    input,
    date: normalized,
    epochMs: normalized.getTime(),
    isoDate: normalized.toISOString().slice(0, 10)
  };
}
