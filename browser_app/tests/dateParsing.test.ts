import { describe, expect, it } from "vitest";
import { parseClinicalDate, sortByParsedDate } from "../src/lib/dateParsing";

describe("parseClinicalDate", () => {
  it("parses common clinical date formats", () => {
    expect(parseClinicalDate("2024-01-02").isoDate).toBe("2024-01-02");
    expect(parseClinicalDate("1/2/2024").isoDate).toBe("2024-01-02");
    expect(parseClinicalDate("Jan 2, 2024").isoDate).toBe("2024-01-02");
  });

  it("sorts by parsed datetime rather than string order", () => {
    const records = [
      { label: "feb", ...parseClinicalDate("2/1/2024") },
      { label: "jan", ...parseClinicalDate("10/1/2023") },
      { label: "mar", ...parseClinicalDate("2024-03-01") }
    ];
    expect(sortByParsedDate(records).map((record) => record.label)).toEqual(["jan", "feb", "mar"]);
  });

  it("throws on unparseable dates", () => {
    expect(() => parseClinicalDate("not a date")).toThrow(/Could not parse date/);
  });
});
