import { describe, expect, it } from "vitest";
import { ctGovStudyUrl } from "../src/lib/ctGov";

describe("ctGovStudyUrl", () => {
  it("builds a ClinicalTrials.gov study URL from an NCT ID", () => {
    expect(ctGovStudyUrl(" NCT01234567 ")).toBe("https://clinicaltrials.gov/study/NCT01234567");
  });

  it("returns an empty string without an NCT ID", () => {
    expect(ctGovStudyUrl("")).toBe("");
  });
});
