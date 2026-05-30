import { describe, expect, it } from "vitest";
import { fillPrompt, splitBoilerplate } from "../src/lib/text";

describe("text helpers", () => {
  it("splits boilerplate section from patient summary", () => {
    const out = splitBoilerplate("Age: 70\nCancer type: Lung cancer\n\nBoilerplate conditions:\nECOG 1.");
    expect(out.patientSummary).toContain("Lung cancer");
    expect(out.patientBoilerplate).toBe("ECOG 1.");
  });

  it("fills prompt variables", () => {
    expect(fillPrompt("Hello {name}. {name} again.", { name: "patient" })).toBe("Hello patient. patient again.");
  });
});
