import { describe, expect, it } from "vitest";
import { docTagsToPlainText } from "../src/services/graniteDoclingOcr";

describe("docTagsToPlainText", () => {
  it("removes DocTags and location tags while preserving text order", () => {
    const text = docTagsToPlainText(
      [
        "<doctag>",
        "<page_header><loc_10><loc_20><loc_30><loc_40>Clinic Note</page_header>",
        "<section_header_level_1><loc_1><loc_2><loc_3><loc_4>Assessment</section_header_level_1>",
        "<text><loc_1><loc_2><loc_3><loc_4>Metastatic lung cancer with KRAS G12C.</text>",
        "<list_item><loc_1><loc_2><loc_3><loc_4>Prior platinum therapy.</list_item>",
        "</doctag><|end_of_text|>"
      ].join("")
    );

    expect(text).toBe("Clinic Note\nAssessment\nMetastatic lung cancer with KRAS G12C.\nPrior platinum therapy.");
  });
});
