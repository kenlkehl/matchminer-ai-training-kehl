import { describe, expect, it } from "vitest";
import { normalizeEmbeddedTrialIndexUrl, parseEmbeddedTrialIndexText } from "../src/services/trialImport";

describe("parseEmbeddedTrialIndexText", () => {
  it("normalizes real-time eval pre-embedded rows", () => {
    const records = parseEmbeddedTrialIndexText(
      JSON.stringify([
        {
          id: 123,
          nct_id: "NCT01234567",
          this_cohort: "Age range allowed: 18 years and older. Cancer type allowed: lung cancer.",
          boilerplate_text: "No uncontrolled infection.",
          embedding: [0.1, 0.2, 0.3]
        }
      ]),
      "trial_space_embeddings.json"
    );

    expect(records).toHaveLength(1);
    expect(records[0]).toMatchObject({
      spaceId: "123",
      nctId: "NCT01234567",
      title: "NCT01234567",
      trialSpaceText: "Age range allowed: 18 years and older. Cancer type allowed: lung cancer.",
      boilerplateText: "No uncontrolled infection.",
      url: "https://clinicaltrials.gov/study/NCT01234567"
    });
    expect(records[0].embedding).toEqual([0.1, 0.2, 0.3]);
  });

  it("parses CSV rows with JSON-stringified embeddings", () => {
    const records = parseEmbeddedTrialIndexText(
      [
        "id,nct_id,this_cohort,boilerplate_text,embedding",
        'space-a,NCT00000001,Trial space text,,"[0.5, 0.25]"'
      ].join("\n"),
      "trial_space_embeddings.csv"
    );

    expect(records[0].spaceId).toBe("space-a");
    expect(records[0].embedding).toEqual([0.5, 0.25]);
  });

  it("accepts wrapped browser-native records", () => {
    const records = parseEmbeddedTrialIndexText(
      JSON.stringify({
        records: [
          {
            spaceId: "space-a",
            nctId: "NCT00000001",
            title: "A trial",
            trialSpaceText: "Trial space text",
            boilerplateText: "",
            embedding: ["1", "0"]
          }
        ]
      }),
      "browser-index.json"
    );

    expect(records[0].title).toBe("A trial");
    expect(records[0].embedding).toEqual([1, 0]);
  });

  it("parses JSONL even when the source URL has query parameters", () => {
    const records = parseEmbeddedTrialIndexText(
      [
        JSON.stringify({
          id: "space-a",
          nct_id: "NCT00000001",
          this_cohort: "Trial space text",
          boilerplate_text: "",
          embedding: [1, 0]
        }),
        JSON.stringify({
          id: "space-b",
          nct_id: "NCT00000002",
          this_cohort: "Another trial space",
          boilerplate_text: "",
          embedding: [0, 1]
        })
      ].join("\n"),
      "https://huggingface.co/org/repo/resolve/main/trials.jsonl?download=true"
    );

    expect(records.map((record) => record.spaceId)).toEqual(["space-a", "space-b"]);
  });

  it("normalizes Hugging Face blob URLs to downloadable dataset files", () => {
    expect(
      normalizeEmbeddedTrialIndexUrl("https://huggingface.co/datasets/ksg-dfci/mmai-synthetic-0526/blob/main/trial_space_embeddings_6-2-26.parquet")
    ).toBe("https://huggingface.co/datasets/ksg-dfci/mmai-synthetic-0526/resolve/main/trial_space_embeddings_6-2-26.parquet?download=true");
  });
});
