import { describe, expect, it } from "vitest";
import { cosineSimilarity, retrieveByEmbedding } from "../src/services/matching";
import type { TrialSpaceRecord } from "../src/types";

describe("matching helpers", () => {
  it("computes cosine similarity", () => {
    expect(cosineSimilarity([1, 0], [1, 0])).toBeCloseTo(1);
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0);
  });

  it("retrieves trials in descending similarity order", () => {
    const trials: TrialSpaceRecord[] = [
      trial("a", [0, 1]),
      trial("b", [1, 0]),
      trial("c", [0.5, 0.5])
    ];
    expect(retrieveByEmbedding([1, 0], trials, 2).map((item) => item.trial.spaceId)).toEqual(["b", "c"]);
  });
});

function trial(spaceId: string, embedding: number[]): TrialSpaceRecord {
  return {
    spaceId,
    nctId: `NCT-${spaceId}`,
    title: spaceId,
    trialSpaceText: spaceId,
    boilerplateText: "",
    embedding
  };
}
