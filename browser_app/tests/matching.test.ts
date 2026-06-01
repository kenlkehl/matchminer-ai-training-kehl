import { beforeEach, describe, expect, it, vi } from "vitest";
import { scoreBoilerplateChecker, scoreTrialChecker } from "../src/services/modelRuntime";
import { cosineSimilarity, retrieveByEmbedding, scoreAndRankMatches } from "../src/services/matching";
import type { TrialSpaceRecord } from "../src/types";

vi.mock("../src/services/modelRuntime", () => ({
  scoreBoilerplateChecker: vi.fn(),
  scoreTrialChecker: vi.fn()
}));

describe("matching helpers", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

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

  it("ranks display matches by TrialChecker score instead of retrieval similarity", async () => {
    vi.mocked(scoreTrialChecker).mockResolvedValue([0.1, 0.9, 0.6]);
    vi.mocked(scoreBoilerplateChecker).mockResolvedValue([0.2, 0.3, 0.4]);

    const matches = await scoreAndRankMatches({
      patientSummary: "patient summary",
      patientBoilerplate: "patient boilerplate",
      candidates: [
        { trial: trial("best-retrieval", [1, 0]), cosineSimilarity: 0.95 },
        { trial: trial("best-trial-checker", [0, 1]), cosineSimilarity: 0.25 },
        { trial: trial("middle-trial-checker", [0.5, 0.5]), cosineSimilarity: 0.5 }
      ],
      trialCheckerModelId: "trial-checker",
      boilerplateCheckerModelId: "boilerplate-checker",
      dtype: "auto",
      displayCount: 3
    });

    expect(matches.map((match) => match.trial.spaceId)).toEqual([
      "best-trial-checker",
      "middle-trial-checker",
      "best-retrieval"
    ]);
    expect(matches.map((match) => match.rank)).toEqual([1, 2, 3]);
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
