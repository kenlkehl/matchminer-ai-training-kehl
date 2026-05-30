import { describe, expect, it } from "vitest";
import { cosineSimilarity } from "../src/services/matching";
import { hashTextEmbedding } from "../src/lib/hashEmbedding";

describe("hashTextEmbedding", () => {
  it("creates normalized deterministic vectors", () => {
    const a = hashTextEmbedding("lung cancer kras g12c", 64);
    const b = hashTextEmbedding("lung cancer kras g12c", 64);
    expect(a).toEqual(b);
    expect(cosineSimilarity(a, b)).toBeCloseTo(1);
  });
});
