import { describe, expect, it } from "vitest";
import { DEFAULT_MODEL_SETTINGS } from "../src/data/defaultSettings";

describe("default model settings", () => {
  it("uses the desktop LLM context and summary chunk defaults", () => {
    expect(DEFAULT_MODEL_SETTINGS.llmContextTokens).toBe(30000);
    expect(DEFAULT_MODEL_SETTINGS.maxSummaryTokens).toBe(15000);
    expect(DEFAULT_MODEL_SETTINGS.summaryChunkTokens).toBe(10000);
  });
});
