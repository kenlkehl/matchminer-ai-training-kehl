import { createRequire } from "node:module";
import { describe, expect, it } from "vitest";
import { DEFAULT_LLAMA_GGUF_FILE, DEFAULT_LLAMA_GGUF_REPO, DEFAULT_MODEL_SETTINGS } from "../src/data/defaultSettings";

const require = createRequire(import.meta.url);
const electronArtifacts = require("../electron/runtime/artifacts.cjs");

describe("default model settings", () => {
  it("uses the desktop LLM context and summary chunk defaults", () => {
    expect(DEFAULT_MODEL_SETTINGS.llmContextTokens).toBe(30000);
    expect(DEFAULT_MODEL_SETTINGS.maxSummaryTokens).toBe(15000);
    expect(DEFAULT_MODEL_SETTINGS.summaryChunkTokens).toBe(10000);
  });

  it("uses OncoReasoning as the default native GGUF artifact", () => {
    expect(DEFAULT_LLAMA_GGUF_REPO).toBe("ksg-dfci/OncoReasoning-0526-GGUF");
    expect(DEFAULT_LLAMA_GGUF_FILE).toBe("OncoReasoning-0526.Q4_K_M.gguf");
    expect(electronArtifacts.DEFAULT_LLAMA_REPO).toBe(DEFAULT_LLAMA_GGUF_REPO);
    expect(electronArtifacts.DEFAULT_LLAMA_FILE).toBe(DEFAULT_LLAMA_GGUF_FILE);
  });
});
