import { describe, expect, it, vi } from "vitest";
import { DEFAULT_PROMPTS } from "../src/data/defaultPrompts";
import { chunkClinicalNotesByTokens, chunkTextByTokens, fillPrompt, splitBoilerplate } from "../src/lib/text";
import { stripThinkingBlocks, tokenIdsFromTokenizerOutput } from "../src/services/modelRuntime";

describe("text helpers", () => {
  it("splits boilerplate section from patient summary", () => {
    const out = splitBoilerplate("Age: 70\nCancer type: Lung cancer\n\nBoilerplate conditions:\nECOG 1.");
    expect(out.patientSummary).toContain("Lung cancer");
    expect(out.patientBoilerplate).toBe("ECOG 1.");
  });

  it("fills prompt variables", () => {
    expect(fillPrompt("Hello {name}. {name} again.", { name: "patient" })).toBe("Hello patient. patient again.");
  });

  it("chunks dated clinical notes by token count with overlap", async () => {
    const notes = [
      { id: "a", dateInput: "2024-01-01", isoDate: "2024-01-01", epochMs: 1, text: "A".repeat(90) },
      { id: "b", dateInput: "2024-02-01", isoDate: "2024-02-01", epochMs: 2, text: "B".repeat(90) }
    ];
    const chunks = await chunkClinicalNotesByTokens(notes, asciiCharacterTokenizer, {
      chunkSizeTokens: 256,
      overlapTokens: 10
    });

    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks[0].firstDate).toBe("2024-01-01");
    expect(chunks.at(-1)?.lastDate).toBe("2024-02-01");
    expect(chunks[1].tokenStart).toBe(chunks[0].tokenEnd - 10);
    expect(chunks.every((chunk) => chunk.tokenCount <= 256)).toBe(true);
  });

  it("splits an existing summary chunk without losing fallback dates", async () => {
    const chunks = await chunkTextByTokens("x".repeat(700), asciiCharacterTokenizer, {
      chunkSizeTokens: 300,
      overlapTokens: 50,
      fallbackFirstDate: "2024-03-01",
      fallbackLastDate: "2024-04-01",
      useFallbackDateRangeWhenNoHeaders: true
    });

    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks[0].firstDate).toBe("2024-03-01");
    expect(chunks.at(-1)?.lastDate).toBe("2024-04-01");
    expect(chunks[1].tokenStart).toBe(chunks[0].tokenEnd - 50);
  });

  it("extracts token ids from Transformers.js tensor-like tokenizer output", () => {
    const tensorLikeOutput = {
      input_ids: {
        dims: [1, 4],
        data: new BigInt64Array([101n, 102n, 103n, 104n])
      }
    };

    expect(tokenIdsFromTokenizerOutput(tensorLikeOutput)).toEqual([101, 102, 103, 104]);
  });

  it("keeps the final answer separate from local reasoning model thinking", () => {
    const infoSpy = vi.spyOn(console, "info").mockImplementation(() => undefined);
    try {
      expect(stripThinkingBlocks("<think>private reasoning</think>\nFinal summary")).toBe("Final summary");
      expect(stripThinkingBlocks("<think>unfinished\nFinal summary")).toBe("unfinished\nFinal summary");
    } finally {
      infoSpy.mockRestore();
    }
  });

  it("keeps default prompts aligned with the main repo prompt style", () => {
    const promptText = Object.values(DEFAULT_PROMPTS).map((prompt) => prompt.defaultValue).join("\n");
    expect(promptText).not.toMatch(/<think>|chain-of-thought|Do not output reasoning/i);
    expect(promptText).toContain("Reference: common systemic therapy regimen abbreviations");
    expect(promptText).toContain("After reasoning step by step, compute a score from 0 to 5");
  });
});

const asciiCharacterTokenizer = {
  encode: (text: string) => Array.from(text, (char) => char.charCodeAt(0)),
  decode: (tokenIds: number[]) => tokenIds.map((id) => String.fromCharCode(id)).join("")
};
