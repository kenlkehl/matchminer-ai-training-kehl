import { describe, expect, it } from "vitest";
import { chunkClinicalNotesByTokens, chunkTextByTokens, fillPrompt, splitBoilerplate } from "../src/lib/text";
import { tokenIdsFromTokenizerOutput } from "../src/services/modelRuntime";

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
});

const asciiCharacterTokenizer = {
  encode: (text: string) => Array.from(text, (char) => char.charCodeAt(0)),
  decode: (tokenIds: number[]) => tokenIds.map((id) => String.fromCharCode(id)).join("")
};
