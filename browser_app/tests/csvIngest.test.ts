import { describe, expect, it } from "vitest";
import { parseCsvPatientFile } from "../src/services/csvIngest";

describe("parseCsvPatientFile", () => {
  it("requires date and text columns and sorts by parsed dates", async () => {
    const file = new File(
      [
        "text,date\n",
        "later note,2/1/2024\n",
        "earlier note,10/1/2023\n"
      ],
      "patient.csv",
      { type: "text/csv" }
    );

    const doc = await parseCsvPatientFile(file);
    expect(doc.source).toBe("csv");
    expect(doc.notes.map((note) => note.text)).toEqual(["earlier note", "later note"]);
  });

  it("rejects missing required columns", async () => {
    const file = new File(["when,body\n2024-01-01,text\n"], "bad.csv", { type: "text/csv" });
    await expect(parseCsvPatientFile(file)).rejects.toThrow(/date and text/);
  });
});
