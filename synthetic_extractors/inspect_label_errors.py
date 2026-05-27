import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_parquet")
    parser.add_argument("--gold-standard-label-column", required=True)
    parser.add_argument("--ai-labeled-column", required=True)
    parser.add_argument("--reasoning-column", default="raw_response")
    parser.add_argument("--text-column", default="RPT_TEXT")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--positive-value", default="1",
                        help="Value considered 'positive' for FP/FN computation; compared after string coercion (default: '1')")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.input_parquet)
    df = df[df["split"].astype(str).str.contains("train", na=False)].copy()

    gold = args.gold_standard_label_column
    pred = args.ai_labeled_column

    def norm(s):
        return s.astype(str).str.strip().str.lower().replace({"1.0": "1", "0.0": "0", "true": "1", "false": "0"})

    gold_norm = norm(df[gold])
    pred_norm = norm(df[pred])
    pos = str(args.positive_value).strip().lower()
    pos = {"1.0": "1", "0.0": "0", "true": "1", "false": "0"}.get(pos, pos)

    print(f"Train rows: {len(df)}")
    print(f"\nCrosstab of {gold} (rows) vs {pred} (cols):")
    print(pd.crosstab(gold_norm, pred_norm, dropna=False))

    fp_mask = (gold_norm != pos) & (pred_norm == pos)
    fn_mask = (gold_norm == pos) & (pred_norm != pos)

    fp = df[fp_mask]
    fn = df[fn_mask]

    print(f"\nFalse positives: {len(fp)}    False negatives: {len(fn)}")

    n = args.n_samples
    fp_sample = fp.sample(n=min(n, len(fp)), random_state=args.random_seed) if len(fp) else fp
    fn_sample = fn.sample(n=min(n, len(fn)), random_state=args.random_seed) if len(fn) else fn

    def dump(label, sample):
        print("\n" + "=" * 80)
        print(f"{label} ({len(sample)} shown)")
        print("=" * 80)
        for i, (_, row) in enumerate(sample.iterrows(), 1):
            print(f"\n--- {label} #{i} (gold={row[gold]!r}, pred={row[pred]!r}) ---")
            print("RPT_TEXT:")
            print(row.get(args.text_column, ""))
            print(f"\n{args.reasoning_column}:")
            print(row.get(args.reasoning_column, ""))

    dump("FALSE POSITIVE", fp_sample)
    dump("FALSE NEGATIVE", fn_sample)


if __name__ == "__main__":
    main()
