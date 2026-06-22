"""
download_cuad.py
=================
Downloads CUAD-DeepSeek dataset from HuggingFace.

Dataset: zenml/cuad-deepseek
  - 32,700 contract clauses
  - 41 clause types + NONE category
  - Expert rationales (why clause belongs to category)
  - DeepSeek reasoning traces (how to arrive at classification)
  - clause_with_context (150 chars before/after for disambiguation)

Splits:
  train      : 26,100 rows
  validation :  3,270 rows
  test       :  3,270 rows

Usage:
    python download_cuad.py
    python download_cuad.py --output raw/cuad  --splits train validation
"""

import argparse
import pandas as pd
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Download CUAD-DeepSeek dataset")
    ap.add_argument("--output", default="raw/cuad",  help="Output folder (default: raw/cuad)")
    ap.add_argument("--splits", nargs="+",
                    default=["train", "validation", "test"],
                    help="Splits to download (default: all three)")
    args = ap.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  CUAD-DeepSeek Downloader")
    print("="*60)
    print(f"  Dataset : zenml/cuad-deepseek")
    print(f"  Splits  : {args.splits}")
    print(f"  Output  : {output_dir}/")
    print("="*60)

    try:
        from datasets import load_dataset
    except ImportError:
        print("\n[ERROR] datasets not installed.")
        print("  Run: pip install datasets")
        return

    print("\nDownloading dataset (first run ~200MB)...")
    ds = load_dataset("zenml/cuad-deepseek")
    print(f"  Available splits: {list(ds.keys())}")

    all_dfs = []

    for split in args.splits:
        if split not in ds:
            print(f"  [SKIP] Split '{split}' not found")
            continue

        print(f"\nProcessing split: {split}")
        df = ds[split].to_pandas()
        print(f"  Rows    : {len(df)}")
        print(f"  Columns : {df.columns.tolist()}")

        # Save raw split
        out_path = output_dir / f"cuad_{split}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved   : {out_path}")
        all_dfs.append(df)

    # ── Dataset stats ─────────────────────────────────────────────────────────
    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)

        print(f"\n{'='*60}")
        print(f"  DATASET STATISTICS")
        print(f"{'='*60}")
        print(f"  Total rows          : {len(df_all):,}")
        print(f"  Unique labels       : {df_all['label'].nunique()}")
        print(f"  Unique contracts    : {df_all['contract_name'].nunique()}")

        print(f"\n  Label distribution (top 15):")
        for label, count in df_all['label'].value_counts().head(15).items():
            pct = count / len(df_all) * 100
            bar = "█" * int(pct / 2)
            print(f"  {label:<35} {count:>5}  ({pct:>4.1f}%)  {bar}")

        print(f"\n  NONE label count: {(df_all['label'] == 'NONE').sum():,}")
        print(f"  Non-NONE count  : {(df_all['label'] != 'NONE').sum():,}")

        # Clause length stats
        df_all["clause_len"] = df_all["clause"].apply(lambda x: len(str(x).split()))
        print(f"\n  Clause length (words):")
        print(f"    Min    : {df_all['clause_len'].min()}")
        print(f"    Median : {df_all['clause_len'].median():.0f}")
        print(f"    Mean   : {df_all['clause_len'].mean():.0f}")
        print(f"    Max    : {df_all['clause_len'].max()}")

        # Rationale length stats
        df_all["rationale_len"] = df_all["rationale"].apply(lambda x: len(str(x).split()))
        print(f"\n  Rationale length (words):")
        print(f"    Min    : {df_all['rationale_len'].min()}")
        print(f"    Median : {df_all['rationale_len'].median():.0f}")
        print(f"    Mean   : {df_all['rationale_len'].mean():.0f}")
        print(f"    Max    : {df_all['rationale_len'].max()}")

        # Sample row
        print(f"\n  Sample row (train[0]):")
        row = pd.read_csv(output_dir / "cuad_train.csv").iloc[0]
        print(f"    label          : {row['label']}")
        print(f"    contract_type  : {row['contract_type']}")
        print(f"    clause         : {str(row['clause'])[:200]}...")
        print(f"    rationale      : {str(row['rationale'])[:200]}...")

        print(f"\n{'='*60}")
        print(f"  DOWNLOAD COMPLETE")
        print(f"{'='*60}")
        print(f"  Files saved in {output_dir}/:")
        for f in sorted(output_dir.iterdir()):
            size_mb = f.stat().st_size / 1e6
            lines   = sum(1 for _ in open(f, encoding="utf-8")) - 1
            print(f"    {f.name:<30} {lines:>6} rows  {size_mb:.1f} MB")
        print()
        print("  Next step:")
        print("    python format_cuad.py --data_dir raw/cuad --output finetune_data/cuad")


if __name__ == "__main__":
    main()