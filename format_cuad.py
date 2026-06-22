"""
format_cuad.py v2
==================
Changes from v1:
  1. NONE examples dropped entirely (--none_ratio 0 default)
  2. Rationale cleaning improved:
     - Removes "Classification Rationale: Label" header artifacts
     - Removes markdown bold, numbered lists, bullet points
     - Caps at 150 words to control output length
  3. Default max_train reduced to 5000 (better time estimate)

Usage:
    python format_cuad.py --data_dir raw/cuad --output finetune_data/cuad

Options:
    --max_train     Max training examples (default 5000)
    --none_ratio    Fraction of NONE to keep, 0=drop all (default 0)
    --min_rationale Min words in rationale (default 30)
    --max_rationale Max words in rationale output (default 150)
    --seed          Random seed (default 42)
"""

import re
import json
import argparse
import random
import pandas as pd
import numpy as np
from pathlib import Path


CLAUSE_RISK = {
    "Change Of Control":                "High",
    "Anti-Assignment":                  "High",
    "Non-Compete":                      "High",
    "Exclusivity":                      "High",
    "Ip Ownership Assignment":          "High",
    "License Grant":                    "High",
    "Non-Transferable License":         "High",
    "Unlimited/All-You-Can-Eat-License":"High",
    "Irrevocability Or Perpetual":      "High",
    "Third Party Beneficiary":          "High",
    "Termination For Convenience":      "Medium-High",
    "Governing Law":                    "Medium-High",
    "Indemnification":                  "Medium-High",
    "Insurance":                        "Medium-High",
    "Liquidated Damages":               "Medium-High",
    "Price Restrictions":               "Medium-High",
    "Rofr/Rofo/Rofn":                   "Medium-High",
    "Source Code Escrow":               "Medium-High",
    "Cap On Liability":                 "Medium-High",
    "Warranty Duration":                "Medium-High",
    "Confidentiality":                  "Medium",
    "Non-Disparagement":                "Medium",
    "Audit Rights":                     "Medium",
    "Most Favored Nation":              "Medium",
    "Minimum Commitment":               "Medium",
    "Volume Restriction":               "Medium",
    "Competitive Restriction Exception":"Medium",
    "Covenant Not To Sue":              "Medium",
    "Post-Termination Services":        "Medium",
    "Renewal Term":                     "Medium",
    "Agreement Date":                   "Low",
    "Document Name":                    "Low",
    "Parties":                          "Low",
    "Effective Date":                   "Low",
    "Expiration Date":                  "Low",
    "Notice Period To Terminate Renewal":"Low",
    "Auto-Renewal":                     "Low",
    "Revenue/Profit Sharing":           "Low",
    "Joint Ip Ownership":               "Low",
    "Affiliate License-Licensor":       "Low",
    "Affiliate License-Licensee":       "Low",
    "NONE":                             "Not Applicable",
}

SYSTEM_PROMPT = (
    "You are an expert legal AI assistant specializing in commercial contract review. "
    "Your role is to analyze contract clauses and provide clear, structured legal analysis "
    "that helps lawyers and business professionals understand key provisions and potential risks.\n\n"
    "For each clause provided, you will:\n"
    "1. Identify the clause type from standard legal categories\n"
    "2. Provide a concise analysis explaining the legal significance\n"
    "3. Assign a risk level: Low, Medium, Medium-High, or High\n\n"
    "Be precise, professional, and flag any provisions that require careful attention."
)


# ─────────────────────────────────────────────────────────────────────────────
# RATIONALE CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_rationale(text: str, max_words: int = 150) -> str:
    """
    Clean DeepSeek rationale for use as training target.

    Removes:
    - "Classification Rationale: Label" header (DeepSeek prompt artifact)
    - "Rationale for Classification as X" headers
    - Markdown bold (**text**)
    - Markdown headers (##, ###)
    - Numbered list markers (1. 2. 3.)
    - Bullet points (- • *)
    - Excessive newlines

    Caps output at max_words words.
    """
    if not text or pd.isna(text):
        return ""

    text = str(text)

    # Remove bold FIRST so header is exposed as plain text
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)

    # NOW remove the exposed header artifact
    text = re.sub(
        r'^(Classification Rationale|Rationale for Classification)[^\n]*\n?',
        '', text, flags=re.MULTILINE | re.IGNORECASE
    )

    # FIX 2 — Remove markdown bold **text** → text
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)

    # FIX 3 — Remove markdown italic *text* → text
    text = re.sub(r'\*(.+?)\*', r'\1', text)

    # FIX 4 — Remove markdown headers (## Heading)
    text = re.sub(r'^#{1,4}\s+', '', text, flags=re.MULTILINE)

    # FIX 5 — Remove numbered list markers (1. 2. a. b.)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[a-z]\.\s+', '', text, flags=re.MULTILINE)

    # FIX 6 — Remove bullet points (- • *)
    text = re.sub(r'^\s*[-•*]\s+', '', text, flags=re.MULTILINE)

    # FIX 7 — Remove quoted clause echoing (often just repeats the input)
    # Pattern: lines starting with " or containing full clause in quotes
    text = re.sub(r'^"[^"]{20,}"[^\n]*\n?', '', text, flags=re.MULTILINE)

    # FIX 8 — Collapse excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # FIX 9 — Clean up per-line whitespace
    lines = [l.strip() for l in text.split('\n')]
    lines = [l for l in lines if len(l) > 2 or l == '']
    text  = '\n'.join(lines).strip()

    # FIX 10 — Cap at max_words
    words = text.split()
    if len(words) > max_words:
        text = ' '.join(words[:max_words]) + '...'

    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT BUILDING
# ─────────────────────────────────────────────────────────────────────────────

def build_output(row: pd.Series, max_words: int = 150) -> str:
    """
    Structured legal analysis report — natural language, no JSON.
    Format:
        Clause Type: [label]

        Analysis:
        [cleaned rationale]

        Risk Level: [Low / Medium / Medium-High / High]
    """
    label     = str(row["label"]).strip()
    rationale = clean_rationale(str(row["rationale"]), max_words)
    risk      = CLAUSE_RISK.get(label, "Medium")

    return (
        f"Clause Type: {label}\n\n"
        f"Analysis:\n"
        f"{rationale}\n\n"
        f"Risk Level: {risk}"
    )


def build_user_message(row: pd.Series) -> str:
    """
    User prompt — clause_with_context preferred over raw clause.
    Includes contract type for domain context.
    """
    clause        = str(row["clause"]).strip()
    context       = str(row.get("clause_with_context", "")).strip()
    contract_type = str(row.get("contract_type", "Commercial Contract")).strip()

    # Use context version if it's meaningfully longer
    display_text = context if len(context) > len(clause) + 20 else clause

    return (
        f"Contract Type: {contract_type}\n\n"
        f"Clause to analyze:\n\n"
        f"{display_text}\n\n"
        f"Please provide a legal analysis of this clause."
    )


def format_chat(row: pd.Series, max_words: int = 150) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": build_user_message(row)},
            {"role": "assistant", "content": build_output(row, max_words)},
        ]
    }


def format_alpaca(row: pd.Series, max_words: int = 150) -> dict:
    return {
        "instruction": (
            "Analyze this legal contract clause. "
            "Identify the clause type, provide a concise legal analysis, "
            "and assess the risk level."
        ),
        "input":  build_user_message(row),
        "output": build_output(row, max_words),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Format CUAD-DeepSeek for fine-tuning (Task C)")
    ap.add_argument("--data_dir",       default="raw/cuad",          help="Folder with cuad_*.csv files")
    ap.add_argument("--output",         default="finetune_data/cuad", help="Output folder")
    ap.add_argument("--max_train",      type=int,   default=5000,    help="Max training examples (default 5000)")
    ap.add_argument("--none_ratio",     type=float, default=0.0,     help="Fraction of NONE to keep, 0=drop all (default 0)")
    ap.add_argument("--min_rationale",  type=int,   default=30,      help="Min words in rationale to include (default 30)")
    ap.add_argument("--max_rationale",  type=int,   default=150,     help="Max words in output rationale (default 150)")
    ap.add_argument("--seed",           type=int,   default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*65)
    print("  CUAD-DeepSeek Formatter v2 — Task C: Legal Clause Analysis")
    print("="*65)
    print(f"  NONE examples   : {'dropped entirely' if args.none_ratio == 0 else f'kept at {args.none_ratio:.0%}'}")
    print(f"  Max rationale   : {args.max_rationale} words")
    print(f"  Max train       : {args.max_train}")

    # ── Load splits ───────────────────────────────────────────────────────────
    dfs = {}
    for split in ["train", "validation", "test"]:
        fpath = data_dir / f"cuad_{split}.csv"
        if fpath.exists():
            dfs[split] = pd.read_csv(fpath)
            print(f"\n  Loaded {split:<12}: {len(dfs[split]):>6} rows")
        else:
            print(f"  [SKIP] {fpath} not found")

    if not dfs:
        print("\n[ERROR] No CSV files found. Run download_cuad.py first.")
        return

    # ── Process each split ────────────────────────────────────────────────────
    for split, df in dfs.items():

        print(f"\n{'='*65}")
        print(f"  Processing: {split}")
        print(f"{'='*65}")
        print(f"  Raw rows                    : {len(df):>6}")

        # Filter 1 — drop NONE
        if args.none_ratio == 0:
            df = df[df["label"] != "NONE"].copy()
            print(f"  After dropping NONE         : {len(df):>6}")
        else:
            none_mask   = df["label"] == "NONE"
            df_none     = df[none_mask]
            df_non_none = df[~none_mask]
            n_keep      = int(len(df_non_none) * args.none_ratio / (1 - args.none_ratio))
            n_keep      = min(n_keep, len(df_none))
            df          = pd.concat(
                [df_non_none, df_none.sample(n=n_keep, random_state=args.seed)],
                ignore_index=True
            )
            print(f"  After NONE sampling ({args.none_ratio:.0%})    : {len(df):>6}")

        # Filter 2 — rationale quality
        df["rationale_wc"] = df["rationale"].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        df = df[df["rationale_wc"] >= args.min_rationale].copy()
        print(f"  After rationale filter      : {len(df):>6}")

        # Filter 3 — clause quality
        df["clause_wc"] = df["clause"].apply(lambda x: len(str(x).split()))
        df = df[df["clause_wc"] >= 5].copy()
        print(f"  After clause filter         : {len(df):>6}")

        # Shuffle
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

        # Cap train
        if split == "train" and len(df) > args.max_train:
            df = df.sample(n=args.max_train, random_state=args.seed).reset_index(drop=True)
            print(f"  Capped at {args.max_train}               : {len(df):>6}")

        # ── Label distribution ────────────────────────────────────────────────
        print(f"\n  Label distribution (top 12):")
        for label, count in df["label"].value_counts().head(12).items():
            risk = CLAUSE_RISK.get(label, "Medium")
            pct  = count / len(df) * 100
            print(f"    {label:<35} {count:>5} ({pct:>4.1f}%)  [{risk}]")

        # ── Verify cleaning on sample ─────────────────────────────────────────
        if split == "train":
            sample_row = df.iloc[0]
            raw_rat    = str(sample_row["rationale"])
            clean_rat  = clean_rationale(raw_rat, args.max_rationale)

            print(f"\n  Cleaning verification (first example):")
            print(f"    Label          : {sample_row['label']}")
            print(f"    Contract type  : {sample_row['contract_type']}")
            print(f"    Clause (first 100 chars): {str(sample_row['clause'])[:100]}...")
            print(f"    Raw rationale  ({len(raw_rat.split())} words): {raw_rat[:150]}...")
            print(f"    Clean rationale ({len(clean_rat.split())} words): {clean_rat[:150]}...")

            print(f"\n  Full formatted output:")
            print("-"*65)
            output = build_output(sample_row, args.max_rationale)
            for line in output.split('\n'):
                print(f"    {line}")
            print("-"*65)

        # ── Write JSONL ───────────────────────────────────────────────────────
        chat_path   = output_dir / f"cuad_{split}_chat.jsonl"
        alpaca_path = output_dir / f"cuad_{split}_alpaca.jsonl"

        with open(chat_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(format_chat(row, args.max_rationale)) + "\n")

        with open(alpaca_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(format_alpaca(row, args.max_rationale)) + "\n")

        size_mb = chat_path.stat().st_size / 1e6
        print(f"\n  Saved:")
        print(f"    {chat_path.name:<45} {len(df):>5} examples  {size_mb:.1f} MB")
        print(f"    {alpaca_path.name:<45} {len(df):>5} examples")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  FORMATTING COMPLETE")
    print(f"{'='*65}")

    train_chat = output_dir / "cuad_train_chat.jsonl"
    val_chat   = output_dir / "cuad_validation_chat.jsonl"

    if train_chat.exists():
        n_train = sum(1 for _ in open(train_chat, encoding="utf-8"))
        # Rough time estimate: ~150 tokens output, 1024 max_length
        # RTX 4070 laptop: ~800 tokens/sec throughput
        # Steps = n_train * epochs / effective_batch
        steps_per_epoch = n_train // 8
        time_3ep = round(steps_per_epoch * 3 * 1024 / 800 / 60, 0)
        time_5ep = round(steps_per_epoch * 5 * 1024 / 800 / 60, 0)

        print(f"\n  Training examples  : {n_train}")
        print(f"  Validation examples: {sum(1 for _ in open(val_chat, encoding='utf-8'))}")
        print(f"\n  Estimated training time:")
        print(f"    3 epochs  →  ~{int(time_3ep//60)}h {int(time_3ep%60)}m")
        print(f"    5 epochs  →  ~{int(time_5ep//60)}h {int(time_5ep%60)}m")
        print(f"\n  Run training:")
        print(f"    python finetune_cuad.py \\")
        print(f"      --data {train_chat} \\")
        print(f"      --val_data {val_chat} \\")
        print(f"      --output ./legal_qwen_lora")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()