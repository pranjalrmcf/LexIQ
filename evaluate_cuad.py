"""
evaluate_cuad.py
=================
Evaluates fine-tuned CUAD legal clause analysis model.

Metrics:
  1. Clause Type Accuracy    — exact match on extracted clause type
  2. Risk Level Accuracy     — exact match on extracted risk level
  3. ROUGE-L                 — longest common subsequence overlap with reference
  4. BERTScore               — semantic similarity (best single metric for generation)
  5. Analysis completeness   — did model generate full analysis or truncate

Comparison:
  Base Qwen2.5-1.5B (no fine-tuning) vs Fine-tuned CUAD model

Usage:
    # Full evaluation with base comparison
    python evaluate_cuad.py ^
        --adapter ./legal_qwen_lora ^
        --test_data finetune_data/cuad/cuad_test_chat.jsonl ^
        --compare_base

    # Quick test on 50 samples
    python evaluate_cuad.py ^
        --adapter ./legal_qwen_lora ^
        --test_data finetune_data/cuad/cuad_test_chat.jsonl ^
        --max_samples 50

Install:
    pip install rouge-score bert-score
"""

import re
import sys
import json
import argparse
import warnings
import torch
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

BNBCONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

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

VALID_RISK_LEVELS = {"low", "medium", "medium-high", "high", "not applicable"}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model(adapter_path: str, load_adapter: bool = True):
    label = "fine-tuned" if load_adapter else "base"
    print(f"\n  Loading {label} model: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BNBCONFIG,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base, adapter_path) if load_adapter else base
    model.eval()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM: {vram:.2f} GB")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(user_content: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def predict_batch(user_contents: list, model, tokenizer,
                  batch_size: int = 4, max_new_tokens: int = 300) -> list:
    responses = []
    total     = len(user_contents)

    for i in range(0, total, batch_size):
        batch   = user_contents[i: i + batch_size]
        prompts = [build_prompt(c) for c in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=900,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for out in outputs:
            new_toks = out[inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(new_toks, skip_special_tokens=True).strip()
            responses.append(response)

        done = min(i + batch_size, total)
        print(f"  Progress: {done}/{total} ({100*done//total}%)", end="\r")

    print()
    return responses


# ─────────────────────────────────────────────────────────────────────────────
# PARSING MODEL OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def extract_clause_type(text: str) -> str:
    """Extract 'Clause Type: X' from model output."""
    match = re.search(r"Clause Type\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def extract_risk_level(text: str) -> str:
    """Extract 'Risk Level: X' from model output."""
    match = re.search(r"Risk Level\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    return ""


def extract_analysis(text: str) -> str:
    """Extract the analysis section between 'Analysis:' and 'Risk Level:'"""
    match = re.search(
        r"Analysis\s*:\s*\n(.+?)(?=Risk Level|$)",
        text, re.IGNORECASE | re.DOTALL
    )
    if match:
        return match.group(1).strip()
    # Fallback — return everything between clause type and risk level
    match2 = re.search(
        r"Clause Type.*?\n\n(.+?)(?=Risk Level|$)",
        text, re.IGNORECASE | re.DOTALL
    )
    if match2:
        return match2.group(1).strip()
    return text.strip()


def normalize_clause_type(text: str) -> str:
    """Normalize clause type for comparison — lowercase, strip punctuation."""
    return re.sub(r'[^a-z0-9\s/]', '', text.lower()).strip()


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_rouge_l(predictions: list, references: list) -> dict:
    """ROUGE-L scores between generated and reference analyses."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = []
        for pred, ref in zip(predictions, references):
            if pred and ref:
                score = scorer.score(ref, pred)
                scores.append(score['rougeL'].fmeasure)
        return {
            "mean":   round(np.mean(scores), 4) if scores else 0,
            "median": round(np.median(scores), 4) if scores else 0,
            "std":    round(np.std(scores), 4) if scores else 0,
        }
    except ImportError:
        print("  [SKIP] rouge-score not installed. Run: pip install rouge-score")
        return {"mean": None, "median": None, "std": None}


def compute_bertscore(predictions: list, references: list) -> dict:
    """BERTScore F1 — semantic similarity between generated and reference."""
    try:
        from bert_score import score as bert_score
        # Use a small model to save VRAM during eval
        P, R, F1 = bert_score(
            predictions, references,
            model_type="distilbert-base-uncased",
            lang="en",
            verbose=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        f1_scores = F1.numpy()
        return {
            "mean":   round(float(np.mean(f1_scores)), 4),
            "median": round(float(np.median(f1_scores)), 4),
            "std":    round(float(np.std(f1_scores)), 4),
        }
    except ImportError:
        print("  [SKIP] bert-score not installed. Run: pip install bert-score")
        return {"mean": None, "median": None, "std": None}


def compute_clause_type_accuracy(
        pred_types: list, ref_types: list) -> dict:
    """
    Exact match accuracy on clause type.
    Also computes partial match (predicted type is substring of reference).
    """
    exact   = 0
    partial = 0
    parsed  = 0

    for pred, ref in zip(pred_types, ref_types):
        pred_n = normalize_clause_type(pred)
        ref_n  = normalize_clause_type(ref)

        if pred_n:
            parsed += 1
        if pred_n == ref_n:
            exact += 1
        elif pred_n and (pred_n in ref_n or ref_n in pred_n):
            partial += 1

    n = len(pred_types)
    return {
        "exact_match":   round(exact / n, 4) if n else 0,
        "partial_match": round((exact + partial) / n, 4) if n else 0,
        "parse_rate":    round(parsed / n, 4) if n else 0,
        "n":             n,
    }


def compute_risk_accuracy(pred_risks: list, ref_risks: list) -> dict:
    """Exact match accuracy on risk level."""
    correct = 0
    parsed  = 0

    for pred, ref in zip(pred_risks, ref_risks):
        pred_n = pred.strip().lower()
        ref_n  = ref.strip().lower()
        if pred_n in VALID_RISK_LEVELS:
            parsed += 1
        if pred_n == ref_n:
            correct += 1

    n = len(pred_risks)
    return {
        "accuracy":   round(correct / n, 4) if n else 0,
        "parse_rate": round(parsed / n, 4) if n else 0,
        "n":          n,
    }


def compute_completeness(responses: list) -> dict:
    """
    Check if model generated a complete response with all three sections:
    Clause Type, Analysis, Risk Level.
    """
    complete     = 0
    has_type     = 0
    has_analysis = 0
    has_risk     = 0

    for resp in responses:
        t = bool(extract_clause_type(resp))
        a = bool(extract_analysis(resp))
        r = bool(extract_risk_level(resp))
        has_type     += t
        has_analysis += a
        has_risk     += r
        if t and a and r:
            complete += 1

    n = len(responses)
    return {
        "complete_responses":  round(complete / n, 4) if n else 0,
        "has_clause_type":     round(has_type / n, 4) if n else 0,
        "has_analysis":        round(has_analysis / n, 4) if n else 0,
        "has_risk_level":      round(has_risk / n, 4) if n else 0,
        "avg_response_words":  round(np.mean([len(r.split()) for r in responses]), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(responses: list, references: list,
                   ref_types: list, ref_risks: list,
                   label: str, skip_bertscore: bool = False) -> dict:

    print(f"\n{'='*68}")
    print(f"  {label}")
    print(f"{'='*68}")

    # Extract predictions
    pred_types    = [extract_clause_type(r)  for r in responses]
    pred_risks    = [extract_risk_level(r)   for r in responses]
    pred_analyses = [extract_analysis(r)     for r in responses]
    ref_analyses  = [extract_analysis(r)     for r in references]

    # ── 1. Completeness ───────────────────────────────────────────────────────
    comp = compute_completeness(responses)
    print(f"\n  OUTPUT COMPLETENESS:")
    print(f"    Complete responses (all 3 sections) : {comp['complete_responses']*100:.1f}%")
    print(f"    Has Clause Type                     : {comp['has_clause_type']*100:.1f}%")
    print(f"    Has Analysis                        : {comp['has_analysis']*100:.1f}%")
    print(f"    Has Risk Level                      : {comp['has_risk_level']*100:.1f}%")
    print(f"    Avg response length (words)         : {comp['avg_response_words']:.0f}")

    # ── 2. Clause type accuracy ───────────────────────────────────────────────
    ct = compute_clause_type_accuracy(pred_types, ref_types)
    print(f"\n  CLAUSE TYPE ACCURACY:")
    print(f"    Exact match accuracy  : {ct['exact_match']*100:.1f}%")
    print(f"    Partial match         : {ct['partial_match']*100:.1f}%")
    print(f"    Parse rate            : {ct['parse_rate']*100:.1f}%")

    # ── 3. Risk level accuracy ────────────────────────────────────────────────
    rl = compute_risk_accuracy(pred_risks, ref_risks)
    print(f"\n  RISK LEVEL ACCURACY:")
    print(f"    Accuracy              : {rl['accuracy']*100:.1f}%")
    print(f"    Parse rate            : {rl['parse_rate']*100:.1f}%")

    # ── 4. ROUGE-L ────────────────────────────────────────────────────────────
    print(f"\n  ROUGE-L (analysis section):")
    rouge = compute_rouge_l(pred_analyses, ref_analyses)
    if rouge["mean"] is not None:
        print(f"    Mean   : {rouge['mean']:.4f}")
        print(f"    Median : {rouge['median']:.4f}")
        print(f"    Std    : {rouge['std']:.4f}")
    else:
        print(f"    Skipped (rouge-score not installed)")

    # ── 5. BERTScore ──────────────────────────────────────────────────────────
    bertscore = {"mean": None}
    if not skip_bertscore:
        print(f"\n  BERTScore F1 (semantic similarity):")
        print(f"    Computing... (uses distilbert, ~2-3 min)")
        bertscore = compute_bertscore(pred_analyses, ref_analyses)
        if bertscore["mean"] is not None:
            print(f"    Mean   : {bertscore['mean']:.4f}")
            print(f"    Median : {bertscore['median']:.4f}")
            print(f"    Std    : {bertscore['std']:.4f}")
        else:
            print(f"    Skipped (bert-score not installed)")
    else:
        print(f"\n  BERTScore: skipped (--skip_bertscore)")

    return {
        "label":           label,
        "completeness":    comp,
        "clause_type":     ct,
        "risk_level":      rl,
        "rouge_l":         rouge,
        "bertscore":       bertscore,
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(results_ft: dict, results_base: dict = None):
    print(f"\n{'='*68}")
    print(f"  SUMMARY COMPARISON")
    print(f"{'='*68}")

    has_base = results_base is not None

    header = f"  {'Metric':<35}  {'Fine-Tuned':>12}"
    if has_base:
        header += f"  {'Base Model':>12}  {'Delta':>8}"
    print(header)
    print(f"  {'-'*35}  {'-'*12}" + (f"  {'-'*12}  {'-'*8}" if has_base else ""))

    def row(name, ft_val, base_val=None, fmt=".1f", suffix="%", higher_better=True):
        ft_str = f"{ft_val:{fmt}}{suffix}" if ft_val is not None else "N/A"
        line   = f"  {name:<35}  {ft_str:>12}"
        if has_base and base_val is not None:
            base_str = f"{base_val:{fmt}}{suffix}"
            delta    = ft_val - base_val if (ft_val is not None and base_val is not None) else None
            if delta is not None:
                flag     = "↑" if (delta > 0) == higher_better else "↓" if delta != 0 else "→"
                delta_str = f"{delta:+.1f}{suffix}"
                line += f"  {base_str:>12}  {delta_str:>7} {flag}"
            else:
                line += f"  {base_str:>12}  {'N/A':>8}"
        print(line)

    ft  = results_ft
    bas = results_base or {}

    # Completeness
    row("Complete responses",
        ft["completeness"]["complete_responses"]*100,
        bas.get("completeness", {}).get("complete_responses", 0)*100 if has_base else None)

    row("Avg response length (words)",
        ft["completeness"]["avg_response_words"],
        bas.get("completeness", {}).get("avg_response_words") if has_base else None,
        fmt=".0f", suffix=" w")

    # Clause type
    row("Clause type exact match",
        ft["clause_type"]["exact_match"]*100,
        bas.get("clause_type", {}).get("exact_match", 0)*100 if has_base else None)

    row("Clause type partial match",
        ft["clause_type"]["partial_match"]*100,
        bas.get("clause_type", {}).get("partial_match", 0)*100 if has_base else None)

    # Risk level
    row("Risk level accuracy",
        ft["risk_level"]["accuracy"]*100,
        bas.get("risk_level", {}).get("accuracy", 0)*100 if has_base else None)

    # ROUGE-L
    if ft["rouge_l"]["mean"] is not None:
        row("ROUGE-L (analysis)",
            ft["rouge_l"]["mean"]*100,
            bas.get("rouge_l", {}).get("mean", 0)*100 if has_base else None)

    # BERTScore
    if ft["bertscore"]["mean"] is not None:
        row("BERTScore F1 (semantic)",
            ft["bertscore"]["mean"]*100,
            bas.get("bertscore", {}).get("mean", 0)*100 if has_base else None)

    print(f"{'='*68}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Evaluate CUAD Legal Analysis Model")
    ap.add_argument("--adapter",        required=True, help="Path to LoRA adapter")
    ap.add_argument("--test_data",      required=True, help="Path to cuad_test_chat.jsonl")
    ap.add_argument("--compare_base",   action="store_true", help="Also run base model")
    ap.add_argument("--max_samples",    type=int, default=None, help="Limit samples for quick test")
    ap.add_argument("--batch_size",     type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--skip_bertscore", action="store_true",
                    help="Skip BERTScore (saves ~3 min but loses best metric)")
    ap.add_argument("--show_examples",  type=int, default=3,
                    help="Show N example outputs (default 3)")
    args = ap.parse_args()

    print(f"\n{'='*68}")
    print(f"  CUAD LEGAL ANALYSIS — EVALUATION")
    print(f"{'='*68}")

    # ── Load test data ────────────────────────────────────────────────────────
    print(f"\nLoading: {args.test_data}")
    records = []
    with open(args.test_data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.max_samples:
        import random
        random.seed(42)
        random.shuffle(records)   
        records = records[:args.max_samples]
    print(f"  Test examples: {len(records)}")

    # Extract inputs and references
    user_contents = [r["messages"][1]["content"] for r in records]
    references    = [r["messages"][2]["content"] for r in records]
    ref_types     = [extract_clause_type(r)  for r in references]
    ref_risks     = [extract_risk_level(r)   for r in references]

    # Label distribution in test set
    from collections import Counter
    label_counts = Counter(ref_types)
    print(f"\n  Test set clause type distribution (top 8):")
    for label, count in label_counts.most_common(8):
        print(f"    {label:<35} {count:>4}")

    # ── Base model ────────────────────────────────────────────────────────────
    results_base = None
    if args.compare_base:
        print(f"\nRunning BASE model inference...")
        model_base, tok_base = load_model(args.adapter, load_adapter=False)
        responses_base = predict_batch(
            user_contents, model_base, tok_base,
            args.batch_size, args.max_new_tokens
        )
        results_base = evaluate_model(
            responses_base, references, ref_types, ref_risks,
            "BASE MODEL — Qwen2.5-1.5B (no fine-tuning)",
            skip_bertscore=args.skip_bertscore,
        )
        del model_base
        torch.cuda.empty_cache()
        print(f"\n  Base model unloaded.")

    # ── Fine-tuned model ──────────────────────────────────────────────────────
    print(f"\nRunning FINE-TUNED model inference...")
    model_ft, tok_ft = load_model(args.adapter, load_adapter=True)
    responses_ft = predict_batch(
        user_contents, model_ft, tok_ft,
        args.batch_size, args.max_new_tokens
    )
    results_ft = evaluate_model(
        responses_ft, references, ref_types, ref_risks,
        "FINE-TUNED — Qwen2.5-1.5B + CUAD LoRA",
        skip_bertscore=args.skip_bertscore,
    )

    # ── Comparison ────────────────────────────────────────────────────────────
    print_comparison(results_ft, results_base)

    # ── Example outputs ───────────────────────────────────────────────────────
    if args.show_examples > 0:
        print(f"\n{'='*68}")
        print(f"  EXAMPLE OUTPUTS (first {args.show_examples})")
        print(f"{'='*68}")

        for i in range(min(args.show_examples, len(records))):
            clause_preview = user_contents[i].split("Clause to analyze:\n\n")[-1][:200]
            print(f"\n  Example {i+1}:")
            print(f"  Clause   : {clause_preview}...")
            print(f"  Reference: {references[i][:200]}...")
            print(f"  Predicted: {responses_ft[i][:200]}...")
            print(f"  Type match: {normalize_clause_type(extract_clause_type(responses_ft[i]))} "
                  f"vs {normalize_clause_type(ref_types[i])}")

    # ── Portfolio talking point ───────────────────────────────────────────────
    ct_acc   = results_ft["clause_type"]["exact_match"] * 100
    rl_acc   = results_ft["risk_level"]["accuracy"] * 100
    rouge    = results_ft["rouge_l"]["mean"]
    bscore   = results_ft["bertscore"]["mean"]

    print(f"\n{'='*68}")
    print(f"  PORTFOLIO TALKING POINT")
    print(f"{'='*68}")
    print(f"  Fine-tuned Qwen2.5-1.5B via QLoRA (r=16) on CUAD-DeepSeek")
    print(f"  for legal contract clause analysis.")
    print(f"  Achieved {ct_acc:.1f}% clause type accuracy, {rl_acc:.1f}% risk")
    print(f"  level accuracy", end="")
    if rouge:
        print(f", ROUGE-L {rouge:.3f}", end="")
    if bscore:
        print(f", BERTScore F1 {bscore:.3f}", end="")
    print(f" on held-out CUAD test set.")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()