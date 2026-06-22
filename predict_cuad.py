"""
predict_cuad.py
================
Inference script for the CUAD legal clause analysis model.

Usage:
    # Interactive mode
    python predict_cuad.py --adapter ./legal_qwen_lora --interactive

    # Single clause
    python predict_cuad.py --adapter ./legal_qwen_lora ^
        --clause "Either party may terminate this Agreement upon 30 days written notice."

    # From file
    python predict_cuad.py --adapter ./legal_qwen_lora --file contract.txt

    # Compare fine-tuned vs base
    python predict_cuad.py --adapter ./legal_qwen_lora ^
        --clause "your clause here" --compare
"""

import sys
import argparse
import warnings
import torch
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


def load_model(adapter_path: str, load_adapter: bool = True):
    label = "fine-tuned" if load_adapter else "base"
    print(f"\n  Loading {label} model...")
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
    print(f"  VRAM: {vram:.2f} GB  |  Ready.")
    return model, tokenizer


def build_prompt(clause: str, contract_type: str = "Commercial Contract") -> str:
    user_msg = (
        f"Contract Type: {contract_type}\n\n"
        f"Clause to analyze:\n\n"
        f"{clause}\n\n"
        f"Please provide a legal analysis of this clause."
    )
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def predict(clause: str, model, tokenizer,
            contract_type: str = "Commercial Contract",
            max_new_tokens: int = 300) -> str:
    prompt = build_prompt(clause, contract_type)
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=900).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def print_result(clause: str, response: str, label: str = "FINE-TUNED MODEL"):
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  CLAUSE:")
    for line in clause[:300].split('\n'):
        print(f"    {line}")
    if len(clause) > 300:
        print(f"    ...")
    print(f"\n  ANALYSIS:")
    print(f"  {'-'*63}")
    for line in response.split('\n'):
        print(f"  {line}")
    print(f"{'='*65}")


def main():
    ap = argparse.ArgumentParser(description="CUAD Legal Clause Analyzer")
    ap.add_argument("--adapter",       required=True,  help="Path to LoRA adapter")
    ap.add_argument("--clause",        default=None,   help="Clause text to analyze")
    ap.add_argument("--file",          default=None,   help="Text file with clause")
    ap.add_argument("--contract_type", default="Commercial Contract")
    ap.add_argument("--interactive",   action="store_true")
    ap.add_argument("--compare",       action="store_true", help="Compare vs base model")
    ap.add_argument("--max_tokens",    type=int, default=300)
    args = ap.parse_args()

    print("\nLoading fine-tuned model...")
    model_ft, tokenizer = load_model(args.adapter, load_adapter=True)

    model_base = None
    if args.compare:
        print("Loading base model for comparison...")
        model_base, _ = load_model(args.adapter, load_adapter=False)

    def analyze(clause: str):
        response_ft = predict(clause, model_ft, tokenizer,
                              args.contract_type, args.max_tokens)
        print_result(clause, response_ft, "FINE-TUNED — Qwen2.5-1.5B + CUAD LoRA")

        if model_base:
            response_base = predict(clause, model_base, tokenizer,
                                    args.contract_type, args.max_tokens)
            print_result(clause, response_base, "BASE MODEL — Qwen2.5-1.5B (no fine-tuning)")

    if args.clause:
        analyze(args.clause)

    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            analyze(f.read().strip())

    elif args.interactive:
        print("\nInteractive Legal Clause Analyzer")
        print("Paste a contract clause, press Enter twice to analyze.")
        print("Type 'quit' to exit.\n")

        while True:
            print("─"*65)
            print("Enter clause (blank line to submit):")
            lines = []
            while True:
                line = input()
                if line.strip().lower() == "quit":
                    sys.exit(0)
                if line == "" and lines:
                    break
                if line:
                    lines.append(line)
            if lines:
                analyze(" ".join(lines))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()