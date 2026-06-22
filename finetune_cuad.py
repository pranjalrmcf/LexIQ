"""
finetune_cuad.py
=================
QLoRA Fine-Tuning: Qwen2.5-1.5B on CUAD Legal Clause Analysis
Optimized for ~1.5 hours on RTX 4070 Laptop 8GB

Task    : Legal clause analysis — identify type, explain provisions, flag risk
Model   : Qwen/Qwen2.5-1.5B-Instruct
Method  : QLoRA (4-bit NF4 + LoRA adapters)
Dataset : CUAD-DeepSeek, 5000 train / 1065 val

Hyperparams tuned for 2hr budget:
  r=16          — balanced capacity/speed (r=8 too weak, r=32 too slow)
  epochs=5      — enough for legal reasoning to form
  lr=1e-4       — slightly higher than OCEAN task (shorter outputs = less overfit risk)
  max_length=1024 — covers clause + context + 150 word output
  grad_accum=8  — effective batch=8

Usage:
    python finetune_cuad.py ^
        --data finetune_data/cuad/cuad_train_chat.jsonl ^
        --val_data finetune_data/cuad/cuad_validation_chat.jsonl ^
        --output ./legal_qwen_lora
"""

import os
import sys
import json
import argparse
import warnings
import torch
warnings.filterwarnings("ignore")

print("\n" + "="*60)
print("  CUAD Legal Clause — QLoRA Fine-Tuner")
print("="*60)

if not torch.cuda.is_available():
    print("[ERROR] No CUDA GPU detected.")
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  GPU     : {gpu_name}")
print(f"  VRAM    : {vram_gb:.1f} GB")
print(f"  CUDA    : {torch.version.cuda}")
print(f"  PyTorch : {torch.__version__}")
print("="*60)

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

BNBCONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# r=16 — middle ground between OCEAN (r=8 too weak) and v2 (r=32 too slow)
# Legal reasoning needs more capacity than classification but not max
LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,       # scaling ratio 2x
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj",
        "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_qwen_prompt(example: dict) -> str:
    """Qwen2.5 ChatML format."""
    messages = example.get("messages", [])
    prompt   = ""
    for msg in messages:
        role    = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    return prompt


def prepare_dataset(jsonl_path: str, tokenizer, max_length: int):
    print(f"\n  Loading: {jsonl_path}")
    records = load_jsonl(jsonl_path)
    print(f"  Raw examples      : {len(records)}")

    texts = []
    for rec in records:
        prompt = format_qwen_prompt(rec)
        if prompt.strip():
            texts.append({"text": prompt})

    print(f"  Formatted         : {len(texts)}")

    kept    = []
    skipped = 0
    for item in texts:
        ids = tokenizer(item["text"], truncation=False)["input_ids"]
        if len(ids) <= max_length:
            kept.append(item)
        else:
            skipped += 1

    print(f"  Within {max_length} tokens  : {len(kept)}")
    if skipped:
        print(f"  Skipped (too long): {skipped}")

    return Dataset.from_list(kept)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="QLoRA Fine-tune Qwen2.5 on CUAD Legal")
    ap.add_argument("--data",        required=True,                   help="Path to cuad_train_chat.jsonl")
    ap.add_argument("--val_data",    required=True,                   help="Path to cuad_validation_chat.jsonl")
    ap.add_argument("--output",      default="./legal_qwen_lora",     help="Output dir for adapter weights")
    ap.add_argument("--epochs",      type=int,   default=5,           help="Training epochs (default 5)")
    ap.add_argument("--batch_size",  type=int,   default=1,           help="Batch size (default 1)")
    ap.add_argument("--max_length",  type=int,   default=1024,        help="Max token length (default 1024)")
    ap.add_argument("--lr",          type=float, default=1e-4,        help="Learning rate (default 1e-4)")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── 1. Tokenizer ──────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size : {tokenizer.vocab_size:,}")

    # ── 2. Dataset ────────────────────────────────────────────────────────────
    print(f"\n[2/5] Preparing datasets...")
    train_ds = prepare_dataset(args.data,     tokenizer, args.max_length)
    val_ds   = prepare_dataset(args.val_data, tokenizer, args.max_length)

    print(f"\n  Final sizes — Train: {len(train_ds)}  Val: {len(val_ds)}")

    # ── 3. Load model ─────────────────────────────────────────────────────────
    print(f"\n[3/5] Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BNBCONFIG,
        device_map="auto",
        max_memory={0: "7GB", "cpu": "16GB"},
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.config.use_cache = False

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM after load : {vram:.2f} GB")

    # ── 4. LoRA ───────────────────────────────────────────────────────────────
    print(f"\n[4/5] Attaching LoRA (r={LORA_CONFIG.r})...")
    model = get_peft_model(model, LORA_CONFIG)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Trainable : {trainable:,}  ({100*trainable/total:.3f}%)")
    print(f"  Frozen    : {total-trainable:,}")

    vram2 = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM with adapters : {vram2:.2f} GB")

    # ── VRAM estimate ─────────────────────────────────────────────────────────
    print(f"\n  VRAM estimate for training:")
    print(f"    Base model (NF4)       ~{vram2:.1f} GB")
    print(f"    Activations + grads    ~2.0 GB")
    print(f"    LoRA r=16              ~0.08 GB")
    print(f"    Optimizer (8-bit)      ~0.6 GB")
    print(f"    ─────────────────────────────")
    print(f"    Total                  ~{vram2 + 2.68:.1f} GB  (of {vram_gb:.1f} GB)")

    # ── Time estimate ─────────────────────────────────────────────────────────
    steps_per_epoch = len(train_ds) // 8
    total_steps     = steps_per_epoch * args.epochs
    est_minutes     = total_steps * 1024 / 800 / 60
    print(f"\n  Time estimate:")
    print(f"    Steps per epoch : {steps_per_epoch}")
    print(f"    Total steps     : {total_steps}")
    print(f"    Estimated time  : ~{int(est_minutes//60)}h {int(est_minutes%60)}m")

    # ── 5. Train ──────────────────────────────────────────────────────────────
    effective_batch = args.batch_size * 8
    print(f"\n[5/5] Training...")
    print(f"  Epochs         : {args.epochs}")
    print(f"  Batch size     : {args.batch_size} (grad_accum=8 → effective={effective_batch})")
    print(f"  LR             : {args.lr}")
    print(f"  LoRA rank      : {LORA_CONFIG.r}")
    print(f"  Max length     : {args.max_length}")
    print(f"  Train examples : {len(train_ds)}")
    print(f"  Val examples   : {len(val_ds)}")
    print(f"  Output         : {args.output}")
    print()

    sft_config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=args.lr,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=False,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,          # less frequent eval — val set is larger
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        dataset_text_field="text",
        max_length=args.max_length,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    print("  Training started:")
    print("-"*60)
    train_result = trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n  Saving adapter weights...")
    trainer.model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    metrics_path = os.path.join(args.output, "train_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(train_result.metrics, f, indent=2)

    train_loss    = train_result.metrics.get("train_loss", 0)
    train_runtime = train_result.metrics.get("train_runtime", 0) / 60

    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print("="*60)
    print(f"  Train loss   : {train_loss:.4f}")
    print(f"  Train time   : {train_runtime:.1f} min")
    print(f"  Saved to     : {args.output}/")
    print()
    print("  Loss targets for legal reasoning task:")
    print("    < 1.2  → excellent — model generating coherent legal analysis")
    print("    1.2-1.8 → good — usable analysis with some inconsistencies")
    print("    > 2.0  → poor — likely format issues or insufficient training")
    print()
    print("  Next steps:")
    print(f"    python predict_cuad.py --adapter {args.output} --interactive")
    print("="*60)


if __name__ == "__main__":
    main()