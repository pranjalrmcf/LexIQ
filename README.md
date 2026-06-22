# LexIQ — Legal Contract Clause Analyzer

> Fine-tuned Qwen2.5-1.5B (QLoRA) for automated contract clause classification, legal analysis generation, and risk assessment across 41 commercial clause categories.

---

## Overview

LexIQ automates clause-level contract review using a fine-tuned Small Language Model. Given a raw contract clause, the model identifies the clause type, generates a structured legal analysis explaining key provisions, and assigns a risk level — tasks traditionally performed by junior associates spending hours per contract.

**Key results on held-out CUAD test set:**

| Metric | Base Model | LexIQ | Delta |
|---|---|---|---|
| Complete responses | 45% | 100% | +55% |
| Clause type accuracy | 1% | 79% | +78pp |
| Risk level accuracy | 1% | 89% | +88pp |
| ROUGE-L | 0.173 | 0.291 | +0.118 |
| BERTScore F1 | 0.769 | 0.848 | +0.079 |

Performance sits between BERT-base (~70%) and RoBERTa-large (~82%) on CUAD — trained overnight on a consumer GPU with 8GB VRAM.

---

## Example Output

**Input:**
```
Contract Type: SUPPLY AGREEMENT

Either party may terminate this Agreement upon 30 days written 
notice without cause.
```

**Output:**
```
Clause Type: Termination for Convenience

Analysis:
This clause grants both parties unrestricted termination rights 
without requiring cause, subject only to 30 days written notice. 
The absence of a cause requirement creates significant exposure 
for either party. The notice period provides minimal protection 
against abrupt termination in long-term commercial relationships.

Risk Level: Medium-High
```

---

## Dataset

Built on **CUAD-DeepSeek** (`zenml/cuad-deepseek`) — a DeepSeek-R1-enriched version of the Contract Understanding Atticus Dataset:

- 510 commercial legal contracts
- 13,000+ attorney-annotated clause examples
- 41 clause categories (Anti-Assignment, Cap on Liability, IP Ownership, Governing Law, Termination for Convenience, etc.)
- Expert rationales averaging 211 words per clause

**After preprocessing:**
- 5,000 training examples (NONE-class filtered, quality-filtered)
- 1,065 validation examples
- 1,058 test examples

---

## Clause Categories & Risk Tiers

| Risk Level | Clause Types |
|---|---|
| **High** | Change of Control, Anti-Assignment, Non-Compete, Exclusivity, IP Ownership Assignment, License Grant |
| **Medium-High** | Termination for Convenience, Governing Law, Indemnification, Cap on Liability, Warranty Duration |
| **Medium** | Confidentiality, Audit Rights, Non-Disparagement, Most Favored Nation, Minimum Commitment |
| **Low** | Agreement Date, Parties, Effective Date, Expiration Date, Auto-Renewal |

---

## Model Architecture

| Component | Choice | Reason |
|---|---|---|
| Base model | Qwen2.5-1.5B-Instruct | Strong instruction-following at small scale, BF16 native |
| Quantization | 4-bit NF4 (bitsandbytes) | NormalFloat4 — optimal for normally-distributed LLM weights |
| LoRA rank | r=16, alpha=32 | Balanced capacity vs VRAM for generation task |
| Target modules | q/k/v/o/gate/up/down proj | All attention + MLP layers |
| Trainable params | ~18M (1.16% of 1.55B) | Base model fully frozen |
| Training VRAM | ~4.5GB | Runs on RTX 4070 Laptop 8GB |

---

## Training Configuration

```python
epochs          = 5  (early stopping at epoch ~3.2, eval_loss=1.0118)
learning_rate   = 1e-4  (cosine decay + 5% warmup)
batch_size      = 1  (grad_accum=8 → effective batch=8)
max_length      = 1024
optimizer       = paged_adamw_8bit
precision       = BFloat16
eval_steps      = 100
save_steps      = 200
```

Training stopped at epoch 3.2 when eval loss began increasing for two consecutive evaluations (1.0118 → 1.0176 → 1.0177). Best checkpoint saved automatically via `load_best_model_at_end=True`.

---

## Data Engineering Pipeline

Raw CUAD-DeepSeek required significant preprocessing:

1. **NONE-class filtering** — dropped all 18,833 NONE examples (57.7% of raw data) to ensure every training example contains substantive legal reasoning
2. **Rationale cleaning** — 10-stage normalization removing DeepSeek prompt artifacts, markdown formatting, numbered lists, bullet points, and quoted clause echoes
3. **Output capping** — rationales truncated to 150 words to control training sequence length
4. **Risk annotation** — all 41 clause types mapped to Low / Medium / Medium-High / High risk tiers
5. **Instruction formatting** — converted to Qwen2.5 ChatML format with structured three-section output

---

## Evaluation Metrics

| Metric | Why It Was Chosen |
|---|---|
| **Clause type accuracy** | Hard classification signal — exact match against 41 ground-truth labels |
| **Risk level accuracy** | Most practically useful for production — getting risk wrong is a real-world failure |
| **ROUGE-L** | Longest common subsequence overlap — better than ROUGE-1/2 for legal prose |
| **BERTScore F1** | Semantic similarity using DistilBERT — catches paraphrasing ROUGE misses |
| **Output completeness** | Production sanity check — did the model generate all three sections |

**Note on BERTScore:** The base model scored 0.769 BERTScore despite 1% accuracy by generating verbose, legally-fluent but structurally incorrect text (226 avg words). LexIQ scored 0.848 with concise, accurate output (155 avg words). This confirms BERTScore alone is insufficient for structured generation tasks — task-specific metrics are the ground truth.

---

## Repository Structure

```
├── download_cuad.py          # Download CUAD-DeepSeek from HuggingFace
├── format_cuad.py            # Preprocess + format as instruction-tuning JSONL
├── finetune_cuad.py          # QLoRA fine-tuning script
├── evaluate_cuad.py          # Full evaluation with base model comparison
├── predict_cuad.py           # Inference script (interactive + single clause)
├── finetune_data/
│   └── cuad/
│       ├── cuad_train_chat.jsonl       # 5,000 training examples
│       ├── cuad_validation_chat.jsonl  # 1,065 validation examples
│       └── cuad_test_chat.jsonl        # 1,058 test examples
└── legal_qwen_lora/
    ├── adapter_config.json             # LoRA architecture config
    ├── adapter_model.safetensors       # Trained adapter weights (~60MB)
    └── train_metrics.json             # Training loss, runtime, steps
```

---

## Setup & Usage

### Install dependencies
```bash
pip install transformers trl peft bitsandbytes accelerate torch
pip install rouge-score bert-score datasets scikit-learn
```

### Download and format data
```bash
python download_cuad.py
python format_cuad.py --data_dir raw/cuad --output finetune_data/cuad
```

### Train
```bash
python finetune_cuad.py \
  --data finetune_data/cuad/cuad_train_chat.jsonl \
  --val_data finetune_data/cuad/cuad_validation_chat.jsonl \
  --output ./legal_qwen_lora
```

### Evaluate
```bash
# Full evaluation with base model comparison
python evaluate_cuad.py \
  --adapter ./legal_qwen_lora \
  --test_data finetune_data/cuad/cuad_test_chat.jsonl \
  --compare_base

# Quick test on 100 samples
python evaluate_cuad.py \
  --adapter ./legal_qwen_lora \
  --test_data finetune_data/cuad/cuad_test_chat.jsonl \
  --max_samples 100
```

### Interactive inference
```bash
python predict_cuad.py --adapter ./legal_qwen_lora --interactive

# Compare fine-tuned vs base model
python predict_cuad.py \
  --adapter ./legal_qwen_lora \
  --clause "Either party may terminate this Agreement upon 30 days written notice." \
  --compare
```

---

## Hardware Requirements

| Component | Minimum | Used |
|---|---|---|
| GPU VRAM | 6GB | RTX 4070 Laptop 8GB |
| RAM | 16GB | 32GB |
| Storage | 10GB | ~15GB (model + data + checkpoints) |
| Training time | — | ~1h 20m (5 epochs, early stop at 3.2) |

Fully offline — no cloud API required at training or inference time.

---

## Key Findings

**1. Data quality beats data quantity.** 5,000 clean filtered examples with rich rationales produced better results than 26,000 raw noisy examples would have.

**2. NONE-class filtering is critical.** The raw dataset is 57.7% NONE examples. Including them teaches the model to hedge rather than commit to a clause type — hurting precision on the 41 actual categories.

**3. Early stopping matters more than epoch count.** The model peaked at epoch 3.2 and began overfitting. Monitoring eval loss per checkpoint is non-negotiable for generation tasks.

**4. BERTScore is misleading for structured generation.** A verbose, incoherent base model can outscore a precise, accurate fine-tuned model on BERTScore. Always pair embedding-based metrics with task-specific accuracy metrics.

---

## Tech Stack

```
Fine-tuning  : HuggingFace Transformers, TRL (SFTTrainer), PEFT
Quantization : bitsandbytes (4-bit NF4 + double quantization)
Base model   : Qwen/Qwen2.5-1.5B-Instruct
Evaluation   : rouge-score, bert-score, scikit-learn
Dataset      : zenml/cuad-deepseek (HuggingFace)
Hardware     : NVIDIA RTX 4070 Laptop GPU (8GB VRAM)
```

---

## References

- [CUAD: An Expert-Annotated NLP Dataset for Legal Contracts](https://arxiv.org/abs/2103.06268)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675)
