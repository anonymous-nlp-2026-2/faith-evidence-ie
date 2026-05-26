# Evidence at What Cost? Decoupling and Scaling in LLM-Based Document Relation Extraction

Code for the paper "Evidence at What Cost? Decoupling and Scaling in LLM-Based Document Relation Extraction" (under review at EMNLP 2026).

## Abstract

LLM-based Document-level Relation Extraction (DocRE) now matches or exceeds encoder baselines in extraction accuracy, yet outputs bare relation triplets without identifying supporting sentences. Encoder-based evidence methods (SAIS, DREEAM) provide evidence but trail LLM approaches in accuracy, creating a gap between extraction performance and verifiability. We study evidence-grounded DocRE as structured generation and compare Group Relative Policy Optimization (GRPO), Direct Preference Optimization (DPO), and Rejection Sampling Fine-Tuning (RSFT) across three fully evaluated scales (1.7B, 4B, 8B) with 14B validation, two model families, and two datasets. The evidence tax—the extraction cost of generating evidence—is scale-dependent: beneficial at 1.7B (−5.63pp Ign-F1) but approximately neutral at ≥4B, challenging the assumption that evidence always harms accuracy. Gradient-based optimization amplifies evidence-extraction competition (GRPO catastrophic, DPO gradual), while RSFT avoids decoupling because discrete output selection structurally cannot exploit the extraction-evidence trade-off. We define the Evidence-Decoupling Ratio (EDCR) to quantify this misalignment, validated at per-relation level (ρ=−0.79, p<0.001). Our 1.7B RSFT model approaches 8B SFT performance at 4.7× fewer parameters, and test-set evaluation preserves model ranking across all scales.

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA support
- See `requirements.txt` for full dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Data

This project uses [DocRED](https://github.com/thunlp/DocRED) and [Re-DocRED](https://github.com/tonytan48/Re-DocRED) datasets.
Download and place in `data/` directory.

## Usage

### SFT Training

```bash
python -m training.sft_trainer \
    --model_name Qwen/Qwen3-4B \
    --data_dir data/docred \
    --output_dir outputs/sft
```

Multi-GPU with DeepSpeed:
```bash
accelerate launch --config_file configs/accelerate_4gpu.yaml \
    -m training.sft_trainer \
    --deepspeed configs/deepspeed_zero2.json \
    --output_dir outputs/sft
```

### RSFT with CED Scoring

```bash
# Step 1: Generate candidates from SFT model
python -m training.rsft_generate \
    --model_path outputs/sft \
    --data_dir data/docred \
    --num_generations 8

# Step 2: CED scoring and filtering
python -m training.rsft_score_filter \
    --input_dir outputs/rsft_candidates \
    --scoring_mode ced \
    --selection_strategy top_k --top_k 1

# Step 3: RSFT training
python -m training.rsft_trainer \
    --sft_adapter outputs/sft \
    --rsft_data_path outputs/rsft_scored \
    --output_dir outputs/rsft
```

### GRPO Training

```bash
python -m training.grpo_trainer \
    --sft_adapter outputs/sft \
    --data_dir data/docred \
    --output_dir outputs/grpo
```

### DPO Training

```bash
python -m training.dpo_trainer \
    --sft_adapter outputs/sft \
    --data_dir data/docred \
    --output_dir outputs/dpo
```

### Evaluation

```bash
python -m eval.inference \
    --model_path outputs/rsft \
    --data_dir data/docred \
    --split dev \
    --output_dir eval_outputs/rsft_dev
```

## Project Structure

```
├── analysis/      # CED signal validation and analysis
├── configs/       # DeepSpeed and Accelerate configs
├── data/          # DocRED data processor
├── eval/          # Inference, evaluation, CED reranking
├── rewards/       # CED reward computation
├── scripts/       # Training and evaluation shell scripts
├── tests/         # Unit tests
└── training/      # SFT, RSFT, GRPO, DPO trainers
```
