# FREIGE: Faithful and Robust Evidence-Grounded Information Extraction

Code for the paper (under review at EMNLP 2026).

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
python -m freige.training.sft_trainer \
    --model_name Qwen/Qwen3-4B \
    --data_dir data/docred \
    --output_dir outputs/sft
```

Multi-GPU with DeepSpeed:
```bash
accelerate launch --config_file configs/accelerate_4gpu.yaml \
    -m freige.training.sft_trainer \
    --deepspeed configs/deepspeed_zero2.json \
    --output_dir outputs/sft
```

### RSFT with CED Scoring

```bash
# Step 1: Generate candidates from SFT model
python -m freige.training.rsft_generate \
    --model_path outputs/sft \
    --data_dir data/docred \
    --num_generations 8

# Step 2: CED scoring and filtering
python -m freige.training.rsft_score_filter \
    --input_dir outputs/rsft_candidates \
    --scoring_mode ced \
    --selection_strategy top_k --top_k 1

# Step 3: RSFT training
python -m freige.training.rsft_trainer \
    --sft_adapter outputs/sft \
    --rsft_data_path outputs/rsft_scored \
    --output_dir outputs/rsft
```

### GRPO Training

```bash
python -m freige.training.grpo_trainer \
    --sft_adapter outputs/sft \
    --data_dir data/docred \
    --output_dir outputs/grpo
```

### Evaluation

```bash
python -m freige.eval.inference \
    --model_path outputs/rsft \
    --data_dir data/docred \
    --split dev \
    --output_dir eval_outputs/rsft_dev
```

## Project Structure

```
freige/
├── analysis/      # CED signal validation and analysis
├── configs/       # DeepSpeed and Accelerate configs
├── data/          # DocRED data processor
├── eval/          # Inference, evaluation, CED reranking
├── rewards/       # CED reward computation
├── scripts/       # Training and evaluation shell scripts
├── tests/         # Unit tests
└── training/      # SFT, RSFT, GRPO, DPO trainers
```
