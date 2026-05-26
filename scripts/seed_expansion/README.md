# Seed Expansion Scripts

## Training (5 jobs)
All use identical hyperparams to D102 k=1 runs, only seed and output_dir differ.

| Script | Model | Seed | Data | ~Time |
|--------|-------|------|------|-------|
| train_1_7b_s43.sh | Qwen3-1.7B | 43 | 3021 samples | ~15min |
| train_1_7b_s44.sh | Qwen3-1.7B | 44 | 3021 samples | ~15min |
| train_8b_s44.sh | Qwen3-8B | 44 | 3018 samples | ~48min |
| train_llama_s43.sh | LLaMA-3.1-8B | 43 | 2952 samples | ~43min |
| train_llama_s44.sh | LLaMA-3.1-8B | 44 | 2952 samples | ~43min |

## Usage
```bash
# Single GPU:
GPU=2 bash scripts/seed_expansion/train_1_7b_s43.sh

# Eval after training:
GPU=2 MODEL_PATH=rsft_output_qwen3_1_7b_k1_s43 \
  BASE_MODEL=Qwen/Qwen3-1.7B \
  SFT_ADAPTER=sft_output_qwen3_1_7b_bf16 \
  OUTPUT_DIR=eval_results/d102_1_7b_k1_s43 \
  SEED=43 \
  bash scripts/seed_expansion/eval_template.sh
```

## C001/C002 constraints
- Check nvidia-smi before eval (GPU must have <2GB used)
- CUDA_VISIBLE_DEVICES is set in each script via GPU env var
