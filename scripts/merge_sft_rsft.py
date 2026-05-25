"""Pre-merge base + SFT + RSFT for GRPO training."""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/workspace/models/Qwen3-4B"
SFT = "/workspace/sft_output"
RSFT = "/workspace/rsft_output_ced_s43/checkpoint-510"
OUT = "/workspace/merged_sft_rsft_ced_s43"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cpu")

print("Merging SFT adapter...")
model = PeftModel.from_pretrained(model, SFT)
model = model.merge_and_unload()

print("Merging RSFT adapter...")
model = PeftModel.from_pretrained(model, RSFT)
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained(OUT)

print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(SFT)
tokenizer.save_pretrained(OUT)

print(f"Merge complete -> {OUT}")
