"""Analyze token length distribution of SFT training data for 14B max_length decision."""
import json
import sys
import numpy as np

sys.path.insert(0, "/workspace/freige")

from transformers import AutoTokenizer
from freige.data.docred_processor import DocREDProcessor
from freige.training.sft_trainer import build_chat_messages, SYSTEM_PROMPT

tokenizer = AutoTokenizer.from_pretrained(
    "/workspace/models/Qwen/Qwen3-14B",
    trust_remote_code=True,
)

processor = DocREDProcessor(data_dir="/workspace/data/docred")
samples = processor.process("train")
groups = processor.group_by_document(samples)

lengths = []
doc_info = []  # (doc_id, token_len, n_sents, n_relations)

for doc_id, doc_samples in groups.items():
    sft = processor.format_sft_sample(doc_samples)
    msgs = build_chat_messages(sft["instruction"], sft["input"], sft["output"])
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    toks = tokenizer.encode(text)
    tlen = len(toks)
    lengths.append(tlen)
    doc_info.append((doc_id, tlen, len(doc_samples[0].sents), len(doc_samples)))

lengths = np.array(lengths)

print(f"=== Token 长度分布 ===")
print(f"样本数: {len(lengths)}")
print(f"mean: {lengths.mean():.0f}, median: {np.median(lengths):.0f}")
print(f"p90: {np.percentile(lengths, 90):.0f}, p95: {np.percentile(lengths, 95):.0f}")
print(f"p99: {np.percentile(lengths, 99):.0f}, max: {lengths.max()}")

print(f"\n=== 截断比例 ===")
for threshold in [2048, 3072, 4096]:
    n_trunc = (lengths > threshold).sum()
    pct = n_trunc / len(lengths) * 100
    print(f"max_length={threshold}: {pct:.1f}% 截断 ({n_trunc}/{len(lengths)} 样本)")

# Analyze truncated samples at 2048
trunc_2048 = [d for d in doc_info if d[1] > 2048]
if trunc_2048:
    trunc_2048.sort(key=lambda x: -x[1])
    print(f"\n=== 被 2048 截断的样本特征 ===")
    t_lens = [d[1] for d in trunc_2048]
    t_sents = [d[2] for d in trunc_2048]
    t_rels = [d[3] for d in trunc_2048]
    print(f"token 长度 mean={np.mean(t_lens):.0f}, max={max(t_lens)}")
    print(f"句子数 mean={np.mean(t_sents):.1f}, max={max(t_sents)}")
    print(f"关系数 mean={np.mean(t_rels):.1f}, max={max(t_rels)}")
    
    # Non-truncated for comparison
    non_trunc = [d for d in doc_info if d[1] <= 2048]
    nt_sents = [d[2] for d in non_trunc]
    nt_rels = [d[3] for d in non_trunc]
    print(f"\n未截断样本对比:")
    print(f"句子数 mean={np.mean(nt_sents):.1f}")
    print(f"关系数 mean={np.mean(nt_rels):.1f}")

    print(f"\n最长 5 个样本:")
    for doc_id, tlen, nsents, nrels in trunc_2048[:5]:
        print(f"  {doc_id}: {tlen} tokens, {nsents} sents, {nrels} relations")

# Distribution histogram
bins = [0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 99999]
labels = ["0-512", "512-1024", "1024-1536", "1536-2048", "2048-2560", "2560-3072", "3072-3584", "3584-4096", "4096+"]
hist, _ = np.histogram(lengths, bins=bins)
print(f"\n=== 分布直方图 ===")
for label, count in zip(labels, hist):
    pct = count / len(lengths) * 100
    bar = "#" * int(pct)
    print(f"{label:>10s}: {count:4d} ({pct:5.1f}%) {bar}")
