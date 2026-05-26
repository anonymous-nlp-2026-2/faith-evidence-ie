import json
import os
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "./scripts/analysis/results"

PAIRS = {
    "Qwen3-1.7B": {
        "with": "eval_results/qwen3_1_7b_sft_eval/predictions.json",
        "without": "eval_results/qwen3_1_7b_no_evidence_eval_v3/predictions.json",
    },
    "Qwen3-4B": {
        "with": "eval_results/d111_4b_sft_s42_reeval/predictions.json",
        "without": "eval_results/plan_014_no_evidence_eval_d076/predictions.json",
    },
    "Qwen3-8B": {
        "with": "eval_results/qwen3_8b_sft_eval/predictions.json",
        "without": "eval_results/qwen3_8b_no_evidence_eval_v2/predictions.json",
    },
    "Llama-3.1-8B": {
        "with": "eval_results/llama_3_1_8b_sft_eval/predictions.json",
        "without": "eval_results/llama_3_1_8b_no_evidence_eval/predictions.json",
    },
}

METRICS = {
    "Qwen3-1.7B": {
        "with": "eval_results/qwen3_1_7b_sft_eval/metrics.json",
        "without": "eval_results/qwen3_1_7b_no_evidence_eval_v3/metrics.json",
    },
    "Qwen3-4B": {
        "with": "eval_results/d111_4b_sft_s42_reeval/metrics.json",
        "without": "eval_results/plan_014_no_evidence_eval_d076/metrics.json",
    },
    "Qwen3-8B": {
        "with": "eval_results/qwen3_8b_sft_eval/metrics.json",
        "without": "eval_results/qwen3_8b_no_evidence_eval_v2/metrics.json",
    },
    "Llama-3.1-8B": {
        "with": "eval_results/llama_3_1_8b_sft_eval/metrics.json",
        "without": "eval_results/llama_3_1_8b_no_evidence_eval/metrics.json",
    },
}

def load_json(path):
    with open(path) as f:
        return json.load(f)

def compute_fp_fn(pred_triples, gold_triples):
    pred_set = set((t["head"], t["relation"], t["tail"]) for t in pred_triples)
    gold_set = set((t["head"], t["relation"], t["tail"]) for t in gold_triples)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn

scales = list(PAIRS.keys())

# =========================================================
# Fig 1: Evidence Tax overview (F1 + truncation + duplicates)
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: F1 comparison
f1_with = [load_json(METRICS[s]["with"])["f1"] for s in scales]
f1_no = [load_json(METRICS[s]["without"])["f1"] for s in scales]
x = np.arange(len(scales))
w = 0.35
axes[0].bar(x - w/2, f1_with, w, label="With Evidence", color="#4C72B0")
axes[0].bar(x + w/2, f1_no, w, label="No Evidence", color="#DD8452")
axes[0].set_ylabel("F1 Score")
axes[0].set_title("(a) RE F1 by Scale")
axes[0].set_xticks(x)
axes[0].set_xticklabels(scales, fontsize=8)
axes[0].legend(fontsize=8)
axes[0].set_ylim(0.3, 0.6)
for i in range(len(scales)):
    tax = (f1_with[i] - f1_no[i]) * 100
    axes[0].annotate(f"{tax:+.1f}pp", (x[i], max(f1_with[i], f1_no[i]) + 0.005),
                     ha="center", fontsize=7, color="red" if tax < 0 else "green")

# Panel B: Truncation rate
trunc_with = []
trunc_no = []
for s in scales:
    pw = load_json(PAIRS[s]["with"])
    pn = load_json(PAIRS[s]["without"])
    trunc_with.append(sum(1 for d in pw if d["truncated"]) / len(pw) * 100)
    trunc_no.append(sum(1 for d in pn if d["truncated"]) / len(pn) * 100)
axes[1].bar(x - w/2, trunc_with, w, label="With Evidence", color="#4C72B0")
axes[1].bar(x + w/2, trunc_no, w, label="No Evidence", color="#DD8452")
axes[1].set_ylabel("Truncation Rate (%)")
axes[1].set_title("(b) Output Truncation Rate")
axes[1].set_xticks(x)
axes[1].set_xticklabels(scales, fontsize=8)
axes[1].legend(fontsize=8)

# Panel C: Duplicate rate
dup_with = []
dup_no = []
for s in scales:
    pw = load_json(PAIRS[s]["with"])
    pn = load_json(PAIRS[s]["without"])
    for label, preds, lst in [("w", pw, dup_with), ("n", pn, dup_no)]:
        from collections import Counter
        total_all, total_dup = 0, 0
        for d in preds:
            seen = Counter((t["head"], t["relation"], t["tail"]) for t in d["parsed_triples"])
            total_all += sum(seen.values())
            total_dup += sum(seen.values()) - len(seen)
        lst.append(total_dup / total_all * 100 if total_all > 0 else 0)

axes[2].bar(x - w/2, dup_with, w, label="With Evidence", color="#4C72B0")
axes[2].bar(x + w/2, dup_no, w, label="No Evidence", color="#DD8452")
axes[2].set_ylabel("Duplicate Triple Rate (%)")
axes[2].set_title("(c) Duplicate Triples")
axes[2].set_xticks(x)
axes[2].set_xticklabels(scales, fontsize=8)
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "evidence_tax_overview.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved evidence_tax_overview.png")

# =========================================================
# Fig 2: Output length distributions (violin/box plots)
# =========================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, scale in enumerate(scales):
    ax = axes[idx // 2][idx % 2]
    pw = load_json(PAIRS[scale]["with"])
    pn = load_json(PAIRS[scale]["without"])
    lens_w = [len(d["raw_output"]) for d in pw]
    lens_n = [len(d["raw_output"]) for d in pn]
    
    bp = ax.boxplot([lens_w, lens_n], labels=["With Evidence", "No Evidence"],
                    patch_artist=True, widths=0.6)
    bp["boxes"][0].set_facecolor("#4C72B0")
    bp["boxes"][1].set_facecolor("#DD8452")
    for b in bp["boxes"]:
        b.set_alpha(0.7)
    
    ax.set_title(f"{scale}")
    ax.set_ylabel("Output Length (chars)")
    
    tw = sum(1 for d in pw if d["truncated"])
    tn = sum(1 for d in pn if d["truncated"])
    ax.text(0.02, 0.98, f"Trunc: {tw/len(pw):.1%} vs {tn/len(pn):.1%}",
            transform=ax.transAxes, va="top", fontsize=8, color="gray")

plt.suptitle("Output Length Distribution: With vs Without Evidence", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "output_length_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved output_length_distributions.png")

# =========================================================
# Fig 3: Precision-Recall scatter + FP analysis
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Precision vs Recall
for s in scales:
    mw = load_json(METRICS[s]["with"])
    mn = load_json(METRICS[s]["without"])
    axes[0].scatter(mw["recall"], mw["precision"], marker="o", s=100, zorder=5, label=f"{s} with-ev")
    axes[0].scatter(mn["recall"], mn["precision"], marker="^", s=100, zorder=5, label=f"{s} no-ev")
    axes[0].annotate("", xy=(mn["recall"], mn["precision"]),
                     xytext=(mw["recall"], mw["precision"]),
                     arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")
axes[0].set_title("(a) Precision vs Recall")
axes[0].legend(fontsize=6, ncol=2)

# Panel B: FP rate comparison
fp_with = []
fp_no = []
for s in scales:
    pw = load_json(PAIRS[s]["with"])
    pn = load_json(PAIRS[s]["without"])
    for preds, lst in [(pw, fp_with), (pn, fp_no)]:
        total_tp, total_fp = 0, 0
        for d in preds:
            tp, fp, fn = compute_fp_fn(d["parsed_triples"], d["gold_triples"])
            total_tp += tp; total_fp += fp
        lst.append(total_fp / (total_tp + total_fp) * 100 if (total_tp + total_fp) > 0 else 0)

axes[1].bar(x - w/2, fp_with, w, label="With Evidence", color="#4C72B0")
axes[1].bar(x + w/2, fp_no, w, label="No Evidence", color="#DD8452")
axes[1].set_ylabel("False Positive Rate (%)")
axes[1].set_title("(b) False Positive Rate by Scale")
axes[1].set_xticks(x)
axes[1].set_xticklabels(scales, fontsize=8)
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "precision_recall_fp.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved precision_recall_fp.png")

# =========================================================
# Fig 4: Evidence overhead breakdown
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Chars per triple
costs_w_all = []
costs_n_all = []
for s in scales:
    pw = load_json(PAIRS[s]["with"])
    pn = load_json(PAIRS[s]["without"])
    costs_w = []
    costs_n = []
    for d in pw:
        try:
            triples = json.loads(d["raw_output"])
            if isinstance(triples, list) and len(triples) > 0:
                costs_w.append(len(d["raw_output"]) / len(triples))
        except: pass
    for d in pn:
        try:
            triples = json.loads(d["raw_output"])
            if isinstance(triples, list) and len(triples) > 0:
                costs_n.append(len(d["raw_output"]) / len(triples))
        except: pass
    costs_w_all.append(statistics.mean(costs_w) if costs_w else 0)
    costs_n_all.append(statistics.mean(costs_n) if costs_n else 0)

axes[0].bar(x - w/2, costs_w_all, w, label="With Evidence", color="#4C72B0")
axes[0].bar(x + w/2, costs_n_all, w, label="No Evidence", color="#DD8452")
axes[0].set_ylabel("Avg Chars per Triple")
axes[0].set_title("(a) Per-Triple Character Cost")
axes[0].set_xticks(x)
axes[0].set_xticklabels(scales, fontsize=8)
axes[0].legend(fontsize=8)
for i in range(len(scales)):
    if costs_n_all[i] > 0:
        ratio = costs_w_all[i] / costs_n_all[i]
        axes[0].annotate(f"{ratio:.2f}x", (x[i], max(costs_w_all[i], costs_n_all[i]) + 1),
                        ha="center", fontsize=7)

# Panel B: Evidence field overhead %
ev_pcts = []
for s in scales:
    pw = load_json(PAIRS[s]["with"])
    ev_chars_list = []
    total_chars_list = []
    for d in pw:
        try:
            triples = json.loads(d["raw_output"])
            if not isinstance(triples, list): continue
            ev_c = 0
            for t in triples:
                if "evidence" in t:
                    ev_c += len(json.dumps({"evidence": t["evidence"]})) + 2
            if len(d["raw_output"]) > 0:
                ev_chars_list.append(ev_c)
                total_chars_list.append(len(d["raw_output"]))
        except: pass
    mean_ev = statistics.mean(ev_chars_list) if ev_chars_list else 0
    mean_total = statistics.mean(total_chars_list) if total_chars_list else 1
    ev_pcts.append(mean_ev / mean_total * 100)

axes[1].bar(x, ev_pcts, 0.5, color="#4C72B0")
axes[1].set_ylabel("Evidence Overhead (%)")
axes[1].set_title("(b) Evidence Field % of Total Output")
axes[1].set_xticks(x)
axes[1].set_xticklabels(scales, fontsize=8)
for i in range(len(scales)):
    axes[1].annotate(f"{ev_pcts[i]:.1f}%", (x[i], ev_pcts[i] + 0.3), ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "evidence_overhead.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved evidence_overhead.png")

# =========================================================
# Fig 5: Per-doc paired analysis - F1 difference histogram
# =========================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, scale in enumerate(scales):
    ax = axes[idx // 2][idx % 2]
    pw = load_json(PAIRS[scale]["with"])
    pn = load_json(PAIRS[scale]["without"])
    with_by_id = {d["doc_id"]: d for d in pw}
    no_by_id = {d["doc_id"]: d for d in pn}
    
    diffs = []
    for doc_id in set(with_by_id.keys()) & set(no_by_id.keys()):
        dw = with_by_id[doc_id]
        dn = no_by_id[doc_id]
        tpw, fpw, fnw = compute_fp_fn(dw["parsed_triples"], dw["gold_triples"])
        tpn, fpn, fnn = compute_fp_fn(dn["parsed_triples"], dn["gold_triples"])
        f1w = 2*tpw/(2*tpw+fpw+fnw) if (2*tpw+fpw+fnw) > 0 else 0
        f1n = 2*tpn/(2*tpn+fpn+fnn) if (2*tpn+fpn+fnn) > 0 else 0
        diffs.append(f1w - f1n)
    
    ax.hist(diffs, bins=50, color="#4C72B0", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    mean_diff = statistics.mean(diffs)
    ax.axvline(mean_diff, color="green", linestyle="-", linewidth=1.5)
    ax.set_title(f"{scale} (mean diff = {mean_diff:+.3f})")
    ax.set_xlabel("F1(with-ev) - F1(no-ev)")
    ax.set_ylabel("Count")
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    ax.text(0.02, 0.95, f"with>no: {pos}, no>with: {neg}",
            transform=ax.transAxes, va="top", fontsize=8)

plt.suptitle("Per-Document F1 Difference: With Evidence minus No Evidence", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "per_doc_f1_diff.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved per_doc_f1_diff.png")

print("\nAll plots saved to", RESULTS_DIR)
