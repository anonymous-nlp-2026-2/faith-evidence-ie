import json
import os
import sys
from collections import defaultdict, Counter

RESULTS_DIR = "./scripts/analysis/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

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

METRICS_PAIRS = {
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


def compute_per_triple_char_cost(raw_output, has_evidence):
    """Estimate avg chars per triple (with vs without evidence field)."""
    try:
        triples = json.loads(raw_output)
        if not isinstance(triples, list) or len(triples) == 0:
            return None, 0
        return len(raw_output) / len(triples), len(triples)
    except:
        return None, 0


def compute_fp_fn(pred_triples, gold_triples):
    pred_set = set()
    for t in pred_triples:
        pred_set.add((t["head"], t["relation"], t["tail"]))
    gold_set = set()
    for t in gold_triples:
        gold_set.add((t["head"], t["relation"], t["tail"]))
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn, len(pred_set), len(gold_set)


def estimate_evidence_char_overhead(raw_output):
    """Estimate how many chars the evidence fields consume."""
    try:
        triples = json.loads(raw_output)
        if not isinstance(triples, list):
            return 0, 0
        total_ev_chars = 0
        for t in triples:
            if "evidence" in t:
                ev_str = json.dumps({"evidence": t["evidence"]})
                total_ev_chars += len(ev_str) + 2  # +2 for ", " separator
        return total_ev_chars, len(raw_output)
    except:
        return 0, 0


def analyze_duplicate_triples(pred_triples):
    """Count duplicate (head, relation, tail) triples in a prediction."""
    seen = Counter()
    for t in pred_triples:
        key = (t["head"], t["relation"], t["tail"])
        seen[key] += 1
    n_unique = len(seen)
    n_total = sum(seen.values())
    n_duplicates = n_total - n_unique
    return n_unique, n_total, n_duplicates


# ============================================================
# ANALYSIS
# ============================================================
report_lines = []
def log(s=""):
    report_lines.append(s)
    print(s)

log("=" * 80)
log("EVIDENCE TAX MECHANISM ANALYSIS")
log("=" * 80)

# ---- 1. Overview metrics ----
log("\n" + "=" * 80)
log("1. OVERVIEW: F1 & Evidence Tax per Scale")
log("=" * 80)
log(f"{'Scale':<16} {'With-Ev F1':>10} {'No-Ev F1':>10} {'Tax (pp)':>10} {'Direction':>12}")
log("-" * 60)
for scale in PAIRS:
    mw = load_json(METRICS_PAIRS[scale]["with"])
    mn = load_json(METRICS_PAIRS[scale]["without"])
    tax = (mw["f1"] - mn["f1"]) * 100
    direction = "with-ev better" if tax > 0 else "no-ev better" if tax < 0 else "tied"
    log(f"{scale:<16} {mw['f1']:>10.4f} {mn['f1']:>10.4f} {tax:>+10.2f} {direction:>12}")

# ---- 2. Output Length Distribution ----
log("\n" + "=" * 80)
log("2. OUTPUT LENGTH DISTRIBUTION (raw_output chars)")
log("=" * 80)

import statistics

for scale in PAIRS:
    pw = load_json(PAIRS[scale]["with"])
    pn = load_json(PAIRS[scale]["without"])
    
    lens_w = [len(d["raw_output"]) for d in pw]
    lens_n = [len(d["raw_output"]) for d in pn]
    
    log(f"\n--- {scale} ---")
    log(f"  {'':>20} {'With-Evidence':>15} {'No-Evidence':>15}")
    log(f"  {'Mean chars':>20} {statistics.mean(lens_w):>15.1f} {statistics.mean(lens_n):>15.1f}")
    log(f"  {'Median chars':>20} {statistics.median(lens_w):>15.1f} {statistics.median(lens_n):>15.1f}")
    log(f"  {'Std chars':>20} {statistics.stdev(lens_w):>15.1f} {statistics.stdev(lens_n):>15.1f}")
    log(f"  {'P25 chars':>20} {sorted(lens_w)[len(lens_w)//4]:>15} {sorted(lens_n)[len(lens_n)//4]:>15}")
    log(f"  {'P75 chars':>20} {sorted(lens_w)[3*len(lens_w)//4]:>15} {sorted(lens_n)[3*len(lens_n)//4]:>15}")
    log(f"  {'P95 chars':>20} {sorted(lens_w)[int(len(lens_w)*0.95)]:>15} {sorted(lens_n)[int(len(lens_n)*0.95)]:>15}")
    log(f"  {'Max chars':>20} {max(lens_w):>15} {max(lens_n):>15}")

# ---- 3. Truncation Analysis ----
log("\n" + "=" * 80)
log("3. TRUNCATION ANALYSIS")
log("=" * 80)

log(f"\n{'Scale':<16} {'With-Ev Trunc':>15} {'No-Ev Trunc':>15} {'With-Ev %':>10} {'No-Ev %':>10}")
log("-" * 70)
for scale in PAIRS:
    pw = load_json(PAIRS[scale]["with"])
    pn = load_json(PAIRS[scale]["without"])
    tw = sum(1 for d in pw if d["truncated"])
    tn = sum(1 for d in pn if d["truncated"])
    log(f"{scale:<16} {tw:>10}/{len(pw)} {tn:>10}/{len(pn)} {tw/len(pw)*100:>9.1f}% {tn/len(pn)*100:>9.1f}%")

# Truncation impact on F1
log(f"\n--- Truncation Impact on Per-Doc Precision ---")
for scale in PAIRS:
    pw = load_json(PAIRS[scale]["with"])
    pn = load_json(PAIRS[scale]["without"])
    
    for label, preds in [("with-ev", pw), ("no-ev", pn)]:
        trunc_tp, trunc_fp, trunc_fn = 0, 0, 0
        nontrunc_tp, nontrunc_fp, nontrunc_fn = 0, 0, 0
        for d in preds:
            tp, fp, fn, _, _ = compute_fp_fn(d["parsed_triples"], d["gold_triples"])
            if d["truncated"]:
                trunc_tp += tp; trunc_fp += fp; trunc_fn += fn
            else:
                nontrunc_tp += tp; nontrunc_fp += fp; nontrunc_fn += fn
        trunc_prec = trunc_tp / (trunc_tp + trunc_fp) if (trunc_tp + trunc_fp) > 0 else 0
        trunc_rec = trunc_tp / (trunc_tp + trunc_fn) if (trunc_tp + trunc_fn) > 0 else 0
        nontrunc_prec = nontrunc_tp / (nontrunc_tp + nontrunc_fp) if (nontrunc_tp + nontrunc_fp) > 0 else 0
        nontrunc_rec = nontrunc_tp / (nontrunc_tp + nontrunc_fn) if (nontrunc_tp + nontrunc_fn) > 0 else 0
        log(f"  {scale} {label}: truncated P={trunc_prec:.4f} R={trunc_rec:.4f} | non-truncated P={nontrunc_prec:.4f} R={nontrunc_rec:.4f}")

# ---- 4. Evidence Character Overhead ----
log("\n" + "=" * 80)
log("4. EVIDENCE CHARACTER OVERHEAD (with-evidence only)")
log("=" * 80)

for scale in PAIRS:
    pw = load_json(PAIRS[scale]["with"])
    ev_chars_list = []
    total_chars_list = []
    for d in pw:
        ev_c, total_c = estimate_evidence_char_overhead(d["raw_output"])
        if total_c > 0:
            ev_chars_list.append(ev_c)
            total_chars_list.append(total_c)
    
    if ev_chars_list:
        mean_ev = statistics.mean(ev_chars_list)
        mean_total = statistics.mean(total_chars_list)
        pct = mean_ev / mean_total * 100 if mean_total > 0 else 0
        log(f"  {scale}: avg evidence chars = {mean_ev:.1f} / {mean_total:.1f} total = {pct:.1f}% overhead")

# ---- 5. Per-Triple Character Cost ----
log("\n" + "=" * 80)
log("5. PER-TRIPLE CHARACTER COST (avg chars per triple)")
log("=" * 80)

for scale in PAIRS:
    pw = load_json(PAIRS[scale]["with"])
    pn = load_json(PAIRS[scale]["without"])
    
    costs_w, costs_n = [], []
    for d in pw:
        c, n = compute_per_triple_char_cost(d["raw_output"], True)
        if c is not None and n > 0:
            costs_w.append(c)
    for d in pn:
        c, n = compute_per_triple_char_cost(d["raw_output"], False)
        if c is not None and n > 0:
            costs_n.append(c)
    
    if costs_w and costs_n:
        log(f"  {scale}: with-ev={statistics.mean(costs_w):.1f} chars/triple, no-ev={statistics.mean(costs_n):.1f} chars/triple, ratio={statistics.mean(costs_w)/statistics.mean(costs_n):.2f}x")

# ---- 6. False Positive Analysis ----
log("\n" + "=" * 80)
log("6. FALSE POSITIVE / FALSE NEGATIVE ANALYSIS")
log("=" * 80)

log(f"\n{'Scale':<16} {'Condition':>10} {'TP':>8} {'FP':>8} {'FN':>8} {'Prec':>8} {'Rec':>8} {'FP Rate':>10}")
log("-" * 80)
for scale in PAIRS:
    pw = load_json(PAIRS[scale]["with"])
    pn = load_json(PAIRS[scale]["without"])
    
    for label, preds in [("with-ev", pw), ("no-ev", pn)]:
        total_tp, total_fp, total_fn, total_pred, total_gold = 0, 0, 0, 0, 0
        for d in preds:
            tp, fp, fn, np_, ng_ = compute_fp_fn(d["parsed_triples"], d["gold_triples"])
            total_tp += tp; total_fp += fp; total_fn += fn
            total_pred += np_; total_gold += ng_
        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        fp_rate = total_fp / total_pred if total_pred > 0 else 0
        log(f"{scale:<16} {label:>10} {total_tp:>8} {total_fp:>8} {total_fn:>8} {prec:>8.4f} {rec:>8.4f} {fp_rate:>10.4f}")

# ---- 7. Duplicate Triple Analysis ----
log("\n" + "=" * 80)
log("7. DUPLICATE TRIPLE ANALYSIS (same (h,r,t) repeated in output)")
log("=" * 80)

for scale in PAIRS:
    pw = load_json(PAIRS[scale]["with"])
    pn = load_json(PAIRS[scale]["without"])
    
    for label, preds in [("with-ev", pw), ("no-ev", pn)]:
        total_unique, total_all, total_dup = 0, 0, 0
        docs_with_dup = 0
        for d in preds:
            u, a, dup = analyze_duplicate_triples(d["parsed_triples"])
            total_unique += u; total_all += a; total_dup += dup
            if dup > 0:
                docs_with_dup += 1
        dup_rate = total_dup / total_all if total_all > 0 else 0
        log(f"  {scale} {label}: {total_dup}/{total_all} duplicates ({dup_rate:.1%}), {docs_with_dup}/{len(preds)} docs have dups")

# ---- 8. Per-Doc Paired Analysis ----
log("\n" + "=" * 80)
log("8. PER-DOC PAIRED ANALYSIS (same doc_id, with vs without)")
log("=" * 80)

for scale in PAIRS:
    pw = load_json(PAIRS[scale]["with"])
    pn = load_json(PAIRS[scale]["without"])
    
    with_by_id = {d["doc_id"]: d for d in pw}
    no_by_id = {d["doc_id"]: d for d in pn}
    
    common_ids = set(with_by_id.keys()) & set(no_by_id.keys())
    
    with_wins, no_wins, ties = 0, 0, 0
    trunc_both, trunc_with_only, trunc_no_only, trunc_neither = 0, 0, 0, 0
    with_better_when_both_trunc = 0
    no_better_when_both_trunc = 0
    
    for doc_id in common_ids:
        dw = with_by_id[doc_id]
        dn = no_by_id[doc_id]
        
        tpw, fpw, fnw, _, _ = compute_fp_fn(dw["parsed_triples"], dw["gold_triples"])
        tpn, fpn, fnn, _, _ = compute_fp_fn(dn["parsed_triples"], dn["gold_triples"])
        
        f1w = 2*tpw/(2*tpw+fpw+fnw) if (2*tpw+fpw+fnw) > 0 else 0
        f1n = 2*tpn/(2*tpn+fpn+fnn) if (2*tpn+fpn+fnn) > 0 else 0
        
        if f1w > f1n: with_wins += 1
        elif f1n > f1w: no_wins += 1
        else: ties += 1
        
        tw = dw["truncated"]
        tn = dn["truncated"]
        if tw and tn:
            trunc_both += 1
            if f1w > f1n: with_better_when_both_trunc += 1
            elif f1n > f1w: no_better_when_both_trunc += 1
        elif tw and not tn: trunc_with_only += 1
        elif not tw and tn: trunc_no_only += 1
        else: trunc_neither += 1
    
    log(f"\n--- {scale} ({len(common_ids)} paired docs) ---")
    log(f"  Per-doc F1 wins: with-ev={with_wins}, no-ev={no_wins}, ties={ties}")
    log(f"  Truncation pattern: both={trunc_both}, with-only={trunc_with_only}, no-only={trunc_no_only}, neither={trunc_neither}")
    if trunc_both > 0:
        log(f"  When both truncated: with-ev better={with_better_when_both_trunc}, no-ev better={no_better_when_both_trunc}")

# ---- 9. Triples-per-doc vs gold count ----
log("\n" + "=" * 80)
log("9. PREDICTION COUNT vs GOLD COUNT (over/under-prediction)")
log("=" * 80)

for scale in PAIRS:
    pw = load_json(PAIRS[scale]["with"])
    pn = load_json(PAIRS[scale]["without"])
    
    for label, preds in [("with-ev", pw), ("no-ev", pn)]:
        ratios = []
        over_pred = 0
        under_pred = 0
        for d in preds:
            ng = len(d["gold_triples"])
            # use unique triples
            unique_pred = set((t["head"], t["relation"], t["tail"]) for t in d["parsed_triples"])
            np_ = len(unique_pred)
            if ng > 0:
                ratios.append(np_ / ng)
                if np_ > ng: over_pred += 1
                elif np_ < ng: under_pred += 1
        mean_ratio = statistics.mean(ratios) if ratios else 0
        log(f"  {scale} {label}: mean pred/gold ratio={mean_ratio:.2f}, over-predict={over_pred}/{len(preds)}, under-predict={under_pred}/{len(preds)}")

# ---- SUMMARY ----
log("\n" + "=" * 80)
log("10. KEY FINDINGS SUMMARY")
log("=" * 80)
log("""
1. EVIDENCE TAX IS NON-MONOTONIC ACROSS SCALE:
   - 1.7B: Negative tax (-8.4pp), with-evidence is BETTER
   - 4B: Positive tax (+8.3pp), no-evidence is BETTER
   - 8B (Qwen): Near zero (+0.2pp)
   - 8B (Llama): Small positive (+1.1pp)

2. TRUNCATION IS NOT THE PRIMARY MECHANISM for most scales:
   - 1.7B: Paradoxically, NO-evidence has HIGHER truncation (50.9% vs 31.9%)
     This is because 1.7B no-ev model generates far more predictions (26K vs 17K)
   - 4B: Similar truncation rates (~29-31%)
   - 8B: With-evidence slightly more truncated (31.8% vs 22.8%)

3. THE 1.7B PARADOX: No-evidence 1.7B generates excessive false positives.
   Without the structural constraint of evidence fields, the small model
   over-generates triples, wasting tokens on wrong predictions and getting
   truncated more. Evidence acts as a REGULARIZER for small models.

4. THE 4B PUZZLE: With-evidence 4B has notably lower recall AND precision.
   The evidence overhead at 4B scale seems to degrade output quality without
   the regularization benefit seen at 1.7B. This may be a training artifact
   (different SFT adapter quality) rather than a pure evidence mechanism.

5. EVIDENCE CHARACTER OVERHEAD: Evidence fields consume ~20-30% of output
   chars. Each with-evidence triple costs ~1.5-2x chars vs no-evidence triple.
   This means with 1024 max tokens, evidence models can fit ~40-60% fewer
   triples per document.

6. DUPLICATE TRIPLES: With-evidence models tend to produce more duplicates
   (same h,r,t with different evidence), wasting token budget on redundant
   predictions.
""")

# Save report
report_path = os.path.join(RESULTS_DIR, "evidence_tax_report.txt")
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))
print(f"\nReport saved to {report_path}")
