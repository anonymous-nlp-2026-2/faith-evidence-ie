"""Evidence Tax 机制分析脚本。

分析 with-evidence vs no-evidence 模型在各尺度上的性能差异机制。
不碰 GPU，纯 CPU 分析。

Usage:
    python scripts/evidence_tax_analysis.py [--module A|B|C|D|all]
"""
import json
import argparse
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, '.')

# ============================================================
# Data paths
# ============================================================
EVAL_ROOT = Path("eval_results")
DATA_ROOT = Path("data/docred")

# SFT with-evidence vs no-evidence pairs
SFT_PAIRS = {
    "Qwen3-1.7B": {
        "with_evi": EVAL_ROOT / "qwen3_1_7b_sft_eval",
        "no_evi":   EVAL_ROOT / "qwen3_1_7b_no_evidence_eval_v3",
    },
    "Qwen3-4B": {
        "with_evi": EVAL_ROOT / "sft_baseline_d076_reeval",
        "no_evi":   EVAL_ROOT / "plan_014_no_evidence_eval_d076",
    },
    "Qwen3-8B": {
        "with_evi": EVAL_ROOT / "qwen3_8b_sft_eval",
        "no_evi":   EVAL_ROOT / "qwen3_8b_no_evidence_eval_v2",
    },
    "LLaMA-3.1-8B": {
        "with_evi": EVAL_ROOT / "llama_3_1_8b_sft_eval",
        "no_evi":   EVAL_ROOT / "llama_3_1_8b_no_evidence_eval",
    },
}

# RSFT results (with-evidence only, for context)
RSFT_RESULTS = {
    "Qwen3-1.7B RSFT-k1":  EVAL_ROOT / "d102_1_7b_k1_s42",
    "Qwen3-4B RSFT-CED":   EVAL_ROOT / "rsft_ced_s44_reeval",
    "Qwen3-8B RSFT":       EVAL_ROOT / "plan_006_qwen3_8b_rsft_eval",
    "LLaMA-8B RSFT":       EVAL_ROOT / "plan_006_llama_rsft_r3_eval",
}


def load_metrics(path):
    mf = path / "metrics.json"
    if mf.exists():
        return json.loads(mf.read_text())
    return None


def load_predictions(path):
    pf = path / "predictions.json"
    if pf.exists():
        return json.loads(pf.read_text())
    return None


def load_dev_data():
    with open(DATA_ROOT / "dev.json") as f:
        return json.load(f)


def doc_num_sentences(dev_data):
    """Map doc title -> number of sentences."""
    return {doc["title"]: len(doc["sents"]) for doc in dev_data}


def doc_num_tokens(dev_data):
    """Map doc title -> total token count (word-level from sents)."""
    return {doc["title"]: sum(len(s) for s in doc["sents"]) for doc in dev_data}


# ============================================================
# Module A: Output Length Distribution
# ============================================================
def module_a_output_length():
    """Compare output length distributions between with-evidence and no-evidence."""
    import numpy as np
    print("=" * 70)
    print("  MODULE A: OUTPUT LENGTH DISTRIBUTION")
    print("=" * 70)

    dev_data = load_dev_data()
    doc_nsent = doc_num_sentences(dev_data)
    doc_ntok = doc_num_tokens(dev_data)

    for model_name, paths in SFT_PAIRS.items():
        we_preds = load_predictions(paths["with_evi"])
        ne_preds = load_predictions(paths["no_evi"])

        if we_preds is None and ne_preds is None:
            print(f"\n  {model_name}: No predictions.json available, skipping")
            continue

        print(f"\n{'─'*70}")
        print(f"  {model_name}")
        print(f"{'─'*70}")

        for label, preds in [("With-Evidence", we_preds), ("No-Evidence", ne_preds)]:
            if preds is None:
                print(f"  {label}: No predictions.json")
                continue

            output_lens = [len(p["raw_output"]) for p in preds]
            n_triples = [len(p["parsed_triples"]) for p in preds]
            truncated = [p["truncated"] for p in preds]

            arr = np.array(output_lens)
            tri = np.array(n_triples)
            trunc_arr = np.array(truncated)

            print(f"\n  {label} ({len(preds)} docs):")
            print(f"    Output chars:  mean={arr.mean():.0f}  median={np.median(arr):.0f}  "
                  f"p90={np.percentile(arr, 90):.0f}  max={arr.max()}")
            print(f"    N triples:     mean={tri.mean():.1f}  median={np.median(tri):.0f}  "
                  f"p90={np.percentile(tri, 90):.0f}  max={tri.max()}")
            print(f"    Truncated:     {trunc_arr.sum()}/{len(preds)} ({trunc_arr.mean()*100:.1f}%)")

            # Truncated vs non-truncated output length
            trunc_lens = arr[trunc_arr]
            clean_lens = arr[~trunc_arr]
            if len(trunc_lens) > 0 and len(clean_lens) > 0:
                print(f"    Trunc chars:   mean={trunc_lens.mean():.0f}  vs  Clean: mean={clean_lens.mean():.0f}")
                print(f"    Trunc triples: mean={tri[trunc_arr].mean():.1f}  vs  Clean: mean={tri[~trunc_arr].mean():.1f}")

            # By document length (short/medium/long based on sentences)
            short, medium, long_ = [], [], []
            for p in preds:
                nsent = doc_nsent.get(p["doc_id"], 0)
                if nsent <= 6:
                    short.append(p)
                elif nsent <= 10:
                    medium.append(p)
                else:
                    long_.append(p)

            print(f"    By doc length:")
            for bucket_name, bucket in [("short (<=6 sent)", short),
                                         ("medium (7-10)", medium),
                                         ("long (>10)", long_)]:
                if not bucket:
                    continue
                b_lens = np.array([len(p["raw_output"]) for p in bucket])
                b_trunc = sum(1 for p in bucket if p["truncated"])
                b_tri = np.array([len(p["parsed_triples"]) for p in bucket])
                print(f"      {bucket_name:20s}: n={len(bucket):4d}  "
                      f"out_chars={b_lens.mean():6.0f}  "
                      f"triples={b_tri.mean():5.1f}  "
                      f"trunc={b_trunc}/{len(bucket)} ({b_trunc/len(bucket)*100:.1f}%)")


# ============================================================
# Module B: Truncation Impact
# ============================================================
def module_b_truncation_impact():
    """Analyze how truncation rate varies by model size and its impact on RE performance."""
    print("\n" + "=" * 70)
    print("  MODULE B: TRUNCATION IMPACT")
    print("=" * 70)

    # Summary table from metrics
    print(f"\n  {'Model':<20} {'Mode':<14} {'Ign-F1':>8} {'Trunc':>8} {'Trunc%':>8} {'Preds':>8}")
    print(f"  {'─'*20} {'─'*14} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for model_name, paths in SFT_PAIRS.items():
        for mode, path in paths.items():
            m = load_metrics(path)
            if m is None:
                continue
            label = "with-evi" if mode == "with_evi" else "no-evi"
            print(f"  {model_name:<20} {label:<14} {m['ign_f1']:>8.4f} "
                  f"{m['n_truncated']:>8} {m['n_truncated']/m['n_documents']*100:>7.1f}% "
                  f"{m['n_predictions']:>8}")

    # Per-document truncation analysis (where predictions available)
    print(f"\n  Per-Document Truncation Analysis (on prediction files):")
    print(f"  {'─'*70}")

    from eval.evaluator import DocREDEvaluator, gold_from_docred
    dev_data = load_dev_data()
    gold = gold_from_docred(dev_data)
    evaluator = DocREDEvaluator.from_train_file(str(DATA_ROOT / "train_annotated.json"))

    for model_name, paths in SFT_PAIRS.items():
        we_preds = load_predictions(paths["with_evi"])
        ne_preds = load_predictions(paths["no_evi"])

        for label, preds in [("with-evi", we_preds), ("no-evi", ne_preds)]:
            if preds is None:
                continue

            trunc_docs = [p for p in preds if p["truncated"]]
            clean_docs = [p for p in preds if not p["truncated"]]

            trunc_ids = set(p["doc_id"] for p in trunc_docs)
            clean_ids = set(p["doc_id"] for p in clean_docs)

            # Compute per-subset F1
            def subset_f1(doc_ids):
                ids = set(doc_ids)
                sub_preds = []
                sub_gold = []
                for p in preds:
                    if p["doc_id"] not in ids:
                        continue
                    for t in p["parsed_triples"]:
                        if isinstance(t, dict):
                            sub_preds.append({
                                "doc_id": p["doc_id"],
                                "head": t.get("head", ""),
                                "tail": t.get("tail", ""),
                                "relation": t.get("relation", ""),
                            })
                    for t in p["gold_triples"]:
                        if isinstance(t, dict):
                            sub_gold.append({
                                "doc_id": p["doc_id"],
                                "head": t.get("head", ""),
                                "tail": t.get("tail", ""),
                                "relation": t.get("relation", ""),
                            })
                return evaluator.compute_f1(sub_preds, sub_gold)

            if trunc_ids and clean_ids:
                trunc_m = subset_f1(trunc_ids)
                clean_m = subset_f1(clean_ids)
                print(f"\n  {model_name} {label}:")
                print(f"    Truncated ({len(trunc_ids)} docs): Ign-F1={trunc_m['ign_f1']:.4f}  "
                      f"P={trunc_m['precision']:.4f}  R={trunc_m['recall']:.4f}")
                print(f"    Clean ({len(clean_ids)} docs):     Ign-F1={clean_m['ign_f1']:.4f}  "
                      f"P={clean_m['precision']:.4f}  R={clean_m['recall']:.4f}")
                print(f"    Gap: {(clean_m['ign_f1'] - trunc_m['ign_f1'])*100:+.2f}pp")


# ============================================================
# Module C: Per-Relation Analysis
# ============================================================
def module_c_per_relation():
    """Compute per-relation recall for with-evi vs no-evi models."""
    print("\n" + "=" * 70)
    print("  MODULE C: PER-RELATION ANALYSIS")
    print("=" * 70)

    for model_name, paths in SFT_PAIRS.items():
        we_preds = load_predictions(paths["with_evi"])
        ne_preds = load_predictions(paths["no_evi"])
        if we_preds is None or ne_preds is None:
            print(f"\n  {model_name}: Missing predictions for one mode, skipping")
            continue

        print(f"\n{'─'*70}")
        print(f"  {model_name}")
        print(f"{'─'*70}")

        # Build gold relation counts by relation type
        gold_by_rel = defaultdict(set)  # rel -> set of (doc_id, head, tail)
        for p in we_preds:  # gold_triples same in both
            for t in p["gold_triples"]:
                if isinstance(t, dict):
                    key = (p["doc_id"], t["head"].lower().strip(),
                           t["tail"].lower().strip(), t["relation"].lower().strip())
                    gold_by_rel[t["relation"].lower().strip()].add(key)

        # Build pred sets for each mode
        def build_pred_set(preds):
            pred_by_rel = defaultdict(set)
            for p in preds:
                for t in p["parsed_triples"]:
                    if isinstance(t, dict):
                        key = (p["doc_id"], t["head"].lower().strip(),
                               t["tail"].lower().strip(), t["relation"].lower().strip())
                        pred_by_rel[t["relation"].lower().strip()].add(key)
            return pred_by_rel

        we_pred_by_rel = build_pred_set(we_preds)
        ne_pred_by_rel = build_pred_set(ne_preds)

        # Compute per-relation recall
        results = []
        for rel, gold_keys in gold_by_rel.items():
            n_gold = len(gold_keys)
            if n_gold < 5:
                continue
            we_tp = len(gold_keys & we_pred_by_rel.get(rel, set()))
            ne_tp = len(gold_keys & ne_pred_by_rel.get(rel, set()))
            we_recall = we_tp / n_gold
            ne_recall = ne_tp / n_gold
            tax = ne_recall - we_recall
            results.append((rel, n_gold, we_recall, ne_recall, tax))

        # Sort by tax (evidence tax = no_evi_recall - with_evi_recall)
        results.sort(key=lambda x: x[4], reverse=True)

        print(f"\n  {'Relation':<45} {'Gold':>5} {'WE-R':>6} {'NE-R':>6} {'Tax':>7}")
        print(f"  {'─'*45} {'─'*5} {'─'*6} {'─'*6} {'─'*7}")

        # Top 10 where no-evidence wins most (positive tax)
        print(f"\n  Top 10 relations where no-evidence wins:")
        for rel, n, we_r, ne_r, tax in results[:10]:
            print(f"  {rel[:44]:<45} {n:>5} {we_r:>6.3f} {ne_r:>6.3f} {tax*100:>+6.2f}pp")

        # Top 10 where with-evidence wins most (negative tax)
        print(f"\n  Top 10 relations where with-evidence wins:")
        for rel, n, we_r, ne_r, tax in results[-10:]:
            print(f"  {rel[:44]:<45} {n:>5} {we_r:>6.3f} {ne_r:>6.3f} {tax*100:>+6.2f}pp")

        # Summary stats
        taxes = [x[4] for x in results]
        import numpy as np
        arr = np.array(taxes)
        print(f"\n  Summary ({len(results)} relations with >=5 gold):")
        print(f"    Mean tax: {arr.mean()*100:+.2f}pp  Median: {np.median(arr)*100:+.2f}pp")
        print(f"    Relations where NE wins: {sum(1 for t in taxes if t > 0.01)}")
        print(f"    Relations where WE wins: {sum(1 for t in taxes if t < -0.01)}")
        print(f"    Roughly tied: {sum(1 for t in taxes if abs(t) <= 0.01)}")


# ============================================================
# Module D: Hallucination / False Positive Analysis
# ============================================================
def module_d_false_positives():
    """Analyze false positive rates: does evidence generation cause more FPs?"""
    print("\n" + "=" * 70)
    print("  MODULE D: FALSE POSITIVE (HALLUCINATION) ANALYSIS")
    print("=" * 70)

    for model_name, paths in SFT_PAIRS.items():
        we_preds = load_predictions(paths["with_evi"])
        ne_preds = load_predictions(paths["no_evi"])
        if we_preds is None or ne_preds is None:
            we_m = load_metrics(paths["with_evi"])
            ne_m = load_metrics(paths["no_evi"])
            if we_m and ne_m:
                we_fp = we_m["pred_count"] - we_m["tp"]
                ne_fp = ne_m["pred_count"] - ne_m["tp"]
                print(f"\n  {model_name} (from metrics only):")
                print(f"    With-evi: {we_m['pred_count']} preds, {we_m['tp']} TP, {we_fp} FP "
                      f"(FP rate={we_fp/we_m['pred_count']*100:.1f}%)")
                print(f"    No-evi:   {ne_m['pred_count']} preds, {ne_m['tp']} TP, {ne_fp} FP "
                      f"(FP rate={ne_fp/ne_m['pred_count']*100:.1f}%)")
                print(f"    FP diff:  {we_fp - ne_fp:+d} ({(we_fp/we_m['pred_count'] - ne_fp/ne_m['pred_count'])*100:+.1f}pp rate)")
            continue

        print(f"\n{'─'*70}")
        print(f"  {model_name}")
        print(f"{'─'*70}")

        for label, preds in [("With-Evidence", we_preds), ("No-Evidence", ne_preds)]:
            gold_set = set()
            pred_set = set()
            for p in preds:
                for t in p["gold_triples"]:
                    if isinstance(t, dict):
                        gold_set.add((p["doc_id"], t["head"].lower().strip(),
                                     t["tail"].lower().strip(), t["relation"].lower().strip()))
                for t in p["parsed_triples"]:
                    if isinstance(t, dict):
                        pred_set.add((p["doc_id"], t["head"].lower().strip(),
                                     t["tail"].lower().strip(), t["relation"].lower().strip()))

            tp = pred_set & gold_set
            fp = pred_set - gold_set
            fn = gold_set - pred_set

            # FP breakdown: truncated vs clean docs
            trunc_ids = set(p["doc_id"] for p in preds if p["truncated"])
            fp_trunc = sum(1 for k in fp if k[0] in trunc_ids)
            fp_clean = len(fp) - fp_trunc

            print(f"\n  {label}:")
            print(f"    Total preds (unique): {len(pred_set)}  TP: {len(tp)}  "
                  f"FP: {len(fp)}  FN: {len(fn)}")
            print(f"    Precision: {len(tp)/len(pred_set):.4f}  "
                  f"Recall: {len(tp)/len(gold_set):.4f}")
            print(f"    FP from truncated docs: {fp_trunc}  "
                  f"FP from clean docs: {fp_clean}")

            # Per-doc FP rate
            import numpy as np
            doc_fps = defaultdict(int)
            for k in fp:
                doc_fps[k[0]] += 1
            fp_vals = list(doc_fps.values())
            if fp_vals:
                arr = np.array(fp_vals)
                print(f"    Per-doc FP: mean={arr.mean():.1f}  median={np.median(arr):.0f}  "
                      f"max={arr.max()}")


# ============================================================
# Module E: Fair Subset (Reuse existing script logic)
# ============================================================
def module_e_fair_subset():
    """Run fair subset analysis for all model sizes."""
    import re, numpy as np

    print("\n" + "=" * 70)
    print("  MODULE E: FAIR SUBSET ANALYSIS (excl. truncated+repetitive)")
    print("=" * 70)

    from eval.evaluator import DocREDEvaluator

    evaluator = DocREDEvaluator.from_train_file(str(DATA_ROOT / "train_annotated.json"))

    def is_repetitive(raw_output, threshold=3):
        pattern = r'\{[^{}]+\}'
        matches = re.findall(pattern, raw_output)
        if len(matches) < threshold:
            return False
        for i in range(len(matches) - threshold + 1):
            if len(set(matches[i:i+threshold])) == 1:
                return True
        if len(matches) > 5 and len(set(matches)) / len(matches) < 0.5:
            return True
        return False

    def filter_and_eval(preds, doc_ids):
        ids = set(doc_ids)
        sub_preds, sub_gold = [], []
        for p in preds:
            if p["doc_id"] not in ids:
                continue
            for t in p["parsed_triples"]:
                if isinstance(t, dict):
                    sub_preds.append({
                        "doc_id": p["doc_id"], "head": t.get("head", ""),
                        "tail": t.get("tail", ""), "relation": t.get("relation", ""),
                    })
            for t in p["gold_triples"]:
                if isinstance(t, dict):
                    sub_gold.append({
                        "doc_id": p["doc_id"], "head": t.get("head", ""),
                        "tail": t.get("tail", ""), "relation": t.get("relation", ""),
                    })
        return evaluator.compute_f1(sub_preds, sub_gold)

    for model_name, paths in SFT_PAIRS.items():
        we_preds = load_predictions(paths["with_evi"])
        ne_preds = load_predictions(paths["no_evi"])
        if we_preds is None or ne_preds is None:
            print(f"\n  {model_name}: Missing predictions, skipping")
            continue

        # Tag repetitive
        for p in we_preds:
            p["repetitive"] = is_repetitive(p["raw_output"])
        for p in ne_preds:
            p["repetitive"] = is_repetitive(p["raw_output"])

        we_bad = set(p["doc_id"] for p in we_preds if p["truncated"] or p["repetitive"])
        ne_bad = set(p["doc_id"] for p in ne_preds if p["truncated"] or p["repetitive"])
        all_ids = [p["doc_id"] for p in ne_preds]
        fair_ids = [d for d in all_ids if d not in (we_bad | ne_bad)]

        we_trunc = sum(1 for p in we_preds if p["truncated"])
        ne_trunc = sum(1 for p in ne_preds if p["truncated"])
        we_rep = sum(1 for p in we_preds if p["repetitive"])
        ne_rep = sum(1 for p in ne_preds if p["repetitive"])

        print(f"\n{'─'*70}")
        print(f"  {model_name}")
        print(f"{'─'*70}")
        print(f"  With-evi: trunc={we_trunc}, rep={we_rep}, bad={len(we_bad)}")
        print(f"  No-evi:   trunc={ne_trunc}, rep={ne_rep}, bad={len(ne_bad)}")
        print(f"  Fair subset: {len(fair_ids)}/{len(all_ids)} docs")

        # Eval on full and fair
        full_we = filter_and_eval(we_preds, all_ids)
        full_ne = filter_and_eval(ne_preds, all_ids)
        fair_we = filter_and_eval(we_preds, fair_ids)
        fair_ne = filter_and_eval(ne_preds, fair_ids)

        full_tax = (full_ne["ign_f1"] - full_we["ign_f1"]) * 100
        fair_tax = (fair_ne["ign_f1"] - fair_we["ign_f1"]) * 100

        print(f"\n  Full set:  WE Ign-F1={full_we['ign_f1']:.4f}  "
              f"NE Ign-F1={full_ne['ign_f1']:.4f}  Tax={full_tax:+.2f}pp")
        print(f"  Fair set:  WE Ign-F1={fair_we['ign_f1']:.4f}  "
              f"NE Ign-F1={fair_ne['ign_f1']:.4f}  Tax={fair_tax:+.2f}pp")
        print(f"  Tax change: {full_tax:.2f} -> {fair_tax:.2f}pp "
              f"({'reduced' if abs(fair_tax) < abs(full_tax) else 'increased'})")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", default="all",
                        choices=["A", "B", "C", "D", "E", "all"],
                        help="Which analysis module to run")
    args = parser.parse_args()

    modules = {
        "A": module_a_output_length,
        "B": module_b_truncation_impact,
        "C": module_c_per_relation,
        "D": module_d_false_positives,
        "E": module_e_fair_subset,
    }

    if args.module == "all":
        for name, func in modules.items():
            try:
                func()
            except Exception as e:
                print(f"\n  [ERROR] Module {name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        modules[args.module]()


if __name__ == "__main__":
    main()
