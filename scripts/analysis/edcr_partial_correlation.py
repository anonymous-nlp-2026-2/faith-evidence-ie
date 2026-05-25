#!/usr/bin/env python3
"""EDCR per-relation partial correlation analysis.

Validates EDCR as an evidence grounding indicator by computing partial
correlations controlling for relation frequency and avg evidence count.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REL_INFO_PATH = "/workspace/baselines/dreeam/meta/rel_info.json"
GOLD_PATH = "/workspace/data/re-docred-repo/data/dev_revised.json"

EVAL_DIRS = {
    "Qwen3-8B RSFT": "/workspace/eval_results/d102_8b_k1_s42",
    "Qwen3-1.7B RSFT": "/workspace/eval_results/d102_1_7b_k1_s42",
    "Llama-3.1-8B RSFT": "/workspace/eval_results/d102_llama_k1_s42",
}

MIN_SAMPLES = 5


def load_gold(gold_path, rel_info):
    with open(gold_path) as f:
        data = json.load(f)

    gold_by_doc = {}
    rel_stats = defaultdict(lambda: {"count": 0, "total_evi_sents": 0})

    for doc in data:
        doc_id = doc["title"]
        vertex_set = doc["vertexSet"]
        n_sents = len(doc["sents"])
        entries = {}

        for label in doc.get("labels", []):
            h_name = vertex_set[label["h"]][0]["name"]
            t_name = vertex_set[label["t"]][0]["name"]
            rel_name = rel_info.get(label["r"], label["r"])
            evidence = label.get("evidence", [])
            key = (h_name.lower().strip(), t_name.lower().strip(), rel_name.lower().strip())

            if key in entries:
                entries[key]["evidence"] |= set(evidence)
            else:
                entries[key] = {
                    "head": h_name, "tail": t_name, "relation": rel_name,
                    "evidence": set(evidence), "n_sents": n_sents,
                }

        gold_by_doc[doc_id] = entries
        for key, info in entries.items():
            rel_stats[info["relation"]]["count"] += 1
            rel_stats[info["relation"]]["total_evi_sents"] += len(info["evidence"])

    per_relation = {}
    for rel, st in rel_stats.items():
        per_relation[rel] = {
            "n_gold": st["count"],
            "avg_evi_count": st["total_evi_sents"] / st["count"] if st["count"] > 0 else 0,
        }
    return gold_by_doc, per_relation


def load_predictions(pred_path):
    with open(pred_path) as f:
        data = json.load(f)
    pred_by_doc = defaultdict(list)
    for doc in data:
        for triple in doc.get("parsed_triples", []):
            pred_by_doc[doc["doc_id"]].append(triple)
    return pred_by_doc


def compute_per_relation_metrics(gold_by_doc, pred_by_doc, per_relation_gold):
    rel_tp = defaultdict(int)
    rel_fp = defaultdict(int)
    rel_fn = defaultdict(int)
    rel_distractor = defaultdict(int)
    rel_total_cites = defaultdict(int)
    # TP-only EDCR
    rel_tp_distractor = defaultdict(int)
    rel_tp_total_cites = defaultdict(int)

    all_doc_ids = set(list(gold_by_doc.keys()) + list(pred_by_doc.keys()))

    for doc_id in all_doc_ids:
        doc_gold = gold_by_doc.get(doc_id, {})
        doc_preds = pred_by_doc.get(doc_id, [])
        matched = set()

        for pred in doc_preds:
            key = (pred["head"].lower().strip(), pred["tail"].lower().strip(),
                   pred["relation"].lower().strip())
            rel_name = pred["relation"]
            pred_evi = pred.get("evidence", [])
            if not isinstance(pred_evi, list):
                pred_evi = []

            if key in doc_gold:
                rel_tp[rel_name] += 1
                matched.add(key)
                gold_evi = doc_gold[key]["evidence"]
                for s in pred_evi:
                    rel_total_cites[rel_name] += 1
                    rel_tp_total_cites[rel_name] += 1
                    if s not in gold_evi:
                        rel_distractor[rel_name] += 1
                        rel_tp_distractor[rel_name] += 1
            else:
                rel_fp[rel_name] += 1
                rel_total_cites[rel_name] += len(pred_evi)
                rel_distractor[rel_name] += len(pred_evi)

        for key, info in doc_gold.items():
            if key not in matched:
                rel_fn[info["relation"]] += 1

    results = {}
    all_rels = set(list(rel_tp.keys()) + list(rel_fp.keys()) + list(rel_fn.keys()))

    for rel in all_rels:
        tp = rel_tp[rel]; fp = rel_fp[rel]; fn = rel_fn[rel]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        tc = rel_total_cites[rel]; dc = rel_distractor[rel]
        edcr_all = dc / tc if tc > 0 else float('nan')

        tc_tp = rel_tp_total_cites[rel]; dc_tp = rel_tp_distractor[rel]
        edcr_tp = dc_tp / tc_tp if tc_tp > 0 else float('nan')

        gold_info = per_relation_gold.get(rel, {"n_gold": 0, "avg_evi_count": 0})
        results[rel] = {
            "n_gold": gold_info["n_gold"], "avg_evi_count": gold_info["avg_evi_count"],
            "tp": tp, "fp": fp, "fn": fn, "n_pred": tp + fp,
            "precision": prec, "recall": rec, "f1": f1,
            "edcr_all": edcr_all, "edcr_tp": edcr_tp,
            "n_total_cites": tc, "n_distractor_cites": dc,
        }
    return results


def partial_corr(x, y, Z):
    """Partial correlation via residualization. Returns (pearson_r, pearson_p, spearman_rho, spearman_p)."""
    from numpy.linalg import lstsq
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    D = np.column_stack([np.ones(len(x)), Z])
    res_x = x - D @ lstsq(D, x, rcond=None)[0]
    res_y = y - D @ lstsq(D, y, rcond=None)[0]
    pr, pp = stats.pearsonr(res_x, res_y)
    sr, sp = stats.spearmanr(res_x, res_y)
    return pr, pp, sr, sp


def run_analysis(name, per_rel, min_n=MIN_SAMPLES):
    rows = []
    for rel, m in per_rel.items():
        if m["n_gold"] >= min_n and m["n_pred"] > 0 and not np.isnan(m["edcr_all"]):
            rows.append({
                "relation": rel, "n_gold": m["n_gold"],
                "avg_evi_count": m["avg_evi_count"],
                "f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
                "edcr_all": m["edcr_all"],
                "edcr_tp": m["edcr_tp"] if not np.isnan(m.get("edcr_tp", float('nan'))) else m["edcr_all"],
                "tp": m["tp"],
            })
    df = pd.DataFrame(rows)
    if len(df) < 10:
        print(f"\n## {name}: Too few relations ({len(df)})")
        return None

    print(f"\n{'='*72}")
    print(f"  {name}")
    print(f"{'='*72}")
    print(f"  N relations (≥{min_n} gold, ≥1 pred): {len(df)}")

    edcr = df["edcr_all"].values
    f1 = df["f1"].values
    freq = df["n_gold"].values
    log_freq = np.log1p(freq)
    avg_evi = df["avg_evi_count"].values
    prec = df["precision"].values
    rec = df["recall"].values

    print(f"\n  --- Raw Correlations ---")
    for label, a, b in [
        ("EDCR ↔ F1", edcr, f1),
        ("log(freq) ↔ F1", log_freq, f1),
        ("EDCR ↔ log(freq)", edcr, log_freq),
        ("avg_evi ↔ F1", avg_evi, f1),
        ("EDCR ↔ avg_evi", edcr, avg_evi),
    ]:
        pr, pp = stats.pearsonr(a, b)
        sr, sp = stats.spearmanr(a, b)
        print(f"    {label:<22}  Pearson r={pr:+.4f} (p={pp:.1e})  Spearman ρ={sr:+.4f} (p={sp:.1e})")

    print(f"\n  --- Partial Correlations ---")
    results = {}
    for label, a, b, Z_arr in [
        ("EDCR ↔ F1 | freq", edcr, f1, log_freq),
        ("EDCR ↔ F1 | freq,evi", edcr, f1, np.column_stack([log_freq, avg_evi])),
        ("EDCR ↔ Prec | freq", edcr, prec, log_freq),
        ("EDCR ↔ Recall | freq", edcr, rec, log_freq),
    ]:
        pr, pp, sr, sp = partial_corr(a, b, Z_arr)
        n_cov = Z_arr.shape[1] if Z_arr.ndim > 1 else 1
        df_val = len(a) - n_cov - 2
        sig = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else "n.s."
        print(f"    {label:<28}  r={pr:+.4f} (p={pp:.1e}) {sig}  ρ={sr:+.4f} (p={sp:.1e})  df={df_val}")
        results[label] = {"pearson_r": float(pr), "pearson_p": float(pp),
                          "spearman_rho": float(sr), "spearman_p": float(sp), "df": df_val}

    # Robustness: exclude relations with F1=0
    df_pos = df[df["f1"] > 0]
    if len(df_pos) >= 10:
        print(f"\n  --- Robustness: excluding F1=0 relations (N={len(df_pos)}) ---")
        e2, f2, lf2, ae2 = df_pos["edcr_all"].values, df_pos["f1"].values, np.log1p(df_pos["n_gold"].values), df_pos["avg_evi_count"].values
        pr, pp, sr, sp = partial_corr(e2, f2, lf2)
        sig = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else "n.s."
        print(f"    EDCR ↔ F1 | freq (F1>0)    r={pr:+.4f} (p={pp:.1e}) {sig}  ρ={sr:+.4f} (p={sp:.1e})  N={len(df_pos)}")
        results["robustness_f1_pos"] = {"pearson_r": float(pr), "pearson_p": float(pp), "n": len(df_pos)}

    # TP-only EDCR robustness
    df_tp = df[df["tp"] > 0].copy()
    if len(df_tp) >= 10:
        print(f"\n  --- Robustness: EDCR_tp (TP-only evidence, N={len(df_tp)}) ---")
        e3, f3, lf3 = df_tp["edcr_tp"].values, df_tp["f1"].values, np.log1p(df_tp["n_gold"].values)
        pr, pp, sr, sp = partial_corr(e3, f3, lf3)
        sig = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else "n.s."
        print(f"    EDCR_tp ↔ F1 | freq         r={pr:+.4f} (p={pp:.1e}) {sig}  ρ={sr:+.4f} (p={sp:.1e})  N={len(df_tp)}")
        results["robustness_edcr_tp"] = {"pearson_r": float(pr), "pearson_p": float(pp), "n": len(df_tp)}

    return {"model": name, "n_relations": len(df), "partial_correlations": results,
            "raw_edcr_f1_r": float(stats.pearsonr(edcr, f1)[0]),
            "raw_edcr_f1_p": float(stats.pearsonr(edcr, f1)[1])}


def main():
    print("# EDCR Per-Relation Partial Correlation Analysis")
    print(f"# Min gold per relation: {MIN_SAMPLES}")

    with open(REL_INFO_PATH) as f:
        rel_info = json.load(f)

    gold_by_doc, per_relation_gold = load_gold(GOLD_PATH, rel_info)
    n_docs = len(gold_by_doc)
    n_triples = sum(len(v) for v in gold_by_doc.values())
    print(f"\nGold: {n_docs} docs, {n_triples} triples, {len(per_relation_gold)} relation types")

    all_results = {}
    for model_name, eval_dir in EVAL_DIRS.items():
        pred_path = Path(eval_dir) / "predictions.json"
        if not pred_path.exists():
            print(f"\n## {model_name}: predictions not found")
            continue
        preds = load_predictions(str(pred_path))
        per_rel = compute_per_relation_metrics(gold_by_doc, preds, per_relation_gold)
        res = run_analysis(model_name, per_rel)
        if res:
            all_results[model_name] = res

    # Summary
    if all_results:
        print(f"\n{'='*72}")
        print(f"  SUMMARY")
        print(f"{'='*72}")
        print(f"\n  {'Model':<25} {'Raw r':<10} {'Partial r':<12} {'p-value':<12} {'Sig'}")
        print(f"  {'-'*65}")
        for m, r in all_results.items():
            pc = r["partial_correlations"].get("EDCR ↔ F1 | freq", {})
            sig = "***" if pc.get("pearson_p",1) < 0.001 else "**" if pc.get("pearson_p",1) < 0.01 else "*" if pc.get("pearson_p",1) < 0.05 else "n.s."
            print(f"  {m:<25} {r['raw_edcr_f1_r']:+.4f}    {pc.get('pearson_r',0):+.4f}      {pc.get('pearson_p',1):.1e}    {sig}")

        print(f"\n  Interpretation:")
        consistent = all(r["partial_correlations"].get("EDCR ↔ F1 | freq", {}).get("pearson_p", 1) < 0.05
                         for r in all_results.values())
        direction = all(r["partial_correlations"].get("EDCR ↔ F1 | freq", {}).get("pearson_r", 0) < 0
                        for r in all_results.values())
        if consistent and direction:
            print("  EDCR IS a reliable indicator of evidence grounding quality.")
            print("  Lower EDCR (fewer distractor citations) → higher per-relation F1,")
            print("  even after controlling for relation frequency.")
        elif consistent:
            print("  Significant partial correlation exists but direction needs examination.")
        else:
            print("  Results are mixed across models. EDCR validity as an independent")
            print("  indicator requires further investigation.")

    out = "/workspace/eval_results/edcr_partial_correlation_analysis.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
