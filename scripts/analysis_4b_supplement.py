"""4B Duplication + Truncation + Complexity 分析补充脚本。

CPU-only，不碰 GPU。
从 metrics.json 和 predictions.json 提取 4B SFT with/no-evidence 数据。
"""

import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, '.')

EVAL_ROOT = Path("eval_results")
DATA_ROOT = Path("data/docred")

# 4B eval paths
WE_DIR = EVAL_ROOT / "sft_baseline_d076_reeval"  # with-evidence (metrics only)
NE_DIR = EVAL_ROOT / "plan_014_no_evidence_eval_d076"  # no-evidence (has predictions.json)

# Other models for comparison context
ALL_PAIRS = {
    "Qwen3-1.7B": {
        "with_evi": EVAL_ROOT / "qwen3_1_7b_sft_eval",
        "no_evi":   EVAL_ROOT / "qwen3_1_7b_no_evidence_eval_v3",
    },
    "Qwen3-4B": {
        "with_evi": WE_DIR,
        "no_evi":   NE_DIR,
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


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_dev_gold_counts():
    """Map doc title -> number of gold triples."""
    from freige.data.docred_processor import DOCRED_REL_INFO
    dev_data = load_json(DATA_ROOT / "dev.json")
    counts = {}
    for doc in dev_data:
        title = doc.get("title", "")
        counts[title] = len(doc.get("labels", []))
    return counts


def compute_duplication_from_predictions(preds):
    """Per-doc duplication analysis from predictions.json."""
    total_raw = 0
    total_unique = 0
    docs_with_dups = 0
    
    for doc in preds:
        triples = doc.get("parsed_triples", [])
        raw_count = len(triples)
        
        unique_set = set()
        for t in triples:
            if isinstance(t, dict):
                key = (
                    t.get("head", "").lower().strip(),
                    t.get("relation", "").lower().strip(),
                    t.get("tail", "").lower().strip(),
                )
                unique_set.add(key)
        
        unique_count = len(unique_set)
        total_raw += raw_count
        total_unique += unique_count
        if raw_count > unique_count:
            docs_with_dups += 1
    
    dup_rate = 1 - total_unique / total_raw if total_raw > 0 else 0
    return {
        "raw_preds": total_raw,
        "unique_preds": total_unique,
        "dup_rate": dup_rate,
        "docs_with_dups": docs_with_dups,
        "n_docs": len(preds),
    }


def compute_duplication_from_metrics(metrics):
    """Aggregate duplication from metrics.json (no per-doc breakdown)."""
    raw = metrics["n_predictions"]
    unique = metrics["pred_count"]
    dup_rate = 1 - unique / raw if raw > 0 else 0
    return {
        "raw_preds": raw,
        "unique_preds": unique,
        "dup_rate": dup_rate,
        "docs_with_dups": "N/A",
        "n_docs": metrics.get("n_documents", metrics.get("n_total", 0)),
    }


# ============================================================
# Section 1: Duplication Rate Table (all models)
# ============================================================
def section_1_duplication():
    print("=" * 80)
    print("  SECTION 1: DUPLICATION RATE (ALL MODELS)")
    print("=" * 80)
    print()
    print(f"{'Model':<18} {'Condition':<15} {'Raw preds':>10} {'Unique':>8} {'Dup rate':>9} {'Docs w/dups':>12}")
    print("-" * 80)
    
    for model_name, paths in ALL_PAIRS.items():
        for cond, cond_dir in [("With-Evidence", paths["with_evi"]), ("No-Evidence", paths["no_evi"])]:
            pred_file = cond_dir / "predictions.json"
            metrics_file = cond_dir / "metrics.json"
            
            if pred_file.exists():
                preds = load_json(pred_file)
                stats = compute_duplication_from_predictions(preds)
                source = "preds"
            elif metrics_file.exists():
                metrics = load_json(metrics_file)
                stats = compute_duplication_from_metrics(metrics)
                source = "metrics"
            else:
                print(f"{model_name:<18} {cond:<15} {'NO DATA':>10}")
                continue
            
            dwd = f"{stats['docs_with_dups']}" if isinstance(stats['docs_with_dups'], int) else stats['docs_with_dups']
            note = f" [{source}]" if source == "metrics" else ""
            print(f"{model_name:<18} {cond:<15} {stats['raw_preds']:>10} {stats['unique_preds']:>8} "
                  f"{stats['dup_rate']:>8.1%} {dwd:>12}{note}")
    print()


# ============================================================
# Section 2: 4B Truncation Crossover
# ============================================================
def section_2_truncation_crossover():
    print("=" * 80)
    print("  SECTION 2: TRUNCATION CROSSOVER ANALYSIS")
    print("=" * 80)
    
    from freige.eval.evaluator import DocREDEvaluator
    evaluator = DocREDEvaluator.from_train_file(str(DATA_ROOT / "train_annotated.json"))
    
    for model_name, paths in ALL_PAIRS.items():
        we_pred_file = paths["with_evi"] / "predictions.json"
        ne_pred_file = paths["no_evi"] / "predictions.json"
        
        if not we_pred_file.exists() or not ne_pred_file.exists():
            we_metrics = load_json(paths["with_evi"] / "metrics.json") if (paths["with_evi"] / "metrics.json").exists() else None
            ne_metrics = load_json(paths["no_evi"] / "metrics.json") if (paths["no_evi"] / "metrics.json").exists() else None
            
            print(f"\n{'─'*80}")
            print(f"  {model_name}: AGGREGATE ONLY (missing predictions.json)")
            if we_metrics and ne_metrics:
                print(f"  With-evi truncated: {we_metrics.get('n_truncated', '?')}/{we_metrics.get('n_documents', '?')}")
                print(f"  No-evi truncated:   {ne_metrics.get('n_truncated', '?')}/{ne_metrics.get('n_documents', '?')}")
            continue
        
        we_preds = load_json(we_pred_file)
        ne_preds = load_json(ne_pred_file)
        
        we_trunc = {p["doc_id"]: p.get("truncated", False) for p in we_preds}
        ne_trunc = {p["doc_id"]: p.get("truncated", False) for p in ne_preds}
        
        # Build crossover categories
        categories = {
            "Both truncated": [],
            "Only evi truncated": [],
            "Only noevi truncated": [],
            "Neither truncated": [],
        }
        
        all_doc_ids = set(we_trunc.keys()) & set(ne_trunc.keys())
        for doc_id in all_doc_ids:
            wt = we_trunc.get(doc_id, False)
            nt = ne_trunc.get(doc_id, False)
            if wt and nt:
                categories["Both truncated"].append(doc_id)
            elif wt and not nt:
                categories["Only evi truncated"].append(doc_id)
            elif not wt and nt:
                categories["Only noevi truncated"].append(doc_id)
            else:
                categories["Neither truncated"].append(doc_id)
        
        # Compute F1 per category
        def compute_f1_for_docs(preds, doc_ids):
            ids = set(doc_ids)
            pred_list, gold_list = [], []
            for p in preds:
                if p["doc_id"] not in ids:
                    continue
                for t in p["parsed_triples"]:
                    if isinstance(t, dict):
                        pred_list.append({
                            "doc_id": p["doc_id"],
                            "head": t.get("head", ""),
                            "tail": t.get("tail", ""),
                            "relation": t.get("relation", ""),
                        })
                for t in p["gold_triples"]:
                    if isinstance(t, dict):
                        gold_list.append({
                            "doc_id": p["doc_id"],
                            "head": t.get("head", ""),
                            "tail": t.get("tail", ""),
                            "relation": t.get("relation", ""),
                            "evidence": t.get("evidence", []),
                        })
            if not pred_list or not gold_list:
                return {"f1": 0, "ign_f1": 0}
            return evaluator.compute_f1(pred_list, gold_list)
        
        print(f"\n{'─'*80}")
        print(f"  {model_name}")
        print(f"{'─'*80}")
        print(f"{'Category':<25} {'Count':>6} {'Evi F1':>8} {'NoEvi F1':>9} {'Delta':>8}")
        print("-" * 60)
        
        for cat_name, doc_ids in categories.items():
            n = len(doc_ids)
            if n > 0:
                we_f1 = compute_f1_for_docs(we_preds, doc_ids)
                ne_f1 = compute_f1_for_docs(ne_preds, doc_ids)
                delta = (ne_f1["ign_f1"] - we_f1["ign_f1"]) * 100
                print(f"{cat_name:<25} {n:>6} {we_f1['ign_f1']:>8.4f} {ne_f1['ign_f1']:>9.4f} {delta:>+7.2f}pp")
            else:
                print(f"{cat_name:<25} {n:>6} {'—':>8} {'—':>9} {'—':>8}")
    print()


# ============================================================
# Section 3: Document Complexity Interaction
# ============================================================
def section_3_doc_complexity():
    print("=" * 80)
    print("  SECTION 3: DOCUMENT COMPLEXITY INTERACTION (TRUNCATION BY GOLD TRIPLE BIN)")
    print("=" * 80)
    
    gold_counts = load_dev_gold_counts()
    
    bins = [(0, 5), (6, 10), (11, 20), (21, 200)]
    bin_labels = ["0-5", "6-10", "11-20", "21+"]
    
    def get_bin(count):
        for i, (lo, hi) in enumerate(bins):
            if lo <= count <= hi:
                return i
        return len(bins) - 1
    
    for model_name, paths in ALL_PAIRS.items():
        print(f"\n{'─'*80}")
        print(f"  {model_name}")
        print(f"{'─'*80}")
        
        for cond, cond_dir in [("With-Evidence", paths["with_evi"]), ("No-Evidence", paths["no_evi"])]:
            pred_file = cond_dir / "predictions.json"
            metrics_file = cond_dir / "metrics.json"
            
            if not pred_file.exists():
                if metrics_file.exists():
                    m = load_json(metrics_file)
                    print(f"\n  {cond}: AGGREGATE ONLY — {m.get('n_truncated', '?')}/{m.get('n_documents', '?')} truncated")
                continue
            
            preds = load_json(pred_file)
            
            bin_total = [0] * len(bins)
            bin_trunc = [0] * len(bins)
            
            for p in preds:
                doc_id = p["doc_id"]
                gc = gold_counts.get(doc_id, 0)
                bi = get_bin(gc)
                bin_total[bi] += 1
                if p.get("truncated", False):
                    bin_trunc[bi] += 1
            
            print(f"\n  {cond}:")
            print(f"    {'Bin':<8} {'Docs':>6} {'Truncated':>10} {'Rate':>8}")
            print(f"    {'-'*36}")
            for i, label in enumerate(bin_labels):
                rate = bin_trunc[i] / bin_total[i] if bin_total[i] > 0 else 0
                print(f"    {label:<8} {bin_total[i]:>6} {bin_trunc[i]:>10} {rate:>7.1%}")
    print()


# ============================================================
# Section 4: Capacity Valley Verification
# ============================================================
def section_4_capacity_valley():
    print("=" * 80)
    print("  SECTION 4: CAPACITY VALLEY VERIFICATION FOR 4B")
    print("=" * 80)
    
    we_m = load_json(WE_DIR / "metrics.json")
    ne_m = load_json(NE_DIR / "metrics.json")
    
    we_dup = 1 - we_m["pred_count"] / we_m["n_predictions"] if we_m["n_predictions"] > 0 else 0
    ne_dup = 1 - ne_m["pred_count"] / ne_m["n_predictions"] if ne_m["n_predictions"] > 0 else 0
    
    tax = (ne_m["ign_f1"] - we_m["ign_f1"]) * 100
    
    print(f"\n  4B Qwen3-4B SFT Capacity Valley Analysis:")
    print(f"  ─────────────────────────────────────────")
    print(f"  With-evidence Ign-F1:     {we_m['ign_f1']:.4f}")
    print(f"  No-evidence Ign-F1:       {ne_m['ign_f1']:.4f}")
    print(f"  Evidence Tax:             {tax:+.2f}pp")
    print(f"")
    print(f"  With-evidence dup rate:   {we_dup:.1%} (raw={we_m['n_predictions']}, unique={we_m['pred_count']})")
    print(f"  No-evidence dup rate:     {ne_dup:.1%} (raw={ne_m['n_predictions']}, unique={ne_m['pred_count']})")
    print(f"")
    print(f"  With-evidence truncated:  {we_m['n_truncated']}/{we_m['n_documents']} ({we_m['n_truncated']/we_m['n_documents']:.1%})")
    print(f"  No-evidence truncated:    {ne_m['n_truncated']}/{ne_m['n_documents']} ({ne_m['n_truncated']/ne_m['n_documents']:.1%})")
    print(f"")
    print(f"  With-evidence recall:     {we_m['recall']:.4f}")
    print(f"  No-evidence recall:       {ne_m['recall']:.4f}")
    print(f"  Recall gain (no-evi):     {(ne_m['recall'] - we_m['recall'])*100:+.2f}pp")
    print(f"")
    
    # Compare with other models
    print(f"  Cross-model comparison (from metrics):")
    print(f"  {'Model':<18} {'WE dup%':>8} {'NE dup%':>8} {'Tax pp':>8} {'WE trunc':>10} {'NE trunc':>10}")
    print(f"  {'-'*70}")
    for model_name, paths in ALL_PAIRS.items():
        we_mf = paths["with_evi"] / "metrics.json"
        ne_mf = paths["no_evi"] / "metrics.json"
        if not we_mf.exists() or not ne_mf.exists():
            continue
        wm = load_json(we_mf)
        nm = load_json(ne_mf)
        
        # For 1.7B and 8B, also get metrics_v2 if available
        we_v2 = paths["with_evi"] / "metrics_v2.json"
        ne_v2 = paths["no_evi"] / "metrics_v2.json"
        if we_v2.exists():
            wm_v2 = load_json(we_v2)
            wm_ign = wm_v2.get("ign_f1", wm["ign_f1"])
        else:
            wm_ign = wm["ign_f1"]
        if ne_v2.exists():
            nm_v2 = load_json(ne_v2)
            nm_ign = nm_v2.get("ign_f1", nm["ign_f1"])
        else:
            nm_ign = nm["ign_f1"]
        
        wd = 1 - wm["pred_count"] / wm["n_predictions"] if wm["n_predictions"] > 0 else 0
        nd = 1 - nm["pred_count"] / nm["n_predictions"] if nm["n_predictions"] > 0 else 0
        t = (nm_ign - wm_ign) * 100
        
        print(f"  {model_name:<18} {wd:>7.1%} {nd:>7.1%} {t:>+7.2f} "
              f"{wm['n_truncated']:>5}/{wm['n_documents']} {nm['n_truncated']:>5}/{nm['n_documents']}")
    print()


# ============================================================
# Section 5: 1.7B config check (base_model field)
# ============================================================
def section_5_config_check():
    print("=" * 80)
    print("  SECTION 5: CONFIG BASE_MODEL FIELD CHECK")
    print("=" * 80)
    
    dirs_to_check = [
        ("qwen3_1_7b_sft_eval", EVAL_ROOT / "qwen3_1_7b_sft_eval"),
        ("qwen3_1_7b_no_evidence_eval_v3", EVAL_ROOT / "qwen3_1_7b_no_evidence_eval_v3"),
    ]
    
    for name, d in dirs_to_check:
        mf = d / "metrics.json"
        if mf.exists():
            m = load_json(mf)
            config = m.get("config", {})
            base = config.get("base_model", "?")
            model = config.get("model_path", "?")
            adapter = config.get("sft_adapter", "?")
            print(f"\n  {name}:")
            print(f"    base_model:  {base}")
            print(f"    model_path:  {model}")
            print(f"    sft_adapter: {adapter}")
    print()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    section_1_duplication()
    section_2_truncation_crossover()
    section_3_doc_complexity()
    section_4_capacity_valley()
    section_5_config_check()
