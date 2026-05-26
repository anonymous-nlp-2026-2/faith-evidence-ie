"""Cross-method EDCR diagnostic analysis.

Input:
  --methods NAME:PRED_PATH [NAME:PRED_PATH ...]
    Each entry is method_name:path_to_predictions.json
    predictions.json format: list of {doc_id, parsed_triples, gold_triples}
    where each triple has {head, relation, tail, evidence}

  OR --auto-discover (scan eval_results/*/predictions.json)

  --gold PATH  DocRED dev.json for gold (only needed if predictions.json lacks gold_triples)
  --output-dir PATH  Where to write results (default: eval_results/)

Output (in output_dir):
  edcr_diagnostic.json  — per-method: overall EDCR, per-relation-type EDCR, per-document EDCR stats
  edcr_diagnostic.tex   — LaTeX table ready for paper
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Allow importing freige when run from ./
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.evaluator import DocREDEvaluator, _triple_key


# --------------------------------------------------------------------------
# Default method configs for --auto-discover
# --------------------------------------------------------------------------
AUTO_METHODS = [
    ("SFT baseline (quant)", "./sft_baseline_eval_results/predictions.json"),
    ("SFT baseline (noquant)", "eval_results/sft_baseline_noquant/predictions.json"),
    ("RSFT-CED s42", "eval_results/rsft_r1/predictions.json"),
    ("RSFT-CED s43", "eval_results/rsft_s43/predictions.json"),
    ("RSFT-CED s44", "eval_results/rsft_s44/predictions.json"),
    ("RSFT-CED r2a", "eval_results/rsft_r2a/predictions.json"),
    ("RSFT-CED r2b", "eval_results/rsft_r2b/predictions.json"),
    ("RSFT-flatNLI s42", "eval_results/rsft_flat_nli/predictions.json"),
    ("SFT no-evidence", "eval_results/sft_no_evidence/predictions.json"),
    ("GRPO bf16 ckpt100", "eval_results/grpo_g8_bf16_ckpt100/predictions.json"),
    ("GRPO QLoRA kl001 ckpt100", "eval_results/grpo_ced_kl001_step100/checkpoint-100/predictions.json"),
    ("GRPO QLoRA lr5e6 ckpt100", "eval_results/lr5e6_r1_ckpt100/checkpoint-100/predictions.json"),
    ("DPO-CED 1ep (fixed)", "eval_results/dpo_ced_1ep_fixed/predictions.json"),
]


def load_predictions(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def flatten_predictions(docs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Flatten per-doc predictions into evaluator format.
    Returns (predictions, gold) — both as flat lists of {doc_id, head, tail, relation, evidence}.
    """
    preds = []
    gold = []
    for doc in docs:
        doc_id = doc["doc_id"]
        for t in doc.get("parsed_triples", []):
            preds.append({
                "doc_id": doc_id,
                "head": t["head"],
                "tail": t["tail"],
                "relation": t["relation"],
                "evidence": t.get("evidence", []),
            })
        for g in doc.get("gold_triples", []):
            gold.append({
                "doc_id": doc_id,
                "head": g["head"],
                "tail": g["tail"],
                "relation": g["relation"],
                "evidence": g.get("evidence", []),
            })
    return preds, gold


def compute_per_relation_edcr(predictions: list[dict], gold: list[dict]) -> dict:
    """EDCR broken down by relation type.
    Returns {relation_name: {edcr, n_distractor, n_total, n_triples}}.
    """
    gold_map = {}
    for g in gold:
        key = _triple_key(g["doc_id"], g["head"], g["tail"], g["relation"])
        gold_map.setdefault(key, set()).update(g.get("evidence", []))

    per_rel = defaultdict(lambda: {"n_distractor": 0, "n_total": 0, "n_triples": 0})

    for p in predictions:
        pred_evi = set(p.get("evidence", []))
        if not pred_evi:
            continue
        rel = p["relation"].lower().strip()
        key = _triple_key(p["doc_id"], p["head"], p["tail"], p["relation"])
        gold_evi = gold_map.get(key, set())

        distractors = pred_evi - gold_evi
        per_rel[rel]["n_distractor"] += len(distractors)
        per_rel[rel]["n_total"] += len(pred_evi)
        per_rel[rel]["n_triples"] += 1

    result = {}
    for rel, counts in sorted(per_rel.items(), key=lambda x: -x[1]["n_triples"]):
        edcr = counts["n_distractor"] / counts["n_total"] if counts["n_total"] else 0.0
        result[rel] = {
            "edcr": round(edcr, 4),
            "n_distractor": counts["n_distractor"],
            "n_total": counts["n_total"],
            "n_triples": counts["n_triples"],
        }
    return result


def compute_per_document_edcr(predictions: list[dict], gold: list[dict]) -> dict:
    """Per-document EDCR, returning distribution statistics."""
    gold_map = {}
    for g in gold:
        key = _triple_key(g["doc_id"], g["head"], g["tail"], g["relation"])
        gold_map.setdefault(key, set()).update(g.get("evidence", []))

    doc_distractor = defaultdict(int)
    doc_total = defaultdict(int)

    for p in predictions:
        pred_evi = set(p.get("evidence", []))
        if not pred_evi:
            continue
        doc_id = p["doc_id"]
        key = _triple_key(p["doc_id"], p["head"], p["tail"], p["relation"])
        gold_evi = gold_map.get(key, set())

        doc_distractor[doc_id] += len(pred_evi - gold_evi)
        doc_total[doc_id] += len(pred_evi)

    doc_edcrs = []
    for doc_id in doc_total:
        if doc_total[doc_id] > 0:
            doc_edcrs.append(doc_distractor[doc_id] / doc_total[doc_id])

    if not doc_edcrs:
        return {"mean": 0, "median": 0, "std": 0, "q25": 0, "q75": 0, "n_docs": 0}

    doc_edcrs.sort()
    n = len(doc_edcrs)
    q25_idx = int(n * 0.25)
    q75_idx = int(n * 0.75)

    return {
        "mean": round(statistics.mean(doc_edcrs), 4),
        "median": round(statistics.median(doc_edcrs), 4),
        "std": round(statistics.stdev(doc_edcrs), 4) if n > 1 else 0.0,
        "q25": round(doc_edcrs[q25_idx], 4),
        "q75": round(doc_edcrs[q75_idx], 4),
        "min": round(doc_edcrs[0], 4),
        "max": round(doc_edcrs[-1], 4),
        "n_docs": n,
    }


def analyze_method(name: str, pred_path: str) -> dict | None:
    path = Path(pred_path)
    if not path.exists():
        print(f"  SKIP {name}: {pred_path} not found")
        return None

    docs = load_predictions(str(path))
    preds, gold = flatten_predictions(docs)

    if not preds:
        print(f"  SKIP {name}: no predictions")
        return None

    evaluator = DocREDEvaluator()

    overall = evaluator.compute_edcr(preds, gold)
    f1_metrics = evaluator.compute_f1(preds, gold)
    evi_metrics = evaluator.compute_evi_f1(preds, gold)

    per_rel = compute_per_relation_edcr(preds, gold)
    per_doc = compute_per_document_edcr(preds, gold)

    return {
        "method": name,
        "rel_f1": round(f1_metrics["f1"], 4),
        "evi_f1": round(evi_metrics["evi_f1"], 4),
        "edcr": round(overall["edcr"], 4),
        "n_distractor_citations": overall["n_distractor_citations"],
        "n_total_citations": overall["n_total_citations"],
        "n_predictions": len(preds),
        "per_relation_edcr": per_rel,
        "per_document_edcr": per_doc,
    }


def generate_latex_table(results: list[dict]) -> str:
    """Generate LaTeX table with overall metrics + per-doc EDCR stats."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{EDCR Cross-Method Diagnostic}")
    lines.append(r"\label{tab:edcr-diagnostic}")
    lines.append(r"\begin{tabular}{l ccc cc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Rel-F1 & Evi-F1 & EDCR & EDCR$_\text{med}$ & EDCR$_\text{std}$ \\")
    lines.append(r"\midrule")

    for r in results:
        name = r["method"].replace("_", r"\_")
        doc = r["per_document_edcr"]
        lines.append(
            f"  {name} & {r['rel_f1']:.4f} & {r['evi_f1']:.4f} & "
            f"{r['edcr']:.4f} & {doc['median']:.4f} & {doc['std']:.4f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_per_relation_latex(results: list[dict], top_k: int = 10) -> str:
    """LaTeX table showing EDCR per relation type for top-K most frequent relations."""
    # Collect all relation types across methods, rank by total triples
    rel_counts = defaultdict(int)
    for r in results:
        for rel, info in r["per_relation_edcr"].items():
            rel_counts[rel] += info["n_triples"]
    top_rels = sorted(rel_counts.keys(), key=lambda x: -rel_counts[x])[:top_k]

    n_methods = len(results)
    col_spec = "l" + " c" * n_methods
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Per-Relation-Type EDCR (top-%d relations by frequency)}" % top_k)
    lines.append(r"\label{tab:edcr-per-relation}")
    lines.append(r"\begin{tabular}{%s}" % col_spec)
    lines.append(r"\toprule")

    header = "Relation"
    for r in results:
        short = r["method"][:12]
        header += f" & {short}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for rel in top_rels:
        row = rel.replace("_", r"\_")
        for r in results:
            info = r["per_relation_edcr"].get(rel)
            if info:
                row += f" & {info['edcr']:.3f}"
            else:
                row += " & --"
        row += r" \\"
        lines.append(f"  {row}")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-method EDCR diagnostic analysis"
    )
    parser.add_argument(
        "--methods", nargs="+", metavar="NAME:PATH",
        help="Method entries as name:predictions_path pairs"
    )
    parser.add_argument(
        "--auto-discover", action="store_true",
        help="Auto-discover methods from built-in config"
    )
    parser.add_argument(
        "--output-dir", default="eval_results/",
        help="Output directory for diagnostic files"
    )
    parser.add_argument(
        "--top-k-relations", type=int, default=10,
        help="Number of top relations for per-relation table"
    )
    args = parser.parse_args()

    method_list = []
    if args.auto_discover:
        method_list = AUTO_METHODS
    elif args.methods:
        for m in args.methods:
            if ":" not in m:
                parser.error(f"Invalid method format: {m} (expected NAME:PATH)")
            name, path = m.split(":", 1)
            method_list.append((name, path))
    else:
        parser.error("Specify --methods or --auto-discover")

    results = []
    for name, path in method_list:
        print(f"Analyzing: {name}")
        r = analyze_method(name, path)
        if r is not None:
            results.append(r)

    if not results:
        print("No valid results.")
        return

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Summary JSON
    json_path = out / "edcr_diagnostic.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON -> {json_path}")

    # LaTeX tables
    tex_path = out / "edcr_diagnostic.tex"
    with open(tex_path, "w") as f:
        f.write("% Auto-generated by edcr_diagnostic.py\n\n")
        f.write(generate_latex_table(results))
        f.write("\n\n")
        f.write(generate_per_relation_latex(results, top_k=args.top_k_relations))
    print(f"LaTeX -> {tex_path}")

    # Console summary
    print(f"\n{'Method':<28} {'rel_f1':>7} {'evi_f1':>7} {'EDCR':>7} {'med':>7} {'std':>7} {'n_doc':>6}")
    print("-" * 78)
    for r in results:
        doc = r["per_document_edcr"]
        print(f"{r['method']:<28} {r['rel_f1']:>7.4f} {r['evi_f1']:>7.4f} "
              f"{r['edcr']:>7.4f} {doc['median']:>7.4f} {doc['std']:>7.4f} {doc['n_docs']:>6}")


if __name__ == "__main__":
    main()
