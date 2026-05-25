"""Deep error analysis for FREIGE relation extraction predictions.

Input:
  - predictions.json from each eval dir (parsed_triples, gold_triples, doc_id, truncated)
  - dev.json (DocRED gold with sentence-level structure)
  - train_annotated.json (for relation frequency and ign-triple filtering)

Output:
  - Per-eval error classification CSV
  - Multi-eval comparison LaTeX table
  - Per-relation breakdown JSON

Dependencies: standard library only (json, csv, collections, argparse)
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ── DocRED relation mapping (P-code → name) ──

DOCRED_REL_INFO = {
    "P6": "head of government", "P17": "country", "P19": "place of birth",
    "P20": "place of death", "P22": "father", "P25": "mother",
    "P26": "spouse", "P27": "country of citizenship", "P30": "continent",
    "P31": "instance of", "P35": "head of state", "P36": "capital",
    "P37": "official language", "P39": "position held", "P40": "child",
    "P50": "author", "P54": "member of sports team", "P57": "director",
    "P58": "screenwriter", "P69": "educated at", "P86": "composer",
    "P102": "member of political party", "P108": "employer",
    "P112": "founded by", "P118": "league", "P123": "publisher",
    "P127": "owned by",
    "P131": "located in the administrative territorial entity",
    "P136": "genre", "P137": "operator", "P140": "religion",
    "P150": "contains administrative territorial entity",
    "P155": "follows", "P156": "followed by",
    "P159": "headquarters location", "P161": "cast member",
    "P162": "producer", "P166": "award received", "P170": "creator",
    "P171": "parent taxon", "P172": "ethnic group", "P175": "performer",
    "P176": "manufacturer", "P178": "developer", "P179": "series",
    "P190": "sister city", "P194": "legislative body",
    "P205": "basin country",
    "P206": "located in or next to body of water",
    "P241": "military branch", "P264": "record label",
    "P272": "production company", "P276": "location",
    "P279": "subclass of", "P355": "subsidiary", "P361": "part of",
    "P364": "original language of work", "P400": "platform",
    "P403": "mouth of the watercourse", "P449": "original network",
    "P463": "member of", "P488": "chairperson",
    "P495": "country of origin", "P527": "has part",
    "P551": "residence", "P569": "date of birth",
    "P570": "date of death", "P571": "inception",
    "P576": "dissolved, abolished or demolished",
    "P577": "publication date", "P580": "start time",
    "P582": "end time", "P585": "point in time", "P607": "conflict",
    "P674": "characters", "P676": "lyrics by",
    "P706": "located on terrain feature", "P710": "participant",
    "P737": "influenced by", "P740": "location of formation",
    "P749": "parent organization", "P800": "notable work",
    "P807": "separated from", "P840": "narrative location",
    "P937": "work location", "P1001": "applies to jurisdiction",
    "P1056": "product or material produced",
    "P1198": "unemployment rate", "P1336": "territory claimed by",
    "P1344": "participant of", "P1365": "replaces",
    "P1366": "replaced by", "P1376": "capital of",
    "P1412": "languages spoken, written or signed",
    "P1441": "present in work", "P3373": "sibling",
}


def _norm(s):
    return s.lower().strip()


def _triple_key(head, tail, relation):
    return (_norm(head), _norm(tail), _norm(relation))


def _pair_key(head, tail):
    return (_norm(head), _norm(tail))


# ── Data loading ──

def load_predictions(pred_path):
    with open(pred_path) as f:
        return json.load(f)


def load_gold_data(gold_path):
    """Load DocRED dev.json, return dict: title → {labels, vertexSet, sents}."""
    with open(gold_path) as f:
        data = json.load(f)
    return {doc["title"]: doc for doc in data}


def build_train_facts(train_path):
    """Build set of (h_name_norm, t_name_norm, rel_name_norm) from training data."""
    with open(train_path) as f:
        train_data = json.load(f)
    facts = set()
    for doc in train_data:
        vs = doc["vertexSet"]
        for label in doc.get("labels", []):
            h_name = vs[label["h"]][0]["name"]
            t_name = vs[label["t"]][0]["name"]
            rel_name = DOCRED_REL_INFO.get(label["r"], label["r"])
            facts.add(_triple_key(h_name, t_name, rel_name))
    return facts


def compute_train_relation_freq(train_path):
    """Return Counter of relation names in training data."""
    with open(train_path) as f:
        train_data = json.load(f)
    freq = Counter()
    for doc in train_data:
        for label in doc.get("labels", []):
            rel_name = DOCRED_REL_INFO.get(label["r"], label["r"])
            freq[_norm(rel_name)] += 1
    return freq


def get_rare_relations(rel_freq, percentile=25):
    """Return set of relation names in the bottom `percentile`% by frequency."""
    if not rel_freq:
        return set()
    counts = sorted(rel_freq.values())
    threshold_idx = max(0, int(len(counts) * percentile / 100) - 1)
    threshold = counts[threshold_idx]
    return {r for r, c in rel_freq.items() if c <= threshold}


# ── Gold evidence lookup from dev.json ──

def build_gold_evidence_map(gold_data_raw):
    """Build (title, h_norm, t_norm, rel_norm) → set(evidence_sent_ids) from dev.json."""
    evi_map = {}
    for title, doc in gold_data_raw.items():
        vs = doc["vertexSet"]
        for label in doc.get("labels", []):
            h_name = vs[label["h"]][0]["name"]
            t_name = vs[label["t"]][0]["name"]
            rel_name = DOCRED_REL_INFO.get(label["r"], label["r"])
            key = (title, *_triple_key(h_name, t_name, rel_name))
            evi_map[key] = set(label.get("evidence", []))
    return evi_map


# ── Error classification ──

def classify_errors(doc_preds, train_facts, rare_relations, gold_evi_map):
    """Classify FP, FN, and evidence errors for a single document.

    Returns dict with counts and per-triple detail lists.
    """
    doc_id = doc_preds["doc_id"]
    pred_triples = doc_preds.get("parsed_triples", [])
    gold_triples = doc_preds.get("gold_triples", [])
    truncated = doc_preds.get("truncated", False)

    # Build keyed sets
    pred_by_key = defaultdict(list)
    for t in pred_triples:
        k = _triple_key(t["head"], t["tail"], t["relation"])
        pred_by_key[k].append(t)

    gold_by_key = {}
    gold_pairs = defaultdict(set)  # (h,t) → set of relations
    for t in gold_triples:
        k = _triple_key(t["head"], t["tail"], t["relation"])
        gold_by_key[k] = t
        gold_pairs[_pair_key(t["head"], t["tail"])].add(_norm(t["relation"]))

    pred_keys = set(pred_by_key.keys())
    gold_keys = set(gold_by_key.keys())
    tp_keys = pred_keys & gold_keys
    fp_keys = pred_keys - gold_keys
    fn_keys = gold_keys - pred_keys

    # ── FP classification ──
    fp_hallucinated = []
    fp_relation_confusion = []
    fp_duplicate_triples = []

    # Duplicates: same key predicted more than once
    for k, preds in pred_by_key.items():
        if len(preds) > 1:
            fp_duplicate_triples.extend(preds[1:])

    for k in fp_keys:
        h, t, r = k
        pair = (h, t)
        representative = pred_by_key[k][0]
        if pair in gold_pairs:
            fp_relation_confusion.append({
                **representative,
                "gold_relations": list(gold_pairs[pair]),
            })
        else:
            fp_hallucinated.append(representative)

    # ── FN classification ──
    fn_cross_sentence = []
    fn_rare_relation = []
    fn_truncation_loss = []
    fn_other = []

    for k in fn_keys:
        gold_t = gold_by_key[k]
        evi = set(gold_t.get("evidence", []))
        is_cross = len(evi) >= 2
        is_rare = _norm(gold_t["relation"]) in rare_relations
        is_trunc = truncated

        categorized = False
        if is_trunc:
            fn_truncation_loss.append(gold_t)
            categorized = True
        if is_cross:
            fn_cross_sentence.append(gold_t)
            categorized = True
        if is_rare:
            fn_rare_relation.append(gold_t)
            categorized = True
        if not categorized:
            fn_other.append(gold_t)

    # ── Evidence error classification (TP only) ──
    evi_exact = []
    evi_over = []
    evi_under = []
    evi_shift = []
    evi_no_gold = []

    for k in tp_keys:
        pred_t = pred_by_key[k][0]
        pred_evi = set(pred_t.get("evidence", []))

        # Try gold_evi_map (from dev.json) first, fall back to gold_triples
        map_key = (doc_id, *k)
        if map_key in gold_evi_map:
            gold_evi = gold_evi_map[map_key]
        else:
            gold_t = gold_by_key[k]
            gold_evi = set(gold_t.get("evidence", []))

        if not gold_evi:
            evi_no_gold.append(pred_t)
            continue

        if pred_evi == gold_evi:
            evi_exact.append(pred_t)
        elif not pred_evi & gold_evi:
            evi_shift.append({**pred_t, "gold_evidence": sorted(gold_evi)})
        elif pred_evi > gold_evi:
            evi_over.append({**pred_t, "gold_evidence": sorted(gold_evi)})
        elif pred_evi < gold_evi:
            evi_under.append({**pred_t, "gold_evidence": sorted(gold_evi)})
        else:
            # Partial overlap (not a subset in either direction)
            if len(pred_evi - gold_evi) > len(pred_evi & gold_evi):
                evi_shift.append({**pred_t, "gold_evidence": sorted(gold_evi)})
            elif len(pred_evi) > len(gold_evi):
                evi_over.append({**pred_t, "gold_evidence": sorted(gold_evi)})
            else:
                evi_under.append({**pred_t, "gold_evidence": sorted(gold_evi)})

    # ── Ign filtering ──
    tp_ign = {k for k in tp_keys if k not in train_facts}
    fp_ign = {k for k in fp_keys if k not in train_facts}
    fn_ign = {k for k in fn_keys if k not in train_facts}

    return {
        "doc_id": doc_id,
        "truncated": truncated,
        "n_pred": len(pred_triples),
        "n_gold": len(gold_triples),
        "tp": len(tp_keys),
        "fp": len(fp_keys),
        "fn": len(fn_keys),
        "tp_ign": len(tp_ign),
        "fp_ign": len(fp_ign),
        "fn_ign": len(fn_ign),
        "duplicates": len(fp_duplicate_triples),
        # FP breakdown
        "fp_hallucinated": len(fp_hallucinated),
        "fp_relation_confusion": len(fp_relation_confusion),
        # FN breakdown
        "fn_cross_sentence": len(fn_cross_sentence),
        "fn_rare_relation": len(fn_rare_relation),
        "fn_truncation_loss": len(fn_truncation_loss),
        "fn_other": len(fn_other),
        # Evidence breakdown (counts)
        "evi_exact": len(evi_exact),
        "evi_over_citation": len(evi_over),
        "evi_under_citation": len(evi_under),
        "evi_shift": len(evi_shift),
        # Detail lists (for per-relation analysis)
        "_fp_hallucinated": fp_hallucinated,
        "_fp_relation_confusion": fp_relation_confusion,
        "_fn_cross_sentence": fn_cross_sentence,
        "_fn_rare_relation": fn_rare_relation,
        "_fn_truncation_loss": fn_truncation_loss,
        "_fn_other": fn_other,
        "_evi_exact": evi_exact,
        "_evi_over": evi_over,
        "_evi_under": evi_under,
        "_evi_shift": evi_shift,
        "_tp_keys": tp_keys,
        "_fp_keys": fp_keys,
        "_fn_keys": fn_keys,
    }


def aggregate_results(doc_results):
    """Aggregate per-doc results into eval-level summary."""
    agg = Counter()
    keys = [
        "n_pred", "n_gold", "tp", "fp", "fn",
        "tp_ign", "fp_ign", "fn_ign", "duplicates",
        "fp_hallucinated", "fp_relation_confusion",
        "fn_cross_sentence", "fn_rare_relation", "fn_truncation_loss", "fn_other",
        "evi_exact", "evi_over_citation", "evi_under_citation", "evi_shift",
    ]
    for dr in doc_results:
        for k in keys:
            agg[k] += dr[k]

    n_truncated = sum(1 for dr in doc_results if dr["truncated"])
    n_docs = len(doc_results)

    # Compute F1 / Ign-F1
    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    tp_i, fp_i, fn_i = agg["tp_ign"], agg["fp_ign"], agg["fn_ign"]
    prec_i = tp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else 0.0
    rec_i = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0.0
    ign_f1 = 2 * prec_i * rec_i / (prec_i + rec_i) if (prec_i + rec_i) > 0 else 0.0

    # Evidence distribution percentages (over TP)
    evi_total = agg["evi_exact"] + agg["evi_over_citation"] + agg["evi_under_citation"] + agg["evi_shift"]

    return {
        "n_docs": n_docs,
        "n_truncated": n_truncated,
        **{k: agg[k] for k in keys},
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "ign_precision": prec_i,
        "ign_recall": rec_i,
        "ign_f1": ign_f1,
        "evi_total_tp_with_evi": evi_total,
        "evi_exact_pct": agg["evi_exact"] / evi_total * 100 if evi_total else 0,
        "evi_over_pct": agg["evi_over_citation"] / evi_total * 100 if evi_total else 0,
        "evi_under_pct": agg["evi_under_citation"] / evi_total * 100 if evi_total else 0,
        "evi_shift_pct": agg["evi_shift"] / evi_total * 100 if evi_total else 0,
        # FP breakdown percentages
        "fp_hallucinated_pct": agg["fp_hallucinated"] / agg["fp"] * 100 if agg["fp"] else 0,
        "fp_rel_confusion_pct": agg["fp_relation_confusion"] / agg["fp"] * 100 if agg["fp"] else 0,
    }


def build_per_relation_breakdown(doc_results):
    """Per-relation error counts across all documents."""
    rel_stats = defaultdict(lambda: Counter())

    for dr in doc_results:
        for t in dr["_fp_hallucinated"]:
            rel_stats[_norm(t["relation"])]["fp_hallucinated"] += 1
        for t in dr["_fp_relation_confusion"]:
            rel_stats[_norm(t["relation"])]["fp_relation_confusion"] += 1
        for t in dr["_fn_cross_sentence"]:
            rel_stats[_norm(t["relation"])]["fn_cross_sentence"] += 1
        for t in dr["_fn_rare_relation"]:
            rel_stats[_norm(t["relation"])]["fn_rare_relation"] += 1
        for t in dr["_fn_truncation_loss"]:
            rel_stats[_norm(t["relation"])]["fn_truncation_loss"] += 1
        for t in dr["_fn_other"]:
            rel_stats[_norm(t["relation"])]["fn_other"] += 1
        for t in dr["_evi_exact"]:
            rel_stats[_norm(t["relation"])]["evi_exact"] += 1
        for t in dr["_evi_over"]:
            rel_stats[_norm(t["relation"])]["evi_over"] += 1
        for t in dr["_evi_under"]:
            rel_stats[_norm(t["relation"])]["evi_under"] += 1
        for t in dr["_evi_shift"]:
            rel_stats[_norm(t["relation"])]["evi_shift"] += 1
        for k in dr["_tp_keys"]:
            rel_stats[k[2]]["tp"] += 1
        for k in dr["_fp_keys"]:
            rel_stats[k[2]]["fp_total"] += 1
        for k in dr["_fn_keys"]:
            rel_stats[k[2]]["fn_total"] += 1

    return {rel: dict(counts) for rel, counts in sorted(rel_stats.items())}


# ── Output generation ──

def write_eval_csv(agg, output_path):
    """Write single-eval summary CSV."""
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in agg.items():
            if isinstance(v, float):
                w.writerow([k, f"{v:.4f}"])
            else:
                w.writerow([k, v])


def write_comparison_csv(all_results, output_path):
    """Write multi-eval comparison CSV."""
    if not all_results:
        return
    eval_names = list(all_results.keys())
    metrics = list(all_results[eval_names[0]].keys())

    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + eval_names)
        for m in metrics:
            row = [m]
            for name in eval_names:
                v = all_results[name].get(m, "")
                if isinstance(v, float):
                    row.append(f"{v:.4f}")
                else:
                    row.append(v)
            w.writerow(row)


def generate_latex_table(all_results, output_path):
    """Generate LaTeX table for paper: FP/FN breakdown + evidence errors."""
    eval_names = list(all_results.keys())
    n_cols = len(eval_names)

    # Shorten eval names for column headers
    short_names = []
    for name in eval_names:
        short = name.replace("d102_", "").replace("plan_006_", "").replace("_k1_s42", "")
        short = short.replace("_eval", "").replace("_", " ").title()
        short_names.append(short)

    col_spec = "l" + "r" * n_cols
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header = " & ".join([""] + [f"\\textbf{{{s}}}" for s in short_names]) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Overall metrics
    lines.append(r"\multicolumn{" + str(n_cols + 1) + r"}{l}{\textit{Overall Metrics}} \\")
    for metric, label in [
        ("ign_f1", "Ign F1"), ("f1", "F1"),
        ("precision", "Precision"), ("recall", "Recall"),
    ]:
        vals = [f"{all_results[e].get(metric, 0):.1%}" for e in eval_names]
        lines.append(f"  {label} & " + " & ".join(vals) + r" \\")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(n_cols + 1) + r"}{l}{\textit{False Positive Breakdown}} \\")
    for metric, label in [
        ("fp", "Total FP"),
        ("fp_hallucinated", "Hallucinated Rel."),
        ("fp_relation_confusion", "Relation Confusion"),
        ("duplicates", "Duplicates"),
    ]:
        vals = [str(all_results[e].get(metric, 0)) for e in eval_names]
        lines.append(f"  {label} & " + " & ".join(vals) + r" \\")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(n_cols + 1) + r"}{l}{\textit{False Negative Breakdown}} \\")
    for metric, label in [
        ("fn", "Total FN"),
        ("fn_cross_sentence", "Cross-sent. Miss"),
        ("fn_rare_relation", "Rare Rel. Miss"),
        ("fn_truncation_loss", "Truncation Loss"),
        ("fn_other", "Other FN"),
    ]:
        vals = [str(all_results[e].get(metric, 0)) for e in eval_names]
        lines.append(f"  {label} & " + " & ".join(vals) + r" \\")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(n_cols + 1) + r"}{l}{\textit{Evidence Errors (TP only)}} \\")
    for metric, label in [
        ("evi_exact_pct", "Exact Match (\\%)"),
        ("evi_over_pct", "Over-citation (\\%)"),
        ("evi_under_pct", "Under-citation (\\%)"),
        ("evi_shift_pct", "Evidence Shift (\\%)"),
    ]:
        vals = [f"{all_results[e].get(metric, 0):.1f}" for e in eval_names]
        lines.append(f"  {label} & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Error analysis across models. FP/FN counts are absolute; evidence errors are percentages over true positives. FN sub-categories can overlap (e.g., a triple can be both cross-sentence and truncation loss).}")
    lines.append(r"\label{tab:error_analysis}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Deep error analysis for FREIGE relation extraction predictions."
    )
    parser.add_argument(
        "--eval_dirs", nargs="+", required=True,
        help="Eval directory names (under eval_base)."
    )
    parser.add_argument(
        "--eval_base", default="/workspace/eval_results",
        help="Base directory containing eval result dirs."
    )
    parser.add_argument(
        "--gold_path", default="/workspace/data/docred/dev.json",
        help="Path to DocRED dev.json."
    )
    parser.add_argument(
        "--train_path", default="/workspace/data/docred/train_annotated.json",
        help="Path to DocRED train_annotated.json."
    )
    parser.add_argument(
        "--output_dir", default="/workspace/analysis/error_analysis",
        help="Output directory."
    )
    parser.add_argument(
        "--rare_percentile", type=int, default=25,
        help="Bottom percentile for 'rare relation' threshold."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load shared data
    print(f"Loading gold data from {args.gold_path} ...")
    gold_data_raw = load_gold_data(args.gold_path)
    gold_evi_map = build_gold_evidence_map(gold_data_raw)
    print(f"  {len(gold_data_raw)} documents, {len(gold_evi_map)} gold triples with evidence")

    print(f"Loading train data from {args.train_path} ...")
    train_facts = build_train_facts(args.train_path)
    rel_freq = compute_train_relation_freq(args.train_path)
    rare_relations = get_rare_relations(rel_freq, args.rare_percentile)
    print(f"  {len(train_facts)} train facts, {len(rel_freq)} relations, {len(rare_relations)} rare")
    print(f"  Rare relations (bottom {args.rare_percentile}%): {sorted(rare_relations)[:10]}...")

    # Process each eval
    all_agg = {}
    all_per_relation = {}

    for eval_name in args.eval_dirs:
        pred_path = os.path.join(args.eval_base, eval_name, "predictions.json")
        if not os.path.exists(pred_path):
            print(f"WARNING: {pred_path} not found, skipping.")
            continue

        print(f"\nAnalyzing {eval_name} ...")
        predictions = load_predictions(pred_path)
        print(f"  {len(predictions)} documents")

        doc_results = []
        for doc_pred in predictions:
            dr = classify_errors(doc_pred, train_facts, rare_relations, gold_evi_map)
            doc_results.append(dr)

        agg = aggregate_results(doc_results)
        all_agg[eval_name] = agg

        # Per-eval CSV
        csv_path = output_dir / f"{eval_name}_errors.csv"
        write_eval_csv(agg, csv_path)
        print(f"  Wrote {csv_path}")

        # Per-relation JSON
        per_rel = build_per_relation_breakdown(doc_results)
        all_per_relation[eval_name] = per_rel
        rel_path = output_dir / f"{eval_name}_per_relation.json"
        with open(rel_path, "w") as f:
            json.dump(per_rel, f, indent=2, ensure_ascii=False)
        print(f"  Wrote {rel_path}")

        # Print summary
        print(f"  Ign-F1={agg['ign_f1']:.4f}  F1={agg['f1']:.4f}  "
              f"P={agg['precision']:.4f}  R={agg['recall']:.4f}")
        print(f"  TP={agg['tp']}  FP={agg['fp']}  FN={agg['fn']}  Dup={agg['duplicates']}")
        print(f"  FP: hallucinated={agg['fp_hallucinated']} "
              f"rel_confusion={agg['fp_relation_confusion']}")
        print(f"  FN: cross_sent={agg['fn_cross_sentence']} "
              f"rare_rel={agg['fn_rare_relation']} "
              f"trunc={agg['fn_truncation_loss']} "
              f"other={agg['fn_other']}")
        print(f"  Evi: exact={agg['evi_exact_pct']:.1f}% "
              f"over={agg['evi_over_pct']:.1f}% "
              f"under={agg['evi_under_pct']:.1f}% "
              f"shift={agg['evi_shift_pct']:.1f}%")

    if len(all_agg) > 1:
        # Comparison CSV
        comp_path = output_dir / "comparison.csv"
        write_comparison_csv(all_agg, comp_path)
        print(f"\nWrote comparison CSV: {comp_path}")

    # LaTeX table (even for single eval, useful)
    if all_agg:
        latex_path = output_dir / "error_analysis_table.tex"
        generate_latex_table(all_agg, latex_path)
        print(f"Wrote LaTeX table: {latex_path}")

    # Combined per-relation JSON
    if all_per_relation:
        combined_path = output_dir / "per_relation_all.json"
        with open(combined_path, "w") as f:
            json.dump(all_per_relation, f, indent=2, ensure_ascii=False)
        print(f"Wrote combined per-relation: {combined_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
