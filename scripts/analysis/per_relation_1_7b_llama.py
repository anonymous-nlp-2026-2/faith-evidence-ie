"""Per-relation F1 analysis for 1.7B Qwen and LLaMA-8B (SFT vs RSFT)."""

import json, csv, sys
from collections import defaultdict
from pathlib import Path

EVAL_DIR = Path("eval_results")
GOLD_PATH = Path("data/docred/dev.json")
TRAIN_PATH = Path("data/docred/train_annotated.json")
OUT_DIR = Path("./scripts/analysis/results")

GROUPS = {
    "1_7B": {
        "SFT": EVAL_DIR / "qwen3_1_7b_sft_eval" / "predictions.json",
        "RSFT": EVAL_DIR / "d102_1_7b_k1_s42" / "predictions.json",
    },
    "LLaMA": {
        "SFT": EVAL_DIR / "llama_3_1_8b_sft_eval" / "predictions.json",
        "RSFT": EVAL_DIR / "d102_llama_k1_s42" / "predictions.json",
    },
}

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


def _key(head, tail, relation):
    return (head.lower().strip(), tail.lower().strip(), relation.lower().strip())


def _doc_key(doc_id, head, tail, relation):
    return (doc_id, *_key(head, tail, relation))


def build_train_facts():
    with open(TRAIN_PATH) as f:
        train_data = json.load(f)
    facts = set()
    for doc in train_data:
        vs = doc["vertexSet"]
        for lab in doc.get("labels", []):
            h = vs[lab["h"]][0]["name"]
            t = vs[lab["t"]][0]["name"]
            r = DOCRED_REL_INFO.get(lab["r"], lab["r"])
            facts.add(_key(h, t, r))
    return facts


def load_predictions(path):
    with open(path) as f:
        data = json.load(f)
    docs = {}
    for d in data:
        doc_id = d["doc_id"]
        preds = [(t["head"], t["tail"], t["relation"]) for t in d.get("parsed_triples", [])]
        golds = [(t["head"], t["tail"], t["relation"]) for t in d.get("gold_triples", [])]
        docs[doc_id] = {"preds": preds, "golds": golds}
    return docs


def compute_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def per_relation_analysis(all_preds, train_facts):
    results = {}
    for model_name, docs in all_preds.items():
        rel_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        for doc_id, d in docs.items():
            pred_set = {_doc_key(doc_id, h, t, r) for h, t, r in d["preds"]}
            gold_set = {_doc_key(doc_id, h, t, r) for h, t, r in d["golds"]}
            tp_set = pred_set & gold_set
            fp_set = pred_set - gold_set
            fn_set = gold_set - pred_set
            for k in tp_set:
                rel_stats[k[3]]["tp"] += 1
            for k in fp_set:
                rel_stats[k[3]]["fp"] += 1
            for k in fn_set:
                rel_stats[k[3]]["fn"] += 1
        results[model_name] = dict(rel_stats)
    return results


def analyze_group(group_name, sft_path, rsft_path, train_facts, out_dir):
    sft_label = f"{group_name}_SFT"
    rsft_label = f"{group_name}_RSFT"

    all_preds = {}
    for label, path in [(sft_label, sft_path), (rsft_label, rsft_path)]:
        if not path.exists():
            print(f"  WARNING: {label} not found at {path}")
            continue
        all_preds[label] = load_predictions(path)
        print(f"  {label}: {len(all_preds[label])} docs")

    if len(all_preds) < 2:
        print(f"  Skipping {group_name}: need both SFT and RSFT")
        return None

    rel_results = per_relation_analysis(all_preds, train_facts)

    all_rels = set()
    for model_stats in rel_results.values():
        all_rels.update(model_stats.keys())
    all_rels = sorted(all_rels)

    rows = []
    for rel in all_rels:
        sft_s = rel_results[sft_label].get(rel, {"tp": 0, "fp": 0, "fn": 0})
        rsft_s = rel_results[rsft_label].get(rel, {"tp": 0, "fp": 0, "fn": 0})
        sft_p, sft_r, sft_f1 = compute_f1(sft_s["tp"], sft_s["fp"], sft_s["fn"])
        rsft_p, rsft_r, rsft_f1 = compute_f1(rsft_s["tp"], rsft_s["fp"], rsft_s["fn"])
        gold = sft_s["tp"] + sft_s["fn"]
        gain = rsft_f1 - sft_f1
        rows.append({
            "relation": rel,
            "gold": gold,
            "sft_f1": round(sft_f1, 4),
            "sft_p": round(sft_p, 4),
            "sft_r": round(sft_r, 4),
            "sft_tp": sft_s["tp"],
            "sft_fp": sft_s["fp"],
            "sft_fn": sft_s["fn"],
            "rsft_f1": round(rsft_f1, 4),
            "rsft_p": round(rsft_p, 4),
            "rsft_r": round(rsft_r, 4),
            "rsft_tp": rsft_s["tp"],
            "rsft_fp": rsft_s["fp"],
            "rsft_fn": rsft_s["fn"],
            "gain": round(gain, 4),
        })

    rows.sort(key=lambda x: x["gain"], reverse=True)

    # Write CSV
    csv_path = out_dir / f"per_relation_{group_name.lower()}.csv"
    fieldnames = ["relation", "gold", "sft_f1", "rsft_f1", "gain",
                  "sft_p", "sft_r", "sft_tp", "sft_fp", "sft_fn",
                  "rsft_p", "rsft_r", "rsft_tp", "rsft_fp", "rsft_fn"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Write readable txt
    txt_path = out_dir / f"per_relation_{group_name.lower()}.txt"
    lines = []
    lines.append(f"Per-Relation F1 Breakdown: {group_name} (SFT vs RSFT)")
    lines.append(f"SFT:  {sft_path}")
    lines.append(f"RSFT: {rsft_path}")
    lines.append(f"Total relations found: {len(rows)}")
    lines.append("")

    # Overall micro F1
    for label in [sft_label, rsft_label]:
        total_tp = sum(s["tp"] for s in rel_results[label].values())
        total_fp = sum(s["fp"] for s in rel_results[label].values())
        total_fn = sum(s["fn"] for s in rel_results[label].values())
        _, _, overall_f1 = compute_f1(total_tp, total_fp, total_fn)
        lines.append(f"{label} overall micro-F1: {overall_f1:.4f} (TP={total_tp}, FP={total_fp}, FN={total_fn})")
    lines.append("")

    # Top 20 by gold count
    by_gold = sorted(rows, key=lambda x: x["gold"], reverse=True)[:20]
    lines.append("=" * 90)
    lines.append("Top-20 relations by frequency (gold count)")
    lines.append("=" * 90)
    lines.append(f"{'Relation':<45s} {'Gold':>5s} {'SFT_F1':>7s} {'RSFT_F1':>8s} {'Gain':>7s}")
    lines.append("-" * 90)
    for r in by_gold:
        lines.append(f"{r['relation']:<45s} {r['gold']:>5d} {r['sft_f1']:>7.4f} {r['rsft_f1']:>8.4f} {r['gain']:>+7.4f}")

    # Top 10 RSFT gain
    lines.append("")
    lines.append("=" * 90)
    lines.append("Top-10 relations by RSFT gain (gold >= 10)")
    lines.append("=" * 90)
    lines.append(f"{'Relation':<45s} {'Gold':>5s} {'SFT_F1':>7s} {'RSFT_F1':>8s} {'Gain':>7s}")
    lines.append("-" * 90)
    filtered = [r for r in rows if r["gold"] >= 10]
    for r in filtered[:10]:
        lines.append(f"{r['relation']:<45s} {r['gold']:>5d} {r['sft_f1']:>7.4f} {r['rsft_f1']:>8.4f} {r['gain']:>+7.4f}")

    # Bottom 10 RSFT gain
    lines.append("")
    lines.append("=" * 90)
    lines.append("Bottom-10 relations by RSFT gain (gold >= 10)")
    lines.append("=" * 90)
    lines.append(f"{'Relation':<45s} {'Gold':>5s} {'SFT_F1':>7s} {'RSFT_F1':>8s} {'Gain':>7s}")
    lines.append("-" * 90)
    for r in filtered[-10:]:
        lines.append(f"{r['relation']:<45s} {r['gold']:>5d} {r['sft_f1']:>7.4f} {r['rsft_f1']:>8.4f} {r['gain']:>+7.4f}")

    # Zero-gold relations in RSFT (hallucinated relations)
    rsft_only = [r for r in rows if r["gold"] == 0 and r["rsft_fp"] > 0]
    if rsft_only:
        lines.append("")
        lines.append("=" * 90)
        lines.append(f"Relations with 0 gold but RSFT predicted (FP only): {len(rsft_only)}")
        lines.append("=" * 90)
        for r in sorted(rsft_only, key=lambda x: x["rsft_fp"], reverse=True)[:10]:
            lines.append(f"  {r['relation']:<45s} FP={r['rsft_fp']}")

    # Summary stats
    lines.append("")
    lines.append("=" * 90)
    lines.append("Summary")
    lines.append("=" * 90)
    improved = [r for r in filtered if r["gain"] > 0]
    degraded = [r for r in filtered if r["gain"] < 0]
    unchanged = [r for r in filtered if r["gain"] == 0]
    lines.append(f"Relations with gold >= 10: {len(filtered)}")
    lines.append(f"  Improved (gain > 0):  {len(improved)}")
    lines.append(f"  Degraded (gain < 0):  {len(degraded)}")
    lines.append(f"  Unchanged (gain = 0): {len(unchanged)}")
    if improved:
        avg_gain = sum(r["gain"] for r in improved) / len(improved)
        lines.append(f"  Avg gain (improved):  {avg_gain:+.4f}")
    if degraded:
        avg_loss = sum(r["gain"] for r in degraded) / len(degraded)
        lines.append(f"  Avg loss (degraded):  {avg_loss:+.4f}")

    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n  Written: {csv_path}")
    print(f"  Written: {txt_path}")
    return rows


def main():
    print("Loading train facts...")
    train_facts = build_train_facts()
    print(f"  {len(train_facts)} unique train facts")

    for group_name, paths in GROUPS.items():
        print(f"\n=== {group_name} ===")
        analyze_group(group_name, paths["SFT"], paths["RSFT"], train_facts, OUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
